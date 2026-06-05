"""Property test P6: skip mode + precedência de classificação de erro.

**Validates: Requirements 6.2, 6.3, 6.5, 6.6, 6.7, 6.8, 11.3**

A propriedade afirma que, *para qualquer* exceção ``GcvError`` que carregue
um conjunto não-vazio de classes ``S ⊆ {auth_error, quota_exceeded,
timeout, generic_error}``, com ``on_failure == "skip"``, o
``CloudVisionPipeline`` produz:

- ``result.ocr_text == ""`` e ``result.mean_confidence == 0.0``
  (Requirement 6.2 — resultado vazio mas válido).
- ``result.metadata["error"]`` igual à classe vencedora pela precedência
  fixa ``auth_error > quota_exceeded > timeout > generic_error``
  (Requirements 6.5–6.8).
- ``result.metadata["error_secondary"]`` é uma ``list`` ordenada por
  precedência, contendo exatamente ``S \\ {error}`` (Requirement 6.8).
- ``result.metadata["error_message"]`` é uma ``str`` com
  ``len <= 500`` (Requirement 6.2 — truncamento defensivo).
- ``result.stages[-1].name == "output"`` (o caminho de falha ainda
  preserva o contrato visual da auditoria — Property 1).
- A combinação ``("", 0.0)`` faz o ``QualityEvaluator`` produzir
  ``passed == False``, garantindo que o ``NutritionReader`` prossiga
  para o próximo preset da cascata (Requirement 11.3).

A implementação substitui o ``GcvClient`` real por um stub
(``_RaisingGcvClient``) cujo único papel é levantar uma ``GcvError``
pré-construída — assim o teste cobre o ramo de falha sem depender do
SDK ``google-cloud-vision`` ou de qualquer I/O de rede.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from audit.recorder import AuditRecorder
from nutrition.pipelines.base import PipelineContext
from nutrition.pipelines.cloud_vision import CloudVisionPipeline
from ocr.cloud_vision.options import GcvPresetOptions
from ocr.cloud_vision.types import ERROR_PRECEDENCE, GcvError
from ocr.quality import QualityEvaluator, QualityThresholds
from ocr.service import OcrConfig
from tests.ocr_engine.gcv.strategies import error_class_subsets


# Índice de precedência canônica (Requirement 6.8): quanto menor o índice,
# mais "estrutural" a classe — ``auth_error`` (0) domina ``generic_error``
# (3). Pré-computamos uma vez para evitar buscas repetidas em ``ERROR_PRECEDENCE``
# dentro do hot-loop do Hypothesis.
_PRECEDENCE_INDEX: dict[str, int] = {
    code: idx for idx, code in enumerate(ERROR_PRECEDENCE)
}

# Conjunto canônico de classes válidas reportáveis em ``metadata.error`` /
# ``metadata.error_secondary`` no caminho de falha de chamada à API.
# Usado apenas para validar o invariante "todos os secundários são códigos
# legítimos" — pré-API (``invalid_feature``, ``import_error``) NÃO faz
# parte deste teste por construção (P6 cobre apenas falhas de chamada à
# API; falhas pré-API são objeto de P12 — ``test_invalid_feature.py`` /
# task 10.6).
_VALID_ERROR_CODES: frozenset[str] = frozenset(ERROR_PRECEDENCE)


class _RaisingGcvClient:
    """Stub mínimo de ``GcvClient`` que sempre levanta uma ``GcvError`` fixa.

    Não simula cache, rate-limit nem inicialização lazy: o caminho de falha
    no pipeline depende apenas da semântica do ``GcvError`` levantado e da
    política ``on_failure`` — todo o resto do ``GcvClient`` é desnecessário
    para falsificar P6. Manter o stub minúsculo isola a propriedade da
    dependência ``google-cloud-vision`` (que pode ou não estar instalada
    no ambiente de teste, conforme Requirement 14.3).
    """

    def __init__(self, err: GcvError) -> None:
        # Armazenamos a exceção uma vez no construtor; ``fetch`` apenas
        # re-levanta. Usar atributo privado deixa explícito que o stub não
        # expõe estado público para os testes inspecionarem.
        self._err = err

    def fetch(self, *args: object, **kwargs: object) -> object:  # pragma: no cover - stub
        # Assinatura aceita qualquer entrada (``png_bytes, feature, hints``)
        # porque o pipeline chama com argumentos posicionais; aqui não nos
        # importamos com a forma — só com o efeito (raise).
        raise self._err


def _build_pipeline_with_failure(
    err: GcvError,
) -> CloudVisionPipeline:
    """Constrói um ``CloudVisionPipeline`` em modo ``skip`` com cliente stub.

    Centraliza a montagem para evitar duplicação entre os exemplos do
    Hypothesis e os casos determinísticos auxiliares. As opções do preset
    são fixadas em valores válidos (``DOCUMENT_TEXT_DETECTION`` + ``("pt",)``)
    para que o pipeline siga o caminho de chamada à API — caminho onde a
    ``GcvError`` será efetivamente capturada por ``execute``. Caminhos
    alternativos (``invalid_feature``) são objeto de P14 e ficam fora deste
    teste.
    """

    options = GcvPresetOptions(
        feature="DOCUMENT_TEXT_DETECTION",
        language_hints=("pt",),
        model=None,
        invalid_feature=False,
        raw_feature=None,
    )
    # ``OcrConfig`` é exigido pelo construtor do pipeline por uniformidade,
    # mas o caminho GCV não consome o objeto. Mantemos os defaults para
    # tornar a intenção explícita (não estamos exercitando Tesseract).
    ocr_config = OcrConfig()
    return CloudVisionPipeline(
        gcv_options=options,
        ocr_config=ocr_config,
        client=_RaisingGcvClient(err),
        on_failure="skip",
        # ``ignored_steps_count=0`` reflete um preset GCV canônico
        # (``steps == []`` por convenção — Requirement 3.8). O valor não
        # afeta P6, mas é o caso de produção.
        ignored_steps_count=0,
        gcv_config_warnings=(),
    )


def _make_context(
    project_root: Path,
    preset_name: str = "gcv_doc_text",
) -> tuple[AuditRecorder, PipelineContext]:
    """Materializa a estrutura mínima de auditoria para uma execução.

    Cria a árvore ``extractions/`` e ``images/pipeline/`` exigida pelo
    ``AuditRecorder`` e devolve o par ``(recorder, context)`` pronto para
    ser passado a ``CloudVisionPipeline.execute``. Cada chamada usa um
    ``project_root`` distinto (``tmp_path_factory.mktemp``) para isolar
    iterações sucessivas do Hypothesis.
    """

    (project_root / "extractions").mkdir(parents=True, exist_ok=True)
    (project_root / "images" / "pipeline").mkdir(parents=True, exist_ok=True)
    input_path = project_root / "subjects" / "label_subject.png"

    recorder = AuditRecorder(
        project_root=project_root,
        input_path=input_path,
        # ``clean_previous`` não é relevante aqui (a árvore já está vazia),
        # mas mantemos o default explícito para alinhar com o uso do
        # ``NutritionReader`` em produção.
        clean_previous=True,
    )
    artifacts = recorder.start_attempt(1, preset_name)
    context = PipelineContext(
        input_path=input_path,
        attempt_index=1,
        preset_name=preset_name,
        recorder=recorder,
        artifacts=artifacts,
    )
    return recorder, context


def _expected_primary(error_classes: frozenset[str]) -> str:
    """Determina a classe vencedora pela precedência fixa do design."""

    return min(error_classes, key=_PRECEDENCE_INDEX.__getitem__)


def _expected_secondary(error_classes: frozenset[str], primary: str) -> tuple[str, ...]:
    """Constrói a tupla ``error_secondary`` esperada (ordenada por precedência)."""

    return tuple(
        sorted(error_classes - {primary}, key=_PRECEDENCE_INDEX.__getitem__)
    )


# ---------------------------------------------------------------------------
# Property 6 — corpo principal
# ---------------------------------------------------------------------------


@given(
    error_classes=error_class_subsets(),
    # ``message_payload`` cobre tanto mensagens curtas (típicas de erros do
    # SDK gRPC) quanto excedentes a 500 caracteres (HTTP body com JSON
    # detalhado da Google), exercitando a invariante de truncamento
    # imposta por ``CloudVisionPipeline._populate_error_metadata``.
    message_payload=st.text(min_size=0, max_size=900),
)
@settings(
    max_examples=100,
    deadline=None,
    # Necessário porque ``tmp_path_factory`` é session-scoped e nosso
    # uso por exemplo (via ``mktemp``) já garante isolamento — o aviso de
    # "function-scoped fixture reused" é falso positivo neste padrão.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_skip_mode_produz_resultado_vazio_com_precedencia(
    tmp_path_factory: pytest.TempPathFactory,
    error_classes: frozenset[str],
    message_payload: str,
) -> None:
    """**Property 6**: ``on_failure == "skip"`` → resultado vazio classificado por precedência.

    **Validates: Requirements 6.2, 6.3, 6.5, 6.6, 6.7, 6.8, 11.3**
    """

    # ---------------------------------------------------------------
    # Arrange — derivamos primário/secundários da precedência canônica
    # e construímos um ``GcvError`` exatamente como ``GcvClient._classify``
    # faria em produção. Isso evita sobrepor a lógica de classificação
    # neste teste (P6 é sobre o que o PIPELINE faz com um ``GcvError`` já
    # classificado; a classificação em si é coberta pelo task 6.2).
    # ---------------------------------------------------------------
    expected_primary = _expected_primary(error_classes)
    expected_secondary = _expected_secondary(error_classes, expected_primary)
    err = GcvError(
        error=expected_primary,
        message=message_payload,
        secondary=expected_secondary,
    )

    project_root = tmp_path_factory.mktemp("p6_skip_mode")
    _, context = _make_context(project_root)

    pipeline = _build_pipeline_with_failure(err)

    # Imagem mínima válida — o conteúdo é irrelevante para o caminho de
    # falha (a chamada ao ``client.fetch`` falha antes de qualquer parse).
    # Mantemos 16×16 BGR ``uint8`` por consistência com o gerador de
    # imagens em ``strategies.image_arrays``.
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    # ---------------------------------------------------------------
    # Act
    # ---------------------------------------------------------------
    result = pipeline.execute(image, context)

    # ---------------------------------------------------------------
    # Assert — invariantes da Property 6
    # ---------------------------------------------------------------

    # (1) Requirement 6.2: ``ocr_text`` vazio e ``mean_confidence`` zero.
    assert result.ocr_text == ""
    assert result.mean_confidence == 0.0

    # (2) Requirements 6.5–6.8: precedência canônica determina o código
    # primário e a ordem dos secundários.
    metadata = result.metadata
    assert metadata["error"] == expected_primary
    assert metadata["error_secondary"] == list(expected_secondary)
    # ``error_secondary`` é uma ``list`` (não tupla) para permitir
    # serialização direta em ``_summary.json`` sem custom encoder.
    assert isinstance(metadata["error_secondary"], list)
    # Todos os secundários pertencem ao conjunto canônico de códigos
    # válidos (defesa contra strings espúrias vazando do SDK).
    assert all(code in _VALID_ERROR_CODES for code in metadata["error_secondary"])

    # (3) Requirement 6.2: ``error_message`` é string com ``len <= 500``.
    assert isinstance(metadata["error_message"], str)
    assert len(metadata["error_message"]) <= 500
    # A mensagem original (potencialmente longa) deve coincidir com a
    # truncada nos primeiros 500 caracteres — o truncamento é defensivo,
    # não decorativo: nada de "..." sufixado, conforme design.
    assert metadata["error_message"] == message_payload[:500]

    # (4) Property 1 / Requirement 1.5: o caminho de falha ainda preserva
    # ``stages[-1].name == "output"`` para manter o contrato visual da
    # auditoria — consumidores que rastreiam a saída final pelo último
    # stage (``_summary.json``) continuam alinhados mesmo em ``skip``.
    assert result.stages, "stages list não pode estar vazia mesmo em skip"
    assert result.stages[0].name == "input"
    assert result.stages[-1].name == "output"

    # (5) Requirement 11.3: a combinação ``("", 0.0)`` faz o
    # ``QualityEvaluator`` produzir ``passed == False`` — invariante
    # crítico para que a cascata avance ao próximo preset, INDEPENDENTEMENTE
    # do código de erro classificado em ``metadata.error``.
    quality = QualityEvaluator(QualityThresholds()).evaluate(
        result.ocr_text,
        result.mean_confidence,
    )
    assert quality.passed is False, (
        "skip mode deve produzir um PipelineResult que o QualityEvaluator "
        "marca como passed=False, garantindo que a cascata progrida"
    )


# ---------------------------------------------------------------------------
# Casos determinísticos auxiliares
# ---------------------------------------------------------------------------
#
# Os exemplos abaixo travam casos canônicos da Property 6 que a estratégia
# ``error_class_subsets()`` cobre estatisticamente, mas que vale a pena
# fixar como regressão explícita: subconjuntos de tamanho 1 (uma só
# classe), o subconjunto completo (todas as quatro), e o caso de
# truncamento exato em 500 caracteres.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "single_class",
    [
        "auth_error",
        "quota_exceeded",
        "timeout",
        "generic_error",
    ],
)
def test_skip_mode_classe_unica_sem_secundarios(
    tmp_path_factory: pytest.TempPathFactory,
    single_class: str,
) -> None:
    """``S = {classe}`` ⇒ ``error == classe`` e ``error_secondary == []``.

    Garante que cada classe canônica do conjunto ``ERROR_PRECEDENCE``
    sobrevive ao caminho ``skip`` quando aparece sozinha. Falha aqui
    indica regressão na cópia de ``GcvError.error`` para
    ``metadata.error`` — e o relatório de erro do pytest mostra a classe
    afetada diretamente, sem depender do shrinking do Hypothesis.
    """

    err = GcvError(error=single_class, message="boom", secondary=())
    project_root = tmp_path_factory.mktemp("p6_single_class")
    _, context = _make_context(project_root)
    pipeline = _build_pipeline_with_failure(err)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    result = pipeline.execute(image, context)

    assert result.ocr_text == ""
    assert result.mean_confidence == 0.0
    assert result.metadata["error"] == single_class
    assert result.metadata["error_secondary"] == []
    assert result.metadata["error_message"] == "boom"
    assert result.stages[-1].name == "output"


def test_skip_mode_todas_as_classes_aplicam_precedencia(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``S = ERROR_PRECEDENCE`` ⇒ ``auth_error`` vence; demais como secundários.

    Caso "tudo aplicável simultaneamente" — exercita a precedência fixa
    em sua forma máxima. ``auth_error`` deve dominar e os três restantes
    aparecem em ``error_secondary`` na ordem canônica
    ``[quota_exceeded, timeout, generic_error]``.
    """

    err = GcvError(
        error="auth_error",
        message="multiplas classes",
        secondary=("quota_exceeded", "timeout", "generic_error"),
    )
    project_root = tmp_path_factory.mktemp("p6_all_classes")
    _, context = _make_context(project_root)
    pipeline = _build_pipeline_with_failure(err)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    result = pipeline.execute(image, context)

    assert result.metadata["error"] == "auth_error"
    assert result.metadata["error_secondary"] == [
        "quota_exceeded",
        "timeout",
        "generic_error",
    ]


def test_skip_mode_trunca_mensagem_acima_de_500(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Mensagem de 1000 caracteres ⇒ ``error_message`` exatamente os 500 primeiros.

    Caso explícito do contrato de truncamento do design: o limite é
    fixo em 500 caracteres, sem sufixo ``"..."``. Se algum dia o
    pipeline introduzir um sufixo decorativo, este teste falha
    imediatamente apontando para a mudança não-documentada.
    """

    long_message = "x" * 1000
    err = GcvError(error="timeout", message=long_message, secondary=())
    project_root = tmp_path_factory.mktemp("p6_truncation")
    _, context = _make_context(project_root)
    pipeline = _build_pipeline_with_failure(err)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    result = pipeline.execute(image, context)

    assert len(result.metadata["error_message"]) == 500
    assert result.metadata["error_message"] == "x" * 500


# ---------------------------------------------------------------------------
# Property 12 — Validates: Requirements 6.2, 6.4, 14.3, 14.4
# ---------------------------------------------------------------------------
# P12 cobre falhas **pré-API**: erros que ocorrem dentro de
# ``GcvClient._ensure_client()`` antes de qualquer chamada à rede, e não no
# ``client.fetch()`` diretamente (P6 já cobre esse caso via
# ``_RaisingGcvClient``). Dois tipos de falha pré-API são relevantes:
#
#   1. ``import_error`` — o import lazy de ``google.cloud.vision`` já falhou
#      em uma tentativa anterior; ``_ensure_client`` re-levanta
#      ``GcvError(error="import_error")`` sem repetir o import.
#   2. ``auth_error`` — ``resolve_credentials`` não encontra credenciais
#      válidas; ``_ensure_client`` propaga o ``GcvError(error="auth_error")``
#      levantado por ``auth.resolve_credentials``.
#
# Em ambos os casos a política ``on_failure`` do ``CloudVisionPipeline``
# deve ser respeitada: ``"skip"`` → resultado vazio válido;
# ``"raise"`` → re-propagação da ``GcvError`` com o código original.
# ---------------------------------------------------------------------------


# -- helpers de construção ---------------------------------------------------


def _build_import_error_client_p12(project_root: Path) -> object:
    """Constrói um ``GcvClient`` com ``_import_error`` já cacheado.

    Simula o estado de um processo que tentou importar
    ``google.cloud.vision`` e falhou (DLL ausente, SDK não instalado).
    O ``_api_client`` é ``None`` para garantir que ``_ensure_client``
    não curto-circuite pelo primeiro caminho (cliente já disponível).
    O cache é desabilitado para que ``fetch`` não tente um lookup
    antes de chegar a ``_ensure_client``.
    """

    from ocr.cloud_vision.app_config import GcvAppConfig
    from ocr.cloud_vision.client import GcvClient

    config = GcvAppConfig.from_dict(
        {"cache_enabled": False, "on_failure": "skip"},
        project_root,
    )
    # Construção direta do dataclass: slots=True mas Python permite
    # passagem de todos os campos incluindo os "privados" prefixados com ``_``.
    return GcvClient(
        config=config,
        project_root=project_root,
        cache=None,
        rate_limiter=None,
        _api_client=None,
        _import_error=ImportError("grpc DLL nao encontrada: libgrpc.so"),
    )


def _build_pipeline_with_client_p12(
    client: object,
    on_failure: str,
) -> CloudVisionPipeline:
    """Monta um ``CloudVisionPipeline`` com um cliente arbitrário e política dada."""

    options = GcvPresetOptions(
        feature="DOCUMENT_TEXT_DETECTION",
        language_hints=("pt",),
        model=None,
        invalid_feature=False,
        raw_feature=None,
    )
    return CloudVisionPipeline(
        gcv_options=options,
        ocr_config=OcrConfig(),
        client=client,
        on_failure=on_failure,
        ignored_steps_count=0,
        gcv_config_warnings=(),
    )


# -- import_error + skip ------------------------------------------------------


def test_pre_api_import_error_skip_produces_empty_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``_import_error`` cacheado + ``on_failure="skip"`` → resultado vazio válido.

    **Validates: Requirements 6.2, 14.4**

    Quando ``GcvClient._ensure_client()`` re-levanta
    ``GcvError(error="import_error")`` (import lazy já falhou antes), o
    ``CloudVisionPipeline`` com ``on_failure="skip"`` deve produzir um
    ``PipelineResult`` idêntico ao caminho de ``auth_error`` / ``quota_exceeded``
    do P6: texto vazio, confiança zero, última stage ``"output"``.
    """

    project_root = tmp_path_factory.mktemp("p12_import_skip")
    client = _build_import_error_client_p12(project_root)
    pipeline = _build_pipeline_with_client_p12(client, "skip")
    _, context = _make_context(project_root)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    result = pipeline.execute(image, context)

    assert result.ocr_text == ""
    assert result.mean_confidence == 0.0
    assert result.stages[-1].name == "output"
    assert result.metadata["error"] == "import_error"
    assert isinstance(result.metadata["error_message"], str)
    assert len(result.metadata["error_message"]) <= 500


# -- import_error + raise ------------------------------------------------------


def test_pre_api_import_error_raise_propagates(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``_import_error`` cacheado + ``on_failure="raise"`` → ``GcvError`` re-propagada.

    **Validates: Requirements 6.4, 14.4**

    Com ``on_failure="raise"``, a ``GcvError(error="import_error")`` que
    vem de ``_ensure_client`` deve atravessar o pipeline sem ser engolida.
    O código de erro original deve ser preservado.
    """

    project_root = tmp_path_factory.mktemp("p12_import_raise")
    client = _build_import_error_client_p12(project_root)
    pipeline = _build_pipeline_with_client_p12(client, "raise")
    _, context = _make_context(project_root)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(GcvError) as exc_info:
        pipeline.execute(image, context)

    assert exc_info.value.error == "import_error"


# -- auth_error + skip --------------------------------------------------------


def test_pre_api_auth_error_skip_produces_empty_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``resolve_credentials`` falha + ``on_failure="skip"`` → resultado vazio válido.

    **Validates: Requirements 5.3, 6.2**

    Quando credenciais estão ausentes (``resolve_credentials`` levanta
    ``GcvError(error="auth_error")``), o ``CloudVisionPipeline`` com
    ``on_failure="skip"`` deve retornar um resultado vazio mas válido,
    idêntico ao comportamento de qualquer outra falha pré-API.
    """

    from unittest.mock import patch

    from ocr.cloud_vision.app_config import GcvAppConfig
    from ocr.cloud_vision.client import GcvClient

    project_root = tmp_path_factory.mktemp("p12_auth_skip")
    config = GcvAppConfig.from_dict(
        {"cache_enabled": False, "on_failure": "skip", "credentials_path": None},
        project_root,
    )
    client = GcvClient.build(config, project_root)
    pipeline = _build_pipeline_with_client_p12(client, "skip")
    _, context = _make_context(project_root)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    auth_err = GcvError(error="auth_error", message="credenciais ausentes: nenhuma fonte configurada")
    with patch("ocr.cloud_vision.auth.resolve_credentials", side_effect=auth_err):
        result = pipeline.execute(image, context)

    assert result.ocr_text == ""
    assert result.mean_confidence == 0.0
    assert result.stages[-1].name == "output"
    assert result.metadata["error"] == "auth_error"


# -- auth_error + raise --------------------------------------------------------


def test_pre_api_auth_error_raise_propagates(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``resolve_credentials`` falha + ``on_failure="raise"`` → ``GcvError`` re-propagada.

    **Validates: Requirements 5.3, 6.4**

    Com ``on_failure="raise"``, a ``GcvError(error="auth_error")`` vinda
    de ``resolve_credentials`` (dentro de ``_ensure_client``) deve ser
    re-propagada sem modificação pelo pipeline.
    """

    from unittest.mock import patch

    from ocr.cloud_vision.app_config import GcvAppConfig
    from ocr.cloud_vision.client import GcvClient

    project_root = tmp_path_factory.mktemp("p12_auth_raise")
    config = GcvAppConfig.from_dict(
        {"cache_enabled": False, "on_failure": "raise", "credentials_path": None},
        project_root,
    )
    client = GcvClient.build(config, project_root)
    pipeline = _build_pipeline_with_client_p12(client, "raise")
    _, context = _make_context(project_root)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    auth_err = GcvError(error="auth_error", message="credenciais ausentes")
    with patch("ocr.cloud_vision.auth.resolve_credentials", side_effect=auth_err):
        with pytest.raises(GcvError) as exc_info:
            pipeline.execute(image, context)

    assert exc_info.value.error == "auth_error"
