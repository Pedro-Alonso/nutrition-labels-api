"""Property test P14: ``gcv.feature`` inválido não chama a API.

**Validates: Requirements 3.5**

A propriedade afirma que, *para qualquer* string ``s`` que NÃO pertença
ao conjunto ``ALLOWED_FEATURES`` (``{"TEXT_DETECTION",
"DOCUMENT_TEXT_DETECTION"}``) declarada em ``gcv.feature`` de um preset,
``CloudVisionPipeline.execute`` produz um ``PipelineResult`` com:

- ``metadata["error"] == "invalid_feature"`` (Requirement 3.5);
- ``ocr_text == ""``;
- ``mean_confidence == 0.0``;
- ``GcvClient.fetch`` **nunca** é chamado durante a execução
  (curto-circuito antes de qualquer I/O — design.md, seção
  "Resumo das classificações de erro").

Esta é uma falha de configuração, não de chamada à API: por isso o
contrato exige que o pipeline curto-circuite *independentemente* de
``on_failure``, gerando um ``PipelineResult`` vazio mas válido — sem
consultar cache e sem propagar exceção. A cascata Tesseract toma frente
naturalmente porque o ``QualityEvaluator`` enxerga ``("", 0.0)`` e
marca ``passed=False``.

A implementação substitui o ``GcvClient`` real por um stub
(``_CountingGcvClient``) que registra cada invocação de ``fetch`` em
``fetch_calls``. Após a execução, asseveramos que a lista permanece
vazia — qualquer chamada acidental (por bug de fluxo, por exemplo) é
detectada imediatamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from audit.recorder import AuditRecorder
from nutrition.pipelines.base import PipelineContext
from nutrition.pipelines.cloud_vision import CloudVisionPipeline
from ocr.cloud_vision.options import GcvPresetOptions
from ocr.cloud_vision.types import ALLOWED_FEATURES, GcvFetchResult
from ocr.service import OcrConfig
from tests.ocr_engine.gcv.strategies import feature_strings_invalid


class _CountingGcvClient:
    """Stub de ``GcvClient`` que conta invocações de ``fetch``.

    Não simula cache, rate-limit nem inicialização lazy: a propriedade
    P14 exige apenas detectar se ``fetch`` foi (ou não) chamado. Cada
    invocação registra a tupla ``(feature, language_hints)`` em
    ``fetch_calls`` e retorna um ``GcvFetchResult`` sintético — o
    retorno só importa caso haja uma chamada acidental, situação em
    que o teste já falhará na assertion ``fetch_calls == []`` antes de
    o valor ser consumido.

    Manter o stub minúsculo isola P14 da dependência opcional
    ``google-cloud-vision`` (Requirement 14.3); o teste roda mesmo
    sem o SDK instalado.
    """

    def __init__(self) -> None:
        # ``list`` em vez de contador para que mensagens de falha mostrem
        # quais argumentos foram passados na chamada acidental — isso
        # facilita o diagnóstico do shrinking do Hypothesis quando uma
        # regressão for introduzida.
        self.fetch_calls: list[tuple[str, tuple[str, ...]]] = []

    def fetch(
        self,
        png_bytes: bytes,
        feature: str,
        language_hints: list[str] | tuple[str, ...],
    ) -> GcvFetchResult:  # pragma: no cover - executado só em regressão
        # Convertemos ``language_hints`` para tupla imutável para preservar
        # o contrato de ``GcvFetchResult.language_hints`` e evitar que o
        # registro mude por mutações posteriores no caller.
        self.fetch_calls.append((feature, tuple(language_hints)))
        # Retorno arbitrário; não deve ser consumido em nenhuma execução
        # legítima de P14 (a invariante exige que ``fetch`` jamais seja
        # chamado quando ``invalid_feature=True``).
        return GcvFetchResult(
            response_json={},
            cache_hit=False,
            feature=feature,
            language_hints=tuple(language_hints),
        )


def _make_context(
    project_root: Path,
    preset_name: str = "gcv_invalid_feature_preset",
) -> tuple[AuditRecorder, PipelineContext]:
    """Materializa a estrutura mínima de auditoria para uma execução.

    Cria a árvore ``extractions/`` e ``images/pipeline/`` exigida pelo
    ``AuditRecorder`` e devolve o par ``(recorder, context)`` pronto
    para ser passado a ``CloudVisionPipeline.execute``. Cada chamada
    usa um ``project_root`` distinto (``tmp_path_factory.mktemp``)
    para isolar iterações sucessivas do Hypothesis.
    """

    (project_root / "extractions").mkdir(parents=True, exist_ok=True)
    (project_root / "images" / "pipeline").mkdir(parents=True, exist_ok=True)
    input_path = project_root / "subjects" / "label_subject.png"

    recorder = AuditRecorder(
        project_root=project_root,
        input_path=input_path,
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


def _build_pipeline(
    options: GcvPresetOptions,
    client: _CountingGcvClient,
    on_failure: str,
) -> CloudVisionPipeline:
    """Constrói um ``CloudVisionPipeline`` com o stub injetado.

    Centraliza a montagem para evitar duplicação entre os exemplos do
    Hypothesis e os casos determinísticos auxiliares. ``OcrConfig`` é
    passado por uniformidade com os demais pipelines — o caminho GCV
    não consome o objeto.
    """

    return CloudVisionPipeline(
        gcv_options=options,
        ocr_config=OcrConfig(),
        client=client,
        on_failure=on_failure,
        # Preset GCV canônico tem ``steps == []`` (Requirement 3.8); o
        # ``ignored_steps_count`` é gravado em ``metadata`` para
        # auditoria mas não afeta a invariante P14.
        ignored_steps_count=0,
        gcv_config_warnings=(),
    )


# ---------------------------------------------------------------------------
# Property 14 — corpo principal
# ---------------------------------------------------------------------------


@given(invalid_feature=feature_strings_invalid())
@settings(
    max_examples=100,
    deadline=None,
    # ``tmp_path_factory`` é session-scoped e usamos ``mktemp`` para
    # isolar cada exemplo do Hypothesis — o aviso de "function-scoped
    # fixture reused" é falso positivo neste padrão.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_invalid_feature_does_not_call_api(
    tmp_path_factory: pytest.TempPathFactory,
    invalid_feature: str,
) -> None:
    """**Property 14**: ``gcv.feature`` inválido ⇒ API nunca é chamada.

    **Validates: Requirements 3.5**
    """

    # ---------------------------------------------------------------
    # Arrange — pré-condição: o gerador devolve apenas strings fora de
    # ``ALLOWED_FEATURES``. Reafirmamos como invariante de geração para
    # garantir que o teste falsifique exatamente o que pretende: se o
    # gerador for futuramente expandido com casos válidos por engano,
    # esta assertion captura a divergência cedo.
    # ---------------------------------------------------------------
    assert invalid_feature not in ALLOWED_FEATURES

    options = GcvPresetOptions.from_dict({"feature": invalid_feature})
    # Pré-condição estrutural: ``from_dict`` reconhece o valor declarado
    # como inválido e ativa a sinalização (Requirement 3.5 + Property 4).
    # Esta etapa não é o que P14 testa — ela é um pré-requisito que, se
    # quebrar, indicaria regressão em ``GcvPresetOptions.from_dict``
    # (coberto por outros testes), não no pipeline.
    assert options.invalid_feature is True
    assert options.raw_feature is not None

    project_root = tmp_path_factory.mktemp("p14_invalid_feature")
    _, context = _make_context(project_root)

    client = _CountingGcvClient()
    pipeline = _build_pipeline(options, client, on_failure="skip")

    # Imagem mínima válida — o conteúdo é irrelevante porque o pipeline
    # curto-circuita antes do ``encode_png``. Mantemos 16×16 BGR
    # ``uint8`` por consistência com ``strategies.image_arrays``.
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    # ---------------------------------------------------------------
    # Act
    # ---------------------------------------------------------------
    result = pipeline.execute(image, context)

    # ---------------------------------------------------------------
    # Assert — invariantes da Property 14
    # ---------------------------------------------------------------

    # (1) Invariante crítico: ``fetch`` nunca foi chamado. Esta é a
    # propriedade central de P14: o pipeline NÃO consulta cache nem
    # chama a API quando ``invalid_feature=True``.
    assert client.fetch_calls == [], (
        "GcvClient.fetch foi invocado para feature inválida "
        f"{invalid_feature!r}; chamadas registradas: {client.fetch_calls!r}"
    )

    # (2) Resultado vazio: ``ocr_text == ""`` e ``mean_confidence == 0.0``
    # — texto vazio + confiança zero faz o ``QualityEvaluator`` produzir
    # ``passed=False``, deixando a cascata avançar para o próximo preset.
    assert result.ocr_text == ""
    assert result.mean_confidence == 0.0

    # (3) ``metadata["error"] == "invalid_feature"`` — código canônico
    # exigido pelo Requirement 3.5; consumidores externos
    # (``_summary.json``, UI) usam essa string para distinguir o caminho
    # de erro pré-API dos demais.
    assert result.metadata["error"] == "invalid_feature"


# ---------------------------------------------------------------------------
# Casos determinísticos auxiliares
# ---------------------------------------------------------------------------
#
# Os exemplos abaixo travam casos canônicos da Property 14 que a
# estratégia ``feature_strings_invalid()`` cobre estatisticamente, mas
# que vale a pena fixar como regressão explícita: variações de casing,
# strings vazias e o caminho ``on_failure="raise"`` (P14 exige que a
# invariante valha INDEPENDENTE da política de falha).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_feature",
    [
        "",
        "  ",
        "text_detection",
        "Text_Detection",
        "DOCUMENT TEXT DETECTION",
        "OCR",
        "FACE_DETECTION",
        "LABEL_DETECTION",
    ],
)
def test_invalid_feature_skip_mode_no_fetch_call(
    tmp_path_factory: pytest.TempPathFactory,
    invalid_feature: str,
) -> None:
    """``on_failure == "skip"`` ⇒ resultado vazio, ``fetch`` não chamado.

    Caso explícito da Property 14 com a política default
    (``on_failure="skip"``). Cobre variações de casing, strings vazias
    e nomes de outras APIs do GCV (``LABEL_DETECTION``,
    ``FACE_DETECTION``) que poderiam aparecer em JSONs reais por
    confusão do operador.
    """

    options = GcvPresetOptions.from_dict({"feature": invalid_feature})
    assert options.invalid_feature is True

    project_root = tmp_path_factory.mktemp("p14_skip_deterministic")
    _, context = _make_context(project_root)

    client = _CountingGcvClient()
    pipeline = _build_pipeline(options, client, on_failure="skip")
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    result = pipeline.execute(image, context)

    assert client.fetch_calls == []
    assert result.ocr_text == ""
    assert result.mean_confidence == 0.0
    assert result.metadata["error"] == "invalid_feature"


def test_invalid_feature_raise_mode_still_does_not_call_fetch(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``on_failure == "raise"`` ⇒ ``fetch`` ainda não é chamado.

    O Requirement 3.5 e o design (seção "Resumo das classificações de
    erro") deixam explícito que ``invalid_feature`` é falha de
    *configuração*, não de chamada à API: por isso o pipeline deve
    curto-circuitar com ``PipelineResult`` vazio
    **independentemente** de ``on_failure``, sem propagar exceção.

    Este teste falsifica regressões em que algum reescrita futura
    confunda ``invalid_feature`` com falha de API e tente honrar
    ``on_failure="raise"`` re-levantando o erro — comportamento que
    quebraria o contrato de "a cascata Tesseract toma frente" descrito
    em design.md.
    """

    options = GcvPresetOptions.from_dict({"feature": "INVALID_VALUE_XYZ"})
    assert options.invalid_feature is True

    project_root = tmp_path_factory.mktemp("p14_raise_mode")
    _, context = _make_context(project_root)

    client = _CountingGcvClient()
    pipeline = _build_pipeline(options, client, on_failure="raise")
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    # Em modo ``raise``, ``invalid_feature`` ainda é tratado como falha
    # pré-API e produz ``PipelineResult`` vazio — sem propagar exceção.
    result = pipeline.execute(image, context)

    assert client.fetch_calls == []
    assert result.ocr_text == ""
    assert result.mean_confidence == 0.0
    assert result.metadata["error"] == "invalid_feature"


def test_valid_feature_does_call_fetch(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Sanity check inverso: feature válida ⇒ ``fetch`` é chamado uma vez.

    Garante que P14 não passa trivialmente por causa de algum bug que
    desabilite ``fetch`` em todos os caminhos. Aqui exercitamos um
    preset válido (``DOCUMENT_TEXT_DETECTION``) e confirmamos que o
    stub recebe exatamente uma chamada com a feature e os hints
    declarados — discriminando o caminho ``invalid_feature=False`` do
    caminho testado em P14.
    """

    options = GcvPresetOptions.from_dict(
        {"feature": "DOCUMENT_TEXT_DETECTION", "language_hints": ["pt"]}
    )
    assert options.invalid_feature is False

    project_root = tmp_path_factory.mktemp("p14_valid_sanity")
    _, context = _make_context(project_root)

    client = _CountingGcvClient()
    pipeline = _build_pipeline(options, client, on_failure="skip")
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    pipeline.execute(image, context)

    # Exatamente uma chamada com os argumentos canônicos do preset.
    assert len(client.fetch_calls) == 1
    feature_arg, hints_arg = client.fetch_calls[0]
    assert feature_arg == "DOCUMENT_TEXT_DETECTION"
    assert hints_arg == ("pt",)
