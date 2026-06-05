"""Property test P1: contrato de ``PipelineResult`` e stages do GCV.

**Validates: Requirements 1.4, 1.5, 10.3**

A propriedade afirma que, *para qualquer* imagem BGR ``uint8`` plausível
e *qualquer* resposta sintética da Google Cloud Vision API (com chaves
em ``camelCase``, da forma de ``MessageToDict(AnnotateImageResponse)``),
um ``CloudVisionPipeline`` construído com um ``GcvClient`` stub e uma
política ``on_failure="skip"`` produz um ``PipelineResult`` que:

1. É instância de :class:`nutrition.pipelines.base.PipelineResult`
   (Requirement 1.4 — contrato canônico).
2. Tem todos os campos obrigatórios populados com os tipos corretos
   (``ocr_text: str``, ``mean_confidence: float``, ``stages: list
   [StageRecord]``, ``final_image: np.ndarray``, ``metadata: dict``)
   — Requirement 1.4.
3. Possui ``stages[0].name == "input"`` (stage 01 fixa, base de
   comparação visual) e ``stages[-1].name == "output"`` (stage final
   canônica), conforme Requirements 1.5 e 10.3.
4. Tem ``mean_confidence`` em ``[0, 100]`` — alinhado à escala
   exigida pelo ``QualityEvaluator`` e propagada pelo parser.

A invariante combina o que o pipeline garante em sucesso (chamada
real ou cache hit do stub) — a auditoria simétrica entre os dois é
explorada em P9 (task 10.5). O foco aqui é o **contrato** observável
do ``PipelineResult``: qualquer regressão que altere a forma do
resultado (campo faltando, tipo errado, primeira/última stage com
rótulo diferente, ``mean_confidence`` fora da escala) é detectada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from audit.recorder import AuditRecorder
from nutrition.pipelines.base import (
    PipelineContext,
    PipelineResult,
    StageRecord,
)
from nutrition.pipelines.cloud_vision import CloudVisionPipeline
from ocr.cloud_vision.options import GcvPresetOptions
from ocr.cloud_vision.types import ALLOWED_FEATURES, GcvFetchResult
from ocr.service import OcrConfig
from tests.ocr_engine.gcv.strategies import (
    bcp47_hints,
    gcv_response_dict,
    image_arrays,
)


# ---------------------------------------------------------------------------
# Stub do ``GcvClient`` — duck typing sobre o método ``fetch``
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _StubGcvClient:
    """Stub determinístico do ``GcvClient`` para isolar o pipeline da rede.

    O ``CloudVisionPipeline`` depende apenas do método
    ``fetch(png_bytes, feature, language_hints) -> GcvFetchResult`` do
    cliente (duck typing); este stub satisfaz essa interface devolvendo
    uma resposta sintética injetada no construtor. ``fetch_calls``
    grava cada invocação para que o teste possa verificar que o
    pipeline:

    - chamou ``fetch`` exatamente uma vez (sucesso, sem retries),
    - propagou a ``feature`` declarada no preset,
    - serializou ``language_hints`` como tupla ordem-sensível alinhada
      ao contrato de cache (Requirement 7.3).
    """

    response_json: dict
    cache_hit: bool = False
    fetch_calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def fetch(
        self,
        png_bytes: bytes,
        feature: str,
        language_hints: list[str] | tuple[str, ...],
    ) -> GcvFetchResult:
        # Coerção defensiva: o pipeline chama ``list(self.gcv_options
        # .language_hints)`` antes de repassar; preservamos a ordem em
        # tupla imutável para compor o ``GcvFetchResult`` canônico.
        hints_tuple = tuple(language_hints)
        self.fetch_calls.append((feature, hints_tuple))
        return GcvFetchResult(
            response_json=self.response_json,
            cache_hit=self.cache_hit,
            feature=feature,
            language_hints=hints_tuple,
        )


# ---------------------------------------------------------------------------
# Helpers de construção
# ---------------------------------------------------------------------------


def _build_recorder(project_root: Path, preset_name: str) -> tuple[
    AuditRecorder, "PipelineContext"
]:
    """Monta um ``AuditRecorder`` real e o ``PipelineContext`` associado.

    O recorder grava em uma sub-árvore de ``project_root`` exclusiva
    para esta tentativa. Como o pipeline GCV cria 4 stages
    (``input``, ``gcv_boxes_overlay``, ``gcv_response``, ``output``)
    e a estrutura mínima de diretórios já é provida pelo
    ``AuditRecorder`` em ``__init__``, não precisamos pré-criar
    pastas além do ``project_root``.

    O ``input_path`` aponta para um nome de arquivo plausível dentro
    de ``subjects/`` — o arquivo NÃO precisa existir em disco; o
    ``AuditRecorder`` consome apenas ``stem`` e ``suffix`` para o
    naming de artefatos.
    """

    input_path = project_root / "subjects" / "subject_p1.png"
    recorder = AuditRecorder(project_root, input_path, clean_previous=True)
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
    response_json: dict,
    feature: str,
    language_hints: tuple[str, ...],
    cache_hit: bool,
) -> tuple[CloudVisionPipeline, _StubGcvClient]:
    """Constrói o pipeline GCV sob teste com o stub injetado.

    ``GcvPresetOptions`` é instanciado diretamente (sem passar pelo
    ``from_dict``) para que o teste isole o contrato de
    ``execute`` da lógica de coerção de presets — exercida em P4
    (task 2.4) e P14 (task 10.7). ``invalid_feature=False`` garante
    que o pipeline siga o caminho de sucesso (chamada à ``fetch``);
    o caminho ``invalid_feature`` é objeto de P14.
    """

    options = GcvPresetOptions(
        feature=feature,
        language_hints=language_hints,
        model=None,
        invalid_feature=False,
        raw_feature=None,
    )
    client = _StubGcvClient(response_json=response_json, cache_hit=cache_hit)
    pipeline = CloudVisionPipeline(
        gcv_options=options,
        ocr_config=OcrConfig(),
        client=client,
        on_failure="skip",
        ignored_steps_count=0,
    )
    return pipeline, client


# ---------------------------------------------------------------------------
# Property 1 — Validates: Requirements 1.4, 1.5, 10.3
# ---------------------------------------------------------------------------


@given(
    image=image_arrays(),
    response_json=gcv_response_dict(),
    feature=st.sampled_from(ALLOWED_FEATURES),
    language_hints=bcp47_hints(),
    cache_hit=st.booleans(),
)
@settings(
    max_examples=50,
    deadline=None,
    # ``tmp_path_factory`` é session-scoped, mas o helper continua a
    # ser invocado dentro do test que recebe a fixture function-scoped
    # ``recorder`` interna. Suprimir o health check é seguro: cada
    # exemplo grava em uma sub-árvore distinta via ``mktemp``.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_pipeline_result_contract(
    tmp_path_factory: pytest.TempPathFactory,
    image: np.ndarray,
    response_json: dict,
    feature: str,
    language_hints: tuple[str, ...],
    cache_hit: bool,
) -> None:
    """**Property 1**: contrato de ``PipelineResult`` e stages.

    **Validates: Requirements 1.4, 1.5, 10.3**

    Para qualquer imagem BGR ``uint8`` e qualquer resposta GCV
    sintética bem-formada, o ``CloudVisionPipeline`` em modo
    ``on_failure="skip"`` produz um ``PipelineResult`` cujos campos
    obrigatórios obedecem ao contrato canônico (Requirement 1.4),
    cujas stages começam em ``"input"`` (stage 01) e terminam em
    ``"output"`` (Requirements 1.5 e 10.3) e cuja
    ``mean_confidence`` está em ``[0, 100]`` — escala consumida
    pelo ``QualityEvaluator``.

    A propriedade NÃO depende dos detalhes específicos do
    ``response_json`` (texto, número de palavras, confidências):
    qualquer resposta válida deve produzir um resultado em
    conformidade. Variar ``cache_hit`` exercita as duas
    ramificações de auditoria simétrica (Requirement 10.4) — a
    análise específica desse caso é objeto de P9 (task 10.5).
    """

    project_root = tmp_path_factory.mktemp("p1_pipeline_contract")

    preset_name = "gcv_doc_text"
    recorder, context = _build_recorder(project_root, preset_name)
    pipeline, client = _build_pipeline(
        response_json=response_json,
        feature=feature,
        language_hints=language_hints,
        cache_hit=cache_hit,
    )

    result = pipeline.execute(image, context)

    # ------------------------------------------------------------------
    # Invariante 1: tipo canônico (Requirement 1.4).
    # ------------------------------------------------------------------
    assert isinstance(result, PipelineResult), (
        f"execute deveria retornar PipelineResult; obteve {type(result)!r}"
    )

    # ------------------------------------------------------------------
    # Invariante 2: campos obrigatórios com tipos corretos
    # (Requirement 1.4 — "todos os campos obrigatórios").
    # ------------------------------------------------------------------
    assert isinstance(result.ocr_text, str), (
        f"ocr_text deveria ser str; obteve {type(result.ocr_text)!r}"
    )
    assert isinstance(result.mean_confidence, float), (
        f"mean_confidence deveria ser float; obteve {type(result.mean_confidence)!r}"
    )
    assert isinstance(result.stages, list), (
        f"stages deveria ser list; obteve {type(result.stages)!r}"
    )
    assert all(isinstance(stage, StageRecord) for stage in result.stages), (
        "todos os elementos de stages deveriam ser StageRecord"
    )
    assert isinstance(result.final_image, np.ndarray), (
        f"final_image deveria ser np.ndarray; obteve {type(result.final_image)!r}"
    )
    assert isinstance(result.metadata, dict), (
        f"metadata deveria ser dict; obteve {type(result.metadata)!r}"
    )

    # ------------------------------------------------------------------
    # Invariante 3: contrato das stages (Requirements 1.5 e 10.3).
    # No caminho de sucesso o pipeline grava 4 stages
    # (``input``, ``gcv_boxes_overlay``, ``gcv_response``, ``output``).
    # Validamos apenas o que P1 declara: comprimento ≥ 2, primeira
    # stage = ``input`` e última stage = ``output``. A simetria entre
    # cache hit e miss (que exige 4 stages exatas) é objeto de P9.
    # ------------------------------------------------------------------
    assert len(result.stages) >= 2, (
        f"stages deveria ter ao menos 2 entradas (input e output); "
        f"obteve {len(result.stages)}: "
        f"{[s.name for s in result.stages]}"
    )
    assert result.stages[0].name == "input", (
        f"stages[0].name deveria ser 'input'; obteve "
        f"{result.stages[0].name!r}"
    )
    assert result.stages[-1].name == "output", (
        f"stages[-1].name deveria ser 'output'; obteve "
        f"{result.stages[-1].name!r}"
    )
    # Ordem dos ``order`` é cronológica (1, 2, ...) — sem isso, o
    # ``_summary.json`` exibido pelo controller fica desalinhado.
    assert [stage.order for stage in result.stages] == list(
        range(1, len(result.stages) + 1)
    ), (
        f"stages.order deveria ser sequencial 1..N; obteve "
        f"{[s.order for s in result.stages]}"
    )

    # ------------------------------------------------------------------
    # Invariante 4: ``mean_confidence`` em ``[0, 100]``.
    # Não basta ser ``float``: precisa ser finito e dentro da escala
    # do ``QualityEvaluator`` (que multiplica por 100 internamente
    # se viesse em [0, 1]). O parser aplica ``min(.., 100.0)`` para
    # defender o limite superior contra erro de ponto flutuante
    # (Requirement 9.9 reforça o teto via parser).
    # ------------------------------------------------------------------
    assert math.isfinite(result.mean_confidence), (
        f"mean_confidence deveria ser finito; obteve {result.mean_confidence!r}"
    )
    assert 0.0 <= result.mean_confidence <= 100.0, (
        f"mean_confidence={result.mean_confidence!r} fora da escala [0, 100]"
    )

    # ------------------------------------------------------------------
    # Sanity check do stub: garante que o pipeline efetivamente
    # delegou a chamada (sem retries, sem chamadas extras), passando
    # a feature e hints declarados nas opções. Isso protege a
    # propriedade contra implementações degeneradas que retornariam
    # um ``PipelineResult`` vazio sem sequer consultar o cliente.
    # ------------------------------------------------------------------
    assert len(client.fetch_calls) == 1, (
        f"client.fetch deveria ser chamado exatamente 1 vez; "
        f"obteve {len(client.fetch_calls)} chamadas"
    )
    call_feature, call_hints = client.fetch_calls[0]
    assert call_feature == feature
    assert call_hints == language_hints


# ---------------------------------------------------------------------------
# Caso determinístico auxiliar — sanity check minimalista
# ---------------------------------------------------------------------------


def test_pipeline_result_contract_minimal_response(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Sanity check determinístico do contrato de ``PipelineResult``.

    **Validates: Requirements 1.4, 1.5, 10.3**

    Caso minimalista com resposta GCV "vazia mas bem-formada":
    ``fullTextAnnotation.text == ""`` e ``pages == []``. O parser
    devolve ``ocr_text=""`` e ``mean_confidence=0.0`` (Requirement
    9.5). Mesmo nesse cenário degenerado o pipeline preserva o
    contrato — primeira stage ``input``, última ``output`` — e o
    ``PipelineResult`` permanece válido. Útil como guarda contra
    regressões sem depender da geração aleatória do Hypothesis.
    """

    project_root = tmp_path_factory.mktemp("p1_pipeline_contract_minimal")
    image = np.full((32, 32, 3), 200, dtype=np.uint8)
    response_json = {
        "fullTextAnnotation": {"text": "", "pages": []},
        "textAnnotations": [],
    }

    preset_name = "gcv_doc_text_minimal"
    _, context = _build_recorder(project_root, preset_name)
    pipeline, client = _build_pipeline(
        response_json=response_json,
        feature="DOCUMENT_TEXT_DETECTION",
        language_hints=("pt",),
        cache_hit=False,
    )

    result = pipeline.execute(image, context)

    assert isinstance(result, PipelineResult)
    assert isinstance(result.ocr_text, str)
    assert result.ocr_text == ""
    assert isinstance(result.mean_confidence, float)
    assert result.mean_confidence == 0.0
    assert result.stages[0].name == "input"
    assert result.stages[-1].name == "output"
    assert isinstance(result.final_image, np.ndarray)
    assert result.final_image.shape == image.shape
    assert isinstance(result.metadata, dict)
    assert len(client.fetch_calls) == 1


# ---------------------------------------------------------------------------
# Property 2 — Validates: Requirements 1.7, 2.5
# ---------------------------------------------------------------------------
# P2: ``apply_operation`` é NUNCA chamado pelo ``CloudVisionPipeline``;
# ``metadata["ignored_steps_count"]`` deve ser igual ao valor injetado no
# construtor (independentemente de qual seja — 0, 1, 5 ou qualquer N inteiro
# não-negativo). O pipeline ignora os ``steps`` do preset por contrato
# (Requirements 1.6 e 1.7) e apenas registra a contagem para auditoria.
# ---------------------------------------------------------------------------


@given(ignored_steps_count=st.integers(min_value=0, max_value=15))
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_apply_operation_never_called_and_ignored_count_correct(
    tmp_path_factory: pytest.TempPathFactory,
    ignored_steps_count: int,
) -> None:
    """**Property 2**: ``apply_operation`` nunca é chamado; ``ignored_steps_count`` correto.

    **Validates: Requirements 1.6, 1.7, 2.5**

    Para qualquer valor de ``ignored_steps_count`` (0–15), o
    ``CloudVisionPipeline`` deve:

    1. Nunca invocar ``nutrition.pipelines.linear.apply_operation`` —
       o pipeline GCV não aplica operações do ``OPERATION_REGISTRY`` à
       imagem (Requirement 1.6). A prova é feita via ``patch``: se alguma
       refatoração acidental introduzir uma chamada, o mock a registra e o
       assert falha imediatamente.
    2. Expor ``result.metadata["ignored_steps_count"] == ignored_steps_count``
       — a contagem é meramente auditada, não executada (Requirement 1.7).
    """

    from unittest.mock import MagicMock, patch

    project_root = tmp_path_factory.mktemp("p2_ignored_steps")
    preset_name = "gcv_doc_text"
    _, context = _build_recorder(project_root, preset_name)

    # Imagem e resposta fixas — o conteúdo não importa para esta propriedade.
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    response_json = {"fullTextAnnotation": {"text": "", "pages": []}}

    options = GcvPresetOptions(
        feature="DOCUMENT_TEXT_DETECTION",
        language_hints=("pt",),
        model=None,
        invalid_feature=False,
        raw_feature=None,
    )
    from ocr.cloud_vision.types import GcvFetchResult

    class _FixedStub:
        def fetch(self, png_bytes: bytes, feature: str, language_hints: object) -> GcvFetchResult:
            return GcvFetchResult(
                response_json=response_json,
                cache_hit=False,
                feature=feature,
                language_hints=tuple(language_hints),
            )

    pipeline = CloudVisionPipeline(
        gcv_options=options,
        ocr_config=OcrConfig(),
        client=_FixedStub(),
        on_failure="skip",
        ignored_steps_count=ignored_steps_count,
    )

    with patch("nutrition.pipelines.linear.apply_operation") as mock_apply:
        result = pipeline.execute(image, context)

    # Invariante 1: ``apply_operation`` do ``LinearPipeline`` nunca foi tocado.
    assert mock_apply.call_count == 0, (
        f"apply_operation foi chamado {mock_apply.call_count}x; "
        "CloudVisionPipeline não deve usar o OPERATION_REGISTRY"
    )

    # Invariante 2: ``ignored_steps_count`` reflete exatamente o valor injetado.
    assert result.metadata["ignored_steps_count"] == ignored_steps_count, (
        f"metadata['ignored_steps_count']={result.metadata['ignored_steps_count']!r} "
        f"!= {ignored_steps_count!r}"
    )


def test_apply_operation_never_called_deterministic(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Caso determinístico do P2: ``ignored_steps_count=5`` → sem chamadas a ``apply_operation``.

    **Validates: Requirements 1.6, 1.7, 2.5**

    Fixa o caso ``N=5`` para que o relatório de falha do pytest aponte
    diretamente para o caso concreto, sem depender do shrinking do Hypothesis.
    Complementa o teste de propriedade acima como guarda contra regressões.
    """

    from unittest.mock import MagicMock, patch

    project_root = tmp_path_factory.mktemp("p2_ignored_steps_det")
    preset_name = "gcv_doc_text_det"
    _, context = _build_recorder(project_root, preset_name)

    image = np.full((32, 32, 3), 200, dtype=np.uint8)
    response_json = {
        "fullTextAnnotation": {"text": "valor energetico 75 kcal", "pages": []},
    }
    pipeline, _ = _build_pipeline(
        response_json=response_json,
        feature="DOCUMENT_TEXT_DETECTION",
        language_hints=("pt",),
        cache_hit=False,
    )
    # Substituímos o pipeline construído para ter ignored_steps_count=5.
    options = GcvPresetOptions(
        feature="DOCUMENT_TEXT_DETECTION",
        language_hints=("pt",),
        model=None,
        invalid_feature=False,
        raw_feature=None,
    )
    from ocr.cloud_vision.types import GcvFetchResult

    class _Stub:
        def fetch(self, png_bytes: bytes, feature: str, language_hints: object) -> GcvFetchResult:
            return GcvFetchResult(
                response_json=response_json,
                cache_hit=False,
                feature=feature,
                language_hints=tuple(language_hints),
            )

    pipeline5 = CloudVisionPipeline(
        gcv_options=options,
        ocr_config=OcrConfig(),
        client=_Stub(),
        on_failure="skip",
        ignored_steps_count=5,
    )

    with patch("nutrition.pipelines.linear.apply_operation") as mock_apply:
        result = pipeline5.execute(image, context)

    assert mock_apply.call_count == 0
    assert result.metadata["ignored_steps_count"] == 5
