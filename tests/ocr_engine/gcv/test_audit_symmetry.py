"""Property test P9: auditoria simétrica entre cache hit e cache miss.

**Validates: Requirements 9.7, 10.1, 10.2, 10.4, 10.5**

A propriedade afirma que o ``CloudVisionPipeline`` produz **exatamente
os mesmos artefatos de auditoria** quando a resposta da Cloud Vision API
vem do cache (``cache_hit=True``) quanto quando vem de uma chamada real
(``cache_hit=False``), desde que o ``response_json`` seja idêntico nos
dois casos. Em particular:

- Os rótulos das ``stages`` produzidas são iguais entre as duas
  execuções (Requirement 10.4): em ambos os casos o pipeline emite
  ``input``, ``gcv_boxes_overlay``, ``gcv_response`` e ``output``,
  nessa ordem.
- O conjunto de chaves em ``PipelineResult.metadata`` é idêntico entre
  hit e miss (Requirements 9.7 e 10.5): apenas o **valor** de
  ``cache_hit`` difere; ``feature``, ``language_hints``,
  ``block_count``, ``paragraph_count``, ``word_count`` e
  ``gcv_response_path`` permanecem com os mesmos valores derivados do
  mesmo ``response_json``.
- O arquivo apontado por ``gcv_response_path`` existe em disco em
  ambos os casos (Requirement 10.1 + 10.4): cache hit também grava a
  cópia local da resposta crua, garantindo que a inspeção visual da
  pasta ``images/pipeline/<input>/<NN>_<preset>/`` seja
  indistinguível entre execuções com e sem chamada à API.

Para falsificar a invariante, instanciamos dois ``CloudVisionPipeline``
com stubs de ``GcvClient`` que retornam ``GcvFetchResult`` com o mesmo
``response_json`` mas valores opostos de ``cache_hit``. Comparamos os
dois ``PipelineResult`` obtidos para confirmar a simetria.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from audit.recorder import AuditRecorder
from nutrition.pipelines.base import PipelineContext
from nutrition.pipelines.cloud_vision import CloudVisionPipeline
from ocr.cloud_vision.options import GcvPresetOptions
from ocr.cloud_vision.types import ALLOWED_FEATURES, GcvFetchResult
from ocr.service import OcrConfig
from tests.ocr_engine.gcv.strategies import bcp47_hints, gcv_response_dict, image_arrays


# ---------------------------------------------------------------------------
# Stub do ``GcvClient``
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _StubGcvClient:
    """Stub determinístico do ``GcvClient`` para testes do pipeline.

    Não importa o SDK ``google-cloud-vision``: o pipeline só consome a
    interface ``fetch(png_bytes, feature, language_hints) ->
    GcvFetchResult`` (duck typing). Ao injetar este stub, controlamos
    exatamente o ``response_json`` e o flag ``cache_hit`` retornados,
    isolando o pipeline da camada de rede e do cache em disco.

    Attributes:
        response_json: Payload que o pipeline receberá na chamada
            ``fetch``. Compartilhado byte-a-byte entre as duas
            execuções (hit vs miss) para que P9 possa comparar
            metadados derivados.
        cache_hit: Valor de ``cache_hit`` no ``GcvFetchResult``
            devolvido. ``True`` simula resposta vinda do cache;
            ``False`` simula chamada real à API. A diferença
            observável no pipeline deve se restringir a
            ``metadata.cache_hit``.
        feature: ``feature`` reproduzida no ``GcvFetchResult`` —
            espelha o argumento recebido pelo ``fetch`` para que o
            pipeline tenha simetria com chamadas reais.
        calls: Lista de tuplas ``(png_bytes_len, feature, hints)`` para
            que o teste possa inspecionar quantas vezes o stub foi
            invocado e com que argumentos. Não é obrigatório para a
            propriedade P9, mas serve de sanity check defensivo.
    """

    response_json: dict
    cache_hit: bool
    feature: str
    calls: list[tuple[int, str, tuple[str, ...]]]

    def fetch(
        self,
        png_bytes: bytes,
        feature: str,
        language_hints: list[str] | tuple[str, ...],
    ) -> GcvFetchResult:
        # Coercão para tupla imutável: o pipeline passa ``list`` via
        # ``list(self.gcv_options.language_hints)``; preservamos o
        # contrato de ``GcvFetchResult.language_hints`` (tupla).
        hints_tuple: tuple[str, ...] = tuple(language_hints)
        # Registra invocação para inspeção opcional. ``len(png_bytes)``
        # evita guardar bytes longos na lista de calls.
        self.calls.append((len(png_bytes), feature, hints_tuple))
        return GcvFetchResult(
            response_json=self.response_json,
            cache_hit=self.cache_hit,
            feature=feature,
            language_hints=hints_tuple,
        )


# ---------------------------------------------------------------------------
# Helper — execução isolada de uma instância do pipeline
# ---------------------------------------------------------------------------


def _run_pipeline_once(
    project_root: Path,
    image: np.ndarray,
    response_json: dict,
    feature: str,
    language_hints: tuple[str, ...],
    cache_hit: bool,
    *,
    input_slug: str = "p9_subject",
) -> tuple[Any, list[Any]]:
    """Executa o ``CloudVisionPipeline`` em um ``project_root`` isolado.

    Cada execução recebe seu próprio ``project_root`` para que os
    artefatos de hit e miss não se sobrescrevam entre si — é assim
    que comparamos dois conjuntos completos de stages/metadata sem
    precisar deslocar arquivos manualmente.

    Devolve ``(result, stages)`` onde ``result`` é o ``PipelineResult``
    e ``stages`` é a lista cronológica de ``StageRecord``.
    """

    # Estrutura mínima esperada pelo ``AuditRecorder``.
    (project_root / "extractions").mkdir(parents=True, exist_ok=True)
    (project_root / "images" / "pipeline").mkdir(parents=True, exist_ok=True)
    subjects_dir = project_root / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)

    # ``input_path`` não precisa existir como arquivo real porque o
    # ``AuditRecorder`` só usa ``stem``/``suffix`` para compor o slug.
    input_path = subjects_dir / f"{input_slug}.png"

    recorder = AuditRecorder(
        project_root=project_root,
        input_path=input_path,
        clean_previous=True,
    )
    artifacts = recorder.start_attempt(1, "00_gcv_doc_text")

    context = PipelineContext(
        input_path=input_path,
        attempt_index=1,
        preset_name="00_gcv_doc_text",
        recorder=recorder,
        artifacts=artifacts,
    )

    # Opções coerentes com a feature pedida — ``invalid_feature=False``
    # para que o pipeline siga o caminho de sucesso (P9 não cobre o
    # ramo de curto-circuito; isso é P14).
    gcv_options = GcvPresetOptions(
        feature=feature,
        language_hints=language_hints,
        model=None,
        invalid_feature=False,
        raw_feature=feature,
    )

    stub = _StubGcvClient(
        response_json=response_json,
        cache_hit=cache_hit,
        feature=feature,
        calls=[],
    )

    pipeline = CloudVisionPipeline(
        gcv_options=gcv_options,
        ocr_config=OcrConfig(),
        client=stub,  # type: ignore[arg-type]  # duck typing: só usa .fetch
        on_failure="skip",
        ignored_steps_count=0,
    )

    result = pipeline.execute(image, context)
    return result, list(result.stages)


# ---------------------------------------------------------------------------
# Property 9 — Validates: Requirements 9.7, 10.1, 10.2, 10.4, 10.5
# ---------------------------------------------------------------------------


# Chaves canônicas que ``PipelineResult.metadata`` deve sempre conter no
# caminho de sucesso (Requirements 9.7 e 10.5). Deriva do schema definido
# em ``CloudVisionPipeline.execute`` e em ``design.md``. ``cache_hit`` é
# o único campo cujo *valor* legitimamente diverge entre hit e miss; os
# demais devem ser idênticos.
_REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "feature",
        "language_hints",
        "block_count",
        "paragraph_count",
        "word_count",
        "cache_hit",
        "gcv_response_path",
        "ignored_steps_count",
        "confidence_warning",
        "error",
        "error_message",
        "error_secondary",
        "gcv_config_warnings",
    }
)

# Sequência canônica de stages no caminho de sucesso (Requirements 1.5,
# 10.2 e 10.4). O pipeline emite, nesta ordem:
#   01 input → 02 gcv_boxes_overlay → 03 gcv_response → 04 output.
# Cache hit produz exatamente a mesma sequência (Requirement 10.4).
_EXPECTED_STAGE_NAMES: tuple[str, ...] = (
    "input",
    "gcv_boxes_overlay",
    "gcv_response",
    "output",
)


@given(
    image=image_arrays(),
    response_json=gcv_response_dict(),
    feature=st.sampled_from(ALLOWED_FEATURES),
    language_hints=bcp47_hints(),
)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_audit_is_symmetric_between_cache_hit_and_miss(
    tmp_path_factory: pytest.TempPathFactory,
    image: np.ndarray,
    response_json: dict,
    feature: str,
    language_hints: tuple[str, ...],
) -> None:
    """**Property 9**: auditoria simétrica entre cache hit e cache miss.

    **Validates: Requirements 9.7, 10.1, 10.2, 10.4, 10.5**

    Para qualquer ``response_json`` válido e qualquer combinação de
    ``feature``/``language_hints``, executar o ``CloudVisionPipeline``
    com cache hit (``cache_hit=True``) e com cache miss
    (``cache_hit=False``) deve produzir:

    - A mesma sequência de stages (mesmos rótulos, mesma ordem).
    - O mesmo conjunto de chaves em ``metadata``.
    - Os mesmos valores em ``metadata`` para todos os campos exceto
      ``cache_hit`` (cujo valor reflete a origem da resposta) e
      ``gcv_response_path`` (que aponta para arquivos em
      ``project_root`` diferentes — comparamos pelo nome do arquivo).
    - Existência física do arquivo de resposta apontado por
      ``gcv_response_path`` em ambos os casos (cache hit também
      grava a cópia local — Requirement 10.4).
    """

    # ``tmp_path_factory`` é session-scoped; ``mktemp`` produz um
    # diretório único para cada exemplo do Hypothesis para que hit e
    # miss não conflitem entre si nem com iterações anteriores.
    root_miss = tmp_path_factory.mktemp("p9_miss")
    root_hit = tmp_path_factory.mktemp("p9_hit")

    result_miss, stages_miss = _run_pipeline_once(
        project_root=root_miss,
        image=image,
        response_json=response_json,
        feature=feature,
        language_hints=language_hints,
        cache_hit=False,
    )
    result_hit, stages_hit = _run_pipeline_once(
        project_root=root_hit,
        image=image,
        response_json=response_json,
        feature=feature,
        language_hints=language_hints,
        cache_hit=True,
    )

    # ---------------------------------------------------------------
    # Invariante 1: sequência de stages idêntica entre hit e miss.
    # Requirement 10.4 — a auditoria visual da pasta deve ser
    # indistinguível entre as duas origens; isso começa na própria
    # lista de etapas registradas.
    # ---------------------------------------------------------------
    names_miss = tuple(stage.name for stage in stages_miss)
    names_hit = tuple(stage.name for stage in stages_hit)
    assert names_miss == _EXPECTED_STAGE_NAMES, (
        f"miss: stages esperadas {_EXPECTED_STAGE_NAMES!r}; obtidas {names_miss!r}"
    )
    assert names_hit == _EXPECTED_STAGE_NAMES, (
        f"hit: stages esperadas {_EXPECTED_STAGE_NAMES!r}; obtidas {names_hit!r}"
    )
    assert names_miss == names_hit, (
        "stages divergem entre hit e miss — auditoria não é simétrica"
    )

    # ---------------------------------------------------------------
    # Invariante 2: conjunto de chaves em ``metadata`` idêntico.
    # Requirements 9.7 e 10.5 — o schema de metadata é fixo no caminho
    # de sucesso; nenhuma chave pode aparecer/sumir em função de
    # ``cache_hit``.
    # ---------------------------------------------------------------
    keys_miss = frozenset(result_miss.metadata.keys())
    keys_hit = frozenset(result_hit.metadata.keys())
    assert keys_miss == keys_hit, (
        f"chaves de metadata divergem entre hit e miss: "
        f"miss={keys_miss}, hit={keys_hit}"
    )
    # Defensivo: as chaves obrigatórias declaradas no design devem estar
    # todas presentes. Falsifica regressões em que o pipeline pare de
    # popular alguma chave canônica.
    missing_miss = _REQUIRED_METADATA_KEYS - keys_miss
    assert not missing_miss, (
        f"miss: chaves obrigatórias ausentes em metadata: {missing_miss}"
    )

    # ---------------------------------------------------------------
    # Invariante 3: valores idênticos em todos os campos exceto
    # ``cache_hit`` (cujo valor reflete a origem) e
    # ``gcv_response_path`` (que aponta para project_roots distintos).
    # ---------------------------------------------------------------
    for key in _REQUIRED_METADATA_KEYS - {"cache_hit", "gcv_response_path"}:
        assert result_miss.metadata[key] == result_hit.metadata[key], (
            f"metadata[{key!r}] diverge entre hit e miss: "
            f"miss={result_miss.metadata[key]!r}, "
            f"hit={result_hit.metadata[key]!r}"
        )

    # ``cache_hit`` é o único campo cujo valor legitimamente diverge.
    assert result_miss.metadata["cache_hit"] is False
    assert result_hit.metadata["cache_hit"] is True

    # ---------------------------------------------------------------
    # Invariante 4: ``gcv_response_path`` aponta para arquivo existente
    # em ambos os casos. Requirements 10.1 e 10.4 — cache hit também
    # grava a cópia local da resposta crua para que a auditoria seja
    # autocontida (não depende do estado do cache no disco).
    # ---------------------------------------------------------------
    path_miss = Path(result_miss.metadata["gcv_response_path"])
    path_hit = Path(result_hit.metadata["gcv_response_path"])
    assert path_miss.is_file(), (
        f"miss: gcv_response_path={path_miss!r} não existe em disco — "
        "cópia local da resposta crua não foi gravada"
    )
    assert path_hit.is_file(), (
        f"hit: gcv_response_path={path_hit!r} não existe em disco — "
        "cache hit deveria gravar cópia local idêntica (Requirement 10.4)"
    )
    # Os dois paths são absolutos mas vivem em ``project_root``
    # diferentes; o nome do arquivo (input_slug + preset_slug + NN +
    # stage) é o que pode ser comparado para confirmar simetria de
    # naming canônico (Requirement 10.1).
    assert path_miss.name == path_hit.name, (
        f"naming canônico do gcv_response diverge entre hit e miss: "
        f"miss={path_miss.name!r}, hit={path_hit.name!r}"
    )

    # ---------------------------------------------------------------
    # Invariante 5: stages de overlay e response presentes em ambos os
    # caminhos com o mesmo ``op``. Requirement 10.2 (overlay PNG) +
    # Requirement 10.1 (resposta JSON local).
    # ---------------------------------------------------------------
    for stages in (stages_miss, stages_hit):
        ops_by_name = {stage.name: stage.op for stage in stages}
        assert ops_by_name.get("gcv_boxes_overlay") == "gcv_word_boxes", (
            f"overlay PNG não foi registrado com o op canônico: {ops_by_name!r}"
        )
        assert ops_by_name.get("gcv_response") == "gcv_response_dump", (
            f"resposta JSON não foi registrada com o op canônico: {ops_by_name!r}"
        )
