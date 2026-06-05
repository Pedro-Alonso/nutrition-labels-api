"""Exemplos de integração: ordenação de presets GCV na cascata (wave 7, 11.7).

**Validates: R11.1, R11.2, R11.4**

Três exemplos determinísticos cobrem o comportamento da cascata quando
presets GCV e Tesseract coexistem num mesmo ``PresetRepository``:

- R11.1: preset GCV com prioridade alta (menor número) e resposta de
  qualidade → vence e para a cascata (``stop_on_first_pass=True``).
- R11.2: ``stop_on_first_pass=False`` → todos os presets rodam; GCV
  ainda vence pelo maior ``score`` contínuo.
- R11.4: GCV falha com ``on_failure="skip"`` → cascata avança para o
  preset linear seguinte que produz texto de qualidade.

Os testes constroem um ``NutritionReader`` mínimo (sem Tesseract real):

- Preset GCV: tipo ``cloud_vision``, ``priority=5``.
- Preset linear de fallback: ``LinearPipeline`` injetado via
  ``unittest.mock.patch`` para retornar um ``PipelineResult`` fixo sem
  chamar o Tesseract.
- ``FormatDetector`` substituído por um mock que retorna sempre a
  categoria configurada.
- ``roi_enabled=False`` para evitar a inicialização do ``RoiDetector``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nutrition.format_detector import DetectedFormat, FormatDetector
from nutrition.pipelines.base import PipelineResult, StageRecord
from nutrition.presets import PresetRepository
from nutrition.reader import NutritionReader, ReaderOptions
from imaging.roi import RoiDetectionConfig
from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.client import GcvClient
from ocr.cloud_vision.types import GcvError


# ---------------------------------------------------------------------------
# Constantes de conteúdo
# ---------------------------------------------------------------------------

# Resposta GCV sintética com texto que passa no QualityEvaluator default
# (comprimento >40, todas as 6 keywords esperadas, confidence=80.0).
_GCV_GOOD_RESPONSE: dict = {
    "fullTextAnnotation": {
        "text": (
            "valor energetico 75 kcal carboidratos 15 g proteinas 1.4 g "
            "gorduras totais 1.0 g gorduras saturadas 0.4 g sodio 120 mg"
        ),
        "pages": [
            {
                "blocks": [
                    {
                        "paragraphs": [
                            {
                                "words": [
                                    {"confidence": 0.85},
                                    {"confidence": 0.80},
                                    {"confidence": 0.78},
                                ]
                            }
                        ]
                    }
                ]
            }
        ],
    },
}

# Resultado vazio que o LinearPipeline stub retorna (falha de qualidade).
_LINEAR_EMPTY_RESULT = PipelineResult(
    ocr_text="",
    mean_confidence=0.0,
    stages=[
        StageRecord(order=1, name="input", op="input", output_path="", params={}),
        StageRecord(order=2, name="output", op="output", output_path="", params={}),
    ],
    final_image=np.zeros((16, 16, 3), dtype=np.uint8),
    metadata={},
)

# Resultado de qualidade para o preset linear de fallback (R11.4).
_LINEAR_GOOD_RESULT = PipelineResult(
    ocr_text=(
        "valor energetico 75 kcal carboidratos 15 g proteinas 1.4 g "
        "gorduras totais 1.0 g gorduras saturadas 0.4 g sodio 120 mg"
    ),
    mean_confidence=80.0,
    stages=[
        StageRecord(order=1, name="input", op="input", output_path="", params={}),
        StageRecord(order=2, name="output", op="output", output_path="", params={}),
    ],
    final_image=np.zeros((16, 16, 3), dtype=np.uint8),
    metadata={},
)


# ---------------------------------------------------------------------------
# Helper de construção do reader
# ---------------------------------------------------------------------------


def _make_reader(
    project_root: Path,
    gcv_response: dict,
    on_failure: str = "skip",
    include_linear_fallback: bool = False,
    linear_fallback_priority: int = 50,
) -> tuple[NutritionReader, Path]:
    """Constrói um NutritionReader mínimo com preset GCV (e opcionalmente um linear).

    Cria a estrutura de diretórios necessária, grava os presets em JSON,
    monta um stub de GcvClient e retorna o reader pronto para chamar
    ``read()``.

    Args:
        project_root: Diretório temporário isolado para esta execução.
        gcv_response: Resposta sintética retornada pelo stub da API.
        on_failure: Política GCV (``"skip"`` ou ``"raise"``).
        include_linear_fallback: Se True, grava um segundo preset
            ``linear_table`` na mesma categoria para ser exercitado
            quando o GCV falha/retorna qualidade baixa.
        linear_fallback_priority: Prioridade do preset linear (>= gcv
            priority=5 para garantir que GCV rode primeiro).

    Returns:
        Tupla ``(reader, image_path)`` onde ``image_path`` é o PNG
        escrito em ``project_root/subjects/``.
    """

    # Estrutura de diretórios.
    category = "table"
    preset_dir = project_root / "config" / "presets" / category
    preset_dir.mkdir(parents=True)
    (project_root / "extractions").mkdir(exist_ok=True)
    (project_root / "images" / "pipeline").mkdir(parents=True, exist_ok=True)
    subjects_dir = project_root / "subjects"
    subjects_dir.mkdir(exist_ok=True)

    # Preset GCV (priority=5 → roda antes do linear se present).
    gcv_preset = {
        "name": "00_gcv",
        "description": "GCV preset para teste de cascata",
        "kind": "cloud_vision",
        "priority": 5,
        "steps": [],
        "ocr": {"lang": "por", "psm": 6, "oem": 3},
        "gcv": {
            "feature": "DOCUMENT_TEXT_DETECTION",
            "language_hints": ["pt"],
            "model": None,
        },
        "quality_thresholds": {
            "min_mean_confidence": 75,
            "min_text_length": 40,
            "min_keyword_hits": 3,
            "expected_keywords": [
                "valor energetico", "carboidratos", "proteinas",
                "gorduras totais", "gorduras saturadas", "sodio",
            ],
        },
    }
    (preset_dir / "00_gcv.json").write_text(
        json.dumps(gcv_preset, ensure_ascii=False),
        encoding="utf-8",
    )

    # Preset linear de fallback (opcional, priority=50 → roda depois do GCV).
    if include_linear_fallback:
        linear_preset = {
            "name": "01_linear_fallback",
            "description": "Linear fallback para teste de cascata",
            "kind": "linear_table",
            "priority": linear_fallback_priority,
            "steps": [],
            "ocr": {"lang": "por", "psm": 6, "oem": 3},
            "quality_thresholds": {
                "min_mean_confidence": 65,
                "min_text_length": 40,
                "min_keyword_hits": 3,
                "expected_keywords": [
                    "valor energetico", "carboidratos", "proteinas",
                    "gorduras totais", "gorduras saturadas", "sodio",
                ],
            },
        }
        (preset_dir / "01_linear_fallback.json").write_text(
            json.dumps(linear_preset, ensure_ascii=False),
            encoding="utf-8",
        )

    # config/app.json — apenas o bloco gcv; sem roi real.
    (project_root / "config").mkdir(exist_ok=True)
    app_json: dict = {
        "gcv": {
            "cache_enabled": False,
            "on_failure": on_failure,
            "credentials_path": None,
        }
    }
    (project_root / "config" / "app.json").write_text(
        json.dumps(app_json, ensure_ascii=False),
        encoding="utf-8",
    )

    # PresetRepository, FormatDetector (mock), RoiDetectionConfig mínima.
    preset_repo = PresetRepository(project_root / "config" / "presets")

    class _MockFD(FormatDetector):
        """Stub: retorna sempre category='table' (correspondente ao diretório)."""

        def detect(self, image: np.ndarray) -> DetectedFormat:  # type: ignore[override]
            return DetectedFormat(
                category="table",
                score=1.0,
                grid_density=0.0,
                reasoning="mock",
            )

    roi_config = RoiDetectionConfig(
        prototxt_path=None,
        weights_path=None,
        pb_path=None,
        pbtxt_path=None,
        confidence_threshold=0.2,
        target_class_names=(),
        use_contour_fallback=False,
    )

    # GCV client stub.
    gcv_app_config = GcvAppConfig.from_dict(
        {"cache_enabled": False, "on_failure": on_failure},
        project_root,
    )
    api_stub = MagicMock()
    api_stub.annotate_image.return_value = gcv_response
    gcv_client = GcvClient.build(gcv_app_config, project_root, api_client=api_stub)

    reader = NutritionReader(
        project_root=project_root,
        preset_repo=preset_repo,
        format_detector=_MockFD(),
        roi_config=roi_config,
        gcv_app_config=gcv_app_config,
        gcv_client=gcv_client,
    )

    # Imagem mínima em disco (exigida por read() / AuditRecorder).
    import cv2

    image = np.full((32, 32, 3), 200, dtype=np.uint8)
    image_path = subjects_dir / "test_label.png"
    cv2.imwrite(str(image_path), image)

    return reader, image_path


# ---------------------------------------------------------------------------
# R11.1 — GCV com qualidade alta vence e para a cascata
# ---------------------------------------------------------------------------


def test_gcv_preset_wins_when_quality_passes(tmp_path: Path) -> None:
    """GCV (priority=5) com resposta de qualidade → vence e stop_on_first_pass.

    **Validates: R11.1**

    Com ``stop_on_first_pass=True``, a cascata deve parar no primeiro
    preset cujo ``QualityScore.passed == True``. O preset GCV (priority=5)
    roda antes do linear (priority=50) e produz texto que passa em todos
    os limiares. A cascata deve registrar exatamente 1 attempt.
    """

    reader, image_path = _make_reader(
        tmp_path,
        gcv_response=_GCV_GOOD_RESPONSE,
        on_failure="skip",
        include_linear_fallback=True,
    )

    with patch(
        "nutrition.pipelines.linear.LinearPipeline.execute",
        return_value=_LINEAR_EMPTY_RESULT,
    ):
        outcome = reader.read(
            image_path,
            ReaderOptions(stop_on_first_pass=True, roi_enabled=False),
        )

    assert outcome.winning_preset == "00_gcv", (
        f"GCV deveria vencer; preset vencedor: {outcome.winning_preset!r}"
    )
    assert len(outcome.attempts) == 1, (
        f"Com stop_on_first_pass=True e GCV passando, só 1 attempt esperado; "
        f"obteve {len(outcome.attempts)}"
    )
    assert outcome.passed is True


# ---------------------------------------------------------------------------
# R11.2 — stop_on_first_pass=False roda todos os presets
# ---------------------------------------------------------------------------


def test_stop_on_first_pass_false_runs_all(tmp_path: Path) -> None:
    """``stop_on_first_pass=False`` → ambos os presets executam.

    **Validates: R11.2**

    Com ``stop_on_first_pass=False``, o reader deve executar todos os
    presets independentemente de qualquer preset já ter passado. O preset
    GCV (priority=5) produz boa qualidade; o linear (priority=50) retorna
    texto vazio. O reader deve registrar 2 attempts e eleger o GCV como
    vencedor pelo maior ``score`` contínuo.
    """

    reader, image_path = _make_reader(
        tmp_path,
        gcv_response=_GCV_GOOD_RESPONSE,
        on_failure="skip",
        include_linear_fallback=True,
    )

    with patch(
        "nutrition.pipelines.linear.LinearPipeline.execute",
        return_value=_LINEAR_EMPTY_RESULT,
    ):
        outcome = reader.read(
            image_path,
            ReaderOptions(stop_on_first_pass=False, roi_enabled=False),
        )

    assert len(outcome.attempts) == 2, (
        f"stop_on_first_pass=False deve rodar ambos os presets; "
        f"obteve {len(outcome.attempts)} attempt(s)"
    )
    assert outcome.winning_preset == "00_gcv", (
        f"GCV (maior score) deve vencer; obteve {outcome.winning_preset!r}"
    )


# ---------------------------------------------------------------------------
# R11.4 — GCV falha com skip → cascata avança para o preset linear
# ---------------------------------------------------------------------------


def test_gcv_falls_back_to_linear_on_skip(tmp_path: Path) -> None:
    """GCV falha (``on_failure="skip"``) → linear de fallback vence.

    **Validates: R11.4**

    Quando o stub GCV levanta ``GcvError`` (simula falha de API) e a
    política é ``"skip"``, o pipeline GCV produz resultado vazio
    (``passed=False``). A cascata deve avançar para o preset linear que
    retorna texto de qualidade e se tornar o vencedor.
    """

    # Stub que levanta GcvError em vez de retornar uma resposta válida.
    reader, image_path = _make_reader(
        tmp_path,
        gcv_response=_GCV_GOOD_RESPONSE,  # ignorado; vamos substituir o fetch
        on_failure="skip",
        include_linear_fallback=True,
    )

    # Substituímos o client do reader por um stub que sempre falha.
    class _FailingClient:
        def fetch(self, *args: object, **kwargs: object) -> object:
            raise GcvError(error="generic_error", message="API indisponível")

    reader._gcv_client = _FailingClient()  # type: ignore[assignment]

    with patch(
        "nutrition.pipelines.linear.LinearPipeline.execute",
        return_value=_LINEAR_GOOD_RESULT,
    ):
        outcome = reader.read(
            image_path,
            ReaderOptions(stop_on_first_pass=True, roi_enabled=False),
        )

    assert outcome.winning_preset == "01_linear_fallback", (
        f"Linear fallback deveria vencer após falha do GCV; "
        f"obteve {outcome.winning_preset!r}"
    )
    assert len(outcome.attempts) == 2, (
        f"Ambos os presets devem ter sido tentados; obteve {len(outcome.attempts)}"
    )
