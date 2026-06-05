"""Testes de dispatch do IngredientAnalyzer e do NutritionTextPostProcessor (wave 7, 11.9).

**Validates: R2.5, R12.1, R12.2, R13.1, R13.2**

Quatro exemplos verificam que o ``NutritionReader`` despacha corretamente
os componentes de pós-processamento com base na categoria e nas opções:

- R12.1: ``category_override="ingredient"`` → ``IngredientAnalyzer.analyze``
  é chamado uma vez; ``outcome.ingredient_report`` não é ``None``.
- R12.2: ``category_override="table"`` → ``IngredientAnalyzer.analyze``
  **nunca** é chamado; ``outcome.ingredient_report`` é ``None``.
- R13.1: ``postprocess=True`` → ``NutritionTextPostProcessor.postprocess``
  é chamado ao menos uma vez.
- R13.2: ``postprocess=False`` → ``NutritionTextPostProcessor.postprocess``
  **nunca** é chamado; ``final_ocr_text == final_postprocessed_text``.

Todos os testes usam um preset GCV com stub e ``roi_enabled=False`` para
não depender do Tesseract nem de credenciais reais.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ingredients.analyzer import IngredientReport
from nutrition.format_detector import DetectedFormat, FormatDetector
from nutrition.presets import PresetRepository
from nutrition.reader import NutritionReader, ReaderOptions
from imaging.roi import RoiDetectionConfig
from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.client import GcvClient


# ---------------------------------------------------------------------------
# Constantes de conteúdo
# ---------------------------------------------------------------------------

# Texto de ingredientes para categoria "ingredient".
_INGREDIENT_TEXT = (
    "acucar, sal, farinha de trigo, oleo de soja, lecitina de soja"
)

# Resposta GCV sintética com texto de ingredientes.
_INGREDIENT_RESPONSE: dict = {
    "fullTextAnnotation": {
        "text": _INGREDIENT_TEXT,
        "pages": [
            {
                "blocks": [
                    {
                        "paragraphs": [
                            {"words": [{"confidence": 0.90}, {"confidence": 0.88}]}
                        ]
                    }
                ]
            }
        ],
    },
}

# Resposta GCV com texto nutricional para categoria "table".
_TABLE_RESPONSE: dict = {
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
                            {"words": [{"confidence": 0.85}, {"confidence": 0.80}]}
                        ]
                    }
                ]
            }
        ],
    },
}


# ---------------------------------------------------------------------------
# Helper de construção do reader
# ---------------------------------------------------------------------------


def _build_reader(
    project_root: Path,
    category: str,
    gcv_response: dict,
) -> tuple[NutritionReader, Path]:
    """Constrói NutritionReader com preset GCV na categoria especificada.

    A categoria do diretório de presets deve corresponder ao
    ``preset_category`` calculado pelo reader:

    - ``category_override="ingredient"`` → ``preset_category="ingredients"``
      → diretório ``config/presets/ingredients/``
    - ``category_override="table"`` → ``preset_category="table"``
      → diretório ``config/presets/table/``

    Para simplificar, o caller passa a categoria do diretório de presets
    diretamente.
    """

    preset_dir = project_root / "config" / "presets" / category
    preset_dir.mkdir(parents=True)
    (project_root / "extractions").mkdir(exist_ok=True)
    (project_root / "images" / "pipeline").mkdir(parents=True, exist_ok=True)
    subjects_dir = project_root / "subjects"
    subjects_dir.mkdir(exist_ok=True)

    gcv_preset = {
        "name": "00_gcv",
        "description": "Preset GCV para teste de dispatch",
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
            "min_mean_confidence": 0,
            "min_text_length": 0,
            "min_keyword_hits": 0,
            "expected_keywords": [],
        },
    }
    (preset_dir / "00_gcv.json").write_text(
        json.dumps(gcv_preset, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_root / "config").mkdir(exist_ok=True)
    (project_root / "config" / "app.json").write_text(
        json.dumps({"gcv": {"cache_enabled": False, "on_failure": "skip"}}),
        encoding="utf-8",
    )

    preset_repo = PresetRepository(project_root / "config" / "presets")

    class _MockFD(FormatDetector):
        def detect(self, image: np.ndarray) -> DetectedFormat:  # type: ignore[override]
            # Usado apenas quando category_override não está setado.
            return DetectedFormat(
                category="table", score=1.0, grid_density=0.0, reasoning="mock"
            )

    roi_config = RoiDetectionConfig(
        prototxt_path=None, weights_path=None, pb_path=None, pbtxt_path=None,
        confidence_threshold=0.2, target_class_names=(), use_contour_fallback=False,
    )
    gcv_app_config = GcvAppConfig.from_dict(
        {"cache_enabled": False, "on_failure": "skip"},
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

    image = np.full((32, 32, 3), 200, dtype=np.uint8)
    image_path = subjects_dir / "test_label.png"
    cv2.imwrite(str(image_path), image)

    return reader, image_path


# ---------------------------------------------------------------------------
# R12.1 — categoria "ingredient" aciona IngredientAnalyzer
# ---------------------------------------------------------------------------


def test_ingredient_category_triggers_analyzer(tmp_path: Path) -> None:
    """``category_override="ingredient"`` → ``IngredientAnalyzer.analyze`` chamado.

    **Validates: R12.1**

    O ``NutritionReader`` deve chamar ``IngredientAnalyzer.analyze`` quando
    a categoria detectada/forçada é ``"ingredient"`` e o reader tem uma
    instância de ``IngredientAnalyzer`` (``ontology_diabetes.json`` existe).
    ``outcome.ingredient_report`` não deve ser ``None``.
    """

    reader, image_path = _build_reader(
        tmp_path,
        category="ingredients",
        gcv_response=_INGREDIENT_RESPONSE,
    )

    # O reader.ingredient_analyzer fica None quando config/ontology_diabetes.json
    # não existe no tmp_path. Injetamos um mock diretamente no reader para
    # isolar o teste da presença do arquivo de ontologia real — o invariante
    # que queremos verificar é o *despacho*, não o carregamento da ontologia.
    fake_report = MagicMock(spec=IngredientReport)
    fake_report.tokens_found = ["acucar", "sal"]
    fake_report.to_text_report.return_value = "=== ANALISE ==="
    fake_report.to_dict.return_value = {"risco_global": "ALTO"}

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = fake_report
    reader.ingredient_analyzer = mock_analyzer

    outcome = reader.read(
        image_path,
        ReaderOptions(category_override="ingredient", roi_enabled=False),
    )

    assert mock_analyzer.analyze.call_count == 1, (
        f"IngredientAnalyzer.analyze deveria ser chamado 1x; "
        f"foi chamado {mock_analyzer.analyze.call_count}x"
    )
    assert outcome.ingredient_report is not None, (
        "outcome.ingredient_report não deveria ser None para categoria ingredient"
    )


# ---------------------------------------------------------------------------
# R12.2 — categoria "table" NÃO aciona IngredientAnalyzer
# ---------------------------------------------------------------------------


def test_table_category_does_not_trigger_analyzer(tmp_path: Path) -> None:
    """``category_override="table"`` → ``IngredientAnalyzer.analyze`` não chamado.

    **Validates: R12.2**

    Para categorias diferentes de ``"ingredient"``, o reader não deve
    disparar a análise clínica. ``outcome.ingredient_report`` deve
    ser ``None``.
    """

    reader, image_path = _build_reader(
        tmp_path,
        category="table",
        gcv_response=_TABLE_RESPONSE,
    )

    with patch("ingredients.IngredientAnalyzer.analyze") as mock_analyze:
        outcome = reader.read(
            image_path,
            ReaderOptions(category_override="table", roi_enabled=False),
        )

    assert mock_analyze.call_count == 0, (
        f"IngredientAnalyzer.analyze não deveria ser chamado para categoria 'table'; "
        f"foi chamado {mock_analyze.call_count}x"
    )
    assert outcome.ingredient_report is None, (
        "outcome.ingredient_report deveria ser None para categoria table"
    )


# ---------------------------------------------------------------------------
# R13.1 — postprocess=True aplica o pós-processador
# ---------------------------------------------------------------------------


def test_postprocess_true_applies_postprocessor(tmp_path: Path) -> None:
    """``postprocess=True`` → ``NutritionTextPostProcessor.postprocess`` é chamado.

    **Validates: R13.1**

    Com ``postprocess=True`` (default), o reader deve chamar
    ``NutritionTextPostProcessor.postprocess`` para cada tentativa de
    preset e também para o texto vencedor final.
    """

    reader, image_path = _build_reader(
        tmp_path,
        category="table",
        gcv_response=_TABLE_RESPONSE,
    )

    with patch.object(
        reader.text_postprocessor,
        "postprocess",
        wraps=reader.text_postprocessor.postprocess,
    ) as spy:
        reader.read(
            image_path,
            ReaderOptions(postprocess=True, roi_enabled=False),
        )

    assert spy.call_count >= 1, (
        f"NutritionTextPostProcessor.postprocess deveria ser chamado ao menos "
        f"1x com postprocess=True; foi chamado {spy.call_count}x"
    )


# ---------------------------------------------------------------------------
# R13.2 — postprocess=False não aplica o pós-processador
# ---------------------------------------------------------------------------


def test_postprocess_false_skips_postprocessor(tmp_path: Path) -> None:
    """``postprocess=False`` → pós-processador não é chamado; finals idênticos.

    **Validates: R13.2**

    Com ``postprocess=False``, o reader deve devolver
    ``final_ocr_text == final_postprocessed_text`` sem invocar o
    ``NutritionTextPostProcessor``.
    """

    reader, image_path = _build_reader(
        tmp_path,
        category="table",
        gcv_response=_TABLE_RESPONSE,
    )

    with patch.object(
        reader.text_postprocessor,
        "postprocess",
        wraps=reader.text_postprocessor.postprocess,
    ) as spy:
        outcome = reader.read(
            image_path,
            ReaderOptions(postprocess=False, roi_enabled=False),
        )

    assert spy.call_count == 0, (
        f"NutritionTextPostProcessor.postprocess não deveria ser chamado com "
        f"postprocess=False; foi chamado {spy.call_count}x"
    )
    assert outcome.final_ocr_text == outcome.final_postprocessed_text, (
        "final_ocr_text e final_postprocessed_text devem ser idênticos "
        "quando postprocess=False"
    )
