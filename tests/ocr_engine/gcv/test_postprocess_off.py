"""Property test P21: postprocess=False → final_ocr_text == final_postprocessed_text (wave 7, 11.10).

**Validates: P21, R2.5**

Quando ``ReaderOptions.postprocess=False``, o ``NutritionReader`` não deve
aplicar o ``NutritionTextPostProcessor`` ao texto OCR vencedor:
``outcome.final_ocr_text == outcome.final_postprocessed_text`` — os dois
campos do ``ReadOutcome`` devem ser idênticos (Requirement R2.5).

No backend REST, ``AuditRecorder`` é um no-op (``NullAuditRecorder`` — ver
``audit/recorder.py``): nenhum arquivo (`final.txt`, `final_postprocessed.txt`)
é gravado em disco e ``outcome.summary_path`` é sempre o sentinela
``Path("/dev/null")``. Os testes abaixo verificam apenas os campos em
memória de ``ReadOutcome`` e o contrato no-op de ``summary_path``.

A property test varia a resposta GCV sintética via ``gcv_response_dict()``
para cobrir textos longos, textos com caracteres substituíveis pelo
postprocessor (``O→0``, ``S→5``) e textos vazios, verificando que em
todos os casos o invariante se mantém.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from nutrition.format_detector import DetectedFormat, FormatDetector
from nutrition.presets import PresetRepository
from nutrition.reader import NutritionReader, ReaderOptions
from imaging.roi import RoiDetectionConfig
from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.client import GcvClient
from tests.ocr_engine.gcv.strategies import gcv_response_dict


# ---------------------------------------------------------------------------
# Helper de construção do reader
# ---------------------------------------------------------------------------


def _make_reader(project_root: Path, gcv_response: dict) -> tuple[NutritionReader, Path]:
    """Constrói NutritionReader mínimo com um preset GCV na categoria 'table'."""

    preset_dir = project_root / "config" / "presets" / "table"
    preset_dir.mkdir(parents=True)
    (project_root / "extractions").mkdir(exist_ok=True)
    (project_root / "images" / "pipeline").mkdir(parents=True, exist_ok=True)
    subjects_dir = project_root / "subjects"
    subjects_dir.mkdir(exist_ok=True)

    preset = {
        "name": "00_gcv",
        "description": "Preset GCV para teste P21",
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
        json.dumps(preset, ensure_ascii=False),
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
# P21 — property test
# ---------------------------------------------------------------------------


@given(gcv_response=gcv_response_dict())
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_postprocess_off_both_finals_identical_property(
    tmp_path_factory: pytest.TempPathFactory,
    gcv_response: dict,
) -> None:
    """**P21**: ``postprocess=False`` → ``final_ocr_text == final_postprocessed_text``.

    **Validates: P21, R2.5**

    Para qualquer resposta GCV sintética, quando ``postprocess=False``:
    1. ``outcome.final_ocr_text == outcome.final_postprocessed_text``.
    2. ``outcome.summary_path`` é o sentinela do ``NullAuditRecorder``
       (``Path("/dev/null")``) — nenhum arquivo é gravado em disco.
    """

    project_root = tmp_path_factory.mktemp("p21_postprocess_off")
    reader, image_path = _make_reader(project_root, gcv_response)

    outcome = reader.read(
        image_path,
        ReaderOptions(postprocess=False, roi_enabled=False),
    )

    # Invariante 1: campos do ReadOutcome.
    assert outcome.final_ocr_text == outcome.final_postprocessed_text, (
        "final_ocr_text e final_postprocessed_text diferem com postprocess=False; "
        f"raw={outcome.final_ocr_text!r:.80}, post={outcome.final_postprocessed_text!r:.80}"
    )

    # Invariante 2: contrato no-op do AuditRecorder — nenhum arquivo gravado.
    assert outcome.summary_path == Path("/dev/null")


# ---------------------------------------------------------------------------
# P21 — caso determinístico com texto que o postprocessor modificaria
# ---------------------------------------------------------------------------


def test_postprocess_off_deterministic(tmp_path: Path) -> None:
    """``postprocess=False`` com texto substituível → finals idênticos e brutos.

    **Validates: P21, R2.5**

    Usa uma resposta cujo texto contém caracteres substituídos pelo
    postprocessor (``O→0``, ``S→5`` via ``_OCR_DIGIT_TABLE``) quando
    ``postprocess=True``. Com ``postprocess=False``, ``final_ocr_text`` e
    ``final_postprocessed_text`` devem ser o OCR bruto original, sem
    nenhuma substituição.
    """

    # Texto com "O" e "S" em posições que o postprocessor substituiria por
    # "0" e "5". O teste não valida as substituições específicas — apenas
    # que ``postprocess=False`` as suprime inteiramente.
    raw_ocr_text = "Carboidratos 1Og Proteinas 1Sg sodio 50 mg"
    gcv_response: dict = {
        "fullTextAnnotation": {
            "text": raw_ocr_text,
            "pages": [
                {
                    "blocks": [
                        {
                            "paragraphs": [
                                {"words": [{"confidence": 0.80}, {"confidence": 0.75}]}
                            ]
                        }
                    ]
                }
            ],
        },
    }

    reader, image_path = _make_reader(tmp_path, gcv_response)

    outcome = reader.read(
        image_path,
        ReaderOptions(postprocess=False, roi_enabled=False),
    )

    # Ambos os finals devem ser o texto OCR bruto, sem modificações.
    assert outcome.final_ocr_text == outcome.final_postprocessed_text
    assert outcome.final_ocr_text == raw_ocr_text, (
        f"Com postprocess=False, final_ocr_text deveria ser o texto bruto; "
        f"obteve {outcome.final_ocr_text!r}"
    )

    # Contrato no-op do AuditRecorder — nenhum arquivo gravado.
    assert outcome.summary_path == Path("/dev/null")
