"""Testes de integridade de artefatos em modo batch (wave 7, 11.8).

**Validates: R6.4, R6.10**

Dois exemplos verificam que uma falha de GCV em ``on_failure="raise"``
não corrompe nem apaga os artefatos já produzidos por outras imagens:

- R6.4: ``read()`` bem-sucedida em ``img_a`` grava ``_summary.json``;
  ``read()`` com GCV falhando em ``img_b`` levanta ``GcvError``;
  ``extractions/img_a/_summary.json`` continua íntegro.
- R6.10: quando GCV falha antes de qualquer gravação de cache
  (``cache_enabled=True`` mas fetch levanta antes de ``cache.put``),
  o ``cache_dir`` permanece vazio.

O ``NutritionReader`` usa ``AuditRecorder(clean_previous=True)`` que
apaga apenas a pasta da imagem corrente — as demais nunca são tocadas
pela lógica de cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from nutrition.format_detector import DetectedFormat, FormatDetector
from nutrition.presets import PresetRepository
from nutrition.reader import NutritionReader, ReaderOptions
from imaging.roi import RoiDetectionConfig
from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.client import GcvClient
from ocr.cloud_vision.types import GcvError


# ---------------------------------------------------------------------------
# Resposta sintética de boa qualidade
# ---------------------------------------------------------------------------

_GOOD_RESPONSE: dict = {
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
# Helper de setup
# ---------------------------------------------------------------------------


def _write_tiny_png(path: Path) -> None:
    """Escreve um PNG 32x32 cinza em ``path``."""

    image = np.full((32, 32, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _build_reader_raise_mode(
    project_root: Path,
    api_stub: object,
) -> NutritionReader:
    """Constrói NutritionReader com ``on_failure="raise"`` e stub injetado.

    Usa uma única categoria ``table`` com um preset GCV e sem preset
    Tesseract, para que qualquer falha do stub chegue diretamente ao
    ``on_failure`` do pipeline.
    """

    category = "table"
    preset_dir = project_root / "config" / "presets" / category
    preset_dir.mkdir(parents=True)
    (project_root / "extractions").mkdir(exist_ok=True)
    (project_root / "images" / "pipeline").mkdir(parents=True, exist_ok=True)
    (project_root / "subjects").mkdir(exist_ok=True)

    gcv_preset = {
        "name": "00_gcv",
        "description": "GCV preset para teste de batch",
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
    (project_root / "config").mkdir(exist_ok=True)
    (project_root / "config" / "app.json").write_text(
        json.dumps({"gcv": {"cache_enabled": False, "on_failure": "raise"}}),
        encoding="utf-8",
    )

    preset_repo = PresetRepository(project_root / "config" / "presets")

    class _MockFD(FormatDetector):
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

    gcv_app_config = GcvAppConfig.from_dict(
        {"cache_enabled": False, "on_failure": "raise"},
        project_root,
    )
    gcv_client = GcvClient.build(gcv_app_config, project_root, api_client=api_stub)

    return NutritionReader(
        project_root=project_root,
        preset_repo=preset_repo,
        format_detector=_MockFD(),
        roi_config=roi_config,
        gcv_app_config=gcv_app_config,
        gcv_client=gcv_client,
    )


# ---------------------------------------------------------------------------
# R6.4 — artefatos de img_a preservados quando img_b falha
# ---------------------------------------------------------------------------


def test_raise_mode_leaves_other_image_artifacts_intact(tmp_path: Path) -> None:
    """Falha de GCV em img_b não apaga artefatos de img_a.

    **Validates: R6.4**

    A lógica do ``AuditRecorder(clean_previous=True)`` apaga apenas
    ``extractions/<slug_da_imagem_corrente>/`` antes de cada execução.
    Um erro levantado no meio do processamento de img_b não deve tocar
    em ``extractions/img_a/``.
    """

    call_count = 0

    class _StatefulStub:
        def annotate_image(self, request: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _GOOD_RESPONSE
            raise GcvError(error="generic_error", message="API indisponível")

    reader = _build_reader_raise_mode(tmp_path, _StatefulStub())

    # Imagens de teste.
    img_a = tmp_path / "subjects" / "img_a.png"
    img_b = tmp_path / "subjects" / "img_b.png"
    _write_tiny_png(img_a)
    _write_tiny_png(img_b)

    # img_a: deve ter sucesso e gravar _summary.json.
    outcome_a = reader.read(img_a, ReaderOptions(roi_enabled=False))
    assert outcome_a.summary_path is not None
    summary_a = outcome_a.summary_path
    assert summary_a.exists(), "extractions/img_a/_summary.json deveria existir"

    # Verificar que é JSON válido antes de continuar.
    json.loads(summary_a.read_text(encoding="utf-8"))

    # img_b: deve levantar GcvError.
    with pytest.raises(GcvError) as exc_info:
        reader.read(img_b, ReaderOptions(roi_enabled=False))
    assert exc_info.value.error == "generic_error"

    # Invariante R6.4: artefatos de img_a continuam intactos.
    assert summary_a.exists(), (
        "extractions/img_a/_summary.json foi removido após falha em img_b; "
        "AuditRecorder não deve tocar pastas de outras imagens"
    )
    data = json.loads(summary_a.read_text(encoding="utf-8"))
    assert "winning_preset" in data, "estrutura de _summary.json foi corrompida"


# ---------------------------------------------------------------------------
# R6.10 — cache_dir vazio quando fetch falha antes de cache.put
# ---------------------------------------------------------------------------


def test_raise_mode_does_not_corrupt_cache(tmp_path: Path) -> None:
    """``on_failure="raise"`` com cache habilitado → ``cache_dir`` vazio.

    **Validates: R6.10**

    Quando ``GcvClient.fetch`` levanta antes de chegar ao ``cache.put``
    (erro de API após cache miss), o arquivo de cache não deve ser
    gravado parcialmente. O ``cache_dir`` deve permanecer completamente
    vazio.

    Nota: com ``on_failure="raise"`` o ``CloudVisionPipeline`` propaga
    o ``GcvError`` antes que o reader persista qualquer artefato,
    portanto o ``NutritionReader.read()`` levanta. Testamos apenas que
    o ``cache_dir`` não acumulou entradas.
    """

    cache_dir = tmp_path / "gcv_cache"
    cache_dir.mkdir()

    # Stub que sempre levanta na chamada à API (antes do cache.put).
    class _FailingStub:
        def annotate_image(self, request: dict) -> dict:
            raise GcvError(error="timeout", message="request timed out")

    category = "table"
    preset_dir = tmp_path / "config" / "presets" / category
    preset_dir.mkdir(parents=True)
    (tmp_path / "extractions").mkdir(exist_ok=True)
    (tmp_path / "images" / "pipeline").mkdir(parents=True, exist_ok=True)
    (tmp_path / "subjects").mkdir(exist_ok=True)

    gcv_preset = {
        "name": "00_gcv",
        "kind": "cloud_vision",
        "priority": 5,
        "steps": [],
        "ocr": {"lang": "por", "psm": 6, "oem": 3},
        "gcv": {"feature": "DOCUMENT_TEXT_DETECTION", "language_hints": ["pt"]},
        "quality_thresholds": {},
    }
    (preset_dir / "00_gcv.json").write_text(json.dumps(gcv_preset), encoding="utf-8")
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "app.json").write_text(
        json.dumps({
            "gcv": {
                "cache_enabled": True,
                "cache_dir": str(cache_dir),
                "on_failure": "raise",
            }
        }),
        encoding="utf-8",
    )

    preset_repo = PresetRepository(tmp_path / "config" / "presets")

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
        {"cache_enabled": True, "cache_dir": str(cache_dir), "on_failure": "raise"},
        tmp_path,
    )
    gcv_client = GcvClient.build(gcv_app_config, tmp_path, api_client=_FailingStub())

    reader = NutritionReader(
        project_root=tmp_path,
        preset_repo=preset_repo,
        format_detector=_MockFD(),
        roi_config=roi_config,
        gcv_app_config=gcv_app_config,
        gcv_client=gcv_client,
    )

    image_path = tmp_path / "subjects" / "label.png"
    _write_tiny_png(image_path)

    with pytest.raises(GcvError):
        reader.read(image_path, ReaderOptions(roi_enabled=False))

    # Invariante R6.10: cache_dir sem arquivos.
    files = list(cache_dir.iterdir())
    assert files == [], (
        f"cache_dir contém {len(files)} arquivo(s) após falha pré-cache.put: "
        f"{[f.name for f in files]}"
    )
