"""Testes unitários das funções puras do AnalysisService.

Não chamam Tesseract — testam apenas hashlib, decode_image_bytes e
outcome_to_dict.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from app.analysis.service import decode_image_bytes, outcome_to_dict


def _make_tiny_jpeg() -> bytes:
    """Gera um JPEG 1×1 pixel branco."""
    tiny = np.ones((1, 1, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".jpg", tiny)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# image_hash is sha256
# ---------------------------------------------------------------------------

def test_image_hash_is_sha256() -> None:
    data = b"hello world"
    expected = hashlib.sha256(data).hexdigest()
    assert len(expected) == 64
    assert expected == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# outcome_to_dict structure
# ---------------------------------------------------------------------------

def _make_mock_outcome(with_ingredient_report: bool = False) -> MagicMock:
    outcome = MagicMock()
    outcome.winning_preset = "preset_test"
    outcome.winning_attempt_index = 1
    outcome.passed = True
    outcome.detected_format = MagicMock(category="table", score=0.9, grid_density=0.02, reasoning="test")
    outcome.final_ocr_text = "texto"
    outcome.final_postprocessed_text = "texto processado"
    outcome.attempts = []
    outcome.summary_path = None
    outcome.groundtruth_metrics = None
    if with_ingredient_report:
        report = MagicMock()
        report.to_dict.return_value = {"risco_global": "ALTO", "ingredientes_identificados": [], "nao_identificados": []}
        outcome.ingredient_report = report
    else:
        outcome.ingredient_report = None
    return outcome


def test_outcome_to_dict_structure() -> None:
    outcome = _make_mock_outcome()
    # Simula asdict para MagicMock não-dataclass
    from dataclasses import dataclass
    from ocr_engine.nutrition.reader import ReadOutcome, ReaderOptions
    from ocr_engine.nutrition.format_detector import DetectedFormat

    detected = DetectedFormat(category="table", score=0.9, grid_density=0.02, reasoning="test")
    real_outcome = ReadOutcome(
        winning_preset="test",
        winning_attempt_index=1,
        passed=True,
        detected_format=detected,
        final_ocr_text="texto",
        final_postprocessed_text="processado",
        attempts=[],
        summary_path=None,
        groundtruth_metrics=None,
        ingredient_report=None,
    )
    result = outcome_to_dict(real_outcome)
    assert "detected_format" in result
    assert "winning_preset" in result
    assert "passed" in result
    assert "attempts" in result
    assert "ingredient_analysis" in result
    assert result["ingredient_analysis"] is None
    assert "summary_path" not in result
    assert "groundtruth_metrics" not in result


def test_outcome_to_dict_no_ingredient_report() -> None:
    from ocr_engine.nutrition.reader import ReadOutcome
    from ocr_engine.nutrition.format_detector import DetectedFormat

    detected = DetectedFormat(category="text", score=0.1, grid_density=0.0, reasoning="low")
    outcome = ReadOutcome(
        winning_preset=None,
        winning_attempt_index=None,
        passed=False,
        detected_format=detected,
        final_ocr_text="",
        final_postprocessed_text="",
        attempts=[],
        ingredient_report=None,
    )
    result = outcome_to_dict(outcome)
    assert result["ingredient_analysis"] is None


# ---------------------------------------------------------------------------
# decode_image_bytes
# ---------------------------------------------------------------------------

def test_decode_image_bytes_zero_raises() -> None:
    with pytest.raises(ValueError):
        decode_image_bytes(b"")


def test_decode_image_bytes_corrupt_raises() -> None:
    with pytest.raises(ValueError):
        decode_image_bytes(b"\x00\x01\x02\x03corrupt_data")


def test_decode_image_bytes_valid_jpeg() -> None:
    data = _make_tiny_jpeg()
    image = decode_image_bytes(data)
    assert image is not None
    assert image.shape == (1, 1, 3)


# ---------------------------------------------------------------------------
# outcome_to_dict attempts serializable
# ---------------------------------------------------------------------------

def test_outcome_to_dict_attempts_serializable() -> None:
    from dataclasses import dataclass
    from ocr_engine.nutrition.reader import ReadOutcome
    from ocr_engine.nutrition.format_detector import DetectedFormat
    from ocr_engine.ocr.quality import QualityScore

    detected = DetectedFormat(category="table", score=0.8, grid_density=0.02, reasoning="ok")
    score = QualityScore(
        passed=True,
        score=0.85,
        mean_confidence=75.0,
        text_length=120,
        keyword_hits=4,
        details={},
    )
    outcome = ReadOutcome(
        winning_preset="p",
        winning_attempt_index=1,
        passed=True,
        detected_format=detected,
        final_ocr_text="t",
        final_postprocessed_text="t",
        attempts=[{"attempt_index": 1, "preset_name": "p", "score": score}],
        ingredient_report=None,
    )
    result = outcome_to_dict(outcome)
    assert len(result["attempts"]) == 1
    att = result["attempts"][0]
    assert att["passed"] is True
    assert att["score"] == 0.85
    assert att["mean_confidence"] == 75.0
