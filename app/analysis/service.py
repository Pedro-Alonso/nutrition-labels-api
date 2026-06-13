from __future__ import annotations

import hashlib
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ocr_engine.nutrition.reader import NutritionReader, ReaderOptions, ReadOutcome


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodifica bytes de imagem para numpy array BGR."""
    if not image_bytes:
        raise ValueError("Não foi possível decodificar a imagem. Formato inválido ou corrompido.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("Não foi possível decodificar a imagem. Formato inválido ou corrompido.") from exc
    if image is None:
        raise ValueError("Não foi possível decodificar a imagem. Formato inválido ou corrompido.")
    return image


def outcome_to_dict(outcome: ReadOutcome) -> dict:
    """Converte ReadOutcome para dict serializável em JSON."""
    result = asdict(outcome)
    result.pop("summary_path", None)
    result.pop("groundtruth_metrics", None)

    # Converte ingredient_report para dict serializável
    if outcome.ingredient_report is not None:
        result["ingredient_analysis"] = outcome.ingredient_report.to_dict()
    else:
        result["ingredient_analysis"] = None
    result.pop("ingredient_report", None)

    # Converte attempts: QualityScore é dataclass com slots=True (sem __dict__)
    from dataclasses import is_dataclass

    attempts_out = []
    for att in outcome.attempts:
        score_obj = att.get("score", {})
        if is_dataclass(score_obj) and not isinstance(score_obj, type):
            score_dict = asdict(score_obj)
        elif isinstance(score_obj, dict):
            score_dict = score_obj
        else:
            score_dict = {}

        attempts_out.append(
            {
                "attempt_index": att.get("attempt_index", 0),
                "preset": att.get("preset_name", att.get("preset", "")),
                "passed": score_dict.get("passed", False),
                "score": score_dict.get("score", 0.0),
                "mean_confidence": score_dict.get("mean_confidence", 0.0),
                "text_length": score_dict.get("text_length", 0),
                "keyword_hits": score_dict.get("keyword_hits", 0),
            }
        )
    result["attempts"] = attempts_out

    # detected_format já é dict via asdict
    return result


class AnalysisService:
    def __init__(self, reader: NutritionReader) -> None:
        self._reader = reader

    def read_outcome(
        self,
        image_bytes: bytes,
        category_override: str | None = None,
        roi_enabled: bool = False,
        stop_on_first_pass: bool = True,
        postprocess: bool = True,
    ) -> ReadOutcome:
        """Executa OCR e retorna o ReadOutcome diretamente (não convertido para dict)."""
        options = ReaderOptions(
            category_override=category_override,
            roi_enabled=roi_enabled,
            stop_on_first_pass=stop_on_first_pass,
            postprocess=postprocess,
        )
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)
        try:
            return self._reader.read(tmp_path, options=options)
        finally:
            tmp_path.unlink(missing_ok=True)

    def analyze(
        self,
        image_bytes: bytes,
        category_override: str | None = None,
        roi_enabled: bool = False,
        stop_on_first_pass: bool = True,
        postprocess: bool = True,
    ) -> dict:
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        options = ReaderOptions(
            category_override=category_override,
            roi_enabled=roi_enabled,
            stop_on_first_pass=stop_on_first_pass,
            postprocess=postprocess,
        )

        # NutritionReader.read() espera um Path; salva em tmp, processa, remove.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)

        try:
            outcome = self._reader.read(tmp_path, options=options)
        finally:
            tmp_path.unlink(missing_ok=True)

        result = outcome_to_dict(outcome)
        result["image_hash"] = image_hash
        return result
