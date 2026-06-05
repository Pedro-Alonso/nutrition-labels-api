from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output


@dataclass(slots=True)
class OcrConfig:
    lang: str = "por"
    psm: int = 6
    oem: int = 3
    extra_config: str = ""
    dual_pass_polarity: bool = False

    def as_tesseract_config(self) -> str:
        parts = [f"--oem {int(self.oem)}", f"--psm {int(self.psm)}"]
        if self.extra_config:
            parts.append(self.extra_config)
        return " ".join(parts)


@dataclass(slots=True)
class OcrResult:
    text: str
    confidence: float
    used_inverted: bool = False
    image_for_ocr: np.ndarray | None = field(default=None, repr=False)


class OcrService:
    """Encapsula chamadas ao Tesseract e implementa dual-pass de polaridade."""

    def __init__(self, config: OcrConfig | None = None) -> None:
        self.config = config or OcrConfig()

    def read(self, image: Path | np.ndarray) -> OcrResult:
        if self.config.dual_pass_polarity:
            return self._read_best_polarity(image)
        return self._read_once(image)

    # ------------------------------ internas ------------------------------

    def _read_once(self, image: Path | np.ndarray) -> OcrResult:
        pil = self._to_pil(image)
        config = self.config.as_tesseract_config()
        text = pytesseract.image_to_string(pil, lang=self.config.lang, config=config)
        data = pytesseract.image_to_data(
            pil,
            lang=self.config.lang,
            config=config,
            output_type=Output.DICT,
        )
        return OcrResult(
            text=text,
            confidence=_mean_confidence(data),
            image_for_ocr=image if isinstance(image, np.ndarray) else None,
        )

    def _read_best_polarity(self, image: Path | np.ndarray) -> OcrResult:
        gray = self._to_numpy_gray(image)
        normal = self._read_once(gray)
        inverted_img = cv2.bitwise_not(gray)
        inverted = self._read_once(inverted_img)

        if inverted.confidence > normal.confidence:
            return OcrResult(
                text=inverted.text,
                confidence=inverted.confidence,
                used_inverted=True,
                image_for_ocr=inverted_img,
            )
        return OcrResult(
            text=normal.text,
            confidence=normal.confidence,
            used_inverted=False,
            image_for_ocr=gray,
        )

    # ------------------------------ conversores ------------------------------

    @staticmethod
    def _to_pil(image: Path | np.ndarray) -> Image.Image:
        if isinstance(image, Path):
            return Image.open(image)
        array = np.asarray(image)
        if array.ndim == 2:
            return Image.fromarray(array)
        if array.ndim == 3:
            return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
        raise ValueError("Formato de imagem não suportado para OCR")

    @staticmethod
    def _to_numpy_gray(image: Path | np.ndarray) -> np.ndarray:
        if isinstance(image, Path):
            gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Falha ao carregar imagem para OCR: {image}")
            return gray
        array = np.asarray(image)
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        raise ValueError("Formato de imagem não suportado para OCR")


def _mean_confidence(data: dict[str, list[str]]) -> float:
    values: list[float] = []
    for raw in data.get("conf", []):
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        if score >= 0:
            values.append(score)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
