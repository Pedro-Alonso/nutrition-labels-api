from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Leitura robusta em paths com acentos/espaços (Windows-friendly)."""
    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, mode)
    if image is None:
        raise ValueError(f"Falha ao carregar imagem: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Escrita robusta em paths com acentos/espaços (Windows-friendly)."""
    extension = path.suffix if path.suffix else ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Falha ao codificar imagem: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
