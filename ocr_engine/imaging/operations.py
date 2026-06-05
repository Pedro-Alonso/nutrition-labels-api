"""Operações primitivas de PDI.

Cada função é pura (entrada → saída) e nomeada para casar com as configurações
declarativas nos presets JSON. A tabela `OPERATION_REGISTRY` no fim do arquivo é o
único ponto de acoplamento entre preset JSON e implementação.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


# ------------------------------ primitivas ------------------------------


def resize_max_height(image: np.ndarray, max_height: int = 500) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= max_height:
        return image
    scale = max_height / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, max_height), interpolation=cv2.INTER_AREA)


def resize_min_height(image: np.ndarray, min_height: int = 600) -> np.ndarray:
    """Upscale suave quando a ROI é pequena demais para o Tesseract."""
    h, w = image.shape[:2]
    if h >= min_height:
        return image
    scale = min_height / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, min_height), interpolation=cv2.INTER_CUBIC)


def grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def median_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.medianBlur(image, _odd(kernel_size, minimum=1))


def gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = _odd(kernel_size, minimum=1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=0)


def bilateral_filter(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    gray = grayscale(image)
    tile = (max(1, int(tile_grid_size[0])), max(1, int(tile_grid_size[1])))
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile)
    return op.apply(gray)


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    gray = grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 41,
    c: int = 14,
) -> np.ndarray:
    gray = grayscale(image)
    block = _odd(block_size, minimum=3)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        int(c),
    )


def ensure_black_text_on_white(image: np.ndarray) -> np.ndarray:
    """Normaliza polaridade para texto preto em fundo branco."""
    binary = image if image.ndim == 2 else grayscale(image)
    dark = float(np.mean(binary < 128))
    bright = float(np.mean(binary >= 128))
    if dark > bright:
        return cv2.bitwise_not(binary)
    return binary


def invert(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(image)


def morph_erode(
    image: np.ndarray,
    kernel_size: int = 2,
    iterations: int = 1,
) -> np.ndarray:
    k = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(image, kernel, iterations=max(1, int(iterations)))


def morph_dilate(
    image: np.ndarray,
    kernel_size: int = 2,
    iterations: int = 1,
) -> np.ndarray:
    k = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(image, kernel, iterations=max(1, int(iterations)))


def morph_close(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    k = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(iterations)))


def morph_open(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    k = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=max(1, int(iterations)))


def thin_text_if_thick(image: np.ndarray, dark_ratio_threshold: float = 0.14) -> np.ndarray:
    """Erosão condicional: só afina texto quando ele está grosso demais."""
    binary = image if image.ndim == 2 else grayscale(image)
    dark_ratio = float(np.mean(binary < 128))
    if dark_ratio < dark_ratio_threshold:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.erode(binary, kernel, iterations=1)


def deskew(image: np.ndarray, max_abs_angle: float = 10.0) -> np.ndarray:
    """Corrige inclinação pequena usando minAreaRect nos pixels de texto.

    Usa limite em `max_abs_angle` para evitar girar imagens que já estão retas,
    onde ruído levaria a rotações erradas.
    """
    binary = image if image.ndim == 2 else grayscale(image)
    if float(np.mean(binary < 128)) < 0.02:
        return binary
    coords = np.column_stack(np.where(binary < 128))
    if coords.size == 0:
        return binary
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5 or abs(angle) > max_abs_angle:
        return binary
    h, w = binary.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        binary,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def unsharp_mask(
    image: np.ndarray,
    kernel_size: int = 5,
    amount: float = 1.0,
) -> np.ndarray:
    gray = grayscale(image)
    blurred = cv2.GaussianBlur(gray, (_odd(kernel_size, 1), _odd(kernel_size, 1)), 0)
    sharpened = cv2.addWeighted(gray, 1.0 + float(amount), blurred, -float(amount), 0)
    return sharpened


# ------------------------------ helpers ------------------------------


def _odd(value: int, minimum: int = 1) -> int:
    v = max(minimum, int(value))
    if v % 2 == 0:
        v += 1
    return v


# ------------------------------ registry ------------------------------


OPERATION_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "resize_max_height": resize_max_height,
    "resize_min_height": resize_min_height,
    "grayscale": grayscale,
    "median_blur": median_blur,
    "gaussian_blur": gaussian_blur,
    "bilateral_filter": bilateral_filter,
    "clahe": clahe,
    "otsu_threshold": otsu_threshold,
    "adaptive_threshold": adaptive_threshold,
    "ensure_black_text_on_white": ensure_black_text_on_white,
    "invert": invert,
    "morph_erode": morph_erode,
    "morph_dilate": morph_dilate,
    "morph_close": morph_close,
    "morph_open": morph_open,
    "thin_text_if_thick": thin_text_if_thick,
    "deskew": deskew,
    "unsharp_mask": unsharp_mask,
}
