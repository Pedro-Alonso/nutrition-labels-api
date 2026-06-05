"""Detecção morfológica de estruturas de grade (linhas e células de tabela)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class GridLines:
    horizontal: np.ndarray
    vertical: np.ndarray
    combined: np.ndarray


@dataclass(slots=True, frozen=True)
class Cell:
    x: int
    y: int
    w: int
    h: int

    @property
    def row_center(self) -> float:
        return self.y + self.h / 2.0


def extract_grid_lines(
    binary_image: np.ndarray,
    horizontal_divisor: int = 20,
    vertical_divisor: int = 20,
) -> GridLines:
    """Extrai linhas horizontais e verticais por abertura morfológica.

    Assume imagem binária com texto/linhas escuros (preto sobre branco). Inverte
    internamente para deixar linhas em branco sobre fundo preto e aplica MORPH_OPEN
    com kernels longos proporcionais às dimensões da imagem.
    """
    if binary_image.ndim != 2:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(binary_image)

    h, w = inverted.shape
    h_len = max(3, w // max(1, horizontal_divisor))
    v_len = max(3, h // max(1, vertical_divisor))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, v_kernel)
    combined = cv2.bitwise_or(horizontal, vertical)
    return GridLines(horizontal=horizontal, vertical=vertical, combined=combined)


def remove_grid_lines(
    binary_image: np.ndarray,
    horizontal_divisor: int = 20,
    vertical_divisor: int = 20,
    min_line_area_ratio: float = 0.005,
) -> np.ndarray:
    """Remove linhas horizontais e verticais preservando o texto quando possível."""
    if binary_image.ndim != 2:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    grid = extract_grid_lines(binary_image, horizontal_divisor, vertical_divisor)
    dilated = cv2.dilate(
        grid.combined,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    h, w = binary_image.shape
    if cv2.countNonZero(dilated) < int(min_line_area_ratio * h * w):
        return binary_image

    inverted = cv2.bitwise_not(binary_image)
    cleaned = cv2.subtract(inverted, dilated)
    return cv2.bitwise_not(cleaned)


def detect_cells(
    binary_image: np.ndarray,
    horizontal_divisor: int = 20,
    vertical_divisor: int = 20,
    min_cell_area_ratio: float = 0.001,
    max_cell_area_ratio: float = 0.5,
) -> list[Cell]:
    """Detecta células retangulares da tabela por contornos da grade invertida.

    Retorna as células ordenadas em ordem de leitura (top-to-bottom, left-to-right)
    agrupando por linhas via clustering simples das coordenadas y.
    """
    if binary_image.ndim != 2:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    grid = extract_grid_lines(binary_image, horizontal_divisor, vertical_divisor)

    h, w = binary_image.shape
    image_area = float(h * w)

    grid_mask = cv2.dilate(
        grid.combined,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    non_grid = cv2.bitwise_not(grid_mask)

    contours, _ = cv2.findContours(non_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells: list[Cell] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = float(cw * ch)
        if area <= 0:
            continue
        ratio = area / image_area
        if ratio < min_cell_area_ratio or ratio > max_cell_area_ratio:
            continue
        if cw < 20 or ch < 12:
            continue
        cells.append(Cell(x=x, y=y, w=cw, h=ch))

    cells.sort(key=lambda c: (c.y, c.x))
    if cells:
        mean_cell_h = sum(c.h for c in cells) / len(cells)
        row_tolerance = max(10, int(0.5 * mean_cell_h))
    else:
        row_tolerance = max(10, h // 80)
    return _group_rows_then_columns(cells, row_tolerance=row_tolerance)


def _group_rows_then_columns(cells: list[Cell], row_tolerance: int) -> list[Cell]:
    if not cells:
        return []
    rows: list[list[Cell]] = [[cells[0]]]
    for cell in cells[1:]:
        if abs(cell.y - rows[-1][0].y) <= row_tolerance:
            rows[-1].append(cell)
        else:
            rows.append([cell])
    ordered: list[Cell] = []
    for row in rows:
        row.sort(key=lambda c: c.x)
        ordered.extend(row)
    return ordered


def estimate_grid_density(binary_image: np.ndarray) -> float:
    """Proporção de pixels que pertencem a linhas longas (horizontais ou verticais).

    Valor > ~0.015 costuma indicar tabela bem demarcada. Útil para a etapa de
    roteamento (tabela vs texto corrido).
    """
    if binary_image.ndim != 2:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    grid = extract_grid_lines(binary_image, horizontal_divisor=8, vertical_divisor=8)
    total = float(binary_image.shape[0] * binary_image.shape[1])
    if total <= 0:
        return 0.0
    return cv2.countNonZero(grid.combined) / total


def crop_cell(image: np.ndarray, cell: Cell, padding: int = 2) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, cell.x + padding)
    y1 = max(0, cell.y + padding)
    x2 = min(w, cell.x + cell.w - padding)
    y2 = min(h, cell.y + cell.h - padding)
    if x2 <= x1 or y2 <= y1:
        return image[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]
    return image[y1:y2, x1:x2]
