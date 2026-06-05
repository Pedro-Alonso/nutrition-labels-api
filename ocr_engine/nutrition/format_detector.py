"""Detecção de formato visual: tabela vs texto corrido.

Heurística rápida baseada em duas evidências:

1. Densidade de linhas horizontais/verticais longas — tabelas exibem grade nítida.
2. Distribuição vertical dos componentes conexos — texto corrido tende a formar
   linhas bem regulares; tabela tende a agrupar verticalmente em blocos separados
   por linhas de grade.

A decisão final aceita override por configuração (ver `config/routing.json`) ou
pelo menu interativo, para não empacar o usuário quando a heurística falhar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from imaging import operations as ops
from imaging.morphology import estimate_grid_density


Category = Literal["table", "text", "ingredient"]


@dataclass(slots=True)
class DetectedFormat:
    category: Category
    score: float
    grid_density: float
    reasoning: str


@dataclass(slots=True)
class FormatDetectorConfig:
    grid_density_threshold: float = 0.015
    force_category: Category | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "FormatDetectorConfig":
        if not data:
            return cls()
        force = data.get("force_category")
        if force not in (None, "table", "text", "ingredient"):
            force = None
        return cls(
            grid_density_threshold=float(data.get("grid_density_threshold", 0.015)),
            force_category=force,
        )


class FormatDetector:
    def __init__(self, config: FormatDetectorConfig | None = None) -> None:
        self.config = config or FormatDetectorConfig()

    def detect(self, image_bgr: np.ndarray) -> DetectedFormat:
        if self.config.force_category:
            return DetectedFormat(
                category=self.config.force_category,
                score=1.0,
                grid_density=0.0,
                reasoning="override forçado via configuração",
            )

        gray = ops.grayscale(image_bgr)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            12,
        )
        density = estimate_grid_density(binary)
        is_table = density >= self.config.grid_density_threshold
        category: Category = "table" if is_table else "text"
        reasoning = (
            f"densidade de grade={density:.4f} vs limiar={self.config.grid_density_threshold:.4f}"
        )
        return DetectedFormat(
            category=category,
            score=float(density),
            grid_density=float(density),
            reasoning=reasoning,
        )
