"""Módulo de leitura de informação nutricional.

Aceita dois formatos visuais: tabela estruturada e texto corrido. Expõe um único
ponto de entrada (`NutritionReader`) que decide o formato, carrega os presets
adequados, roda a cascata e devolve o melhor output.
"""

from .reader import NutritionReader, ReadOutcome
from .format_detector import FormatDetector, DetectedFormat
from .presets import Preset, PresetRepository

__all__ = [
    "NutritionReader",
    "ReadOutcome",
    "FormatDetector",
    "DetectedFormat",
    "Preset",
    "PresetRepository",
]
