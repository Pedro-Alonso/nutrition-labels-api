"""Camada de OCR: wrapper do Tesseract, pós-processamento e avaliação de qualidade."""

from .service import OcrService, OcrConfig, OcrResult
from .quality import QualityEvaluator, QualityScore, QualityThresholds
from .postprocessing import NutritionTextPostProcessor
from .metrics import evaluate as evaluate_metrics, wer, cer, OcrMetrics, EditCounts

__all__ = [
    "OcrService",
    "OcrConfig",
    "OcrResult",
    "QualityEvaluator",
    "QualityScore",
    "QualityThresholds",
    "NutritionTextPostProcessor",
    "evaluate_metrics",
    "wer",
    "cer",
    "OcrMetrics",
    "EditCounts",
]
