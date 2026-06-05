"""Módulo de análise clínica de listas de ingredientes para diabetes mellitus.

Arquitetura híbrida (Simbólica + Estatística):
  - Simbólica: ontologia JSON estática (verdade médica validada)
  - Estatística: Levenshtein para correção de erros OCR

Uso rápido:
    from ingredients import IngredientAnalyzer
    from pathlib import Path

    analyzer = IngredientAnalyzer(Path("config/ontology_diabetes.json"))
    report = analyzer.analyze(ocr_text)
    print(report)
"""

from .analyzer import IngredientAnalyzer, IngredientReport, _global_risk
from .tokenizer import tokenize
from .matcher import OntologyMatcher, MatchResult

__all__ = [
    "IngredientAnalyzer",
    "IngredientReport",
    "OntologyMatcher",
    "MatchResult",
    "tokenize",
    "_global_risk",
]
