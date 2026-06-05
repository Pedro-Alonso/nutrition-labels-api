"""Avaliação de qualidade do output de OCR.

Determina se um output é "satisfatório" para que o orquestrador possa decidir
encerrar a cascata de presets ou tentar o próximo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

from .levenshtein import levenshtein_distance


DEFAULT_EXPECTED_KEYWORDS: tuple[str, ...] = (
    "valor energetico",
    "carboidratos",
    "proteinas",
    "gorduras totais",
    "gorduras saturadas",
    "sodio",
)


@dataclass(slots=True)
class QualityThresholds:
    min_mean_confidence: float = 65.0
    min_text_length: int = 40
    min_keyword_hits: int = 3
    keyword_max_distance: int = 3
    expected_keywords: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EXPECTED_KEYWORDS)

    @classmethod
    def from_dict(cls, data: dict | None) -> "QualityThresholds":
        if not data:
            return cls()
        return cls(
            min_mean_confidence=float(data.get("min_mean_confidence", 65.0)),
            min_text_length=int(data.get("min_text_length", 40)),
            min_keyword_hits=int(data.get("min_keyword_hits", 3)),
            keyword_max_distance=int(data.get("keyword_max_distance", 3)),
            expected_keywords=tuple(data.get("expected_keywords", DEFAULT_EXPECTED_KEYWORDS)),
        )


@dataclass(slots=True)
class QualityScore:
    passed: bool
    score: float
    mean_confidence: float
    text_length: int
    keyword_hits: int
    details: dict


class QualityEvaluator:
    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def evaluate(self, text: str, mean_confidence: float) -> QualityScore:
        normalized = _normalize(text)
        text_length = len(text.strip())
        keyword_hits = self._count_keywords(normalized)

        details = {
            "mean_confidence": mean_confidence,
            "text_length": text_length,
            "keyword_hits": keyword_hits,
            "keywords_required": self.thresholds.min_keyword_hits,
        }

        confidence_ok = mean_confidence >= self.thresholds.min_mean_confidence
        length_ok = text_length >= self.thresholds.min_text_length
        keywords_ok = keyword_hits >= self.thresholds.min_keyword_hits

        passed = confidence_ok and length_ok and keywords_ok
        # Score contínuo para ranquear presets quando nenhum passar.
        score = (
            (mean_confidence / 100.0) * 0.5
            + min(keyword_hits / max(1, len(self.thresholds.expected_keywords)), 1.0) * 0.4
            + min(text_length / 500.0, 1.0) * 0.1
        )
        return QualityScore(
            passed=passed,
            score=float(score),
            mean_confidence=mean_confidence,
            text_length=text_length,
            keyword_hits=keyword_hits,
            details=details,
        )

    def _count_keywords(self, normalized_text: str) -> int:
        if not normalized_text:
            return 0
        # Procura cada keyword esperada como substring aproximada em janelas de mesmo tamanho.
        hits = 0
        tokens = normalized_text.split()
        for keyword in self.thresholds.expected_keywords:
            kw_normalized = _normalize(keyword)
            kw_tokens = kw_normalized.split()
            if not kw_tokens:
                continue
            window = len(kw_tokens)
            found = False
            for i in range(0, max(1, len(tokens) - window + 1)):
                candidate = " ".join(tokens[i : i + window])
                if levenshtein_distance(candidate, kw_normalized) <= self.thresholds.keyword_max_distance:
                    found = True
                    break
            if found:
                hits += 1
        return hits


def _normalize(text: str) -> str:
    no_accents = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )
    lowered = no_accents.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()
