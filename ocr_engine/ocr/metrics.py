"""Métricas de avaliação de qualidade do OCR: WER e CER.

Referência canônica citada na Revisão Bibliográfica:
    Rice, S. V. (1996). Measuring the accuracy of page-reading systems.
    PhD Thesis, University of Nevada, Las Vegas.

    WER = (S_w + D_w + I_w) / N_w
    CER = (S_c + D_c + I_c) / N_c

S = substituições, D = deleções, I = inserções, N = comprimento da referência.
Calculados via alinhamento por programação dinâmica (Levenshtein estendido).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict


@dataclass(slots=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def error_count(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.reference_length == 0:
            return 0.0 if self.error_count == 0 else 1.0
        return self.error_count / self.reference_length


@dataclass(slots=True)
class OcrMetrics:
    wer: float
    cer: float
    word_edits: EditCounts
    char_edits: EditCounts

    def to_dict(self) -> dict:
        return {
            "wer": round(self.wer, 4),
            "cer": round(self.cer, 4),
            "word_edits": {
                "substitutions": self.word_edits.substitutions,
                "deletions": self.word_edits.deletions,
                "insertions": self.word_edits.insertions,
                "reference_words": self.word_edits.reference_length,
                "error_words": self.word_edits.error_count,
            },
            "char_edits": {
                "substitutions": self.char_edits.substitutions,
                "deletions": self.char_edits.deletions,
                "insertions": self.char_edits.insertions,
                "reference_chars": self.char_edits.reference_length,
                "error_chars": self.char_edits.error_count,
            },
        }


def _edit_counts(reference: list, hypothesis: list) -> EditCounts:
    """Levenshtein estendido rastreando S, D, I separadamente.

    Usa duas linhas de DP para memória O(m).
    """
    n, m = len(reference), len(hypothesis)

    # Cada célula: (custo_total, substituições, deleções, inserções)
    prev = [(j, 0, 0, j) for j in range(m + 1)]

    for i in range(1, n + 1):
        curr: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * (m + 1)
        curr[0] = (i, 0, i, 0)
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                c_eq, s_eq, d_eq, ins_eq = prev[j - 1]
                option_sub = (c_eq, s_eq, d_eq, ins_eq)
            else:
                c_s, s_s, d_s, ins_s = prev[j - 1]
                option_sub = (c_s + 1, s_s + 1, d_s, ins_s)

            c_d, s_d, d_d, ins_d = prev[j]
            option_del = (c_d + 1, s_d, d_d + 1, ins_d)

            c_i, s_i, d_i, ins_i = curr[j - 1]
            option_ins = (c_i + 1, s_i, d_i, ins_i + 1)

            best = min(option_sub, option_del, option_ins, key=lambda t: t[0])
            curr[j] = best
        prev = curr

    _, S, D, I = prev[m]
    return EditCounts(substitutions=S, deletions=D, insertions=I, reference_length=n)


# Unidades de medida comuns em rótulos nutricionais brasileiros.
_UNIT_PATTERN = re.compile(
    r"(\d)\s+(g|mg|kcal|kj|ml|mcg|ui|iu)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Remove acentos, passa para minúsculas e colapsa pontuação/espaços.

    Também colapsa 'número + espaço + unidade' → 'número+unidade' (ex: '15 g' → '15g',
    '75 kcal' → '75kcal') para não penalizar variações tipográficas irrelevantes do OCR.
    """
    no_accents = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )
    lowered = no_accents.lower()
    # Colapsa espaço entre dígito e unidade antes de remover pontuação.
    collapsed = _UNIT_PATTERN.sub(r"\1\2", lowered)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", collapsed)
    return re.sub(r"\s+", " ", cleaned).strip()


def wer(reference: str, hypothesis: str) -> tuple[float, EditCounts]:
    """Retorna (WER, EditCounts em nível de palavra)."""
    ref_words = _normalize(reference).split()
    hyp_words = _normalize(hypothesis).split()
    counts = _edit_counts(ref_words, hyp_words)
    return counts.rate, counts


def cer(reference: str, hypothesis: str) -> tuple[float, EditCounts]:
    """Retorna (CER, EditCounts em nível de caractere)."""
    ref_chars = list(_normalize(reference))
    hyp_chars = list(_normalize(hypothesis))
    counts = _edit_counts(ref_chars, hyp_chars)
    return counts.rate, counts


def evaluate(reference: str, hypothesis: str) -> OcrMetrics:
    """Calcula WER e CER em uma única chamada."""
    wer_rate, word_counts = wer(reference, hypothesis)
    cer_rate, char_counts = cer(reference, hypothesis)
    return OcrMetrics(wer=wer_rate, cer=cer_rate, word_edits=word_counts, char_edits=char_counts)
