"""Associação fuzzy entre tokens de ingredientes e a ontologia clínica.

Estratégia em quatro camadas — da mais precisa à mais tolerante:

1. Correspondência exata após normalização (sem acento, minúsculas).
2. Contenção com limite de palavra: a chave ontológica está contida no token
   como palavra inteira (não apenas substring), ou vice-versa.
3. Levenshtein com limiar proporcional ao comprimento da chave:
   limiar = max(1, min(3, len(chave) // 6))
4. Decomposição por palavras: tokens compostos ("flocos de trigo") são
   divididos em palavras individuais; cada palavra passa pelas camadas 1-3
   e retorna-se o match de maior risco clínico encontrado.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
import json

from ocr.levenshtein import levenshtein_distance


_RISK_ORDER: dict[str, int] = {
    "ALTO": 0,
    "MODERADO-ALTO": 1,
    "MODERADO": 2,
    "BAIXO": 3,
    "SEGURO": 4,
    "BENEFICO": 5,
    "INFORMATIVO": 6,
}


def _word_boundary_match(key: str, token: str) -> bool:
    idx = token.find(key)
    if idx == -1:
        return False
    before_ok = idx == 0 or token[idx - 1] == " "
    after_ok = idx + len(key) == len(token) or token[idx + len(key)] == " "
    return before_ok and after_ok


@dataclass(slots=True)
class OntologyEntry:
    key: str        # chave canônica normalizada
    display: str    # chave original do JSON
    classe: str
    risco: str
    alerta: str
    indice_glicemico: int | str | None  # inteiro, faixa "55-65", ou None
    fisiopatologia: str | None
    nota_clinica: str | None


@dataclass(slots=True)
class MatchResult:
    token_original: str
    token_normalized: str
    matched_entry: OntologyEntry | None
    match_type: str   # "exact" | "containment" | "levenshtein" | "none"
    edit_distance: int


def _normalize(text: str) -> str:
    no_accents = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    lowered = no_accents.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


class OntologyMatcher:
    """Carrega a ontologia e realiza correspondência fuzzy em tempo de execução."""

    def __init__(self, ontology_path: Path) -> None:
        self._entries: list[OntologyEntry] = []
        self._index: dict[str, OntologyEntry] = {}
        self._load(ontology_path)

    def _load(self, path: Path) -> None:
        data: dict = json.loads(path.read_text(encoding="utf-8"))
        for display_key, spec in data.items():
            ig_raw = spec.get("indice_glicemico")
            entry = OntologyEntry(
                key=_normalize(display_key),
                display=display_key,
                classe=spec.get("classe", ""),
                risco=spec.get("risco", ""),
                alerta=spec.get("alerta", ""),
                indice_glicemico=None if ig_raw in (None, "", "N/A") else ig_raw,
                fisiopatologia=spec.get("fisiopatologia") or None,
                nota_clinica=spec.get("nota_clinica") or None,
            )
            self._entries.append(entry)
            self._index[entry.key] = entry
            for syn in spec.get("sinonimos", []):
                norm_syn = _normalize(syn)
                if norm_syn and norm_syn not in self._index:
                    self._index[norm_syn] = entry

    def match(self, token_original: str, token_normalized: str) -> MatchResult:
        # Camada 1: exato
        if token_normalized in self._index:
            return MatchResult(
                token_original=token_original,
                token_normalized=token_normalized,
                matched_entry=self._index[token_normalized],
                match_type="exact",
                edit_distance=0,
            )

        best_entry: OntologyEntry | None = None
        best_dist = 9999
        best_type = "none"

        for key, entry in self._index.items():
            # Camada 2: contenção com limite de palavra
            if _word_boundary_match(key, token_normalized) or (len(token_normalized) >= 5 and _word_boundary_match(token_normalized, key)):
                dist = abs(len(key) - len(token_normalized))
                if dist < best_dist:
                    best_dist = dist
                    best_entry = entry
                    best_type = "containment"
                continue

            # Camada 3: Levenshtein com limiar proporcional
            threshold = max(1, min(3, len(key) // 6))
            dist = levenshtein_distance(token_normalized, key)
            if dist <= threshold and dist < best_dist:
                best_dist = dist
                best_entry = entry
                best_type = "levenshtein"

        # Camada 4: decomposição por palavras para tokens compostos
        if best_entry is None:
            words = token_normalized.split()
            if len(words) > 1:
                word_best: OntologyEntry | None = None
                for word in words:
                    if len(word) < 3:
                        continue
                    sub = self.match(token_original, word)
                    if sub.matched_entry is None:
                        continue
                    if word_best is None or (
                        _RISK_ORDER.get(sub.matched_entry.risco, 99)
                        < _RISK_ORDER.get(word_best.risco, 99)
                    ):
                        word_best = sub.matched_entry
                if word_best is not None:
                    best_entry = word_best
                    best_type = "word_decomposition"
                    best_dist = 0

        return MatchResult(
            token_original=token_original,
            token_normalized=token_normalized,
            matched_entry=best_entry,
            match_type=best_type,
            edit_distance=best_dist if best_entry else -1,
        )

    def match_all(self, tokens: list[tuple[str, str]]) -> list[MatchResult]:
        return [self.match(orig, norm) for orig, norm in tokens]
