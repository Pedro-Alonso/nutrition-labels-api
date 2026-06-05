"""Segmentação de palavras aglutinadas por kerning apertado (Viterbi PT-BR).

O OCR falha quando espaços entre palavras distintas são menores que um glifo,
produzindo "corantenaturalcaroteno" em vez de "corante natural caroteno".

Algoritmo: Programação dinâmica (Viterbi) que encontra o corte de maior
probabilidade. O vocabulário é carregado de `config/wordlist_pt_food.txt`.

Modelo de custo:
  - Palavra conhecida (freq f): custo = log(total) - log(f)   ← baixo
  - Palavra desconhecida de tamanho n: custo = log(total) + n * UNKNOWN_PENALTY
    A penalidade proporcional ao comprimento garante que dividir "tracosdeamendoa"
    em ["tracos","de","amendoa"] custe MENOS do que mantê-la inteira como
    palavra desconhecida. (Bug original: custo constante = log(total) para
    qualquer desconhecida, tornando-a às vezes mais barata que um split.)

Referência: Viterbi (1967), wordninja (Metzler, 2010).
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Carregamento do vocabulário externo
# ---------------------------------------------------------------------------

_DEFAULT_WORDLIST = Path(__file__).resolve().parent.parent / "config" / "wordlist_pt_food.txt"

# Penalidade por caractere para palavras desconhecidas.
# Valor 5 garante que divisões em palavras conhecidas curtas batam palavras
# desconhecidas longas. Derivado empiricamente (vide demostração em testes).
_UNKNOWN_PENALTY = 5


def _load_vocab(path: Path) -> dict[str, int]:
    """Lê arquivo de vocabulário no formato 'palavra\\tfrequência' (uma por linha)."""
    vocab: dict[str, int] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            word = parts[0].strip().lower()
            freq = int(parts[1]) if len(parts) > 1 else 100
            if word:
                vocab[word] = freq
    except (FileNotFoundError, IOError):
        pass
    return vocab


_VOCAB: dict[str, int] = _load_vocab(_DEFAULT_WORDLIST)

# Garante que o vocabulário não fique vazio mesmo sem o arquivo externo.
_VOCAB_FALLBACK = {
    "de": 10000, "da": 8000, "do": 8000, "e": 9000, "com": 4000,
    "acucar": 2000, "sal": 2000, "agua": 2000, "gordura": 800,
    "vegetal": 1000, "farinha": 1500, "trigo": 1200, "soja": 1000,
    "amido": 1000, "leite": 1500, "oleo": 1500, "aveia": 600,
}
if not _VOCAB:
    _VOCAB = _VOCAB_FALLBACK

_MAX_WORD   = max(len(w) for w in _VOCAB)
_LOG_TOTAL  = math.log(sum(_VOCAB.values()))


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Remove acentos, pontuação e converte para minúsculas."""
    no_acc = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    return re.sub(r"[^a-z]", "", no_acc.lower())


# ---------------------------------------------------------------------------
# Custo Viterbi
# ---------------------------------------------------------------------------

def _word_cost(w: str) -> float:
    freq = _VOCAB.get(w)
    if freq:
        return _LOG_TOTAL - math.log(freq)
    # Penalidade proporcional ao comprimento: palavras desconhecidas longas
    # custam muito mais do que um conjunto de palavras conhecidas curtas.
    return _LOG_TOTAL + len(w) * _UNKNOWN_PENALTY


# ---------------------------------------------------------------------------
# Segmentação principal
# ---------------------------------------------------------------------------

def segment(token: str) -> list[str]:
    """Divide `token` (sem espaços) na lista de palavras mais provável (Viterbi).

    Normaliza (remove acentos) antes do processamento — garantia de que
    'amêndoa' e 'amendoa' são tratados identicamente pelo DP.

    Retorna lista de palavras em minúsculas sem acentos.
    """
    norm = _normalize(token)
    if not norm:
        return []
    if len(norm) <= 2:
        return [norm]

    n = len(norm)
    INF = float("inf")
    # best[i] = (min_custo, palavra_terminando_em_i, posição_anterior)
    best: list[tuple[float, str, int]] = [(INF, "", 0)] * (n + 1)
    best[0] = (0.0, "", 0)

    for i in range(1, n + 1):
        for j in range(max(0, i - _MAX_WORD), i):
            candidate = norm[j:i]
            cost = best[j][0] + _word_cost(candidate)
            if cost < best[i][0]:
                best[i] = (cost, candidate, j)

    # Retrocede pelo caminho ótimo
    words: list[str] = []
    pos = n
    while pos > 0:
        _, word, prev = best[pos]
        if word:
            words.append(word)
            pos = prev
        else:
            break

    if not words:
        return [norm]

    words.reverse()

    # Guarda de qualidade: se a maioria das palavras não está no vocabulário,
    # o segmentador está fabricando divisões — devolve o token original.
    unknown = sum(1 for w in words if w not in _VOCAB)
    if len(words) > 1 and unknown / len(words) > 0.6:
        return [norm]

    return words
