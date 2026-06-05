"""Tokenização de listas de ingredientes extraídas por OCR.

Pipeline:
1. Remove E-numbers / INS-codes e ruído OCR (|, *, †...).
2. Separa por vírgula, ponto e vírgula.
3. Para cada token:
   a. Palavras que o OCR já separou com espaços → mantém como estão (o OCR
      segmentou corretamente → não aplicar Viterbi).
   b. Sub-palavras longas sem espaço (kerning OCR) → aplica segmentação
      Viterbi → filtra stopwords industriais do resultado.
4. Retorna pares (token_original, token_normalizado_significativo).

Stopwords são filtradas SOMENTE nas sub-palavras que passaram pelo
segmentador, não nas palavras já separadas pelo OCR. Isso preserva
compostos como "farinha de trigo" (onde "de" é parte do ingrediente) e
descarta ruído como ["contem", "tracos", "de"] de "contémtraçosdeamêndoa".
"""

from __future__ import annotations

import re
import unicodedata

from .segmenter import segment as _segment


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Sub-palavras com mais de 10 chars sem espaço → candidatas ao segmentador.
_SEG_THRESHOLD = 10

# Ruído OCR: | é confundido com l ou separador; * e outros são rodapé.
_NOISE_RE = re.compile(r"[*†‡§¶#@|]")

_ADDITIVE_RE = re.compile(
    r"\(?\b(?:INS|INA|E|ADT)\s*\d{3}[a-z]?\b\)?",
    re.IGNORECASE,
)
_PARENS_RE = re.compile(r"\([^)]{1,60}\)")
_SPLIT_RE  = re.compile(r"[,;.]")

# Stopwords de rótulos alimentares: instruções de armazenamento,
# conectivos e termos que nunca são ingredientes por si só.
# Aplicadas APENAS em palavras produzidas pelo segmentador Viterbi.
_STOPWORDS: frozenset[str] = frozenset({
    # Preposições e artigos
    "de", "da", "do", "das", "dos", "e", "com", "em", "por", "para",
    "ao", "a", "o", "as", "os", "na", "no", "nas", "nos", "um", "uma",
    "ou", "se", "ate", "sem", "entre", "que",
    # Verbos e instruções de armazenamento
    "manter", "conservar", "guardar", "armazenar", "consumir",
    "utilizar", "usar",
    # Conectivos e qualificadores de rótulo
    "contem", "pode", "tracos", "tracas", "identico", "identica",
    "identicos", "identicas", "semelhante", "semelhantes",
    "tipo", "sabor", "sabores", "similar",
    # Descritores de armazenamento
    "local", "seco", "seca", "limpo", "limpa", "fresco", "fresca",
    "gelado", "gelada", "temperatura", "ambiente", "fechado", "fechada",
    "apos", "abertura", "validade", "fabricacao", "lote",
    # Outros ruídos comuns
    "e", "i",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Minúsculas, sem acentos, sem pontuação."""
    no_accents = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    lowered = no_accents.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _process_subword(subword: str) -> list[str]:
    """Normaliza uma sub-palavra e aplica Viterbi se for longa sem espaço.

    Retorna lista de palavras significativas (após filtro de stopwords
    nas palavras produzidas pelo segmentador).
    """
    norm = _normalize(subword)
    if not norm:
        return []

    if len(norm) <= _SEG_THRESHOLD:
        # Palavra curta — retorna sem segmentar e sem filtrar stopwords.
        # Stopwords curtas ("de", "e", "o") vindas do OCR fazem parte
        # de compostos legítimos como "farinha de trigo".
        return [norm] if norm else []

    # Palavra longa sem espaço — aplica Viterbi e filtra stopwords.
    segmented = _segment(norm)
    meaningful = [w for w in segmented if w not in _STOPWORDS and len(w) >= 3]
    return meaningful if meaningful else [norm]


def _build_normalized(original: str) -> str:
    """Constrói a forma normalizada de um token para o matcher.

    Palavras que o OCR já separou com espaços são normalizadas mas NÃO
    passam pelo segmentador nem pelo filtro de stopwords (o espaço indica
    que o OCR distinguiu as palavras corretamente).

    Sub-palavras longas sem espaço passam pelo Viterbi + filtro de stopwords.
    """
    parts: list[str] = []
    for subword in original.split():
        result = _process_subword(subword)
        parts.extend(result)
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def tokenize(ocr_text: str) -> list[tuple[str, str]]:
    """Divide o texto OCR em tokens de ingredientes.

    Retorna lista de (texto_original, texto_normalizado_significativo).
    Tokens sem palavras significativas após processamento são descartados.
    """
    cleaned = _ADDITIVE_RE.sub("", ocr_text)
    cleaned = _PARENS_RE.sub("", cleaned)
    cleaned = _NOISE_RE.sub("", cleaned)
    # Separa seções como "ALÉRGICOS: ..." mantendo o conteúdo após ":"
    cleaned = re.sub(r"[A-ZÁÉÍÓÚÀÂÊÔÃÕ\s]{4,}:", " ", cleaned)

    tokens: list[tuple[str, str]] = []
    for part in _SPLIT_RE.split(cleaned):
        original = part.strip()
        if not original:
            continue
        normalized = _build_normalized(original)
        if len(normalized) < 3:
            continue
        tokens.append((original, normalized))
    return tokens
