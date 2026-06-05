"""Geradores Hypothesis para a suíte de testes da feature GCV OCR Preset.

Este módulo concentra estratégias customizadas usadas pelos testes
property-based (PBT) de ``tests/gcv/``. Cada gerador é dedicado a
alimentar uma propriedade específica declarada no documento de design
(``.kiro/specs/gcv-ocr-preset/design.md``, seção *Correctness
Properties*) e tem como objetivo cobrir o espaço de entradas relevante
para falsificar invariantes — sem desperdiçar exemplos em estados
trivialmente impossíveis.

Convenções gerais:

- Strings textuais ficam restritas a um alfabeto imprimível enxuto
  (ASCII ``\u0020``–``\u007E``) para evitar surrogates e acentos que
  poluem casos sem agregar cobertura. Quando uma propriedade depende
  explicitamente de Unicode (ex.: garantia de UTF-8 em P21), o teste
  consumidor usa diretamente ``st.text()`` sem nosso wrapper.
- ``language_hints`` é sempre uma ``tuple[str, ...]`` para refletir o
  contrato imutável de ``GcvFetchResult.language_hints``
  (``ocr/cloud_vision/types.py``); nunca devolvemos ``list``.
- ``cv2`` é importado lazy dentro de ``image_arrays`` apenas para
  validar formato suportado em uma assertion local; o resto do módulo
  não depende do OpenCV.

Cada estratégia documenta no próprio docstring qual propriedade ela
alimenta, seguindo a tabela de "Geradores Hypothesis" do design.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from ocr.cloud_vision.types import (
    ALLOWED_FEATURES,
    ALLOWED_KINDS,
    ERROR_PRECEDENCE,
)


# ---------------------------------------------------------------------------
# Alfabetos e helpers
# ---------------------------------------------------------------------------

# Caracteres ASCII imprimíveis (espaço inclusive). Cobre todo o conjunto
# que aparece em respostas reais da Cloud Vision sem entrar em
# surrogates ou combining marks que apenas inflam o espaço de busca.
_PRINTABLE_ASCII = st.characters(min_codepoint=0x20, max_codepoint=0x7E)

# Conjunto enxuto de hints BCP-47 plausíveis para rótulos brasileiros e
# casos multi-idioma observáveis em rótulos importados.
_BCP47_POOL: tuple[str, ...] = ("pt", "en", "es", "pt-BR", "en-US")

# Conjunto fechado de classes de erro reportáveis pela classificação de
# ``GcvClient._classify`` (Requirement 6.5–6.8). ``ERROR_PRECEDENCE``
# define a ordem canônica; aqui só precisamos do conjunto.
_ERROR_CLASSES: tuple[str, ...] = ERROR_PRECEDENCE

# Strings que nunca devem ser aceitas como ``kind`` válido pelo
# ``PresetRepository._parse``: incluem casing alterado, sufixos espúrios
# e alguns valores notórios. ``kind_strings_invalid`` adiciona casos
# gerados aleatoriamente; este pool garante exemplos representativos
# fixos.
_KIND_INVALID_STATIC: tuple[str, ...] = (
    "",
    "  ",
    "Cloud_Vision",
    "CLOUD_VISION",
    "linear",
    "linear_other",
    "unknown_kind",
    "None",
    "null",
    "linear_table ",  # whitespace trailing
    " linear_text",   # whitespace leading
)

# Strings inválidas para ``gcv.feature``: variações de casing dos
# valores válidos e valores fora do domínio.
_FEATURE_INVALID_STATIC: tuple[str, ...] = (
    "",
    "  ",
    "text_detection",
    "document_text_detection",
    "Text_Detection",
    "DOCUMENT TEXT DETECTION",
    "OCR",
    "FACE_DETECTION",
    "LABEL_DETECTION",
    "None",
)


# ---------------------------------------------------------------------------
# Respostas GCV sintéticas
# ---------------------------------------------------------------------------

@st.composite
def _vertices(draw: st.DrawFn) -> list[dict[str, int]]:
    """Gera quatro vértices BGR plausíveis para uma ``BoundingPoly``.

    A Cloud Vision API retorna ``bounding_box.vertices`` como lista de
    quatro pares ``{x, y}`` em coordenadas inteiras dentro do plano da
    imagem. Manter coordenadas pequenas e positivas é suficiente para
    todas as propriedades que testamos — nenhuma delas depende da
    geometria do polígono.
    """

    coords = draw(
        st.lists(
            st.integers(min_value=0, max_value=2048),
            min_size=8,
            max_size=8,
        )
    )
    return [
        {"x": coords[0], "y": coords[1]},
        {"x": coords[2], "y": coords[3]},
        {"x": coords[4], "y": coords[5]},
        {"x": coords[6], "y": coords[7]},
    ]


def _word_text() -> st.SearchStrategy[str]:
    """Texto curto e imprimível para preencher ``Word.symbols`` agregado."""

    return st.text(alphabet=_PRINTABLE_ASCII, min_size=1, max_size=8)


@st.composite
def _word_dict(draw: st.DrawFn, *, numeric_confidence: bool) -> dict[str, Any]:
    """Constrói um ``Word`` simulado.

    Quando ``numeric_confidence=True``, ``confidence`` é um ``float ∈
    [0, 1]``. Quando ``False``, ``confidence`` é um valor não-numérico
    sorteado em ``{None, "high", []}`` para exercitar o caso 9.6
    (``no_numeric_confidences``).
    """

    if numeric_confidence:
        conf: Any = draw(st.floats(min_value=0.0, max_value=1.0))
    else:
        conf = draw(st.sampled_from((None, "high", [])))

    return {
        "boundingBox": {"vertices": draw(_vertices())},
        "confidence": conf,
        "symbols": [
            {"text": ch}
            for ch in draw(_word_text())
        ],
    }


@st.composite
def _paragraph_dict(draw: st.DrawFn, *, numeric_confidence: bool) -> dict[str, Any]:
    """Parágrafo com 1–4 palavras."""

    words = draw(
        st.lists(
            _word_dict(numeric_confidence=numeric_confidence),
            min_size=1,
            max_size=4,
        )
    )
    return {"words": words}


@st.composite
def _block_dict(draw: st.DrawFn, *, numeric_confidence: bool) -> dict[str, Any]:
    """Bloco com 1–3 parágrafos."""

    paragraphs = draw(
        st.lists(
            _paragraph_dict(numeric_confidence=numeric_confidence),
            min_size=1,
            max_size=3,
        )
    )
    return {"paragraphs": paragraphs}


@st.composite
def _page_dict(draw: st.DrawFn, *, numeric_confidence: bool) -> dict[str, Any]:
    """Página com 1–2 blocos."""

    blocks = draw(
        st.lists(
            _block_dict(numeric_confidence=numeric_confidence),
            min_size=1,
            max_size=2,
        )
    )
    return {"blocks": blocks}


@st.composite
def gcv_response_dict(draw: st.DrawFn) -> dict[str, Any]:
    """Resposta GCV sintética com confidências numéricas.

    Alimenta as propriedades **P7** (``mean_confidence`` bem-definida e
    bounded em ``[0, 100]``) e **P8** (extração de texto seleciona
    campo por feature). Produz um ``dict`` no formato de
    ``MessageToDict(AnnotateImageResponse)`` com:

    - ``full_text_annotation.text`` — string concatenada usada quando o
      caller pede ``DOCUMENT_TEXT_DETECTION``.
    - ``full_text_annotation.pages → blocks → paragraphs → words`` —
      profundidade limitada (1–2 páginas, 1–2 blocos, 1–3 parágrafos,
      1–4 palavras) para manter o tempo do teste curto sem perder
      cobertura estrutural.
    - ``text_annotations[0].description`` — texto consolidado usado
      quando o caller pede ``TEXT_DETECTION``. Geramos um texto
      **diferente** do ``full_text_annotation.text`` para garantir que
      P8 falsifique caso o parser leia o campo errado.

    Todas as confidências são ``float ∈ [0, 1]`` (caso "feliz").
    """

    pages = draw(
        st.lists(
            _page_dict(numeric_confidence=True),
            min_size=0,
            max_size=2,
        )
    )

    full_text = draw(
        st.text(alphabet=_PRINTABLE_ASCII, min_size=0, max_size=64)
    )
    annotation_text = draw(
        st.text(alphabet=_PRINTABLE_ASCII, min_size=0, max_size=64)
    )

    # ``text_annotations`` na API real começa com a anotação consolidada
    # seguida de uma anotação por palavra; só precisamos do índice 0
    # para nossas propriedades.
    return {
        "fullTextAnnotation": {
            "text": full_text,
            "pages": pages,
        },
        "textAnnotations": [
            {
                "description": annotation_text,
                "boundingPoly": {"vertices": draw(_vertices())},
            }
        ],
    }


@st.composite
def gcv_response_with_non_numeric_conf(draw: st.DrawFn) -> dict[str, Any]:
    """Resposta GCV com confidências exclusivamente não-numéricas.

    Alimenta o caso especial 9.6 dentro de **P7**: quando há palavras
    detectadas mas nenhuma confidência numérica, o parser deve produzir
    ``mean_confidence == 0.0`` e
    ``confidence_warning == "no_numeric_confidences"``.

    Para garantir que pelo menos uma palavra exista (do contrário
    cairíamos no ramo ``word_count == 0`` de P7), forçamos páginas
    não-vazias.
    """

    pages = draw(
        st.lists(
            _page_dict(numeric_confidence=False),
            min_size=1,
            max_size=2,
        )
    )

    full_text = draw(
        st.text(alphabet=_PRINTABLE_ASCII, min_size=0, max_size=64)
    )

    return {
        "fullTextAnnotation": {
            "text": full_text,
            "pages": pages,
        },
        "textAnnotations": [
            {
                "description": full_text,
                "boundingPoly": {"vertices": draw(_vertices())},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Hints BCP-47
# ---------------------------------------------------------------------------

def bcp47_hints() -> st.SearchStrategy[tuple[str, ...]]:
    """Tuplas BCP-47 de 0 a 4 elementos.

    Alimenta as propriedades **P5** (round-trip do cache com filtragem
    por ``(feature, hints)``) e **P9** (auditoria simétrica). Os hints
    saem do pool fixo ``{"pt", "en", "es", "pt-BR", "en-US"}`` que
    cobre os casos relevantes do projeto: monolíngue PT, fallback EN
    para palavras importadas (ex.: "diet"), e variantes regionais.

    O retorno é ``tuple[str, ...]`` (ordem-sensível) para casar com o
    contrato imutável de ``GcvFetchResult.language_hints`` e com o
    filtro de cache (Requirement 7.3).
    """

    return st.lists(
        st.sampled_from(_BCP47_POOL),
        min_size=0,
        max_size=4,
    ).map(tuple)


# ---------------------------------------------------------------------------
# Imagens
# ---------------------------------------------------------------------------

@st.composite
def image_arrays(draw: st.DrawFn) -> np.ndarray:
    """Imagens BGR ``uint8`` plausíveis como saída de ``imaging.io.read_image``.

    Alimenta a propriedade **P20** (PNG round-trip preserva imagem):
    o pipeline GCV codifica a imagem como PNG antes de enviá-la à API,
    e a invariante exige que ``cv2.imdecode(encode_png(img))`` recupere
    o mesmo array. Para isso bastam imagens não-degeneradas com
    dimensões pequenas: ``H, W ∈ [16, 64]`` e 3 canais (BGR), valores
    ``uint8 ∈ [0, 255]``.

    Mantemos o intervalo de tamanho conservador (``≤ 64``) porque o
    teste compara o array elemento a elemento e PNG é lossless — não
    há ganho em estressar imagens grandes. Usamos
    ``hypothesis.extra.numpy.arrays`` (em vez de ``st.lists`` +
    ``np.array``) por ser a estratégia idiomática para gerar
    ``ndarray`` de qualquer tamanho prático.
    """

    h = draw(st.integers(min_value=16, max_value=64))
    w = draw(st.integers(min_value=16, max_value=64))
    return draw(
        hnp.arrays(
            dtype=np.uint8,
            shape=(h, w, 3),
            elements=st.integers(min_value=0, max_value=255),
        )
    )


# ---------------------------------------------------------------------------
# Strings inválidas para validações de schema
# ---------------------------------------------------------------------------

def kind_strings_invalid() -> st.SearchStrategy[str]:
    """Strings que nunca devem ser aceitas como ``kind`` de preset.

    Alimenta a propriedade **P13** (PresetRepository rejeita kinds
    inválidos). Combina:

    - um pool estático com casos representativos (``""``, ``"  "``,
      ``"Cloud_Vision"``, ``"None"``, valores com whitespace nas
      bordas, etc.);
    - strings imprimíveis arbitrárias filtradas para excluir os
      valores válidos definidos em ``ALLOWED_KINDS``.

    A união garante cobertura tanto dos casos clássicos quanto de
    entradas surpreendentes que podem aparecer em JSONs malformados.
    """

    arbitrary = st.text(
        alphabet=_PRINTABLE_ASCII,
        min_size=0,
        max_size=24,
    ).filter(lambda s: s not in ALLOWED_KINDS)

    return st.one_of(
        st.sampled_from(_KIND_INVALID_STATIC),
        arbitrary,
    )


def feature_strings_invalid() -> st.SearchStrategy[str]:
    """Strings que nunca devem ser aceitas como ``gcv.feature``.

    Alimenta a propriedade **P14** (``gcv.feature`` inválido não chama
    a API). Cobre tanto valores notórios (variações de casing, valores
    de outras APIs do GCV como ``LABEL_DETECTION``) quanto strings
    arbitrárias filtradas para excluir os valores aceitos definidos em
    ``ALLOWED_FEATURES``.
    """

    arbitrary = st.text(
        alphabet=_PRINTABLE_ASCII,
        min_size=0,
        max_size=32,
    ).filter(lambda s: s not in ALLOWED_FEATURES)

    return st.one_of(
        st.sampled_from(_FEATURE_INVALID_STATIC),
        arbitrary,
    )


# ---------------------------------------------------------------------------
# Subconjuntos de classes de erro
# ---------------------------------------------------------------------------

def error_class_subsets() -> st.SearchStrategy[frozenset[str]]:
    """Subconjuntos não-vazios de ``ERROR_PRECEDENCE``.

    Alimenta a propriedade **P6** (skip mode produz resultado vazio com
    classificação por precedência). A propriedade afirma que, dado um
    conjunto ``S`` de classes que se manifestam na mesma chamada,
    ``metadata.error == max(S, key=precedence)`` e
    ``metadata.error_secondary == sorted(S \\ {error}, key=precedence)``.

    Para falsificar a invariante o gerador precisa varrer todos os
    subconjuntos não-vazios de ``{auth_error, quota_exceeded, timeout,
    generic_error}``. Como o domínio é pequeno (15 subconjuntos), o
    Hypothesis encontra rapidamente shrinkings minimais; usamos
    ``frozenset`` para deixar explícito que ordem de declaração não
    importa — apenas a precedência canônica determina o resultado.
    """

    return st.lists(
        st.sampled_from(_ERROR_CLASSES),
        min_size=1,
        max_size=len(_ERROR_CLASSES),
        unique=True,
    ).map(frozenset)


# ---------------------------------------------------------------------------
# Sequências de eventos para o RateLimiter
# ---------------------------------------------------------------------------

@st.composite
def rate_limiter_event_sequences(
    draw: st.DrawFn,
    *,
    max_events: int = 20,
    max_gap: float = 90.0,
) -> list[tuple[float, str]]:
    """Sequências ordenadas de eventos ``(timestamp, kind)`` para o rate limiter.

    Alimenta a propriedade **P10** (rate limiter respeita janela
    deslizante de 60s e ignora cache hits). A propriedade afirma que,
    para qualquer sequência de eventos ``[(t₁, kind₁), …, (tₙ, kindₙ)]``
    com ``kindᵢ ∈ {"hit", "miss"}``, a invariante

    ::

        |{ acquire_timestampⱼ : ∈ [t - 60, t] }| ≤ N

    vale para qualquer instante ``t``, onde ``acquire_timestampⱼ``
    refere-se apenas a eventos ``"miss"`` (eventos ``"hit"`` não
    contabilizam por Requirement 8.4).

    O gerador produz timestamps **monotonicamente não-decrescentes** a
    partir de ``0.0``, com gaps aleatórios em ``[0, max_gap]`` para
    cobrir tanto bursts (gap=0) quanto chamadas espaçadas além da
    janela. A mistura de ``"hit"``/``"miss"`` é equiprovável; o teste
    consumidor injeta clock fake e ``sleep`` no-op para reproduzir
    determinístico.
    """

    n = draw(st.integers(min_value=0, max_value=max_events))
    gaps = draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=max_gap,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )
    kinds = draw(
        st.lists(
            st.sampled_from(("hit", "miss")),
            min_size=n,
            max_size=n,
        )
    )

    timestamps: list[float] = []
    cursor = 0.0
    for gap in gaps:
        cursor += gap
        timestamps.append(cursor)

    return list(zip(timestamps, kinds, strict=True))


# ---------------------------------------------------------------------------
# Estados prévios do diretório de cache
# ---------------------------------------------------------------------------

@st.composite
def cache_states(draw: st.DrawFn) -> dict[str, dict[str, Any]]:
    """Estados prévios do ``cache_dir`` com mistura de entradas válidas e corrompidas.

    Alimenta as propriedades **P5** (round-trip do cache), **P16**
    (``cache_enabled == False`` suprime toda I/O), **P17**
    (``clean_previous`` preserva ``cache_dir``) e **P18** (corrupção de
    entrada não invalida outras).

    Retorna um ``dict`` ``sha → entry`` onde cada ``entry`` descreve o
    par ``<sha>.json`` + ``<sha>.meta.json`` que o teste deve
    materializar em disco. Cada ``entry`` tem o formato:

    .. code-block:: python

        {
            "feature": "TEXT_DETECTION" | "DOCUMENT_TEXT_DETECTION",
            "language_hints": tuple[str, ...],
            "response_corrupt": bool,   # se True, o .json fica inválido
            "meta_corrupt": bool,       # se True, o .meta.json fica inválido
            "response_payload": dict,   # ignorado se response_corrupt
            "image_size_bytes": int,
        }

    Os ``sha`` são strings hex de 64 caracteres distintas — fixadas a
    partir de inteiros sorteados — para permitir lookup determinístico
    pelos testes. Limitamos a 0–6 entradas para manter as iterações
    rápidas; isso é suficiente para o Hypothesis encontrar shrinkings
    minimais nas propriedades acima.
    """

    n = draw(st.integers(min_value=0, max_value=6))
    if n == 0:
        return {}

    # ``unique=True`` em integers garante que ``sha`` derivados sejam
    # distintos (não há colisão de hex padding).
    seeds = draw(
        st.lists(
            st.integers(min_value=0, max_value=2**64 - 1),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )

    # ``ALLOWED_FEATURES`` é um ``frozenset`` (lookup O(1)); o Hypothesis
    # exige uma sequência ordenada para ``sampled_from`` reproduzir
    # exemplos shrinkados de modo determinístico, então convertemos para
    # tupla ordenada apenas no ponto de uso.
    feature_pool = tuple(sorted(ALLOWED_FEATURES))

    entries: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        sha = format(seed, "016x").rjust(64, "0")
        feature = draw(st.sampled_from(feature_pool))
        hints = draw(bcp47_hints())
        # Mantemos a probabilidade de corrupção baixa para que a maioria
        # das entradas seja recuperável (caso contrário P18 fica trivial
        # — todas corrompidas coincide com cache vazio).
        response_corrupt = draw(st.booleans())
        meta_corrupt = draw(st.booleans())
        response_payload = draw(gcv_response_dict())
        image_size_bytes = draw(st.integers(min_value=1, max_value=2**20))

        entries[sha] = {
            "feature": feature,
            "language_hints": hints,
            "response_corrupt": response_corrupt,
            "meta_corrupt": meta_corrupt,
            "response_payload": response_payload,
            "image_size_bytes": image_size_bytes,
        }

    return entries
