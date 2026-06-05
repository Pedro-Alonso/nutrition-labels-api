"""Property tests para o parser de respostas da Google Cloud Vision API.

Cobre as tarefas **3.2**, **3.3** e **3.4** do plano de implementação:

- **Property 7** (task 3.2) — contrato de ``parse_response`` com relação
  a ``mean_confidence`` e ``confidence_warning``. Em palavras: para
  qualquer ``response_json`` válida da Cloud Vision API e qualquer
  ``feature`` aceita, ``parse_response(response_json,
  feature).mean_confidence`` é um ``float`` em ``[0.0, 100.0]``. Mais
  especificamente:

  - Se ``word_count == 0`` ⇒ ``mean_confidence == 0.0`` e
    ``confidence_warning is None`` (Requirement 9.5).
  - Se ``word_count > 0`` e ≥1 palavra tem confidência numérica ⇒
    ``mean_confidence == min(arithmetic_mean(numeric_word_confidences)
    * 100.0, 100.0)`` (Requirements 9.4 e 9.9).
  - Se ``word_count > 0`` e zero palavras têm confidência numérica ⇒
    ``mean_confidence == 0.0`` E ``confidence_warning ==
    "no_numeric_confidences"`` (Requirement 9.6).

- **Property 8** (task 3.3) — extração de texto seleciona o campo
  correto da resposta de acordo com a ``feature`` solicitada. Em
  palavras: para qualquer ``response_json`` com ``fullTextAnnotation``
  e ``textAnnotations`` simultaneamente populados,

  - ``feature == "DOCUMENT_TEXT_DETECTION"`` ⇒ ``parsed.text ==
    response_json["fullTextAnnotation"]["text"]`` (Requirements 9.2 e
    9.8) — o parser **ignora** ``textAnnotations`` mesmo quando o
    campo está presente e populado.
  - ``feature == "TEXT_DETECTION"`` ⇒ ``parsed.text ==
    response_json["textAnnotations"][0]["description"]`` (Requirement
    9.3).

- **Property 20** (task 3.4) — round-trip PNG via ``encode_png``
  preserva a imagem byte a byte (Requirement 9.1). Para qualquer
  imagem BGR ``uint8`` plausível como saída de
  ``imaging.io.read_image``, os bytes produzidos por ``encode_png``
  começam com a assinatura PNG canônica e ``cv2.imdecode`` recupera
  um ``np.ndarray`` exatamente igual ao original — invariante crítica
  para que o SHA-256 dos bytes enviados à API seja determinístico
  (chave do cache, Requirement 7.1).

Os geradores ``gcv_response_dict()`` (caso "feliz" com confidências em
``[0, 1]``) e ``gcv_response_with_non_numeric_conf()`` (caso 9.6) são
combinados via ``one_of`` para varrer os três ramos de P7. O ramo
``word_count == 0`` é exercitado tanto pelo gerador feliz (que pode
produzir ``pages == []``) quanto por um caso degenerado determinístico.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ocr.cloud_vision.parser import encode_png, parse_response
from ocr.cloud_vision.types import ALLOWED_FEATURES
from tests.ocr_engine.gcv.strategies import (
    gcv_response_dict,
    gcv_response_with_non_numeric_conf,
    image_arrays,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_numeric_confidence(value: Any) -> bool:
    """Reproduz o critério de ``parser._is_numeric_confidence``.

    Mantemos uma implementação local em vez de importar a função privada
    do parser para que o teste falsifique a invariante observável
    (``mean_confidence``) sem se acoplar ao detalhe interno. Os dois
    devem permanecer alinhados — qualquer divergência deveria quebrar
    a propriedade.
    """

    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _collect_numeric_confidences(response_json: dict) -> list[float]:
    """Extrai todas as confidências numéricas atravessando ``pages``.

    O parser itera ``full_text_annotation.pages → blocks → paragraphs →
    words`` aceitando chaves em ``camelCase`` ou ``snake_case``. Como os
    geradores em ``tests/gcv/strategies.py`` usam ``camelCase``,
    trabalhamos só com essa variante aqui — alinhado com a forma
    canônica do ``MessageToDict``.
    """

    fta = response_json.get("fullTextAnnotation") or {}
    confidences: list[float] = []
    for page in fta.get("pages", []) or []:
        for block in page.get("blocks", []) or []:
            for paragraph in block.get("paragraphs", []) or []:
                for word in paragraph.get("words", []) or []:
                    if not isinstance(word, dict):
                        continue
                    conf = word.get("confidence")
                    if _is_numeric_confidence(conf):
                        confidences.append(float(conf))
    return confidences


def _count_words(response_json: dict) -> int:
    """Conta palavras totais (independente de confidência ser numérica)."""

    fta = response_json.get("fullTextAnnotation") or {}
    total = 0
    for page in fta.get("pages", []) or []:
        for block in page.get("blocks", []) or []:
            for paragraph in block.get("paragraphs", []) or []:
                total += len(paragraph.get("words", []) or [])
    return total


# ---------------------------------------------------------------------------
# Property 7 — Validates: Requirements 9.4, 9.5, 9.6, 9.9
# ---------------------------------------------------------------------------


@given(
    response_json=st.one_of(
        gcv_response_dict(),
        gcv_response_with_non_numeric_conf(),
    ),
    feature=st.sampled_from(ALLOWED_FEATURES),
)
@settings(max_examples=200, deadline=None)
def test_mean_confidence_is_bounded_in_zero_to_hundred(
    response_json: dict, feature: str
) -> None:
    """**Property 7**: ``mean_confidence`` é sempre um float em ``[0, 100]``.

    **Validates: Requirements 9.4, 9.5, 9.6, 9.9**

    O clamp final ``min(.., 100.0)`` no parser defende contra somas
    com erro de ponto flutuante que poderiam ultrapassar o teto por
    épsilon (Requirement 9.9); aqui afirmamos a invariante bruta sem
    distinguir os ramos — eles são exercidos pelos testes seguintes.
    """

    parsed = parse_response(response_json, feature)

    assert isinstance(parsed.mean_confidence, float)
    assert math.isfinite(parsed.mean_confidence)
    assert 0.0 <= parsed.mean_confidence <= 100.0


@given(response_json=gcv_response_dict(), feature=st.sampled_from(ALLOWED_FEATURES))
@settings(max_examples=200, deadline=None)
def test_mean_confidence_matches_arithmetic_mean_when_numeric(
    response_json: dict, feature: str
) -> None:
    """**Property 7 (caso feliz)**: confidências numéricas viram média × 100.

    **Validates: Requirements 9.4, 9.9**

    Quando há ≥1 palavra com confidência numérica, a invariante exige
    ``mean_confidence == min(mean(confs) * 100, 100)`` — exatamente a
    mesma fórmula que o parser deve aplicar. Comparamos com
    ``math.isclose`` para tolerar a aritmética de ponto flutuante.
    """

    parsed = parse_response(response_json, feature)
    numeric = _collect_numeric_confidences(response_json)
    word_count = _count_words(response_json)

    assert parsed.word_count == word_count

    if word_count == 0:
        assert parsed.mean_confidence == 0.0
        assert parsed.confidence_warning is None
        return

    if not numeric:
        # Cenário improvável neste gerador (todas as confidências são
        # ``float ∈ [0, 1]``) mas ainda permitido pelo gerador caso o
        # Hypothesis encolha para palavras sem ``confidence`` válido.
        # Deixamos o ramo equivalente ao caso 9.6 ser validado pelo
        # próximo teste e apenas verificamos a invariante mínima.
        assert parsed.mean_confidence == 0.0
        assert parsed.confidence_warning == "no_numeric_confidences"
        return

    expected = min(sum(numeric) / len(numeric) * 100.0, 100.0)
    assert math.isclose(
        parsed.mean_confidence, expected, rel_tol=1e-9, abs_tol=1e-9
    ), (
        f"mean_confidence={parsed.mean_confidence!r} difere de "
        f"min(mean({numeric}) * 100, 100) = {expected!r}"
    )
    # Caso feliz nunca dispara o warning de confidências não-numéricas.
    assert parsed.confidence_warning is None


@given(
    response_json=gcv_response_with_non_numeric_conf(),
    feature=st.sampled_from(ALLOWED_FEATURES),
)
@settings(max_examples=200, deadline=None)
def test_mean_confidence_zero_when_all_non_numeric(
    response_json: dict, feature: str
) -> None:
    """**Property 7 (caso 9.6)**: palavras sem confidência numérica.

    **Validates: Requirements 9.6**

    O gerador garante ≥1 página com palavras, e que toda
    ``confidence`` é ``None``, ``"high"`` ou ``[]``. Nessas condições o
    parser deve registrar ``mean_confidence == 0.0`` E sinalizar
    ``confidence_warning == "no_numeric_confidences"`` para que a
    cascata saiba que o ``score`` não tem certeza embutida.
    """

    parsed = parse_response(response_json, feature)

    assert parsed.word_count > 0, (
        "fixture inválida: gerador deveria garantir ≥1 palavra para "
        "exercitar o caminho 9.6"
    )
    assert parsed.mean_confidence == 0.0
    assert parsed.confidence_warning == "no_numeric_confidences"


# ---------------------------------------------------------------------------
# Casos determinísticos auxiliares (cobrem o ramo 9.5 explicitamente)
# ---------------------------------------------------------------------------


def test_mean_confidence_zero_for_empty_word_list() -> None:
    """``word_count == 0`` ⇒ ``mean_confidence == 0`` sem warning.

    **Validates: Requirements 9.5**

    Caso degenerado: resposta sem ``full_text_annotation`` ou com
    ``pages == []``. O parser trata como sucesso degradado — a cascata
    avança via ``QualityEvaluator`` (``score`` baixo por texto vazio).
    Não disparamos ``no_numeric_confidences`` porque não há palavra
    para classificar.
    """

    for response_json in (
        {},
        {"fullTextAnnotation": {"text": "", "pages": []}},
        {
            "fullTextAnnotation": {
                "text": "",
                "pages": [{"blocks": []}],
            }
        },
        {
            "fullTextAnnotation": {
                "text": "",
                "pages": [{"blocks": [{"paragraphs": []}]}],
            }
        },
    ):
        for feature in ALLOWED_FEATURES:
            parsed = parse_response(response_json, feature)
            assert parsed.word_count == 0, (
                f"resposta {response_json!r} deveria ter 0 palavras "
                f"para feature {feature!r}; obteve {parsed.word_count}"
            )
            assert parsed.mean_confidence == 0.0
            assert parsed.confidence_warning is None


def test_mean_confidence_clamped_at_hundred() -> None:
    """Defesa explícita contra ``mean × 100 > 100`` (Requirement 9.9).

    Cenário sintético: confidências exatamente ``1.0`` em todas as
    palavras. A média aritmética é ``1.0`` e ``1.0 * 100 == 100.0``.
    Confirmamos que o parser não estoura o teto e que retorna
    exatamente ``100.0`` (igualdade exata é segura aqui — o cálculo é
    determinístico em ponto flutuante para esse insumo).
    """

    response_json: dict = {
        "fullTextAnnotation": {
            "text": "TOPO",
            "pages": [
                {
                    "blocks": [
                        {
                            "paragraphs": [
                                {
                                    "words": [
                                        {
                                            "boundingBox": {
                                                "vertices": [
                                                    {"x": 0, "y": 0},
                                                    {"x": 1, "y": 0},
                                                    {"x": 1, "y": 1},
                                                    {"x": 0, "y": 1},
                                                ]
                                            },
                                            "confidence": 1.0,
                                            "symbols": [],
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ],
        }
    }

    parsed = parse_response(response_json, "DOCUMENT_TEXT_DETECTION")

    assert parsed.word_count == 1
    assert parsed.mean_confidence == 100.0
    assert parsed.confidence_warning is None


# ---------------------------------------------------------------------------
# Property 8 — Validates: Requirements 9.2, 9.3, 9.8
# ---------------------------------------------------------------------------


@given(response_json=gcv_response_dict())
@settings(max_examples=200, deadline=None)
def test_document_text_detection_reads_full_text_annotation(
    response_json: dict,
) -> None:
    """**Property 8**: ``DOCUMENT_TEXT_DETECTION`` lê ``full_text_annotation``.

    **Validates: Requirements 9.2, 9.8**

    Para toda resposta sintética gerada por ``gcv_response_dict`` —
    que sempre popula tanto ``fullTextAnnotation.text`` quanto
    ``textAnnotations[0].description`` com strings independentes —
    ``parse_response(response_json, "DOCUMENT_TEXT_DETECTION").text``
    deve ser exatamente ``fullTextAnnotation.text``. O parser
    **ignora** ``textAnnotations`` mesmo quando esse campo está
    presente e populado, garantindo que a feature canonical do
    projeto (rótulos densos) sempre consuma o texto consolidado.
    """

    parsed = parse_response(response_json, "DOCUMENT_TEXT_DETECTION")

    expected = response_json["fullTextAnnotation"]["text"]
    assert parsed.text == expected, (
        "DOCUMENT_TEXT_DETECTION deveria ler fullTextAnnotation.text "
        f"={expected!r}; obteve {parsed.text!r}"
    )


@given(response_json=gcv_response_dict())
@settings(max_examples=200, deadline=None)
def test_text_detection_reads_first_text_annotation_description(
    response_json: dict,
) -> None:
    """**Property 8**: ``TEXT_DETECTION`` lê ``textAnnotations[0].description``.

    **Validates: Requirement 9.3**

    Quando o caller solicita ``TEXT_DETECTION``, o parser deve
    devolver a anotação consolidada em
    ``textAnnotations[0].description`` — independente do conteúdo de
    ``fullTextAnnotation.text``. O gerador garante que os dois campos
    contenham strings independentes, então a propriedade falsifica o
    caso de o parser confundir as duas fontes.
    """

    parsed = parse_response(response_json, "TEXT_DETECTION")

    expected = response_json["textAnnotations"][0]["description"]
    assert parsed.text == expected, (
        "TEXT_DETECTION deveria ler textAnnotations[0].description "
        f"={expected!r}; obteve {parsed.text!r}"
    )


@given(response_json=gcv_response_dict())
@settings(max_examples=200, deadline=None)
def test_feature_dispatch_is_independent(response_json: dict) -> None:
    """**Property 8**: as duas features divergem em sua escolha de campo.

    **Validates: Requirements 9.2, 9.3, 9.8**

    Combina os dois ramos numa única invariante: dado o mesmo
    ``response_json``, ``DOCUMENT_TEXT_DETECTION`` retorna
    ``fullTextAnnotation.text`` e ``TEXT_DETECTION`` retorna
    ``textAnnotations[0].description``. Esta invariante falsifica
    qualquer regressão em que uma feature passe a consumir o campo da
    outra (ex.: ``DOCUMENT_TEXT_DETECTION`` caindo silenciosamente em
    ``textAnnotations`` quando ``fullTextAnnotation.text`` é vazio).
    """

    document = parse_response(response_json, "DOCUMENT_TEXT_DETECTION")
    text_only = parse_response(response_json, "TEXT_DETECTION")

    assert document.text == response_json["fullTextAnnotation"]["text"]
    assert (
        text_only.text == response_json["textAnnotations"][0]["description"]
    )


# ---------------------------------------------------------------------------
# Casos determinísticos auxiliares (cobrem 9.8 explicitamente)
# ---------------------------------------------------------------------------


def test_document_text_detection_ignores_text_annotations_when_full_is_empty() -> None:
    """``DOCUMENT_TEXT_DETECTION`` ignora ``textAnnotations`` mesmo se vazia a outra fonte.

    **Validates: Requirement 9.8**

    Caso explícito do design: quando a feature solicitada é
    ``DOCUMENT_TEXT_DETECTION``, o parser **não** deve usar
    ``textAnnotations`` como fallback nem mesmo quando
    ``fullTextAnnotation.text`` é a string vazia. O resultado deve
    ser ``""`` mesmo havendo conteúdo em ``textAnnotations``. Isso
    impede degradação silenciosa entre features.
    """

    response_json: dict = {
        "fullTextAnnotation": {"text": "", "pages": []},
        "textAnnotations": [
            {
                "description": "TEXTO ALTERNATIVO QUE NÃO DEVE VAZAR",
                "boundingPoly": {
                    "vertices": [
                        {"x": 0, "y": 0},
                        {"x": 1, "y": 0},
                        {"x": 1, "y": 1},
                        {"x": 0, "y": 1},
                    ]
                },
            }
        ],
    }

    parsed = parse_response(response_json, "DOCUMENT_TEXT_DETECTION")

    assert parsed.text == ""


def test_text_detection_returns_empty_when_text_annotations_missing() -> None:
    """``TEXT_DETECTION`` devolve ``""`` quando ``textAnnotations`` está ausente ou vazio.

    **Validates: Requirement 9.3**

    Casos degenerados: ausência total de ``textAnnotations``, lista
    vazia, ou primeira entrada sem ``description``. Em todos eles o
    parser deve produzir ``""`` em vez de cair em
    ``fullTextAnnotation`` — preservando a separação canônica entre
    as duas features mesmo em respostas degeneradas.
    """

    cases: list[dict] = [
        {"fullTextAnnotation": {"text": "DENSO", "pages": []}},
        {
            "fullTextAnnotation": {"text": "DENSO", "pages": []},
            "textAnnotations": [],
        },
        {
            "fullTextAnnotation": {"text": "DENSO", "pages": []},
            "textAnnotations": [{"boundingPoly": {"vertices": []}}],
        },
    ]
    for response_json in cases:
        parsed = parse_response(response_json, "TEXT_DETECTION")
        assert parsed.text == "", (
            f"TEXT_DETECTION deveria retornar '' para {response_json!r}; "
            f"obteve {parsed.text!r}"
        )


# ---------------------------------------------------------------------------
# Property 20 — Validates: Requirements 9.1
# ---------------------------------------------------------------------------


# Assinatura canônica de um arquivo PNG (RFC 2083, §3.1):
# bytes 137, 80, 78, 71, 13, 10, 26, 10. A presença desse prefixo no
# resultado de ``encode_png`` é condição necessária para que a Cloud
# Vision API reconheça o payload como imagem PNG; é o sinal mais
# básico de que o codec do OpenCV emitiu o formato esperado.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@given(image=image_arrays())
@settings(max_examples=100, deadline=None)
def test_encode_png_round_trip_preserves_image(image: np.ndarray) -> None:
    """**Property 20**: ``encode_png`` é round-trip lossless.

    **Validates: Requirements 9.1**

    Para qualquer imagem BGR ``uint8`` plausível como saída de
    ``imaging.io.read_image``, a invariante é dupla:

    1. ``encode_png(image)`` começa com a assinatura canônica PNG
       ``\\x89PNG\\r\\n\\x1a\\n`` — o caller (``CloudVisionPipeline``)
       e a própria Cloud Vision API dependem desse prefixo para
       reconhecer o payload, e o teste falsifica qualquer regressão
       em que o codec passe a emitir outro formato (ex.: JPEG por
       configuração equivocada de extensão).
    2. ``cv2.imdecode(np.frombuffer(png_bytes, np.uint8),
       cv2.IMREAD_UNCHANGED)`` devolve um array **exatamente igual**
       ao original (``np.array_equal``). PNG é lossless por
       construção, então a igualdade pode ser exata; tolerância
       numérica indicaria contaminação por compressão com perda.

    A combinação das duas invariantes garante o contrato exigido
    pelo Requirement 9.1: o pipeline codifica a imagem como PNG em
    memória antes de enviá-la à API, sem qualquer alteração visual
    nos pixels — propriedade essencial para que o SHA-256 dos bytes
    enviados (chave do cache, Requirement 7.1) seja determinístico
    e comparável entre execuções.
    """

    png_bytes = encode_png(image)

    # Invariante 1: assinatura PNG.
    assert png_bytes.startswith(_PNG_SIGNATURE), (
        "encode_png deveria produzir bytes começando com a assinatura "
        f"PNG {_PNG_SIGNATURE!r}; obteve prefixo "
        f"{png_bytes[: len(_PNG_SIGNATURE)]!r}"
    )

    # Invariante 2: round-trip lossless via ``cv2.imdecode``.
    # Usamos ``IMREAD_UNCHANGED`` para preservar o número de canais
    # original (3 para BGR), evitando conversões implícitas que
    # ``IMREAD_COLOR`` poderia introduzir.
    decoded = cv2.imdecode(
        np.frombuffer(png_bytes, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )

    assert decoded is not None, (
        "cv2.imdecode retornou None — bytes PNG produzidos por "
        "encode_png não são decodificáveis"
    )
    assert decoded.shape == image.shape, (
        f"shape após round-trip diverge: original={image.shape}, "
        f"decodificado={decoded.shape}"
    )
    assert decoded.dtype == image.dtype, (
        f"dtype após round-trip diverge: original={image.dtype}, "
        f"decodificado={decoded.dtype}"
    )
    assert np.array_equal(decoded, image), (
        "PNG é lossless: cv2.imdecode(encode_png(image)) deveria "
        "recuperar exatamente a imagem original"
    )
