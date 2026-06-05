"""Parser determinístico de respostas da Google Cloud Vision API.

Converte o ``dict`` retornado por ``MessageToDict(AnnotateImageResponse)``
em uma estrutura canônica (``GcvParsedResponse``) consumida pelo
``CloudVisionPipeline`` para preencher ``PipelineResult`` e gerar o
overlay de bounding boxes.

Tolerância a variações de schema
--------------------------------

A serialização de protobuf da Cloud Vision pode emitir tanto chaves em
``camelCase`` (default do ``MessageToDict`` com
``preserving_proto_field_name=False``) quanto em ``snake_case`` (quando
``preserving_proto_field_name=True``). O parser aceita ambas as formas
via o helper interno :func:`_get`, sem assumir nenhum estilo no
chamador. Isso preserva o contrato com fixtures de teste em
``snake_case`` e respostas reais do SDK em ``camelCase``.

Não há rede aqui — o módulo é puro e seguro de importar mesmo sem o
SDK ``google-cloud-vision`` instalado (Requirement 14.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ocr.cloud_vision.types import ALLOWED_FEATURES


# ---------------------------------------------------------------------------
# Estruturas canônicas
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class WordBox:
    """Bounding box axis-aligned de uma palavra detectada.

    A Cloud Vision retorna ``BoundingPoly`` com quatro vértices que
    podem formar um quadrilátero qualquer (rotação, perspectiva). Para
    o overlay simples desenhado pelo pipeline (``cv2.rectangle``)
    precisamos apenas do retângulo axis-aligned envolvente, calculado
    via ``min``/``max`` das coordenadas dos vértices.

    Attributes:
        x1: Coordenada X mínima (canto superior-esquerdo).
        y1: Coordenada Y mínima (canto superior-esquerdo).
        x2: Coordenada X máxima (canto inferior-direito).
        y2: Coordenada Y máxima (canto inferior-direito).
    """

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(slots=True, frozen=True)
class WordToken:
    """Palavra detectada com seu texto e posição na imagem.

    Usado pela reconstrução espacial de tabela no ``CloudVisionPipeline``
    (``table_reconstruction=True``): agrupa palavras por proximidade
    de Y para identificar linhas e por gaps de X para identificar
    colunas, gerando texto tabulado que o ``NutritionTextPostProcessor``
    processa corretamente.

    Attributes:
        text: Texto da palavra conforme detectado pela API.
        box:  Bounding box axis-aligned da palavra.
    """

    text: str
    box: WordBox


@dataclass(slots=True)
class GcvParsedResponse:
    """Resultado canônico de :func:`parse_response`.

    Attributes:
        text: Texto consolidado extraído conforme a feature solicitada
            (``full_text_annotation.text`` para
            ``DOCUMENT_TEXT_DETECTION`` ou
            ``text_annotations[0].description`` para
            ``TEXT_DETECTION``); string vazia quando o campo
            correspondente estiver ausente ou vazio
            (Requirements 9.2, 9.3, 9.8).
        mean_confidence: Confiança média das palavras em escala
            ``[0, 100]`` (Requirements 9.4, 9.5, 9.6, 9.9). Sempre
            ``0.0`` quando ``word_count == 0`` ou quando nenhuma
            palavra tem confidência numérica.
        block_count: Total de blocos em
            ``full_text_annotation.pages → blocks``.
        paragraph_count: Total de parágrafos somando todos os blocos.
        word_count: Total de palavras somando todos os parágrafos
            (independente de a confidência ser numérica).
        word_boxes: Bounding boxes axis-aligned por palavra, na ordem
            em que aparecem em ``pages → blocks → paragraphs → words``.
        word_tokens: Pares (texto, box) extraídos de ``textAnnotations[1:]``
            (lista plana disponível em respostas de ambas as features).
            Usados pela reconstrução espacial de tabela. Campo aditivo —
            pipelines que não usam reconstrução ignoram este campo.
        confidence_warning: ``"no_numeric_confidences"`` quando há
            palavras detectadas mas nenhuma confidência numérica
            (Requirement 9.6); ``None`` caso contrário.
    """

    text: str
    mean_confidence: float
    block_count: int
    paragraph_count: int
    word_count: int
    word_boxes: tuple[WordBox, ...] = ()
    word_tokens: tuple[WordToken, ...] = ()
    confidence_warning: str | None = None


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _get(d: Any, *keys: str) -> Any:
    """Tenta cada chave em ordem em um ``dict``, devolve ``None`` se nenhuma casa.

    A serialização de protobuf da Cloud Vision pode emitir o mesmo
    campo em ``camelCase`` (``fullTextAnnotation``) ou em ``snake_case``
    (``full_text_annotation``) dependendo da configuração do
    ``MessageToDict``. Este helper isola essa ambiguidade dos demais
    callers — eles passam todas as variantes plausíveis e recebem o
    primeiro valor presente.

    Quando ``d`` não é um ``dict`` ou nenhuma das chaves está presente,
    devolve ``None``.
    """

    if not isinstance(d, dict):
        return None
    for key in keys:
        if key in d:
            return d[key]
    return None


def _is_numeric_confidence(value: Any) -> bool:
    """Verifica se um valor de confidência é numérico ``float``/``int`` real.

    A Cloud Vision pode omitir ``confidence``, retornar ``null``, ou —
    em respostas malformadas/serializadas com tipos não usuais —
    devolver strings (``"high"``, ``"n/a"``). Apenas valores
    estritamente numéricos entram no cálculo de
    ``mean_confidence`` (Requirement 9.4).

    O check explícito contra ``bool`` é necessário porque ``True`` e
    ``False`` são instâncias de ``int`` em Python e iriam contaminar a
    média com ``1.0`` e ``0.0`` espúrios.
    """

    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _word_box_from_vertices(vertices: Any) -> WordBox | None:
    """Constrói um :class:`WordBox` axis-aligned a partir de uma lista de vértices.

    Cada vértice é um ``dict`` com chaves opcionais ``x``/``y``;
    coordenadas ausentes são tratadas como ``0`` — esse é o
    comportamento do protobuf da Cloud Vision em casos de palavras
    coladas na borda da imagem (campos com valor zero são omitidos na
    serialização). Quando ``vertices`` não é uma sequência ou está
    vazia, devolve ``None`` (a palavra correspondente é ignorada para
    fins de overlay).
    """

    if not isinstance(vertices, list) or not vertices:
        return None

    xs: list[int] = []
    ys: list[int] = []
    for vertex in vertices:
        if not isinstance(vertex, dict):
            continue
        x = vertex.get("x", 0)
        y = vertex.get("y", 0)
        # Coerção defensiva para ``int`` — algumas serializações usam
        # ``float`` para coordenadas; o overlay com ``cv2.rectangle``
        # exige inteiros.
        try:
            xs.append(int(x))
            ys.append(int(y))
        except (TypeError, ValueError):
            # Vértice mal-formado: pulamos sem invalidar a palavra
            # inteira — ainda pode haver vértices válidos suficientes
            # para um retângulo.
            continue

    if not xs or not ys:
        return None

    return WordBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def _extract_document_text(response_json: dict) -> str:
    """Extrai ``full_text_annotation.text`` (Requirement 9.2/9.8).

    Em ``DOCUMENT_TEXT_DETECTION`` o parser **ignora** ``text_annotations``
    mesmo quando presente — o campo consolidado ``full_text_annotation``
    é a fonte canônica e única para essa feature.
    """

    fta = _get(response_json, "fullTextAnnotation", "full_text_annotation")
    if not isinstance(fta, dict):
        return ""
    text = fta.get("text", "")
    return text if isinstance(text, str) else ""


def _extract_text_annotations_text(response_json: dict) -> str:
    """Extrai ``text_annotations[0].description`` (Requirement 9.3).

    A primeira anotação de ``text_annotations`` é a string consolidada
    do texto completo (a API repete o resultado palavra a palavra nas
    posições seguintes). Se a lista estiver ausente ou vazia,
    devolve ``""``.
    """

    annotations = _get(response_json, "textAnnotations", "text_annotations")
    if not isinstance(annotations, list) or not annotations:
        return ""
    first = annotations[0]
    if not isinstance(first, dict):
        return ""
    description = first.get("description", "")
    return description if isinstance(description, str) else ""


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def parse_response(response_json: dict, feature: str) -> GcvParsedResponse:
    """Converte a resposta crua da Cloud Vision em :class:`GcvParsedResponse`.

    A função segue a tabela de extração definida no design:

    - ``feature == "DOCUMENT_TEXT_DETECTION"`` →
      ``full_text_annotation.text`` (ignora ``text_annotations``).
    - ``feature == "TEXT_DETECTION"`` →
      ``text_annotations[0].description`` (vazio se ausente).
    - Contagens (``block_count``, ``paragraph_count``, ``word_count``)
      iteram ``full_text_annotation.pages → blocks → paragraphs →
      words`` independente da feature, pois a estrutura aninhada está
      sempre presente quando a API processa a imagem com sucesso.
    - ``mean_confidence`` agrega apenas confidências numéricas;
      ``confidence_warning`` é populado conforme Requirement 9.6.
    - ``word_boxes`` é a sequência de retângulos axis-aligned
      consumidos pelo overlay PNG do pipeline (Requirement 10.2).

    Args:
        response_json: ``dict`` da resposta GCV (pode ter chaves em
            ``camelCase`` ou ``snake_case`` — ambos os estilos são
            aceitos transparentemente).
        feature: Modalidade da chamada (``"DOCUMENT_TEXT_DETECTION"``
            ou ``"TEXT_DETECTION"``). Qualquer outro valor cai no
            ramo de ``DOCUMENT_TEXT_DETECTION`` por defesa, mas
            espera-se que o caller já tenha validado o valor antes.

    Returns:
        ``GcvParsedResponse`` com todos os campos populados.
    """

    # --- Texto consolidado conforme a feature -------------------------------
    # ``ALLOWED_FEATURES`` é a fonte canônica do conjunto válido
    # (Requirement 3.4); a validação de pertinência cabe ao
    # ``GcvPresetOptions``, mas referenciamos a constante aqui para
    # manter o vocabulário consistente entre módulos. Quando o caller
    # passa um valor fora desse conjunto (cenário defensivo — o
    # pipeline bloqueia ``invalid_feature`` antes do parser), caímos
    # no default ``DOCUMENT_TEXT_DETECTION`` para nunca falhar.
    if feature not in ALLOWED_FEATURES:
        feature = "DOCUMENT_TEXT_DETECTION"
    if feature == "TEXT_DETECTION":
        text = _extract_text_annotations_text(response_json)
    else:
        text = _extract_document_text(response_json)

    # --- Iteração estrutural para contagens, confidências e bboxes ----------
    block_count = 0
    paragraph_count = 0
    word_count = 0
    numeric_confidences: list[float] = []
    word_boxes: list[WordBox] = []

    fta = _get(response_json, "fullTextAnnotation", "full_text_annotation")
    pages = _get(fta, "pages") if isinstance(fta, dict) else None
    if isinstance(pages, list):
        for page in pages:
            blocks = _get(page, "blocks")
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                block_count += 1
                paragraphs = _get(block, "paragraphs")
                if not isinstance(paragraphs, list):
                    continue
                for paragraph in paragraphs:
                    paragraph_count += 1
                    words = _get(paragraph, "words")
                    if not isinstance(words, list):
                        continue
                    for word in words:
                        word_count += 1
                        if not isinstance(word, dict):
                            continue
                        # Confidência: aceita apenas ``int``/``float``
                        # estritos (descartando ``bool`` por ser
                        # subclasse de ``int``).
                        confidence = word.get("confidence")
                        if _is_numeric_confidence(confidence):
                            numeric_confidences.append(float(confidence))
                        # Bounding box: aceita ``boundingBox`` ou
                        # ``bounding_box``; coordenadas ausentes em
                        # vértices viram ``0``.
                        bbox = _get(word, "boundingBox", "bounding_box")
                        vertices = _get(bbox, "vertices") if isinstance(bbox, dict) else None
                        word_box = _word_box_from_vertices(vertices)
                        if word_box is not None:
                            word_boxes.append(word_box)

    # --- mean_confidence + confidence_warning -------------------------------
    confidence_warning: str | None = None
    if word_count == 0:
        # Caminho 9.5: nenhuma palavra detectada → confidência neutra
        # sem warning (a ausência de texto já é o sinal para a cascata).
        mean_confidence = 0.0
    elif not numeric_confidences:
        # Caminho 9.6: existem palavras mas nenhuma confidência
        # numérica → o pipeline marca o warning para que o operador
        # saiba que o ``score`` da cascata não tem certeza embutida.
        mean_confidence = 0.0
        confidence_warning = "no_numeric_confidences"
    else:
        # Caminho 9.4/9.9: média em ``[0, 1]`` × 100 com clamp em
        # ``100.0`` para defender contra somas com erro de ponto
        # flutuante que ultrapassam o teto por épsilon.
        average = sum(numeric_confidences) / len(numeric_confidences)
        mean_confidence = min(average * 100.0, 100.0)

    # --- word_tokens: pares (texto, box) de textAnnotations[1:] ----------
    # ``textAnnotations`` (lista plana) está presente em respostas de ambas
    # as features e fornece diretamente o texto de cada palavra via
    # ``description``. Saltamos o índice 0 (concatenação completa do texto).
    # As posições vêm de ``boundingPoly.vertices`` — reutilizamos
    # ``_word_box_from_vertices`` já usado para ``word_boxes``.
    word_tokens: list[WordToken] = []
    text_annotations = _get(response_json, "textAnnotations", "text_annotations")
    if isinstance(text_annotations, list):
        for annotation in text_annotations[1:]:
            if not isinstance(annotation, dict):
                continue
            word_text = annotation.get("description", "")
            if not isinstance(word_text, str) or not word_text:
                continue
            bp = _get(annotation, "boundingPoly", "bounding_poly")
            vertices = _get(bp, "vertices") if isinstance(bp, dict) else None
            token_box = _word_box_from_vertices(vertices)
            if token_box is not None:
                word_tokens.append(WordToken(text=word_text, box=token_box))

    return GcvParsedResponse(
        text=text,
        mean_confidence=mean_confidence,
        block_count=block_count,
        paragraph_count=paragraph_count,
        word_count=word_count,
        word_boxes=tuple(word_boxes),
        word_tokens=tuple(word_tokens),
        confidence_warning=confidence_warning,
    )


# ---------------------------------------------------------------------------
# Codificação PNG para envio à API
# ---------------------------------------------------------------------------


def encode_png(image: np.ndarray) -> bytes:
    """Codifica um array ``np.ndarray`` como PNG em memória (Requirement 9.1).

    A Cloud Vision API aceita bytes brutos de imagem em qualquer formato
    suportado; usamos PNG por ser lossless (o hash SHA-256 dos bytes é
    a chave do cache, então a codificação precisa ser determinística e
    sem perda) e por estar disponível no OpenCV via
    ``cv2.imencode(".png", image)`` sem dependências extras.

    Args:
        image: Array no layout esperado pelo OpenCV (BGR ``uint8`` ou
            grayscale ``uint8``). O tamanho/formato é validado pelo
            próprio ``cv2.imencode``.

    Returns:
        Bytes PNG codificados, prontos para serem enviados à API ou
        gravados em disco.

    Raises:
        RuntimeError: Quando ``cv2.imencode`` retorna ``success=False``
            (formato não suportado, array vazio, dtype incompatível).
    """

    success, buffer = cv2.imencode(".png", image)
    if not success:
        # ``cv2.imencode`` falha silenciosamente devolvendo
        # ``success=False`` em vez de levantar; explicitamos a falha
        # como ``RuntimeError`` para que o pipeline classifique-a
        # como erro pré-API e aplique ``on_failure``.
        raise RuntimeError("falha ao codificar imagem como PNG")
    return bytes(buffer)
