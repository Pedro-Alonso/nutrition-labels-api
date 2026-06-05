"""Tipos canônicos e constantes da integração com a Google Cloud Vision API.

Centraliza os tipos compartilhados entre ``GcvClient``, ``CloudVisionPipeline``
e ``PresetRepository``. Não importa o SDK ``google-cloud-vision`` para que este
módulo seja seguro de carregar mesmo quando a dependência opcional não está
instalada (ver Requirement 14.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constantes compartilhadas
# ---------------------------------------------------------------------------

# Conjunto de ``kind`` aceitos pelo ``PresetRepository``. ``cloud_vision`` é o
# novo executor desta feature; os demais valores preservam o contrato Tesseract
# já existente (Requirement 1.2). Tupla imutável: preserva ordem para
# documentação/iteração e funciona naturalmente com ``in``.
ALLOWED_KINDS: tuple[str, ...] = (
    "linear_table",
    "linear_text",
    "linear_ingredient",
    "cell_based",
    "cloud_vision",
)

# Modalidades aceitas pelo bloco ``gcv.feature`` do preset (Requirement 3.4).
# Qualquer valor declarado fora deste conjunto deve ser tratado como
# ``invalid_feature`` e disparar o caminho de falha definido em
# ``CloudVisionPipeline`` (Requirement 3.5).
ALLOWED_FEATURES: tuple[str, ...] = (
    "TEXT_DETECTION",
    "DOCUMENT_TEXT_DETECTION",
)

# Precedência fixa para classificação de erros da chamada à GCV (Requirements
# 6.5–6.8). Quando múltiplas classes se aplicam à mesma exceção, a primeira
# desta tupla vence e as demais vão para ``GcvError.secondary`` ordenadas por
# esta mesma precedência. ``import_error`` e ``invalid_feature`` são códigos
# pré-API e fluem por caminhos disjuntos — não participam desta ordenação
# (ver design.md, seção "Resumo das classificações de erro").
ERROR_PRECEDENCE: tuple[str, ...] = (
    "auth_error",
    "quota_exceeded",
    "timeout",
    "generic_error",
)


# ---------------------------------------------------------------------------
# Exceção classificada
# ---------------------------------------------------------------------------


class GcvError(Exception):
    """Erro classificado da camada Google Cloud Vision.

    Encapsula a classificação canônica usada pelo pipeline em ``metadata.error``
    e ``metadata.error_secondary``. A representação textual inclui tanto o
    código de classificação (``error``) quanto a descrição original
    (``message``), facilitando inspeção em logs estruturados sem perder a
    classe que decidirá o caminho do ``on_failure``.

    Attributes:
        error: Classe primária do erro. Pode pertencer a ``ERROR_PRECEDENCE``
            (erros de chamada à API) ou ser um código pré-API
            (``"import_error"``, ``"invalid_feature"``) — ver design.md.
        message: Descrição original da exceção (truncada a 500 caracteres
            pelo cliente, conforme Requirement 6.2).
        secondary: Tupla com classes adicionais que também se aplicam à mesma
            exceção, ordenadas por precedência. Vazia quando apenas uma classe
            foi identificada ou em casos pré-API.
    """

    error: str
    message: str
    secondary: tuple[str, ...]

    def __init__(
        self,
        error: str,
        message: str,
        secondary: tuple[str, ...] = (),
    ) -> None:
        # Passamos uma representação combinada para ``Exception`` para que
        # ``str(exc)`` carregue tanto a classificação quanto a mensagem
        # original — útil em logs e tracebacks sem perder o contexto do
        # ``on_failure`` que decidirá o caminho de tratamento.
        super().__init__(f"[{error}] {message}")
        self.error = error
        self.message = message
        self.secondary = tuple(secondary)


# ---------------------------------------------------------------------------
# Resultado canônico do cliente
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GcvFetchResult:
    """Resultado canônico devolvido por ``GcvClient.fetch``.

    Reúne a resposta crua da API (ou do cache) junto com os metadados
    necessários para o ``CloudVisionPipeline`` decidir auditoria, parsing e
    propagação de ``cache_hit`` (Requirements 7.4, 9.7, 10.5).

    Attributes:
        response_json: Resposta da GCV serializada como ``dict`` (formato de
            ``MessageToDict(AnnotateImageResponse)``). Em cache hit, é o
            conteúdo lido do disco; em chamada real, é o resultado fresco.
        cache_hit: ``True`` quando a resposta veio do ``GcvCache``;
            ``False`` quando houve chamada real à API.
        feature: Modalidade efetivamente usada (``TEXT_DETECTION`` ou
            ``DOCUMENT_TEXT_DETECTION``); coincide com o que foi gravado no
            ``.meta.json`` do cache.
        language_hints: Hints BCP-47 enviados à API (ou usados como chave de
            filtro do cache). Tupla para preservar imutabilidade
            ordem-sensível, alinhada com o filtro de compatibilidade do cache
            (Requirement 7.3).
    """

    response_json: dict
    cache_hit: bool
    feature: str
    language_hints: tuple[str, ...] = field(default_factory=tuple)
