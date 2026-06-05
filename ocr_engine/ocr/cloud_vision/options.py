"""Opções declarativas do bloco ``gcv`` de cada preset GCV.

Espelha o subset do JSON de preset que o ``CloudVisionPipeline`` consome para
montar a request à Google Cloud Vision API. A leitura é tolerante a campos
ausentes (aplica defaults documentados — Requirements 3.3 e 3.6) e a valores
inválidos de ``feature`` (sinaliza ``invalid_feature=True`` para que o
pipeline curto-circuite *antes* de qualquer I/O — Requirement 3.5).

A classe é ``frozen=True`` para que instâncias possam ser reutilizadas com
segurança entre tentativas/imagens sem risco de mutação acidental.
"""

from __future__ import annotations

from dataclasses import dataclass

from ocr.cloud_vision.types import ALLOWED_FEATURES


# Default canônico quando ``gcv.feature`` está ausente do preset (Requirement
# 3.3) — também é o valor "seguro" exposto em ``feature`` quando o usuário
# declarou um valor inválido, de modo que o pipeline tenha sempre uma string
# coerente para anexar a ``metadata`` mesmo no caminho ``invalid_feature``.
_DEFAULT_FEATURE: str = "DOCUMENT_TEXT_DETECTION"

# Default canônico para ``gcv.language_hints`` (Requirement 3.6). Tupla
# imutável: ordem-sensível é o contrato esperado pela GCV (lista de
# prioridade) e pelo filtro de cache (``GcvCache.get`` compara por igualdade
# de tupla).
_DEFAULT_LANGUAGE_HINTS: tuple[str, ...] = ("pt",)


@dataclass(slots=True, frozen=True)
class GcvPresetOptions:
    """Opções operacionais derivadas do bloco ``gcv`` do preset.

    Attributes:
        feature: Modalidade efetivamente utilizada na request. Sempre uma
            string pertencente a ``ALLOWED_FEATURES``; quando o usuário
            declarou um valor inválido, ``feature`` recebe o default
            (``DOCUMENT_TEXT_DETECTION``) e ``invalid_feature=True`` sinaliza
            que o pipeline deve curto-circuitar antes da chamada.
        language_hints: Hints BCP-47 enviados à API. Tupla imutável para
            preservar a ordem (alinhada ao contrato do cache) e evitar
            mutação compartilhada entre tentativas.
        model: Identificador opcional de modelo repassado à API quando
            declarado. ``None`` quando ausente do preset.
        invalid_feature: ``True`` apenas quando a chave ``"feature"`` foi
            declarada no preset com um valor fora de ``ALLOWED_FEATURES``
            (Requirement 3.5). ``False`` quando a chave está ausente
            (default aplicado) ou quando o valor declarado é válido.
        raw_feature: Valor original declarado em ``data["feature"]``,
            preservado para diagnóstico em ``metadata`` quando
            ``invalid_feature=True``. ``None`` quando a chave está ausente.
            Em casos válidos, contém a mesma string de ``feature`` para
            simplificar o consumo (não há ramos especiais para "valor não
            declarado vs declarado igual ao default").
        table_reconstruction: Quando ``True``, o ``CloudVisionPipeline``
            usa as posições dos word tokens (``textAnnotations``) para
            reconstruir a estrutura de tabela, produzindo texto com ``\t``
            entre colunas detectadas por gap espacial. Ativar apenas em
            presets da categoria ``table``. Default ``False`` preserva o
            comportamento anterior para presets de texto e ingredientes.
    """

    feature: str
    language_hints: tuple[str, ...]
    model: str | None
    invalid_feature: bool
    raw_feature: str | None
    # Dimensão máxima (em pixels, lado maior) antes de codificar para PNG e
    # enviar à API. Imagens acima desse limite são redimensionadas
    # proporcionalmente pelo ``CloudVisionPipeline`` antes de ``encode_png``,
    # evitando timeouts e o limite de 20 MB da API para conteúdo inline.
    # ``0`` desabilita o redimensionamento (útil para testes com imagens
    # sintéticas pequenas). Default 1500 cobre a maioria dos rótulos
    # fotografados em alta resolução sem perda perceptível de OCR.
    max_image_dimension: int = 1500
    table_reconstruction: bool = False

    @classmethod
    def from_dict(cls, data: dict | None) -> "GcvPresetOptions":
        """Constrói uma instância a partir do bloco ``gcv`` do preset.

        Aplica defaults documentados quando campos estão ausentes
        (Requirements 3.3 e 3.6) e marca ``invalid_feature=True`` apenas
        quando a chave ``"feature"`` foi declarada no JSON com um valor
        fora de ``ALLOWED_FEATURES`` (Requirement 3.5). Ausência da chave
        não dispara a flag — cai silenciosamente no default.

        Args:
            data: Dicionário do bloco ``gcv`` do preset. Pode ser ``None``
                (bloco ausente) ou ``{}`` (bloco declarado vazio); ambos
                produzem o mesmo resultado: defaults para todos os campos
                e ``invalid_feature=False``.

        Returns:
            Instância imutável com os campos canônicos. Quando o usuário
            declara ``feature`` inválido, ``feature`` ainda recebe o
            default ``DOCUMENT_TEXT_DETECTION`` (string segura para
            consumidores), ``raw_feature`` retém o valor original e
            ``invalid_feature=True`` instrui o pipeline a curto-circuitar
            antes de qualquer chamada à API ou consulta ao cache.
        """

        # Tratamento uniforme de ``None`` e ``{}``: ambos significam "bloco
        # ``gcv`` ausente do preset" e devem cair em todos os defaults sem
        # marcar ``invalid_feature``.
        source: dict = data if data else {}

        # Campo ``feature``:
        # - chave ausente → ``feature`` recebe o default e ``raw_feature``
        #   permanece ``None`` (Requirement 3.3 + Property 4 do design).
        # - chave presente e valor válido → ``feature`` e ``raw_feature``
        #   recebem o valor declarado.
        # - chave presente e valor inválido → ``feature`` recebe o default
        #   (string segura), ``raw_feature`` preserva o valor original
        #   (coercido para ``str`` quando necessário) e ``invalid_feature``
        #   é ativado (Requirement 3.5).
        if "feature" in source:
            declared = source["feature"]
            # Preserva tipo string quando possível; demais tipos são
            # convertidos para ``str`` apenas para diagnóstico em
            # ``metadata`` (não afetam o caminho de execução, que é
            # decidido por ``invalid_feature``).
            raw_feature: str | None = declared if isinstance(declared, str) else str(declared)
            if isinstance(declared, str) and declared in ALLOWED_FEATURES:
                feature = declared
                invalid_feature = False
            else:
                # Valor declarado fora do conjunto aceito (string desconhecida,
                # ``None``, número, etc.). Conforme contrato: mantém um
                # ``feature`` seguro para consumidores e sinaliza o erro via
                # ``invalid_feature``; o pipeline traduzirá isso em
                # ``metadata.error == "invalid_feature"`` sem chamar a API.
                feature = _DEFAULT_FEATURE
                invalid_feature = True
        else:
            feature = _DEFAULT_FEATURE
            invalid_feature = False
            raw_feature = None

        # Campo ``language_hints``:
        # - chave ausente → default ``("pt",)`` (Requirement 3.6).
        # - chave presente → coerção para tupla, preservando ordem
        #   (ordem-sensível por contrato da GCV e do cache).
        if "language_hints" in source:
            hints_value = source["language_hints"]
            language_hints: tuple[str, ...] = tuple(hints_value)
        else:
            language_hints = _DEFAULT_LANGUAGE_HINTS

        # Campo ``model``:
        # - chave ausente OU declarada como ``None`` → ``None``.
        # - demais valores → repassados como estão (a API aceita strings).
        model = source.get("model", None)

        # Campo ``max_image_dimension``:
        # - chave ausente → default 1500 px (adequado para rótulos reais).
        # - valor ≤ 0 → desabilita redimensionamento (0 = sem limite).
        # - valores não-inteiros são coercidos ou descartados para o default.
        raw_dim = source.get("max_image_dimension", 1500)
        try:
            max_image_dimension = max(0, int(raw_dim))
        except (TypeError, ValueError):
            max_image_dimension = 1500

        # Campo ``table_reconstruction``:
        # - chave ausente → ``False`` (comportamento legado preservado).
        # - valor truthy → ``True``; falsy → ``False``. Coerção via
        #   ``bool()`` aceita 1/0, "true"/"false" não são verificados
        #   mas o JSON correto usa ``true``/``false`` booleano.
        table_reconstruction = bool(source.get("table_reconstruction", False))

        return cls(
            feature=feature,
            language_hints=language_hints,
            model=model,
            invalid_feature=invalid_feature,
            raw_feature=raw_feature,
            max_image_dimension=max_image_dimension,
            table_reconstruction=table_reconstruction,
        )
