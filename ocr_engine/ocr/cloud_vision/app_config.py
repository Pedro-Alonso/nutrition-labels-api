"""Configuração global do bloco ``gcv`` em ``config/app.json``.

Espelha o subset operacional consumido por ``GcvClient``, ``GcvCache`` e
``RateLimiter``. A leitura é tolerante a campos ausentes (aplica defaults
documentados no design — Requirement 4.2) e a valores inválidos de
``max_requests_per_minute``, que são silenciosamente coercidos para ``None``
gerando um warning humanamente legível em ``config_warnings`` (Requirement
8.5). Esses warnings são consumidos uma única vez pelo ``NutritionReader``
na primeira tentativa GCV da execução, via ``metadata.gcv_config_warnings``.

A classe é ``frozen=True`` para que a configuração resolvida possa ser
compartilhada entre o cliente, o cache e o rate limiter sem risco de
mutação acidental durante a execução de um lote.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults documentados no design (Requirement 4.2)
# ---------------------------------------------------------------------------

# Política default de tratamento de falha da chamada à API. ``"skip"`` permite
# que a cascata Tesseract continue como degradação graciosa (ver Requirement
# 6.1).
_DEFAULT_ON_FAILURE: str = "skip"

# Conjunto fechado de políticas aceitas (Requirement 6.1). Qualquer outro
# valor declarado em ``app.json`` cai silenciosamente no default — não há
# erro fatal de boot porque a feature inteira é opcional (Requirement 14.3).
_ALLOWED_ON_FAILURE: frozenset[str] = frozenset({"skip", "raise"})

# Cache habilitado por padrão (Requirement 4.2). Evita custos repetidos em
# re-execuções idênticas e desacopla os experimentos da disponibilidade da
# rede (Requirement 7.1).
_DEFAULT_CACHE_ENABLED: bool = True

# Diretório default do cache, relativo ao ``project_root`` (Requirement 4.2).
# Vive ao lado de ``extractions/<input>/`` mas FORA dele, de modo que
# ``AuditRecorder.clean_previous`` não toque suas entradas (Requirement 7.6).
_DEFAULT_CACHE_DIR_PARTS: tuple[str, ...] = ("extractions", ".gcv_cache")

# Timeout default da chamada à API em segundos (Requirement 4.2). Aplicado
# pelo ``GcvClient`` via ``concurrent.futures`` para classificação canônica
# de ``timeout`` (Requirement 6.7).
_DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 30.0


@dataclass(slots=True, frozen=True)
class GcvAppConfig:
    """Configuração resolvida do bloco ``gcv`` de ``app.json``.

    Attributes:
        credentials_path: Path absoluto do Service Account quando declarado
            em ``app.json::gcv.credentials_path`` (resolvido relativo a
            ``project_root`` quando o valor for relativo). ``None`` quando o
            campo está ausente, é ``null``, vazio ou de tipo incompatível —
            nesse caso o ``auth.resolve_credentials`` cairá no fallback de
            ``GOOGLE_APPLICATION_CREDENTIALS`` (Requirements 5.1, 5.2). A
            existência do arquivo NÃO é validada aqui — esta classe é
            puramente declarativa.
        on_failure: Política de tratamento de falha (``"skip"`` ou
            ``"raise"``). Valores fora do conjunto aceito caem no default
            ``"skip"`` silenciosamente (Requirement 6.1).
        cache_enabled: ``True`` quando o cache em disco está habilitado
            (Requirements 4.2, 7.5). Coerção via ``bool(...)`` quando o
            valor declarado não for booleano puro.
        cache_dir: Path absoluto do diretório do cache em disco, resolvido
            relativo a ``project_root`` quando o valor declarado for
            relativo. Default ``project_root/extractions/.gcv_cache``.
        max_requests_per_minute: Inteiro positivo quando o operador declara
            um limite explícito (Requirement 8.2); ``None`` quando o campo
            está ausente, é ``null``, ou foi coercido por valor inválido
            (Requirements 8.1, 8.5).
        request_timeout_seconds: Timeout em segundos para a chamada
            síncrona à API (Requirement 6.7). Default 30.0. Valores
            inválidos caem no default sem warning (apenas o coercion do
            rate limiter é exigido pelo Requirement 8.5).
        config_warnings: Tupla imutável de mensagens em português
            descrevendo coerções aplicadas durante o parsing (ex.:
            ``"max_requests_per_minute=-5 inválido; rate limiter
            desabilitado"``). O ``NutritionReader`` consome esta tupla uma
            única vez, propagando-a para ``metadata.gcv_config_warnings``
            da primeira tentativa GCV da execução (Requirement 8.5).
    """

    credentials_path: Path | None
    on_failure: str
    cache_enabled: bool
    cache_dir: Path
    max_requests_per_minute: int | None
    request_timeout_seconds: float
    config_warnings: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        data: dict | None,
        project_root: Path,
    ) -> "GcvAppConfig":
        """Constrói uma instância a partir do bloco ``gcv`` de ``app.json``.

        Aplica defaults documentados quando campos estão ausentes
        (Requirement 4.2) e coage ``max_requests_per_minute`` para ``None``
        com warning textual quando o valor declarado é não-positivo,
        booleano, string, float ou de tipo incompatível (Requirements 8.1
        e 8.5). Demais campos com valores inválidos caem silenciosamente
        no default — apenas a coerção do rate limiter é exigida pelo
        contrato.

        Args:
            data: Dicionário do bloco ``gcv`` de ``app.json``. Pode ser
                ``None`` (bloco ausente) ou ``{}`` (bloco declarado vazio);
                ambos produzem o mesmo resultado: defaults para todos os
                campos e ``config_warnings`` vazio (Requirement 4.2).
            project_root: Raiz do projeto usada para resolver paths
                relativos declarados em ``credentials_path`` e
                ``cache_dir``. Esperado como ``Path`` absoluto pelo
                chamador (``build_default_reader`` em ``main.py``).

        Returns:
            Instância imutável com paths resolvidos a absolutos e valores
            coerentes para o consumo direto por ``GcvClient.build`` e
            ``CloudVisionPipeline``.
        """

        # Tratamento uniforme de ``None`` e ``{}``: ambos significam "bloco
        # ``gcv`` ausente de ``app.json``" e devem produzir defaults para
        # todos os campos sem gerar warnings (Requirement 4.2).
        source: dict = data if data else {}

        # Acumulador de warnings de coerção. É uma lista mutável durante a
        # construção e congelada em tupla no retorno para preservar a
        # imutabilidade da instância ``frozen``.
        warnings: list[str] = []

        # ------------------------------------------------------------------
        # Campo ``credentials_path``:
        # - chave ausente, ``None`` ou string vazia → ``None`` (auth resolver
        #   tentará o fallback de env var; Requirement 5.2).
        # - string truthy → resolvido relativo a ``project_root`` quando
        #   relativo, mantido como está quando absoluto.
        # - tipos incompatíveis → ``None`` silenciosamente (a feature inteira
        #   é opcional — Requirement 14.3 — e não justifica abortar o boot).
        # A existência do arquivo NÃO é validada aqui; ``auth
        # .resolve_credentials`` é o ponto canônico dessa verificação
        # (Requirement 5.1).
        # ------------------------------------------------------------------
        raw_credentials = source.get("credentials_path")
        credentials_path: Path | None
        if isinstance(raw_credentials, str) and raw_credentials.strip():
            candidate = Path(raw_credentials)
            credentials_path = candidate if candidate.is_absolute() else project_root / candidate
        else:
            credentials_path = None

        # ------------------------------------------------------------------
        # Campo ``on_failure``:
        # - chave ausente → default ``"skip"`` (Requirement 6.1), sem warning.
        # - valor válido (``"skip"`` ou ``"raise"``) → repassado.
        # - valor declarado mas fora de ``_ALLOWED_ON_FAILURE`` → fallback
        #   ``"skip"`` + warning textual em ``config_warnings``. O contrato
        #   da feature explicita que valores inválidos devem ser sinalizados
        #   ao operador (mesma diretriz aplicada a ``max_requests_per_minute``)
        #   para evitar surpresas silenciosas em produção.
        # ------------------------------------------------------------------
        on_failure: str
        if "on_failure" not in source:
            on_failure = _DEFAULT_ON_FAILURE
        else:
            raw_on_failure = source["on_failure"]
            if isinstance(raw_on_failure, str) and raw_on_failure in _ALLOWED_ON_FAILURE:
                on_failure = raw_on_failure
            else:
                on_failure = _DEFAULT_ON_FAILURE
                warnings.append(
                    f"on_failure={raw_on_failure!r} inválido (esperado 'skip' "
                    f"ou 'raise'); usando default {_DEFAULT_ON_FAILURE!r}"
                )

        # ------------------------------------------------------------------
        # Campo ``cache_enabled``:
        # - chave ausente → default ``True`` (Requirement 4.2).
        # - valor presente → coerção via ``bool(...)`` para tolerar
        #   ``0/1``/``"true"``-como-truthy. Não há warning porque a coerção
        #   booleana é trivial e o operador raramente declara o campo com
        #   tipo errado.
        # ------------------------------------------------------------------
        if "cache_enabled" in source:
            cache_enabled = bool(source["cache_enabled"])
        else:
            cache_enabled = _DEFAULT_CACHE_ENABLED

        # ------------------------------------------------------------------
        # Campo ``cache_dir``:
        # - chave ausente, ``None`` ou string vazia → default
        #   ``project_root/extractions/.gcv_cache`` (Requirement 4.2).
        # - string truthy → resolvido relativo a ``project_root`` quando
        #   relativo. Permite que o operador relocate o cache para um
        #   volume separado declarando absolute path.
        # ------------------------------------------------------------------
        raw_cache_dir = source.get("cache_dir")
        cache_dir: Path
        if isinstance(raw_cache_dir, str) and raw_cache_dir.strip():
            candidate_dir = Path(raw_cache_dir)
            cache_dir = candidate_dir if candidate_dir.is_absolute() else project_root / candidate_dir
        else:
            cache_dir = project_root.joinpath(*_DEFAULT_CACHE_DIR_PARTS)

        # ------------------------------------------------------------------
        # Campo ``max_requests_per_minute`` — única coerção com warning
        # exigida pelo contrato (Requirement 8.5 + Property 11 do design).
        # Aceitos sem warning:
        #   - chave ausente → ``None`` (default global, Requirement 4.2).
        #   - inteiro positivo (NÃO ``bool``, que é subclasse de ``int``) →
        #     mantido como está (Requirement 8.2).
        # Coercidos para ``None`` com warning quando a chave é DECLARADA com
        # qualquer outro valor (Property 11 do design e tasks.md §2.3):
        #   - ``None`` explícito → o operador declarou ``null`` e o
        #     contrato registra essa escolha como warning para que o
        #     ``_summary.json`` da primeira tentativa GCV documente o
        #     comportamento (rate limiter desabilitado intencionalmente).
        #   - inteiro ``≤ 0`` → não-positivo.
        #   - ``bool`` (``True``/``False``) → booleano nunca representa
        #     uma cota válida; tratado antes do branch ``int`` porque
        #     ``bool`` é subclasse de ``int`` em Python e ``True == 1``
        #     enganaria a validação seguinte.
        #   - ``str``, ``float``, qualquer outro tipo → não-numérico no
        #     domínio do contrato (Requirement 8.2 exige inteiro estrito).
        # O warning carrega ``repr(valor_original)`` para que o operador
        # possa identificar exatamente o que estava em ``app.json``.
        # ------------------------------------------------------------------
        max_requests_per_minute: int | None
        if "max_requests_per_minute" not in source:
            max_requests_per_minute = None
        else:
            raw_rpm = source["max_requests_per_minute"]
            if raw_rpm is None:
                # ``null`` declarado explicitamente. Property 11 do design
                # exige warning também neste caso, distinguindo "não
                # declarado" (default silencioso) de "declarado como
                # null" (escolha consciente que merece registro em
                # ``metadata.gcv_config_warnings``).
                max_requests_per_minute = None
                warnings.append(
                    "max_requests_per_minute=None inválido "
                    "(valor nulo explícito); rate limiter desabilitado"
                )
            elif isinstance(raw_rpm, bool):
                # ``bool`` é subclasse de ``int`` em Python; tratamos antes
                # do branch ``int`` para coibir ``True`` como "1 chamada por
                # minuto" — comportamento confuso e quase certamente
                # acidental no JSON.
                max_requests_per_minute = None
                warnings.append(
                    f"max_requests_per_minute={raw_rpm!r} inválido (booleano); "
                    "rate limiter desabilitado"
                )
            elif isinstance(raw_rpm, int):
                if raw_rpm > 0:
                    max_requests_per_minute = raw_rpm
                else:
                    max_requests_per_minute = None
                    warnings.append(
                        f"max_requests_per_minute={raw_rpm!r} inválido "
                        "(não-positivo); rate limiter desabilitado"
                    )
            elif isinstance(raw_rpm, float):
                # Floats são rejeitados mesmo quando positivos: o contrato é
                # de inteiro estrito (R8.2 "inteiro positivo `N`").
                max_requests_per_minute = None
                warnings.append(
                    f"max_requests_per_minute={raw_rpm!r} inválido "
                    "(número de ponto flutuante); rate limiter desabilitado"
                )
            elif isinstance(raw_rpm, str):
                max_requests_per_minute = None
                warnings.append(
                    f"max_requests_per_minute={raw_rpm!r} inválido (string); "
                    "rate limiter desabilitado"
                )
            else:
                # Cobre listas, dicts, objetos arbitrários — qualquer tipo
                # fora do domínio aceito.
                max_requests_per_minute = None
                warnings.append(
                    f"max_requests_per_minute={raw_rpm!r} inválido "
                    f"(tipo {type(raw_rpm).__name__!s}); rate limiter desabilitado"
                )

        # ------------------------------------------------------------------
        # Campo ``request_timeout_seconds``:
        # - chave ausente → default 30.0 (Requirement 4.2).
        # - número (int/float, exceto ``bool``) → coerção para ``float``.
        # - demais tipos ou valores não-positivos → default silencioso. O
        #   contrato (Requirement 8.5) exige warning apenas para
        #   ``max_requests_per_minute``; para o timeout, o silêncio é
        #   aceitável porque ``30s`` é um valor seguro e o operador
        #   percebe o erro pela ausência de efeito.
        # ------------------------------------------------------------------
        raw_timeout = source.get("request_timeout_seconds")
        request_timeout_seconds: float
        if (
            isinstance(raw_timeout, (int, float))
            and not isinstance(raw_timeout, bool)
            and raw_timeout > 0
        ):
            request_timeout_seconds = float(raw_timeout)
        else:
            request_timeout_seconds = _DEFAULT_REQUEST_TIMEOUT_SECONDS

        return cls(
            credentials_path=credentials_path,
            on_failure=on_failure,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
            max_requests_per_minute=max_requests_per_minute,
            request_timeout_seconds=request_timeout_seconds,
            config_warnings=tuple(warnings),
        )
