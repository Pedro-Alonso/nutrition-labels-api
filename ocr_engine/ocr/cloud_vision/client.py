"""Cliente da Google Cloud Vision API com cache, rate limiting e classificação.

``GcvClient`` é a fachada usada por ``CloudVisionPipeline`` para emitir
chamadas à GCV. Centraliza quatro responsabilidades operacionais:

1. **Inicialização lazy do SDK** — ``from google.cloud import vision`` só
   acontece na primeira chamada a :meth:`fetch`. Falhas de import
   (``ImportError``/``ModuleNotFoundError`` ou ``OSError`` de DLL/.so) são
   cacheadas em ``_import_error`` e traduzidas em
   ``GcvError(error="import_error")`` (Requirements 14.3 e 14.4).
2. **Cache em disco** — quando ``config.cache_enabled`` é truthy, a
   resposta da API é gravada via ``GcvCache`` indexada por SHA-256 dos
   bytes PNG enviados (Requirements 7.1, 7.3, 7.5).
3. **Rate limiting** — quando ``config.max_requests_per_minute`` é um
   inteiro positivo, ``RateLimiter`` é consultado APENAS em chamadas
   reais à API (Requirement 8.4 — cache hits não consomem cota).
4. **Timeout e classificação canônica** — chamadas reais são executadas
   dentro de um ``concurrent.futures.ThreadPoolExecutor`` com
   ``request_timeout_seconds`` de teto (Requirement 6.7); qualquer
   exceção é traduzida em ``GcvError`` via :meth:`_classify` aplicando a
   precedência ``auth_error > quota_exceeded > timeout > generic_error``
   (Requirements 6.5–6.8).

Convenções de design seguidas aqui:

- ``api_client`` injetado pelo construtor pula completamente o caminho
  ``_ensure_client``: nenhum import do SDK é feito e nenhuma resolução
  de credencial ocorre. Isso permite que testes usem stubs duck-typed
  com método ``annotate_image`` sem instalar ``google-cloud-vision``
  (Requirement 14.3).
- A classificação por nome de classe (``type(exc).__mro__``) substitui
  o ``isinstance`` direto contra classes do ``google.api_core``: o
  módulo nunca importa o SDK em tempo de carga, preservando a
  compatibilidade com ambientes sem GCV.
- Nada é impresso em ``stdout`` (``AGENTS.md`` reforça: apenas o
  controller imprime). Os erros são propagados via ``GcvError`` e o
  ``CloudVisionPipeline`` decide o que fazer com eles.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.auth import resolve_credentials
from ocr.cloud_vision.cache import GcvCache
from ocr.cloud_vision.rate_limiter import RateLimiter
from ocr.cloud_vision.types import ERROR_PRECEDENCE, GcvError, GcvFetchResult


# ---------------------------------------------------------------------------
# Constantes de classificação
# ---------------------------------------------------------------------------

# Tamanho máximo da mensagem de erro propagada em ``GcvError.message``
# (Requirement 6.2). O ``CloudVisionPipeline`` também aplica o mesmo limite
# defensivamente em ``metadata.error_message``; manter a constante aqui
# evita que mensagens longas inflem ``_summary.json`` quando o pipeline
# escolhe ``raise``.
_ERROR_MESSAGE_MAX_LEN = 500

# Conjuntos de nomes de classes usados pela classificação. Optamos por
# casar pelo NOME da classe (via ``type(exc).__mro__``) em vez de
# ``isinstance`` direto contra ``google.api_core.exceptions`` para
# manter este módulo livre de qualquer dependência de carga sobre o
# SDK ``google-cloud-vision`` (Requirement 14.3 — feature opcional).
# A precedência canônica está em ``ERROR_PRECEDENCE``; aqui só
# definimos quais classes mapeiam para cada código.

# Classes do google-api-core que sinalizam falha de autenticação ou
# autorização (Requirement 6.5). ``Forbidden`` cobre o caso REST
# (``403``) que pode emergir quando o SDK opera por HTTP em vez de
# gRPC.
_AUTH_CLASS_NAMES: frozenset[str] = frozenset(
    {"PermissionDenied", "Unauthenticated", "Forbidden"}
)

# Classes do google-api-core para cota excedida (Requirement 6.6).
# ``ResourceExhausted`` é gRPC; HTTP 429 é tratado separadamente em
# ``_is_http_429`` para cobrir clientes REST e exceções genéricas que
# carregam ``code``/``status_code``.
_QUOTA_CLASS_NAMES: frozenset[str] = frozenset({"ResourceExhausted"})

# Classes do google-api-core para timeout (Requirement 6.7). Em paralelo,
# ``concurrent.futures.TimeoutError`` (timeout local imposto pelo
# ``ThreadPoolExecutor``) e ``socket.timeout`` (timeout de transporte)
# também caem em ``timeout`` por ``isinstance`` em ``_classify``.
_TIMEOUT_CLASS_NAMES: frozenset[str] = frozenset({"DeadlineExceeded"})


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GcvClient:
    """Fachada para a Google Cloud Vision API.

    O cliente é construído pelo :meth:`build` no boot do ``NutritionReader``
    quando há ao menos um GCV preset disponível (Requirement 4.4). Toda a
    inicialização "cara" (resolução de credenciais e import do SDK) é lazy:
    acontece na primeira chamada a :meth:`fetch` via
    :meth:`_ensure_client`.

    Attributes:
        config: Configuração resolvida do bloco ``gcv`` de ``app.json``.
            O cliente consome ``cache_enabled``/``cache_dir`` (para o
            ``GcvCache``), ``max_requests_per_minute`` (para o
            ``RateLimiter``) e ``request_timeout_seconds`` (para o
            timeout local). ``credentials_path`` é repassado a
            :func:`auth.resolve_credentials` na inicialização lazy.
        project_root: Raiz do projeto, usada por
            :func:`auth.resolve_credentials` para ancorar paths
            relativos de ``credentials_path`` (Requirement 5.1).
        cache: ``GcvCache`` quando ``config.cache_enabled`` é truthy;
            ``None`` quando o operador desabilitou o cache
            (Requirement 7.5). Inicializado por :meth:`build` para que
            o pipeline tenha visibilidade sobre o estado configurado.
        rate_limiter: ``RateLimiter`` quando
            ``config.max_requests_per_minute`` é um inteiro positivo;
            ``None`` quando o limite está desabilitado
            (Requirement 8.1). Inicializado por :meth:`build`.
        _api_client: Instância do SDK (``vision.ImageAnnotatorClient``)
            ou stub injetado em testes. Quando não é ``None`` na
            construção, :meth:`_ensure_client` curto-circuita —
            nenhuma credencial é resolvida e nenhum import é feito,
            permitindo testes sem o SDK instalado (Requirement 14.3).
        _import_error: Exceção cacheada quando o import lazy do SDK
            falhou. Tentativas subsequentes de :meth:`fetch` re-lançam
            ``GcvError(error="import_error")`` sem repetir o import
            (Requirement 14.4 — caminho determinístico após a primeira
            falha).
    """

    config: GcvAppConfig
    project_root: Path
    cache: GcvCache | None = None
    rate_limiter: RateLimiter | None = None
    _api_client: Any | None = None
    _import_error: Exception | None = None

    # ------------------------------------------------------------------
    # Construtor canônico
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        config: GcvAppConfig,
        project_root: Path,
        api_client: Any | None = None,
    ) -> "GcvClient":
        """Constrói um ``GcvClient`` a partir da configuração resolvida.

        Inicializa eagerly o ``GcvCache`` e o ``RateLimiter`` conforme a
        configuração permitir, mas NÃO toca em ``google.cloud.vision``
        nem em credenciais — esse trabalho fica para :meth:`_ensure_client`
        (lazy). Essa separação é intencional: o ``NutritionReader`` chama
        ``build`` no boot mesmo sem ter certeza de que algum GCV preset
        será exercitado, e queremos zero custo nesse caminho silencioso
        (Requirement 4.4).

        Args:
            config: Configuração já parseada de ``app.json::gcv`` via
                :meth:`GcvAppConfig.from_dict`.
            project_root: Raiz do projeto, repassada ao
                :func:`auth.resolve_credentials` em
                :meth:`_ensure_client`.
            api_client: Stub opcional injetado por testes. Quando
                fornecido, :meth:`_ensure_client` o devolve diretamente,
                pulando a resolução de credenciais e o import do SDK.

        Returns:
            Instância pronta para receber chamadas a :meth:`fetch`.
        """

        # ``GcvCache`` só é instanciado quando o cache está habilitado —
        # do contrário, ``self.cache is None`` sinaliza ao :meth:`fetch`
        # que toda I/O de cache deve ser suprimida (Requirement 7.5).
        cache = (
            GcvCache(cache_dir=config.cache_dir) if config.cache_enabled else None
        )
        # ``RateLimiter`` só existe quando ``max_requests_per_minute`` é
        # um inteiro positivo. ``GcvAppConfig.from_dict`` já coage valores
        # inválidos para ``None`` (Requirement 8.5), então qualquer valor
        # truthy aqui é seguro.
        rate_limiter = (
            RateLimiter(max_per_minute=config.max_requests_per_minute)
            if config.max_requests_per_minute
            else None
        )
        return cls(
            config=config,
            project_root=project_root,
            cache=cache,
            rate_limiter=rate_limiter,
            _api_client=api_client,
            _import_error=None,
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fetch(
        self,
        png_bytes: bytes,
        feature: str,
        language_hints: list[str] | tuple[str, ...],
    ) -> GcvFetchResult:
        """Consulta a Cloud Vision API (ou o cache) e devolve a resposta canônica.

        O fluxo segue o flowchart do design (seção *Cache: hit vs miss*):

        1. Se ``cache`` está habilitado, calcula SHA-256 dos ``png_bytes``
           e consulta ``GcvCache.get(sha, feature, hints)``. Em hit,
           devolve ``GcvFetchResult(cache_hit=True, ...)`` SEM consumir
           cota do rate limiter (Requirement 8.4).
        2. Em miss, chama :meth:`_ensure_client` (lazy init na primeira
           chamada). Falhas aqui sobem como ``GcvError`` já classificado
           — credenciais ausentes geram ``auth_error`` (Requirement 5.3)
           e import quebrado gera ``import_error`` (Requirement 14.4).
        3. Adquire vaga no ``rate_limiter`` (se houver) — apenas chamadas
           reais contam.
        4. Submete ``client.annotate_image(request)`` a um
           ``ThreadPoolExecutor`` com timeout
           ``config.request_timeout_seconds``. Qualquer exceção é
           classificada via :meth:`_classify` aplicando a precedência
           canônica.
        5. Em sucesso, grava a resposta em ``GcvCache.put`` (se cache
           habilitado) e devolve ``GcvFetchResult(cache_hit=False, ...)``.

        Args:
            png_bytes: Bytes PNG já codificados pelo pipeline. O SHA-256
                desses bytes é a chave determinística do cache
                (Requirement 7.1).
            feature: Modalidade GCV (``"TEXT_DETECTION"`` ou
                ``"DOCUMENT_TEXT_DETECTION"``). Validade já é garantida
                pelo pipeline via ``GcvPresetOptions.invalid_feature``;
                este método assume valor canônico.
            language_hints: Hints BCP-47 a enviar à API. Aceita ``list``
                ou ``tuple`` por flexibilidade dos callers; é convertido
                para ``tuple`` antes de qualquer uso para preservar
                imutabilidade no resultado canônico.

        Returns:
            ``GcvFetchResult`` com a resposta crua (``response_json``),
            ``cache_hit`` indicando a origem da resposta, e os
            ``feature``/``language_hints`` efetivamente usados.

        Raises:
            GcvError: Qualquer falha classificada (auth, quota, timeout,
                generic, import). O ``CloudVisionPipeline`` traduz isso
                em ``metadata.error`` segundo a política ``on_failure``.
        """

        # Normalização imutável dos hints. ``tuple`` é o formato canônico
        # propagado em ``GcvFetchResult.language_hints`` e usado como
        # parte da chave de filtro do cache (Requirement 7.3).
        hints = tuple(language_hints)

        # ---------------------------------------------------------------
        # 1) Cache lookup. ``sha`` é cacheado localmente para evitar
        # recomputar o hash em ``cache.put``. Em ``cache=None`` (modo
        # ``cache_enabled=false``) o lookup e a gravação são pulados
        # silenciosamente (Requirement 7.5).
        # ---------------------------------------------------------------
        sha: str | None = None
        if self.cache is not None:
            sha = hashlib.sha256(png_bytes).hexdigest()
            cached = self.cache.get(sha, feature, hints)
            if cached is not None:
                return GcvFetchResult(
                    response_json=cached,
                    cache_hit=True,
                    feature=feature,
                    language_hints=hints,
                )

        # ---------------------------------------------------------------
        # 2) Lazy init do SDK / resolução de credenciais. Falhas aqui
        # já vêm classificadas como ``GcvError`` (auth_error ou
        # import_error) — propagamos como estão, porque
        # ``CloudVisionPipeline`` espera o código canônico em
        # ``GcvError.error``.
        # ---------------------------------------------------------------
        try:
            client = self._ensure_client()
        except GcvError:
            # Já classificado em ``_ensure_client`` — não passamos pelo
            # ``_classify`` para preservar o código original e evitar
            # double-wrap.
            raise
        except Exception as exc:  # noqa: BLE001 — qualquer outra exceção do init
            # Qualquer exceção não-classificada que escape de
            # ``_ensure_client`` (ex.: erro durante construção do
            # ``ImageAnnotatorClient``) cai na classificação canônica.
            raise self._classify(exc) from exc

        # ---------------------------------------------------------------
        # 3) Rate limiter — apenas chamadas reais (Requirement 8.4). O
        # ``acquire`` bloqueia o thread atual sem polling ativo até
        # liberar vaga (ver ``rate_limiter.py`` para detalhes do
        # algoritmo de janela deslizante).
        # ---------------------------------------------------------------
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()

        # ---------------------------------------------------------------
        # 4) Chamada à API com timeout. Qualquer exceção (timeout local,
        # erro do SDK, erro de transporte) é traduzida em ``GcvError``
        # pela classificação canônica.
        # ---------------------------------------------------------------
        try:
            response_json = self._call_api_with_timeout(
                client, png_bytes, feature, list(hints)
            )
        except GcvError:
            # Defensivo: se ``_call_api`` algum dia levantar um
            # ``GcvError`` (atualmente nunca), preservamos o código
            # original sem reclassificar.
            raise
        except Exception as exc:  # noqa: BLE001 — captura ampla intencional
            raise self._classify(exc) from exc

        # ---------------------------------------------------------------
        # 5) Cache put em sucesso. ``sha`` pode ainda estar ``None``
        # quando o cache está desabilitado; recomputamos preguiçosamente
        # apenas no caminho ``cache habilitado + miss``.
        # ---------------------------------------------------------------
        if self.cache is not None:
            if sha is None:  # pragma: no cover - branch defensivo
                # Cobre cenário hipotético em que ``self.cache`` foi
                # alterado entre o lookup e o put; mantém o invariante
                # "se ``cache`` é truthy aqui, gravamos sempre".
                sha = hashlib.sha256(png_bytes).hexdigest()
            self.cache.put(sha, feature, hints, response_json, len(png_bytes))

        return GcvFetchResult(
            response_json=response_json,
            cache_hit=False,
            feature=feature,
            language_hints=hints,
        )

    # ------------------------------------------------------------------
    # Inicialização lazy
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        """Garante que ``self._api_client`` está pronto para uso.

        Comportamento:

        - ``self._api_client`` já set (por injeção de teste ou por
          inicialização anterior bem-sucedida) → retorna imediatamente.
        - ``self._import_error`` cacheado de uma tentativa anterior →
          re-levanta ``GcvError(error="import_error")`` sem repetir o
          import (caminho determinístico — uma vez que ``vision`` falha
          ao carregar, falha sempre dentro do mesmo processo).
        - Caso contrário: resolve credenciais via
          :func:`auth.resolve_credentials` (que pode levantar
          ``GcvError(auth_error)``) e tenta importar
          ``google.cloud.vision``. Falhas de import (``ImportError``,
          ``ModuleNotFoundError``, ``OSError`` de DLL/.so) são cacheadas
          em ``self._import_error`` e traduzidas em
          ``GcvError(import_error)``. Em sucesso, instancia
          ``vision.ImageAnnotatorClient(credentials=...)`` e armazena em
          ``self._api_client``.

        Returns:
            O ``ImageAnnotatorClient`` (ou stub injetado) pronto para
            receber ``annotate_image(request)``.

        Raises:
            GcvError: ``error="auth_error"`` quando ``resolve_credentials``
                falha; ``error="import_error"`` quando o SDK não pode
                ser carregado.
        """

        # Caminho 1: cliente já disponível. Cobre tanto a injeção em
        # testes (``api_client`` passado em ``build``) quanto chamadas
        # subsequentes a ``fetch`` após a primeira inicialização.
        if self._api_client is not None:
            return self._api_client

        # Caminho 2: import já falhou anteriormente. Re-levantamos o
        # mesmo código sem repetir o import — é determinístico que ele
        # falhará de novo dentro do mesmo processo.
        if self._import_error is not None:
            raise GcvError(
                error="import_error",
                message=_truncate(str(self._import_error)),
            )

        # Caminho 3: inicialização real. Primeiro resolvemos credenciais
        # (Requirement 5.1–5.3) — uma falha aqui sai como
        # ``GcvError(auth_error)`` e propaga sem cache.
        creds_path = resolve_credentials(self.config, self.project_root)

        # Import lazy do SDK. Cobrimos as três classes de falha
        # documentadas em Requirement 14.4: ``ImportError`` (módulo
        # não instalado), ``ModuleNotFoundError`` (subclasse de
        # ``ImportError`` mas listada explicitamente para clareza) e
        # ``OSError`` (carregamento de DLL/.so subjacente — comum em
        # sistemas com ``grpc`` quebrado ou bibliotecas C ausentes).
        try:
            from google.cloud import vision  # type: ignore[import-not-found]
            from google.oauth2 import service_account  # type: ignore[import-not-found]
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            # Cacheia para que tentativas subsequentes não repitam o
            # import e retornem o mesmo erro determinístico.
            self._import_error = exc
            raise GcvError(
                error="import_error",
                message=_truncate(str(exc)),
            ) from exc

        # Construção do client. Falhas aqui são raras (credenciais
        # malformadas, problema de inicialização do gRPC channel) e
        # NÃO são cacheadas em ``_import_error`` — o operador pode
        # corrigir o Service Account e tentar de novo sem reiniciar o
        # processo. ``_classify`` decide o código canônico.
        try:
            credentials = service_account.Credentials.from_service_account_file(
                str(creds_path)
            )
            client = vision.ImageAnnotatorClient(credentials=credentials)
        except Exception as exc:  # noqa: BLE001 — captura ampla intencional
            raise self._classify(exc) from exc

        self._api_client = client
        return client

    # ------------------------------------------------------------------
    # Chamada com timeout
    # ------------------------------------------------------------------

    def _call_api_with_timeout(
        self,
        client: Any,
        png_bytes: bytes,
        feature: str,
        language_hints: list[str],
    ) -> dict:
        """Executa ``client.annotate_image`` com teto de tempo via thread pool.

        Usamos ``concurrent.futures.ThreadPoolExecutor`` em vez de
        ``signal.alarm`` para que o timeout funcione corretamente em
        threads não-principais (e no Windows, onde ``alarm`` não está
        disponível). O executor é descartado com ``cancel_futures=True``
        (Python 3.9+) e ``wait=False`` para não bloquear no shutdown
        quando o timeout dispara — a thread em background pode continuar
        rodando até completar ou ser interrompida pelo SO no fim do
        processo, o que é aceitável para um CLI single-shot.

        Args:
            client: ``ImageAnnotatorClient`` (ou stub) já inicializado.
            png_bytes: Bytes PNG da imagem.
            feature: Modalidade GCV.
            language_hints: Lista de hints BCP-47 já normalizada para
                ``list[str]`` (a API do SDK aceita ``list`` direto).

        Returns:
            ``dict`` com a resposta crua (formato de
            ``MessageToDict(AnnotateImageResponse)``).

        Raises:
            concurrent.futures.TimeoutError: Quando ``request_timeout_seconds``
                é excedido. ``_classify`` mapeia para ``timeout``.
            Exception: Qualquer exceção levantada por
                ``client.annotate_image`` é re-lançada por
                ``future.result`` e classificada em :meth:`fetch`.
        """

        timeout = self.config.request_timeout_seconds
        # ``thread_name_prefix`` facilita a identificação em tracebacks
        # quando uma chamada longa fica pendente após o timeout.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gcv-fetch"
        )
        try:
            future = executor.submit(
                self._call_api, client, png_bytes, feature, language_hints
            )
            # ``future.result(timeout=...)`` re-levanta a exceção
            # original quando ``_call_api`` falha, ou levanta
            # ``concurrent.futures.TimeoutError`` (alias de
            # ``TimeoutError`` builtin no Python 3.11+) quando o teto
            # de tempo é excedido.
            return future.result(timeout=timeout)
        finally:
            # ``cancel_futures=True`` (Python 3.9+) cancela tarefas que
            # ainda não começaram; ``wait=False`` evita bloquear o
            # shutdown quando o timeout disparou e a thread está
            # genuinamente atrasada na chamada à API.
            executor.shutdown(wait=False, cancel_futures=True)

    def _call_api(
        self,
        client: Any,
        png_bytes: bytes,
        feature: str,
        language_hints: list[str],
    ) -> dict:
        """Invoca ``client.annotate_image`` e normaliza a resposta para ``dict``.

        Aceita tanto o ``ImageAnnotatorClient`` real quanto stubs de
        teste. O request é construído como ``dict`` (formato JSON do
        protobuf) — o SDK aceita esse formato e converte internamente
        para ``AnnotateImageRequest``; stubs de teste o consomem via
        duck typing sem precisar do SDK instalado.

        A resposta pode vir em três formatos:

        - ``dict`` (stubs de teste) → repassado como está.
        - Objeto SDK com atributo ``_pb`` (``AnnotateImageResponse``
          real) → convertido via ``MessageToDict`` para alinhar com o
          contrato esperado por ``parser.parse_response``.
        - Outro tipo → conversão best-effort via ``dict(response)``;
          falhas caem em ``{}`` para não derrubar o pipeline (cenário
          defensivo, raro na prática).
        """

        request = self._build_request(png_bytes, feature, language_hints)
        response = client.annotate_image(request)
        return self._normalize_response(response)

    @staticmethod
    def _build_request(
        png_bytes: bytes,
        feature: str,
        language_hints: list[str],
    ) -> dict:
        """Monta o request da API como ``dict`` JSON-protobuf compatível.

        Formato esperado pelo SDK (e por stubs duck-typed):

        .. code-block:: python

            {
                "image": {"content": <PNG bytes>},
                "features": [{"type_": "DOCUMENT_TEXT_DETECTION"}],
                "image_context": {"language_hints": ["pt"]},
            }

        Usamos ``type_`` (com underscore final) porque ``type`` é um
        builtin em Python e o SDK ``google-cloud-vision`` mapeia o
        campo protobuf ``type`` para o atributo ``type_`` em sua API
        Python. Aceitamos ``language_hints`` mesmo quando a lista está
        vazia — o SDK trata isso como auto-detecção.
        """

        return {
            "image": {"content": png_bytes},
            "features": [{"type_": feature}],
            "image_context": {"language_hints": list(language_hints)},
        }

    @staticmethod
    def _normalize_response(response: Any) -> dict:
        """Converte qualquer formato de resposta para ``dict`` JSON.

        - Quando ``response`` já é ``dict`` (stubs), repassa.
        - Quando tem atributo ``_pb`` (resposta real do SDK), converte
          via ``MessageToDict`` preservando ``camelCase`` (default do
          ``protobuf``) — o ``parser.parse_response`` aceita ambos os
          estilos via ``_get(d, *keys)``.
        - Caso contrário, tenta ``dict(response)`` como último recurso.
        """

        if isinstance(response, dict):
            return response
        if hasattr(response, "_pb"):
            try:
                from google.protobuf.json_format import (  # type: ignore[import-not-found]
                    MessageToDict,
                )
            except (ImportError, ModuleNotFoundError, OSError):  # pragma: no cover
                # Sem protobuf disponível, devolvemos dict vazio — o
                # parser ainda funcionará mas extrairá ``text=""`` e
                # ``mean_confidence=0``. Cenário extremamente raro
                # (protobuf é dependência transitiva de
                # ``google-cloud-vision``).
                return {}
            return MessageToDict(response._pb, preserving_proto_field_name=False)
        try:  # pragma: no cover - fallback defensivo
            return dict(response)
        except (TypeError, ValueError):
            return {}

    # ------------------------------------------------------------------
    # Classificação canônica de exceções
    # ------------------------------------------------------------------

    def _classify(self, exc: BaseException) -> GcvError:
        """Traduz uma exceção em ``GcvError`` aplicando precedência canônica.

        Mapeamento (ver design.md, seção *Resumo das classificações de
        erro*):

        - ``PermissionDenied`` / ``Unauthenticated`` / ``Forbidden`` →
          ``auth_error`` (Requirement 6.5).
        - ``ResourceExhausted`` ou HTTP 429 → ``quota_exceeded``
          (Requirement 6.6).
        - ``DeadlineExceeded`` / ``concurrent.futures.TimeoutError`` /
          ``socket.timeout`` → ``timeout`` (Requirement 6.7).
        - Qualquer outra exceção → ``generic_error``.

        Quando uma única exceção carrega múltiplas classificações
        (cenário raro mas possível em tracebacks compostos), aplicamos
        a precedência fixa ``ERROR_PRECEDENCE`` para escolher a
        primária; as demais vão para ``GcvError.secondary`` ordenadas
        pela mesma precedência (Requirement 6.8).

        ``GcvError`` já classificado é devolvido como está — evita o
        double-wrap quando ``_ensure_client`` levanta um ``GcvError``
        e o caller do ``_classify`` o passa adiante por engano.

        Args:
            exc: Exceção a classificar. ``BaseException`` em vez de
                ``Exception`` para tolerar ``KeyboardInterrupt`` e
                ``SystemExit`` quando vierem de uma thread em background
                (cenário raro mas possível com ``ThreadPoolExecutor``).

        Returns:
            ``GcvError`` com ``error`` na precedência canônica,
            ``message`` truncada em 500 caracteres e ``secondary`` com
            as classes não-vencedoras em ordem de precedência.
        """

        if isinstance(exc, GcvError):
            # Já classificado por outra camada — preservamos código
            # primário e secundários sem reaplicar a precedência (que
            # seria idempotente, mas evita custo redundante).
            return exc

        classes: set[str] = set()

        # Inspeção pelo NOME da classe ao longo da MRO. Isso cobre
        # subclasses do google-api-core (ex.: classes geradas por gRPC
        # que herdam de ``PermissionDenied``) sem importar o SDK.
        cls_names = {c.__name__ for c in type(exc).__mro__}

        if cls_names & _AUTH_CLASS_NAMES:
            classes.add("auth_error")

        if cls_names & _QUOTA_CLASS_NAMES or _is_http_429(exc):
            classes.add("quota_exceeded")

        # Timeout é detectado tanto por nome (``DeadlineExceeded``)
        # quanto por isinstance contra os tipos builtin/socket — esses
        # últimos não dependem do SDK e cobrem os timeouts locais
        # impostos pelo ``ThreadPoolExecutor`` e o transporte HTTP.
        if (
            cls_names & _TIMEOUT_CLASS_NAMES
            or isinstance(exc, concurrent.futures.TimeoutError)
            or isinstance(exc, socket.timeout)
        ):
            classes.add("timeout")

        # Quando nada bate, ``generic_error`` é o fallback
        # (Requirement 6.5–6.8 — implícito pela precedência terminar
        # em ``generic_error``).
        if not classes:
            classes.add("generic_error")

        # Aplica precedência canônica. ``min`` com ``ERROR_PRECEDENCE.index``
        # como key escolhe a classe de menor índice (mais "estrutural"),
        # garantindo que ``auth_error`` domine ``quota_exceeded`` que
        # domina ``timeout`` que domina ``generic_error``.
        primary = min(classes, key=ERROR_PRECEDENCE.index)
        secondary = tuple(
            sorted(classes - {primary}, key=ERROR_PRECEDENCE.index)
        )

        return GcvError(
            error=primary,
            message=_truncate(str(exc)),
            secondary=secondary,
        )


# ---------------------------------------------------------------------------
# Helpers a nível de módulo
# ---------------------------------------------------------------------------


def _is_http_429(exc: BaseException) -> bool:
    """Detecta exceções com código HTTP 429 (rate limit/quota).

    Cobre clientes REST e exceções genéricas que carregam o status code
    em ``code`` (convenção do ``google-api-core``) ou ``status_code``
    (convenção do ``requests``/``urllib3``). A detecção complementa o
    casamento por nome de classe — ``ResourceExhausted`` é a forma gRPC
    e cobre o caminho oficial do SDK; este helper protege contra
    exceções menos canônicas que ainda assim representam quota
    excedida.
    """

    code = getattr(exc, "code", None)
    if code == 429:
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    return False


def _truncate(message: str | None) -> str:
    """Trunca a mensagem em ``_ERROR_MESSAGE_MAX_LEN`` (500 chars).

    Defensivo contra ``None`` (algumas exceções têm ``str(exc) == ""``
    sem ``args``, o que pode resultar em string vazia mas nunca em
    ``None``; mantemos a guarda por consistência com o pipeline).
    Não adiciona sufixo ``"..."`` — o contrato do design exige
    truncamento puro para que o operador veja exatamente o início da
    mensagem original sem decoração.
    """

    if message is None:
        return ""
    if len(message) > _ERROR_MESSAGE_MAX_LEN:
        return message[:_ERROR_MESSAGE_MAX_LEN]
    return message
