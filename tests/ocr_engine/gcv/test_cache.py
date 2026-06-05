"""Property test P5: round-trip do cache com filtragem por ``(feature, hints)``.

Validates: Requirements 7.1, 7.3, 7.4

A propriedade afirma três invariantes complementares sobre ``GcvCache``:

1. **Round-trip determinístico** — para qualquer ``(sha256, feature,
   language_hints, response_json)``, ``GcvCache.put(...)`` seguido de
   ``GcvCache.get(sha256, feature, language_hints)`` devolve um
   ``dict`` igual a ``response_json`` (Requirement 7.1: o SHA-256 é a
   chave; Requirement 7.4: ``cache_hit`` é detectável pela presença de
   resposta retornada).
2. **Filtro por ``feature``** — quando o caller pede o mesmo
   ``sha256`` mas com uma ``feature`` diferente da gravada, ``get``
   devolve ``None`` sem servir a resposta (Requirement 7.3). Isso
   evita servir uma resposta de ``TEXT_DETECTION`` quando o preset
   pediu ``DOCUMENT_TEXT_DETECTION`` (e vice-versa).
3. **Filtro por ``language_hints``** — quando o caller pede o mesmo
   ``sha256`` e a mesma ``feature`` mas com ``language_hints`` diferentes
   em conteúdo OU em ordem, ``get`` devolve ``None`` (Requirement 7.3).
   A ordem importa porque a GCV trata ``language_hints`` como lista de
   prioridade.

Os geradores de ``tests/gcv/strategies.py`` (``gcv_response_dict()``,
``bcp47_hints()``) alimentam um espaço amplo de respostas sintéticas e
combinações de hints; o ``sha256`` é gerado como string hexadecimal de
64 caracteres para refletir o formato real da chave (saída de
``hashlib.sha256``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ocr.cloud_vision.cache import GcvCache
from ocr.cloud_vision.types import ALLOWED_FEATURES
from tests.ocr_engine.gcv.strategies import bcp47_hints, cache_states, gcv_response_dict


# ---------------------------------------------------------------------------
# Estratégias auxiliares
# ---------------------------------------------------------------------------

# SHA-256 hexadecimal de 64 caracteres minúsculos. Reproduz o formato
# canônico do ``hashlib.sha256(...).hexdigest()`` usado pelo
# ``CloudVisionPipeline`` ao calcular a chave do cache (Requirement 7.1).
_SHA256_HEX = st.from_regex(r"[0-9a-f]{64}", fullmatch=True)

# Estratégia sobre as duas modalidades aceitas pelo bloco ``gcv.feature``.
# ``ALLOWED_FEATURES`` é uma tupla ordenada — perfeita para
# ``sampled_from`` reproduzir shrinkings determinísticos.
_FEATURE = st.sampled_from(ALLOWED_FEATURES)

# Tamanho de imagem em bytes — registrado em ``.meta.json`` mas
# semanticamente irrelevante para o filtro do ``get``. Mantemos os
# valores positivos pequenos para não inflar o disco temporário do
# Hypothesis em iterações repetidas.
_IMAGE_SIZE = st.integers(min_value=1, max_value=2**20)


def _fresh_cache(tmp_path_factory: pytest.TempPathFactory, label: str) -> GcvCache:
    """Constrói um ``GcvCache`` apontado para um diretório temporário único.

    Cada exemplo do Hypothesis recebe um diretório próprio para evitar
    colisão de chaves entre iterações sucessivas. ``tmp_path_factory`` é
    session-scoped, então não há problema em invocá-lo dentro de uma
    função decorada com ``@given``.
    """

    cache_dir = tmp_path_factory.mktemp(label)
    return GcvCache(cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Property 5 — round-trip determinístico
# ---------------------------------------------------------------------------


@given(
    sha256=_SHA256_HEX,
    feature=_FEATURE,
    hints=bcp47_hints(),
    response=gcv_response_dict(),
    image_size=_IMAGE_SIZE,
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_p5_roundtrip_returns_same_response_json(
    tmp_path_factory: pytest.TempPathFactory,
    sha256: str,
    feature: str,
    hints: tuple[str, ...],
    response: dict,
    image_size: int,
) -> None:
    """**Property 5 (round-trip)**: ``put`` → ``get`` recupera o ``response_json``.

    **Validates: Requirements 7.1, 7.4**

    Para qualquer combinação válida de ``(sha256, feature, hints,
    response)``, ``GcvCache.get`` devolve um ``dict`` cuja igualdade
    estrutural (``==``) bate com o ``response_json`` original. A
    comparação após ``json.dumps`` + ``json.loads`` é exata para os
    tipos JSON-nativos produzidos por ``gcv_response_dict()`` (str,
    int, float finito em ``[0, 1]``, list, dict).
    """

    cache = _fresh_cache(tmp_path_factory, "p5_roundtrip")

    json_path = cache.put(
        sha256=sha256,
        feature=feature,
        language_hints=hints,
        response_json=response,
        image_size_bytes=image_size,
    )

    # Sanity: ``put`` retorna o caminho do ``.json`` recém-gravado e o
    # arquivo correspondente existe em disco. Isso falsifica trivialmente
    # qualquer regressão que retorne ``None`` ou um path bogus.
    assert isinstance(json_path, Path)
    assert json_path.is_file()
    assert json_path.name == f"{sha256}.json"

    cached = cache.get(sha256, feature, hints)

    # ``cache_hit`` (Requirement 7.4) é representado pelo retorno
    # não-``None`` de ``GcvCache.get``: o ``CloudVisionPipeline``
    # converte essa presença em ``metadata.cache_hit = True`` no nível
    # superior. Aqui validamos a base do contrato: get não-vazio ⇒ a
    # resposta exata gravada por ``put`` é recuperada.
    assert cached is not None, "cache hit esperado após put imediato"
    assert cached == response


# ---------------------------------------------------------------------------
# Property 5 — filtro por feature
# ---------------------------------------------------------------------------


@given(
    sha256=_SHA256_HEX,
    hints=bcp47_hints(),
    response=gcv_response_dict(),
    image_size=_IMAGE_SIZE,
    feature_pair=st.permutations(list(ALLOWED_FEATURES)),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_p5_get_mismatched_feature_returns_none(
    tmp_path_factory: pytest.TempPathFactory,
    sha256: str,
    hints: tuple[str, ...],
    response: dict,
    image_size: int,
    feature_pair: list[str],
) -> None:
    """**Property 5 (mismatched feature)**: feature divergente ⇒ ``None``.

    **Validates: Requirements 7.3**

    Grava a entrada com ``feature_pair[0]`` e consulta com
    ``feature_pair[1]``. Como ``ALLOWED_FEATURES`` é estritamente
    ``("TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION")``, a permutação
    garante divergência e força o filtro de compatibilidade do cache a
    rejeitar a entrada.

    O sanity check com a feature original mantida confirma que a
    entrada *está* no cache — ou seja, o ``None`` no caso divergente é
    causado pelo filtro, não por uma falha de gravação acidental.
    """

    stored_feature, queried_feature = feature_pair
    # ``permutations`` em uma sequência de 2 elementos sempre devolve um
    # par distinto; reforçamos a intenção com um assert explícito.
    assert stored_feature != queried_feature

    cache = _fresh_cache(tmp_path_factory, "p5_feature_filter")
    cache.put(
        sha256=sha256,
        feature=stored_feature,
        language_hints=hints,
        response_json=response,
        image_size_bytes=image_size,
    )

    # Sanity: a feature gravada continua recuperável.
    assert cache.get(sha256, stored_feature, hints) == response

    # Invariante: feature divergente ⇒ ``None``.
    assert cache.get(sha256, queried_feature, hints) is None


# ---------------------------------------------------------------------------
# Property 5 — filtro por language_hints (conteúdo ou ordem divergente)
# ---------------------------------------------------------------------------


@given(
    sha256=_SHA256_HEX,
    feature=_FEATURE,
    hints_pair=st.tuples(bcp47_hints(), bcp47_hints()).filter(
        lambda pair: pair[0] != pair[1]
    ),
    response=gcv_response_dict(),
    image_size=_IMAGE_SIZE,
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_p5_get_mismatched_hints_returns_none(
    tmp_path_factory: pytest.TempPathFactory,
    sha256: str,
    feature: str,
    hints_pair: tuple[tuple[str, ...], tuple[str, ...]],
    response: dict,
    image_size: int,
) -> None:
    """**Property 5 (mismatched hints)**: hints divergentes ⇒ ``None``.

    **Validates: Requirements 7.3**

    Cobre tanto divergência de **conteúdo** (ex.: ``("pt",)`` vs
    ``("en",)``) quanto de **ordem** (ex.: ``("pt", "en")`` vs
    ``("en", "pt")``). O ``filter`` garante que o par é estritamente
    distinto; quando o filtro descarta um exemplo coincidente, o
    Hypothesis sorteia outro.

    O sanity check com os hints originais confirma que a entrada *está*
    no cache — provando que o ``None`` no caso divergente é fruto do
    filtro de compatibilidade ordem-sensível.
    """

    stored_hints, queried_hints = hints_pair
    assert stored_hints != queried_hints

    cache = _fresh_cache(tmp_path_factory, "p5_hints_filter")
    cache.put(
        sha256=sha256,
        feature=feature,
        language_hints=stored_hints,
        response_json=response,
        image_size_bytes=image_size,
    )

    # Sanity: hints idênticos continuam recuperáveis.
    assert cache.get(sha256, feature, stored_hints) == response

    # Invariante: hints divergentes (conteúdo ou ordem) ⇒ ``None``.
    assert cache.get(sha256, feature, queried_hints) is None


# ---------------------------------------------------------------------------
# Property 5 — caso explícito: ordem importa para hints não-palindromos
# ---------------------------------------------------------------------------


@given(
    sha256=_SHA256_HEX,
    feature=_FEATURE,
    hints=bcp47_hints().filter(
        # Apenas hints não-palindromos: para tuplas de tamanho 0 ou 1, e
        # para tuplas simétricas como ``("pt", "pt")``, a inversão é
        # idêntica e não exerceria a invariante de ordem.
        lambda h: tuple(reversed(h)) != h
    ),
    response=gcv_response_dict(),
    image_size=_IMAGE_SIZE,
)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_p5_reversed_hints_returns_none(
    tmp_path_factory: pytest.TempPathFactory,
    sha256: str,
    feature: str,
    hints: tuple[str, ...],
    response: dict,
    image_size: int,
) -> None:
    """**Property 5 (ordem de hints)**: inversão de hints ⇒ ``None``.

    **Validates: Requirements 7.3**

    Reforça explicitamente a invariante de **ordem-sensibilidade** do
    filtro de hints. Para qualquer tupla de hints cuja inversão difere
    do original (ex.: ``("pt", "en")`` ⇒ ``("en", "pt")``), gravar com
    ``hints`` e consultar com ``reversed(hints)`` deve devolver
    ``None``. Este caso é uma especialização do teste anterior
    garantida por construção — útil porque shrinkers do Hypothesis
    podem reduzir o teste mais geral a um par com conteúdo igual mas
    ordem trocada, e queremos um sinal direto de que essa redução
    sempre resulta em ``None``.
    """

    reversed_hints = tuple(reversed(hints))
    assert hints != reversed_hints

    cache = _fresh_cache(tmp_path_factory, "p5_reversed_hints")
    cache.put(
        sha256=sha256,
        feature=feature,
        language_hints=hints,
        response_json=response,
        image_size_bytes=image_size,
    )

    assert cache.get(sha256, feature, hints) == response
    assert cache.get(sha256, feature, reversed_hints) is None


# ===========================================================================
# Property 18 — Corrupção de uma entrada não invalida outras
# ===========================================================================
#
# **Validates: Requirements 7.7**
#
# Cenário: dado um cache populado com várias entradas válidas via
# ``put``, escolhe-se uma entrada-alvo e corrompe-se um de seus dois
# arquivos em disco (``<sha>.json`` ou ``<sha>.meta.json``). A
# invariante exige:
#
# 1. ``cache.get(target_sha, target_feature, target_hints)`` devolve
#    ``None`` — a entrada corrompida é descartada silenciosamente
#    (Requirement 7.7).
# 2. Para cada entrada vizinha não-tocada, ``cache.get(sha, feature,
#    hints)`` continua devolvendo o ``response_json`` original — a
#    corrupção é estritamente local àquela chave.
# 3. Os arquivos das entradas vizinhas permanecem **fisicamente
#    presentes** no diretório — o cache nunca remove vizinhos como
#    efeito colateral da leitura de uma entrada corrompida.
#
# A estratégia ``cache_states()`` de ``tests/gcv/strategies.py`` gera a
# distribuição de SHAs/feature/hints; aqui filtramos para tamanho ≥ 2
# (a invariante "não invalida outras" só é significativa quando
# existe pelo menos uma "outra" entrada). Os flags ``response_corrupt``
# / ``meta_corrupt`` da estratégia são **ignorados** neste teste —
# materializamos todas as entradas como saudáveis via ``put`` e
# controlamos explicitamente qual arquivo corromper, para que o teste
# tenha um único eixo de variação semântico.


# Tipo de corrupção a aplicar: substituir o conteúdo do arquivo por
# bytes não-parseáveis como JSON. Cobrimos os dois lados do par
# (``.json`` e ``.meta.json``) porque a implementação de
# ``GcvCache.get`` lê primeiro a meta e depois a resposta — ambos os
# caminhos de erro devem resultar no mesmo comportamento (return
# ``None`` sem efeito colateral).
_CORRUPTION_KIND = st.sampled_from(("json", "meta"))


# Conteúdo "corrompido" a gravar no arquivo escolhido. Misturamos:
# - strings que claramente não são JSON (texto livre, bytes binários
#   embutidos como string);
# - JSON quase-válido truncado;
# - arquivos vazios.
# Todos devem provocar ``json.JSONDecodeError`` (subclasse de
# ``ValueError``) na hora da leitura, que ``GcvCache.get`` absorve.
_CORRUPT_PAYLOADS = st.sampled_from(
    (
        "",                          # arquivo vazio
        "not json at all",           # texto livre
        "{",                         # JSON truncado (objeto aberto)
        "[1, 2, 3",                  # JSON truncado (array aberto)
        "\x00\x01\x02\xff\xfe",      # bytes binários como string
        "{\"feature\": ",            # par chave: incompleto
        "}{",                        # delimitadores invertidos
    )
)


@given(
    states=cache_states().filter(lambda entries: len(entries) >= 2),
    corruption_kind=_CORRUPTION_KIND,
    corrupt_payload=_CORRUPT_PAYLOADS,
    # ``target_seed`` escolhe a entrada-alvo de forma determinística
    # via módulo sobre os SHAs ordenados — sortear por índice direto
    # exigiria conhecer ``len(states)`` antes de gerar.
    target_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_p18_corruption_of_one_entry_does_not_invalidate_others(
    tmp_path_factory: pytest.TempPathFactory,
    states: dict[str, dict],
    corruption_kind: str,
    corrupt_payload: str,
    target_seed: int,
) -> None:
    """**Property 18**: corromper uma entrada não invalida vizinhas.

    **Validates: Requirements 7.7**

    Para qualquer estado prévio do cache com ``n ≥ 2`` entradas
    saudáveis, a corrupção de exatamente uma entrada (``.json`` ou
    ``.meta.json``) deve:

    - tornar ``get`` daquela entrada ``None``;
    - manter ``get`` das demais entradas funcional, devolvendo o
      ``response_json`` exato gravado por ``put``;
    - preservar fisicamente os arquivos das demais entradas.
    """

    cache = _fresh_cache(tmp_path_factory, "p18_corruption_isolation")

    # ----------------------------------------------------------------
    # Etapa 1: materializar todas as entradas como saudáveis.
    # ----------------------------------------------------------------
    # Iteramos em ordem determinística (SHAs ordenados) para que o
    # estado de disco seja reproduzível entre execuções com mesma
    # seed do Hypothesis. Também coletamos um snapshot do
    # ``response_payload`` de cada entrada para comparar mais tarde.
    sorted_shas = sorted(states.keys())
    for sha in sorted_shas:
        entry = states[sha]
        cache.put(
            sha256=sha,
            feature=entry["feature"],
            language_hints=entry["language_hints"],
            response_json=entry["response_payload"],
            image_size_bytes=entry["image_size_bytes"],
        )

    # Sanity: todas as entradas estão recuperáveis antes da corrupção.
    # Esse passo falsifica trivialmente regressões em ``put`` que
    # afetariam o cenário de corrupção de modo enganoso.
    for sha in sorted_shas:
        entry = states[sha]
        assert (
            cache.get(sha, entry["feature"], entry["language_hints"])
            == entry["response_payload"]
        ), "round-trip pré-corrupção deve recuperar o payload original"

    # ----------------------------------------------------------------
    # Etapa 2: escolher e corromper uma entrada-alvo.
    # ----------------------------------------------------------------
    target_sha = sorted_shas[target_seed % len(sorted_shas)]
    target_entry = states[target_sha]

    if corruption_kind == "json":
        target_path = cache.cache_dir / f"{target_sha}.json"
    else:  # "meta"
        target_path = cache.cache_dir / f"{target_sha}.meta.json"

    # Sanity: o arquivo-alvo realmente foi gravado por ``put`` antes
    # de tentarmos sobrescrevê-lo. Sem essa garantia o teste poderia
    # passar por engano caso ``put`` falhasse silenciosamente.
    assert target_path.is_file(), (
        f"arquivo-alvo {target_path} não foi gravado por put"
    )

    # Sobrescreve o arquivo com bytes não-parseáveis. Usamos
    # ``write_bytes`` em vez de ``write_text`` para que sequências
    # binárias como ``\x00\x01...`` cheguem ao disco intactas — o
    # ``json.loads`` ainda assim falhará na leitura.
    target_path.write_bytes(corrupt_payload.encode("utf-8", errors="replace"))

    # ----------------------------------------------------------------
    # Etapa 3: invariantes pós-corrupção.
    # ----------------------------------------------------------------
    # (a) A entrada corrompida vira inacessível.
    assert (
        cache.get(
            target_sha,
            target_entry["feature"],
            target_entry["language_hints"],
        )
        is None
    ), (
        "GcvCache.get deve devolver None para entrada corrompida "
        f"(corruption_kind={corruption_kind!r})"
    )

    # (b) Todas as outras entradas continuam recuperáveis com o
    # payload original. Esse é o coração da P18: a corrupção é
    # estritamente local.
    for sha in sorted_shas:
        if sha == target_sha:
            continue
        neighbor = states[sha]
        assert (
            cache.get(sha, neighbor["feature"], neighbor["language_hints"])
            == neighbor["response_payload"]
        ), (
            f"vizinha {sha} foi indevidamente afetada pela corrupção "
            f"de {target_sha}"
        )

    # (c) Os arquivos físicos das vizinhas continuam presentes — o
    # cache nunca remove vizinhos como efeito colateral.
    for sha in sorted_shas:
        if sha == target_sha:
            continue
        assert (cache.cache_dir / f"{sha}.json").is_file(), (
            f"{sha}.json foi removido indevidamente"
        )
        assert (cache.cache_dir / f"{sha}.meta.json").is_file(), (
            f"{sha}.meta.json foi removido indevidamente"
        )


# ===========================================================================
# Smoke test (task 4.5) — schema do par ``<sha>.json`` + ``<sha>.meta.json``
# ===========================================================================
#
# **Validates: Requirements 7.2**
#
# Diferente das propriedades acima (P5 e P18), este é um teste de exemplo
# único — sem ``@given`` — focado em fixar o **schema concreto** do par de
# arquivos gravado por ``GcvCache.put``. Em particular, verifica:
#
# 1. ``<sha>.json`` existe e seu conteúdo é o ``response_json`` original
#    (round-trip de leitura crua, sem passar por ``GcvCache.get``).
# 2. ``<sha>.meta.json`` existe e contém EXATAMENTE as quatro chaves
#    obrigatórias declaradas no Requirement 7.2: ``created_at``,
#    ``feature``, ``language_hints``, ``image_size_bytes``.
# 3. Os tipos das chaves obrigatórias são coerentes com o design:
#    - ``created_at``: string ISO-8601 parseável por
#      ``datetime.fromisoformat``;
#    - ``feature``: string pertencente a ``ALLOWED_FEATURES``;
#    - ``language_hints``: lista de strings (JSON não tem tupla);
#    - ``image_size_bytes``: inteiro não-negativo.
#
# Esse smoke test complementa as property tests servindo de "guard rail"
# rápido: qualquer regressão que altere o nome de uma chave, troque o
# tipo de ``language_hints`` para string única, ou deixe de gravar o
# timestamp deve ser detectada aqui antes de chegar nos testes mais
# pesados.


def test_smoke_meta_json_schema_and_types(tmp_path: Path) -> None:
    """Garante o schema documentado do par ``<sha>.json`` + ``.meta.json``.

    **Validates: Requirements 7.2**

    Constrói um ``GcvCache`` apontando para ``tmp_path``, executa um
    ``put`` com valores conhecidos e inspeciona diretamente o ``.json``
    e o ``.meta.json`` resultantes. A leitura é feita via
    ``json.loads`` cru (sem passar por ``GcvCache.get``) para que o
    teste falhe se a *forma* do arquivo no disco mudar — mesmo que o
    ``get`` continue funcional por compensar internamente.
    """

    import json
    from datetime import datetime

    cache = GcvCache(cache_dir=tmp_path / "smoke_cache")

    # Valores fixos e bem-tipados — facilitam comparação direta nas
    # asserções abaixo. ``sha256`` segue o formato hexadecimal canônico
    # de 64 chars (saída de ``hashlib.sha256().hexdigest()``).
    sha256 = "a" * 64
    feature = "DOCUMENT_TEXT_DETECTION"
    language_hints = ("pt", "en")
    response_json = {
        "fullTextAnnotation": {"text": "Valor Energético 75kcal"},
        "textAnnotations": [],
    }
    image_size_bytes = 4096

    json_path = cache.put(
        sha256=sha256,
        feature=feature,
        language_hints=language_hints,
        response_json=response_json,
        image_size_bytes=image_size_bytes,
    )

    # ----------------------------------------------------------------
    # (1) ``<sha>.json`` existe e mantém o ``response_json`` original.
    # ----------------------------------------------------------------
    assert json_path.is_file()
    assert json_path.name == f"{sha256}.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == response_json

    # ----------------------------------------------------------------
    # (2) ``<sha>.meta.json`` existe e tem as quatro chaves obrigatórias.
    # ----------------------------------------------------------------
    meta_path = cache.cache_dir / f"{sha256}.meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    required_keys = {
        "created_at",
        "feature",
        "language_hints",
        "image_size_bytes",
    }
    assert required_keys.issubset(meta.keys()), (
        f"meta.json deve conter as chaves {required_keys}; "
        f"presentes: {set(meta.keys())}"
    )

    # ----------------------------------------------------------------
    # (3) Tipos coerentes para cada chave obrigatória.
    # ----------------------------------------------------------------

    # ``created_at``: string ISO-8601 parseável. ``fromisoformat`` em
    # Python 3.11+ aceita o offset ``+00:00`` produzido por
    # ``datetime.now(timezone.utc).isoformat()``.
    assert isinstance(meta["created_at"], str)
    parsed_ts = datetime.fromisoformat(meta["created_at"])
    # Sanity adicional: o timestamp deve ter timezone (UTC) e não ser
    # naive — o cache grava com ``timezone.utc`` explicitamente.
    assert parsed_ts.tzinfo is not None

    # ``feature``: string em ``ALLOWED_FEATURES``.
    assert isinstance(meta["feature"], str)
    assert meta["feature"] in ALLOWED_FEATURES
    assert meta["feature"] == feature

    # ``language_hints``: lista de strings (JSON não tem tupla; o
    # ``put`` converte para ``list`` antes de serializar).
    assert isinstance(meta["language_hints"], list)
    assert all(isinstance(h, str) for h in meta["language_hints"])
    assert meta["language_hints"] == list(language_hints)

    # ``image_size_bytes``: inteiro não-negativo. ``bool`` é subclasse de
    # ``int`` em Python; o cache aceita ``int(image_size_bytes)`` e a
    # serialização preserva ``int``.
    assert isinstance(meta["image_size_bytes"], int)
    assert not isinstance(meta["image_size_bytes"], bool)
    assert meta["image_size_bytes"] >= 0
    assert meta["image_size_bytes"] == image_size_bytes


# ===========================================================================
# Property 19 — Entradas de cache não expiram automaticamente
# ===========================================================================
#
# **Validates: Requirements 7.8**
#
# Cenário: dado um cache populado com uma entrada saudável via ``put``,
# substitui-se manualmente o campo ``created_at`` de ``.meta.json`` por
# um timestamp arbitrariamente antigo (ex.: epoch UNIX em 1970,
# virada do milênio, ou quaisquer valores em décadas passadas). A
# invariante exige que ``cache.get(...)`` continue devolvendo o
# ``response_json`` original — ``GcvCache`` não aplica TTL, não
# invalida por idade e não invalida por mismatch de schema/código/
# modelo (Requirement 7.8). Apenas a corrupção descrita em R7.7 pode
# causar descarte automático.
#
# A propriedade complementa P18: P18 prova que corrupção é local; P19
# prova que idade nunca é causa de descarte. Juntas elas caracterizam
# o conjunto fechado de razões pelas quais ``get`` pode devolver
# ``None`` para uma entrada gravada por ``put``.


# Conjunto de "épocas" arbitrariamente antigas usadas para reescrever
# ``created_at``. Cada entrada é uma string ISO-8601 plausível. O
# Hypothesis amostra uniformemente a tupla; o tamanho enxuto facilita
# shrinking determinístico para o exemplo minimal mais simbólico
# (epoch UNIX).
_ANCIENT_TIMESTAMPS: tuple[str, ...] = (
    # Epoch UNIX exato — referência simbólica máxima de "muito antigo".
    "1970-01-01T00:00:00+00:00",
    # Antes do epoch — formato ISO-8601 estendido com ano negativo é
    # permitido pela RFC 3339 mas raro em produção; serve como sinal
    # adversarial para forçar o cache a não tentar parsear o campo.
    "1900-01-01T00:00:00+00:00",
    # Década pré-internet ampla.
    "1985-06-15T12:34:56+00:00",
    # Virada do milênio Y2K — emblemático para sistemas legados.
    "1999-12-31T23:59:59+00:00",
    "2000-01-01T00:00:00+00:00",
    # Qualquer ponto anterior ao desenvolvimento da feature GCV.
    "2010-07-04T08:15:30+00:00",
    # Sufixo ``Z`` (formato Zulu) em vez de ``+00:00`` — variação
    # ortográfica que ``GcvCache`` deve igualmente ignorar.
    "1970-01-01T00:00:00Z",
    # Valores totalmente fora do padrão ISO-8601 — ``GcvCache`` não
    # parseia o campo, então strings arbitrárias devem ser ignoradas
    # silenciosamente.
    "definitely-not-a-timestamp",
    "",
)


@given(
    sha256=_SHA256_HEX,
    feature=_FEATURE,
    hints=bcp47_hints(),
    response=gcv_response_dict(),
    image_size=_IMAGE_SIZE,
    ancient_created_at=st.sampled_from(_ANCIENT_TIMESTAMPS),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_p19_entries_do_not_expire_by_age(
    tmp_path_factory: pytest.TempPathFactory,
    sha256: str,
    feature: str,
    hints: tuple[str, ...],
    response: dict,
    image_size: int,
    ancient_created_at: str,
) -> None:
    """**Property 19**: ``created_at`` antigo não causa descarte automático.

    **Validates: Requirements 7.8**

    Para qualquer entrada saudável gravada por ``put``, reescrever o
    campo ``created_at`` de ``.meta.json`` para um timestamp
    arbitrariamente antigo (ou mesmo uma string sem formato válido)
    NÃO deve afetar a recuperação via ``get``. O contrato é
    explicitamente "sem TTL e sem invalidação por idade"
    (Requirement 7.8): apenas a corrupção dos arquivos JSON
    (Requirement 7.7) pode causar descarte automático.

    Para qualquer entrada saudável gravada por ``put``, reescrever o
    campo ``created_at`` de ``.meta.json`` para um timestamp
    arbitrariamente antigo (ou mesmo uma string sem formato válido)
    NÃO deve afetar a recuperação via ``get``. O contrato é
    explicitamente "sem TTL e sem invalidação por idade"
    (Requirement 7.8): apenas a corrupção dos arquivos JSON
    (Requirement 7.7) pode causar descarte automático.

    O teste:

    1. Grava a entrada via ``put`` (que define ``created_at`` para o
       agora UTC).
    2. Lê ``.meta.json``, sobrescreve o campo ``created_at`` com um
       valor antigo, e regrava o arquivo preservando os demais
       campos (``feature``, ``language_hints``, ``image_size_bytes``).
    3. Confirma que ``cache.get(sha, feature, hints)`` ainda devolve
       o ``response_json`` original — provando que a idade não foi
       usada como critério de invalidação.
    """

    cache = _fresh_cache(tmp_path_factory, "p19_no_expiration")

    cache.put(
        sha256=sha256,
        feature=feature,
        language_hints=hints,
        response_json=response,
        image_size_bytes=image_size,
    )

    # Sanity: a entrada acabou de ser gravada e é recuperável com
    # ``created_at`` "atual". Garante que qualquer falha posterior é
    # consequência exclusiva da reescrita de ``created_at``.
    assert cache.get(sha256, feature, hints) == response

    # Reescreve ``created_at`` em ``.meta.json`` mantendo intactos os
    # campos usados pelo filtro de compatibilidade (``feature`` e
    # ``language_hints``). Isso isola o efeito do timestamp antigo.
    meta_path = cache.cache_dir / f"{sha256}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["created_at"] = ancient_created_at
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Invariante: a recuperação continua intacta independentemente de
    # quão antigo é o ``created_at``. Mesmo timestamps malformados ou
    # vazios são ignorados pelo cache — ele não parseia o campo.
    cached = cache.get(sha256, feature, hints)
    assert cached is not None, (
        "GcvCache.get devolveu None para entrada com created_at antigo "
        f"({ancient_created_at!r}); entradas válidas não devem expirar "
        "por idade (R7.8)"
    )
    assert cached == response


# ---------------------------------------------------------------------------
# Property 16 — Validates: Requirements 7.5, 14.5
# ---------------------------------------------------------------------------
# P16: quando ``cache_enabled=False``, o ``GcvClient`` deve suprimir
# completamente qualquer I/O sobre o ``cache_dir`` — nem lookup nem
# gravação. Após uma chamada a ``GcvClient.fetch()`` bem-sucedida, o
# diretório de cache informado na configuração deve permanecer completamente
# vazio (sem arquivos ``.json`` nem ``.meta.json``).
#
# A prova usa um ``api_client`` stub injetado via ``GcvClient.build(...)``
# para que nenhuma credencial real seja necessária, e um ``cache_dir``
# isolado em ``tmp_path`` para que a verificação seja determinística.
# ---------------------------------------------------------------------------


def _tiny_png_bytes() -> bytes:
    """Devolve bytes PNG mínimos válidos (imagem 1x1 pixel preto).

    Usados como ``png_bytes`` na chamada a ``GcvClient.fetch`` quando
    o test não precisa exercitar o parser (o stub ignora os bytes e
    devolve o response_json injetado).
    """

    import numpy as np

    from ocr.cloud_vision.parser import encode_png

    image = np.zeros((1, 1, 3), dtype=np.uint8)
    return encode_png(image)


def test_cache_disabled_creates_no_files_example(tmp_path: Path) -> None:
    """``cache_enabled=False`` → zero arquivos em ``cache_dir`` após ``fetch``.

    **Validates: Requirements 7.5, 14.5**

    Caso determinístico: configura ``GcvAppConfig`` com
    ``cache_enabled=False``, injeta um stub de API via
    ``GcvClient.build(..., api_client=stub)``, chama ``fetch()`` e
    verifica que o ``cache_dir`` continua vazio. Isso garante que o
    ``GcvClient`` não gravou nem leu nenhum arquivo de cache mesmo
    quando a chamada à API foi bem-sucedida.
    """

    from unittest.mock import MagicMock

    from ocr.cloud_vision.app_config import GcvAppConfig
    from ocr.cloud_vision.client import GcvClient

    cache_dir = tmp_path / "gcv_cache"
    cache_dir.mkdir()

    config = GcvAppConfig.from_dict(
        {
            "cache_enabled": False,
            "cache_dir": str(cache_dir),
            "on_failure": "skip",
            "credentials_path": None,
            "request_timeout_seconds": 30,
        },
        tmp_path,
    )

    stub = MagicMock()
    stub.annotate_image.return_value = {
        "fullTextAnnotation": {"text": "Carboidratos 15 g", "pages": []},
    }

    client = GcvClient.build(config, tmp_path, api_client=stub)
    png_bytes = _tiny_png_bytes()
    result = client.fetch(png_bytes, "DOCUMENT_TEXT_DETECTION", ["pt"])

    # O fetch deve ter retornado um resultado real (cache_hit=False).
    assert result.cache_hit is False

    # Invariante P16: zero arquivos no cache_dir.
    files = list(cache_dir.iterdir())
    assert files == [], (
        f"cache_enabled=False mas {len(files)} arquivo(s) encontrado(s) "
        f"em cache_dir: {[f.name for f in files]}"
    )


@given(
    feature=st.sampled_from(ALLOWED_FEATURES),
    hints=bcp47_hints(),
)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_cache_disabled_creates_no_files_property(
    tmp_path_factory: pytest.TempPathFactory,
    feature: str,
    hints: tuple[str, ...],
) -> None:
    """``cache_enabled=False`` → zero arquivos para qualquer ``(feature, hints)``.

    **Validates: Requirements 7.5, 14.5**

    Variante property-based: varia ``feature`` e ``language_hints`` para
    garantir que nenhuma combinação aciona gravação de cache quando
    ``cache_enabled=False``, mesmo que ``(feature, hints)`` sejam
    diferentes entre exemplos (descarta a hipótese de hard-coded
    ``feature == "DOCUMENT_TEXT_DETECTION"``).
    """

    from unittest.mock import MagicMock

    from ocr.cloud_vision.app_config import GcvAppConfig
    from ocr.cloud_vision.client import GcvClient

    project_root = tmp_path_factory.mktemp("p16_cache_disabled")
    cache_dir = project_root / "gcv_cache"
    cache_dir.mkdir()

    config = GcvAppConfig.from_dict(
        {
            "cache_enabled": False,
            "cache_dir": str(cache_dir),
            "on_failure": "skip",
            "credentials_path": None,
        },
        project_root,
    )

    stub = MagicMock()
    stub.annotate_image.return_value = {
        "fullTextAnnotation": {"text": "test", "pages": []},
    }

    client = GcvClient.build(config, project_root, api_client=stub)
    png_bytes = _tiny_png_bytes()
    client.fetch(png_bytes, feature, list(hints))

    files = list(cache_dir.iterdir())
    assert files == [], (
        f"cache_enabled=False + feature={feature!r} + hints={hints!r} "
        f"produziu {len(files)} arquivo(s): {[f.name for f in files]}"
    )
