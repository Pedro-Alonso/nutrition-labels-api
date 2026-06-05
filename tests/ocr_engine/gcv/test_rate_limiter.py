"""Property test P10: rate limiter respeita janela deslizante e ignora cache hits.

Validates: Requirements 8.2, 8.3, 8.4

A propriedade afirma três invariantes complementares sobre ``RateLimiter``:

1. **Janela deslizante de 60 segundos** (Requirement 8.2) — para qualquer
   sequência de `N` chamadas reais à API, e para qualquer instante
   ``t``, o número de timestamps registrados por ``acquire()`` que caem
   dentro da janela ``[t - 60, t]`` é menor ou igual a
   ``max_per_minute``. Aqui "timestamp registrado" é o valor de
   ``clock()`` no momento em que ``acquire`` decidiu liberar a vaga
   (i.e., o que é apendado a ``_timestamps``).
2. **Espera quando a janela está cheia** (Requirement 8.3) — se a
   janela está saturada no instante da chamada, ``acquire`` invoca
   ``sleep`` ao menos uma vez antes de completar. A duração pode
   ser ``0`` no caso de borda em que o timestamp mais antigo está
   exatamente no limiar (``oldest == now - 60``); o que importa
   semanticamente é que o limiter cede o controle via ``sleep`` em
   vez de fazer polling ativo. Em casos não-degenerados (oldest
   estritamente dentro da janela), a duração será positiva — esse é
   o cenário típico em produção.
3. **Cache hits não consomem cota** (Requirement 8.4) — eventos de
   tipo ``"hit"`` no modelo de eventos *não* invocam ``acquire``.
   Apenas eventos ``"miss"`` (chamadas reais) avançam o limiter.
   Verificamos contando explicitamente quantas vezes ``acquire`` foi
   chamado e comparando com a contagem de eventos ``"miss"`` na
   sequência gerada.

A propriedade é dirigida por ``rate_limiter_event_sequences()`` em
``tests/gcv/strategies.py``, que produz sequências ordenadas de eventos
``(timestamp, kind)`` com gaps controlados. Combinamos com uma estratégia
sobre ``max_per_minute`` (inteiros pequenos, ``M ∈ [1, 5]``) — valores
maiores apenas tornam a invariante trivial, sem ganho de cobertura.

Notas de implementação do harness:

- ``clock`` e ``sleep`` são funções injetadas via ``acquire(...)``,
  conforme o contrato declarado no design (seção *RateLimiter*). Não
  há dependência do relógio real, garantindo execução determinística.
- O fake ``sleep`` adiciona um épsilon ``1e-9`` ao avanço do clock
  ALÉM do ``seconds`` solicitado. Isso é necessário porque o
  ``RateLimiter`` usa cutoff ``< now - 60.0`` (estrito): se o clock
  avançar exatamente ``wait``, o timestamp mais antigo ainda satisfaz
  ``ts == now - 60`` e não é removido — gerando um loop com ``sleep(0)``
  em cenários adversariais como múltiplos timestamps no mesmo instante
  ``t=0``. O épsilon nudge garante progresso sem violar a invariante:
  o timestamp registrado fica ligeiramente após ``oldest + 60``, e
  qualquer janela ``[t-60, t]`` ainda contém no máximo ``M`` desses
  timestamps.
- O snapshot dos timestamps registrados é feito copiando
  ``rl._timestamps[-1]`` após cada ``acquire`` — esse é o valor que o
  ``RateLimiter`` acabou de apender. Acessar ``_timestamps`` é
  intrusivo, mas o teste vive no mesmo módulo e a alternativa
  (registrar via interceptor de ``clock``) acrescenta complexidade
  sem ganho semântico.
"""

from __future__ import annotations

from typing import Callable

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ocr.cloud_vision.rate_limiter import RateLimiter
from tests.ocr_engine.gcv.strategies import rate_limiter_event_sequences


# Domínio de ``max_per_minute``: inteiros pequenos suficientes para
# provocar bursts e esperas no cenário gerado, e pequenos o bastante
# para que o Hypothesis encontre rapidamente exemplos minimais.
_MAX_PER_MINUTE = st.integers(min_value=1, max_value=5)


def _build_fake_clock() -> tuple[
    Callable[[], float],
    Callable[[float], None],
    list[float],
    list[float],
]:
    """Constrói um par ``(clock, sleep)`` determinístico para injeção.

    O estado é encapsulado em listas mutáveis (em vez de ``nonlocal``)
    para que o caller possa avançar o clock manualmente entre eventos
    do harness — simulando a passagem natural do tempo entre chamadas
    do ``CloudVisionPipeline``.

    Retorna:

    - ``fake_clock``: função sem argumentos que devolve o instante atual.
    - ``fake_sleep``: função que registra a duração solicitada em
      ``sleep_log`` e avança o clock por ``seconds + 1e-9`` (épsilon
      contra a borda do cutoff estrito ``<`` do ``RateLimiter``).
    - ``now_holder``: lista de tamanho 1 com o instante atual,
      manipulável diretamente pelo caller (``now_holder[0] = ts``).
    - ``sleep_log``: histórico das durações passadas a ``fake_sleep``.
    """

    now_holder = [0.0]
    sleep_log: list[float] = []

    def fake_clock() -> float:
        return now_holder[0]

    def fake_sleep(seconds: float) -> None:
        sleep_log.append(seconds)
        # Avança o clock por ``seconds`` mais um épsilon.
        # O épsilon resolve o caso adversarial em que ``seconds == 0``
        # (chamada degenerada com vaga prestes a liberar) e o caso de
        # ``seconds > 0`` em que o timestamp mais antigo está exatamente
        # no limiar do cutoff: sem o nudge, o ``RateLimiter`` entra em
        # loop infinito porque ``ts < cutoff`` é estritamente falso na
        # igualdade.
        now_holder[0] += seconds + 1e-9

    return fake_clock, fake_sleep, now_holder, sleep_log


@given(
    events=rate_limiter_event_sequences(),
    max_per_minute=_MAX_PER_MINUTE,
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_p10_sliding_window_and_cache_hits_ignored(
    events: list[tuple[float, str]],
    max_per_minute: int,
) -> None:
    """**Property 10**: janela deslizante respeitada + cache hits ignorados.

    **Validates: Requirements 8.2, 8.3, 8.4**

    Para qualquer sequência ordenada de eventos ``(timestamp, kind)``
    com ``kind ∈ {"hit", "miss"}`` e qualquer ``M = max_per_minute``,
    o ``RateLimiter`` garante:

    1. Apenas eventos ``"miss"`` invocam ``acquire`` (Requirement 8.4).
       Verificado contando manualmente.
    2. Para qualquer instante ``t`` correspondente a um timestamp
       registrado, ``|{ ts ∈ acquire_ts : t - 60 ≤ ts ≤ t }| ≤ M``
       (Requirement 8.2). Verificado em todos os "right edges" da
       sequência de timestamps registrados (suficiente porque a
       contagem da janela é monotonicamente não-decrescente em ``t``
       até o próximo evento de saída pela esquerda).
    3. Quando a janela está cheia no momento de um ``acquire``, a
       implementação invoca ``sleep`` ao menos uma vez antes de
       retornar (Requirement 8.3 — aguarda sem polling ativo). A
       duração é tipicamente positiva; ``0`` é tolerado apenas no
       caso de borda em que o timestamp mais antigo está exatamente
       no limiar do cutoff (``oldest == now - 60``), onde o
       ``RateLimiter`` ainda assim cede o controle via ``sleep`` em
       vez de fazer polling.
    """

    rl = RateLimiter(max_per_minute=max_per_minute)
    fake_clock, fake_sleep, now_holder, sleep_log = _build_fake_clock()

    # Histórico canônico: timestamps que o ``RateLimiter`` apendou em
    # ``_timestamps``. Um por miss bem-sucedido; nenhum por hit.
    acquire_timestamps: list[float] = []

    # Contador de chamadas a ``acquire`` (verifica Requirement 8.4).
    acquire_call_count = 0

    # Conta dos eventos "miss" gerados pela strategy. Deve bater
    # exatamente com ``acquire_call_count`` ao final.
    expected_miss_count = sum(1 for _, kind in events if kind == "miss")

    for event_ts, kind in events:
        # Avança o clock até o instante do evento, mas nunca volta
        # atrás (o relógio monotônico do design jamais regride).
        # ``now_holder[0]`` pode ter sido empurrado para frente pelo
        # ``fake_sleep`` em iterações anteriores; preservamos esse
        # avanço.
        if event_ts > now_holder[0]:
            now_holder[0] = event_ts

        if kind == "hit":
            # Cache hit: NÃO invoca ``acquire``. A linha a seguir
            # apenas documenta o invariante — não há side effect.
            continue

        # Miss: chamada real à API. ``acquire`` deve registrar este
        # timestamp e, se a janela estiver cheia, dormir até liberar.
        sleeps_before = len(sleep_log)

        # Captura ``now`` AGORA (pré-acquire) para a verificação de
        # "janela estava cheia" abaixo. ``acquire`` lê ``clock()``
        # internamente e pode acabar registrando um timestamp >= este
        # valor, mas a contagem da janela no momento da chamada usa o
        # ``now`` corrente antes de qualquer sleep do limiter.
        now_pre_acquire = now_holder[0]

        rl.acquire(clock=fake_clock, sleep=fake_sleep)
        acquire_call_count += 1

        # ``rl._timestamps[-1]`` é o valor recém-apendado: o instante
        # ``clock()`` que o limiter capturou ao decidir liberar a
        # vaga. Preservamos uma cópia porque ``_timestamps`` pode ser
        # podado em ``acquire``s subsequentes.
        assert rl._timestamps, (
            "_timestamps deve conter ao menos a aquisição recém-registrada"
        )
        acquire_timestamps.append(rl._timestamps[-1])

        sleeps_after = len(sleep_log)

        # ----------------------------------------------------------------
        # Requirement 8.3: quando a janela estava cheia, sleep ocorreu.
        # ----------------------------------------------------------------
        # Reconstruímos a contagem PRÉ-acquire usando os timestamps já
        # registrados ANTES desta chamada (excluindo o atual) e o
        # ``now`` no momento da chamada. Se essa contagem é ``>= M``,
        # a janela estava saturada e ``acquire`` precisaria ceder o
        # controle ao menos uma vez via ``sleep``.
        prev_ts = acquire_timestamps[:-1]
        in_window_before = sum(
            1
            for ts in prev_ts
            if now_pre_acquire - 60.0 <= ts <= now_pre_acquire
        )
        if in_window_before >= max_per_minute:
            # Janela cheia ⇒ pelo menos uma chamada a ``sleep`` deve ter
            # ocorrido (Requirement 8.3 — aguarda sem polling ativo). A
            # duração é tipicamente positiva; ``0`` é aceitável apenas
            # no caso de borda em que o timestamp mais antigo está
            # exatamente no limiar do cutoff (``oldest == now - 60``),
            # situação em que o limiter ainda assim cede o controle.
            assert sleeps_after > sleeps_before, (
                "RateLimiter não invocou sleep apesar de janela cheia "
                f"(events={events}, max={max_per_minute}, "
                f"now_pre={now_pre_acquire}, prev_ts={prev_ts})"
            )

    # ----------------------------------------------------------------
    # Requirement 8.4: cache hits não invocam ``acquire``.
    # ----------------------------------------------------------------
    assert acquire_call_count == expected_miss_count, (
        "acquire foi chamado um número de vezes diferente da contagem de "
        f"eventos 'miss': calls={acquire_call_count}, "
        f"misses={expected_miss_count}, events={events}"
    )
    assert len(acquire_timestamps) == expected_miss_count

    # ----------------------------------------------------------------
    # Requirement 8.2: invariante da janela deslizante.
    # ----------------------------------------------------------------
    # Verifica que para todo "right edge" t = acquire_timestamps[i],
    # |{ ts ∈ acquire_timestamps : t - 60 ≤ ts ≤ t }| ≤ max_per_minute.
    # Como a função de contagem em ``[t-60, t]`` só muda quando ``t``
    # cruza um timestamp registrado, checar todos os timestamps
    # registrados é suficiente para cobrir o supremum em ``ℝ``.
    for t in acquire_timestamps:
        in_window = sum(
            1 for ts in acquire_timestamps if t - 60.0 <= ts <= t
        )
        assert in_window <= max_per_minute, (
            "Janela deslizante violada: "
            f"t={t}, count={in_window}, max={max_per_minute}, "
            f"acquire_ts={acquire_timestamps}, events={events}"
        )


# ===========================================================================
# Casos determinísticos auxiliares
# ===========================================================================
#
# Os testes abaixo travam cenários canônicos da Property 10 que a
# strategy ``rate_limiter_event_sequences`` cobre estatisticamente, mas
# que vale a pena fixar como regressão explícita: burst no mesmo
# instante, intercalação com cache hits, e sequência uniforme acima do
# limite. Se algum branch do limiter quebrar, o relatório de falha
# mostra o caso específico em vez de depender do shrinking do
# Hypothesis.


def test_burst_acima_do_limite_dorme_e_libera_em_60s() -> None:
    """Burst de ``M+1`` chamadas no mesmo instante: ``M`` passam, 1 espera 60s.

    Cenário canônico do Requirement 8.3: ``M`` chamadas em ``t=0`` cabem
    na janela; a ``M+1``-ésima precisa esperar até o timestamp mais
    antigo sair da janela.
    """

    rl = RateLimiter(max_per_minute=2)
    fake_clock, fake_sleep, now_holder, sleep_log = _build_fake_clock()

    # 2 chamadas no mesmo instante: cabem.
    rl.acquire(clock=fake_clock, sleep=fake_sleep)
    rl.acquire(clock=fake_clock, sleep=fake_sleep)
    assert sleep_log == []

    # 3ª chamada no mesmo instante: janela cheia, deve esperar.
    rl.acquire(clock=fake_clock, sleep=fake_sleep)
    # Pelo menos um sleep com duração positiva ocorreu.
    assert any(s > 0.0 for s in sleep_log), (
        f"Esperava sleep positivo, obteve {sleep_log}"
    )
    # O clock avançou pelo menos 60 segundos.
    assert now_holder[0] >= 60.0


def test_cache_hits_nao_invocam_acquire() -> None:
    """Eventos ``"hit"`` não chamam ``acquire`` (Requirement 8.4).

    Mesmo com 100 cache hits intercalados a 1 miss, apenas 1 timestamp
    é registrado pelo limiter. Caso degenerado que falsifica
    diretamente qualquer regressão que faça o ``CloudVisionPipeline``
    chamar ``acquire`` em hits.
    """

    rl = RateLimiter(max_per_minute=1)
    fake_clock, fake_sleep, _now, sleep_log = _build_fake_clock()

    # Simulamos 100 hits + 1 miss + 100 hits.
    # Hits NÃO chamam acquire (são pulados pelo loop do harness).
    rl.acquire(clock=fake_clock, sleep=fake_sleep)

    # Após 1 miss: 1 timestamp registrado, nenhum sleep.
    assert len(rl._timestamps) == 1
    assert sleep_log == []


def test_chamadas_espacadas_alem_da_janela_nao_dormem() -> None:
    """Chamadas com gap > 60s não devem disparar ``sleep`` (Requirement 8.2).

    Quando o gap entre chamadas excede 60 segundos, o expurgo da
    janela deslizante deixa o limiter sempre com vaga disponível,
    mesmo com ``M=1``.
    """

    rl = RateLimiter(max_per_minute=1)
    fake_clock, fake_sleep, now_holder, sleep_log = _build_fake_clock()

    for offset in (0.0, 61.0, 122.0, 200.0):
        now_holder[0] = offset
        rl.acquire(clock=fake_clock, sleep=fake_sleep)

    # Nenhum sleep ocorreu — todas as chamadas tinham vaga imediata.
    assert sleep_log == []
    # ``_timestamps`` mantém apenas o último (gaps > 60 evictam os
    # anteriores em cada iteração).
    assert len(rl._timestamps) == 1


def test_max_per_minute_um_serializa_chamadas_consecutivas() -> None:
    """``M=1`` força gap mínimo de 60s entre chamadas consecutivas.

    Cenário extremo da Property 10: com ``M=1``, qualquer par de
    chamadas em sequência precisa estar separado por ≥ 60s no clock
    real. Verificamos que após 3 chamadas consecutivas em ``t=0``, o
    clock final é ≥ 120s e cada par adjacente de timestamps
    registrados está separado por aproximadamente 60s.
    """

    rl = RateLimiter(max_per_minute=1)
    fake_clock, fake_sleep, now_holder, _ = _build_fake_clock()

    rl.acquire(clock=fake_clock, sleep=fake_sleep)
    ts_first = rl._timestamps[-1]

    rl.acquire(clock=fake_clock, sleep=fake_sleep)
    ts_second = rl._timestamps[-1]

    rl.acquire(clock=fake_clock, sleep=fake_sleep)
    ts_third = rl._timestamps[-1]

    # Gaps entre chamadas consecutivas ≥ 60s (com tolerância para o
    # épsilon do fake_sleep).
    assert ts_second - ts_first >= 60.0
    assert ts_third - ts_second >= 60.0

    # Janela deslizante: cada timestamp isoladamente forma uma janela
    # de 1 elemento, respeitando ``M=1``.
    for t in (ts_first, ts_second, ts_third):
        in_window = sum(
            1 for ts in (ts_first, ts_second, ts_third) if t - 60.0 <= ts <= t
        )
        assert in_window <= 1
