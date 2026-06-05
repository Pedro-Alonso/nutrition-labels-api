"""Rate limiter com janela deslizante de 60 segundos para chamadas à GCV.

Implementa o comportamento descrito nos Requirements 8.1–8.4 e na seção
"RateLimiter" do design: limita ``max_per_minute`` chamadas reais à API em
qualquer janela deslizante de 60s, bloqueando o thread chamador até liberar
vaga sem polling ativo.

Notas de implementação:

- ``clock`` e ``sleep`` são parâmetros injetáveis por chamada (defaults
  ``time.monotonic`` / ``time.sleep``). Isso é proposital para permitir
  testes determinísticos: a propriedade P10 do design exige verificar a
  invariante "em qualquer janela de 60s, no máximo ``N`` chamadas" sem
  depender de relógio real.
- O ``threading.Lock`` é segurado **apenas** durante a leitura/escrita da
  lista de timestamps; o ``sleep`` ocorre fora do lock para não bloquear
  outros threads que poderiam ter vaga após o expurgo da próxima iteração.
- A janela é estritamente ``< now - 60.0`` (semi-aberta à esquerda):
  timestamps com idade exata de 60.0s ainda contam como dentro da janela,
  o que é a interpretação conservadora (não viola o limite por arredondamento
  de relógios monotônicos).
- Cache hits NÃO devem chamar ``acquire`` (Requirement 8.4); essa decisão
  pertence a ``CloudVisionPipeline``/``GcvClient``, não a este módulo.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(slots=True)
class RateLimiter:
    """Limita ``max_per_minute`` aquisições em qualquer janela deslizante de 60s.

    Attributes:
        max_per_minute: Número máximo de aquisições permitidas em qualquer
            janela contígua de 60 segundos. Deve ser inteiro positivo —
            valores não-positivos ou inválidos são coibidos a montante (no
            ``build_default_reader``, conforme Requirement 8.5), portanto
            esta classe assume entrada já validada.
        _timestamps: Histórico ordenado dos timestamps de aquisições
            recentes (em segundos, no relógio fornecido por ``clock``).
            Mantido em ordem cronológica de inserção; o expurgo remove o
            prefixo correspondente a entradas mais antigas que ``now - 60``.
        _lock: Mutex que protege leituras/escritas de ``_timestamps``. Nunca
            é mantido durante ``sleep`` para não bloquear outros chamadores.
    """

    max_per_minute: int
    _timestamps: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Bloqueia até que uma chamada possa ocorrer sem violar o limite.

        Algoritmo (janela deslizante):

        1. Adquire o lock.
        2. Lê ``now = clock()``.
        3. Remove ``in-place`` os timestamps com idade ``>= 60.0`` em relação
           a ``now`` (i.e., timestamps menores que ``now - 60.0``).
        4. Se há vaga (``len(_timestamps) < max_per_minute``): registra ``now``
           em ``_timestamps``, libera o lock e retorna.
        5. Senão, calcula ``wait = max(0.0, (oldest_timestamp + 60.0) - now)``
           — o tempo mínimo até o timestamp mais antigo sair da janela e
           liberar uma vaga. Libera o lock, dorme ``sleep(wait)`` e recomeça
           do passo 1.

        O loop é necessário porque, ao acordar, outros threads podem ter
        consumido a vaga liberada; nesse caso o expurgo da próxima iteração
        recalcula o ``wait`` correto.

        Args:
            clock: Função sem argumentos que devolve o instante atual em
                segundos. Default ``time.monotonic`` (imune a ajustes do
                relógio do sistema). Em testes, injeta-se um clock falso
                determinístico.
            sleep: Função que dorme ``segundos`` sem polling ativo. Default
                ``time.sleep``. Em testes, injeta-se uma stub que apenas
                avança o clock falso.
        """
        while True:
            self._lock.acquire()
            try:
                now = clock()
                # Expurga timestamps fora da janela de 60s (in-place para
                # preservar a referência da lista do dataclass).
                cutoff = now - 60.0
                # Como ``_timestamps`` é mantido em ordem cronológica de
                # inserção, basta encontrar o primeiro índice ainda dentro
                # da janela e descartar o prefixo. Loop manual em vez de
                # ``bisect`` para manter zero dependências externas.
                drop = 0
                for ts in self._timestamps:
                    if ts < cutoff:
                        drop += 1
                    else:
                        break
                if drop:
                    del self._timestamps[:drop]

                if len(self._timestamps) < self.max_per_minute:
                    # Há vaga: registra esta aquisição e retorna.
                    self._timestamps.append(now)
                    return

                # Janela cheia: calcula o tempo mínimo até a próxima vaga.
                # ``_timestamps[0]`` é o mais antigo dentro da janela; ele
                # sai ao completar 60s desde o seu registro.
                oldest = self._timestamps[0]
                wait = (oldest + 60.0) - now
                if wait < 0.0:
                    wait = 0.0
            finally:
                # Libera o lock ANTES de dormir para não bloquear outros
                # threads que poderiam ter vaga após o expurgo da próxima
                # iteração (Requirement 8.3: aguarda sem polling ativo).
                self._lock.release()

            sleep(wait)
