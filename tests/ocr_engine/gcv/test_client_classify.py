"""Testes exemplo da classificação de exceções em ``GcvClient._classify``.

Cobre a tarefa **6.2** do plano de implementação: dado o contrato em
``ocr/cloud_vision/client.py``, qualquer exceção exposta pelo SDK
``google-cloud-vision`` (ou pelas camadas de rede/transporte) deve ser
traduzida em ``GcvError`` com:

- ``error`` na precedência canônica
  ``auth_error > quota_exceeded > timeout > generic_error``
  (Requirements 6.5–6.8);
- ``message`` igual a ``str(exc)`` truncada em 500 caracteres
  (Requirement 6.2);
- ``secondary`` ordenado pela mesma precedência quando há mais de uma
  classe aplicável (Requirement 6.8).

A estratégia de teste evita dependência do SDK real: o ``_classify``
inspeciona ``type(exc).__mro__`` por **nome** de classe (ver constantes
``_AUTH_CLASS_NAMES``, ``_QUOTA_CLASS_NAMES`` e ``_TIMEOUT_CLASS_NAMES``
no módulo). Basta declarar stubs locais com os nomes esperados
(``PermissionDenied``, ``ResourceExhausted``, ``DeadlineExceeded``,
``GoogleAPICallError``) e a classificação reage como se fossem as
classes do ``google.api_core.exceptions``. ``concurrent.futures.TimeoutError``
é importada do stdlib diretamente, pois é coberta por um ``isinstance``
explícito em ``_classify``.

Seguindo as convenções do projeto (``AGENTS.md``):
- Comentários e docstrings em português; identificadores em inglês.
- Imports absolutos a partir da raiz do projeto.
- Tipos modernos via ``from __future__ import annotations``.
- Nada vai para ``stdout`` — as asserções falam por si.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any

import pytest

from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.client import GcvClient
from ocr.cloud_vision.types import GcvError


# ---------------------------------------------------------------------------
# Stubs de exceções do ``google.api_core.exceptions``
# ---------------------------------------------------------------------------

# A classificação em ``GcvClient._classify`` casa pelo NOME da classe ao
# longo da MRO (``type(exc).__mro__``). Isso significa que basta declarar
# classes locais com os mesmos nomes do SDK para exercitar cada ramo de
# classificação sem instalar ``google-cloud-vision`` no ambiente de
# testes (Requirement 14.3 — feature opcional). Documentamos a origem de
# cada nome para que a manutenção fique rastreável caso o SDK mude.


class PermissionDenied(Exception):
    """Stub de ``google.api_core.exceptions.PermissionDenied``.

    Mapeia para ``auth_error`` em ``_classify`` (Requirement 6.5).
    """


class ResourceExhausted(Exception):
    """Stub de ``google.api_core.exceptions.ResourceExhausted``.

    Mapeia para ``quota_exceeded`` em ``_classify`` (Requirement 6.6).
    """


class DeadlineExceeded(Exception):
    """Stub de ``google.api_core.exceptions.DeadlineExceeded``.

    Mapeia para ``timeout`` em ``_classify`` (Requirement 6.7).
    """


class GoogleAPICallError(Exception):
    """Stub de ``google.api_core.exceptions.GoogleAPICallError``.

    Não pertence a nenhum dos conjuntos de classificação canônica;
    portanto, em ``_classify`` deve cair no fallback ``generic_error``.
    """


# ---------------------------------------------------------------------------
# Fixture local: cliente com SDK pré-injetado
# ---------------------------------------------------------------------------


@pytest.fixture
def client(
    tmp_project_root: Path,
    gcv_app_config_default: dict[str, Any],
) -> GcvClient:
    """Constrói uma instância de ``GcvClient`` mínima para chamar ``_classify``.

    ``_classify`` é um método de instância mas não consulta nenhum
    estado persistente do cliente: opera apenas sobre a exceção
    recebida. Mesmo assim, instanciamos via ``GcvClient.build`` para
    seguir o caminho canônico de construção (e cobrir indiretamente o
    contrato do construtor). Injetamos um ``api_client`` arbitrário
    apenas para sinalizar a ``_ensure_client`` que não precisa importar
    o SDK — embora ``_classify`` jamais o invoque, isso protege contra
    surpresas se a fixture for reutilizada em testes futuros.
    """

    config = GcvAppConfig.from_dict(
        gcv_app_config_default, project_root=tmp_project_root
    )
    # ``api_client`` é um sentinel arbitrário; ``_classify`` não toca
    # nele. Qualquer objeto truthy serve para impedir o caminho de
    # inicialização lazy.
    return GcvClient.build(
        config=config,
        project_root=tmp_project_root,
        api_client=object(),
    )


# ---------------------------------------------------------------------------
# Casos de classificação (Requirements 6.5–6.7)
# ---------------------------------------------------------------------------


def test_permission_denied_classifies_as_auth_error(client: GcvClient) -> None:
    """``PermissionDenied`` → ``auth_error`` (Requirement 6.5).

    A classe carrega o nome canônico do ``google.api_core`` e deve ser
    detectada via ``type(exc).__mro__`` em ``_AUTH_CLASS_NAMES``.
    Como nenhuma outra classificação se aplica, ``secondary`` deve
    estar vazio.
    """

    exc = PermissionDenied("credenciais inválidas")

    err = client._classify(exc)

    assert isinstance(err, GcvError)
    assert err.error == "auth_error"
    assert err.message == "credenciais inválidas"
    assert err.secondary == ()


def test_resource_exhausted_classifies_as_quota_exceeded(client: GcvClient) -> None:
    """``ResourceExhausted`` → ``quota_exceeded`` (Requirement 6.6).

    Equivalente gRPC de cota excedida; o ramo HTTP 429 é coberto por
    ``_is_http_429`` em outro teste e por testes do pipeline.
    """

    exc = ResourceExhausted("quota exceeded for project")

    err = client._classify(exc)

    assert err.error == "quota_exceeded"
    assert err.message == "quota exceeded for project"
    assert err.secondary == ()


def test_deadline_exceeded_classifies_as_timeout(client: GcvClient) -> None:
    """``DeadlineExceeded`` → ``timeout`` (Requirement 6.7).

    Forma gRPC do timeout reportado pelo backend GCV. ``_classify``
    deve detectá-la pelo nome de classe em ``_TIMEOUT_CLASS_NAMES``.
    """

    exc = DeadlineExceeded("deadline excedido pelo servidor")

    err = client._classify(exc)

    assert err.error == "timeout"
    assert err.message == "deadline excedido pelo servidor"
    assert err.secondary == ()


def test_concurrent_futures_timeout_classifies_as_timeout(client: GcvClient) -> None:
    """``concurrent.futures.TimeoutError`` → ``timeout`` (Requirement 6.7).

    Timeout local imposto pelo ``ThreadPoolExecutor`` em
    ``_call_api_with_timeout``. Em Python 3.11+ é alias de
    ``TimeoutError`` builtin, mas a classificação cobre o tipo via
    ``isinstance`` direto, independentemente do nome da classe.
    """

    exc = concurrent.futures.TimeoutError("estouro do teto local")

    err = client._classify(exc)

    assert err.error == "timeout"
    assert err.message == "estouro do teto local"
    assert err.secondary == ()


def test_google_api_call_error_classifies_as_generic_error(client: GcvClient) -> None:
    """``GoogleAPICallError`` (sem subclasse específica) → ``generic_error``.

    Classe genérica do ``google.api_core.exceptions`` que NÃO está em
    nenhum dos conjuntos ``_AUTH_CLASS_NAMES``/``_QUOTA_CLASS_NAMES``/
    ``_TIMEOUT_CLASS_NAMES`` e não carrega ``code``/``status_code`` 429.
    Cai no fallback ``generic_error`` definido no fim de ``_classify``.
    """

    exc = GoogleAPICallError("falha de transporte indeterminada")

    err = client._classify(exc)

    assert err.error == "generic_error"
    assert err.message == "falha de transporte indeterminada"
    assert err.secondary == ()


# ---------------------------------------------------------------------------
# Truncamento da mensagem (Requirement 6.2)
# ---------------------------------------------------------------------------


def test_message_truncated_to_500_chars(client: GcvClient) -> None:
    """``GcvError.message`` é truncado em 500 caracteres (Requirement 6.2).

    A constante ``_ERROR_MESSAGE_MAX_LEN`` em ``client.py`` define o
    limite e ``_truncate`` aplica corte puro (sem reticências) para
    preservar o início literal da mensagem original. Validamos com
    uma mensagem de 600 caracteres: o resultado deve ter exatamente
    500 caracteres e coincidir com o prefixo da mensagem original.
    """

    long_message = "x" * 600
    # Usamos ``GoogleAPICallError`` apenas para garantir um caminho de
    # classificação não-vazio; o teste não depende do código primário —
    # o que importa é o tamanho de ``GcvError.message``.
    exc = GoogleAPICallError(long_message)

    err = client._classify(exc)

    assert len(err.message) == 500
    assert err.message == long_message[:500]
