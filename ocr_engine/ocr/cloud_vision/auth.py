"""Resolução de credenciais Service Account para a Google Cloud Vision API.

Implementa o fallback documentado em Requirements 5.1–5.4: primeiro tenta
``gcv.credentials_path`` declarado em ``config/app.json``; em seguida, a
variável de ambiente ``GOOGLE_APPLICATION_CREDENTIALS`` (padrão Google);
caso nenhuma fonte resolva um arquivo existente, levanta
``GcvError(error="auth_error", ...)`` para que o ``CloudVisionPipeline``
aplique a política de ``on_failure``.

A função NUNCA lê o conteúdo do Service Account — apenas verifica
existência via ``Path.is_file()``. A leitura efetiva fica a cargo do SDK
``google.oauth2.service_account.Credentials.from_service_account_file``,
o que satisfaz Requirement 5.4 (sem Service Account em logs ou artefatos).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from ocr.cloud_vision.types import GcvError

if TYPE_CHECKING:  # pragma: no cover - import só para análise estática
    # Evita import circular em tempo de execução: o módulo ``app_config`` é
    # construído em paralelo e a função apenas lê ``config.credentials_path``,
    # caracterizando duck typing nativo.
    from ocr.cloud_vision.app_config import GcvAppConfig


# Mensagem canônica usada no caminho de falha (Requirement 5.3). Definida como
# constante para facilitar matching exato em testes e localização.
_AUTH_ERROR_MESSAGE = (
    "credenciais ausentes: gcv.credentials_path inválido e "
    "GOOGLE_APPLICATION_CREDENTIALS ausente ou inválido"
)


def resolve_credentials(
    config: "GcvAppConfig",
    project_root: Path,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve o ``Path`` do Service Account a ser usado pelo ``GcvClient``.

    A ordem de resolução segue exatamente Requirements 5.1–5.3:

    1. ``config.credentials_path`` quando truthy e o arquivo existe. Paths
       relativos são resolvidos contra ``project_root``.
    2. ``env["GOOGLE_APPLICATION_CREDENTIALS"]`` (default ``os.environ``)
       quando definida e o arquivo existe.
    3. Caso nenhuma fonte resolva, levanta ``GcvError(error="auth_error")``
       com mensagem canônica.

    Args:
        config: Configuração já parseada de ``app.json::gcv``. A função
            consome apenas o atributo ``credentials_path`` (duck typing).
        project_root: Diretório raiz do projeto, usado para ancorar paths
            relativos declarados em ``config.credentials_path``.
        env: Mapping de variáveis de ambiente. Default ``os.environ`` é
            resolvido em tempo de chamada (não no momento de definição) para
            que mudanças via ``monkeypatch`` em testes sejam observadas.

    Returns:
        Caminho absoluto (ou tal-como-fornecido) para o arquivo do Service
        Account validado por ``Path.is_file()``.

    Raises:
        GcvError: Quando nem ``config.credentials_path`` nem
            ``GOOGLE_APPLICATION_CREDENTIALS`` apontam para um arquivo
            existente. ``error == "auth_error"``.
    """

    # Resolução tardia de ``os.environ``: replicar o estado vigente no momento
    # da chamada e permitir injeção em testes (Requirement 5.2).
    effective_env: Mapping[str, str] = env if env is not None else os.environ

    # 1) Caminho explícito vindo da configuração tem prioridade absoluta.
    config_path = config.credentials_path
    if config_path:
        candidate = Path(config_path)
        if not candidate.is_absolute():
            # Paths relativos são ancorados a ``project_root`` e normalizados
            # via ``.resolve()`` para alinhar com o restante do projeto
            # (ver ``AGENTS.md``: Path handling).
            candidate = (project_root / candidate).resolve()
        if candidate.is_file():
            return candidate

    # 2) Fallback para o padrão Google via variável de ambiente.
    env_path = effective_env.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path:
        env_candidate = Path(env_path)
        if env_candidate.is_file():
            return env_candidate

    # 3) Nenhuma fonte resolveu — sinaliza falha classificada para que o
    # ``GcvClient`` (ou o ``CloudVisionPipeline``) aplique ``on_failure``.
    raise GcvError(error="auth_error", message=_AUTH_ERROR_MESSAGE)
