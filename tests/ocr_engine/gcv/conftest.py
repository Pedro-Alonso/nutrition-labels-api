"""Fixtures base para a suíte de testes da feature GCV OCR Preset.

Este módulo concentra fixtures pytest comuns a vários testes da feature:

- ``tmp_project_root``: cria uma raiz de projeto temporária com a
  estrutura mínima esperada pelo ``NutritionReader`` e pelo
  ``AuditRecorder`` (``extractions/``, ``images/pipeline/``,
  ``subjects/``).
- ``gcv_app_config_default``: devolve um dicionário com os defaults
  documentados do bloco ``gcv`` em ``config/app.json``. A escolha por
  retornar um ``dict`` (em vez de uma instância de ``GcvAppConfig``) é
  deliberada: a classe ``GcvAppConfig`` ainda não existe nesta fase do
  plano de implementação (será introduzida na task 2.3). Quando a classe
  estiver disponível, os testes consumidores podem simplesmente fazer
  ``GcvAppConfig.from_dict(gcv_app_config_default, project_root)`` sem
  precisar atualizar a fixture.
- ``fake_service_account_path``: grava em disco um arquivo JSON falso
  que NÃO segue o schema real de Service Account do Google. O conteúdo
  embute um marcador textual único (``GCV_TEST_MARKER_<uuid>``) que
  permite verificar, em testes de privacidade (P15), que nenhum
  artefato gravado pelo pipeline contém o marcador — provando que o
  conteúdo do Service Account jamais vaza para ``extractions/`` ou
  ``images/pipeline/``.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest


# Prefixo fixo do marcador de privacidade. O sufixo único é gerado a
# cada invocação de ``fake_service_account_path``. Testes podem usar a
# função ``read_marker_from_service_account`` para recuperar o marcador
# corrente sem precisar passar o valor explicitamente.
SERVICE_ACCOUNT_MARKER_PREFIX = "GCV_TEST_MARKER_"
_MARKER_PATTERN = re.compile(rf"{SERVICE_ACCOUNT_MARKER_PREFIX}[0-9a-f]+")


def read_marker_from_service_account(sa_path: Path) -> str:
    """Lê o marcador de privacidade gravado em um Service Account falso.

    O marcador segue o padrão ``GCV_TEST_MARKER_<hex>`` e está embutido
    no campo ``private_key`` do JSON. Esta função é usada por testes
    de privacidade (Requirement 5.4) que precisam buscar o marcador em
    artefatos sem assumir conhecimento prévio do sufixo único.
    """

    content = sa_path.read_text(encoding="utf-8")
    match = _MARKER_PATTERN.search(content)
    if match is None:  # pragma: no cover - guarda defensiva
        raise AssertionError(
            f"Marcador de privacidade não encontrado em {sa_path}"
        )
    return match.group(0)


# ---------------------------------------------------------------------------
# Estrutura de diretórios de projeto
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project_root(tmp_path: Path) -> Path:
    """Cria uma raiz de projeto temporária com a estrutura mínima.

    A árvore resultante reproduz apenas as pastas indispensáveis ao
    fluxo do ``NutritionReader`` durante os testes:

    .. code-block::

        <tmp>/
        ├── extractions/
        ├── images/
        │   └── pipeline/
        └── subjects/

    Pastas como ``config/`` ou ``subjects_groundtruth/`` são criadas
    sob demanda pelos próprios testes que precisarem delas, mantendo
    esta fixture enxuta e estável.
    """

    (tmp_path / "extractions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "pipeline").mkdir(parents=True, exist_ok=True)
    (tmp_path / "subjects").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Configuração default do bloco ``gcv`` em ``app.json``
# ---------------------------------------------------------------------------

@pytest.fixture
def gcv_app_config_default(tmp_project_root: Path) -> dict[str, Any]:
    """Retorna o dicionário default para o bloco ``gcv`` em ``app.json``.

    Os valores espelham os defaults documentados no design (Requirement
    4.2): credenciais ausentes, modo de falha ``skip``, cache habilitado
    apontando para ``<project_root>/extractions/.gcv_cache``, sem rate
    limiting e timeout de 30 segundos.

    O ``cache_dir`` é resolvido como caminho absoluto ancorado em
    ``tmp_project_root`` para que testes que materializam o cache em
    disco fiquem isolados entre si — independentemente de como
    ``GcvAppConfig.from_dict`` venha a tratar paths relativos no
    futuro.
    """

    return {
        "credentials_path": None,
        "on_failure": "skip",
        "cache_enabled": True,
        "cache_dir": str(tmp_project_root / "extractions" / ".gcv_cache"),
        "max_requests_per_minute": None,
        "request_timeout_seconds": 30,
    }


# ---------------------------------------------------------------------------
# Service Account falso com marcador de privacidade
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_service_account_path(tmp_project_root: Path) -> Path:
    """Grava um Service Account falso com marcador único de privacidade.

    O arquivo gerado tem extensão ``.json`` e contém um payload mínimo
    com um campo ``private_key`` cujo valor é uma string contendo o
    marcador ``GCV_TEST_MARKER_<uuid>``. O marcador é regenerado a cada
    invocação da fixture para reduzir a chance de colisão com strings
    legítimas presentes em outros artefatos.

    Uso típico em testes de privacidade (Requirement 5.4 / Property
    15): após executar o pipeline, varre-se ``extractions/`` e
    ``images/pipeline/`` em busca do marcador; sua ausência é
    evidência de que o conteúdo do Service Account não foi exposto em
    logs, ``_summary.json`` ou outros artefatos.
    """

    marker = f"{SERVICE_ACCOUNT_MARKER_PREFIX}{uuid.uuid4().hex}"
    sa_path = tmp_project_root / "fake_service_account.json"
    payload = {
        "type": "service_account",
        "project_id": "fake-project",
        "private_key_id": "fake-key-id",
        "private_key": (
            "-----BEGIN PRIVATE KEY-----\n"
            f"{marker}\n"
            "-----END PRIVATE KEY-----\n"
        ),
        "client_email": "fake@fake-project.iam.gserviceaccount.com",
        "client_id": "000000000000000000000",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    sa_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Não anexamos o marcador como atributo dinâmico ao ``Path`` porque
    # ``pathlib.Path`` usa ``__slots__`` em Python 3.11+. Testes que
    # precisam do valor exato do marcador devem usar
    # ``read_marker_from_service_account(sa_path)``.
    return sa_path
