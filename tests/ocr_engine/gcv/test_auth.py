"""Testes exemplo da ordem de resolução de credenciais GCV.

Cobre os três caminhos descritos em ``ocr/cloud_vision/auth.py`` e nos
Requirements 5.1–5.3:

1. ``config.credentials_path`` aponta para arquivo existente → função
   retorna esse ``Path`` sem consultar a variável de ambiente.
2. ``config.credentials_path`` é ``None`` (ou inválido) e a variável de
   ambiente ``GOOGLE_APPLICATION_CREDENTIALS`` aponta para arquivo
   existente → função retorna o ``Path`` da variável de ambiente.
3. Nenhuma das duas fontes resolve um arquivo válido → função levanta
   ``GcvError(error="auth_error", ...)``.

Os testes usam ``tmp_path`` do pytest para criar Service Accounts falsos
em disco e ``monkeypatch`` para isolar ``os.environ`` (a função
``resolve_credentials`` usa ``os.environ`` por default — Requirement
5.2).

Seguindo as convenções do projeto (``AGENTS.md``):
- Comentários e docstrings em português; identificadores em inglês.
- Imports absolutos a partir da raiz do projeto.
- Tipos modernos via ``from __future__ import annotations``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.auth import resolve_credentials
from ocr.cloud_vision.types import GcvError


def _build_config(
    credentials_path: str | None,
    project_root: Path,
) -> GcvAppConfig:
    """Constrói uma ``GcvAppConfig`` mínima para os testes.

    Apenas o campo ``credentials_path`` é relevante em
    ``resolve_credentials``; os demais ficam em seus defaults
    documentados (Requirement 4.2). Usa ``GcvAppConfig.from_dict`` para
    garantir que a normalização de paths relativos vs absolutos siga o
    mesmo caminho da configuração real lida de ``app.json``.
    """

    return GcvAppConfig.from_dict(
        {"credentials_path": credentials_path},
        project_root=project_root,
    )


def test_resolve_uses_config_credentials_path_when_file_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caminho 1 (Requirement 5.1): config válido tem prioridade absoluta.

    Quando ``config.credentials_path`` aponta para um arquivo existente,
    ``resolve_credentials`` deve devolvê-lo SEM consultar a variável
    de ambiente — mesmo que ela também esteja apontando para outro
    arquivo válido. Garantimos isso definindo um caminho de env var
    inexistente: se a função (incorretamente) priorizasse o env, o
    teste falharia ao não receber o arquivo do config.
    """

    sa_from_config = tmp_path / "config_sa.json"
    sa_from_config.write_text("{}", encoding="utf-8")

    # Env var é definida apontando para arquivo INEXISTENTE para
    # provar que a função não cai no fallback quando o config já
    # resolve. Caso a precedência fosse invertida, o teste capturaria
    # o erro porque o env apontaria para um path inválido.
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(tmp_path / "does_not_exist.json"),
    )

    config = _build_config(str(sa_from_config), project_root=tmp_path)

    resolved = resolve_credentials(config, project_root=tmp_path)

    assert resolved == sa_from_config


def test_resolve_falls_back_to_env_var_when_config_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caminho 2 (Requirement 5.2): fallback para env var.

    Quando ``config.credentials_path`` é ``None`` (ou inválido),
    ``resolve_credentials`` deve consultar
    ``GOOGLE_APPLICATION_CREDENTIALS``. Se a variável aponta para um
    arquivo existente, esse ``Path`` é o resultado.
    """

    sa_from_env = tmp_path / "env_sa.json"
    sa_from_env.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_from_env))

    config = _build_config(credentials_path=None, project_root=tmp_path)

    resolved = resolve_credentials(config, project_root=tmp_path)

    assert resolved == sa_from_env


def test_resolve_raises_auth_error_when_no_source_resolves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caminho 3 (Requirement 5.3): falha total → ``GcvError``.

    Quando nem ``config.credentials_path`` nem
    ``GOOGLE_APPLICATION_CREDENTIALS`` apontam para um arquivo
    existente, ``resolve_credentials`` deve levantar
    ``GcvError(error="auth_error", ...)`` para que o pipeline aplique
    a política de ``on_failure``.

    Aqui também removemos explicitamente a variável de ambiente
    ``GOOGLE_APPLICATION_CREDENTIALS`` (caso esteja set no host real),
    isolando o teste de qualquer leak de configuração externa.
    """

    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    config = _build_config(credentials_path=None, project_root=tmp_path)

    with pytest.raises(GcvError) as excinfo:
        resolve_credentials(config, project_root=tmp_path)

    assert excinfo.value.error == "auth_error"
