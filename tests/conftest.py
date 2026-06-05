"""Fixtures globais para todos os testes da API.

Garante que ocr_engine/ esteja em sys.path (via importação do pacote)
antes que qualquer teste da suíte GCV seja coletado.
"""
from __future__ import annotations

import ocr_engine  # noqa: F401 — garante que ocr_engine/ entre em sys.path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, get_engine
from app.main import app


# ---------------------------------------------------------------------------
# Client de sessão: lifespan (e build_reader) é invocado apenas uma vez
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def session_client() -> AsyncClient:
    """AsyncClient que compartilha o lifespan da app por toda a sessão.

    O NutritionReader é carregado uma única vez no startup — invocá-lo
    por teste tornaria a suíte impraticavelmente lenta.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Sessão de banco com rollback automático
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Sessão de banco isolada por teste via savepoint + rollback.

    Usa ``join_transaction_mode="create_savepoint"`` para que qualquer
    ``await session.commit()`` dentro do código da app crie um savepoint
    em vez de commitar a transação externa — garantindo que todos os
    dados sejam revertidos ao fim de cada teste.
    """
    engine = get_engine()
    conn = await engine.connect()
    trans = await conn.begin()
    session = AsyncSession(
        bind=conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    yield session
    await session.close()
    await trans.rollback()
    await conn.close()


# ---------------------------------------------------------------------------
# Client por teste: injeta a sessão isolada via override de dependência
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client(session_client: AsyncClient, db_session: AsyncSession) -> AsyncClient:
    """AsyncClient por teste com override de get_db apontando para a sessão isolada."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield session_client
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Usuário de teste e token JWT
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_user(client: AsyncClient) -> dict:
    """Registra um usuário de teste e retorna suas credenciais."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "senha12345",
            "display_name": "Test User",
        },
    )
    assert resp.status_code == 201, f"Falha ao criar usuário de teste: {resp.text}"
    return {"email": "test@example.com", "password": "senha12345"}


@pytest_asyncio.fixture
async def auth_token(client: AsyncClient, test_user: dict) -> str:
    """JWT access token do usuário de teste."""
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": test_user["email"], "password": test_user["password"]},
    )
    assert resp.status_code == 200, f"Falha ao fazer login: {resp.text}"
    return resp.json()["access_token"]
