"""Fixtures globais para todos os testes da API.

Garante que ocr_engine/ esteja em sys.path (via importação do pacote)
antes que qualquer teste da suíte GCV seja coletado.

Estratégia de isolamento de banco:
- NullPool no engine de teste: cada conexão é criada e fechada sem pooling,
  evitando o erro "cannot use Connection.transaction() in a manually started
  transaction" do asyncpg quando pool_pre_ping=True encontra conexões sujas.
- clean_db autouse trunca as tabelas antes de cada teste.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import ocr_engine  # noqa: F401 — garante que ocr_engine/ entre em sys.path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.analysis.models import Scan
from app.core.config import get_settings
from app.core.database import get_db
from app.main import app

# Engine de teste com NullPool — sem compartilhamento de conexões entre fixtures
_test_engine = None


def _get_test_engine():
    global _test_engine
    if _test_engine is None:
        settings = get_settings()
        _test_engine = create_async_engine(
            settings.database_url,
            echo=False,
            poolclass=NullPool,
        )
    return _test_engine


# ---------------------------------------------------------------------------
# Client de sessão: carrega o reader OCR apenas uma vez por sessão
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def session_client() -> AsyncClient:
    """AsyncClient para cada teste.

    ASGITransport do httpx não executa o lifespan ASGI, portanto o
    NutritionReader é carregado diretamente se ainda não foi inicializado.
    build_reader() é rápido (carrega JSONs), então o overhead é aceitável.
    """
    if not hasattr(app.state, "reader") or app.state.reader is None:
        from ocr_engine import build_reader
        app.state.reader = build_reader()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Limpeza de banco entre testes
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(autouse=True)
async def clean_db() -> None:
    """Trunca as tabelas antes de cada teste para isolamento completo."""
    async with _get_test_engine().begin() as conn:
        await conn.execute(text("TRUNCATE TABLE revoked_tokens, scans, product_summaries, ingredient_lists, nutritional_tables, products, users RESTART IDENTITY CASCADE"))


# ---------------------------------------------------------------------------
# Sessão de banco por teste
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """AsyncSession isolada por teste via NullPool (sem reúso de conexão)."""
    factory = async_sessionmaker(bind=_get_test_engine(), expire_on_commit=False)
    async with factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Client por teste: injeta a sessão via override de dependência
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client(session_client: AsyncClient, db_session: AsyncSession) -> AsyncClient:
    """AsyncClient por teste com override de get_db apontando para a sessão de teste."""

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


@pytest_asyncio.fixture
async def test_user_2(client: AsyncClient) -> dict:
    """Segundo usuário de teste para verificar isolamento entre usuários."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "other@example.com",
            "password": "outrosenha12345",
            "display_name": "Other User",
        },
    )
    assert resp.status_code == 201, f"Falha ao criar segundo usuário: {resp.text}"
    return {"email": "other@example.com", "password": "outrosenha12345"}


@pytest_asyncio.fixture
async def auth_token_2(client: AsyncClient, test_user_2: dict) -> str:
    """JWT access token do segundo usuário de teste."""
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": test_user_2["email"], "password": test_user_2["password"]},
    )
    assert resp.status_code == 200, f"Falha ao fazer login do segundo usuário: {resp.text}"
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Helper para inserir scans diretamente no banco (sem OCR)
# ---------------------------------------------------------------------------

async def _insert_scans(
    db: AsyncSession,
    user_id: str,
    count: int,
    image_hash_prefix: str = "",
) -> list[Scan]:
    """Insere `count` scans para `user_id` diretamente via ORM, sem OCR."""
    scans: list[Scan] = []
    for i in range(count):
        scan = Scan(
            id=str(uuid.uuid4()),
            user_id=user_id,
            image_hash=f"{image_hash_prefix}{i:064d}",
            detected_format="table",
            winning_preset="preset_test",
            passed=True,
            risco_global=None,
            result_json={
                "scan_id": str(uuid.uuid4()),
                "cache_hit": False,
                "detected_format": {"category": "table", "score": 0.9, "grid_density": 0.02, "reasoning": "test"},
                "winning_preset": "preset_test",
                "winning_attempt_index": 1,
                "passed": True,
                "final_ocr_text": f"scan {i}",
                "final_postprocessed_text": f"scan {i}",
                "attempts": [],
                "ingredient_analysis": None,
            },
            created_at=datetime.now(timezone.utc),
        )
        db.add(scan)
        scans.append(scan)
    await db.commit()
    return scans
