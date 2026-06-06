"""Testes unitários da camada de serviço de usuários.

Cobre:
- get_user_by_id encontrado e não encontrado
- update_user display_name
- list_user_scans vazia, paginação na segunda página
- get_scan_by_id encontrado e usuário errado
- delete_scan encontrado, não encontrado e usuário errado
- delete_user com cascata de scans
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.models import Scan
from app.users.models import User
from app.users import service as user_service
from tests.conftest import _insert_scans


async def _create_user(db: AsyncSession, email: str = "u@test.com") -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        password_hash="hash",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# get_user_by_id
# ---------------------------------------------------------------------------

async def test_get_user_by_id_found(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    found = await user_service.get_user_by_id(db_session, user.id)
    assert found is not None
    assert found.id == user.id


async def test_get_user_by_id_not_found(db_session: AsyncSession) -> None:
    found = await user_service.get_user_by_id(db_session, str(uuid.uuid4()))
    assert found is None


# ---------------------------------------------------------------------------
# update_user
# ---------------------------------------------------------------------------

async def test_update_user_display_name(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    updated = await user_service.update_user(db_session, user, display_name="Novo Nome")
    assert updated.display_name == "Novo Nome"


# ---------------------------------------------------------------------------
# list_user_scans
# ---------------------------------------------------------------------------

async def test_list_user_scans_empty_returns_zero_total(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    scans, total = await user_service.list_user_scans(db_session, user.id)
    assert total == 0
    assert scans == []


async def test_list_user_scans_pagination_second_page(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    await _insert_scans(db_session, user.id, count=25)

    scans, total = await user_service.list_user_scans(db_session, user.id, page=2, per_page=20)
    assert total == 25
    assert len(scans) == 5


# ---------------------------------------------------------------------------
# get_scan_by_id
# ---------------------------------------------------------------------------

async def test_get_scan_by_id_found(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    inserted = await _insert_scans(db_session, user.id, count=1)
    scan = await user_service.get_scan_by_id(db_session, inserted[0].id, user.id)
    assert scan is not None
    assert scan.id == inserted[0].id


async def test_get_scan_by_id_wrong_user_returns_none(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    other = await _create_user(db_session, email="other@test.com")
    inserted = await _insert_scans(db_session, user.id, count=1)
    scan = await user_service.get_scan_by_id(db_session, inserted[0].id, other.id)
    assert scan is None


# ---------------------------------------------------------------------------
# delete_scan
# ---------------------------------------------------------------------------

async def test_delete_scan_returns_true(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    inserted = await _insert_scans(db_session, user.id, count=1)
    result = await user_service.delete_scan(db_session, inserted[0].id, user.id)
    assert result is True


async def test_delete_scan_not_found_returns_false(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    result = await user_service.delete_scan(db_session, str(uuid.uuid4()), user.id)
    assert result is False


async def test_delete_scan_wrong_user_returns_false(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    other = await _create_user(db_session, email="other@test.com")
    inserted = await _insert_scans(db_session, user.id, count=1)
    result = await user_service.delete_scan(db_session, inserted[0].id, other.id)
    assert result is False


# ---------------------------------------------------------------------------
# delete_user
# ---------------------------------------------------------------------------

async def test_delete_user_cascades_scans(db_session: AsyncSession) -> None:
    user = await _create_user(db_session)
    await _insert_scans(db_session, user.id, count=3)

    await user_service.delete_user(db_session, user.id)

    deleted_user = await user_service.get_user_by_id(db_session, user.id)
    assert deleted_user is None

    scans, total = await user_service.list_user_scans(db_session, user.id)
    assert total == 0
