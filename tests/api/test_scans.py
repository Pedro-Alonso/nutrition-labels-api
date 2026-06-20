"""Testes dos endpoints GET/DELETE /me/scans/{scan_id}, paginação e cache-hit.

Cobre:
- GET scan detail encontrado
- GET scan não encontrado → 404
- GET scan de outro usuário → 404
- DELETE scan → 204
- DELETE scan não encontrado → 404
- DELETE scan de outro usuário → 404
- Paginação: página 1 e página 3 (parcial)
- Paginação em lista vazia
- cache_hit quando mesma imagem do mesmo usuário
- sem cache_hit para usuário diferente com mesma imagem
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.models import Scan
from tests.conftest import _insert_scans


async def _get_user_id(client: AsyncClient, token: str) -> str:
    resp = await client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# GET /me/scans/{scan_id}
# ---------------------------------------------------------------------------

async def test_get_scan_detail_returns_result_json(
    client: AsyncClient, auth_token: str, db_session: AsyncSession
) -> None:
    user_id = await _get_user_id(client, auth_token)
    scans = await _insert_scans(db_session, user_id, count=1)

    resp = await client.get(
        f"/api/v1/users/me/scans/{scans[0].id}",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == scans[0].id
    assert "result_json" in data


async def test_get_scan_not_found_returns_404(
    client: AsyncClient, auth_token: str
) -> None:
    resp = await client.get(
        f"/api/v1/users/me/scans/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 404


async def test_get_scan_other_users_scan_returns_404(
    client: AsyncClient,
    auth_token: str,
    auth_token_2: str,
    db_session: AsyncSession,
) -> None:
    user_id_2 = await _get_user_id(client, auth_token_2)
    scans = await _insert_scans(db_session, user_id_2, count=1)

    resp = await client.get(
        f"/api/v1/users/me/scans/{scans[0].id}",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /me/scans/{scan_id}
# ---------------------------------------------------------------------------

async def test_delete_scan_returns_204(
    client: AsyncClient, auth_token: str, db_session: AsyncSession
) -> None:
    user_id = await _get_user_id(client, auth_token)
    scans = await _insert_scans(db_session, user_id, count=1)

    resp = await client.delete(
        f"/api/v1/users/me/scans/{scans[0].id}",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 204

    # confirma que sumiu
    resp2 = await client.get(
        f"/api/v1/users/me/scans/{scans[0].id}",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp2.status_code == 404


async def test_delete_scan_not_found_returns_404(
    client: AsyncClient, auth_token: str
) -> None:
    resp = await client.delete(
        f"/api/v1/users/me/scans/{uuid.uuid4()}",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 404


async def test_delete_scan_other_users_scan_returns_404(
    client: AsyncClient,
    auth_token: str,
    auth_token_2: str,
    db_session: AsyncSession,
) -> None:
    user_id_2 = await _get_user_id(client, auth_token_2)
    scans = await _insert_scans(db_session, user_id_2, count=1)

    resp = await client.delete(
        f"/api/v1/users/me/scans/{scans[0].id}",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Paginação
# ---------------------------------------------------------------------------

async def test_list_scans_pagination_page_1(
    client: AsyncClient, auth_token: str, db_session: AsyncSession
) -> None:
    user_id = await _get_user_id(client, auth_token)
    await _insert_scans(db_session, user_id, count=25)

    resp = await client.get(
        "/api/v1/users/me/scans?page=1&per_page=20",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 25
    assert len(data["items"]) == 20
    assert data["page"] == 1


async def test_list_scans_pagination_page_3(
    client: AsyncClient, auth_token: str, db_session: AsyncSession
) -> None:
    user_id = await _get_user_id(client, auth_token)
    await _insert_scans(db_session, user_id, count=25)

    resp = await client.get(
        "/api/v1/users/me/scans?page=3&per_page=10",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 25
    assert len(data["items"]) == 5


async def test_list_scans_empty(
    client: AsyncClient, auth_token: str
) -> None:
    resp = await client.get(
        "/api/v1/users/me/scans",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []


async def test_list_scans_includes_product_name_and_brand(
    client: AsyncClient, auth_token: str, db_session: AsyncSession
) -> None:
    user_id = await _get_user_id(client, auth_token)
    await _insert_scans(db_session, user_id, count=1, name="Refrigerante Cola", brand="Marca X")

    resp = await client.get(
        "/api/v1/users/me/scans",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["name"] == "Refrigerante Cola"
    assert item["brand"] == "Marca X"


async def test_list_scans_without_product_name_returns_null(
    client: AsyncClient, auth_token: str, db_session: AsyncSession
) -> None:
    user_id = await _get_user_id(client, auth_token)
    await _insert_scans(db_session, user_id, count=1)

    resp = await client.get(
        "/api/v1/users/me/scans",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["name"] is None
    assert item["brand"] is None


# ---------------------------------------------------------------------------
# DELETE /me/scans (clear all)
# ---------------------------------------------------------------------------

async def test_clear_all_scans_returns_204(
    client: AsyncClient, auth_token: str, db_session: AsyncSession
) -> None:
    user_id = await _get_user_id(client, auth_token)
    await _insert_scans(db_session, user_id, count=5)

    resp = await client.delete(
        "/api/v1/users/me/scans",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 204

    resp2 = await client.get(
        "/api/v1/users/me/scans",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["total"] == 0
    assert resp2.json()["items"] == []


async def test_clear_all_scans_empty_history_returns_204(
    client: AsyncClient, auth_token: str
) -> None:
    resp = await client.delete(
        "/api/v1/users/me/scans",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 204


async def test_clear_all_scans_does_not_affect_other_users(
    client: AsyncClient,
    auth_token: str,
    auth_token_2: str,
    db_session: AsyncSession,
) -> None:
    user_id_1 = await _get_user_id(client, auth_token)
    user_id_2 = await _get_user_id(client, auth_token_2)
    await _insert_scans(db_session, user_id_1, count=3, image_hash_prefix="a")
    await _insert_scans(db_session, user_id_2, count=4, image_hash_prefix="b")

    resp = await client.delete(
        "/api/v1/users/me/scans",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 204

    resp2 = await client.get(
        "/api/v1/users/me/scans",
        headers={"Authorization": f"Bearer {auth_token_2}"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["total"] == 4


async def test_clear_all_scans_requires_auth(client: AsyncClient) -> None:
    resp = await client.delete("/api/v1/users/me/scans")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------

async def test_cache_hit_same_user_same_image(
    client: AsyncClient, auth_token: str, db_session: AsyncSession
) -> None:
    """Inserir scan com hash específico; nova requisição com mesma imagem deve retornar cache_hit."""
    import hashlib

    user_id = await _get_user_id(client, auth_token)

    fake_bytes = b"fake-image-data"
    image_hash = hashlib.sha256(fake_bytes).hexdigest()

    cached_scan = Scan(
        id=str(uuid.uuid4()),
        user_id=user_id,
        image_hash=image_hash,
        detected_format="table",
        winning_preset="preset_test",
        passed=True,
        risco_global=None,
        result_json={
            "scan_id": str(uuid.uuid4()),
            "cache_hit": False,
            "detected_format": {"category": "table", "score": 0.9, "grid_density": 0.02, "reasoning": "ok"},
            "winning_preset": "preset_test",
            "winning_attempt_index": 1,
            "passed": True,
            "final_ocr_text": "cached text",
            "final_postprocessed_text": "cached text",
            "attempts": [],
            "ingredient_analysis": None,
        },
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(cached_scan)
    await db_session.commit()

    import io
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("test.jpg", io.BytesIO(fake_bytes), "image/jpeg")},
    )
    # O endpoint retorna 400 para imagem corrompida, mas antes verifica cache.
    # Se o cache existir, retorna antes do OCR.
    assert resp.status_code == 200
    assert resp.json()["cache_hit"] is True


async def test_cache_miss_different_user(
    client: AsyncClient,
    auth_token: str,
    auth_token_2: str,
    db_session: AsyncSession,
) -> None:
    """Cache de um usuário não vaza para outro usuário com mesma imagem."""
    import hashlib
    import io

    fake_bytes = b"fake-image-data-shared"
    image_hash = hashlib.sha256(fake_bytes).hexdigest()

    user_id = await _get_user_id(client, auth_token)

    # Insere cache apenas para o user 1
    cached_scan = Scan(
        id=str(uuid.uuid4()),
        user_id=user_id,
        image_hash=image_hash,
        detected_format="table",
        winning_preset="preset_test",
        passed=True,
        risco_global=None,
        result_json={
            "scan_id": str(uuid.uuid4()),
            "cache_hit": False,
            "detected_format": {"category": "table", "score": 0.9, "grid_density": 0.0, "reasoning": "ok"},
            "winning_preset": "preset_test",
            "winning_attempt_index": 1,
            "passed": True,
            "final_ocr_text": "user1 cache",
            "final_postprocessed_text": "user1 cache",
            "attempts": [],
            "ingredient_analysis": None,
        },
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(cached_scan)
    await db_session.commit()

    # User 2 envia a mesma imagem; deve ir para OCR (não cache do user 1)
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token_2}"},
        files={"file": ("test.jpg", io.BytesIO(fake_bytes), "image/jpeg")},
    )
    # Para user2 não existe cache; a imagem corrompida vai falhar no OCR com 400
    # O importante é que NÃO veio do cache (não é 200 com cache_hit=True)
    assert resp.status_code != 200 or resp.json().get("cache_hit") is not True
