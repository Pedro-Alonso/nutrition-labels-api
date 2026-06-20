"""Testes dos endpoints de autenticação: register, login, refresh, upgrade.

Cobre:
- Register com e-mail novo → 201
- Register com e-mail duplicado → 409
- Register guest com is_guest=true → 201 com is_guest=true
- Login correto → 200 com tokens
- Login com senha errada → 401
- Refresh token válido → 200 novo access token
- Refresh token inválido → 401
- Senha muito curta → 422 (validação Pydantic)
- Upgrade guest → 200 com is_guest=false
- Upgrade non-guest → 403
- Upgrade com email duplicado → 409
- Upgrade com senha curta → 422
- Login após upgrade funciona
- Guest delete account → 204
"""
from __future__ import annotations

from httpx import AsyncClient


async def test_register_new_user(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "novo@example.com", "password": "senha12345", "display_name": "Novo"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["email"] == "novo@example.com"
    assert data["display_name"] == "Novo"
    assert "id" in data
    assert "password_hash" not in data


async def test_register_duplicate_email(client: AsyncClient) -> None:
    body = {"email": "dup@example.com", "password": "senha12345"}
    r1 = await client.post("/api/v1/auth/register", json=body)
    assert r1.status_code == 201

    r2 = await client.post("/api/v1/auth/register", json=body)
    assert r2.status_code == 409


async def test_register_without_display_name(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "nodisplay@example.com", "password": "senha12345"},
    )
    assert resp.status_code == 201
    assert resp.json()["display_name"] is None


async def test_register_persists_profile_fields(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "perfil@example.com",
            "password": "senha12345",
            "diabetes_type": "DM2",
            "language_level": "leigo",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["diabetes_type"] == "DM2"
    assert data["language_level"] == "leigo"

    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "perfil@example.com", "password": "senha12345"},
    )
    access = login.json()["access_token"]
    me = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {access}"},
    )
    assert me.status_code == 200
    assert me.json()["diabetes_type"] == "DM2"
    assert me.json()["language_level"] == "leigo"


async def test_login_success(client: AsyncClient, test_user: dict) -> None:
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": test_user["email"], "password": test_user["password"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


async def test_login_wrong_password(client: AsyncClient, test_user: dict) -> None:
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": test_user["email"], "password": "senhaerrada"},
    )
    assert resp.status_code == 401


async def test_login_unknown_email(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "naoexiste@example.com", "password": "senha12345"},
    )
    assert resp.status_code == 401


async def test_refresh_valid_token(client: AsyncClient, test_user: dict) -> None:
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": test_user["email"], "password": test_user["password"]},
    )
    refresh_token = login.json()["refresh_token"]

    resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


async def test_refresh_invalid_token(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "token.invalido.aqui"},
    )
    assert resp.status_code == 401


async def test_register_short_password(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "short@example.com", "password": "abc"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Guest mode tests
# ---------------------------------------------------------------------------


async def _register_guest(client: AsyncClient) -> tuple[dict, str]:
    """Helper: register a guest user and return (response_data, access_token)."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "guest_abc@guest.local",
            "password": "randompass12345678",
            "display_name": "guest12345",
            "is_guest": True,
        },
    )
    assert resp.status_code == 201
    data = resp.json()

    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "guest_abc@guest.local", "password": "randompass12345678"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    return data, token


async def test_register_guest(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "guest_reg@guest.local",
            "password": "randompass12345678",
            "display_name": "guest99999",
            "is_guest": True,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["is_guest"] is True
    assert data["email"] == "guest_reg@guest.local"
    assert data["display_name"] == "guest99999"


async def test_upgrade_guest_success(client: AsyncClient) -> None:
    _, token = await _register_guest(client)

    resp = await client.post(
        "/api/v1/auth/upgrade",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "email": "upgraded@example.com",
            "password": "realpass12345",
            "display_name": "Real User",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_guest"] is False
    assert data["email"] == "upgraded@example.com"
    assert data["display_name"] == "Real User"


async def test_upgrade_non_guest_forbidden(client: AsyncClient, auth_token: str) -> None:
    resp = await client.post(
        "/api/v1/auth/upgrade",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"email": "new@example.com", "password": "newpass12345"},
    )
    assert resp.status_code == 403


async def test_upgrade_duplicate_email(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "taken@example.com", "password": "senha12345"},
    )

    _, token = await _register_guest(client)

    resp = await client.post(
        "/api/v1/auth/upgrade",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": "taken@example.com", "password": "newpass12345"},
    )
    assert resp.status_code == 409


async def test_upgrade_short_password(client: AsyncClient) -> None:
    _, token = await _register_guest(client)

    resp = await client.post(
        "/api/v1/auth/upgrade",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": "up@example.com", "password": "short"},
    )
    assert resp.status_code == 422


async def test_login_after_upgrade(client: AsyncClient) -> None:
    _, token = await _register_guest(client)

    await client.post(
        "/api/v1/auth/upgrade",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": "afterup@example.com", "password": "afteruppass123"},
    )

    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "afterup@example.com", "password": "afteruppass123"},
    )
    assert login.status_code == 200
    assert "access_token" in login.json()


async def test_guest_delete_account(client: AsyncClient) -> None:
    _, token = await _register_guest(client)

    resp = await client.delete(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 204
