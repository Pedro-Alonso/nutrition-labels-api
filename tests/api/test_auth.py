"""Testes dos endpoints de autenticação: register, login, refresh.

Cobre:
- Register com e-mail novo → 201
- Register com e-mail duplicado → 409
- Login correto → 200 com tokens
- Login com senha errada → 401
- Refresh token válido → 200 novo access token
- Refresh token inválido → 401
- Senha muito curta → 422 (validação Pydantic)
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
