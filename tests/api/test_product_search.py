from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import _insert_product


async def test_search_empty_query_returns_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/products/search?q=")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


async def test_search_short_query_returns_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/products/search?q=a")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


async def test_search_by_name_finds_product(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _insert_product(db_session, "7891000100103", name="Refrigerante de Cola", brand="MarcaX")

    resp = await client.get("/api/v1/products/search?q=cola")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    barcodes = [item["barcode"] for item in data["items"]]
    assert "7891000100103" in barcodes


async def test_search_by_brand_finds_product(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _insert_product(db_session, "7891000100110", name="Refrigerante", brand="Coca-Cola")

    resp = await client.get("/api/v1/products/search?q=coca")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    barcodes = [item["barcode"] for item in data["items"]]
    assert "7891000100110" in barcodes


async def test_search_accent_insensitive(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _insert_product(db_session, "7891000100127", name="Açúcar Mascavo")

    resp = await client.get("/api/v1/products/search?q=acucar")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    barcodes = [item["barcode"] for item in data["items"]]
    assert "7891000100127" in barcodes


async def test_search_barcode_exact_match(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _insert_product(db_session, "7891234567890", name="Produto Teste")

    resp = await client.get("/api/v1/products/search?q=7891234567890")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["barcode"] == "7891234567890"


async def test_search_barcode_not_found(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/products/search?q=7891234567890")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []


async def test_search_numeric_non_barcode_length(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _insert_product(db_session, "7891000100134", name="Produto 12345")

    resp = await client.get("/api/v1/products/search?q=12345")
    assert resp.status_code == 200
    # 5 digits is not a barcode length, so it runs name search


async def test_search_pagination(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    for i in range(25):
        await _insert_product(
            db_session,
            f"{7890000000000 + i}",
            name=f"Produto Teste {i:03d}",
            brand="MarcaComum",
        )

    resp = await client.get("/api/v1/products/search?q=MarcaComum&per_page=10&page=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 25
    assert len(data["items"]) == 10
    assert data["page"] == 1
    assert data["per_page"] == 10


async def test_search_no_auth_required(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/products/search?q=test")
    assert resp.status_code == 200
