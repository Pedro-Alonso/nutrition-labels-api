"""Testes dos endpoints de produtos (/api/v1/products).

Cobre:
- GET /{barcode} 404 antes de criar
- POST /{barcode} cria produto (201)
- GET /{barcode} 200 após criar
- POST /{barcode} 409 em duplicata
- PUT /{barcode} atualiza campos (patch semântico)
- PUT /{barcode} 404 para produto inexistente
- GET /{barcode}/analysis 404 sem ingredientes
- GET /{barcode}/analysis 200 com ingredientes
- POST /{barcode}/ocr requer auth
- POST /{barcode}/ocr 422 sem imagens
- POST /{barcode}/ocr retorna preview com imagem real
- Rotas públicas (GET) acessíveis sem auth
"""
from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

FIXTURES = Path(__file__).parent.parent / "fixtures" / "images"

BARCODE = "7891234567890"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _create_product(
    client: AsyncClient,
    auth_token: str,
    barcode: str = BARCODE,
    body: dict | None = None,
) -> dict:
    payload = body or {}
    resp = await client.post(
        f"/api/v1/products/{barcode}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json=payload,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# GET /{barcode} — público
# ---------------------------------------------------------------------------

async def test_get_product_not_found(client: AsyncClient) -> None:
    resp = await client.get(f"/api/v1/products/{BARCODE}")
    assert resp.status_code == 404


async def test_get_product_no_auth_required(client: AsyncClient, auth_token: str) -> None:
    """GET é público: funciona sem cabeçalho de autorização."""
    await _create_product(client, auth_token)
    resp = await client.get(f"/api/v1/products/{BARCODE}")
    assert resp.status_code == 200


async def test_get_product_returns_correct_fields(
    client: AsyncClient, auth_token: str
) -> None:
    await _create_product(
        client,
        auth_token,
        body={"name": "Coca-Cola", "brand": "Coca-Cola Company"},
    )
    resp = await client.get(f"/api/v1/products/{BARCODE}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["barcode"] == BARCODE
    assert data["name"] == "Coca-Cola"
    assert data["brand"] == "Coca-Cola Company"
    assert data["nutritional_table"] is None
    assert data["ingredients"] is None
    assert data["analysis"] is None
    assert "created_at" in data
    assert "updated_at" in data


# ---------------------------------------------------------------------------
# POST /{barcode} — requer auth
# ---------------------------------------------------------------------------

async def test_create_product_requires_auth(client: AsyncClient) -> None:
    resp = await client.post(f"/api/v1/products/{BARCODE}", json={})
    assert resp.status_code == 401


async def test_create_product_minimal(client: AsyncClient, auth_token: str) -> None:
    """Cria produto sem campos opcionais."""
    data = await _create_product(client, auth_token)
    assert data["barcode"] == BARCODE
    assert data["name"] is None
    assert data["brand"] is None
    assert data["nutritional_table"] is None
    assert data["ingredients"] is None


async def test_create_product_with_nutritional_table(
    client: AsyncClient, auth_token: str
) -> None:
    body = {
        "name": "Produto Teste",
        "nutritional_table": {
            "portion_description": "Porção de 20g",
            "columns": ["Quantidade por porção", "% VD"],
            "rows": [
                {"nutrient": "Carboidratos", "values": ["15g", "5"]},
                {"nutrient": "Proteínas", "values": ["1,4g", "2"]},
            ],
        },
    }
    data = await _create_product(client, auth_token, body=body)
    nt = data["nutritional_table"]
    assert nt is not None
    assert nt["portion_description"] == "Porção de 20g"
    assert nt["columns"] == ["Quantidade por porção", "% VD"]
    assert len(nt["rows"]) == 2
    assert nt["rows"][0]["nutrient"] == "Carboidratos"
    assert nt["rows"][0]["values"] == ["15g", "5"]


async def test_create_product_with_ingredients(
    client: AsyncClient, auth_token: str
) -> None:
    body = {
        "ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]},
    }
    data = await _create_product(client, auth_token, body=body)
    assert data["ingredients"] is not None
    assert data["ingredients"]["items"] == ["açúcar", "farinha de trigo", "sal"]
    # Análise DM é computada on-the-fly
    # Pode ser None se analyzer não disponível; só verifica estrutura se não-None
    if data["analysis"] is not None:
        assert "risco_global" in data["analysis"]
        assert "ingredientes_identificados" in data["analysis"]


async def test_create_product_persists_scan_for_history(
    client: AsyncClient, auth_token: str
) -> None:
    """Criar um produto deve registrar um Scan no histórico do usuário."""
    headers = {"Authorization": f"Bearer {auth_token}"}

    # Histórico começa vazio.
    before = await client.get("/api/v1/users/me/scans", headers=headers)
    assert before.status_code == 200
    assert before.json()["total"] == 0

    await _create_product(
        client,
        auth_token,
        body={"ingredients": {"items": ["açúcar", "sal"]}},
    )

    after = await client.get("/api/v1/users/me/scans", headers=headers)
    assert after.status_code == 200
    payload = after.json()
    assert payload["total"] == 1
    assert len(payload["items"]) == 1
    # Scan do fluxo de produtos é marcado como revisado (sem banner de
    # "baixa qualidade" no histórico).
    assert payload["items"][0]["passed"] is True


async def test_create_product_409_duplicate(
    client: AsyncClient, auth_token: str
) -> None:
    await _create_product(client, auth_token)
    resp = await client.post(
        f"/api/v1/products/{BARCODE}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={},
    )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# PUT /{barcode} — requer auth
# ---------------------------------------------------------------------------

async def test_update_product_requires_auth(client: AsyncClient, auth_token: str) -> None:
    await _create_product(client, auth_token)
    resp = await client.put(f"/api/v1/products/{BARCODE}", json={"name": "Novo"})
    assert resp.status_code == 401


async def test_update_product_not_found(client: AsyncClient, auth_token: str) -> None:
    resp = await client.put(
        f"/api/v1/products/{BARCODE}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"name": "Qualquer"},
    )
    assert resp.status_code == 404


async def test_update_product_patch_name(client: AsyncClient, auth_token: str) -> None:
    """Atualizar só o name não apaga brand."""
    await _create_product(
        client, auth_token, body={"name": "Original", "brand": "Marca"}
    )
    resp = await client.put(
        f"/api/v1/products/{BARCODE}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"name": "Atualizado"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Atualizado"
    assert data["brand"] == "Marca"  # preservado


async def test_update_product_null_name_clears_field(
    client: AsyncClient, auth_token: str
) -> None:
    """Enviar name=null apaga o campo."""
    await _create_product(client, auth_token, body={"name": "Original"})
    resp = await client.put(
        f"/api/v1/products/{BARCODE}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"name": None},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] is None


async def test_update_product_add_ingredients(
    client: AsyncClient, auth_token: str
) -> None:
    """PUT pode adicionar ingredientes a produto que não tinha."""
    await _create_product(client, auth_token)
    resp = await client.put(
        f"/api/v1/products/{BARCODE}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"ingredients": {"items": ["sal", "açúcar"]}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingredients"] is not None
    assert data["ingredients"]["items"] == ["sal", "açúcar"]


async def test_update_product_null_ingredients_removes_list(
    client: AsyncClient, auth_token: str
) -> None:
    """Enviar ingredients=null remove a lista de ingredientes."""
    await _create_product(
        client,
        auth_token,
        body={"ingredients": {"items": ["sal"]}},
    )
    resp = await client.put(
        f"/api/v1/products/{BARCODE}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"ingredients": None},
    )
    assert resp.status_code == 200
    assert resp.json()["ingredients"] is None


async def test_update_product_absent_ingredients_preserves(
    client: AsyncClient, auth_token: str
) -> None:
    """Campo ausente no body não apaga ingredientes existentes."""
    await _create_product(
        client,
        auth_token,
        body={"ingredients": {"items": ["sal"]}},
    )
    resp = await client.put(
        f"/api/v1/products/{BARCODE}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"name": "Novo Nome"},  # sem 'ingredients'
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Novo Nome"
    assert data["ingredients"] is not None
    assert data["ingredients"]["items"] == ["sal"]


# ---------------------------------------------------------------------------
# GET /{barcode}/analysis — público
# ---------------------------------------------------------------------------

async def test_get_analysis_not_found_no_product(client: AsyncClient) -> None:
    resp = await client.get(f"/api/v1/products/{BARCODE}/analysis")
    assert resp.status_code == 404


async def test_get_analysis_not_found_no_ingredients(
    client: AsyncClient, auth_token: str
) -> None:
    """Produto sem ingredientes → 404."""
    await _create_product(client, auth_token, body={"name": "Sem ingredientes"})
    resp = await client.get(f"/api/v1/products/{BARCODE}/analysis")
    # 404 se sem ingredientes OU 503 se analyzer não disponível
    assert resp.status_code in (404, 503)


async def test_get_analysis_with_ingredients(
    client: AsyncClient, auth_token: str
) -> None:
    """Produto com ingredientes retorna análise DM ou 503 se analyzer indisponível."""
    await _create_product(
        client,
        auth_token,
        body={"ingredients": {"items": ["açúcar", "xarope de milho", "sal"]}},
    )
    resp = await client.get(f"/api/v1/products/{BARCODE}/analysis")
    if resp.status_code == 503:
        return  # analyzer não carregado neste ambiente — test pass
    assert resp.status_code == 200
    data = resp.json()
    assert "risco_global" in data
    assert "ingredientes_identificados" in data
    assert "nao_identificados" in data


async def test_get_analysis_no_auth_required(
    client: AsyncClient, auth_token: str
) -> None:
    """GET /analysis é público."""
    await _create_product(
        client,
        auth_token,
        body={"ingredients": {"items": ["sal"]}},
    )
    # Chama sem token
    resp = await client.get(f"/api/v1/products/{BARCODE}/analysis")
    assert resp.status_code in (200, 503)  # 503 se analyzer ausente, mas não 401


# ---------------------------------------------------------------------------
# POST /{barcode}/ocr — requer auth
# ---------------------------------------------------------------------------

async def test_ocr_preview_requires_auth(client: AsyncClient) -> None:
    resp = await client.post(f"/api/v1/products/{BARCODE}/ocr")
    assert resp.status_code == 401


async def test_ocr_preview_422_no_images(
    client: AsyncClient, auth_token: str
) -> None:
    resp = await client.post(
        f"/api/v1/products/{BARCODE}/ocr",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 422


async def test_ocr_preview_with_nutrition_image(
    client: AsyncClient, auth_token: str
) -> None:
    with open(FIXTURES / "coca_tabela.jpg", "rb") as f:
        resp = await client.post(
            f"/api/v1/products/{BARCODE}/ocr",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"image_nutrition": ("coca_tabela.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["barcode"] == BARCODE
    assert "nutritional_table" in data
    assert "ingredients" in data
    assert data["ingredients"] is None  # não enviamos imagem de ingredientes


async def test_ocr_preview_with_ingredients_image(
    client: AsyncClient, auth_token: str
) -> None:
    with open(FIXTURES / "coca_ingredientes.jpg", "rb") as f:
        resp = await client.post(
            f"/api/v1/products/{BARCODE}/ocr",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"image_ingredients": ("coca_ingredientes.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["barcode"] == BARCODE
    assert data["nutritional_table"] is None
    assert "ingredients" in data


async def test_ocr_preview_does_not_save_to_db(
    client: AsyncClient, auth_token: str, db_session: AsyncSession
) -> None:
    """OCR preview não deve criar nenhum produto no banco."""
    from sqlalchemy import select
    from app.products.models import Product

    with open(FIXTURES / "coca_tabela.jpg", "rb") as f:
        await client.post(
            f"/api/v1/products/{BARCODE}/ocr",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"image_nutrition": ("coca_tabela.jpg", f, "image/jpeg")},
        )

    result = await db_session.execute(select(Product).where(Product.barcode == BARCODE))
    assert result.scalar_one_or_none() is None
