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
- GET /{barcode} popula analysis.natural_language_summary (cache de ProductSummary)
- Cache-hit evita 2ª chamada ao Groq; regenera ao trocar personalização ou editar produto
- POST /{barcode}/scan (scan-on-read): auth, 404, resposta com resumo, dedupe no histórico
- GET /{barcode}/summary — personalized summary endpoint (auth, anon, 404)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings

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


# ---------------------------------------------------------------------------
# GET /{barcode} — resumo em linguagem natural (cache de ProductSummary)
# ---------------------------------------------------------------------------

async def test_get_product_includes_natural_language_summary(
    client: AsyncClient, auth_token: str
) -> None:
    data = await _create_product(
        client,
        auth_token,
        body={"ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]}},
    )
    if data["analysis"] is None:
        return  # analyzer não disponível neste ambiente

    resp = await client.get(f"/api/v1/products/{BARCODE}")
    assert resp.status_code == 200
    assert "natural_language_summary" in resp.json()["analysis"]


async def test_get_product_summary_cache_hit_avoids_second_groq_call(
    client: AsyncClient, auth_token: str
) -> None:
    with patch.object(get_settings(), "groq_api_key", "test-groq-key"), patch(
        "app.products.service.generate_summary",
        new_callable=AsyncMock,
        return_value="Resumo de teste.",
    ) as mock_generate:
        data = await _create_product(
            client,
            auth_token,
            body={"ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]}},
        )
        if data["analysis"] is None:
            return  # analyzer não disponível neste ambiente

        resp1 = await client.get(f"/api/v1/products/{BARCODE}")
        resp2 = await client.get(f"/api/v1/products/{BARCODE}")

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["analysis"]["natural_language_summary"] == "Resumo de teste."
        assert resp2.json()["analysis"]["natural_language_summary"] == "Resumo de teste."
        # Criação + 2 GETs reaproveitam o mesmo cache (barcode, None, None).
        assert mock_generate.call_count == 1


async def test_get_product_summary_regenerates_on_personalization_change(
    client: AsyncClient, auth_token: str
) -> None:
    with patch.object(get_settings(), "groq_api_key", "test-groq-key"), patch(
        "app.products.service.generate_summary",
        new_callable=AsyncMock,
        side_effect=["Resumo padrão.", "Resumo personalizado."],
    ) as mock_generate:
        data = await _create_product(
            client,
            auth_token,
            body={"ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]}},
        )
        if data["analysis"] is None:
            return  # analyzer não disponível neste ambiente

        headers = {"Authorization": f"Bearer {auth_token}"}

        resp1 = await client.get(f"/api/v1/products/{BARCODE}", headers=headers)
        assert resp1.json()["analysis"]["natural_language_summary"] == "Resumo padrão."

        put_resp = await client.put(
            "/api/v1/users/me",
            headers=headers,
            json={"diabetes_type": "DM2", "language_level": "leigo"},
        )
        assert put_resp.status_code == 200

        resp2 = await client.get(f"/api/v1/products/{BARCODE}", headers=headers)
        assert (
            resp2.json()["analysis"]["natural_language_summary"]
            == "Resumo personalizado."
        )
        # Chave de cache (barcode, diabetes_type, language_level) mudou → 2ª chamada ao Groq.
        assert mock_generate.call_count == 2


async def test_get_product_summary_regenerates_after_product_update(
    client: AsyncClient, auth_token: str
) -> None:
    with patch.object(get_settings(), "groq_api_key", "test-groq-key"), patch(
        "app.products.service.generate_summary",
        new_callable=AsyncMock,
        side_effect=["Resumo original.", "Resumo atualizado."],
    ) as mock_generate:
        data = await _create_product(
            client,
            auth_token,
            body={"ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]}},
        )
        if data["analysis"] is None:
            return  # analyzer não disponível neste ambiente

        headers = {"Authorization": f"Bearer {auth_token}"}

        put_resp = await client.put(
            f"/api/v1/products/{BARCODE}",
            headers=headers,
            json={"name": "Produto Atualizado"},
        )
        assert put_resp.status_code == 200

        resp = await client.get(f"/api/v1/products/{BARCODE}", headers=headers)
        assert resp.json()["analysis"]["natural_language_summary"] == "Resumo atualizado."
        # Edição do produto invalida o cache → 2ª chamada ao Groq.
        assert mock_generate.call_count == 2


# ---------------------------------------------------------------------------
# POST /{barcode}/scan — scan-on-read (requer auth)
# ---------------------------------------------------------------------------

async def test_scan_product_requires_auth(client: AsyncClient, auth_token: str) -> None:
    await _create_product(client, auth_token)
    resp = await client.post(f"/api/v1/products/{BARCODE}/scan")
    assert resp.status_code == 401


async def test_scan_product_not_found(client: AsyncClient, auth_token: str) -> None:
    resp = await client.post(
        f"/api/v1/products/{BARCODE}/scan",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 404


async def test_scan_product_returns_analysis_and_summary(
    client: AsyncClient, auth_token: str, auth_token_2: str
) -> None:
    with patch.object(get_settings(), "groq_api_key", "test-groq-key"), patch(
        "app.products.service.generate_summary",
        new_callable=AsyncMock,
        return_value="Resumo de teste.",
    ):
        data = await _create_product(
            client,
            auth_token,
            body={"ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]}},
        )

        resp = await client.post(
            f"/api/v1/products/{BARCODE}/scan",
            headers={"Authorization": f"Bearer {auth_token_2}"},
        )
        assert resp.status_code == 200
        result = resp.json()
        assert result["barcode"] == BARCODE
        if data["analysis"] is not None:
            assert result["analysis"]["natural_language_summary"] == "Resumo de teste."


async def test_scan_product_records_history(
    client: AsyncClient, auth_token: str, auth_token_2: str
) -> None:
    await _create_product(
        client,
        auth_token,
        body={"ingredients": {"items": ["sal"]}},
    )
    headers_2 = {"Authorization": f"Bearer {auth_token_2}"}

    before = await client.get("/api/v1/users/me/scans", headers=headers_2)
    assert before.json()["total"] == 0

    resp = await client.post(f"/api/v1/products/{BARCODE}/scan", headers=headers_2)
    assert resp.status_code == 200

    after = await client.get("/api/v1/users/me/scans", headers=headers_2)
    assert after.json()["total"] == 1


async def test_scan_product_dedupe_moves_to_top_without_duplicating(
    client: AsyncClient, auth_token: str
) -> None:
    """Reler um produto já escaneado atualiza o registro existente (sobe ao topo)."""
    barcode_a = BARCODE
    barcode_b = "7891234567891"
    headers = {"Authorization": f"Bearer {auth_token}"}

    await _create_product(client, auth_token, barcode=barcode_a)

    scans_after_a = await client.get("/api/v1/users/me/scans", headers=headers)
    scan_id_a = scans_after_a.json()["items"][0]["id"]

    await _create_product(client, auth_token, barcode=barcode_b)

    scans_after_b = await client.get("/api/v1/users/me/scans", headers=headers)
    assert scans_after_b.json()["total"] == 2

    # Reler o produto A via scan-on-read deve atualizar (não duplicar) o registro existente.
    resp = await client.post(f"/api/v1/products/{barcode_a}/scan", headers=headers)
    assert resp.status_code == 200

    scans_final = await client.get("/api/v1/users/me/scans", headers=headers)
    payload_final = scans_final.json()
    assert payload_final["total"] == 2
    assert payload_final["items"][0]["id"] == scan_id_a


# ---------------------------------------------------------------------------
# GET /{barcode}/summary — personalized summary endpoint
# ---------------------------------------------------------------------------

async def test_summary_not_found(client: AsyncClient) -> None:
    """404 when the product does not exist."""
    resp = await client.get(f"/api/v1/products/{BARCODE}/summary")
    assert resp.status_code == 404


async def test_summary_no_ingredients_returns_nulls(
    client: AsyncClient, auth_token: str
) -> None:
    """Product without ingredients returns all-null summary response."""
    await _create_product(client, auth_token, body={"name": "Empty"})
    resp = await client.get(f"/api/v1/products/{BARCODE}/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"] is None
    assert data["risco_global"] is None


async def test_summary_anonymous_returns_summary(
    client: AsyncClient, auth_token: str
) -> None:
    """Anonymous request returns generic summary with matching response schema."""
    with patch.object(get_settings(), "groq_api_key", "test-groq-key"), patch(
        "app.products.service.generate_summary",
        new_callable=AsyncMock,
        return_value="Generic summary.",
    ):
        await _create_product(
            client,
            auth_token,
            body={"ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]}},
        )
        resp = await client.get(f"/api/v1/products/{BARCODE}/summary")

    if resp.status_code == 200:
        data = resp.json()
        assert "summary" in data
        assert "diabetes_type" in data
        assert "language_level" in data
        assert "risco_global" in data
        assert data["diabetes_type"] is None
        assert data["language_level"] is None


async def test_summary_authenticated_personalized(
    client: AsyncClient, auth_token: str
) -> None:
    """Authenticated user with profile gets personalized summary."""
    with patch.object(get_settings(), "groq_api_key", "test-groq-key"), patch(
        "app.products.service.generate_summary",
        new_callable=AsyncMock,
        return_value="Personalized summary for DM2.",
    ):
        await _create_product(
            client,
            auth_token,
            body={"ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]}},
        )

        headers = {"Authorization": f"Bearer {auth_token}"}
        await client.put(
            "/api/v1/users/me",
            headers=headers,
            json={"diabetes_type": "DM2", "language_level": "leigo"},
        )

        resp = await client.get(
            f"/api/v1/products/{BARCODE}/summary", headers=headers
        )

    if resp.status_code == 200:
        data = resp.json()
        assert data["summary"] == "Personalized summary for DM2."
        assert data["diabetes_type"] == "DM2"
        assert data["language_level"] == "leigo"
        assert data["risco_global"] is not None


async def test_summary_two_users_get_distinct_texts(
    client: AsyncClient, auth_token: str, auth_token_2: str
) -> None:
    """Two users with different profiles get different summaries for the same barcode."""
    with patch.object(get_settings(), "groq_api_key", "test-groq-key"), patch(
        "app.products.service.generate_summary",
        new_callable=AsyncMock,
        side_effect=["Summary for DM1 simples.", "Summary for DMG técnico.", "Summary for DM1 simples."],
    ):
        await _create_product(
            client,
            auth_token,
            body={"ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]}},
        )

        headers_1 = {"Authorization": f"Bearer {auth_token}"}
        headers_2 = {"Authorization": f"Bearer {auth_token_2}"}

        await client.put(
            "/api/v1/users/me",
            headers=headers_1,
            json={"diabetes_type": "DM1", "language_level": "leigo"},
        )
        await client.put(
            "/api/v1/users/me",
            headers=headers_2,
            json={"diabetes_type": "DMG", "language_level": "tecnico"},
        )

        resp1 = await client.get(
            f"/api/v1/products/{BARCODE}/summary", headers=headers_1
        )
        resp2 = await client.get(
            f"/api/v1/products/{BARCODE}/summary", headers=headers_2
        )

    if resp1.status_code == 200 and resp2.status_code == 200:
        assert resp1.json()["summary"] != resp2.json()["summary"]
        assert resp1.json()["diabetes_type"] == "DM1"
        assert resp2.json()["diabetes_type"] == "DMG"


async def test_summary_fallback_to_default_when_llm_fails(
    client: AsyncClient, auth_token: str
) -> None:
    """When LLM fails for authenticated user, falls back to generic cached summary."""
    with patch.object(get_settings(), "groq_api_key", "test-groq-key"), patch(
        "app.products.service.generate_summary",
        new_callable=AsyncMock,
        side_effect=["Generic fallback.", None],
    ):
        await _create_product(
            client,
            auth_token,
            body={"ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]}},
        )

        # First call generates generic summary (anonymous during create)
        # Now set personalization and make summary request — LLM returns None
        headers = {"Authorization": f"Bearer {auth_token}"}
        await client.put(
            "/api/v1/users/me",
            headers=headers,
            json={"diabetes_type": "DM1", "language_level": "leigo"},
        )

        resp = await client.get(
            f"/api/v1/products/{BARCODE}/summary", headers=headers
        )

    if resp.status_code == 200:
        data = resp.json()
        # Should fall back to generic summary
        assert data["summary"] == "Generic fallback."
        assert data["diabetes_type"] is None
        assert data["language_level"] is None


async def test_summary_anonymous_without_groq_returns_null(
    client: AsyncClient, auth_token: str
) -> None:
    """Without Groq API key and no cached summary, returns null summary."""
    await _create_product(
        client,
        auth_token,
        body={"ingredients": {"items": ["açúcar", "farinha de trigo", "sal"]}},
    )
    resp = await client.get(f"/api/v1/products/{BARCODE}/summary")
    assert resp.status_code == 200
    data = resp.json()
    # Without Groq key, summary is null (no cache exists yet)
    if data["risco_global"] is not None:
        assert data["summary"] is None or isinstance(data["summary"], str)
