"""Testes do endpoint POST /api/v1/analyze.

Cobre:
- Análise sem autenticação → 401
- Upload de arquivo não-imagem → 400
- Upload de imagem válida (tabela nutricional) → 200 com JSON válido
- Upload de imagem de ingredientes → 200 com ingredient_analysis
- Scan salvo no banco após análise bem-sucedida
"""
from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.models import Scan

FIXTURES = Path(__file__).parent.parent / "fixtures" / "images"


async def test_analyze_requires_auth(client: AsyncClient) -> None:
    """Requisição sem token JWT deve retornar 401."""
    with open(FIXTURES / "coca_tabela.jpg", "rb") as f:
        resp = await client.post(
            "/api/v1/analyze",
            files={"file": ("coca_tabela.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 401


async def test_analyze_rejects_non_image(client: AsyncClient, auth_token: str) -> None:
    """Upload de arquivo com Content-Type não-imagem deve retornar 400."""
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("nota.txt", b"isso nao e imagem", "text/plain")},
    )
    assert resp.status_code == 400


async def test_analyze_table_returns_valid_json(
    client: AsyncClient, auth_token: str
) -> None:
    """Imagem de tabela nutricional deve retornar 200 com estrutura correta."""
    with open(FIXTURES / "coca_tabela.jpg", "rb") as f:
        resp = await client.post(
            "/api/v1/analyze",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("coca_tabela.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 200
    data = resp.json()

    # Campos obrigatórios do AnalyzeResponse
    assert "scan_id" in data
    assert "detected_format" in data
    assert "category" in data["detected_format"]
    assert "winning_preset" in data
    assert "passed" in data
    assert isinstance(data["passed"], bool)
    assert "final_ocr_text" in data
    assert "final_postprocessed_text" in data
    assert "attempts" in data
    assert isinstance(data["attempts"], list)


async def test_analyze_ingredients_returns_ingredient_analysis(
    client: AsyncClient, auth_token: str
) -> None:
    """Imagem analisada como ingredientes deve retornar ingredient_analysis."""
    with open(FIXTURES / "coca_ingredientes.jpg", "rb") as f:
        resp = await client.post(
            "/api/v1/analyze",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("coca_ingredientes.jpg", f, "image/jpeg")},
            data={"category_override": "ingredient"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "ingredient_analysis" in data
    # ingredient_analysis pode ser None se o texto OCR não foi reconhecido;
    # mas a chave deve estar presente e, quando presente, ter estrutura válida.
    ia = data["ingredient_analysis"]
    if ia is not None:
        assert "risco_global" in ia
        assert "ingredientes_identificados" in ia
        assert "nao_identificados" in ia


async def test_scan_persisted_in_db(
    client: AsyncClient,
    auth_token: str,
    db_session: AsyncSession,
) -> None:
    """Após uma análise bem-sucedida, um Scan deve existir no banco."""
    with open(FIXTURES / "coca_tabela.jpg", "rb") as f:
        resp = await client.post(
            "/api/v1/analyze",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("coca_tabela.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 200
    scan_id = resp.json()["scan_id"]
    assert scan_id is not None

    result = await db_session.execute(select(Scan).where(Scan.id == scan_id))
    scan = result.scalar_one_or_none()
    assert scan is not None, f"Scan {scan_id} não encontrado no banco"
    assert scan.image_hash  # SHA-256 do upload deve estar salvo


async def test_presets_endpoint_no_auth(client: AsyncClient) -> None:
    """GET /presets não exige autenticação e retorna as categorias."""
    resp = await client.get("/api/v1/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert "table" in data
    assert "text" in data
    assert "ingredients" in data
    # Deve haver ao menos um preset em cada categoria
    assert len(data["table"]) > 0
    assert len(data["text"]) > 0
    assert len(data["ingredients"]) > 0
