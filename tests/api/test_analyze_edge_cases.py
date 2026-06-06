"""Testes de edge-cases do endpoint POST /analyze.

Cobre:
- arquivo muito grande → 413
- arquivo de zero bytes → 400
- imagem corrompida → 400
- content-type não suportado → 400
- category_override válidos (table, text, ingredient)
- category_override inválido cai no OCR normalmente (sem erro)
- passed=False em imagem mínima (1×1 px branco)
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from httpx import AsyncClient


def _make_tiny_jpeg() -> bytes:
    tiny = np.ones((1, 1, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".jpg", tiny)
    return buf.tobytes()


def _make_jpeg_bytes(h: int = 64, w: int = 64) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Validação de tamanho e conteúdo
# ---------------------------------------------------------------------------

async def test_file_too_large_returns_413(
    client: AsyncClient, auth_token: str
) -> None:
    # Gera um payload maior que o limite (10MB padrão)
    large_data = b"x" * (11 * 1024 * 1024)
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("big.jpg", io.BytesIO(large_data), "image/jpeg")},
    )
    assert resp.status_code == 413


async def test_zero_byte_file_returns_400(
    client: AsyncClient, auth_token: str
) -> None:
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert resp.status_code == 400


async def test_corrupt_image_returns_400(
    client: AsyncClient, auth_token: str
) -> None:
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("corrupt.jpg", io.BytesIO(b"\x00\x01\x02\x03bad"), "image/jpeg")},
    )
    assert resp.status_code == 400


async def test_unsupported_content_type_returns_400(
    client: AsyncClient, auth_token: str
) -> None:
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# category_override
# ---------------------------------------------------------------------------

async def test_category_override_table(
    client: AsyncClient, auth_token: str
) -> None:
    data = _make_jpeg_bytes()
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("img.jpg", io.BytesIO(data), "image/jpeg")},
        data={"category_override": "table"},
    )
    assert resp.status_code == 200
    assert resp.json()["detected_format"]["category"] == "table"


async def test_category_override_text(
    client: AsyncClient, auth_token: str
) -> None:
    data = _make_jpeg_bytes()
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("img.jpg", io.BytesIO(data), "image/jpeg")},
        data={"category_override": "text"},
    )
    assert resp.status_code == 200
    assert resp.json()["detected_format"]["category"] == "text"


async def test_category_override_ingredient(
    client: AsyncClient, auth_token: str
) -> None:
    data = _make_jpeg_bytes()
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("img.jpg", io.BytesIO(data), "image/jpeg")},
        data={"category_override": "ingredient"},
    )
    assert resp.status_code == 200
    assert resp.json()["detected_format"]["category"] == "ingredient"


async def test_category_override_invalid_falls_back(
    client: AsyncClient, auth_token: str
) -> None:
    """Override inválido é ignorado silenciosamente; roteamento automático ocorre."""
    data = _make_jpeg_bytes()
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("img.jpg", io.BytesIO(data), "image/jpeg")},
        data={"category_override": "invalid_category"},
    )
    assert resp.status_code == 200
    # A categoria detectada deve ser "table" ou "text" (heurística)
    assert resp.json()["detected_format"]["category"] in ("table", "text", "ingredient")


# ---------------------------------------------------------------------------
# Imagem mínima → passed=False (Tesseract não extrai texto de 1×1 px)
# ---------------------------------------------------------------------------

async def test_passed_false_on_minimal_image(
    client: AsyncClient, auth_token: str
) -> None:
    """Imagem 1×1 branco → Tesseract retorna vazio → passed=False."""
    data = _make_tiny_jpeg()
    resp = await client.post(
        "/api/v1/analyze",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("tiny.jpg", io.BytesIO(data), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is False
    assert body["winning_preset"] is not None
