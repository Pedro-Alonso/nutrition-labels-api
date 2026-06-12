"""Testes unitários dos extratores LLM (Groq) e das guardas de saída.

Nenhum teste faz chamada real à Groq — `AsyncGroq` é mockado via
`unittest.mock.patch`, espelhando o padrão de `tests/unit/test_analysis_service.py`.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.products.llm_service import (
    _is_refusal,
    clean_ingredients_text,
    clean_nutritional_table,
)
from app.products.router import _looks_like_single_phrase


def _mock_groq_client(content: str) -> MagicMock:
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=content))]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


# ---------------------------------------------------------------------------
# clean_nutritional_table
# ---------------------------------------------------------------------------

async def test_clean_nutritional_table_success() -> None:
    response_json = json.dumps(
        {
            "portion_description": "Porção de 13g (1 colher de sopa)",
            "columns": ["Quantidade por porção", "% VD"],
            "rows": [
                {"nutrient": "Valor energético", "values": ["75kcal", "4%"]},
                {"nutrient": "Carboidratos", "values": ["0g", "0%"]},
                {"nutrient": "Proteínas", "values": ["0g", "0%"]},
                {"nutrient": "Gorduras totais", "values": ["8g", "15%"]},
                {"nutrient": "Gorduras saturadas", "values": ["1,2g", "5%"]},
                {"nutrient": "Sódio", "values": ["80mg", "3%"]},
            ],
        }
    )
    client = _mock_groq_client(response_json)
    with patch("app.products.llm_service.AsyncGroq", return_value=client):
        result = await clean_nutritional_table("texto ocr cru da maionese", "fake-key")

    assert result is not None
    assert result.portion_description == "Porção de 13g (1 colher de sopa)"
    assert result.columns == ["Quantidade por porção", "% VD"]
    assert len(result.rows) == 6
    assert result.rows[0].nutrient == "Valor energético"
    assert result.rows[0].values == ["75kcal", "4%"]


async def test_clean_nutritional_table_empty_rows_returns_none() -> None:
    response_json = json.dumps({"portion_description": None, "columns": [], "rows": []})
    client = _mock_groq_client(response_json)
    with patch("app.products.llm_service.AsyncGroq", return_value=client):
        result = await clean_nutritional_table("texto ilegível da lata monster", "fake-key")

    assert result is None


async def test_clean_nutritional_table_invalid_json_returns_none() -> None:
    client = _mock_groq_client("isto não é um json válido")
    with patch("app.products.llm_service.AsyncGroq", return_value=client):
        result = await clean_nutritional_table("texto qualquer", "fake-key")

    assert result is None


async def test_clean_nutritional_table_exception_returns_none() -> None:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("network error"))
    with patch("app.products.llm_service.AsyncGroq", return_value=client):
        result = await clean_nutritional_table("texto qualquer", "fake-key")

    assert result is None


# ---------------------------------------------------------------------------
# clean_ingredients_text — guardas de recusa (bug da lata Monster)
# ---------------------------------------------------------------------------

async def test_clean_ingredients_text_normal_passthrough() -> None:
    client = _mock_groq_client("água, óleo de soja, vinagre, sal")
    with patch("app.products.llm_service.AsyncGroq", return_value=client):
        result = await clean_ingredients_text("texto ocr", "fake-key")

    assert result == "água, óleo de soja, vinagre, sal"


async def test_clean_ingredients_text_refusal_returns_empty() -> None:
    client = _mock_groq_client("Não há ingredientes listados no texto fornecido.")
    with patch("app.products.llm_service.AsyncGroq", return_value=client):
        result = await clean_ingredients_text("texto ilegível da lata monster", "fake-key")

    assert result == ""


async def test_clean_ingredients_text_long_phrase_without_comma_returns_empty() -> None:
    long_phrase = (
        "Esta imagem apresenta apenas ruído visual sem qualquer lista legível "
        "de itens alimentares reconhecíveis"
    )
    client = _mock_groq_client(long_phrase)
    with patch("app.products.llm_service.AsyncGroq", return_value=client):
        result = await clean_ingredients_text("texto ilegível", "fake-key")

    assert result == ""


@pytest.mark.parametrize(
    "text,expected",
    [
        ("água, sal, açúcar", False),
        ("Não há ingredientes listados no texto fornecido.", True),
        ("Nenhum ingrediente foi encontrado.", True),
        ("", False),
        ("sal", False),
    ],
)
def test_is_refusal(text: str, expected: bool) -> None:
    assert _is_refusal(text) == expected


# ---------------------------------------------------------------------------
# _looks_like_single_phrase — guarda de item único no ocr_preview
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "items,expected",
    [
        (["Não há ingredientes listados no texto fornecido, infelizmente."], True),
        (["água", "sal", "açúcar"], False),
        (["água"], False),
        ([], False),
    ],
)
def test_looks_like_single_phrase(items: list[str], expected: bool) -> None:
    assert _looks_like_single_phrase(items) == expected
