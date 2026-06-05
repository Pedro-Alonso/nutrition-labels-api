"""Testes de integração: ``build_default_reader`` com e sem preset GCV.

**Validates: Requirements 4.3, 4.4**

Verifica que ``build_default_reader`` injeta corretamente os componentes GCV
no ``NutritionReader`` quando há ao menos um preset ``cloud_vision`` em
``config/presets/``, e que não exige credenciais quando não há nenhum.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nutrition.reader import build_default_reader
from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.client import GcvClient


# ---------------------------------------------------------------------------
# Helpers de setup
# ---------------------------------------------------------------------------

_GCV_PRESET: dict = {
    "name": "gcv_doc_text",
    "description": "Teste: GCV DOCUMENT_TEXT_DETECTION.",
    "kind": "cloud_vision",
    "priority": 5,
    "steps": [],
    "ocr": {"lang": "por", "psm": 6, "oem": 3},
    "gcv": {
        "feature": "DOCUMENT_TEXT_DETECTION",
        "language_hints": ["pt"],
        "model": None,
    },
    "quality_thresholds": {
        "min_mean_confidence": 75,
        "min_text_length": 40,
        "min_keyword_hits": 3,
        "expected_keywords": [
            "valor energetico",
            "carboidratos",
            "proteinas",
        ],
    },
}

_LINEAR_PRESET: dict = {
    "name": "otsu_basic",
    "description": "Teste: preset linear sem GCV.",
    "kind": "linear_table",
    "priority": 10,
    "steps": [{"op": "grayscale"}],
    "ocr": {"lang": "por", "psm": 6, "oem": 3},
    "quality_thresholds": {},
}


def _minimal_project(root: Path) -> Path:
    """Cria estrutura mínima de projeto para ``build_default_reader``."""
    (root / "config" / "presets" / "table").mkdir(parents=True)
    (root / "config" / "presets" / "text").mkdir(parents=True)
    (root / "config" / "presets" / "ingredients").mkdir(parents=True)
    (root / "extractions").mkdir(parents=True)
    (root / "images" / "pipeline").mkdir(parents=True)
    return root


def _write_preset(root: Path, category: str, filename: str, data: dict) -> None:
    path = root / "config" / "presets" / category / filename
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Cenário 1: sem preset GCV
# ---------------------------------------------------------------------------


def test_sem_preset_gcv_gcv_client_eh_none(tmp_path: Path) -> None:
    """Sem preset ``cloud_vision``: ``gcv_client`` deve ser ``None``.

    **Validates: Requirement 4.3**

    Quando o repositório de presets não contém nenhum preset com
    ``kind == "cloud_vision"``, nenhuma credencial GCV deve ser
    exigida e ``reader._gcv_client`` deve ser ``None``.
    """
    root = _minimal_project(tmp_path)
    _write_preset(root, "table", "10_linear.json", _LINEAR_PRESET)

    reader = build_default_reader(root)

    assert reader._gcv_client is None, (
        "gcv_client deveria ser None quando não há preset cloud_vision"
    )


def test_sem_preset_gcv_gcv_app_config_usa_defaults(tmp_path: Path) -> None:
    """Sem preset GCV: ``_gcv_app_config`` usa defaults do design.

    ``build_default_reader`` sempre constrói um ``GcvAppConfig`` — mesmo
    sem preset GCV — para que os defaults sejam refletidos em metadata
    caso um preset seja adicionado depois. A instância usa os valores
    padrão quando ``app.json`` não tem bloco ``gcv``.
    """
    root = _minimal_project(tmp_path)

    reader = build_default_reader(root)

    assert reader._gcv_app_config is not None
    assert isinstance(reader._gcv_app_config, GcvAppConfig)
    # Defaults canônicos (Requirement 4.2)
    assert reader._gcv_app_config.on_failure == "skip"
    assert reader._gcv_app_config.cache_enabled is True
    assert reader._gcv_app_config.max_requests_per_minute is None


# ---------------------------------------------------------------------------
# Cenário 2: com preset GCV
# ---------------------------------------------------------------------------


def test_com_preset_gcv_gcv_client_eh_instanciado(tmp_path: Path) -> None:
    """Com preset ``cloud_vision``: ``gcv_client`` deve ser um ``GcvClient``.

    **Validates: Requirement 4.4**

    Quando o repositório contém ao menos um preset com
    ``kind == "cloud_vision"``, ``build_default_reader`` deve construir
    e injetar um ``GcvClient`` no reader. O cliente é criado pelo
    ``GcvClient.build`` sem fazer chamadas de rede ou resolver
    credenciais (inicialização lazy).
    """
    root = _minimal_project(tmp_path)
    _write_preset(root, "table", "00_gcv.json", _GCV_PRESET)

    reader = build_default_reader(root)

    assert reader._gcv_client is not None, (
        "gcv_client não deveria ser None quando há preset cloud_vision"
    )
    assert isinstance(reader._gcv_client, GcvClient)


def test_com_preset_gcv_gcv_app_config_tambem_injetado(tmp_path: Path) -> None:
    """Com preset GCV: ``_gcv_app_config`` também é injetado no reader."""
    root = _minimal_project(tmp_path)
    _write_preset(root, "table", "00_gcv.json", _GCV_PRESET)

    reader = build_default_reader(root)

    assert reader._gcv_app_config is not None
    assert isinstance(reader._gcv_app_config, GcvAppConfig)


def test_com_preset_gcv_em_multiplas_categorias(tmp_path: Path) -> None:
    """Preset GCV em qualquer categoria aciona instanciação do cliente."""
    root = _minimal_project(tmp_path)
    # Coloca o preset GCV em ``text``, não em ``table``
    gcv_text = {**_GCV_PRESET, "name": "gcv_text"}
    _write_preset(root, "text", "00_gcv.json", gcv_text)

    reader = build_default_reader(root)

    assert reader._gcv_client is not None
    assert isinstance(reader._gcv_client, GcvClient)


def test_app_json_gcv_block_alimenta_app_config(tmp_path: Path) -> None:
    """Bloco ``gcv`` em ``app.json`` é refletido no ``GcvAppConfig`` injetado."""
    root = _minimal_project(tmp_path)
    app_json = {
        "gcv": {
            "credentials_path": None,
            "on_failure": "raise",
            "cache_enabled": False,
            "cache_dir": "extractions/.gcv_cache",
            "max_requests_per_minute": None,
            "request_timeout_seconds": 60,
        }
    }
    (root / "config" / "app.json").write_text(
        json.dumps(app_json), encoding="utf-8"
    )
    _write_preset(root, "table", "00_gcv.json", _GCV_PRESET)

    reader = build_default_reader(root)

    assert reader._gcv_app_config is not None
    assert reader._gcv_app_config.on_failure == "raise"
    assert reader._gcv_app_config.cache_enabled is False
    assert reader._gcv_app_config.request_timeout_seconds == 60.0
