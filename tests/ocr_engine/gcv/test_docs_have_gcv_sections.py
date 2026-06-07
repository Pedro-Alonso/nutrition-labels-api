"""Smoke test 13.5: arquivos de documentação mencionam a feature GCV.

**Validates: Requirements 15.1, 15.2, 15.3, 15.4**

Verifica que cada documento atualizado pela feature ``cloud_vision``
contém ao menos um marcador esperado relacionado ao GCV. Falha se
algum arquivo não mencionar a feature — sinaliza que a seção de
documentação foi esquecida ou acidentalmente removida.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Raiz do monolito (teste-pytesseract/) — contém docs/PRESETS.md, docs/CONFIG.md,
# docs/ARCHITECTURE.md e .kiro/steering/. Os testes GCV verificam se esses
# arquivos do monolito foram atualizados para mencionar a feature cloud_vision.
# Estrutura: tests/ocr_engine/gcv/  →  parents[3] = nutrition-labels-api/  →  parents[4] = monolito
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Arquivos que devem mencionar GCV, com pelo menos um dos marcadores listados.
# Estrutura: (caminho_relativo_ao_projeto, marcadores_esperados)
_DOC_REQUIREMENTS: list[tuple[str, list[str]]] = [
    (
        "docs/PRESETS.md",
        ["cloud_vision", "gcv", "GCV"],
    ),
    (
        "docs/CONFIG.md",
        ["cloud_vision", "gcv", "GCV"],
    ),
    (
        "docs/ARCHITECTURE.md",
        ["cloud_vision", "CloudVisionPipeline", "GCV"],
    ),
    (
        ".kiro/steering/nutrition.md",
        ["cloud_vision", "CloudVisionPipeline"],
    ),
    (
        ".kiro/steering/nutrition-pipelines.md",
        ["cloud_vision", "CloudVisionPipeline"],
    ),
]


@pytest.mark.parametrize("rel_path,markers", _DOC_REQUIREMENTS)
def test_doc_menciona_gcv(rel_path: str, markers: list[str]) -> None:
    """Verifica que ``rel_path`` contém ao menos um dos marcadores GCV."""
    doc_path = _PROJECT_ROOT / rel_path
    assert doc_path.exists(), (
        f"Arquivo de documentação não encontrado: {doc_path}"
    )
    content = doc_path.read_text(encoding="utf-8")
    found = any(marker in content for marker in markers)
    assert found, (
        f"{rel_path} não menciona nenhum dos marcadores GCV esperados: "
        f"{markers!r} — a seção de documentação pode estar faltando."
    )
