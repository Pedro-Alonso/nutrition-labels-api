"""Smoke test dos pins de dependência em ``requirements.txt``.

Cobre a tarefa **11.13** do plano de implementação: garantir que o
``requirements.txt`` na raiz do projeto contém **simultaneamente** a
dependência ``google-cloud-vision`` (necessária para o novo pipeline
``cloud_vision``) e o pin ``numpy<2.0.0`` (preservado por compatibilidade
com TensorFlow, conforme ``AGENTS.md``).

Validates: Requirements 14.1, 14.2.

O teste é puramente textual — não tenta importar pacotes nem resolver a
árvore de dependências. Lê ``requirements.txt`` como UTF-8, ignora linhas
em branco e comentários (``#``), e procura, ao menos, uma linha que
declare cada pin esperado. Essa simplicidade é proposital: o objetivo é
flagrar regressões acidentais em PRs que mexem em ``requirements.txt``,
não validar resolução de versões.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# Raiz do backend resolvida a partir deste arquivo:
# ``tests/ocr_engine/gcv/test_pins.py`` → sobe 3 níveis (gcv → ocr_engine →
# tests → nutrition-labels-api). O requirements.txt está na raiz do backend.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REQUIREMENTS_FILE = _PROJECT_ROOT / "requirements.txt"


def _read_requirement_lines() -> list[str]:
    """Lê ``requirements.txt`` e devolve linhas relevantes (sem comentários).

    Preserva o conteúdo original de cada linha (sem normalizar caixa ou
    espaços) para que as asserções verifiquem o texto real do arquivo.
    Linhas em branco e linhas iniciadas por ``#`` são descartadas porque
    não declaram pins.
    """

    text = _REQUIREMENTS_FILE.read_text(encoding="utf-8")
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def test_requirements_file_exists() -> None:
    """``requirements.txt`` precisa existir na raiz do projeto.

    Pré-condição para os demais asserts; falha cedo com mensagem clara
    caso o arquivo seja movido ou removido por engano.
    """

    assert _REQUIREMENTS_FILE.is_file(), (
        f"requirements.txt não encontrado em {_REQUIREMENTS_FILE}"
    )


def test_google_cloud_vision_dependency_is_declared() -> None:
    """Validates: Requirements 14.1.

    Pelo menos uma linha não comentada de ``requirements.txt`` declara
    ``google-cloud-vision`` (com ou sem especificador de versão).
    """

    lines = _read_requirement_lines()
    matches = [line for line in lines if line.lower().startswith("google-cloud-vision")]

    assert matches, (
        "Esperado ao menos um requirement declarando 'google-cloud-vision'; "
        f"linhas encontradas: {lines!r}"
    )


def test_numpy_pin_is_preserved() -> None:
    """Validates: Requirements 14.2.

    O pin ``numpy<2.0.0`` (sem espaços) deve continuar presente em
    ``requirements.txt``. Comparamos a substring exata para flagrar
    relaxamentos como ``numpy<3`` ou remoções acidentais.
    """

    lines = _read_requirement_lines()
    matches = [line for line in lines if "numpy<2.0.0" in line.replace(" ", "")]

    assert matches, (
        "Pin 'numpy<2.0.0' ausente de requirements.txt; "
        f"linhas encontradas: {lines!r}"
    )


@pytest.mark.parametrize(
    "expected_substring",
    [
        "google-cloud-vision",
        "numpy<2.0.0",
    ],
)
def test_required_pins_coexist(expected_substring: str) -> None:
    """Sanidade combinada: ambos os pins coexistem no mesmo arquivo.

    Reforça a co-presença explicitada na tarefa 11.13 (``google-cloud-vision``
    presente **E** ``numpy<2.0.0`` presente). Implementado como teste
    parametrizado para que o relatório do pytest mostre claramente qual
    pin falhou em caso de regressão.
    """

    raw_text = _REQUIREMENTS_FILE.read_text(encoding="utf-8").replace(" ", "")
    assert expected_substring in raw_text, (
        f"Substring '{expected_substring}' não encontrada em requirements.txt"
    )
