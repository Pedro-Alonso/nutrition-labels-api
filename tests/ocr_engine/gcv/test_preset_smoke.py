"""Smoke test da convenção ``steps == []`` em GCV presets.

Cobre a tarefa **11.12** do plano de implementação: garantir que **todo**
preset declarado com ``kind == "cloud_vision"`` em ``config/presets/`` tem
``steps`` igual a uma lista vazia (``[]``), conforme R3.8 do documento de
requirements:

    The GCV_Preset SHALL declare ``steps`` as an empty list (``[]``) by
    convention, making it explicit that there is no local PDI before the
    API call.

Validates: Requirements 3.8.

O teste é puramente textual/declarativo — apenas inspeciona arquivos JSON
em disco, sem instanciar ``PresetRepository`` nem o pipeline. A intenção é
flagrar regressões em PRs que modifiquem presets GCV ou criem novos. O
teste passa silenciosamente quando não há nenhum preset GCV no
repositório (o predicado `for all GCV preset, steps == []` é vacuamente
verdadeiro), mas emite uma asserção informativa quando o diretório
``config/presets/`` não existe.
"""

from __future__ import annotations

import json
from pathlib import Path


# Raiz do backend resolvida a partir deste arquivo:
# ``tests/ocr_engine/gcv/test_preset_smoke.py`` → sobe 3 níveis (gcv →
# ocr_engine → tests → rotulos-backend). Os presets estão em
# ``ocr_engine/config/presets/`` no backend.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PRESETS_ROOT = _PROJECT_ROOT / "ocr_engine" / "config" / "presets"


def _iter_preset_files() -> list[Path]:
    """Lista todos os JSONs em ``config/presets/`` recursivamente.

    Retorna lista ordenada para reprodutibilidade entre execuções (a
    ordem de ``Path.rglob`` não é garantida em todas as plataformas).
    """

    return sorted(_PRESETS_ROOT.rglob("*.json"))


def _load_preset(path: Path) -> dict:
    """Carrega um preset JSON do disco com encoding UTF-8 explícito.

    Mantemos o ``json.loads`` cru (sem ``PresetRepository``) para que o
    smoke test não dependa do parser da feature: sua função é validar a
    convenção declarativa nos arquivos, não o comportamento do loader.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def test_presets_root_exists() -> None:
    """``config/presets/`` precisa existir na raiz do projeto.

    Pré-condição estrutural — falha cedo se o diretório for removido ou
    renomeado por engano, antes que o assert principal silencie a
    regressão por iterar uma lista vazia.
    """

    assert _PRESETS_ROOT.is_dir(), (
        f"Diretório de presets não encontrado em {_PRESETS_ROOT}"
    )


def test_cloud_vision_presets_have_empty_steps() -> None:
    """Validates: Requirements 3.8.

    Para cada preset com ``kind == "cloud_vision"`` em
    ``config/presets/**/*.json``, o campo ``steps`` deve ser
    exatamente uma lista vazia. A convenção torna explícito que não
    existe PDI local antes do envio à API (a imagem é encaminhada
    como veio do Reader).

    Falha incluindo o caminho do preset ofensor e o valor encontrado
    para facilitar o diagnóstico em CI/local.
    """

    offenders: list[tuple[Path, object]] = []
    for preset_file in _iter_preset_files():
        data = _load_preset(preset_file)
        if data.get("kind") != "cloud_vision":
            continue
        steps = data.get("steps", None)
        if steps != []:
            offenders.append((preset_file, steps))

    assert not offenders, (
        "Presets cloud_vision devem declarar steps == [] (R3.8). "
        f"Violações encontradas: {[(str(p), s) for p, s in offenders]}"
    )
