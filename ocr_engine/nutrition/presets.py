"""Carregamento e validação dos presets declarativos.

Cada preset vive em `config/presets/<categoria>/<nome>.json` onde `<categoria>`
é `table` (para imagens tabulares ou baseadas em células) ou `text` (para
informação nutricional escrita em texto corrido).

Schema de um preset:

```json
{
  "name": "otsu_basic",
  "description": "...",
  "kind": "linear_table" | "linear_text" | "cell_based" | "cloud_vision",
  "priority": 10,
  "steps": [
    {"op": "grayscale"},
    {"op": "median_blur", "kernel_size": 3},
    ...
  ],
  "ocr": {
    "lang": "por",
    "psm": 6,
    "oem": 3,
    "extra_config": "",
    "dual_pass_polarity": true
  },
  "cell_detection": {           # apenas quando kind == "cell_based"
    "horizontal_divisor": 20,
    "vertical_divisor": 20,
    "min_cell_area_ratio": 0.001,
    "max_cell_area_ratio": 0.5,
    "cell_ocr_psm": 7
  },
  "gcv": {                      # apenas quando kind == "cloud_vision"
    "feature": "DOCUMENT_TEXT_DETECTION",
    "language_hints": ["pt"],
    "model": null
  },
  "quality_thresholds": {
    "min_mean_confidence": 65,
    "min_text_length": 40,
    "min_keyword_hits": 3,
    "expected_keywords": ["valor energetico", "carboidratos", "proteinas"]
  }
}
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal
import json

from ocr.cloud_vision.types import ALLOWED_KINDS


PresetKind = Literal[
    "linear_table",
    "linear_text",
    "cell_based",
    "linear_ingredient",
    "cloud_vision",
]


@dataclass(slots=True)
class Preset:
    name: str
    description: str
    kind: PresetKind
    priority: int
    steps: list[dict]
    ocr: dict
    cell_detection: dict
    quality_thresholds: dict
    source_path: Path
    gcv: dict = field(default_factory=dict)

    @property
    def category(self) -> str:
        # Para presets ``cloud_vision`` a categoria deriva do diretório de
        # origem (Requirement 2.6) — o mesmo ``kind`` aparece em ``table``,
        # ``text`` e ``ingredients``, então o ``kind`` sozinho não é suficiente
        # para determinar a categoria.
        if self.kind == "cloud_vision":
            return self.source_path.parent.name
        if self.kind == "linear_text":
            return "text"
        if self.kind == "linear_ingredient":
            return "ingredients"
        return "table"


class PresetRepository:
    """Carrega presets de `config/presets/` agrupando por categoria e ordenando por `priority`."""

    def __init__(self, presets_root: Path) -> None:
        self.presets_root = presets_root
        self._presets: dict[str, list[Preset]] = {"table": [], "text": [], "ingredients": []}
        self._load()

    def for_category(self, category: str) -> list[Preset]:
        return list(self._presets.get(category, []))

    def all(self) -> list[Preset]:
        result: list[Preset] = []
        for items in self._presets.values():
            result.extend(items)
        return result

    def _load(self) -> None:
        if not self.presets_root.exists():
            return
        for category_dir in sorted(self.presets_root.iterdir()):
            if not category_dir.is_dir():
                continue
            if category_dir.name not in self._presets:
                continue
            loaded: list[Preset] = []
            for preset_file in sorted(category_dir.glob("*.json")):
                loaded.append(self._parse(preset_file))
            loaded.sort(key=lambda p: (p.priority, p.name))
            self._presets[category_dir.name] = loaded

    @staticmethod
    def _parse(path: Path) -> Preset:
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"name", "kind", "steps", "ocr"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Preset {path} faltando chaves: {missing}")

        kind = data["kind"]
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"Preset {path} kind inválido: {kind}")

        return Preset(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            kind=kind,
            priority=int(data.get("priority", 100)),
            steps=list(data["steps"]),
            ocr=dict(data["ocr"]),
            cell_detection=dict(data.get("cell_detection", {})),
            quality_thresholds=dict(data.get("quality_thresholds", {})),
            source_path=path,
            gcv=dict(data.get("gcv", {})),
        )


def iter_preset_files(presets_root: Path) -> Iterable[Path]:
    if not presets_root.exists():
        return []
    return sorted(presets_root.rglob("*.json"))
