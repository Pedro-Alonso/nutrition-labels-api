"""NullAuditRecorder — implementa a interface de AuditRecorder sem I/O em disco.

No backend REST os resultados são retornados como JSON HTTP, não persistidos em
arquivos locais. Todos os métodos são no-ops que retornam valores sentinela.

A classe é exposta como ``AuditRecorder`` para que o código copiado do monolito
(nutrition/reader.py) importe ``from audit.recorder import AuditRecorder`` e
receba esta implementação sem nenhuma alteração.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class AttemptArtifacts:
    """Versão sentinela de AttemptArtifacts — não aponta para arquivos reais."""

    preset_name: str = ""
    attempt_index: int = 0
    stages_dir: Path = field(default_factory=lambda: Path("/dev/null"))
    ocr_path: Path = field(default_factory=lambda: Path("/dev/null"))
    postprocessed_path: Path = field(default_factory=lambda: Path("/dev/null"))
    score_path: Path = field(default_factory=lambda: Path("/dev/null"))


class AuditRecorder:
    """NullAuditRecorder: todos os métodos são no-ops.

    Implementa a mesma interface pública do ``AuditRecorder`` real do monolito
    (audit/recorder.py) para que ``NutritionReader`` funcione sem modificações.
    """

    def __init__(
        self,
        project_root: Path,
        input_path: Path,
        clean_previous: bool = True,
    ) -> None:
        self.input_slug = input_path.stem

    def set_format_detection(self, detection: dict) -> None:
        pass

    def start_attempt(
        self, attempt_index: int, preset_name: str
    ) -> AttemptArtifacts:
        return AttemptArtifacts(
            preset_name=preset_name,
            attempt_index=attempt_index,
        )

    def save_stage(
        self,
        attempt: AttemptArtifacts,
        stage_index: int,
        stage_name: str,
        image: np.ndarray,
    ) -> Path:
        return Path("/dev/null")

    def save_stage_json(
        self,
        attempt: AttemptArtifacts,
        stage_index: int,
        stage_name: str,
        data: dict,
    ) -> Path:
        return Path("/dev/null")

    def save_attempt_texts(
        self,
        attempt: AttemptArtifacts,
        ocr_text: str,
        postprocessed_text: str,
        score: dict,
    ) -> None:
        pass

    def record_attempt(
        self,
        attempt: AttemptArtifacts,
        stages: list[dict],
        score: dict,
        passed: bool,
    ) -> None:
        pass

    def finalize(
        self,
        winning_attempt_index: int | None,
        winning_preset: str | None,
        final_ocr_text: str,
        final_postprocessed_text: str,
        groundtruth_text: str | None = None,
    ) -> Path:
        return Path("/dev/null")

    def save_ingredient_analysis(
        self,
        ocr_tokens: list[str],
        feedback_clinico: str,
        analysis_dict: dict,
    ) -> None:
        pass
