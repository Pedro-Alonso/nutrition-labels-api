"""Abstrações compartilhadas pelos executores de pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audit.recorder import AuditRecorder, AttemptArtifacts


@dataclass(slots=True)
class StageRecord:
    order: int
    name: str
    op: str
    output_path: str
    params: dict


@dataclass(slots=True)
class PipelineContext:
    """Dados de runtime compartilhados entre etapas de um preset."""

    input_path: Path
    attempt_index: int
    preset_name: str
    recorder: AuditRecorder
    artifacts: AttemptArtifacts
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class PipelineResult:
    ocr_text: str
    mean_confidence: float
    stages: list[StageRecord]
    final_image: np.ndarray
    metadata: dict = field(default_factory=dict)


class Pipeline(ABC):
    """Interface mínima de um executor de preset."""

    @abstractmethod
    def execute(
        self,
        image: np.ndarray,
        context: PipelineContext,
    ) -> PipelineResult:
        raise NotImplementedError
