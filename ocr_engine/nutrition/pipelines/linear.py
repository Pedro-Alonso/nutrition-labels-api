"""Executor linear: aplica uma sequência de operações e roda OCR no final."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from imaging import operations as ops
from imaging.morphology import remove_grid_lines
from ocr.service import OcrService, OcrConfig

from .base import Pipeline, PipelineContext, PipelineResult, StageRecord


def apply_operation(op_name: str, image: np.ndarray, params: dict) -> np.ndarray:
    """Despacha uma operação pelo registry declarativo.

    Duas operações extras vivem fora de `imaging.operations` por dependerem de
    `imaging.morphology`: `remove_grid_lines` e, futuramente, detectores
    adicionais específicos do domínio.
    """
    if op_name == "remove_grid_lines":
        return remove_grid_lines(
            image,
            horizontal_divisor=int(params.get("horizontal_divisor", 20)),
            vertical_divisor=int(params.get("vertical_divisor", 20)),
            min_line_area_ratio=float(params.get("min_line_area_ratio", 0.005)),
        )

    func = ops.OPERATION_REGISTRY.get(op_name)
    if func is None:
        raise ValueError(f"Operação desconhecida no preset: {op_name}")
    return func(image, **params)


class LinearPipeline(Pipeline):
    def __init__(self, steps: Sequence[dict], ocr_config: OcrConfig) -> None:
        self.steps = list(steps)
        self.ocr_service = OcrService(ocr_config)

    def execute(self, image: np.ndarray, context: PipelineContext) -> PipelineResult:
        stages: list[StageRecord] = []
        current = image
        recorder = context.recorder
        artifacts = context.artifacts

        # Etapa 01: sempre a entrada original, para conferência.
        stage_index = 1
        path_input = recorder.save_stage(artifacts, stage_index, "input", current)
        stages.append(StageRecord(stage_index, "input", "input", str(path_input), {}))

        for raw_step in self.steps:
            stage_index += 1
            op_name = raw_step["op"]
            stage_label = raw_step.get("name", op_name)
            params = {k: v for k, v in raw_step.items() if k not in {"op", "name"}}
            current = apply_operation(op_name, current, params)
            stage_path = recorder.save_stage(artifacts, stage_index, stage_label, current)
            stages.append(
                StageRecord(
                    order=stage_index,
                    name=stage_label,
                    op=op_name,
                    output_path=str(stage_path),
                    params=params,
                )
            )

        ocr_result = self.ocr_service.read(current)
        if ocr_result.image_for_ocr is not None and ocr_result.used_inverted:
            stage_index += 1
            label = "dual_pass_inverted"
            stage_path = recorder.save_stage(artifacts, stage_index, label, ocr_result.image_for_ocr)
            stages.append(StageRecord(stage_index, label, "invert", str(stage_path), {}))
            current = ocr_result.image_for_ocr

        # Última etapa marcada como "output" para facilitar inspeção visual.
        stage_index += 1
        output_path = recorder.save_stage(artifacts, stage_index, "output", current)
        stages.append(StageRecord(stage_index, "output", "output", str(output_path), {}))

        return PipelineResult(
            ocr_text=ocr_result.text,
            mean_confidence=ocr_result.confidence,
            stages=stages,
            final_image=current,
            metadata={"used_inverted_polarity": ocr_result.used_inverted},
        )
