"""Executor baseado em células.

Para tabelas onde iluminação irregular atrapalha OCR da imagem inteira, esta
abordagem:

1. Aplica uma preparação curta (`prep_steps`) para chegar a um binário limpo.
2. (Opcional) OCRiza o topo da imagem linearmente como cabeçalho — captura
   linhas como "Porção de 20g" que ficam acima da grade de células.
3. Recorta geograficamente a região do cabeçalho para que a morfologia não
   confunda o contorno do título com uma célula da tabela.
4. Detecta células da grade usando morfologia.
5. Constrói uma matriz 2D (linhas × colunas) por clustering de x-centros,
   com slots None para células ausentes — preserva alinhamento de colunas
   mesmo quando o OCR de uma célula retorna vazio.
6. OCRiza cada célula individualmente e preenche a matriz.
7. Reconstrói o texto em ordem de leitura (linha a linha, coluna a coluna).
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from imaging.morphology import Cell, crop_cell, detect_cells
from ocr.service import OcrService, OcrConfig

from .base import Pipeline, PipelineContext, PipelineResult, StageRecord
from .linear import apply_operation


class CellBasedPipeline(Pipeline):
    def __init__(
        self,
        prep_steps: Sequence[dict],
        ocr_config: OcrConfig,
        horizontal_divisor: int = 20,
        vertical_divisor: int = 20,
        min_cell_area_ratio: float = 0.001,
        max_cell_area_ratio: float = 0.5,
        cell_ocr_psm: int = 7,
        header_region_ratio: float = 0.0,
        vd_column_psm: int = 7,
    ) -> None:
        self.prep_steps = list(prep_steps)
        self.ocr_config = ocr_config
        self.horizontal_divisor = horizontal_divisor
        self.vertical_divisor = vertical_divisor
        self.min_cell_area_ratio = min_cell_area_ratio
        self.max_cell_area_ratio = max_cell_area_ratio
        self.cell_ocr_psm = cell_ocr_psm
        self.vd_column_psm = vd_column_psm
        # Fração [0, 0.5] da altura tratada como cabeçalho.
        # Se > 0: OCR linear no topo + recorte geográfico para detect_cells.
        self.header_region_ratio = max(0.0, min(float(header_region_ratio), 0.5))

    def execute(self, image: np.ndarray, context: PipelineContext) -> PipelineResult:
        stages: list[StageRecord] = []
        recorder = context.recorder
        artifacts = context.artifacts
        confidences: list[float] = []

        stage_index = 1
        path_input = recorder.save_stage(artifacts, stage_index, "input", image)
        stages.append(StageRecord(stage_index, "input", "input", str(path_input), {}))

        current = image
        for raw_step in self.prep_steps:
            stage_index += 1
            op_name = raw_step["op"]
            stage_label = raw_step.get("name", op_name)
            params = {k: v for k, v in raw_step.items() if k not in {"op", "name"}}
            current = apply_operation(op_name, current, params)
            stage_path = recorder.save_stage(artifacts, stage_index, stage_label, current)
            stages.append(
                StageRecord(stage_index, stage_label, op_name, str(stage_path), params)
            )

        # --- 1. Cabeçalho linear (fora da grade) ---
        header_text = ""
        header_h = 0
        if self.header_region_ratio > 0.0:
            header_h = int(current.shape[0] * self.header_region_ratio)
            if header_h > 0:
                header_crop = current[:header_h, :]
                stage_index += 1
                hdr_path = recorder.save_stage(artifacts, stage_index, "header_crop", header_crop)
                stages.append(
                    StageRecord(
                        stage_index,
                        "header_crop",
                        "header_region",
                        str(hdr_path),
                        {"header_region_ratio": self.header_region_ratio},
                    )
                )
                header_cfg = OcrConfig(
                    lang=self.ocr_config.lang,
                    psm=6,
                    oem=self.ocr_config.oem,
                    extra_config=self.ocr_config.extra_config,
                    dual_pass_polarity=self.ocr_config.dual_pass_polarity,
                )
                hdr_result = OcrService(header_cfg).read(header_crop)
                header_text = hdr_result.text.strip()
                if hdr_result.confidence > 0:
                    confidences.append(hdr_result.confidence)

                # Recorte geográfico: entrega à morfologia apenas a região abaixo
                # do cabeçalho, evitando que o contorno do título seja detectado
                # como célula da tabela (o que causaria duplicação no output).
                current = current[header_h:, :]
                stage_index += 1
                grid_path = recorder.save_stage(artifacts, stage_index, "grid_region", current)
                stages.append(
                    StageRecord(
                        stage_index,
                        "grid_region",
                        "geographic_crop",
                        str(grid_path),
                        {"header_h_px": header_h},
                    )
                )

        # --- 2. Detecção de células na região da grade ---
        cells = detect_cells(
            current,
            horizontal_divisor=self.horizontal_divisor,
            vertical_divisor=self.vertical_divisor,
            min_cell_area_ratio=self.min_cell_area_ratio,
            max_cell_area_ratio=self.max_cell_area_ratio,
        )

        stage_index += 1
        overlay = _draw_cells_overlay(current, cells)
        overlay_path = recorder.save_stage(artifacts, stage_index, "cells_overlay", overlay)
        stages.append(
            StageRecord(
                stage_index,
                "cells_overlay",
                "detect_cells",
                str(overlay_path),
                {
                    "count": len(cells),
                    "horizontal_divisor": self.horizontal_divisor,
                    "vertical_divisor": self.vertical_divisor,
                },
            )
        )

        # --- 3. OCR por matriz 2D ---
        cell_ocr_config = OcrConfig(
            lang=self.ocr_config.lang,
            psm=self.cell_ocr_psm,
            oem=self.ocr_config.oem,
            extra_config=self.ocr_config.extra_config,
            dual_pass_polarity=self.ocr_config.dual_pass_polarity,
        )
        cell_service = OcrService(cell_ocr_config)

        vd_ocr_config = OcrConfig(
            lang=self.ocr_config.lang,
            psm=self.vd_column_psm,
            oem=self.ocr_config.oem,
            extra_config=self.ocr_config.extra_config,
            dual_pass_polarity=self.ocr_config.dual_pass_polarity,
        )
        vd_service = OcrService(vd_ocr_config)

        rows = _group_cells_by_row(cells)
        matrix = _assign_column_slots(cells, rows)  # list[list[Cell | None]]
        n_cols = len(matrix[0]) if matrix else 0

        row_texts: list[str] = []

        for row_idx, row in enumerate(matrix):
            row_pieces: list[str] = []
            for col_idx, cell in enumerate(row):
                if cell is None:
                    # Slot ausente: célula não detectada pela morfologia nessa posição.
                    row_pieces.append("")
                    continue

                crop = crop_cell(current, cell, padding=2)
                if crop.size == 0:
                    row_pieces.append("")
                    continue
                # Borda branca: Tesseract precisa de espaço em branco ao redor
                # do glifo para calibrar escala sem confundir a borda da célula.
                crop = cv2.copyMakeBorder(
                    crop, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255
                )

                stage_index += 1
                label = f"cell_r{row_idx:02d}_c{col_idx:02d}"
                cell_path = recorder.save_stage(artifacts, stage_index, label, crop)
                stages.append(
                    StageRecord(
                        stage_index,
                        label,
                        "cell_crop",
                        str(cell_path),
                        {"x": cell.x, "y": cell.y, "w": cell.w, "h": cell.h},
                    )
                )
                service = vd_service if (n_cols > 1 and col_idx == n_cols - 1) else cell_service
                result = service.read(crop)
                clean = result.text.replace("\n", " ").strip()
                row_pieces.append(clean)
                if result.confidence > 0:
                    confidences.append(result.confidence)

            if any(row_pieces):
                row_texts.append("\t".join(row_pieces))

        grid_text = "\n".join(row_texts)
        reconstructed = (header_text + "\n" + grid_text).strip() if header_text else grid_text

        # Fallback: se detecção falhar, OCRiza a imagem inteira para não devolver vazio.
        if not reconstructed.strip():
            fallback = OcrService(self.ocr_config).read(current)
            reconstructed = fallback.text
            confidences.append(fallback.confidence)

        stage_index += 1
        output_path = recorder.save_stage(artifacts, stage_index, "output", current)
        stages.append(StageRecord(stage_index, "output", "output", str(output_path), {}))

        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return PipelineResult(
            ocr_text=reconstructed,
            mean_confidence=mean_conf,
            stages=stages,
            final_image=current,
            metadata={"cell_count": len(cells), "row_count": len(rows), "col_count": n_cols},
        )


# ---------------------------------------------------------------------------
# Funções auxiliares de geometria
# ---------------------------------------------------------------------------


def _group_cells_by_row(cells: list[Cell]) -> list[list[Cell]]:
    """Agrupa células por linha usando tolerância baseada na altura média."""
    if not cells:
        return []
    ordered = sorted(cells, key=lambda c: (c.y, c.x))
    mean_h = sum(c.h for c in ordered) / len(ordered)
    tolerance = max(8, int(0.5 * mean_h))
    rows: list[list[Cell]] = [[ordered[0]]]
    for cell in ordered[1:]:
        if abs(cell.y - rows[-1][0].y) <= tolerance:
            rows[-1].append(cell)
        else:
            rows.append([cell])
    for row in rows:
        row.sort(key=lambda c: c.x)
    return rows


def _assign_column_slots(
    cells: list[Cell],
    rows: list[list[Cell]],
) -> list[list[Cell | None]]:
    """Constrói uma matriz 2D (linhas × colunas canônicas) com slots None ausentes.

    Determina as colunas canônicas por clustering dos x-centros de todas as
    células detectadas. Cada célula é então atribuída à coluna canônica mais
    próxima. Células sem correspondente em uma linha ficam como None, preservando
    o alinhamento de %VD mesmo quando o OCR de uma linha retorna vazio.
    """
    if not cells or not rows:
        return []

    # Clustering de x-centros para determinar colunas canônicas.
    mean_w = sum(c.w for c in cells) / len(cells)
    tolerance_x = max(10, int(0.4 * mean_w))

    x_centers = sorted(c.x + c.w // 2 for c in cells)
    col_buckets: list[list[int]] = [[x_centers[0]]]
    for x in x_centers[1:]:
        if x - col_buckets[-1][-1] <= tolerance_x:
            col_buckets[-1].append(x)
        else:
            col_buckets.append([x])

    col_refs = [sum(b) // len(b) for b in col_buckets]
    n_cols = len(col_refs)

    def nearest_col(x_center: int) -> int:
        return min(range(n_cols), key=lambda i: abs(col_refs[i] - x_center))

    matrix: list[list[Cell | None]] = [[None] * n_cols for _ in rows]
    for row_idx, row_cells in enumerate(rows):
        for cell in row_cells:
            col_idx = nearest_col(cell.x + cell.w // 2)
            matrix[row_idx][col_idx] = cell

    return matrix


def _draw_cells_overlay(image: np.ndarray, cells: list[Cell]) -> np.ndarray:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
    for cell in cells:
        cv2.rectangle(
            canvas,
            (cell.x, cell.y),
            (cell.x + cell.w, cell.y + cell.h),
            (0, 255, 0),
            2,
        )
    return canvas
