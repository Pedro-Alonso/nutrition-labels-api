"""Orquestrador de leitura de informação nutricional.

Fluxo de uma execução:

1. Abre a imagem.
2. (Opcional) Detecta ROI com `RoiDetector`.
3. Detecta formato (tabela ou texto corrido) via `FormatDetector`.
4. Seleciona presets da categoria correspondente ordenados por `priority`.
5. Executa cada preset em cascata; interrompe no primeiro preset que passar na
   avaliação de qualidade. Se nenhum passar, mantém o de maior score contínuo.
6. Aplica pós-processamento léxico no output vencedor.
7. Grava todos os artefatos via `AuditRecorder`.

O usuário pode forçar uma categoria (tabela ou texto) pelo menu ou pela config.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING
import json

from audit.recorder import AuditRecorder
from imaging.io import read_image
from imaging.roi import RoiDetectionConfig, RoiDetector
from ingredients import IngredientAnalyzer, IngredientReport
from ocr.cloud_vision.app_config import GcvAppConfig
from ocr.cloud_vision.options import GcvPresetOptions
from ocr.cloud_vision.postprocessor import GcvTablePostProcessor
from ocr.postprocessing import NutritionTextPostProcessor
from ocr.quality import QualityEvaluator, QualityScore, QualityThresholds
from ocr.service import OcrConfig

from .format_detector import FormatDetector, FormatDetectorConfig, DetectedFormat
from .pipelines import CellBasedPipeline, LinearPipeline, Pipeline, PipelineContext
from .pipelines.cloud_vision import CloudVisionPipeline
from .presets import Preset, PresetRepository

if TYPE_CHECKING:  # pragma: no cover - import só para anotação de tipo.
    # ``GcvClient`` é injetado pelo ``build_default_reader``; o reader não
    # depende do SDK ``google-cloud-vision`` em tempo de carga (Requirement
    # 14.3 — feature opcional). O TYPE_CHECKING também sobrevive ao caso de
    # ``ocr/cloud_vision/client.py`` ainda não existir durante o
    # desenvolvimento incremental do plano de tarefas.
    from ocr.cloud_vision.client import GcvClient


@dataclass(slots=True)
class ReadOutcome:
    winning_preset: str | None
    winning_attempt_index: int | None
    passed: bool
    detected_format: DetectedFormat
    final_ocr_text: str
    final_postprocessed_text: str
    attempts: list[dict] = field(default_factory=list)
    summary_path: Path | None = None
    groundtruth_metrics: dict | None = None
    ingredient_report: IngredientReport | None = None


@dataclass(slots=True)
class ReaderOptions:
    category_override: str | None = None  # "table", "text" ou "ingredient"
    roi_enabled: bool = True
    stop_on_first_pass: bool = True
    postprocess: bool = True


class NutritionReader:
    def __init__(
        self,
        project_root: Path,
        preset_repo: PresetRepository,
        format_detector: FormatDetector,
        roi_config: RoiDetectionConfig,
        gcv_app_config: GcvAppConfig | None = None,
        gcv_client: "GcvClient | None" = None,
    ) -> None:
        self.project_root = project_root
        self.preset_repo = preset_repo
        self.format_detector = format_detector
        self.roi_config = roi_config
        self.text_postprocessor = NutritionTextPostProcessor()
        self.gcv_postprocessor = GcvTablePostProcessor()
        ontology_path = project_root / "config" / "ontology_diabetes.json"
        self.ingredient_analyzer: IngredientAnalyzer | None = (
            IngredientAnalyzer(ontology_path) if ontology_path.exists() else None
        )
        # Wiring opcional do pipeline GCV (Requirement 14.3): a feature inteira é
        # opcional, então ambos os parâmetros são ``None`` por padrão e só são
        # usados quando ``_build_pipeline`` encontra ``preset.kind ==
        # "cloud_vision"``. ``build_default_reader`` injeta os dois quando há ao
        # menos um preset GCV em ``config/presets/``.
        self._gcv_app_config: GcvAppConfig | None = gcv_app_config
        self._gcv_client: "GcvClient | None" = gcv_client
        # Flag de consumo único de ``GcvAppConfig.config_warnings`` por execução
        # (Requirement 8.5). É resetada no início de cada ``read()`` e marcada
        # como ``True`` na primeira chamada a ``_consume_gcv_config_warnings``,
        # de modo que tentativas GCV subsequentes da mesma execução enxerguem
        # tupla vazia em ``metadata.gcv_config_warnings``.
        self._gcv_warnings_consumed: bool = False

    def read(
        self,
        image_path: Path,
        options: ReaderOptions | None = None,
        groundtruth_text: str | None = None,
    ) -> ReadOutcome:
        if not image_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        options = options or ReaderOptions()

        # Reset por execução do consumo único de warnings GCV (Requirement
        # 8.5). Cada chamada a ``read()`` é uma "execução" independente; a
        # primeira tentativa GCV desta execução receberá os warnings, as
        # demais receberão tupla vazia.
        self._gcv_warnings_consumed = False

        recorder = AuditRecorder(self.project_root, image_path)
        image_original = read_image(image_path)

        roi_detector = RoiDetector(self.roi_config)

        # 1. Detecção de formato (usada para escolher a estratégia de ROI).
        # Roda na imagem original sem crop para obter o hint de categoria antes
        # de qualquer ROI.
        if options.category_override in {"table", "text", "ingredient"}:
            detected = DetectedFormat(
                category=options.category_override,  # type: ignore[arg-type]
                score=1.0,
                grid_density=0.0,
                reasoning="categoria forçada pelo usuário",
            )
        else:
            detected = self.format_detector.detect(image_original)
        recorder.set_format_detection(asdict(detected))

        # 2. ROI secundário (conteúdo/texto) — só quando o usuário ativou ROI.
        # Tabela → detecção por grade; texto/ingrediente → text-blobbing.
        category_hint: str | None = detected.category
        image_flat = (
            roi_detector.detect(image_original, category_hint=category_hint)
            if options.roi_enabled
            else image_original
        )

        # 3. Seleciona presets e executa a cascata.
        # "ingredient" no FormatDetector → "ingredients" no PresetRepository
        preset_category = (
            "ingredients" if detected.category == "ingredient" else detected.category
        )
        presets = self.preset_repo.for_category(preset_category)
        if not presets:
            raise RuntimeError(
                f"Nenhum preset disponível para categoria '{detected.category}'."
            )

        attempts_summary: list[dict] = []
        best_score_so_far: QualityScore | None = None
        best_attempt_index: int | None = None
        best_preset_name: str | None = None
        best_preset_kind: str | None = None
        best_ocr_text = ""
        passed = False

        for idx, preset in enumerate(presets, start=1):
            image = image_flat

            artifacts = recorder.start_attempt(idx, preset.name)
            context = PipelineContext(
                input_path=image_path,
                attempt_index=idx,
                preset_name=preset.name,
                recorder=recorder,
                artifacts=artifacts,
            )
            pipeline = self._build_pipeline(preset)
            result = pipeline.execute(image, context)

            evaluator = QualityEvaluator(QualityThresholds.from_dict(preset.quality_thresholds))
            score = evaluator.evaluate(result.ocr_text, result.mean_confidence)
            _pp = self.gcv_postprocessor if preset.kind == "cloud_vision" else self.text_postprocessor
            postprocessed = _pp.postprocess(result.ocr_text) if options.postprocess else result.ocr_text
            recorder.save_attempt_texts(artifacts, result.ocr_text, postprocessed, asdict(score))
            recorder.record_attempt(
                attempt=artifacts,
                stages=[asdict(s) for s in result.stages],
                score=asdict(score),
                passed=score.passed,
            )
            attempts_summary.append(
                {
                    "attempt_index": idx,
                    "preset": preset.name,
                    "passed": score.passed,
                    "score": score.score,
                    "mean_confidence": score.mean_confidence,
                    "keyword_hits": score.keyword_hits,
                    "pipeline_metadata": result.metadata,
                }
            )

            if best_score_so_far is None or score.score > best_score_so_far.score:
                best_score_so_far = score
                best_attempt_index = idx
                best_preset_name = preset.name
                best_preset_kind = preset.kind
                best_ocr_text = result.ocr_text

            if score.passed and options.stop_on_first_pass:
                passed = True
                best_attempt_index = idx
                best_preset_name = preset.name
                best_preset_kind = preset.kind
                best_ocr_text = result.ocr_text
                break

        final_ocr_text = best_ocr_text
        _final_pp = self.gcv_postprocessor if best_preset_kind == "cloud_vision" else self.text_postprocessor
        final_postprocessed = _final_pp.postprocess(final_ocr_text) if options.postprocess else final_ocr_text
        summary_path = recorder.finalize(
            winning_attempt_index=best_attempt_index,
            winning_preset=best_preset_name,
            final_ocr_text=final_ocr_text,
            final_postprocessed_text=final_postprocessed,
            groundtruth_text=groundtruth_text,
        )

        # Análise clínica de ingredientes (apenas para categoria "ingredient")
        ingredient_report: IngredientReport | None = None
        if detected.category == "ingredient" and self.ingredient_analyzer:
            ingredient_report = self.ingredient_analyzer.analyze(
                final_ocr_text, image_name=image_path.name
            )
            recorder.save_ingredient_analysis(
                ocr_tokens=ingredient_report.tokens_found,
                feedback_clinico=ingredient_report.to_text_report(),
                analysis_dict=ingredient_report.to_dict(),
            )

        return ReadOutcome(
            winning_preset=best_preset_name,
            winning_attempt_index=best_attempt_index,
            passed=passed or (best_score_so_far is not None and best_score_so_far.passed),
            detected_format=detected,
            final_ocr_text=final_ocr_text,
            final_postprocessed_text=final_postprocessed,
            attempts=attempts_summary,
            summary_path=summary_path,
            groundtruth_metrics=recorder.manifest.groundtruth_metrics,
            ingredient_report=ingredient_report,
        )

    def _build_pipeline(self, preset: Preset) -> Pipeline:
        ocr_config = OcrConfig(
            lang=str(preset.ocr.get("lang", "por")),
            psm=int(preset.ocr.get("psm", 6)),
            oem=int(preset.ocr.get("oem", 3)),
            extra_config=str(preset.ocr.get("extra_config", "")),
            dual_pass_polarity=bool(preset.ocr.get("dual_pass_polarity", False)),
        )
        if preset.kind == "cloud_vision":
            # Branch GCV: a imagem já foi entregue pelo reader (com ROI quando
            # ``options.roi_enabled``). O ``CloudVisionPipeline`` ignora
            # intencionalmente ``preset.steps`` (Requirements 1.6/1.7) e
            # registra a contagem em ``metadata.ignored_steps_count`` para
            # auditoria. ``ocr_config`` é repassado por uniformidade entre os
            # ``kind`` (a GCV não consome o objeto). As políticas
            # operacionais (``on_failure`` e os warnings de coerção de
            # ``GcvAppConfig``) vêm de ``self._gcv_app_config``; quando o
            # bloco ``gcv`` está ausente de ``app.json`` (``self
            # ._gcv_app_config is None``), caímos no default canônico
            # ``"skip"`` (Requirement 6.1) sem warnings — coerente com o
            # comportamento de ``GcvAppConfig.from_dict(None, ...)``.
            on_failure = (
                self._gcv_app_config.on_failure
                if self._gcv_app_config is not None
                else "skip"
            )
            return CloudVisionPipeline(
                gcv_options=GcvPresetOptions.from_dict(preset.gcv),
                ocr_config=ocr_config,
                client=self._gcv_client,  # type: ignore[arg-type]
                on_failure=on_failure,
                ignored_steps_count=len(preset.steps),
                gcv_config_warnings=self._consume_gcv_config_warnings(),
            )
        if preset.kind == "cell_based":
            cd = preset.cell_detection
            return CellBasedPipeline(
                prep_steps=preset.steps,
                ocr_config=ocr_config,
                horizontal_divisor=int(cd.get("horizontal_divisor", 20)),
                vertical_divisor=int(cd.get("vertical_divisor", 20)),
                min_cell_area_ratio=float(cd.get("min_cell_area_ratio", 0.001)),
                max_cell_area_ratio=float(cd.get("max_cell_area_ratio", 0.5)),
                cell_ocr_psm=int(cd.get("cell_ocr_psm", 7)),
                header_region_ratio=float(cd.get("header_region_ratio", 0.0)),
                vd_column_psm=int(cd.get("vd_column_psm", 8)),
            )
        return LinearPipeline(steps=preset.steps, ocr_config=ocr_config)

    def _consume_gcv_config_warnings(self) -> tuple[str, ...]:
        """Devolve warnings GCV na primeira chamada e tupla vazia depois.

        Implementa o consumo único exigido pelo Requirement 8.5: os warnings
        produzidos por ``GcvAppConfig.from_dict`` (coerções de
        ``max_requests_per_minute``, ``on_failure`` etc.) são propagados
        para ``metadata.gcv_config_warnings`` apenas na primeira tentativa
        GCV de cada execução do reader. Tentativas subsequentes — ainda na
        mesma execução — recebem tupla vazia para evitar ruído duplicado
        no ``_summary.json``.

        A flag ``_gcv_warnings_consumed`` é resetada no início de
        ``read()``, garantindo que execuções independentes tenham seu
        próprio ciclo de warnings.

        Retorna tupla vazia também quando o bloco ``gcv`` está ausente de
        ``app.json`` (``self._gcv_app_config is None``), caso em que não há
        warnings a propagar.
        """

        if self._gcv_warnings_consumed or self._gcv_app_config is None:
            return ()
        self._gcv_warnings_consumed = True
        return self._gcv_app_config.config_warnings


def build_default_reader(project_root: Path) -> NutritionReader:
    """Constrói um NutritionReader a partir dos arquivos padrão em config/."""
    config_root = project_root / "config"
    presets_root = config_root / "presets"
    routing_cfg_path = config_root / "routing.json"
    app_cfg_path = config_root / "app.json"

    routing_data = _load_json(routing_cfg_path) or {}
    app_data = _load_json(app_cfg_path) or {}

    preset_repo = PresetRepository(presets_root)
    format_detector = FormatDetector(FormatDetectorConfig.from_dict(routing_data))

    roi_cfg_dict = app_data.get("roi", {})
    roi_config = RoiDetectionConfig(
        prototxt_path=_opt_path(roi_cfg_dict.get("prototxt_path"), project_root),
        weights_path=_opt_path(roi_cfg_dict.get("weights_path"), project_root),
        pb_path=_opt_path(roi_cfg_dict.get("pb_path"), project_root) or _autodetect(project_root, "*.pb"),
        pbtxt_path=_opt_path(roi_cfg_dict.get("pbtxt_path"), project_root) or _autodetect(project_root, "*.pbtxt"),
        confidence_threshold=float(roi_cfg_dict.get("confidence_threshold", 0.2)),
        target_class_names=tuple(
            roi_cfg_dict.get(
                "target_class_names",
                ("diningtable", "tvmonitor", "bottle", "tabela_nutricional"),
            )
        ),
        use_contour_fallback=bool(roi_cfg_dict.get("use_contour_fallback", True)),
    )
    # Wiring GCV opcional (Requirements 4.3, 4.4, 14.3): instancia o cliente
    # somente quando há ao menos um preset ``cloud_vision`` em
    # ``config/presets/``. Sem preset GCV → ``gcv_client=None`` e nenhuma
    # credencial é exigida. Import tardio mantém o módulo livre de dependência
    # do SDK ``google-cloud-vision`` em tempo de carga do reader.
    gcv_app_config = GcvAppConfig.from_dict(app_data.get("gcv"), project_root)
    has_gcv_preset = any(p.kind == "cloud_vision" for p in preset_repo.all())
    gcv_client = None
    if has_gcv_preset:
        from ocr.cloud_vision.client import GcvClient  # noqa: PLC0415
        gcv_client = GcvClient.build(gcv_app_config, project_root)

    return NutritionReader(
        project_root, preset_repo, format_detector, roi_config,
        gcv_app_config=gcv_app_config,
        gcv_client=gcv_client,
    )


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _opt_path(value: str | None, project_root: Path) -> Path | None:
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = project_root / p
    return p if p.exists() else None


def _autodetect(project_root: Path, pattern: str) -> Path | None:
    matches = sorted(project_root.glob(pattern))
    return matches[0] if matches else None
