from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DetectedFormatSchema(BaseModel):
    category: str
    score: float
    grid_density: float = 0.0
    reasoning: str = ""


class AttemptSchema(BaseModel):
    attempt_index: int
    preset: str
    passed: bool
    score: float
    mean_confidence: float
    text_length: int
    keyword_hits: int


class IngredientItemSchema(BaseModel):
    nome_lido: str
    classe: str
    risco: str
    alerta: str
    indice_glicemico: Any = None
    nota_clinica: str | None = None


class IngredientAnalysisSchema(BaseModel):
    risco_global: str
    ingredientes_identificados: list[IngredientItemSchema]
    nao_identificados: list[str]
    high_risk_ingredients: list[str] = []
    safe_sweeteners: list[str] = []


class AnalyzeResponse(BaseModel):
    scan_id: str | None = None
    cache_hit: bool = False
    detected_format: DetectedFormatSchema
    winning_preset: str | None
    winning_attempt_index: int | None
    passed: bool
    final_ocr_text: str
    final_postprocessed_text: str
    attempts: list[AttemptSchema]
    ingredient_analysis: IngredientAnalysisSchema | None = None
