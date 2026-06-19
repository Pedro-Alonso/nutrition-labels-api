from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.analysis.schemas import IngredientAnalysisSchema


class NutritionalRowData(BaseModel):
    nutrient: str
    values: list[str]


class NutritionalTableData(BaseModel):
    portion_description: str | None = None
    columns: list[str]
    rows: list[NutritionalRowData]


class IngredientsData(BaseModel):
    items: list[str]


class ProductResponse(BaseModel):
    barcode: str
    name: str | None
    brand: str | None
    nutritional_table: NutritionalTableData | None
    ingredients: IngredientsData | None
    analysis: IngredientAnalysisSchema | None
    created_at: datetime
    updated_at: datetime


class ProductCreateRequest(BaseModel):
    name: str | None = None
    brand: str | None = None
    nutritional_table: NutritionalTableData | None = None
    ingredients: IngredientsData | None = None


class ProductUpdateRequest(BaseModel):
    name: str | None = None
    brand: str | None = None
    nutritional_table: NutritionalTableData | None = None
    ingredients: IngredientsData | None = None


class SummaryResponse(BaseModel):
    summary: str | None
    diabetes_type: str | None
    language_level: str | None
    risco_global: str | None


class OcrPreviewResponse(BaseModel):
    barcode: str
    nutritional_table: NutritionalTableData | None
    ingredients: IngredientsData | None
