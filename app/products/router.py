from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.schemas import IngredientAnalysisSchema
from app.analysis.service import AnalysisService
from app.core.config import get_settings
from app.core.database import get_db
from app.core.dependencies import get_current_user_id
from app.products import service as product_service
from app.products.schemas import (
    IngredientsData,
    OcrPreviewResponse,
    ProductCreateRequest,
    ProductResponse,
    ProductUpdateRequest,
)

router = APIRouter()

SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


def _get_analysis_service(request: Request) -> AnalysisService:
    return AnalysisService(request.app.state.reader)


def _get_analyzer(request: Request):
    reader = request.app.state.reader
    return getattr(reader, "ingredient_analyzer", None)


async def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    image_bytes = await upload.read()
    if len(image_bytes) > max_bytes:
        settings = get_settings()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Arquivo muito grande. Limite: {settings.max_upload_size_mb}MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Arquivo vazio.")
    ct = (upload.content_type or "").lower()
    if ct and ct not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato de arquivo não suportado. Use JPEG, PNG, WEBP ou BMP.",
        )
    return image_bytes


# -----------------------------------------------------------------------
# GET /{barcode}/analysis — deve vir ANTES de GET /{barcode}
# -----------------------------------------------------------------------

@router.get("/{barcode}/analysis", response_model=IngredientAnalysisSchema)
async def get_product_analysis(
    barcode: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Retorna análise clínica DM para os ingredientes do produto. Não requer autenticação."""
    analyzer = _get_analyzer(request)
    if analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analisador de ingredientes não disponível (ontologia ausente).",
        )

    product = await product_service.get_by_barcode(db, barcode)
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Produto não encontrado.")

    if product.ingredient_list is None or not product.ingredient_list.items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Produto não possui lista de ingredientes cadastrada.",
        )

    analysis = product_service._compute_analysis(
        analyzer, list(product.ingredient_list.items), barcode
    )
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Produto não possui lista de ingredientes cadastrada.",
        )
    return analysis


# -----------------------------------------------------------------------
# GET /{barcode}
# -----------------------------------------------------------------------

@router.get("/{barcode}", response_model=ProductResponse)
async def get_product(
    barcode: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Retorna dados completos de um produto pelo código de barras. Não requer autenticação."""
    product = await product_service.get_by_barcode(db, barcode)
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Produto não encontrado.")
    analyzer = _get_analyzer(request)
    return product_service.build_product_response(product, analyzer)


# -----------------------------------------------------------------------
# POST /{barcode}/ocr — deve vir ANTES de POST /{barcode}
# -----------------------------------------------------------------------

@router.post("/{barcode}/ocr", response_model=OcrPreviewResponse)
async def ocr_preview(
    barcode: str,
    request: Request,
    image_nutrition: UploadFile | None = File(None),
    image_ingredients: UploadFile | None = File(None),
    user_id: str = Depends(get_current_user_id),
):
    """Processa imagens via OCR e retorna preview estruturado. Não salva nada no banco."""
    if image_nutrition is None and image_ingredients is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Pelo menos uma imagem deve ser enviada (image_nutrition ou image_ingredients).",
        )

    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    ocr_service = _get_analysis_service(request)

    nt_data = None
    ing_data = None

    if image_nutrition is not None:
        nutrition_bytes = await _read_upload(image_nutrition, max_bytes)
        try:
            outcome = await asyncio.to_thread(
                ocr_service.read_outcome,
                nutrition_bytes,
                "table",
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        nt_data = product_service.parse_postprocessed_to_nutritional_table(
            outcome.final_postprocessed_text
        )

    if image_ingredients is not None:
        ingredients_bytes = await _read_upload(image_ingredients, max_bytes)
        try:
            outcome = await asyncio.to_thread(
                ocr_service.read_outcome,
                ingredients_bytes,
                "ingredient",
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        if outcome.ingredient_report is not None:
            items = list(outcome.ingredient_report.tokens_found)
        else:
            raw = outcome.final_ocr_text.strip()
            items = [t.strip() for t in re.split(r"[,;]", raw) if t.strip()] if raw else []

        ing_data = IngredientsData(items=items) if items else None

    return OcrPreviewResponse(
        barcode=barcode,
        nutritional_table=nt_data,
        ingredients=ing_data,
    )


# -----------------------------------------------------------------------
# POST /{barcode}
# -----------------------------------------------------------------------

@router.post("/{barcode}", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    barcode: str,
    body: ProductCreateRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Cria um novo produto na base comunitária. Requer autenticação."""
    existing = await product_service.get_by_barcode(db, barcode)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Código de barras já cadastrado. Use PUT para atualizar.",
        )

    product = await product_service.create_product(db, barcode, user_id, body)
    analyzer = _get_analyzer(request)
    return product_service.build_product_response(product, analyzer)


# -----------------------------------------------------------------------
# PUT /{barcode}
# -----------------------------------------------------------------------

@router.put("/{barcode}", response_model=ProductResponse)
async def update_product(
    barcode: str,
    body: ProductUpdateRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Atualiza um produto existente (patch: campos ausentes mantêm valor atual). Requer autenticação."""
    product = await product_service.get_by_barcode(db, barcode)
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Produto não encontrado.")

    product = await product_service.update_product(db, product, user_id, body)
    analyzer = _get_analyzer(request)
    return product_service.build_product_response(product, analyzer)
