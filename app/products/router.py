from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.schemas import IngredientAnalysisSchema
from app.analysis.service import AnalysisService
from app.core.config import get_settings
from app.core.database import get_db
from app.core.dependencies import get_current_user_id
from app.core.security import verify_access_token
from app.products import service as product_service
from app.products.llm_service import (
    clean_ingredients_text,
    clean_nutritional_table,
)
from app.products.schemas import (
    IngredientsData,
    OcrPreviewResponse,
    PaginatedProducts,
    ProductCreateRequest,
    ProductResponse,
    ProductSearchItem,
    ProductUpdateRequest,
    SummaryResponse,
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


async def _get_optional_user(request: Request, db: AsyncSession):
    """Retorna User se token Bearer válido presente; None caso contrário."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    payload = verify_access_token(auth.split(" ", 1)[1])
    if not payload:
        return None
    from app.users import service as user_service
    return await user_service.get_user_by_id(db, payload["sub"])


_PHRASE_LIKE_MIN_LENGTH = 60


def _looks_like_single_phrase(items: list[str]) -> bool:
    """Detecta um item único e longo sem separadores — provável frase/recusa
    do OCR ou da LLM, não um ingrediente real."""
    return len(items) == 1 and len(items[0]) > _PHRASE_LIKE_MIN_LENGTH


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
# GET /search — deve vir ANTES de rotas com {barcode}
# -----------------------------------------------------------------------

@router.get("/search", response_model=PaginatedProducts)
async def search_products_endpoint(
    q: str = Query("", min_length=0, max_length=200),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    products, total = await product_service.search_products(
        db, q, page=page, per_page=per_page
    )
    return PaginatedProducts(
        items=[
            ProductSearchItem(
                barcode=p.barcode,
                name=p.name,
                brand=p.brand,
                created_at=p.created_at,
            )
            for p in products
        ],
        total=total,
        page=page,
        per_page=per_page,
    )


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

    settings = get_settings()
    items = list(product.ingredient_list.items)

    # 1. Limpeza pré-análise: remove alegações e ruído OCR da lista de ingredientes
    if settings.groq_api_key:
        raw_text = ", ".join(items)
        cleaned_text = await clean_ingredients_text(raw_text, settings.groq_api_key)
        cleaned_items = [t.strip() for t in cleaned_text.split(",") if t.strip()]
        if cleaned_items:
            items = cleaned_items

    # 2. Análise ontológica
    analysis = product_service._compute_analysis(analyzer, items, barcode)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Produto não possui lista de ingredientes cadastrada.",
        )

    # 3. Resumo em linguagem natural (personalizado se usuário autenticado, com cache)
    user = await _get_optional_user(request, db)
    analysis.natural_language_summary = await product_service.get_or_create_summary(
        db, product, analysis, user, settings.groq_api_key
    )

    return analysis


# -----------------------------------------------------------------------
# GET /{barcode}/summary — deve vir ANTES de GET /{barcode}
# -----------------------------------------------------------------------

@router.get("/{barcode}/summary", response_model=SummaryResponse)
async def get_product_summary(
    barcode: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Retorna resumo personalizado enxuto. Auth opcional: personaliza pelo perfil do usuário."""
    product = await product_service.get_by_barcode(db, barcode)
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Produto não encontrado.")

    analyzer = _get_analyzer(request)
    if analyzer is None or product.ingredient_list is None or not product.ingredient_list.items:
        return SummaryResponse(summary=None, diabetes_type=None, language_level=None, risco_global=None)

    items = list(product.ingredient_list.items)
    analysis = product_service._compute_analysis(analyzer, items, barcode)
    if analysis is None:
        return SummaryResponse(summary=None, diabetes_type=None, language_level=None, risco_global=None)

    user = await _get_optional_user(request, db)
    diabetes_type = getattr(user, "diabetes_type", None)
    language_level = getattr(user, "language_level", None)

    settings = get_settings()
    summary_text = await product_service.get_or_create_summary(
        db, product, analysis, user, settings.groq_api_key
    )

    if summary_text is None and (diabetes_type is not None or language_level is not None):
        summary_text = await product_service.get_or_create_summary(
            db, product, analysis, None, settings.groq_api_key
        )
        if summary_text is not None:
            diabetes_type = None
            language_level = None

    return SummaryResponse(
        summary=summary_text,
        diabetes_type=diabetes_type,
        language_level=language_level,
        risco_global=analysis.risco_global,
    )


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
    response = product_service.build_product_response(product, analyzer)

    if response.analysis is not None:
        settings = get_settings()
        user = await _get_optional_user(request, db)
        response.analysis.natural_language_summary = await product_service.get_or_create_summary(
            db, product, response.analysis, user, settings.groq_api_key
        )

    return response


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

        # Extração estruturada via LLM (preferida); parser regex é o fallback.
        if settings.groq_api_key:
            nt_data = await clean_nutritional_table(
                outcome.final_postprocessed_text, settings.groq_api_key
            )
        if nt_data is None:
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

        if outcome.ingredient_report is not None and outcome.ingredient_report.tokens_found:
            items = list(outcome.ingredient_report.tokens_found)
        else:
            items = product_service.split_ingredient_text(outcome.final_ocr_text)

        # Limpeza LLM opcional: remove alegações e ruído OCR antes do preview.
        if settings.groq_api_key and items:
            cleaned = await clean_ingredients_text(", ".join(items), settings.groq_api_key)
            cleaned_items = [t.strip() for t in cleaned.split(",") if t.strip()]
            if cleaned_items:
                items = cleaned_items

        # Descarta item único sem separadores que pareça frase/recusa (OCR
        # ilegível), em vez de salvá-lo como "ingrediente".
        if _looks_like_single_phrase(items):
            items = []

        ing_data = IngredientsData(items=items) if items else None

    return OcrPreviewResponse(
        barcode=barcode,
        nutritional_table=nt_data,
        ingredients=ing_data,
    )


# -----------------------------------------------------------------------
# POST /{barcode}/scan — deve vir ANTES de POST /{barcode}
# -----------------------------------------------------------------------

@router.post("/{barcode}/scan", response_model=ProductResponse)
async def scan_product(
    barcode: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Registra a leitura de um produto já cadastrado no histórico do usuário (scan-on-read)."""
    product = await product_service.get_by_barcode(db, barcode)
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Produto não encontrado.")

    analyzer = _get_analyzer(request)
    response = product_service.build_product_response(product, analyzer)

    settings = get_settings()
    user = await _get_optional_user(request, db)

    if response.analysis is not None:
        response.analysis.natural_language_summary = await product_service.get_or_create_summary(
            db, product, response.analysis, user, settings.groq_api_key
        )

    await product_service.record_product_scan(
        db, user_id, barcode, response.analysis, name=product.name, brand=product.brand
    )

    return response


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
    response = product_service.build_product_response(product, analyzer)

    settings = get_settings()
    user = await _get_optional_user(request, db)

    # Resumo em linguagem natural (personalizado, com cache).
    if response.analysis is not None:
        response.analysis.natural_language_summary = await product_service.get_or_create_summary(
            db, product, response.analysis, user, settings.groq_api_key
        )

    # Persiste um Scan para o histórico do usuário.
    await product_service.record_product_scan(
        db, user_id, barcode, response.analysis, name=product.name, brand=product.brand
    )

    return response


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
