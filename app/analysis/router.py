from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.analysis.schemas import AnalyzeResponse
from app.analysis.service import AnalysisService
from app.core.config import get_settings

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
    reader = request.app.state.reader
    return AnalysisService(reader)


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze_label(
    request: Request,
    file: UploadFile = File(..., description="Imagem do rótulo (JPEG, PNG, WEBP, BMP; max 10MB)"),
    category_override: str | None = Form(None),
    roi_enabled: bool = Form(True),
    stop_on_first_pass: bool = Form(True),
    postprocess: bool = Form(True),
):
    """Analisa uma foto de rótulo alimentício e retorna informação nutricional estruturada."""
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    image_bytes = await file.read()

    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Arquivo muito grande. Limite: {settings.max_upload_size_mb}MB.",
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato de arquivo não suportado. Use JPEG, PNG, WEBP ou BMP.",
        )

    service = _get_analysis_service(request)

    try:
        result = service.analyze(
            image_bytes=image_bytes,
            category_override=category_override,
            roi_enabled=roi_enabled,
            stop_on_first_pass=stop_on_first_pass,
            postprocess=postprocess,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return result


@router.get("/presets", tags=["analysis"])
async def list_presets(request: Request):
    """Lista os presets disponíveis por categoria. Não requer autenticação."""
    reader = request.app.state.reader
    preset_repo = reader._preset_repo

    result: dict[str, list[dict]] = {"table": [], "text": [], "ingredients": []}
    for preset in preset_repo.all():
        entry = {
            "name": preset.name,
            "description": preset.description,
            "kind": preset.kind,
            "priority": preset.priority,
        }
        cat = preset.category
        if cat in result:
            result[cat].append(entry)
        else:
            result[cat] = [entry]

    return result
