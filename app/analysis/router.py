from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.models import Scan
from app.analysis.schemas import AnalyzeResponse
from app.analysis.service import AnalysisService
from app.core.config import get_settings
from app.core.database import get_db
from app.core.dependencies import get_current_user_id

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
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
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

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arquivo vazio.",
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato de arquivo não suportado. Use JPEG, PNG, WEBP ou BMP.",
        )

    image_hash = hashlib.sha256(image_bytes).hexdigest()

    cached_result = await db.execute(
        select(Scan)
        .where(Scan.user_id == user_id, Scan.image_hash == image_hash)
        .order_by(Scan.created_at.desc())
        .limit(1)
    )
    cached_scan = cached_result.scalar_one_or_none()
    if cached_scan:
        result = dict(cached_scan.result_json)
        result["cache_hit"] = True
        result["scan_id"] = cached_scan.id
        return result

    service = _get_analysis_service(request)

    try:
        result = await asyncio.to_thread(
            service.analyze,
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

    result["cache_hit"] = False
    scan_id = str(uuid.uuid4())
    result["scan_id"] = scan_id

    ingredient_analysis = result.get("ingredient_analysis")
    risco_global = ingredient_analysis.get("risco_global") if ingredient_analysis else None

    scan = Scan(
        id=scan_id,
        user_id=user_id,
        image_hash=image_hash,
        detected_format=result.get("detected_format", {}).get("category"),
        winning_preset=result.get("winning_preset"),
        passed=result.get("passed", False),
        risco_global=risco_global,
        result_json=result,
        created_at=datetime.now(timezone.utc),
    )
    db.add(scan)
    await db.commit()

    return result


@router.get("/presets", tags=["analysis"])
async def list_presets(request: Request):
    """Lista os presets disponíveis por categoria. Não requer autenticação."""
    reader = request.app.state.reader
    preset_repo = reader.preset_repo

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
