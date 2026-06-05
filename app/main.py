from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.analysis.router import router as analysis_router
from app.core.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando carregamento do motor OCR...")
    from ocr_engine import build_reader
    app.state.reader = build_reader()
    logger.info("Motor OCR carregado: %s", type(app.state.reader).__name__)
    yield
    logger.info("Shutdown: motor OCR não possui recursos a liberar.")


app = FastAPI(
    title="Rótulos Backend",
    description="API de análise de rótulos alimentícios para pacientes diabéticos.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])


@app.get("/api/v1/health", tags=["health"])
async def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "version": "1.0.0",
        "dependencies": {
            "database": "not_checked",
            "tesseract": "not_checked",
            "gcv_configured": settings.google_application_credentials is not None,
        },
    }
