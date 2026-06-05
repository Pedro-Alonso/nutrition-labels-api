from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fase 0: startup mínimo — reader OCR será carregado na Fase 2
    yield


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
