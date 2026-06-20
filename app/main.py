from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.analysis.router import router as analysis_router
from app.auth.router import router as auth_router
from app.core.config import get_settings
from app.core.limiter import limiter
from app.core.middleware import LoggingMiddleware
from app.products.router import router as products_router
from app.users.router import router as users_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando carregamento do motor OCR...")
    from ocr_engine import build_reader
    app.state.reader = build_reader()
    logger.info("Motor OCR carregado: %s", type(app.state.reader).__name__)

    cleanup_task = asyncio.create_task(_cleanup_expired_tokens())

    yield

    cleanup_task.cancel()
    logger.info("Shutdown: motor OCR não possui recursos a liberar.")


async def _cleanup_expired_tokens() -> None:
    from datetime import datetime, timezone

    from sqlalchemy import delete

    from app.auth.models import RevokedToken
    from app.core.database import get_session_factory

    while True:
        await asyncio.sleep(3600)
        try:
            async with get_session_factory()() as db:
                await db.execute(
                    delete(RevokedToken).where(
                        RevokedToken.expires_at < datetime.now(timezone.utc)
                    )
                )
                await db.commit()
        except Exception:
            logger.exception("Erro ao limpar tokens revogados expirados.")


settings = get_settings()

app = FastAPI(
    title="Rótulos Backend",
    description="API de análise de rótulos alimentícios para pacientes diabéticos.",
    version="1.4.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(LoggingMiddleware)

origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users_router, prefix="/api/v1/users", tags=["users"])
app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])
app.include_router(products_router, prefix="/api/v1/products", tags=["products"])


@app.get("/api/v1/health", tags=["health"])
async def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "version": "1.1.0",
        "dependencies": {
            "database": "not_checked",
            "tesseract": "not_checked",
            "gcv_configured": settings.google_application_credentials is not None,
        },
    }
