# ─────────────────────────────────────────────
# Base: Python 3.11 slim (Debian Bookworm)
# NÃO usar alpine: OpenCV precisa de glibc (musl quebra)
# ─────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─────────────────────────────────────────────
# Dependências de sistema
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-por \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# ─────────────────────────────────────────────
# Instalar dependências Python
# ─────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Copiar código da aplicação
# ─────────────────────────────────────────────
COPY app/ ./app/
COPY ocr_engine/ ./ocr_engine/
COPY alembic/ ./alembic/
COPY alembic.ini .

# ─────────────────────────────────────────────
# Usuário não-root
# ─────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8000

# Um worker: NutritionReader é singleton em memória.
# Para escalar, prefira múltiplos containers (horizontal).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ─────────────────────────────────────────────
# Stage de teste (docker-compose.test.yml)
# ─────────────────────────────────────────────
FROM base AS test

USER root

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY tests/ ./tests/
COPY pytest.ini .

USER appuser
