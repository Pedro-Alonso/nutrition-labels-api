from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://rotulos_user:rotulos_pass@db:5432/rotulos_db"
    secret_key: str = "dev-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 30

    google_application_credentials: str | None = None
    groq_api_key: str | None = None

    max_upload_size_mb: int = 10

    tessdata_prefix: str | None = None

    allowed_origins: str = "*"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
