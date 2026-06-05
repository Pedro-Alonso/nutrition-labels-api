from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    display_name: str | None = None


class ScanSummaryResponse(BaseModel):
    id: str
    created_at: datetime
    detected_format: str | None
    passed: bool
    winning_preset: str | None
    risco_global: str | None

    model_config = {"from_attributes": True}


class PaginatedScans(BaseModel):
    items: list[ScanSummaryResponse]
    total: int
    page: int
    per_page: int
