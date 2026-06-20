from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    is_guest: bool = False
    language_level: str | None = None
    diabetes_type: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    display_name: str | None = None
    language_level: str | None = None
    diabetes_type: str | None = None


class ScanSummaryResponse(BaseModel):
    id: str
    created_at: datetime
    detected_format: str | None
    passed: bool
    winning_preset: str | None
    risco_global: str | None
    name: str | None = None
    brand: str | None = None

    model_config = {"from_attributes": True}


class ScanDetailResponse(BaseModel):
    id: str
    created_at: datetime
    detected_format: str | None
    passed: bool
    winning_preset: str | None
    risco_global: str | None
    result_json: dict

    model_config = {"from_attributes": True}


class PaginatedScans(BaseModel):
    items: list[ScanSummaryResponse]
    total: int
    page: int
    per_page: int
