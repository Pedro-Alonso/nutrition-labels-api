from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Product(Base):
    __tablename__ = "products"

    barcode: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    brand: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    created_by_user_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    nutritional_table: Mapped[NutritionalTable | None] = relationship(
        back_populates="product", uselist=False, lazy="selectin", cascade="all, delete-orphan"
    )
    ingredient_list: Mapped[IngredientList | None] = relationship(
        back_populates="product", uselist=False, lazy="selectin", cascade="all, delete-orphan"
    )


class NutritionalTable(Base):
    __tablename__ = "nutritional_tables"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    product_barcode: Mapped[str] = mapped_column(
        String, ForeignKey("products.barcode", ondelete="CASCADE"), nullable=False, unique=True
    )
    portion_description: Mapped[str | None] = mapped_column(String, nullable=True)
    columns: Mapped[list] = mapped_column(JSONB, nullable=False)
    rows: Mapped[list] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_by_user_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    product: Mapped[Product] = relationship(back_populates="nutritional_table")


class IngredientList(Base):
    __tablename__ = "ingredient_lists"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    product_barcode: Mapped[str] = mapped_column(
        String, ForeignKey("products.barcode", ondelete="CASCADE"), nullable=False, unique=True
    )
    items: Mapped[list] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_by_user_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    product: Mapped[Product] = relationship(back_populates="ingredient_list")
