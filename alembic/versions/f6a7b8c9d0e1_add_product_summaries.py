"""add_product_summaries

Revision ID: f6a7b8c9d0e1
Revises: d1e2f3a4b5c6
Create Date: 2026-06-12 10:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "product_summaries",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("barcode", sa.String(), nullable=False),
        sa.Column("diabetes_type", sa.String(), nullable=True),
        sa.Column("language_level", sa.String(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["barcode"], ["products.barcode"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "barcode", "diabetes_type", "language_level", name="uq_product_summary_cache_key"
        ),
    )
    op.create_index(
        op.f("ix_product_summaries_barcode"), "product_summaries", ["barcode"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_product_summaries_barcode"), table_name="product_summaries")
    op.drop_table("product_summaries")
