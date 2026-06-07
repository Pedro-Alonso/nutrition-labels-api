"""add_products_nutritional_tables_ingredient_lists

Revision ID: c3d4e5f6a7b8
Revises: a1b2c3d4e5f6
Create Date: 2026-06-06 12:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "products",
        sa.Column("barcode", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("brand", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by_user_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("barcode"),
    )
    op.create_index("ix_products_created_by_user_id", "products", ["created_by_user_id"])

    op.create_table(
        "nutritional_tables",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("product_barcode", sa.String(), nullable=False),
        sa.Column("portion_description", sa.String(), nullable=True),
        sa.Column("columns", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rows", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_by_user_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["product_barcode"], ["products.barcode"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["updated_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("product_barcode", name="uq_nutritional_tables_product_barcode"),
    )

    op.create_table(
        "ingredient_lists",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("product_barcode", sa.String(), nullable=False),
        sa.Column("items", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_by_user_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["product_barcode"], ["products.barcode"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["updated_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("product_barcode", name="uq_ingredient_lists_product_barcode"),
    )


def downgrade() -> None:
    op.drop_table("ingredient_lists")
    op.drop_table("nutritional_tables")
    op.drop_index("ix_products_created_by_user_id", table_name="products")
    op.drop_table("products")
