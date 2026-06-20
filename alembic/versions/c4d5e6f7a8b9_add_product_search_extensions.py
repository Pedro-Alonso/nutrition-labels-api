"""add_product_search_extensions

Revision ID: c4d5e6f7a8b9
Revises: b3c4d5e6f7a8
Create Date: 2026-06-19 18:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "c4d5e6f7a8b9"
down_revision: Union[str, None] = "b3c4d5e6f7a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    op.execute("""
        CREATE OR REPLACE FUNCTION f_unaccent(text)
        RETURNS text
        LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
        AS $func$
            SELECT public.unaccent('public.unaccent', $1)
        $func$
    """)
    op.execute("""
        CREATE INDEX ix_products_name_brand_trgm
        ON products
        USING gin (
            f_unaccent(COALESCE(name, '') || ' ' || COALESCE(brand, '')) gin_trgm_ops
        )
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_products_name_brand_trgm")
    op.execute("DROP FUNCTION IF EXISTS f_unaccent(text)")
    op.execute("DROP EXTENSION IF EXISTS unaccent")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
