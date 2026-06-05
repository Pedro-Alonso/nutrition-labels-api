"""initial_schema

Revision ID: f372a4df148e
Revises:
Create Date: 2026-06-04 23:11:53.067376

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = 'f372a4df148e'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "scans",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("image_hash", sa.String(64), nullable=False),
        sa.Column("detected_format", sa.String(), nullable=True),
        sa.Column("winning_preset", sa.String(), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("risco_global", sa.String(), nullable=True),
        sa.Column("result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_scans_user_id"), "scans", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_scans_user_id"), table_name="scans")
    op.drop_table("scans")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
