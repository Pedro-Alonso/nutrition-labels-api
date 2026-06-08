"""add_language_level_diabetes_type_to_users

Revision ID: d1e2f3a4b5c6
Revises: c3d4e5f6a7b8
Create Date: 2026-06-08 12:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("language_level", sa.String(), nullable=True))
    op.add_column("users", sa.Column("diabetes_type", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "diabetes_type")
    op.drop_column("users", "language_level")
