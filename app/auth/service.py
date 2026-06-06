from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import RevokedToken
from app.core.security import hash_password, verify_password
from app.users.models import User


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email.lower()))
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    email: str,
    password: str,
    display_name: str | None = None,
) -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=email.lower(),
        password_hash=hash_password(password),
        display_name=display_name,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate_user(
    db: AsyncSession, email: str, password: str
) -> User | None:
    user = await get_user_by_email(db, email)
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


async def revoke_token(
    db: AsyncSession, jti: str, user_id: str, expires_at: datetime
) -> None:
    stmt = (
        insert(RevokedToken)
        .values(jti=jti, user_id=user_id, expires_at=expires_at)
        .on_conflict_do_nothing(index_elements=["jti"])
    )
    await db.execute(stmt)
    await db.commit()


async def change_password(
    db: AsyncSession, user: User, current: str, new: str
) -> bool:
    if not verify_password(current, user.password_hash):
        return False
    user.password_hash = hash_password(new)
    user.updated_at = datetime.now(timezone.utc)
    await db.commit()
    return True
