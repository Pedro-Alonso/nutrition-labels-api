from __future__ import annotations

from sqlalchemy import delete as sql_delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.users.models import User
from app.analysis.models import Scan


async def get_user_by_id(db: AsyncSession, user_id: str) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def update_user(
    db: AsyncSession, user: User, display_name: str | None
) -> User:
    if display_name is not None:
        user.display_name = display_name
    await db.commit()
    await db.refresh(user)
    return user


async def get_scan_by_id(db: AsyncSession, scan_id: str, user_id: str) -> Scan | None:
    result = await db.execute(
        select(Scan).where(Scan.id == scan_id, Scan.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def delete_scan(db: AsyncSession, scan_id: str, user_id: str) -> bool:
    scan = await get_scan_by_id(db, scan_id, user_id)
    if scan is None:
        return False
    await db.delete(scan)
    await db.commit()
    return True


async def delete_user(db: AsyncSession, user_id: str) -> None:
    user = await get_user_by_id(db, user_id)
    if user:
        await db.execute(sql_delete(Scan).where(Scan.user_id == user_id))
        await db.delete(user)
        await db.commit()


async def list_user_scans(
    db: AsyncSession, user_id: str, page: int = 1, per_page: int = 20
) -> tuple[list[Scan], int]:
    offset = (page - 1) * per_page

    total_result = await db.execute(
        select(func.count()).select_from(Scan).where(Scan.user_id == user_id)
    )
    total = total_result.scalar_one()

    scans_result = await db.execute(
        select(Scan)
        .where(Scan.user_id == user_id)
        .order_by(Scan.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    return list(scans_result.scalars()), total
