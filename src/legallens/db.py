"""Async PostgreSQL session helper.

One engine per process, created lazily on first use. SQLAlchemy's async
engine already pools connections, so we don't add a layer of our own.

We use asyncpg (not psycopg) because asyncpg is significantly faster on
bulk INSERTs (the ingest hot path).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from legallens.config import get_settings


@lru_cache(maxsize=1)
def _engine() -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        pool_size=10,
        max_overflow=10,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def _sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(_engine(), expire_on_commit=False)


@asynccontextmanager
async def session() -> AsyncIterator[AsyncSession]:
    """`async with session() as s: ...` — handles commit/rollback for you."""
    sm = _sessionmaker()
    async with sm() as s:
        try:
            yield s
            await s.commit()
        except Exception:
            await s.rollback()
            raise


async def close_engine() -> None:
    """Called from FastAPI lifespan shutdown."""
    await _engine().dispose()
