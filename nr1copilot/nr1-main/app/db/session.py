"""
Async SQLAlchemy session and engine setup for Render.com/PostgreSQL.
"""
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings

DATABASE_URL = os.getenv("DATABASE_URL") or getattr(settings, "DATABASE_URL", None)
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable must be set for deployment.")

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

async def get_async_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
