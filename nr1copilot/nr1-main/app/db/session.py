
"""
Production-grade database session management
Supports both PostgreSQL (SQLAlchemy) and MongoDB (Motor)
"""

import logging
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import get_settings

logger = logging.getLogger("database")
settings = get_settings()

# SQLAlchemy setup
Base = declarative_base()

# Create async engine
engine = create_async_engine(
    settings.database_url_async,
    echo=not settings.is_production,
    poolclass=NullPool if settings.is_production else None,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# MongoDB client
mongo_client: Optional[AsyncIOMotorClient] = None
mongo_db = None


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def connect_to_mongo():
    """Connect to MongoDB"""
    global mongo_client, mongo_db
    
    if not settings.MONGODB_URI:
        logger.warning("MongoDB URI not provided, skipping MongoDB connection")
        return
    
    try:
        mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
        mongo_db = mongo_client.get_default_database()
        
        # Test connection
        await mongo_client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close MongoDB connection"""
    global mongo_client
    
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")


def get_mongo_db():
    """Get MongoDB database instance"""
    if mongo_db is None:
        raise RuntimeError("MongoDB not connected")
    return mongo_db


async def init_database():
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


async def close_database():
    """Close database connections"""
    await engine.dispose()
    await close_mongo_connection()
    logger.info("Database connections closed")
