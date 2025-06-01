
"""
Database Session Management
Handles MongoDB and Redis connections with proper error handling
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis
from app.config import get_settings

logger = logging.getLogger("database")
settings = get_settings()

# Global database instances
mongodb_client: AsyncIOMotorClient = None
database = None
redis_client = None

async def connect_to_mongo():
    """Create database connection"""
    global mongodb_client, database
    
    try:
        mongodb_client = AsyncIOMotorClient(settings.MONGODB_URI)
        database = mongodb_client[settings.DATABASE_NAME]
        
        # Test connection
        await database.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        # In development, continue without database
        if settings.ENVIRONMENT == "development":
            logger.warning("Running without MongoDB in development mode")
        else:
            raise

async def create_indexes():
    """Create database indexes for performance"""
    try:
        # Users collection indexes
        await database.users.create_index("email", unique=True)
        await database.users.create_index("created_at")
        
        # Videos collection indexes
        await database.videos.create_index("user_id")
        await database.videos.create_index("youtube_url")
        await database.videos.create_index("created_at")
        await database.videos.create_index("status")
        
        # Analytics collection indexes
        await database.analytics.create_index("user_id")
        await database.analytics.create_index("video_id")
        await database.analytics.create_index("timestamp")
        
        # Feedback collection indexes
        await database.feedback.create_index("user_id")
        await database.feedback.create_index("created_at")
        
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")

async def connect_to_redis():
    """Create Redis connection"""
    global redis_client
    
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        logger.info("Successfully connected to Redis")
        
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        if settings.ENVIRONMENT == "development":
            logger.warning("Running without Redis in development mode")
        else:
            raise

async def close_mongo_connection():
    """Close database connection"""
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB connection closed")

async def close_redis_connection():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")

def get_database():
    """Get database instance"""
    return database

def get_redis():
    """Get Redis instance"""
    return redis_client
