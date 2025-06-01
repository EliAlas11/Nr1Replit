"""
Production-grade database session management
Handles MongoDB connections with proper error handling and connection pooling
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from ..config import get_settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Singleton database manager for MongoDB connections"""
    
    _instance: Optional['DatabaseManager'] = None
    _client: Optional[AsyncIOMotorClient] = None
    _database: Optional[AsyncIOMotorDatabase] = None
    
    def __new__(cls) -> 'DatabaseManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def connect(self) -> None:
        """Establish database connection"""
        if self._client is not None:
            return
            
        settings = get_settings()
        
        try:
            logger.info("Connecting to MongoDB...")
            
            self._client = AsyncIOMotorClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                retryWrites=True,
                retryReads=True
            )
            
            # Test connection
            await self._client.admin.command('ping')
            
            self._database = self._client[settings.database_name]
            
            logger.info(f"✅ Connected to MongoDB database: {settings.database_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected database error: {e}")
            raise

    async def disconnect(self) -> None:
        """Close database connection"""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("📡 Disconnected from MongoDB")

    def get_database(self) -> AsyncIOMotorDatabase:
        """Get database instance"""
        if self._database is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._database

    def get_collection(self, name: str):
        """Get collection from database"""
        return self.get_database()[name]

# Global instance
db_manager = DatabaseManager()

async def get_database() -> AsyncIOMotorDatabase:
    """Dependency to get database instance"""
    await db_manager.connect()
    return db_manager.get_database()

# Global database manager instance

async def get_client() -> AsyncIOMotorClient:
    """Get client instance"""
    if db_manager._client is None:
        await db_manager.connect()
    return db_manager.client

# Import time for health check
import time
async def health_check(self) -> dict:
    """Check database health"""
    try:
        if self._client is None:
            return {"status": "disconnected", "error": "No client connection"}
        
        start_time = time.time()
        await self._client.admin.command('ping')
        response_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time * 1000, 2),
            "database": self._database.name if self._database else None
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
DatabaseManager.health_check = health_check