
"""
Netflix-Level Database Connection Manager
Enterprise PostgreSQL connection with pooling and monitoring
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, AsyncGenerator
from urllib.parse import urlparse
import os
from dataclasses import dataclass
from enum import Enum

import asyncpg
from asyncpg import Pool, Connection
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ConnectionStatus(str, Enum):
    """Database connection status enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    ERROR = "error"
    POOL_EXHAUSTED = "pool_exhausted"


@dataclass
class ConnectionStats:
    """Database connection statistics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    avg_query_time: float = 0.0
    total_queries: int = 0


class NetflixLevelDatabaseManager:
    """Enterprise database connection manager with Netflix-level performance"""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.status = ConnectionStatus.DISCONNECTED
        self.stats = ConnectionStats()
        self.query_times: list = []
        self.connection_retry_count = 0
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Connection configuration
        self.database_url = self._get_database_url()
        self.pool_config = {
            "min_size": 5,
            "max_size": 20,
            "command_timeout": 30,
            "server_settings": {
                "application_name": "ViralClip_Pro_v10",
                "jit": "off"
            }
        }
    
    def _get_database_url(self) -> str:
        """Get database URL from environment or use Replit PostgreSQL"""
        # Check for Replit Database URL first
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            # Validate URL format
            try:
                parsed = urlparse(db_url)
                if not all([parsed.scheme, parsed.hostname, parsed.path]):
                    raise ValueError("Invalid database URL format")
            except Exception as e:
                logger.error(f"Invalid DATABASE_URL format: {e}")
                raise ConnectionError(f"Database URL validation failed: {e}")
            
            # Use pooler for better performance
            if ".us-east-2" in db_url and "-pooler" not in db_url:
                db_url = db_url.replace(".us-east-2", "-pooler.us-east-2")
                logger.info("✅ Using Replit PostgreSQL connection pooler")
            return db_url
        
        # Fallback configuration for development
        logger.warning("⚠️ Using development database URL - not for production")
        return "postgresql://user:password@localhost:5432/viralclip_pro"
    
    async def initialize(self) -> bool:
        """Initialize database connection pool with retries"""
        if self.pool and not self.pool.closed:
            logger.info("Database pool already initialized")
            return True
        
        self.status = ConnectionStatus.CONNECTING
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Initializing database pool (attempt {attempt + 1}/{self.max_retries})")
                
                # Create connection pool
                self.pool = await asyncpg.create_pool(
                    self.database_url,
                    **self.pool_config
                )
                
                # Test connection
                async with self.pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                
                self.status = ConnectionStatus.CONNECTED
                self.connection_retry_count = 0
                
                logger.info("✅ Database pool initialized successfully")
                await self._setup_database()
                return True
                
            except Exception as e:
                self.connection_retry_count += 1
                logger.error(f"Database connection failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    self.status = ConnectionStatus.ERROR
                    logger.error("❌ Failed to initialize database after all retries")
                    return False
        
        return False
    
    async def _setup_database(self):
        """Setup database schema and initial data"""
        try:
            async with self.get_connection() as conn:
                # Create extension for UUID generation
                await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
                
                # Create schema if not exists
                await conn.execute("CREATE SCHEMA IF NOT EXISTS viralclip")
                
                logger.info("Database schema setup completed")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """Get database connection with automatic cleanup"""
        if not self.pool or self.pool.closed:
            if not await self.initialize():
                raise ConnectionError("Database pool not available")
        
        start_time = time.time()
        conn = None
        
        try:
            conn = await self.pool.acquire()
            self.stats.active_connections += 1
            yield conn
            
        except asyncpg.exceptions.TooManyConnectionsError:
            self.status = ConnectionStatus.POOL_EXHAUSTED
            logger.error("Database pool exhausted")
            raise
            
        except Exception as e:
            self.stats.failed_connections += 1
            logger.error(f"Database connection error: {e}")
            raise
            
        finally:
            if conn:
                try:
                    await self.pool.release(conn)
                    self.stats.active_connections -= 1
                    
                    # Track query performance
                    query_time = time.time() - start_time
                    self.query_times.append(query_time)
                    if len(self.query_times) > 1000:
                        self.query_times = self.query_times[-1000:]
                    
                    self.stats.avg_query_time = sum(self.query_times) / len(self.query_times)
                    self.stats.total_queries += 1
                    
                except Exception as e:
                    logger.error(f"Error releasing connection: {e}")
    
    async def execute_query(self, query: str, *args, fetch: str = "none") -> Any:
        """Execute database query with error handling and performance tracking"""
        async with self.get_connection() as conn:
            try:
                if fetch == "all":
                    return await conn.fetch(query, *args)
                elif fetch == "one":
                    return await conn.fetchrow(query, *args)
                elif fetch == "val":
                    return await conn.fetchval(query, *args)
                else:
                    return await conn.execute(query, *args)
                    
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Args: {args}")
                raise
    
    async def execute_transaction(self, queries: list) -> list:
        """Execute multiple queries in a transaction"""
        async with self.get_connection() as conn:
            async with conn.transaction():
                results = []
                for query_data in queries:
                    if isinstance(query_data, tuple):
                        query, args = query_data[0], query_data[1:]
                    else:
                        query, args = query_data, ()
                    
                    result = await conn.execute(query, *args)
                    results.append(result)
                
                return results
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        if not self.pool:
            return {"status": "not_initialized"}
        
        return {
            "status": self.status.value,
            "pool_size": self.pool.get_size(),
            "pool_idle_size": self.pool.get_idle_size(),
            "pool_min_size": self.pool_config["min_size"],
            "pool_max_size": self.pool_config["max_size"],
            "stats": {
                "total_connections": self.stats.total_connections,
                "active_connections": self.stats.active_connections,
                "failed_connections": self.stats.failed_connections,
                "avg_query_time": round(self.stats.avg_query_time, 4),
                "total_queries": self.stats.total_queries
            },
            "backup_enabled": bool(os.getenv("BACKUP_ENABLED")),
            "read_replica_enabled": bool(os.getenv("READ_REPLICA_URL"))
        }
    
    async def create_backup(self) -> Dict[str, Any]:
        """Create database backup (if backup URL configured)"""
        backup_url = os.getenv("BACKUP_URL")
        if not backup_url:
            return {"status": "not_configured", "message": "Backup URL not configured"}
        
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_name = f"viralclip_backup_{timestamp}"
            
            # In production, this would trigger actual backup process
            # For now, we log the backup request
            logger.info(f"📦 Backup requested: {backup_name}")
            
            return {
                "status": "requested",
                "backup_name": backup_name,
                "timestamp": timestamp
            }
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive database health check"""
        try:
            start_time = time.time()
            
            async with self.get_connection() as conn:
                # Test basic connectivity
                await conn.execute("SELECT 1")
                
                # Test query performance
                await conn.execute("SELECT COUNT(*) FROM information_schema.tables")
                
            response_time = time.time() - start_time
            
            pool_stats = await self.get_pool_stats()
            
            return {
                "healthy": True,
                "status": self.status.value,
                "response_time_ms": round(response_time * 1000, 2),
                "pool_stats": pool_stats,
                "connection_retries": self.connection_retry_count
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "status": ConnectionStatus.ERROR.value
            }
    
    async def shutdown(self):
        """Gracefully shutdown database connections"""
        if self.pool and not self.pool.closed:
            logger.info("Shutting down database pool...")
            await self.pool.close()
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("✅ Database pool shutdown complete")


# Global database manager instance
db_manager = NetflixLevelDatabaseManager()


# Convenience functions
async def get_db_connection():
    """Get database connection"""
    return db_manager.get_connection()


async def execute_query(query: str, *args, fetch: str = "none"):
    """Execute database query"""
    return await db_manager.execute_query(query, *args, fetch=fetch)


async def execute_transaction(queries: list):
    """Execute transaction"""
    return await db_manager.execute_transaction(queries)
