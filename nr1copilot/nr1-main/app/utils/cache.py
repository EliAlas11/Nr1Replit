"""
Redis Cache Manager
High-performance caching for improved application speed
"""

import json
import pickle
from typing import Any, Optional, Union
import redis.asyncio as redis
from datetime import datetime, timedelta

from ..config import get_settings
from ..logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


class CacheManager:
    """High-performance Redis cache manager"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False

    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            logger.info("✅ Redis cache initialized successfully")

        except Exception as e:
            logger.warning(f"⚠️ Redis cache initialization failed: {e}")
            self.redis_client = None
            self.is_connected = False

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("🔌 Redis cache connection closed")

    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        serialize: str = "json"
    ) -> bool:
        """
        Set a value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Serialization method ('json' or 'pickle')
        """
        if not self.is_connected:
            return False

        try:
            # Serialize value
            if serialize == "json":
                serialized_value = json.dumps(value, default=str)
            elif serialize == "pickle":
                serialized_value = pickle.dumps(value)
            else:
                serialized_value = str(value)

            # Set with TTL
            if ttl:
                await self.redis_client.setex(key, ttl, serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)

            logger.debug(f"📦 Cached key: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to set cache key {key}: {e}")
            return False

    async def get(
        self, 
        key: str, 
        default: Any = None,
        serialize: str = "json"
    ) -> Any:
        """
        Get a value from cache

        Args:
            key: Cache key
            default: Default value if key not found
            serialize: Serialization method used when setting
        """
        if not self.is_connected:
            return default

        try:
            value = await self.redis_client.get(key)
            if value is None:
                return default

            # Deserialize value
            if serialize == "json":
                return json.loads(value)
            elif serialize == "pickle":
                return pickle.loads(value)
            else:
                return value.decode() if isinstance(value, bytes) else value

        except Exception as e:
            logger.error(f"❌ Failed to get cache key {key}: {e}")
            return default

    async def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        if not self.is_connected:
            return False

        try:
            deleted_count = await self.redis_client.delete(key)
            logger.debug(f"🗑️ Deleted cache key: {key}")
            return deleted_count > 0

        except Exception as e:
            logger.error(f"❌ Failed to delete cache key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.is_connected:
            return False

        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"❌ Failed to check cache key {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment a counter in cache"""
        if not self.is_connected:
            return 0

        try:
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.incrby(key, amount)
            if ttl:
                pipe.expire(key, ttl)
            results = await pipe.execute()

            return results[0]

        except Exception as e:
            logger.error(f"❌ Failed to increment cache key {key}: {e}")
            return 0

    async def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.is_connected:
            return {"status": "disconnected"}

        try:
            info = await self.redis_client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "0"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                ) * 100
            }

        except Exception as e:
            logger.error(f"❌ Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern"""
        if not self.is_connected:
            return 0

        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted_count = await self.redis_client.delete(*keys)
                logger.info(f"🧹 Cleared {deleted_count} keys matching pattern: {pattern}")
                return deleted_count
            return 0

        except Exception as e:
            logger.error(f"❌ Failed to clear pattern {pattern}: {e}")
            return 0