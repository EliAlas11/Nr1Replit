"""
Cache Management Utilities
Netflix-level caching for optimal performance
"""

import json
import logging
import hashlib
from typing import Any, Optional, Dict
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    """Netflix-level cache management"""

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.memory_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    value = await self.redis_client.get(key)
                    if value:
                        self.cache_stats["hits"] += 1
                        return json.loads(value)
                except Exception as e:
                    logger.warning(f"Redis get error: {e}")

            # Fallback to memory cache
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry["expires_at"] > datetime.now():
                    self.cache_stats["hits"] += 1
                    return entry["value"]
                else:
                    # Expired entry
                    del self.memory_cache[key]

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache"""
        try:
            serialized_value = json.dumps(value, default=str)

            # Try Redis first
            if self.redis_client:
                try:
                    await self.redis_client.setex(key, ttl, serialized_value)
                    self.cache_stats["sets"] += 1
                    return True
                except Exception as e:
                    logger.warning(f"Redis set error: {e}")

            # Fallback to memory cache
            expires_at = datetime.now() + timedelta(seconds=ttl)
            self.memory_cache[key] = {
                "value": value,
                "expires_at": expires_at
            }
            self.cache_stats["sets"] += 1
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            deleted = False

            # Try Redis first
            if self.redis_client:
                try:
                    result = await self.redis_client.delete(key)
                    if result:
                        deleted = True
                except Exception as e:
                    logger.warning(f"Redis delete error: {e}")

            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                deleted = True

            if deleted:
                self.cache_stats["deletes"] += 1

            return deleted

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests * 100 
            if total_requests > 0 else 0
        )

        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache)
        }

    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            # Clear Redis
            if self.redis_client:
                try:
                    await self.redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Redis clear error: {e}")

            # Clear memory cache
            self.memory_cache.clear()

            return True

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()