
"""
Enhanced Cache Management System
Provides Redis-based caching with fallback to in-memory cache
"""

import json
import time
import hashlib
import logging
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..config import get_settings

logger = logging.getLogger(__name__)

class CacheManager:
    """Enhanced cache manager with Redis and in-memory fallback"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, Dict] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "type": "memory"
        }
        self.max_memory_items = 1000
        self.default_ttl = 3600  # 1 hour
        
    async def initialize(self):
        """Initialize cache system with Redis fallback to memory"""
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    self.settings.redis_url,
                    password=self.settings.redis_password,
                    db=self.settings.redis_db,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                await self.redis_client.ping()
                self.cache_stats["type"] = "redis"
                logger.info("✅ Redis cache initialized successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ Redis unavailable, using memory cache: {e}")
                self.redis_client = None
                self.cache_stats["type"] = "memory"
        else:
            logger.warning("⚠️ Redis package not available, using memory cache")
            self.cache_stats["type"] = "memory"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                return await self._get_redis(key)
            else:
                return await self._get_memory(key)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            if self.redis_client:
                return await self._set_redis(key, value, ttl)
            else:
                return await self._set_memory(key, value, ttl)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if self.redis_client:
                result = await self.redis_client.delete(key)
                return result > 0
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if self.redis_client:
                return await self.redis_client.exists(key) > 0
            else:
                if key in self.memory_cache:
                    item = self.memory_cache[key]
                    if item["expires_at"] > time.time():
                        return True
                    else:
                        del self.memory_cache[key]
                return False
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            if self.redis_client:
                await self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def _get_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            value = await self.redis_client.get(key)
            if value is not None:
                self.cache_stats["hits"] += 1
                return json.loads(value)
            else:
                self.cache_stats["misses"] += 1
                return None
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in cache for key: {key}")
            await self.redis_client.delete(key)
            return None
    
    async def _set_redis(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Set value in Redis"""
        try:
            serialized = json.dumps(value, default=str)
            if ttl:
                await self.redis_client.setex(key, ttl, serialized)
            else:
                await self.redis_client.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def _get_memory(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        if key in self.memory_cache:
            item = self.memory_cache[key]
            if item["expires_at"] > time.time():
                self.cache_stats["hits"] += 1
                return item["value"]
            else:
                del self.memory_cache[key]
                self.cache_stats["misses"] += 1
        else:
            self.cache_stats["misses"] += 1
        return None
    
    async def _set_memory(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Set value in memory cache"""
        # Implement LRU eviction if cache is full
        if len(self.memory_cache) >= self.max_memory_items:
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k]["created_at"])
            del self.memory_cache[oldest_key]
        
        expires_at = time.time() + (ttl or self.default_ttl)
        self.memory_cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time()
        }
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache_stats.copy()
        
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update({
                    "redis_memory_used": info.get("used_memory_human", "Unknown"),
                    "redis_connected_clients": info.get("connected_clients", 0),
                    "redis_keyspace_hits": info.get("keyspace_hits", 0),
                    "redis_keyspace_misses": info.get("keyspace_misses", 0)
                })
            except:
                pass
        else:
            stats.update({
                "memory_cache_size": len(self.memory_cache),
                "memory_cache_max_size": self.max_memory_items
            })
        
        return stats
    
    async def close(self):
        """Close cache connections"""
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
        
        self.memory_cache.clear()
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key"""
        key_parts = [prefix]
        key_parts.extend(str(arg) for arg in args)
        
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend(f"{k}:{v}" for k, v in sorted_kwargs)
        
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
