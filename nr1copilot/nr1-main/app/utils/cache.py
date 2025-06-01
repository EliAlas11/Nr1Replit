
"""
Netflix-Level Cache Management
High-performance caching with Redis integration
"""

import json
import hashlib
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import asyncio

try:
    from redis.asyncio import Redis
except ImportError:
    Redis = None

logger = logging.getLogger(__name__)

class CacheManager:
    """Netflix-level cache management with advanced features"""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis_client = redis_client
        self.local_cache = {}  # Fallback local cache
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback"""
        try:
            if self.redis_client:
                # Try Redis first
                value = await self.redis_client.get(f"viralclip:{key}")
                if value:
                    self.cache_stats["hits"] += 1
                    return json.loads(value)
                    
            # Fallback to local cache
            if key in self.local_cache:
                entry = self.local_cache[key]
                if entry["expires_at"] > datetime.now():
                    self.cache_stats["hits"] += 1
                    return entry["value"]
                else:
                    del self.local_cache[key]
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            if self.redis_client:
                # Set in Redis
                await self.redis_client.setex(
                    f"viralclip:{key}", 
                    ttl, 
                    json.dumps(value, default=str)
                )
            
            # Also set in local cache as backup
            self.local_cache[key] = {
                "value": value,
                "expires_at": datetime.now() + timedelta(seconds=ttl)
            }
            
            self.cache_stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.redis_client:
                await self.redis_client.delete(f"viralclip:{key}")
            
            if key in self.local_cache:
                del self.local_cache[key]
            
            self.cache_stats["deletes"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear_all(self) -> bool:
        """Clear all cache entries"""
        try:
            if self.redis_client:
                # Clear only our keys
                keys = await self.redis_client.keys("viralclip:*")
                if keys:
                    await self.redis_client.delete(*keys)
            
            self.local_cache.clear()
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache),
            "redis_available": self.redis_client is not None
        }
    
    async def cleanup_expired(self):
        """Clean up expired local cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.local_cache.items()
            if entry["expires_at"] <= now
        ]
        
        for key in expired_keys:
            del self.local_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
