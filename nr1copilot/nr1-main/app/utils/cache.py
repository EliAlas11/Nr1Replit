
"""
Netflix-Level Caching System
"""

import json
import pickle
from typing import Any, Optional, Union
from redis.asyncio import Redis

class CacheManager:
    """Netflix-level caching with Redis backend"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.redis.get(f"cache:{key}")
            if value is None:
                return None
            
            # Try to deserialize as JSON first, then pickle
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return pickle.loads(value)
                
        except Exception:
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            
            # Try to serialize as JSON first, then pickle
            try:
                serialized_value = json.dumps(value)
            except (TypeError, ValueError):
                serialized_value = pickle.dumps(value)
            
            await self.redis.setex(f"cache:{key}", ttl, serialized_value)
            return True
            
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            result = await self.redis.delete(f"cache:{key}")
            return result > 0
        except Exception:
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            result = await self.redis.exists(f"cache:{key}")
            return result > 0
        except Exception:
            return False
    
    async def flush_all(self) -> bool:
        """Flush all cached data"""
        try:
            keys = await self.redis.keys("cache:*")
            if keys:
                await self.redis.delete(*keys)
            return True
        except Exception:
            return False
    
    async def get_stats(self) -> dict:
        """Get cache statistics"""
        try:
            info = await self.redis.info()
            keys = await self.redis.keys("cache:*")
            
            return {
                "total_keys": len(keys),
                "memory_usage": info.get("used_memory_human", "unknown"),
                "hit_rate": info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)),
                "connected_clients": info.get("connected_clients", 0)
            }
        except Exception:
            return {"error": "Unable to get cache stats"}
