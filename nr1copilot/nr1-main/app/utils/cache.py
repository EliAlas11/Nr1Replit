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
"""
Netflix-Level Caching System
"""

import asyncio
import json
import time
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Netflix-level caching with Redis and local fallback"""
    
    def __init__(self, redis_client=None, default_ttl: int = 3600):
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.local_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
        
        # Local cache with TTL tracking
        self.local_cache_ttl = {}
        self.max_local_entries = 1000
        
        # Start cleanup task
        self._cleanup_task = None
        self.start_cleanup_task()
    
    def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
    
    async def _cleanup_expired_entries(self):
        """Clean up expired entries from local cache"""
        while True:
            try:
                current_time = time.time()
                expired_keys = [
                    key for key, ttl in self.local_cache_ttl.items()
                    if ttl < current_time
                ]
                
                for key in expired_keys:
                    del self.local_cache[key]
                    del self.local_cache_ttl[key]
                    self.cache_stats["evictions"] += 1
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(60)  # Clean every minute
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(30)
    
    def _generate_cache_key(self, key: str, namespace: str = "default") -> str:
        """Generate a cache key with namespace"""
        return f"viralclip:{namespace}:{key}"
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for caching"""
        try:
            return json.dumps(value, default=str)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return str(value)
    
    def _deserialize_value(self, value: str) -> Any:
        """Deserialize cached value"""
        try:
            return json.loads(value)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return value
    
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    value = await self.redis_client.get(cache_key)
                    if value is not None:
                        self.cache_stats["hits"] += 1
                        return self._deserialize_value(value)
                except Exception as e:
                    logger.error(f"Redis get error: {e}")
            
            # Fallback to local cache
            if cache_key in self.local_cache:
                # Check TTL
                if cache_key in self.local_cache_ttl:
                    if self.local_cache_ttl[cache_key] > time.time():
                        self.cache_stats["hits"] += 1
                        return self.local_cache[cache_key]
                    else:
                        # Expired
                        del self.local_cache[cache_key]
                        del self.local_cache_ttl[cache_key]
                        self.cache_stats["evictions"] += 1
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        namespace: str = "default"
    ) -> bool:
        """Set value in cache"""
        cache_key = self._generate_cache_key(key, namespace)
        ttl = ttl or self.default_ttl
        
        try:
            serialized_value = self._serialize_value(value)
            
            # Try Redis first
            if self.redis_client:
                try:
                    await self.redis_client.setex(cache_key, ttl, serialized_value)
                    self.cache_stats["sets"] += 1
                    return True
                except Exception as e:
                    logger.error(f"Redis set error: {e}")
            
            # Fallback to local cache
            self._set_local_cache(cache_key, value, ttl)
            self.cache_stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def _set_local_cache(self, key: str, value: Any, ttl: int):
        """Set value in local cache with TTL"""
        # Evict oldest entries if cache is full
        if len(self.local_cache) >= self.max_local_entries:
            # Remove 10% of oldest entries
            entries_to_remove = max(1, self.max_local_entries // 10)
            oldest_keys = sorted(
                self.local_cache_ttl.keys(),
                key=lambda k: self.local_cache_ttl[k]
            )[:entries_to_remove]
            
            for old_key in oldest_keys:
                del self.local_cache[old_key]
                del self.local_cache_ttl[old_key]
                self.cache_stats["evictions"] += 1
        
        self.local_cache[key] = value
        self.local_cache_ttl[key] = time.time() + ttl
    
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from cache"""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            deleted = False
            
            # Delete from Redis
            if self.redis_client:
                try:
                    result = await self.redis_client.delete(cache_key)
                    deleted = result > 0
                except Exception as e:
                    logger.error(f"Redis delete error: {e}")
            
            # Delete from local cache
            if cache_key in self.local_cache:
                del self.local_cache[cache_key]
                deleted = True
            
            if cache_key in self.local_cache_ttl:
                del self.local_cache_ttl[cache_key]
            
            if deleted:
                self.cache_stats["deletes"] += 1
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists in cache"""
        value = await self.get(key, namespace)
        return value is not None
    
    async def expire(self, key: str, ttl: int, namespace: str = "default") -> bool:
        """Set expiration for existing key"""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            # Update Redis TTL
            if self.redis_client:
                try:
                    result = await self.redis_client.expire(cache_key, ttl)
                    if result:
                        return True
                except Exception as e:
                    logger.error(f"Redis expire error: {e}")
            
            # Update local cache TTL
            if cache_key in self.local_cache:
                self.local_cache_ttl[cache_key] = time.time() + ttl
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache expire error: {e}")
            return False
    
    async def clear_namespace(self, namespace: str = "default") -> int:
        """Clear all keys in a namespace"""
        pattern = self._generate_cache_key("*", namespace)
        cleared_count = 0
        
        try:
            # Clear from Redis
            if self.redis_client:
                try:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        cleared_count += await self.redis_client.delete(*keys)
                except Exception as e:
                    logger.error(f"Redis clear namespace error: {e}")
            
            # Clear from local cache
            prefix = self._generate_cache_key("", namespace)
            local_keys_to_remove = [
                key for key in self.local_cache.keys()
                if key.startswith(prefix)
            ]
            
            for key in local_keys_to_remove:
                del self.local_cache[key]
                if key in self.local_cache_ttl:
                    del self.local_cache_ttl[key]
                cleared_count += 1
            
            return cleared_count
            
        except Exception as e:
            logger.error(f"Cache clear namespace error: {e}")
            return 0
    
    async def get_keys(self, pattern: str = "*", namespace: str = "default") -> List[str]:
        """Get all keys matching pattern"""
        cache_pattern = self._generate_cache_key(pattern, namespace)
        keys = []
        
        try:
            # Get from Redis
            if self.redis_client:
                try:
                    redis_keys = await self.redis_client.keys(cache_pattern)
                    keys.extend([key.decode() if isinstance(key, bytes) else key for key in redis_keys])
                except Exception as e:
                    logger.error(f"Redis keys error: {e}")
            
            # Get from local cache
            prefix = self._generate_cache_key("", namespace)
            local_keys = [
                key for key in self.local_cache.keys()
                if key.startswith(prefix)
            ]
            
            # Remove duplicates and return
            all_keys = set(keys + local_keys)
            return list(all_keys)
            
        except Exception as e:
            logger.error(f"Cache get keys error: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache_stats.copy()
        
        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        stats["hit_rate"] = (stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        # Local cache info
        stats["local_cache_size"] = len(self.local_cache)
        stats["local_cache_max_size"] = self.max_local_entries
        
        # Redis info
        if self.redis_client:
            try:
                stats["redis_connected"] = await self.redis_client.ping()
                info = await self.redis_client.info("memory")
                stats["redis_memory_usage"] = info.get("used_memory_human", "N/A")
            except Exception:
                stats["redis_connected"] = False
        else:
            stats["redis_connected"] = False
        
        return stats
    
    async def warm_cache(self, data: Dict[str, Any], namespace: str = "warmup"):
        """Warm the cache with initial data"""
        try:
            for key, value in data.items():
                await self.set(key, value, namespace=namespace)
            
            logger.info(f"Cache warmed with {len(data)} entries in namespace '{namespace}'")
            
        except Exception as e:
            logger.error(f"Cache warm error: {e}")
    
    def cache_key_for_video_analysis(self, url: str, params: Dict[str, Any]) -> str:
        """Generate cache key for video analysis"""
        # Create a hash of URL and parameters for consistent caching
        content = f"{url}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def cache_key_for_processing(self, session_id: str, clips: List[Dict[str, Any]]) -> str:
        """Generate cache key for processing results"""
        content = f"{session_id}:{json.dumps(clips, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def cache_video_analysis(
        self, 
        url: str, 
        params: Dict[str, Any], 
        analysis_result: Dict[str, Any],
        ttl: int = 3600
    ):
        """Cache video analysis results"""
        cache_key = self.cache_key_for_video_analysis(url, params)
        await self.set(cache_key, analysis_result, ttl=ttl, namespace="video_analysis")
    
    async def get_cached_video_analysis(
        self, 
        url: str, 
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached video analysis results"""
        cache_key = self.cache_key_for_video_analysis(url, params)
        return await self.get(cache_key, namespace="video_analysis")
"""
Caching System
Netflix-level caching with fallbacks
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional, Dict
import hashlib

logger = logging.getLogger(__name__)

class CacheManager:
    """Netflix-level caching system"""
    
    def __init__(self, redis_client: Optional[Any] = None):
        self.redis_client = redis_client
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.max_local_entries = 1000
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = await self._get_from_redis(key)
                if value is not None:
                    return value
            
            # Fallback to local cache
            return await self._get_from_local(key)
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600
    ) -> bool:
        """Set value in cache"""
        try:
            success = False
            
            if self.redis_client:
                success = await self._set_in_redis(key, value, ttl)
            
            # Always set in local cache as fallback
            await self._set_in_local(key, value, ttl)
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            success = False
            
            if self.redis_client:
                success = await self._delete_from_redis(key)
            
            # Also delete from local cache
            if key in self.local_cache:
                del self.local_cache[key]
            
            return success
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            success = False
            
            if self.redis_client:
                await self.redis_client.flushdb()
                success = True
            
            self.local_cache.clear()
            return success
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get from Redis"""
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def _set_in_redis(
        self,
        key: str,
        value: Any,
        ttl: int
    ) -> bool:
        """Set in Redis"""
        try:
            serialized = json.dumps(value)
            await self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def _delete_from_redis(self, key: str) -> bool:
        """Delete from Redis"""
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def _get_from_local(self, key: str) -> Optional[Any]:
        """Get from local cache"""
        if key in self.local_cache:
            entry = self.local_cache[key]
            
            # Check if expired
            if time.time() > entry["expires"]:
                del self.local_cache[key]
                return None
            
            return entry["value"]
        
        return None
    
    async def _set_in_local(
        self,
        key: str,
        value: Any,
        ttl: int
    ):
        """Set in local cache"""
        # Clean up if too many entries
        if len(self.local_cache) >= self.max_local_entries:
            await self._cleanup_local_cache()
        
        self.local_cache[key] = {
            "value": value,
            "expires": time.time() + ttl,
            "created": time.time()
        }
    
    async def _cleanup_local_cache(self):
        """Clean up expired local cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.local_cache.items()
            if current_time > entry["expires"]
        ]
        
        for key in expired_keys:
            del self.local_cache[key]
        
        # If still too many, remove oldest entries
        if len(self.local_cache) >= self.max_local_entries:
            sorted_entries = sorted(
                self.local_cache.items(),
                key=lambda x: x[1]["created"]
            )
            
            # Remove oldest 25%
            remove_count = len(sorted_entries) // 4
            for key, _ in sorted_entries[:remove_count]:
                del self.local_cache[key]
    
    def generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_data = ":".join(str(arg) for arg in args)
        return hashlib.md5(key_data.encode()).hexdigest()
