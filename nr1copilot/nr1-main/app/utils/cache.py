
"""
Netflix-Level Caching System
Multi-tier caching with Redis, memory cache, and intelligent invalidation
"""

import asyncio
import json
import pickle
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from functools import wraps
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import threading
from collections import OrderedDict, defaultdict

from ..config import settings
from ..logging_config import LoggerMixin, PerformanceLogger

# Type variables
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class CacheLevel(Enum):
    """Cache level enumeration"""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


class CacheStrategy(Enum):
    """Cache strategy enumeration"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float] = None
    tags: Optional[List[str]] = None
    size_bytes: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds"""
        return time.time() - self.created_at


class CacheStats:
    """Cache statistics tracking"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.total_size_bytes = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_hit(self):
        with self.lock:
            self.hits += 1
    
    def record_miss(self):
        with self.lock:
            self.misses += 1
    
    def record_set(self, size_bytes: int = 0):
        with self.lock:
            self.sets += 1
            self.total_size_bytes += size_bytes
    
    def record_delete(self, size_bytes: int = 0):
        with self.lock:
            self.deletes += 1
            self.total_size_bytes = max(0, self.total_size_bytes - size_bytes)
    
    def record_eviction(self, size_bytes: int = 0):
        with self.lock:
            self.evictions += 1
            self.total_size_bytes = max(0, self.total_size_bytes - size_bytes)
    
    def record_error(self):
        with self.lock:
            self.errors += 1
    
    @property
    def hit_rate(self) -> float:
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "deletes": self.deletes,
                "evictions": self.evictions,
                "errors": self.errors,
                "hit_rate": self.hit_rate,
                "total_size_bytes": self.total_size_bytes,
                "uptime_seconds": self.uptime_seconds
            }


class CacheBackend(ABC):
    """Abstract cache backend"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        pass


class MemoryCacheBackend(CacheBackend):
    """High-performance in-memory cache backend"""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 512, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.lock = asyncio.Lock()
        
        # Strategy-specific data
        self.access_counts = defaultdict(int)
        self.access_times = {}
    
    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            if key not in self.cache:
                self.stats.record_miss()
                return None
            
            entry = self.cache[key]
            
            if entry.is_expired:
                del self.cache[key]
                self.stats.record_miss()
                return None
            
            # Update access patterns
            entry.accessed_at = time.time()
            entry.access_count += 1
            self.access_counts[key] += 1
            self.access_times[key] = time.time()
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self.cache.move_to_end(key)
            
            self.stats.record_hit()
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            # Calculate size
            size_bytes = len(pickle.dumps(value))
            
            async with self.lock:
                # Check if we need to evict
                while (len(self.cache) >= self.max_size or 
                       self.stats.total_size_bytes + size_bytes > self.max_memory_bytes):
                    if not await self._evict_one():
                        break
                
                # Create entry
                entry = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    access_count=1,
                    ttl=ttl,
                    size_bytes=size_bytes
                )
                
                self.cache[key] = entry
                self.access_counts[key] = 1
                self.access_times[key] = time.time()
                
                self.stats.record_set(size_bytes)
                return True
                
        except Exception as e:
            logging.error(f"Memory cache set error: {e}")
            self.stats.record_error()
            return False
    
    async def delete(self, key: str) -> bool:
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                size_bytes = entry.size_bytes or 0
                del self.cache[key]
                
                # Clean up tracking
                self.access_counts.pop(key, None)
                self.access_times.pop(key, None)
                
                self.stats.record_delete(size_bytes)
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        async with self.lock:
            if key not in self.cache:
                return False
            
            entry = self.cache[key]
            if entry.is_expired:
                del self.cache[key]
                return False
            
            return True
    
    async def clear(self) -> bool:
        async with self.lock:
            self.cache.clear()
            self.access_counts.clear()
            self.access_times.clear()
            self.stats = CacheStats()
            return True
    
    async def _evict_one(self) -> bool:
        """Evict one entry based on strategy"""
        if not self.cache:
            return False
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used (first item)
            key, entry = self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            entry = self.cache.pop(key)
        else:  # TTL or default
            # Remove oldest entry
            key, entry = self.cache.popitem(last=False)
        
        # Clean up tracking
        self.access_counts.pop(key, None)
        self.access_times.pop(key, None)
        
        size_bytes = entry.size_bytes or 0
        self.stats.record_eviction(size_bytes)
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self.lock:
            return {
                **self.stats.to_dict(),
                "entries_count": len(self.cache),
                "max_size": self.max_size,
                "max_memory_bytes": self.max_memory_bytes,
                "strategy": self.strategy.value
            }


class RedisCacheBackend(CacheBackend):
    """Redis cache backend with connection pooling"""
    
    def __init__(self):
        self.redis = None
        self.stats = CacheStats()
        self.connection_pool = None
        self.initialized = False
    
    async def _initialize(self):
        """Initialize Redis connection"""
        if self.initialized:
            return
        
        try:
            import redis.asyncio as redis
            
            self.connection_pool = redis.ConnectionPool.from_url(
                settings.redis.url,
                max_connections=settings.redis.max_connections,
                socket_timeout=settings.redis.socket_timeout,
                socket_connect_timeout=settings.redis.socket_connect_timeout,
                retry_on_timeout=settings.redis.retry_on_timeout,
                decode_responses=settings.redis.decode_responses
            )
            
            self.redis = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis.ping()
            self.initialized = True
            
        except Exception as e:
            logging.error(f"Redis initialization failed: {e}")
            self.initialized = False
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            await self._initialize()
            if not self.redis:
                return None
            
            data = await self.redis.get(key)
            if data is None:
                self.stats.record_miss()
                return None
            
            value = pickle.loads(data)
            self.stats.record_hit()
            return value
            
        except Exception as e:
            logging.error(f"Redis get error: {e}")
            self.stats.record_error()
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            await self._initialize()
            if not self.redis:
                return False
            
            data = pickle.dumps(value)
            
            if ttl:
                result = await self.redis.setex(key, ttl, data)
            else:
                result = await self.redis.set(key, data)
            
            if result:
                self.stats.record_set(len(data))
            return bool(result)
            
        except Exception as e:
            logging.error(f"Redis set error: {e}")
            self.stats.record_error()
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            await self._initialize()
            if not self.redis:
                return False
            
            result = await self.redis.delete(key)
            if result:
                self.stats.record_delete()
            return bool(result)
            
        except Exception as e:
            logging.error(f"Redis delete error: {e}")
            self.stats.record_error()
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            await self._initialize()
            if not self.redis:
                return False
            
            return bool(await self.redis.exists(key))
            
        except Exception as e:
            logging.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        try:
            await self._initialize()
            if not self.redis:
                return False
            
            await self.redis.flushdb()
            self.stats = CacheStats()
            return True
            
        except Exception as e:
            logging.error(f"Redis clear error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.to_dict()
        
        try:
            if self.redis:
                info = await self.redis.info()
                stats.update({
                    "redis_connected_clients": info.get("connected_clients", 0),
                    "redis_used_memory": info.get("used_memory", 0),
                    "redis_hits": info.get("keyspace_hits", 0),
                    "redis_misses": info.get("keyspace_misses", 0),
                })
        except Exception:
            pass
        
        return stats


class HybridCacheBackend(CacheBackend):
    """Hybrid cache using both memory and Redis"""
    
    def __init__(self):
        self.memory_cache = MemoryCacheBackend(max_size=1000, max_memory_mb=128)
        self.redis_cache = RedisCacheBackend()
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            self.stats.record_hit()
            return value
        
        # Try Redis cache
        value = await self.redis_cache.get(key)
        if value is not None:
            # Promote to memory cache
            await self.memory_cache.set(key, value)
            self.stats.record_hit()
            return value
        
        self.stats.record_miss()
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        # Set in both caches
        memory_result = await self.memory_cache.set(key, value, ttl)
        redis_result = await self.redis_cache.set(key, value, ttl)
        
        if memory_result or redis_result:
            self.stats.record_set()
            return True
        return False
    
    async def delete(self, key: str) -> bool:
        memory_result = await self.memory_cache.delete(key)
        redis_result = await self.redis_cache.delete(key)
        
        if memory_result or redis_result:
            self.stats.record_delete()
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        return (await self.memory_cache.exists(key) or 
                await self.redis_cache.exists(key))
    
    async def clear(self) -> bool:
        memory_result = await self.memory_cache.clear()
        redis_result = await self.redis_cache.clear()
        self.stats = CacheStats()
        return memory_result and redis_result
    
    async def get_stats(self) -> Dict[str, Any]:
        memory_stats = await self.memory_cache.get_stats()
        redis_stats = await self.redis_cache.get_stats()
        
        return {
            "hybrid": self.stats.to_dict(),
            "memory": memory_stats,
            "redis": redis_stats
        }


class NetflixLevelCacheManager(LoggerMixin):
    """Netflix-level cache manager with intelligent caching strategies"""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        self.backend = backend or self._create_default_backend()
        self.cache_tags = defaultdict(set)
        self.tag_lock = asyncio.Lock()
        
    def _create_default_backend(self) -> CacheBackend:
        """Create default cache backend based on configuration"""
        if settings.environment == "production":
            return HybridCacheBackend()
        else:
            return MemoryCacheBackend()
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with PerformanceLogger(f"cache_get:{key}", self.logger):
            value = await self.backend.get(key)
            return value if value is not None else default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> bool:
        """Set value in cache with optional tags"""
        with PerformanceLogger(f"cache_set:{key}", self.logger):
            result = await self.backend.set(key, value, ttl)
            
            if result and tags:
                async with self.tag_lock:
                    for tag in tags:
                        self.cache_tags[tag].add(key)
            
            return result
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with PerformanceLogger(f"cache_delete:{key}", self.logger):
            result = await self.backend.delete(key)
            
            # Remove from tags
            async with self.tag_lock:
                for tag_keys in self.cache_tags.values():
                    tag_keys.discard(key)
            
            return result
    
    async def delete_by_tag(self, tag: str) -> int:
        """Delete all cache entries with specific tag"""
        async with self.tag_lock:
            keys = self.cache_tags.get(tag, set()).copy()
            self.cache_tags[tag].clear()
        
        deleted_count = 0
        for key in keys:
            if await self.delete(key):
                deleted_count += 1
        
        return deleted_count
    
    async def get_or_set(self, key: str, factory: Callable, ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> Any:
        """Get value from cache or set it using factory function"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Generate value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        await self.set(key, value, ttl, tags)
        return value
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        async with self.tag_lock:
            self.cache_tags.clear()
        
        return await self.backend.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        backend_stats = await self.backend.get_stats()
        
        async with self.tag_lock:
            tag_stats = {
                "total_tags": len(self.cache_tags),
                "tags": {tag: len(keys) for tag, keys in self.cache_tags.items()}
            }
        
        return {
            "backend": backend_stats,
            "tags": tag_stats
        }


# Decorators for caching
def cached(ttl: Optional[int] = None, key_prefix: str = "", tags: Optional[List[str]] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(cache_key, result, ttl, tags)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def cache_invalidate_tag(tag: str):
    """Decorator to invalidate cache by tag after function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            await cache_manager.delete_by_tag(tag)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(cache_manager.delete_by_tag(tag))
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global cache manager instance
cache_manager = NetflixLevelCacheManager()


# Cache warming utilities
class CacheWarmer:
    """Utility for warming up cache with frequently accessed data"""
    
    def __init__(self, cache_manager: NetflixLevelCacheManager):
        self.cache_manager = cache_manager
        self.logger = logging.getLogger("app.cache.warmer")
    
    async def warm_upload_metadata(self):
        """Warm cache with upload-related metadata"""
        try:
            # Cache supported formats
            await self.cache_manager.set(
                "supported_formats",
                settings.upload.supported_formats,
                ttl=3600,
                tags=["config", "upload"]
            )
            
            # Cache upload limits
            await self.cache_manager.set(
                "upload_limits",
                {
                    "max_file_size": settings.upload.max_file_size,
                    "chunk_size": settings.upload.chunk_size,
                    "max_concurrent_uploads": settings.upload.max_concurrent_uploads
                },
                ttl=3600,
                tags=["config", "upload"]
            )
            
            self.logger.info("Upload metadata cache warmed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to warm upload metadata cache: {e}")
    
    async def warm_ai_models(self):
        """Warm cache with AI model metadata"""
        try:
            # This would typically load model metadata from a service
            model_metadata = {
                "model_name": settings.ai.model_name,
                "max_tokens": settings.ai.max_tokens,
                "temperature": settings.ai.temperature
            }
            
            await self.cache_manager.set(
                "ai_model_metadata",
                model_metadata,
                ttl=1800,
                tags=["ai", "models"]
            )
            
            self.logger.info("AI model cache warmed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to warm AI model cache: {e}")
    
    async def warm_all(self):
        """Warm all cache categories"""
        await asyncio.gather(
            self.warm_upload_metadata(),
            self.warm_ai_models(),
            return_exceptions=True
        )


# Initialize cache warmer
cache_warmer = CacheWarmer(cache_manager)
