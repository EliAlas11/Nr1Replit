
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
"""
ViralClip Pro v8.0 - Netflix-Level Enterprise Cache System
High-performance caching with distributed features and advanced optimization
"""

import asyncio
import json
import logging
import pickle
import time
import uuid
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import weakref
import gc

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with comprehensive metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    priority: int = 1  # 1=low, 5=high
    compressed: bool = False
    serialization_method: str = "pickle"


@dataclass
class CacheStatistics:
    """Comprehensive cache statistics"""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    compression_savings_bytes: int = 0
    average_access_time_ms: float = 0.0
    memory_usage_percent: float = 0.0
    last_cleanup: Optional[datetime] = None
    performance_score: float = 100.0


class CacheBackend(ABC):
    """Abstract cache backend interface"""
    
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
    async def clear(self) -> bool:
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        pass


class MemoryCache(CacheBackend):
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.statistics = CacheStatistics()
        self.lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            entry = self.entries.get(key)
            
            if not entry:
                self.statistics.miss_count += 1
                return None
            
            # Check expiration
            if entry.expires_at and datetime.utcnow() > entry.expires_at:
                await self._remove_entry(key)
                self.statistics.miss_count += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            
            # Move to end (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.statistics.hit_count += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        async with self.lock:
            # Calculate entry size
            try:
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
            except Exception:
                logger.warning(f"Failed to serialize cache value for key: {key}")
                return False
            
            # Check memory limits
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Cache entry too large: {size_bytes} bytes")
                return False
            
            # Evict if necessary
            await self._ensure_capacity(size_bytes)
            
            # Create entry
            expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            # Remove old entry if exists
            if key in self.entries:
                await self._remove_entry(key)
            
            # Add new entry
            self.entries[key] = entry
            self.access_order.append(key)
            
            # Update statistics
            self.statistics.total_entries = len(self.entries)
            self.statistics.total_size_bytes += size_bytes
            
            return True
    
    async def delete(self, key: str) -> bool:
        async with self.lock:
            if key in self.entries:
                await self._remove_entry(key)
                return True
            return False
    
    async def clear(self) -> bool:
        async with self.lock:
            self.entries.clear()
            self.access_order.clear()
            self.statistics = CacheStatistics()
            return True
    
    async def exists(self, key: str) -> bool:
        return key in self.entries
    
    async def get_stats(self) -> Dict[str, Any]:
        hit_rate = 0.0
        total_requests = self.statistics.hit_count + self.statistics.miss_count
        if total_requests > 0:
            hit_rate = self.statistics.hit_count / total_requests
        
        return {
            "backend": "memory",
            "total_entries": len(self.entries),
            "total_size_bytes": self.statistics.total_size_bytes,
            "hit_rate": hit_rate,
            "hit_count": self.statistics.hit_count,
            "miss_count": self.statistics.miss_count,
            "eviction_count": self.statistics.eviction_count,
            "memory_usage_percent": (self.statistics.total_size_bytes / self.max_memory_bytes) * 100
        }
    
    async def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry"""
        
        # Size-based eviction
        while (self.statistics.total_size_bytes + new_entry_size > self.max_memory_bytes and 
               self.access_order):
            oldest_key = self.access_order[0]
            await self._remove_entry(oldest_key)
        
        # Count-based eviction
        while len(self.entries) >= self.max_size and self.access_order:
            oldest_key = self.access_order[0]
            await self._remove_entry(oldest_key)
    
    async def _remove_entry(self, key: str):
        """Remove entry and update statistics"""
        if key in self.entries:
            entry = self.entries[key]
            del self.entries[key]
            
            if key in self.access_order:
                self.access_order.remove(key)
            
            self.statistics.total_size_bytes -= entry.size_bytes
            self.statistics.eviction_count += 1


class RedisCache(CacheBackend):
    """Redis-based distributed cache backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.statistics = CacheStatistics()
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            import aioredis
            self.redis_client = await aioredis.from_url(self.redis_url)
            logger.info("✅ Redis cache backend initialized")
        except ImportError:
            logger.error("aioredis not installed, falling back to memory cache")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        if not self.redis_client:
            return None
        
        try:
            data = await self.redis_client.get(key)
            if data:
                self.statistics.hit_count += 1
                return pickle.loads(data)
            else:
                self.statistics.miss_count += 1
                return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.statistics.miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self.redis_client:
            return False
        
        try:
            data = pickle.dumps(value)
            await self.redis_client.set(key, data, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        hit_rate = 0.0
        total_requests = self.statistics.hit_count + self.statistics.miss_count
        if total_requests > 0:
            hit_rate = self.statistics.hit_count / total_requests
        
        stats = {
            "backend": "redis",
            "hit_rate": hit_rate,
            "hit_count": self.statistics.hit_count,
            "miss_count": self.statistics.miss_count
        }
        
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update({
                    "used_memory": info.get("used_memory", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                })
            except Exception as e:
                logger.error(f"Failed to get Redis stats: {e}")
        
        return stats


class EnterpriseCache:
    """Netflix-level enterprise caching system with advanced features"""
    
    def __init__(self, backend: str = "memory", **kwargs):
        self.backend_name = backend
        self.backend: Optional[CacheBackend] = None
        self.fallback_backend: MemoryCache = MemoryCache()
        
        # Performance monitoring
        self.access_times: List[float] = []
        self.performance_threshold_ms = 100
        
        # Cache warming
        self.warming_functions: List[Callable] = []
        
        # Tagging system
        self.tag_index: Dict[str, List[str]] = {}
        
        # Compression
        self.compression_enabled = kwargs.get("compression", True)
        self.compression_threshold = kwargs.get("compression_threshold", 1024)
        
        logger.info(f"🚀 Enterprise cache initialized with {backend} backend")
    
    async def initialize_cache_clusters(self):
        """Initialize cache backend clusters"""
        try:
            if self.backend_name == "redis":
                self.backend = RedisCache()
                await self.backend.initialize()
            else:
                self.backend = MemoryCache()
            
            # Warm up cache
            await self._warm_up_cache()
            
            logger.info("✅ Cache clusters initialized successfully")
            
        except Exception as e:
            logger.error(f"Cache initialization failed, using fallback: {e}")
            self.backend = self.fallback_backend
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with performance monitoring"""
        start_time = time.time()
        
        try:
            # Try primary backend
            if self.backend:
                value = await self.backend.get(key)
                if value is not None:
                    self._record_access_time(time.time() - start_time)
                    return value
            
            # Try fallback
            value = await self.fallback_backend.get(key)
            if value is not None:
                self._record_access_time(time.time() - start_time)
                return value
            
            self._record_access_time(time.time() - start_time)
            return default
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._record_access_time(time.time() - start_time)
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                 tags: Optional[List[str]] = None, priority: int = 1) -> bool:
        """Set value in cache with advanced features"""
        try:
            # Process value (compression, etc.)
            processed_value = await self._process_value_for_storage(value)
            
            # Store in primary backend
            success = False
            if self.backend:
                success = await self.backend.set(key, processed_value, ttl)
            
            # Store in fallback
            if not success:
                success = await self.fallback_backend.set(key, processed_value, ttl)
            
            # Update tag index
            if success and tags:
                await self._update_tag_index(key, tags)
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            success = False
            
            if self.backend:
                success = await self.backend.delete(key)
            
            await self.fallback_backend.delete(key)
            await self._remove_from_tag_index(key)
            
            return success
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete all cache entries with specific tags"""
        deleted_count = 0
        
        for tag in tags:
            if tag in self.tag_index:
                keys_to_delete = self.tag_index[tag].copy()
                for key in keys_to_delete:
                    if await self.delete(key):
                        deleted_count += 1
        
        return deleted_count
    
    async def get_or_set(self, key: str, factory: Callable, ttl: Optional[int] = None,
                        tags: Optional[List[str]] = None) -> Any:
        """Get value from cache or compute and store it"""
        value = await self.get(key)
        
        if value is None:
            # Compute value
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()
            
            # Store in cache
            await self.set(key, value, ttl, tags)
        
        return value
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        primary_stats = {}
        if self.backend:
            primary_stats = await self.backend.get_stats()
        
        fallback_stats = await self.fallback_backend.get_stats()
        
        # Calculate performance metrics
        avg_access_time = 0.0
        if self.access_times:
            avg_access_time = sum(self.access_times) / len(self.access_times) * 1000
        
        return {
            "primary_backend": primary_stats,
            "fallback_backend": fallback_stats,
            "performance": {
                "average_access_time_ms": avg_access_time,
                "performance_score": self._calculate_performance_score(),
                "slow_queries": len([t for t in self.access_times if t * 1000 > self.performance_threshold_ms])
            },
            "tag_index_size": len(self.tag_index),
            "warming_functions": len(self.warming_functions)
        }
    
    async def optimize_cache_performance(self):
        """Optimize cache performance"""
        logger.info("🔧 Starting cache performance optimization...")
        
        # Clean up expired entries
        await self._cleanup_expired_entries()
        
        # Optimize memory usage
        await self._optimize_memory_usage()
        
        # Rebalance cache clusters (if applicable)
        await self._rebalance_clusters()
        
        logger.info("✅ Cache performance optimization completed")
    
    async def warm_up_cache(self, warming_functions: List[Callable]):
        """Warm up cache with predefined data"""
        self.warming_functions.extend(warming_functions)
        
        for func in warming_functions:
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(self)
                else:
                    func(self)
                logger.info(f"✅ Cache warmed up with {func.__name__}")
            except Exception as e:
                logger.error(f"Cache warming failed for {func.__name__}: {e}")
    
    async def shutdown_cache_clusters(self):
        """Gracefully shutdown cache clusters"""
        logger.info("🔄 Shutting down cache clusters...")
        
        try:
            # Close Redis connections if any
            if hasattr(self.backend, 'redis_client') and self.backend.redis_client:
                await self.backend.redis_client.close()
            
            # Clear memory caches
            await self.fallback_backend.clear()
            
            # Clear tag index
            self.tag_index.clear()
            
            logger.info("✅ Cache clusters shutdown completed")
            
        except Exception as e:
            logger.error(f"Cache shutdown error: {e}")
    
    def _record_access_time(self, access_time: float):
        """Record cache access time for performance monitoring"""
        self.access_times.append(access_time)
        
        # Keep only last 1000 access times
        if len(self.access_times) > 1000:
            self.access_times = self.access_times[-1000:]
    
    def _calculate_performance_score(self) -> float:
        """Calculate cache performance score (0-100)"""
        if not self.access_times:
            return 100.0
        
        avg_time_ms = sum(self.access_times) / len(self.access_times) * 1000
        
        # Score based on average access time
        if avg_time_ms <= 1:
            return 100.0
        elif avg_time_ms <= 10:
            return 90.0
        elif avg_time_ms <= 50:
            return 75.0
        elif avg_time_ms <= 100:
            return 50.0
        else:
            return 25.0
    
    async def _process_value_for_storage(self, value: Any) -> Any:
        """Process value before storage (compression, etc.)"""
        if not self.compression_enabled:
            return value
        
        try:
            # Serialize to check size
            serialized = pickle.dumps(value)
            
            # Compress if above threshold
            if len(serialized) > self.compression_threshold:
                import zlib
                compressed = zlib.compress(serialized)
                if len(compressed) < len(serialized):
                    return {"__compressed__": True, "data": compressed}
            
            return value
            
        except Exception as e:
            logger.error(f"Value processing error: {e}")
            return value
    
    async def _warm_up_cache(self):
        """Execute cache warming functions"""
        for func in self.warming_functions:
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(self)
                else:
                    func(self)
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
    
    async def _update_tag_index(self, key: str, tags: List[str]):
        """Update tag index for key"""
        for tag in tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            if key not in self.tag_index[tag]:
                self.tag_index[tag].append(key)
    
    async def _remove_from_tag_index(self, key: str):
        """Remove key from tag index"""
        for tag_keys in self.tag_index.values():
            if key in tag_keys:
                tag_keys.remove(key)
    
    async def _cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        # This would be implemented based on the backend
        pass
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage"""
        gc.collect()
    
    async def _rebalance_clusters(self):
        """Rebalance cache clusters (for distributed setups)"""
        pass


# Cache decorators for easy use
def cached(ttl: Optional[int] = None, key_prefix: str = "", tags: Optional[List[str]] = None):
    """Decorator for caching function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            if args:
                key_parts.extend(str(arg) for arg in args)
            if kwargs:
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl, tags)
            
            return result
        
        return wrapper
    return decorator


# Global cache instance
cache = EnterpriseCache()


# Cache warming functions
async def warm_template_cache(cache_instance: EnterpriseCache):
    """Warm up template cache"""
    # This would load frequently used templates
    pass


async def warm_analytics_cache(cache_instance: EnterpriseCache):
    """Warm up analytics cache"""
    # This would precompute common analytics queries
    pass


# Initialize warming functions
cache.warming_functions = [warm_template_cache, warm_analytics_cache]
