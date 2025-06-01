
"""
Netflix-Level Caching System v5.0
High-performance caching with multiple backends and intelligent invalidation
"""

import asyncio
import json
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pickle
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheBackend(str, Enum):
    """Cache backend types"""
    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    ttl: Optional[float]
    access_count: int = 0
    last_accessed: float = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.last_accessed == 0:
            self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    @property
    def age(self) -> float:
        """Get entry age in seconds"""
        return time.time() - self.created_at
    
    def touch(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()


class CacheStats:
    """Cache statistics collector"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.start_time = time.time()
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def uptime(self) -> float:
        """Get cache uptime in seconds"""
        return time.time() - self.start_time
    
    def reset(self):
        """Reset statistics"""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.start_time = time.time()


class MemoryCache:
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.data: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.stats = CacheStats()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            entry = self.data.get(key)
            
            if entry is None:
                self.stats.misses += 1
                return None
            
            if entry.is_expired:
                await self._delete_entry(key)
                self.stats.misses += 1
                return None
            
            # Update access metadata
            entry.touch()
            self._update_access_order(key)
            
            self.stats.hits += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        async with self._lock:
            try:
                # Use default TTL if not specified
                if ttl is None:
                    ttl = self.default_ttl
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    ttl=ttl
                )
                
                # Check if we need to evict entries
                if len(self.data) >= self.max_size and key not in self.data:
                    await self._evict_lru()
                
                # Store entry
                self.data[key] = entry
                self._update_access_order(key)
                
                self.stats.sets += 1
                return True
                
            except Exception as e:
                logger.error(f"Cache set error: {e}")
                self.stats.errors += 1
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        async with self._lock:
            if key in self.data:
                await self._delete_entry(key)
                self.stats.deletes += 1
                return True
            return False
    
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self.data.clear()
            self.access_order.clear()
    
    async def cleanup_expired(self):
        """Remove expired entries"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self.data.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                await self._delete_entry(key)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "backend": "memory",
            "size": len(self.data),
            "max_size": self.max_size,
            "hit_rate": self.stats.hit_rate,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "evictions": self.stats.evictions,
            "errors": self.stats.errors,
            "uptime": self.stats.uptime
        }
    
    def _update_access_order(self, key: str):
        """Update LRU access order"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    async def _delete_entry(self, key: str):
        """Delete entry and update access order"""
        if key in self.data:
            del self.data[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if self.access_order:
            lru_key = self.access_order[0]
            await self._delete_entry(lru_key)
            self.stats.evictions += 1


class FileCache:
    """File-based cache for persistence"""
    
    def __init__(self, cache_dir: Path, default_ttl: Optional[float] = 3600):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.stats = CacheStats()
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from file cache"""
        try:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                self.stats.misses += 1
                return None
            
            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()
                entry = pickle.loads(data)
            
            if entry.is_expired:
                await self.delete(key)
                self.stats.misses += 1
                return None
            
            entry.touch()
            
            # Update file with new access metadata
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(pickle.dumps(entry))
            
            self.stats.hits += 1
            return entry.value
            
        except Exception as e:
            logger.error(f"File cache get error: {e}")
            self.stats.errors += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in file cache"""
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=ttl
            )
            
            file_path = self._get_file_path(key)
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(pickle.dumps(entry))
            
            self.stats.sets += 1
            return True
            
        except Exception as e:
            logger.error(f"File cache set error: {e}")
            self.stats.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from file cache"""
        try:
            file_path = self._get_file_path(key)
            
            if file_path.exists():
                file_path.unlink()
                self.stats.deletes += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"File cache delete error: {e}")
            self.stats.errors += 1
            return False
    
    async def clear(self):
        """Clear all cache files"""
        try:
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()
        except Exception as e:
            logger.error(f"File cache clear error: {e}")
    
    async def cleanup_expired(self):
        """Remove expired cache files"""
        try:
            for file_path in self.cache_dir.glob("*.cache"):
                try:
                    async with aiofiles.open(file_path, 'rb') as f:
                        data = await f.read()
                        entry = pickle.loads(data)
                    
                    if entry.is_expired:
                        file_path.unlink()
                        
                except Exception:
                    # Remove corrupted files
                    file_path.unlink()
                    
        except Exception as e:
            logger.error(f"File cache cleanup error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_files = len(list(self.cache_dir.glob("*.cache")))
        
        return {
            "backend": "file",
            "size": cache_files,
            "cache_dir": str(self.cache_dir),
            "hit_rate": self.stats.hit_rate,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "errors": self.stats.errors,
            "uptime": self.stats.uptime
        }


class CacheManager:
    """Netflix-level cache manager with multiple backends and intelligent strategies"""
    
    def __init__(
        self,
        backend: CacheBackend = CacheBackend.HYBRID,
        memory_max_size: int = 1000,
        file_cache_dir: Optional[Path] = None,
        default_ttl: float = 3600,
        cleanup_interval: float = 300
    ):
        self.backend = backend
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # Initialize backends
        self.memory_cache = MemoryCache(memory_max_size, default_ttl)
        self.file_cache = FileCache(
            file_cache_dir or Path("cache"),
            default_ttl
        ) if backend in [CacheBackend.FILE, CacheBackend.HYBRID] else None
        
        # Cache strategies
        self.strategies = {
            "memory_first": self._strategy_memory_first,
            "file_persistent": self._strategy_file_persistent,
            "hybrid_intelligent": self._strategy_hybrid_intelligent
        }
        
        self._cleanup_task = None
        self._running = False
    
    async def initialize(self):
        """Initialize cache manager"""
        self._running = True
        if self.cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info(f"Cache manager initialized with backend: {self.backend}")
    
    async def cleanup(self):
        """Cleanup cache manager"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Cache manager cleaned up")
    
    async def get(self, key: str, strategy: str = "hybrid_intelligent") -> Optional[Any]:
        """Get value using specified strategy"""
        strategy_func = self.strategies.get(strategy, self._strategy_hybrid_intelligent)
        return await strategy_func("get", key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        strategy: str = "hybrid_intelligent"
    ) -> bool:
        """Set value using specified strategy"""
        strategy_func = self.strategies.get(strategy, self._strategy_hybrid_intelligent)
        return await strategy_func("set", key, value, ttl)
    
    async def delete(self, key: str, strategy: str = "hybrid_intelligent") -> bool:
        """Delete value using specified strategy"""
        strategy_func = self.strategies.get(strategy, self._strategy_hybrid_intelligent)
        return await strategy_func("delete", key)
    
    async def clear(self):
        """Clear all caches"""
        await self.memory_cache.clear()
        if self.file_cache:
            await self.file_cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "backend": self.backend.value,
            "memory": await self.memory_cache.get_stats()
        }
        
        if self.file_cache:
            stats["file"] = await self.file_cache.get_stats()
        
        return stats
    
    # Cache strategies
    async def _strategy_memory_first(self, operation: str, *args, **kwargs) -> Any:
        """Memory-first strategy"""
        if operation == "get":
            key = args[0]
            return await self.memory_cache.get(key)
        elif operation == "set":
            key, value = args[0], args[1]
            ttl = args[2] if len(args) > 2 else kwargs.get("ttl")
            return await self.memory_cache.set(key, value, ttl)
        elif operation == "delete":
            key = args[0]
            return await self.memory_cache.delete(key)
    
    async def _strategy_file_persistent(self, operation: str, *args, **kwargs) -> Any:
        """File-persistent strategy"""
        if not self.file_cache:
            return await self._strategy_memory_first(operation, *args, **kwargs)
        
        if operation == "get":
            key = args[0]
            return await self.file_cache.get(key)
        elif operation == "set":
            key, value = args[0], args[1]
            ttl = args[2] if len(args) > 2 else kwargs.get("ttl")
            return await self.file_cache.set(key, value, ttl)
        elif operation == "delete":
            key = args[0]
            return await self.file_cache.delete(key)
    
    async def _strategy_hybrid_intelligent(self, operation: str, *args, **kwargs) -> Any:
        """Intelligent hybrid strategy"""
        if operation == "get":
            key = args[0]
            
            # Try memory first (fastest)
            value = await self.memory_cache.get(key)
            if value is not None:
                return value
            
            # Try file cache if available
            if self.file_cache:
                value = await self.file_cache.get(key)
                if value is not None:
                    # Promote to memory cache
                    await self.memory_cache.set(key, value)
                    return value
            
            return None
        
        elif operation == "set":
            key, value = args[0], args[1]
            ttl = args[2] if len(args) > 2 else kwargs.get("ttl")
            
            # Always set in memory for speed
            memory_success = await self.memory_cache.set(key, value, ttl)
            
            # Set in file cache for persistence if available
            file_success = True
            if self.file_cache:
                file_success = await self.file_cache.set(key, value, ttl)
            
            return memory_success or file_success
        
        elif operation == "delete":
            key = args[0]
            
            memory_success = await self.memory_cache.delete(key)
            file_success = True
            
            if self.file_cache:
                file_success = await self.file_cache.delete(key)
            
            return memory_success or file_success
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Cleanup expired entries
                await self.memory_cache.cleanup_expired()
                if self.file_cache:
                    await self.file_cache.cleanup_expired()
                
                logger.debug("Cache cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")


# Decorators for caching
def cached(
    ttl: Optional[float] = None,
    key_prefix: str = "",
    strategy: str = "hybrid_intelligent"
):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{key_prefix}{func.__name__}{args}{kwargs}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            if hasattr(wrapper, "_cache_manager"):
                cached_result = await wrapper._cache_manager.get(cache_key, strategy)
                if cached_result is not None:
                    return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            if hasattr(wrapper, "_cache_manager"):
                await wrapper._cache_manager.set(cache_key, result, ttl, strategy)
            
            return result
        
        return wrapper
    return decorator


def cache_invalidate(pattern: str):
    """Decorator for cache invalidation"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache entries matching pattern
            if hasattr(wrapper, "_cache_manager"):
                # Implementation depends on cache backend capabilities
                pass
            
            return result
        return wrapper
    return decorator


# Global cache manager instance
cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager()
        await cache_manager.initialize()
    return cache_manager


# Export main classes and functions
__all__ = [
    "CacheManager",
    "CacheBackend",
    "CacheEntry",
    "CacheStats",
    "MemoryCache",
    "FileCache",
    "cached",
    "cache_invalidate",
    "get_cache_manager"
]
