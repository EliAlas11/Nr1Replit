"""
Netflix-Grade Caching System v2.0
Advanced caching with intelligent optimization and memory management
"""

import asyncio
import time
import json
import logging
import pickle
import hashlib
import weakref
from typing import Any, Dict, Optional, Union, Callable, List, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import wraps
import threading

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata"""
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    access_count: int = 0
    size_bytes: int = 0
    priority: int = 1  # 1=low, 2=medium, 3=high

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    @property
    def age_seconds(self) -> float:
        """Get age in seconds"""
        return time.time() - self.created_at

    def update_access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1

class NetflixCacheManager:
    """Netflix-grade cache manager with intelligent optimization"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 256):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0,
            'memory_usage': 0
        }

        # Optimization settings
        self._optimization_enabled = True
        self._last_optimization = time.time()
        self._optimization_interval = 300  # 5 minutes

        # Background tasks
        self._optimization_task: Optional[asyncio.Task] = None
        self._running = True

    async def start_optimization(self):
        """Start background optimization"""
        if self._optimization_task is None:
            self._optimization_task = asyncio.create_task(self._background_optimization())

    async def stop_optimization(self):
        """Stop background optimization"""
        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass

    async def _background_optimization(self):
        """Background optimization task"""
        while self._running:
            try:
                await asyncio.sleep(self._optimization_interval)
                if self._optimization_enabled:
                    await self.optimize()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 8
            elif isinstance(value, (list, tuple, dict)):
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            else:
                return len(str(value))
        except Exception:
            return 64  # Default estimate

    def _generate_key(self, key: str, namespace: Optional[str] = None) -> str:
        """Generate namespaced cache key"""
        if namespace:
            return f"{namespace}:{key}"
        return key

    def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """Get value from cache with namespace support"""
        cache_key = self._generate_key(key, namespace)

        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is None:
                self.stats['misses'] += 1
                return None

            if entry.is_expired:
                del self._cache[cache_key]
                self.stats['misses'] += 1
                self.stats['deletes'] += 1
                return None

            # Update access statistics
            entry.update_access()

            # Move to end (LRU)
            self._cache.move_to_end(cache_key)

            self.stats['hits'] += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None, 
            namespace: Optional[str] = None, priority: int = 1) -> bool:
        """Set value in cache with enhanced options"""
        cache_key = self._generate_key(key, namespace)
        size_bytes = self._calculate_size(value)

        with self._lock:
            # Check memory limits
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False

            # Create cache entry
            entry = CacheEntry(
                value=value,
                ttl=ttl,
                size_bytes=size_bytes,
                priority=priority
            )

            # Remove old entry if exists
            if cache_key in self._cache:
                old_entry = self._cache[cache_key]
                self.stats['memory_usage'] -= old_entry.size_bytes

            self._cache[cache_key] = entry
            self.stats['memory_usage'] += size_bytes
            self.stats['sets'] += 1

            # Move to end
            self._cache.move_to_end(cache_key)

            # Trigger cleanup if needed
            if len(self._cache) > self.max_size or self.stats['memory_usage'] > self.max_memory_bytes:
                self._evict_entries()

            return True

    def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete entry from cache"""
        cache_key = self._generate_key(key, namespace)

        with self._lock:
            if cache_key in self._cache:
                entry = self._cache.pop(cache_key)
                self.stats['memory_usage'] -= entry.size_bytes
                self.stats['deletes'] += 1
                return True
            return False

    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries"""
        with self._lock:
            if namespace:
                # Clear specific namespace
                keys_to_delete = [
                    key for key in self._cache.keys() 
                    if key.startswith(f"{namespace}:")
                ]
                deleted = 0
                for key in keys_to_delete:
                    entry = self._cache.pop(key)
                    self.stats['memory_usage'] -= entry.size_bytes
                    deleted += 1
                self.stats['deletes'] += deleted
                return deleted
            else:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self.stats['memory_usage'] = 0
                self.stats['deletes'] += count
                return count

    def _evict_entries(self):
        """Intelligent cache eviction based on priority and usage"""
        target_size = int(self.max_size * 0.8)  # Evict to 80% capacity
        target_memory = int(self.max_memory_bytes * 0.8)

        # Build eviction candidates with scores
        candidates = []
        current_time = time.time()

        for key, entry in self._cache.items():
            # Calculate eviction score (higher = more likely to evict)
            score = 0

            # Age factor (older = higher score)
            age_hours = entry.age_seconds / 3600
            score += age_hours * 10

            # Access frequency factor (less accessed = higher score)
            access_freq = entry.access_count / max(age_hours, 0.1)
            score += max(0, 10 - access_freq)

            # Priority factor (lower priority = higher score)
            score += (4 - entry.priority) * 5

            # Last access factor
            hours_since_access = (current_time - entry.last_accessed) / 3600
            score += hours_since_access * 2

            candidates.append((score, key, entry))

        # Sort by score (highest first)
        candidates.sort(reverse=True)

        # Evict entries until under limits
        evicted = 0
        for score, key, entry in candidates:
            if (len(self._cache) <= target_size and 
                self.stats['memory_usage'] <= target_memory):
                break

            del self._cache[key]
            self.stats['memory_usage'] -= entry.size_bytes
            evicted += 1

        self.stats['evictions'] += evicted
        if evicted > 0:
            logger.debug(f"Evicted {evicted} cache entries")

    async def optimize(self):
        """Optimize cache performance"""
        with self._lock:
            start_time = time.time()
            optimizations = 0

            # Remove expired entries
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)

            for key in expired_keys:
                entry = self._cache.pop(key)
                self.stats['memory_usage'] -= entry.size_bytes
                optimizations += 1

            # Compact if memory usage is high
            if self.stats['memory_usage'] > self.max_memory_bytes * 0.9:
                self._evict_entries()
                optimizations += 1

            self._last_optimization = time.time()
            duration = time.time() - start_time

            if optimizations > 0:
                logger.debug(f"Cache optimization completed in {duration:.3f}s, {optimizations} changes")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_bytes': self.stats['memory_usage'],
                'memory_usage_mb': self.stats['memory_usage'] / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'hit_rate_percent': round(hit_rate, 2),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'sets': self.stats['sets'],
                'deletes': self.stats['deletes'],
                'evictions': self.stats['evictions'],
                'last_optimization': datetime.fromtimestamp(self._last_optimization).isoformat(),
                'optimization_enabled': self._optimization_enabled
            }

    async def clear_all(self):
        """Clear all cache entries"""
        return self.clear()

    def cache_decorator(self, ttl: Optional[float] = None, namespace: Optional[str] = None):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
                key = hashlib.md5(str(key_data).encode()).hexdigest()

                # Try to get from cache
                result = self.get(key, namespace)
                if result is not None:
                    return result

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(key, result, ttl, namespace)
                return result

            return wrapper
        return decorator

    async def warm_up(self, warm_up_data: Dict[str, Any]):
        """Warm up cache with initial data"""
        for key, value in warm_up_data.items():
            self.set(key, value, priority=3)  # High priority for warm-up data

        logger.info(f"Cache warmed up with {len(warm_up_data)} entries")

# Global cache manager instance
cache_manager = NetflixCacheManager()

# Backward compatibility
class CacheManager:
    """Legacy cache manager for backward compatibility"""

    def __init__(self):
        self.cache = cache_manager

    async def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        return self.cache.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        return self.cache.delete(key)

    async def clear(self) -> int:
        return self.cache.clear()

# Convenience functions
def get(key: str, namespace: Optional[str] = None) -> Optional[Any]:
    """Get value from global cache"""
    return cache_manager.get(key, namespace)

def set(key: str, value: Any, ttl: Optional[float] = None, 
        namespace: Optional[str] = None, priority: int = 1) -> bool:
    """Set value in global cache"""
    return cache_manager.set(key, value, ttl, namespace, priority)

def delete(key: str, namespace: Optional[str] = None) -> bool:
    """Delete value from global cache"""
    return cache_manager.delete(key, namespace)

def clear(namespace: Optional[str] = None) -> int:
    """Clear cache entries"""
    return cache_manager.clear(namespace)

def cached(ttl: Optional[float] = None, namespace: Optional[str] = None):
    """Decorator for caching function results"""
    return cache_manager.cache_decorator(ttl, namespace)

# Export for compatibility
cache = cache_manager
NetflixEnterpriseCache = NetflixCacheManager

logger.info("✅ Netflix-grade caching system initialized")