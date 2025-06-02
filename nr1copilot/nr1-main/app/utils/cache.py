"""
Netflix-Level Cache Manager
Enterprise caching with intelligent invalidation and performance optimization
"""

import asyncio
import json
import time
import hashlib
import logging
import pickle
import weakref
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata and analytics"""
    value: Any
    created_at: float
    ttl: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    compression_ratio: float = 1.0

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > (self.created_at + self.ttl)

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.created_at

    @property
    def access_frequency(self) -> float:
        """Calculate access frequency per hour"""
        age_hours = max(self.age_seconds / 3600, 0.01)
        return self.access_count / age_hours


class NetflixLevelCacheManager:
    """Netflix-level cache with enterprise patterns and performance optimization"""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        # Core cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_lock = asyncio.Lock()

        # Configuration
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.compression_threshold = 1024  # bytes

        # Performance metrics
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size_bytes": 0,
            "avg_access_time_ms": 0.0,
            "compression_ratio": 0.0
        }

        # Advanced features
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._warming_tasks: Set[asyncio.Task] = set()

        # Background tasks
        self._cleanup_task = None
        self._metrics_task = None
        self._is_running = False

        logger.info("🚀 Netflix-level cache manager initialized")

    async def enterprise_warm_up(self):
        """Enterprise warm-up with preloading and optimization"""
        try:
            start_time = time.time()

            # Start background tasks
            await self._start_background_tasks()

            # Preload critical data
            await self._preload_critical_cache()

            # Initialize performance monitoring
            await self._initialize_performance_monitoring()

            warm_up_time = time.time() - start_time
            logger.info(f"🔥 Cache manager warm-up completed in {warm_up_time:.2f}s")

        except Exception as e:
            logger.error(f"Cache warm-up failed: {e}", exc_info=True)

    async def get_enterprise(
        self, 
        key: str, 
        default: Any = None,
        update_access: bool = True
    ) -> Any:
        """Enterprise get with performance monitoring and analytics"""
        start_time = time.time()

        try:
            async with self._cache_lock:
                entry = self._cache.get(key)

                if entry is None:
                    self.metrics["misses"] += 1
                    self._record_access_pattern(key, False)
                    return default

                if entry.is_expired:
                    await self._remove_entry(key)
                    self.metrics["misses"] += 1
                    self._record_access_pattern(key, False)
                    return default

                # Update access metadata
                if update_access:
                    entry.access_count += 1
                    entry.last_accessed = time.time()

                    # Move to end for LRU
                    self._cache.move_to_end(key)

                self.metrics["hits"] += 1
                self._record_access_pattern(key, True)

                # Update performance metrics
                access_time = (time.time() - start_time) * 1000
                self._update_access_time_metric(access_time)

                return entry.value

        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return default

    async def set_enterprise(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[Set[str]] = None,
        compress: bool = True
    ) -> bool:
        """Enterprise set with compression, tagging, and dependency tracking"""
        try:
            async with self._cache_lock:
                # Prepare value and calculate size
                processed_value, size_bytes, compression_ratio = await self._process_value(
                    value, compress
                )

                # Create cache entry
                entry = CacheEntry(
                    value=processed_value,
                    created_at=time.time(),
                    ttl=ttl or self.default_ttl,
                    size_bytes=size_bytes,
                    tags=tags or set(),
                    dependencies=dependencies or set(),
                    compression_ratio=compression_ratio
                )

                # Check if we need to evict
                await self._ensure_capacity(size_bytes)

                # Store entry
                self._cache[key] = entry
                self._cache.move_to_end(key)

                # Update indexes
                self._update_tag_index(key, tags or set())
                self._update_dependency_graph(key, dependencies or set())

                # Update metrics
                self.metrics["total_size_bytes"] += size_bytes
                self._update_compression_metric(compression_ratio)

                return True

        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False

    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate cache entries by tags with cascading dependencies"""
        invalidated_count = 0

        try:
            async with self._cache_lock:
                keys_to_remove = set()

                # Find all keys with matching tags
                for tag in tags:
                    keys_to_remove.update(self._tag_index.get(tag, set()))

                # Add dependent keys (cascading invalidation)
                cascaded_keys = set()
                for key in keys_to_remove:
                    cascaded_keys.update(self._get_dependent_keys(key))

                keys_to_remove.update(cascaded_keys)

                # Remove entries
                for key in keys_to_remove:
                    if key in self._cache:
                        await self._remove_entry(key)
                        invalidated_count += 1

                logger.info(f"🗑️ Invalidated {invalidated_count} cache entries by tags: {tags}")

        except Exception as e:
            logger.error(f"Tag invalidation failed: {e}")

        return invalidated_count

    async def get_cache_analytics(self) -> Dict[str, Any]:
        """Get comprehensive cache analytics and performance metrics"""
        try:
            async with self._cache_lock:
                total_entries = len(self._cache)
                hit_rate = 0.0
                if self.metrics["hits"] + self.metrics["misses"] > 0:
                    hit_rate = self.metrics["hits"] / (self.metrics["hits"] + self.metrics["misses"])

                # Calculate memory usage by category
                memory_by_tags = defaultdict(int)
                access_patterns = {}

                for key, entry in self._cache.items():
                    for tag in entry.tags:
                        memory_by_tags[tag] += entry.size_bytes

                    access_patterns[key] = {
                        "access_count": entry.access_count,
                        "access_frequency": entry.access_frequency,
                        "age_seconds": entry.age_seconds,
                        "size_bytes": entry.size_bytes
                    }

                # Top accessed entries
                top_entries = sorted(
                    access_patterns.items(),
                    key=lambda x: x[1]["access_frequency"],
                    reverse=True
                )[:10]

                return {
                    "cache_status": {
                        "total_entries": total_entries,
                        "total_size_mb": self.metrics["total_size_bytes"] / 1024 / 1024,
                        "hit_rate": hit_rate,
                        "avg_access_time_ms": self.metrics["avg_access_time_ms"],
                        "compression_ratio": self.metrics["compression_ratio"]
                    },
                    "performance": {
                        "hits": self.metrics["hits"],
                        "misses": self.metrics["misses"],
                        "evictions": self.metrics["evictions"],
                        "memory_efficiency": min(100.0, (hit_rate * 100))
                    },
                    "memory_distribution": dict(memory_by_tags),
                    "top_accessed_entries": [
                        {"key": k, **v} for k, v in top_entries
                    ],
                    "cache_health": {
                        "fragmentation": self._calculate_fragmentation(),
                        "utilization": min(100.0, (total_entries / self.max_size) * 100),
                        "average_entry_age": self._calculate_average_age()
                    }
                }

        except Exception as e:
            logger.error(f"Analytics generation failed: {e}")
            return {"error": str(e)}

    async def _process_value(
        self, 
        value: Any, 
        compress: bool
    ) -> tuple[Any, int, float]:
        """Process value with optional compression"""
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value, default=str).encode('utf-8')
            else:
                serialized = pickle.dumps(value)

            original_size = len(serialized)

            # Apply compression if beneficial
            if compress and original_size > self.compression_threshold:
                import gzip
                compressed = gzip.compress(serialized)

                if len(compressed) < original_size * 0.8:  # 20% savings threshold
                    return compressed, len(compressed), original_size / len(compressed)

            return serialized, original_size, 1.0

        except Exception as e:
            logger.error(f"Value processing failed: {e}")
            return value, len(str(value)), 1.0

    async def _ensure_capacity(self, needed_bytes: int):
        """Ensure cache has capacity using intelligent eviction"""
        current_size = self.metrics["total_size_bytes"]

        if len(self._cache) >= self.max_size or (current_size + needed_bytes) > (self.max_size * 1024):
            # Intelligent eviction based on access patterns and age
            eviction_candidates = []

            for key, entry in self._cache.items():
                # Calculate eviction score (lower is better for eviction)
                age_factor = min(entry.age_seconds / 3600, 10)  # Cap at 10 hours
                access_factor = max(entry.access_frequency, 0.1)
                size_factor = entry.size_bytes / 1024  # KB

                eviction_score = (age_factor * size_factor) / access_factor
                eviction_candidates.append((key, eviction_score))

            # Sort by eviction score (highest first)
            eviction_candidates.sort(key=lambda x: x[1], reverse=True)

            # Evict entries until we have enough space
            for key, _ in eviction_candidates:
                if (len(self._cache) < self.max_size * 0.8 and 
                    self.metrics["total_size_bytes"] + needed_bytes < self.max_size * 1024 * 0.8):
                    break

                await self._remove_entry(key)
                self.metrics["evictions"] += 1

    async def _remove_entry(self, key: str):
        """Remove entry and clean up indexes"""
        if key in self._cache:
            entry = self._cache[key]

            # Update metrics
            self.metrics["total_size_bytes"] -= entry.size_bytes

            # Clean up indexes
            for tag in entry.tags:
                self._tag_index[tag].discard(key)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]

            # Clean up dependencies
            for dep in entry.dependencies:
                self._dependency_graph[dep].discard(key)
                if not self._dependency_graph[dep]:
                    del self._dependency_graph[dep]

            # Remove from cache
            del self._cache[key]

    def _update_tag_index(self, key: str, tags: Set[str]):
        """Update tag index for efficient tag-based queries"""
        for tag in tags:
            self._tag_index[tag].add(key)

    def _update_dependency_graph(self, key: str, dependencies: Set[str]):
        """Update dependency graph for cascading invalidation"""
        for dep in dependencies:
            self._dependency_graph[dep].add(key)

    def _get_dependent_keys(self, key: str) -> Set[str]:
        """Get all keys that depend on the given key"""
        return self._dependency_graph.get(key, set())

    def _record_access_pattern(self, key: str, hit: bool):
        """Record access pattern for analytics"""
        self._access_patterns[key].append(time.time())

        # Keep only recent access patterns (last hour)
        cutoff = time.time() - 3600
        self._access_patterns[key] = [
            t for t in self._access_patterns[key] if t > cutoff
        ]

    def _update_access_time_metric(self, access_time_ms: float):
        """Update rolling average access time"""
        current_avg = self.metrics["avg_access_time_ms"]
        total_requests = self.metrics["hits"] + self.metrics["misses"]

        if total_requests > 0:
            self.metrics["avg_access_time_ms"] = (
                (current_avg * (total_requests - 1) + access_time_ms) / total_requests
            )

    def _update_compression_metric(self, ratio: float):
        """Update rolling average compression ratio"""
        current_ratio = self.metrics["compression_ratio"]
        entries_count = len(self._cache)

        if entries_count > 0:
            self.metrics["compression_ratio"] = (
                (current_ratio * (entries_count - 1) + ratio) / entries_count
            )

    def _calculate_fragmentation(self) -> float:
        """Calculate cache fragmentation percentage"""
        if not self._cache:
            return 0.0

        sizes = [entry.size_bytes for entry in self._cache.values()]
        avg_size = sum(sizes) / len(sizes)
        variance = sum((size - avg_size) ** 2 for size in sizes) / len(sizes)

        return min(100.0, (variance / (avg_size ** 2)) * 100)

    def _calculate_average_age(self) -> float:
        """Calculate average age of cache entries in seconds"""
        if not self._cache:
            return 0.0

        current_time = time.time()
        total_age = sum(
            current_time - entry.created_at 
            for entry in self._cache.values()
        )

        return total_age / len(self._cache)

    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self._is_running = True

        # Cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Metrics task  
        self._metrics_task = asyncio.create_task(self._metrics_loop())

    async def _cleanup_loop(self):
        """Background cleanup of expired entries"""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # 5 minutes

                async with self._cache_lock:
                    expired_keys = []
                    for key, entry in self._cache.items():
                        if entry.is_expired:
                            expired_keys.append(key)

                    for key in expired_keys:
                        await self._remove_entry(key)

                    if expired_keys:
                        logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")

    async def _metrics_loop(self):
        """Background metrics collection and optimization"""
        while self._is_running:
            try:
                await asyncio.sleep(600)  # 10 minutes

                # Log performance metrics
                analytics = await self.get_cache_analytics()
                logger.info(f"📊 Cache performance: {analytics['cache_status']}")

            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")

    async def _preload_critical_cache(self):
        """Preload critical cache data"""
        # This would typically load from a persistent store
        # For now, we'll just initialize some common cache patterns
        await self.set_enterprise("system_config", {"initialized": True}, ttl=86400)
        logger.info("🔥 Critical cache data preloaded")

    async def _initialize_performance_monitoring(self):
        """Initialize performance monitoring"""
        logger.info("📊 Performance monitoring initialized")

    async def graceful_shutdown(self):
        """Graceful shutdown of cache manager"""
        self._is_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()

        # Cancel warming tasks
        for task in self._warming_tasks:
            task.cancel()

        logger.info("✅ Cache manager shutdown complete")