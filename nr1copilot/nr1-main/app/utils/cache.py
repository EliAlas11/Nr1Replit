
"""
Netflix-Level Intelligent Cache Manager
Advanced caching with performance optimization and enterprise features
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Any, Dict, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import weakref
import pickle
import zlib

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Comprehensive cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    average_get_time: float = 0.0
    average_set_time: float = 0.0
    last_cleanup: Optional[datetime] = None

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 100.0 - self.hit_rate


@dataclass
class CacheEntry:
    """Advanced cache entry with comprehensive metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0
    compressed: bool = False
    serialization_format: str = "pickle"

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at.timestamp() > self.ttl

    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds"""
        return time.time() - self.created_at.timestamp()


class EnterpriseIntelligentCacheManager:
    """Netflix-level intelligent cache manager with advanced features"""

    def __init__(self, max_size: int = 10000, max_memory_mb: int = 512):
        # Core configuration
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._tags_to_keys: Dict[str, Set[str]] = defaultdict(set)
        self._access_frequencies: Dict[str, float] = defaultdict(float)
        
        # Performance monitoring
        self.metrics = CacheMetrics()
        self._operation_times: Dict[str, list] = defaultdict(list)
        
        # Enterprise features
        self.compression_threshold = 1024  # Compress values > 1KB
        self.smart_prefetch_enabled = True
        self.adaptive_ttl_enabled = True
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = asyncio.RLock()
        
        logger.info(f"🧠 Intelligent cache manager initialized (max_size={max_size}, max_memory={max_memory_mb}MB)")

    async def enterprise_warm_up(self):
        """Warm up cache manager with enterprise features"""
        try:
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
            self._monitoring_task = asyncio.create_task(self._background_monitoring())
            
            # Pre-warm frequently accessed patterns
            await self._preload_common_patterns()
            
            logger.info("🔥 Cache manager enterprise warm-up completed")
            
        except Exception as e:
            logger.error(f"Cache manager warm-up failed: {e}", exc_info=True)

    async def get_enterprise(
        self, 
        key: str, 
        default: Any = None,
        update_access: bool = True
    ) -> Any:
        """Get value with enterprise features and performance tracking"""
        start_time = time.time()
        
        try:
            async with self._lock:
                self.metrics.total_requests += 1
                
                if key not in self._cache:
                    self.metrics.cache_misses += 1
                    self._record_operation_time('get_miss', start_time)
                    return default
                
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired:
                    await self._remove_entry(key)
                    self.metrics.cache_misses += 1
                    self._record_operation_time('get_expired', start_time)
                    return default
                
                # Update access patterns
                if update_access:
                    entry.last_accessed = datetime.utcnow()
                    entry.access_count += 1
                    self._access_frequencies[key] = self._calculate_access_frequency(entry)
                    
                    # Move to end (LRU)
                    self._cache.move_to_end(key)
                
                self.metrics.cache_hits += 1
                
                # Decompress if needed
                value = await self._deserialize_value(entry.value, entry.compressed, entry.serialization_format)
                
                self._record_operation_time('get_hit', start_time)
                return value
                
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return default

    async def set_enterprise(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None,
        compression: Optional[bool] = None
    ) -> bool:
        """Set value with enterprise features and optimization"""
        start_time = time.time()
        
        try:
            async with self._lock:
                # Serialize and optionally compress value
                serialized_value, is_compressed, size_bytes = await self._serialize_value(
                    value, compression
                )
                
                # Check memory constraints
                if not await self._check_memory_constraints(size_bytes, key):
                    logger.warning(f"Cache set failed for key '{key}': memory constraints")
                    return False
                
                # Remove existing entry if present
                if key in self._cache:
                    await self._remove_entry(key)
                
                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=serialized_value,
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    ttl=ttl,
                    tags=tags or set(),
                    size_bytes=size_bytes,
                    compressed=is_compressed,
                    serialization_format="pickle"
                )
                
                # Add to cache
                self._cache[key] = entry
                
                # Update tag mappings
                for tag in entry.tags:
                    self._tags_to_keys[tag].add(key)
                
                # Update metrics
                self.metrics.memory_usage_bytes += size_bytes
                
                # Trigger cleanup if needed
                await self._enforce_size_limits()
                
                self._record_operation_time('set', start_time)
                return True
                
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False

    async def delete_enterprise(self, key: str) -> bool:
        """Delete entry with enterprise tracking"""
        try:
            async with self._lock:
                if key in self._cache:
                    await self._remove_entry(key)
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False

    async def clear_by_tags(self, tags: Set[str]) -> int:
        """Clear cache entries by tags"""
        try:
            async with self._lock:
                keys_to_remove = set()
                
                for tag in tags:
                    if tag in self._tags_to_keys:
                        keys_to_remove.update(self._tags_to_keys[tag])
                
                for key in keys_to_remove:
                    await self._remove_entry(key)
                
                return len(keys_to_remove)
                
        except Exception as e:
            logger.error(f"Cache clear by tags error: {e}")
            return 0

    async def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        async with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_mb": self.metrics.memory_usage_bytes / 1024 / 1024,
                "max_memory_mb": self.max_memory_bytes / 1024 / 1024,
                "hit_rate": self.metrics.hit_rate,
                "miss_rate": self.metrics.miss_rate,
                "metrics": {
                    "total_requests": self.metrics.total_requests,
                    "cache_hits": self.metrics.cache_hits,
                    "cache_misses": self.metrics.cache_misses,
                    "evictions": self.metrics.evictions,
                    "average_get_time": self.metrics.average_get_time,
                    "average_set_time": self.metrics.average_set_time,
                    "last_cleanup": self.metrics.last_cleanup.isoformat() if self.metrics.last_cleanup else None
                },
                "top_keys": await self._get_top_accessed_keys(10),
                "expiring_soon": await self._get_expiring_entries(60),  # Next 60 seconds
                "tag_counts": {tag: len(keys) for tag, keys in self._tags_to_keys.items()}
            }

    async def _serialize_value(
        self, 
        value: Any, 
        force_compression: Optional[bool] = None
    ) -> Tuple[bytes, bool, int]:
        """Serialize and optionally compress value"""
        try:
            # Serialize
            serialized = pickle.dumps(value)
            size_bytes = len(serialized)
            
            # Determine if compression should be used
            should_compress = (
                force_compression if force_compression is not None
                else size_bytes > self.compression_threshold
            )
            
            if should_compress:
                compressed = zlib.compress(serialized, level=6)
                # Only use compression if it actually reduces size
                if len(compressed) < size_bytes:
                    return compressed, True, len(compressed)
            
            return serialized, False, size_bytes
            
        except Exception as e:
            logger.error(f"Value serialization failed: {e}")
            raise

    async def _deserialize_value(
        self, 
        data: bytes, 
        is_compressed: bool, 
        format_type: str
    ) -> Any:
        """Deserialize and decompress value"""
        try:
            if is_compressed:
                data = zlib.decompress(data)
            
            if format_type == "pickle":
                return pickle.loads(data)
            else:
                raise ValueError(f"Unsupported serialization format: {format_type}")
                
        except Exception as e:
            logger.error(f"Value deserialization failed: {e}")
            raise

    async def _check_memory_constraints(self, new_size: int, key: str) -> bool:
        """Check if adding new entry would exceed memory limits"""
        current_size = self.metrics.memory_usage_bytes
        
        # Account for existing entry size if updating
        if key in self._cache:
            current_size -= self._cache[key].size_bytes
        
        return (current_size + new_size) <= self.max_memory_bytes

    async def _remove_entry(self, key: str):
        """Remove entry and update all tracking structures"""
        if key not in self._cache:
            return
        
        entry = self._cache[key]
        
        # Update memory usage
        self.metrics.memory_usage_bytes -= entry.size_bytes
        
        # Remove from tag mappings
        for tag in entry.tags:
            self._tags_to_keys[tag].discard(key)
            if not self._tags_to_keys[tag]:
                del self._tags_to_keys[tag]
        
        # Remove from frequency tracking
        self._access_frequencies.pop(key, None)
        
        # Remove from cache
        del self._cache[key]

    async def _enforce_size_limits(self):
        """Enforce cache size and memory limits using intelligent eviction"""
        if len(self._cache) <= self.max_size and self.metrics.memory_usage_bytes <= self.max_memory_bytes:
            return
        
        # Calculate how many entries to evict
        entries_to_evict = max(0, len(self._cache) - self.max_size)
        
        # If memory pressure, evict more aggressively
        if self.metrics.memory_usage_bytes > self.max_memory_bytes:
            memory_pressure = self.metrics.memory_usage_bytes / self.max_memory_bytes
            entries_to_evict = max(entries_to_evict, int(len(self._cache) * (memory_pressure - 1.0)))
        
        if entries_to_evict <= 0:
            return
        
        # Get eviction candidates using intelligent scoring
        eviction_candidates = await self._get_eviction_candidates(entries_to_evict)
        
        # Evict entries
        for key in eviction_candidates:
            await self._remove_entry(key)
            self.metrics.evictions += 1

    async def _get_eviction_candidates(self, count: int) -> List[str]:
        """Get intelligent eviction candidates based on multiple factors"""
        candidates = []
        
        for key, entry in self._cache.items():
            # Calculate eviction score (higher = more likely to evict)
            score = await self._calculate_eviction_score(entry)
            candidates.append((score, key))
        
        # Sort by score (highest first) and return top candidates
        candidates.sort(reverse=True)
        return [key for _, key in candidates[:count]]

    async def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate intelligent eviction score based on multiple factors"""
        now = time.time()
        
        # Age factor (older = higher score)
        age_factor = entry.age_seconds / 3600  # Hours
        
        # Access frequency factor (less frequent = higher score)
        access_frequency = self._access_frequencies.get(entry.key, 0.0)
        frequency_factor = 1.0 / (access_frequency + 0.1)  # Avoid division by zero
        
        # Time since last access (longer = higher score)
        last_access_factor = (now - entry.last_accessed.timestamp()) / 3600  # Hours
        
        # Size factor (larger = slightly higher score)
        size_factor = entry.size_bytes / (1024 * 1024)  # MB
        
        # TTL factor (expiring soon = higher score)
        ttl_factor = 0.0
        if entry.ttl is not None:
            remaining_ttl = entry.ttl - entry.age_seconds
            if remaining_ttl > 0:
                ttl_factor = 1.0 / (remaining_ttl + 1)
        
        # Combine factors with weights
        score = (
            age_factor * 0.3 +
            frequency_factor * 0.4 +
            last_access_factor * 0.2 +
            size_factor * 0.05 +
            ttl_factor * 0.05
        )
        
        return score

    def _calculate_access_frequency(self, entry: CacheEntry) -> float:
        """Calculate access frequency with time decay"""
        age_hours = entry.age_seconds / 3600
        time_decay = 1.0 / (1.0 + age_hours * 0.1)  # Decay over time
        return entry.access_count * time_decay

    async def _get_top_accessed_keys(self, limit: int) -> List[Dict[str, Any]]:
        """Get top accessed cache keys"""
        sorted_keys = sorted(
            self._access_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        result = []
        for key, frequency in sorted_keys[:limit]:
            if key in self._cache:
                entry = self._cache[key]
                result.append({
                    "key": key,
                    "access_count": entry.access_count,
                    "frequency": frequency,
                    "age_seconds": entry.age_seconds,
                    "size_bytes": entry.size_bytes
                })
        
        return result

    async def _get_expiring_entries(self, within_seconds: int) -> List[Dict[str, Any]]:
        """Get entries expiring within specified time"""
        current_time = time.time()
        expiring = []
        
        for key, entry in self._cache.items():
            if entry.ttl is not None:
                expires_at = entry.created_at.timestamp() + entry.ttl
                if expires_at <= current_time + within_seconds:
                    expiring.append({
                        "key": key,
                        "expires_in_seconds": max(0, expires_at - current_time),
                        "size_bytes": entry.size_bytes
                    })
        
        return sorted(expiring, key=lambda x: x["expires_in_seconds"])

    async def _background_cleanup(self):
        """Background task for cache cleanup and optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                async with self._lock:
                    # Remove expired entries
                    expired_keys = []
                    for key, entry in self._cache.items():
                        if entry.is_expired:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        await self._remove_entry(key)
                    
                    if expired_keys:
                        logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
                    
                    # Update cleanup timestamp
                    self.metrics.last_cleanup = datetime.utcnow()
                    
                    # Optimize memory usage if needed
                    if self.metrics.memory_usage_bytes > self.max_memory_bytes * 0.8:
                        await self._enforce_size_limits()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")

    async def _background_monitoring(self):
        """Background monitoring and optimization"""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Update average operation times
                for operation, times in self._operation_times.items():
                    if times:
                        avg_time = sum(times) / len(times)
                        if operation.startswith('get'):
                            self.metrics.average_get_time = avg_time
                        elif operation == 'set':
                            self.metrics.average_set_time = avg_time
                        
                        # Keep only recent times (last 100 operations)
                        self._operation_times[operation] = times[-100:]
                
                # Log performance metrics periodically
                if self.metrics.total_requests > 0:
                    logger.info(
                        f"📊 Cache performance: {self.metrics.hit_rate:.1f}% hit rate, "
                        f"{len(self._cache)} entries, "
                        f"{self.metrics.memory_usage_bytes / 1024 / 1024:.1f}MB memory"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")

    def _record_operation_time(self, operation: str, start_time: float):
        """Record operation timing for performance monitoring"""
        duration = time.time() - start_time
        self._operation_times[operation].append(duration)
        
        # Keep only recent times to prevent memory growth
        if len(self._operation_times[operation]) > 1000:
            self._operation_times[operation] = self._operation_times[operation][-500:]

    async def _preload_common_patterns(self):
        """Pre-load common cache patterns for better performance"""
        # This would typically load frequently accessed data
        # For now, just log that pre-loading is happening
        logger.info("🔄 Pre-loading common cache patterns...")

    async def health_check(self) -> bool:
        """Perform health check on cache manager"""
        try:
            # Test basic operations
            test_key = "__health_check__"
            test_value = {"timestamp": time.time(), "test": True}
            
            # Test set
            success = await self.set_enterprise(test_key, test_value, ttl=10)
            if not success:
                return False
            
            # Test get
            retrieved = await self.get_enterprise(test_key)
            if retrieved != test_value:
                return False
            
            # Test delete
            deleted = await self.delete_enterprise(test_key)
            if not deleted:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False

    async def graceful_shutdown(self):
        """Gracefully shutdown cache manager"""
        logger.info("🔄 Shutting down cache manager...")
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Clear cache
        async with self._lock:
            self._cache.clear()
            self._tags_to_keys.clear()
            self._access_frequencies.clear()
            self._operation_times.clear()
        
        logger.info("✅ Cache manager shutdown complete")
