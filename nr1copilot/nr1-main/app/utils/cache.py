
"""
ViralClip Pro v6.0 - Netflix-Level Enterprise Cache Manager
Advanced caching with Redis-like performance, intelligent invalidation, and analytics
"""

import asyncio
import logging
import json
import time
import hashlib
import pickle
import weakref
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import psutil

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Enterprise cache entry with comprehensive metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at.timestamp() > self.ttl
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at.timestamp()


@dataclass
class CacheStats:
    """Comprehensive cache statistics"""
    total_operations: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: int = 0
    entry_count: int = 0
    average_access_time: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        total = self.hits + self.misses
        self.hit_rate = (self.hits / total * 100) if total > 0 else 0.0


class NetflixLevelCacheManager:
    """Netflix-level enterprise cache with advanced features"""

    def __init__(
        self,
        max_memory_mb: int = 512,
        max_entries: int = 10000,
        default_ttl: Optional[float] = 3600,
        cleanup_interval: float = 300,
        enable_analytics: bool = True
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.enable_analytics = enable_analytics
        
        # Core storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.tag_index: Dict[str, set] = defaultdict(set)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance monitoring
        self.stats = CacheStats()
        self.operation_times: List[float] = []
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache")
        self.cleanup_task = None
        self.analytics_task = None
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Start background processes
        asyncio.create_task(self._start_background_tasks())
        
        logger.info(f"🚀 Netflix-level cache initialized: {max_memory_mb}MB, {max_entries} entries")

    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.enable_analytics:
            self.analytics_task = asyncio.create_task(self._analytics_loop())

    async def get_enterprise(
        self,
        key: str,
        default: Any = None,
        update_access: bool = True
    ) -> Any:
        """Get value with enterprise features and analytics"""
        start_time = time.time()
        
        try:
            with self.lock:
                entry = self.cache.get(key)
                
                if entry is None:
                    self.stats.misses += 1
                    self.stats.total_operations += 1
                    await self._emit_event('cache_miss', {'key': key})
                    return default
                
                if entry.is_expired:
                    await self._remove_entry(key)
                    self.stats.misses += 1
                    self.stats.total_operations += 1
                    await self._emit_event('cache_expired', {'key': key, 'age': entry.age_seconds})
                    return default
                
                # Update access metadata
                if update_access:
                    entry.last_accessed = datetime.utcnow()
                    entry.access_count += 1
                    
                    # Move to end for LRU
                    self.cache.move_to_end(key)
                    
                    # Track access patterns
                    if self.enable_analytics:
                        self.access_patterns[key].append(datetime.utcnow())
                        # Keep only recent access history
                        cutoff = datetime.utcnow() - timedelta(hours=24)
                        self.access_patterns[key] = [
                            t for t in self.access_patterns[key] if t > cutoff
                        ]
                
                self.stats.hits += 1
                self.stats.total_operations += 1
                
                return entry.value
                
        finally:
            operation_time = time.time() - start_time
            self.operation_times.append(operation_time)
            
            # Keep only recent operation times for moving average
            if len(self.operation_times) > 1000:
                self.operation_times = self.operation_times[-1000:]
            
            self.stats.average_access_time = sum(self.operation_times) / len(self.operation_times)
            self.stats.update_hit_rate()

    async def set_enterprise(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Set value with enterprise features"""
        try:
            with self.lock:
                # Calculate value size
                size_bytes = await self._calculate_size(value)
                
                # Check memory limits
                if not await self._check_memory_capacity(size_bytes):
                    await self._make_space(size_bytes)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    ttl=ttl or self.default_ttl,
                    size_bytes=size_bytes,
                    tags=tags or [],
                    metadata=metadata or {}
                )
                
                # Remove existing entry if present
                if key in self.cache:
                    await self._remove_entry(key)
                
                # Add new entry
                self.cache[key] = entry
                
                # Update tag index
                for tag in entry.tags:
                    self.tag_index[tag].add(key)
                
                # Update stats
                self.stats.entry_count = len(self.cache)
                self.stats.memory_usage = sum(e.size_bytes for e in self.cache.values())
                
                await self._emit_event('cache_set', {
                    'key': key,
                    'size_bytes': size_bytes,
                    'ttl': entry.ttl,
                    'tags': entry.tags
                })
                
                return True
                
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False

    async def delete_enterprise(self, key: str) -> bool:
        """Delete entry with enterprise cleanup"""
        try:
            with self.lock:
                if key not in self.cache:
                    return False
                
                await self._remove_entry(key)
                await self._emit_event('cache_delete', {'key': key})
                return True
                
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False

    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all entries with specified tags"""
        invalidated = 0
        
        try:
            with self.lock:
                keys_to_remove = set()
                
                for tag in tags:
                    if tag in self.tag_index:
                        keys_to_remove.update(self.tag_index[tag])
                
                for key in keys_to_remove:
                    if key in self.cache:
                        await self._remove_entry(key)
                        invalidated += 1
                
                await self._emit_event('cache_invalidate_tags', {
                    'tags': tags,
                    'invalidated_count': invalidated
                })
                
        except Exception as e:
            logger.error(f"Tag invalidation failed: {e}")
        
        return invalidated

    async def get_multi_enterprise(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values efficiently"""
        results = {}
        
        for key in keys:
            value = await self.get_enterprise(key)
            if value is not None:
                results[key] = value
        
        return results

    async def set_multi_enterprise(
        self,
        items: Dict[str, Any],
        ttl: Optional[float] = None,
        tags: List[str] = None
    ) -> Dict[str, bool]:
        """Set multiple values efficiently"""
        results = {}
        
        for key, value in items.items():
            results[key] = await self.set_enterprise(key, value, ttl, tags)
        
        return results

    async def increment_enterprise(self, key: str, delta: int = 1) -> Optional[int]:
        """Atomic increment operation"""
        try:
            with self.lock:
                entry = self.cache.get(key)
                
                if entry is None:
                    # Create new counter
                    await self.set_enterprise(key, delta)
                    return delta
                
                if not isinstance(entry.value, (int, float)):
                    raise ValueError(f"Cannot increment non-numeric value: {type(entry.value)}")
                
                new_value = entry.value + delta
                entry.value = new_value
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                
                return new_value
                
        except Exception as e:
            logger.error(f"Cache increment failed for key {key}: {e}")
            return None

    async def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive cache analytics"""
        with self.lock:
            memory_usage_mb = self.stats.memory_usage / 1024 / 1024
            
            # Calculate top accessed keys
            top_keys = sorted(
                [(k, e.access_count) for k, e in self.cache.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Calculate tag usage
            tag_usage = {
                tag: len(keys) for tag, keys in self.tag_index.items()
            }
            
            # Calculate expiration info
            expired_count = sum(1 for e in self.cache.values() if e.is_expired)
            
            return {
                "performance": {
                    "hit_rate": round(self.stats.hit_rate, 2),
                    "total_operations": self.stats.total_operations,
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "average_access_time_ms": round(self.stats.average_access_time * 1000, 3)
                },
                "memory": {
                    "usage_mb": round(memory_usage_mb, 2),
                    "max_mb": self.max_memory_bytes / 1024 / 1024,
                    "utilization_percent": round(memory_usage_mb / (self.max_memory_bytes / 1024 / 1024) * 100, 2)
                },
                "entries": {
                    "total_count": len(self.cache),
                    "max_count": self.max_entries,
                    "expired_count": expired_count,
                    "evictions": self.stats.evictions
                },
                "top_keys": [{"key": k, "access_count": c} for k, c in top_keys],
                "tag_usage": dict(sorted(tag_usage.items(), key=lambda x: x[1], reverse=True)[:10]),
                "system": {
                    "process_memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
                    "cache_efficiency": round(len(self.cache) / max(1, self.stats.total_operations) * 100, 2)
                }
            }

    async def enterprise_warm_up(self):
        """Warm up cache with predictive loading"""
        try:
            # Pre-load frequently accessed data patterns
            warm_up_keys = [
                "system:health",
                "app:config",
                "templates:popular",
                "viral:trending_factors"
            ]
            
            # Simulate warming up cache
            for key in warm_up_keys:
                await self.set_enterprise(
                    key,
                    {"status": "warm", "timestamp": datetime.utcnow().isoformat()},
                    ttl=1800,  # 30 minutes
                    tags=["warm_up", "system"]
                )
            
            logger.info(f"🔥 Cache warmed up with {len(warm_up_keys)} entries")
            
        except Exception as e:
            logger.error(f"Cache warm-up failed: {e}")

    async def _cleanup_loop(self):
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
                await self._optimize_memory()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = []
        
        with self.lock:
            for key, entry in self.cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
        
        for key in expired_keys:
            await self._remove_entry(key)
        
        if expired_keys:
            logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

    async def _optimize_memory(self):
        """Optimize memory usage"""
        with self.lock:
            current_memory = sum(e.size_bytes for e in self.cache.values())
            
            if current_memory > self.max_memory_bytes * 0.8:  # 80% threshold
                # Remove least recently used entries
                lru_keys = list(self.cache.keys())[:int(len(self.cache) * 0.1)]  # Remove 10%
                
                for key in lru_keys:
                    await self._remove_entry(key)
                
                logger.info(f"🗑️ Memory optimization: removed {len(lru_keys)} LRU entries")

    async def _analytics_loop(self):
        """Background analytics task"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self._update_analytics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache analytics error: {e}")

    async def _update_analytics(self):
        """Update cache analytics"""
        try:
            analytics = await self.get_analytics()
            
            # Log performance metrics
            if analytics["performance"]["hit_rate"] < 50:
                logger.warning(f"Low cache hit rate: {analytics['performance']['hit_rate']}%")
            
            if analytics["memory"]["utilization_percent"] > 90:
                logger.warning(f"High memory utilization: {analytics['memory']['utilization_percent']}%")
            
        except Exception as e:
            logger.error(f"Analytics update failed: {e}")

    async def _remove_entry(self, key: str):
        """Remove entry and update indexes"""
        if key not in self.cache:
            return
        
        entry = self.cache[key]
        
        # Remove from tag index
        for tag in entry.tags:
            self.tag_index[tag].discard(key)
            if not self.tag_index[tag]:
                del self.tag_index[tag]
        
        # Remove from cache
        del self.cache[key]
        
        # Update stats
        self.stats.entry_count = len(self.cache)
        self.stats.memory_usage = sum(e.size_bytes for e in self.cache.values())

    async def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate

    async def _check_memory_capacity(self, size_bytes: int) -> bool:
        """Check if there's enough memory capacity"""
        current_memory = sum(e.size_bytes for e in self.cache.values())
        return current_memory + size_bytes <= self.max_memory_bytes

    async def _make_space(self, needed_bytes: int):
        """Make space by removing LRU entries"""
        removed_bytes = 0
        removed_count = 0
        
        # Remove entries until we have enough space
        while removed_bytes < needed_bytes and self.cache:
            # Get least recently used entry
            lru_key = next(iter(self.cache))
            entry = self.cache[lru_key]
            
            removed_bytes += entry.size_bytes
            removed_count += 1
            
            await self._remove_entry(lru_key)
            self.stats.evictions += 1
        
        if removed_count > 0:
            logger.debug(f"🗑️ Evicted {removed_count} entries ({removed_bytes} bytes) to make space")

    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit cache events for monitoring"""
        if event_type not in self.event_handlers:
            return
        
        for handler in self.event_handlers[event_type]:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Event handler error for {event_type}: {e}")

    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for cache events"""
        self.event_handlers[event_type].append(handler)

    async def clear_all(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.tag_index.clear()
            self.access_patterns.clear()
            
            # Reset stats
            self.stats = CacheStats()
        
        await self._emit_event('cache_cleared', {})
        logger.info("🧹 All cache entries cleared")

    async def export_data(self, include_values: bool = False) -> Dict[str, Any]:
        """Export cache data for backup/analysis"""
        with self.lock:
            export_data = {
                "metadata": {
                    "export_time": datetime.utcnow().isoformat(),
                    "entry_count": len(self.cache),
                    "total_memory": self.stats.memory_usage
                },
                "entries": []
            }
            
            for key, entry in self.cache.items():
                entry_data = {
                    "key": key,
                    "created_at": entry.created_at.isoformat(),
                    "last_accessed": entry.last_accessed.isoformat(),
                    "access_count": entry.access_count,
                    "ttl": entry.ttl,
                    "size_bytes": entry.size_bytes,
                    "tags": entry.tags,
                    "metadata": entry.metadata
                }
                
                if include_values:
                    try:
                        entry_data["value"] = json.dumps(entry.value, default=str)
                    except:
                        entry_data["value"] = str(entry.value)
                
                export_data["entries"].append(entry_data)
            
            return export_data

    async def graceful_shutdown(self):
        """Gracefully shutdown cache manager"""
        logger.info("🔄 Starting cache manager shutdown...")
        
        # Cancel background tasks
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
        
        if self.analytics_task and not self.analytics_task.done():
            self.analytics_task.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Final analytics
        if self.enable_analytics:
            final_stats = await self.get_analytics()
            logger.info(f"📊 Final cache stats: {final_stats['performance']['hit_rate']}% hit rate, {final_stats['entries']['total_count']} entries")
        
        logger.info("✅ Cache manager shutdown complete")
