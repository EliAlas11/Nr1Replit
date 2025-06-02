
"""
Netflix-Grade Cache Manager v14.0
High-performance caching with intelligent eviction, monitoring, and optimization
"""

import asyncio
import logging
import time
import json
import hashlib
import weakref
from typing import Any, Dict, Optional, Union, Callable, List, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


class CacheLevel(Enum):
    """Cache performance levels"""
    L1_MEMORY = "l1_memory"  # Ultra-fast in-memory cache
    L2_COMPRESSED = "l2_compressed"  # Compressed memory cache
    L3_PERSISTENT = "l3_persistent"  # Persistent disk cache


@dataclass
class CacheEntry:
    """Cache entry with comprehensive metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    cache_level: CacheLevel = CacheLevel.L1_MEMORY
    compression_ratio: float = 1.0
    
    @property
    def age(self) -> float:
        """Get entry age in seconds"""
        return time.time() - self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return self.age > self.ttl
    
    def access(self) -> None:
        """Record access to this entry"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Comprehensive cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    avg_access_time_ms: float = 0.0
    hit_rate_percent: float = 0.0
    memory_usage_mb: float = 0.0
    
    def update_hit_rate(self) -> None:
        """Update hit rate percentage"""
        if self.total_requests > 0:
            self.hit_rate_percent = (self.hits / self.total_requests) * 100


class NetflixCacheManager:
    """Netflix-grade cache manager with multi-level caching and intelligent optimization"""
    
    def __init__(
        self,
        max_size_mb: int = 512,
        default_ttl: float = 3600,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_compression: bool = True
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_compression = enable_compression
        
        # Multi-level cache storage
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l2_cache: Dict[str, CacheEntry] = {}
        self._access_patterns: defaultdict = defaultdict(list)
        
        # Statistics and monitoring
        self.stats = CacheStats()
        self._lock = threading.RLock()
        
        # Optimization settings
        self._optimization_enabled = True
        self._last_optimization = time.time()
        self._optimization_interval = 300  # 5 minutes
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        
        logger.info(f"🧠 Netflix Cache Manager v14.0 initialized - {max_size_mb}MB capacity")
    
    async def initialize(self) -> None:
        """Initialize cache manager with background tasks"""
        try:
            # Start background optimization
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
            self._optimization_task = asyncio.create_task(self._background_optimization())
            
            logger.info("✅ Cache manager background tasks started")
            
        except Exception as e:
            logger.error(f"❌ Cache manager initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown cache manager"""
        try:
            # Cancel background tasks
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
            
            if self._optimization_task and not self._optimization_task.done():
                self._optimization_task.cancel()
            
            # Clear caches
            await self.clear_all()
            
            logger.info("✅ Cache manager shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Cache manager shutdown failed: {e}")
    
    def _generate_key(self, key: Union[str, dict, tuple]) -> str:
        """Generate consistent cache key"""
        if isinstance(key, str):
            return key
        
        # Create hash for complex keys
        key_str = json.dumps(key, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of cache value"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (dict, list, tuple)):
                return len(json.dumps(value, default=str))
            else:
                return len(str(value))
        except Exception:
            return 64  # Default size estimate
    
    async def get(self, key: Union[str, dict, tuple], default: Any = None) -> Any:
        """Get value from cache with intelligent access tracking"""
        start_time = time.time()
        cache_key = self._generate_key(key)
        
        try:
            with self._lock:
                self.stats.total_requests += 1
                
                # Check L1 cache first
                if cache_key in self._l1_cache:
                    entry = self._l1_cache[cache_key]
                    
                    if entry.is_expired:
                        self._remove_entry(cache_key)
                        self.stats.misses += 1
                        return default
                    
                    # Move to end (LRU)
                    self._l1_cache.move_to_end(cache_key)
                    entry.access()
                    
                    self.stats.hits += 1
                    access_time = (time.time() - start_time) * 1000
                    self._update_access_time(access_time)
                    
                    return entry.value
                
                # Check L2 cache
                if cache_key in self._l2_cache:
                    entry = self._l2_cache[cache_key]
                    
                    if entry.is_expired:
                        del self._l2_cache[cache_key]
                        self.stats.misses += 1
                        return default
                    
                    # Promote to L1
                    await self._promote_to_l1(cache_key, entry)
                    entry.access()
                    
                    self.stats.hits += 1
                    access_time = (time.time() - start_time) * 1000
                    self._update_access_time(access_time)
                    
                    return entry.value
                
                # Cache miss
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return default
                
        except Exception as e:
            logger.error(f"❌ Cache get failed for key {cache_key}: {e}")
            self.stats.misses += 1
            return default
    
    async def set(
        self, 
        key: Union[str, dict, tuple], 
        value: Any, 
        ttl: Optional[float] = None,
        cache_level: CacheLevel = CacheLevel.L1_MEMORY
    ) -> bool:
        """Set value in cache with intelligent placement"""
        cache_key = self._generate_key(key)
        
        try:
            with self._lock:
                # Calculate entry size
                size_bytes = self._calculate_size(value)
                
                # Check if value is too large
                if size_bytes > self.max_size_bytes * 0.1:  # 10% of total cache
                    logger.warning(f"⚠️ Value too large for cache: {size_bytes} bytes")
                    return False
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl=ttl or self.default_ttl,
                    size_bytes=size_bytes,
                    cache_level=cache_level
                )
                
                # Ensure space is available
                await self._ensure_space(size_bytes)
                
                # Store in appropriate cache level
                if cache_level == CacheLevel.L1_MEMORY:
                    self._l1_cache[cache_key] = entry
                    # Maintain LRU order
                    self._l1_cache.move_to_end(cache_key)
                else:
                    self._l2_cache[cache_key] = entry
                
                # Update statistics
                self.stats.entry_count += 1
                self.stats.total_size_bytes += size_bytes
                self.stats.memory_usage_mb = self.stats.total_size_bytes / (1024 * 1024)
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Cache set failed for key {cache_key}: {e}")
            return False
    
    async def delete(self, key: Union[str, dict, tuple]) -> bool:
        """Delete entry from cache"""
        cache_key = self._generate_key(key)
        
        try:
            with self._lock:
                removed = False
                
                # Remove from L1
                if cache_key in self._l1_cache:
                    entry = self._l1_cache.pop(cache_key)
                    self._update_stats_on_removal(entry)
                    removed = True
                
                # Remove from L2
                if cache_key in self._l2_cache:
                    entry = self._l2_cache.pop(cache_key)
                    if not removed:  # Only update stats once
                        self._update_stats_on_removal(entry)
                    removed = True
                
                return removed
                
        except Exception as e:
            logger.error(f"❌ Cache delete failed for key {cache_key}: {e}")
            return False
    
    async def clear_all(self) -> None:
        """Clear all cache entries"""
        try:
            with self._lock:
                self._l1_cache.clear()
                self._l2_cache.clear()
                self._access_patterns.clear()
                
                # Reset statistics
                self.stats = CacheStats()
                
            logger.info("🧹 All cache entries cleared")
            
        except Exception as e:
            logger.error(f"❌ Cache clear failed: {e}")
    
    async def _ensure_space(self, required_bytes: int) -> None:
        """Ensure sufficient space is available in cache"""
        current_size = self.stats.total_size_bytes
        
        while current_size + required_bytes > self.max_size_bytes:
            if not self._l1_cache and not self._l2_cache:
                break  # No more entries to evict
            
            # Apply eviction strategy
            evicted = await self._evict_entry()
            if not evicted:
                break
            
            current_size = self.stats.total_size_bytes
    
    async def _evict_entry(self) -> bool:
        """Evict an entry based on the configured strategy"""
        try:
            if self.strategy == CacheStrategy.LRU:
                return self._evict_lru()
            elif self.strategy == CacheStrategy.LFU:
                return self._evict_lfu()
            elif self.strategy == CacheStrategy.TTL:
                return self._evict_expired()
            else:  # ADAPTIVE
                return self._evict_adaptive()
                
        except Exception as e:
            logger.error(f"❌ Cache eviction failed: {e}")
            return False
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry"""
        if self._l1_cache:
            # Get oldest entry from L1
            key = next(iter(self._l1_cache))
            self._remove_entry(key)
            return True
        elif self._l2_cache:
            # Find LRU in L2
            oldest_key = min(
                self._l2_cache.keys(),
                key=lambda k: self._l2_cache[k].last_accessed
            )
            self._remove_entry(oldest_key)
            return True
        
        return False
    
    def _evict_lfu(self) -> bool:
        """Evict least frequently used entry"""
        all_entries = {**self._l1_cache, **self._l2_cache}
        
        if all_entries:
            # Find entry with lowest access count
            lfu_key = min(
                all_entries.keys(),
                key=lambda k: all_entries[k].access_count
            )
            self._remove_entry(lfu_key)
            return True
        
        return False
    
    def _evict_expired(self) -> bool:
        """Evict expired entries first"""
        current_time = time.time()
        
        # Check L1 cache
        for key, entry in list(self._l1_cache.items()):
            if entry.is_expired:
                self._remove_entry(key)
                return True
        
        # Check L2 cache
        for key, entry in list(self._l2_cache.items()):
            if entry.is_expired:
                self._remove_entry(key)
                return True
        
        # If no expired entries, fall back to LRU
        return self._evict_lru()
    
    def _evict_adaptive(self) -> bool:
        """Adaptive eviction based on access patterns"""
        # Use expired entries first
        if self._evict_expired():
            return True
        
        # Then use LFU for better cache efficiency
        return self._evict_lfu()
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and update statistics"""
        entry = None
        
        if key in self._l1_cache:
            entry = self._l1_cache.pop(key)
        elif key in self._l2_cache:
            entry = self._l2_cache.pop(key)
        
        if entry:
            self._update_stats_on_removal(entry)
            self.stats.evictions += 1
    
    def _update_stats_on_removal(self, entry: CacheEntry) -> None:
        """Update statistics when an entry is removed"""
        self.stats.entry_count = max(0, self.stats.entry_count - 1)
        self.stats.total_size_bytes = max(0, self.stats.total_size_bytes - entry.size_bytes)
        self.stats.memory_usage_mb = self.stats.total_size_bytes / (1024 * 1024)
    
    def _update_access_time(self, access_time_ms: float) -> None:
        """Update average access time"""
        if self.stats.avg_access_time_ms == 0:
            self.stats.avg_access_time_ms = access_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * self.stats.avg_access_time_ms
            )
    
    async def _promote_to_l1(self, key: str, entry: CacheEntry) -> None:
        """Promote entry from L2 to L1 cache"""
        try:
            # Remove from L2
            if key in self._l2_cache:
                del self._l2_cache[key]
            
            # Add to L1
            entry.cache_level = CacheLevel.L1_MEMORY
            self._l1_cache[key] = entry
            self._l1_cache.move_to_end(key)
            
        except Exception as e:
            logger.error(f"❌ Cache promotion failed for key {key}: {e}")
    
    async def _background_cleanup(self) -> None:
        """Background task for cleaning expired entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = time.time()
                expired_keys = []
                
                # Find expired entries
                with self._lock:
                    for key, entry in {**self._l1_cache, **self._l2_cache}.items():
                        if entry.is_expired:
                            expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    await self.delete(key)
                
                if expired_keys:
                    logger.debug(f"🧹 Cleaned {len(expired_keys)} expired cache entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Cache cleanup failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _background_optimization(self) -> None:
        """Background task for cache optimization"""
        while True:
            try:
                await asyncio.sleep(self._optimization_interval)
                
                if not self._optimization_enabled:
                    continue
                
                await self._optimize_cache()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Cache optimization failed: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def _optimize_cache(self) -> None:
        """Perform cache optimization"""
        try:
            optimization_start = time.time()
            
            # Calculate cache efficiency
            efficiency = self.get_cache_efficiency()
            
            # Optimize if efficiency is low
            if efficiency < 0.8:  # 80% threshold
                await self._rebalance_cache()
            
            # Update optimization timestamp
            self._last_optimization = time.time()
            
            optimization_time = time.time() - optimization_start
            logger.debug(f"🔧 Cache optimization completed in {optimization_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Cache optimization failed: {e}")
    
    async def _rebalance_cache(self) -> None:
        """Rebalance cache levels based on access patterns"""
        try:
            # Find frequently accessed L2 entries to promote
            l2_candidates = []
            
            for key, entry in self._l2_cache.items():
                if entry.access_count > 5:  # Promote frequently accessed
                    l2_candidates.append((key, entry))
            
            # Sort by access frequency
            l2_candidates.sort(key=lambda x: x[1].access_count, reverse=True)
            
            # Promote top candidates to L1
            promoted_count = 0
            for key, entry in l2_candidates[:10]:  # Promote top 10
                if len(self._l1_cache) < 1000:  # L1 capacity limit
                    await self._promote_to_l1(key, entry)
                    promoted_count += 1
            
            if promoted_count > 0:
                logger.debug(f"🚀 Promoted {promoted_count} entries to L1 cache")
                
        except Exception as e:
            logger.error(f"❌ Cache rebalancing failed: {e}")
    
    def get_cache_efficiency(self) -> float:
        """Calculate cache efficiency score (0-1)"""
        try:
            if self.stats.total_requests == 0:
                return 1.0
            
            hit_rate = self.stats.hits / self.stats.total_requests
            
            # Factor in access time (lower is better)
            time_factor = max(0, 1 - (self.stats.avg_access_time_ms / 100))
            
            # Factor in memory utilization
            memory_utilization = self.stats.total_size_bytes / self.max_size_bytes
            memory_factor = min(1.0, memory_utilization * 2)  # Optimal at 50% usage
            
            # Combined efficiency score
            efficiency = (hit_rate * 0.6 + time_factor * 0.3 + memory_factor * 0.1)
            return min(1.0, efficiency)
            
        except Exception:
            return 0.5  # Default efficiency
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            self.stats.update_hit_rate()
            
            return {
                "performance": {
                    "hit_rate_percent": round(self.stats.hit_rate_percent, 2),
                    "total_requests": self.stats.total_requests,
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "avg_access_time_ms": round(self.stats.avg_access_time_ms, 3)
                },
                "capacity": {
                    "entry_count": self.stats.entry_count,
                    "total_size_mb": round(self.stats.memory_usage_mb, 2),
                    "max_size_mb": self.max_size_bytes / (1024 * 1024),
                    "utilization_percent": round(
                        (self.stats.total_size_bytes / self.max_size_bytes) * 100, 2
                    )
                },
                "eviction": {
                    "evictions": self.stats.evictions,
                    "strategy": self.strategy.value
                },
                "levels": {
                    "l1_entries": len(self._l1_cache),
                    "l2_entries": len(self._l2_cache)
                },
                "optimization": {
                    "last_optimization": self._last_optimization,
                    "efficiency_score": round(self.get_cache_efficiency(), 3),
                    "enabled": self._optimization_enabled
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get cache statistics: {e}")
            return {"error": str(e)}
    
    async def optimize(self) -> None:
        """Manual cache optimization trigger"""
        await self._optimize_cache()
    
    def is_healthy(self) -> bool:
        """Check if cache is healthy"""
        try:
            # Check basic functionality
            if self.stats.hit_rate_percent < 10 and self.stats.total_requests > 100:
                return False  # Very low hit rate
            
            if self.stats.avg_access_time_ms > 50:
                return False  # High access time
            
            return True
            
        except Exception:
            return False


# Global cache manager instance
cache = NetflixCacheManager(
    max_size_mb=512,
    default_ttl=3600,
    strategy=CacheStrategy.ADAPTIVE,
    enable_compression=True
)


# Convenience functions
async def get(key: Union[str, dict, tuple], default: Any = None) -> Any:
    """Get value from global cache"""
    return await cache.get(key, default)


async def set(key: Union[str, dict, tuple], value: Any, ttl: Optional[float] = None) -> bool:
    """Set value in global cache"""
    return await cache.set(key, value, ttl)


async def delete(key: Union[str, dict, tuple]) -> bool:
    """Delete value from global cache"""
    return await cache.delete(key)


async def clear() -> None:
    """Clear global cache"""
    await cache.clear_all()


def stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    return cache.get_statistics()
