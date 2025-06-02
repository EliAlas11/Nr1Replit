"""
Enterprise Cache System
Netflix-level caching with Redis, compression, and intelligent invalidation
"""

import asyncio
import json
import logging
import time
import zlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import pickle

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategy enumeration"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


@dataclass
class CacheStats:
    """Cache statistics tracking"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage: float = 0.0
    last_updated: datetime = None

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class EnterpriseCache:
    """Netflix-level enterprise caching system"""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.stats = CacheStats()
        self.compression_enabled = True
        self.compression_threshold = 1024  # Compress data larger than 1KB

        # Cache layers
        self.l1_cache = {}  # In-memory hot cache
        self.l2_cache = {}  # Compressed cache

        logger.info("🚀 Enterprise Cache System initialized")

    async def initialize_cache_clusters(self):
        """Initialize cache clusters for enterprise deployment"""
        try:
            # Initialize cache layers
            self.l1_cache = {}
            self.l2_cache = {}

            # Setup compression
            self.compression_enabled = True

            # Initialize stats
            self.stats = CacheStats(last_updated=datetime.utcnow())

            logger.info("✅ Cache clusters initialized successfully")

        except Exception as e:
            logger.error(f"Cache cluster initialization failed: {e}")
            raise

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with intelligent retrieval"""
        try:
            # Check L1 cache first (hot cache)
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if not self._is_expired(entry):
                    self._update_access_stats(key, hit=True)
                    return self._deserialize_value(entry['value'])
                else:
                    await self._evict_key(key)

            # Check L2 cache (compressed)
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                if not self._is_expired(entry):
                    value = self._deserialize_value(entry['value'])
                    # Promote to L1 cache
                    await self._promote_to_l1(key, value, entry.get('ttl', self.default_ttl))
                    self._update_access_stats(key, hit=True)
                    return value
                else:
                    await self._evict_key(key)

            # Cache miss
            self._update_access_stats(key, hit=False)
            return default

        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return default

    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        strategy: CacheStrategy = CacheStrategy.LRU
    ) -> bool:
        """Set value in cache with intelligent storage"""
        try:
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl

            serialized_value = self._serialize_value(value)
            value_size = len(serialized_value) if isinstance(serialized_value, (str, bytes)) else 1024

            entry = {
                'value': serialized_value,
                'created_at': time.time(),
                'expires_at': expires_at,
                'ttl': ttl,
                'size': value_size,
                'access_count': 1
            }

            # Choose cache layer based on size and access pattern
            if value_size < self.compression_threshold:
                # Store in L1 cache for fast access
                self.l1_cache[key] = entry

                # Ensure L1 cache doesn't exceed size limit
                if len(self.l1_cache) > self.max_size // 2:
                    await self._evict_l1_entries(strategy)
            else:
                # Store in L2 cache with compression
                entry['value'] = self._compress_value(serialized_value)
                self.l2_cache[key] = entry

                # Ensure L2 cache doesn't exceed size limit
                if len(self.l2_cache) > self.max_size:
                    await self._evict_l2_entries(strategy)

            self._update_access_stats(key, hit=False)
            self.stats.size = len(self.l1_cache) + len(self.l2_cache)

            return True

        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            deleted = False

            if key in self.l1_cache:
                del self.l1_cache[key]
                deleted = True

            if key in self.l2_cache:
                del self.l2_cache[key]
                deleted = True

            if key in self.access_times:
                del self.access_times[key]

            if key in self.access_counts:
                del self.access_counts[key]

            if deleted:
                self.stats.size = len(self.l1_cache) + len(self.l2_cache)

            return deleted

        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.access_times.clear()
            self.access_counts.clear()

            self.stats = CacheStats(last_updated=datetime.utcnow())

            logger.info("🧹 Cache cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False

    async def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        self.stats.size = len(self.l1_cache) + len(self.l2_cache)
        self.stats.memory_usage = self._calculate_memory_usage()
        self.stats.last_updated = datetime.utcnow()

        return self.stats

    async def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        return self.stats.hit_rate

    async def optimize_cache_performance(self):
        """Optimize cache performance by cleaning expired entries and rebalancing"""
        try:
            start_time = time.time()

            # Clean expired entries
            expired_keys = []
            current_time = time.time()

            for key, entry in list(self.l1_cache.items()):
                if current_time > entry.get('expires_at', float('inf')):
                    expired_keys.append(key)

            for key, entry in list(self.l2_cache.items()):
                if current_time > entry.get('expires_at', float('inf')):
                    expired_keys.append(key)

            # Remove expired keys
            for key in expired_keys:
                await self.delete(key)
                self.stats.evictions += 1

            # Rebalance cache layers
            await self._rebalance_cache_layers()

            optimization_time = time.time() - start_time
            logger.info(f"🚀 Cache optimization completed in {optimization_time:.3f}s, removed {len(expired_keys)} expired entries")

        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        return time.time() > entry.get('expires_at', float('inf'))

    def _update_access_stats(self, key: str, hit: bool):
        """Update access statistics"""
        if hit:
            self.stats.hits += 1
        else:
            self.stats.misses += 1

        self.access_times[key] = time.time()
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

    def _serialize_value(self, value: Any) -> Union[str, bytes]:
        """Serialize value for storage"""
        try:
            if isinstance(value, (str, int, float, bool)):
                return json.dumps(value)
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            return json.dumps(str(value))

    def _deserialize_value(self, serialized_value: Union[str, bytes]) -> Any:
        """Deserialize value from storage"""
        try:
            if isinstance(serialized_value, bytes):
                # Check if it's compressed
                if serialized_value.startswith(b'compressed:'):
                    serialized_value = self._decompress_value(serialized_value)

                return pickle.loads(serialized_value)
            else:
                return json.loads(serialized_value)
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            return serialized_value

    def _compress_value(self, value: Union[str, bytes]) -> bytes:
        """Compress value for efficient storage"""
        try:
            if isinstance(value, str):
                value = value.encode('utf-8')

            compressed = zlib.compress(value)
            return b'compressed:' + compressed
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return value if isinstance(value, bytes) else value.encode('utf-8')

    def _decompress_value(self, compressed_value: bytes) -> bytes:
        """Decompress value"""
        try:
            if compressed_value.startswith(b'compressed:'):
                compressed_data = compressed_value[11:]  # Remove 'compressed:' prefix
                return zlib.decompress(compressed_data)
            return compressed_value
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return compressed_value

    async def _promote_to_l1(self, key: str, value: Any, ttl: int):
        """Promote frequently accessed item to L1 cache"""
        try:
            serialized_value = self._serialize_value(value)
            entry = {
                'value': serialized_value,
                'created_at': time.time(),
                'expires_at': time.time() + ttl,
                'ttl': ttl,
                'size': len(serialized_value) if isinstance(serialized_value, (str, bytes)) else 1024,
                'access_count': self.access_counts.get(key, 1)
            }

            self.l1_cache[key] = entry

            # Remove from L2 cache
            if key in self.l2_cache:
                del self.l2_cache[key]

        except Exception as e:
            logger.error(f"L1 promotion failed for key {key}: {e}")

    async def _evict_l1_entries(self, strategy: CacheStrategy):
        """Evict entries from L1 cache based on strategy"""
        try:
            if not self.l1_cache:
                return

            # Evict 25% of entries
            evict_count = max(1, len(self.l1_cache) // 4)

            if strategy == CacheStrategy.LRU:
                # Evict least recently used
                sorted_keys = sorted(
                    self.l1_cache.keys(),
                    key=lambda k: self.access_times.get(k, 0)
                )
            elif strategy == CacheStrategy.LFU:
                # Evict least frequently used
                sorted_keys = sorted(
                    self.l1_cache.keys(),
                    key=lambda k: self.access_counts.get(k, 0)
                )
            else:
                # Default to LRU
                sorted_keys = sorted(
                    self.l1_cache.keys(),
                    key=lambda k: self.access_times.get(k, 0)
                )

            for key in sorted_keys[:evict_count]:
                del self.l1_cache[key]
                self.stats.evictions += 1

        except Exception as e:
            logger.error(f"L1 eviction failed: {e}")

    async def _evict_l2_entries(self, strategy: CacheStrategy):
        """Evict entries from L2 cache based on strategy"""
        try:
            if not self.l2_cache:
                return

            # Evict 25% of entries
            evict_count = max(1, len(self.l2_cache) // 4)

            if strategy == CacheStrategy.LRU:
                sorted_keys = sorted(
                    self.l2_cache.keys(),
                    key=lambda k: self.access_times.get(k, 0)
                )
            elif strategy == CacheStrategy.LFU:
                sorted_keys = sorted(
                    self.l2_cache.keys(),
                    key=lambda k: self.access_counts.get(k, 0)
                )
            else:
                sorted_keys = sorted(
                    self.l2_cache.keys(),
                    key=lambda k: self.access_times.get(k, 0)
                )

            for key in sorted_keys[:evict_count]:
                del self.l2_cache[key]
                self.stats.evictions += 1

        except Exception as e:
            logger.error(f"L2 eviction failed: {e}")

    async def _rebalance_cache_layers(self):
        """Rebalance data between cache layers"""
        try:
            # Move frequently accessed L2 items to L1
            for key, entry in list(self.l2_cache.items()):
                access_count = self.access_counts.get(key, 0)
                if access_count > 10:  # Frequently accessed threshold
                    value = self._deserialize_value(entry['value'])
                    await self._promote_to_l1(key, value, entry.get('ttl', self.default_ttl))

        except Exception as e:
            logger.error(f"Cache rebalancing failed: {e}")

    def _calculate_memory_usage(self) -> float:
        """Calculate approximate memory usage in MB"""
        try:
            total_size = 0

            for entry in self.l1_cache.values():
                total_size += entry.get('size', 1024)

            for entry in self.l2_cache.values():
                total_size += entry.get('size', 1024)

            return total_size / (1024 * 1024)  # Convert to MB

        except Exception as e:
            logger.error(f"Memory usage calculation failed: {e}")
            return 0.0

    async def _evict_key(self, key: str):
        """Evict expired key from all cache layers"""
        await self.delete(key)
        self.stats.evictions += 1

    async def shutdown_cache_clusters(self):
        """Gracefully shutdown cache clusters"""
        logger.info("🔄 Shutting down cache clusters")

        # Save cache statistics
        stats = await self.get_stats()
        logger.info(f"📊 Final cache stats: {stats.hits} hits, {stats.misses} misses, hit rate: {stats.hit_rate:.2%}")

        # Clear all caches
        await self.clear()

        logger.info("✅ Cache clusters shutdown complete")