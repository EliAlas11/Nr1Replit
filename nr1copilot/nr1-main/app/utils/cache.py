
"""
Netflix-Level Cache Manager
Production-grade caching with multiple backends and strategies
"""

import json
import time
import asyncio
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import hashlib

class CacheManager:
    """Advanced cache manager with TTL and persistence"""
    
    def __init__(self):
        # In-memory cache
        self.memory_cache: Dict[str, Dict] = {}
        
        # File-based cache for persistence
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.max_memory_items = 1000
        self.cleanup_interval = 300  # 5 minutes
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired())
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cache value with optional TTL"""
        try:
            ttl = ttl or self.default_ttl
            expiry = time.time() + ttl
            
            cache_entry = {
                "value": value,
                "expiry": expiry,
                "created": time.time(),
                "access_count": 0
            }
            
            # Store in memory
            self.memory_cache[key] = cache_entry
            
            # Enforce memory limits
            await self._enforce_memory_limits()
            
            # Persist to disk for important data
            if await self._should_persist(key):
                await self._persist_to_disk(key, cache_entry)
            
            self.stats["size"] = len(self.memory_cache)
            return True
            
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Check if expired
                if time.time() > entry["expiry"]:
                    del self.memory_cache[key]
                    self.stats["misses"] += 1
                    return None
                
                # Update access stats
                entry["access_count"] += 1
                entry["last_access"] = time.time()
                
                self.stats["hits"] += 1
                return entry["value"]
            
            # Try disk cache
            disk_value = await self._load_from_disk(key)
            if disk_value is not None:
                # Load back to memory with shorter TTL
                await self.set(key, disk_value, ttl=1800)  # 30 minutes
                self.stats["hits"] += 1
                return disk_value
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            print(f"Cache get error: {e}")
            self.stats["misses"] += 1
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete cached value"""
        try:
            # Remove from memory
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            # Remove from disk
            disk_file = self.cache_dir / f"{self._hash_key(key)}.cache"
            if disk_file.exists():
                disk_file.unlink()
            
            self.stats["size"] = len(self.memory_cache)
            return True
            
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return await self.get(key) is not None
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self.memory_cache.clear()
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            self.stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "size": 0
            }
            
            return True
            
        except Exception as e:
            print(f"Cache clear error: {e}")
            return False
    
    async def cache_session_data(self, session_id: str, data: Dict[str, Any]):
        """Cache session-specific data"""
        await self.set(f"session:{session_id}", data, ttl=7200)  # 2 hours
    
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data"""
        return await self.get(f"session:{session_id}")
    
    async def cache_results(self, task_id: str, results: List[Dict[str, Any]]):
        """Cache processing results"""
        await self.set(f"results:{task_id}", results, ttl=86400)  # 24 hours
    
    async def get_results(self, task_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached results"""
        return await self.get(f"results:{task_id}")
    
    async def _enforce_memory_limits(self):
        """Enforce memory cache size limits"""
        if len(self.memory_cache) <= self.max_memory_items:
            return
        
        # Remove least recently used items
        items_to_remove = len(self.memory_cache) - self.max_memory_items
        
        # Sort by last access time (oldest first)
        sorted_items = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].get("last_access", x[1]["created"])
        )
        
        for i in range(items_to_remove):
            key = sorted_items[i][0]
            del self.memory_cache[key]
            self.stats["evictions"] += 1
    
    async def _should_persist(self, key: str) -> bool:
        """Determine if key should be persisted to disk"""
        # Persist session data and results
        return key.startswith(("session:", "results:", "metadata:"))
    
    async def _persist_to_disk(self, key: str, entry: Dict):
        """Persist cache entry to disk"""
        try:
            file_path = self.cache_dir / f"{self._hash_key(key)}.cache"
            
            with open(file_path, 'wb') as f:
                pickle.dump({
                    "key": key,
                    "entry": entry
                }, f)
                
        except Exception as e:
            print(f"Disk persist error: {e}")
    
    async def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load cache entry from disk"""
        try:
            file_path = self.cache_dir / f"{self._hash_key(key)}.cache"
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            entry = data["entry"]
            
            # Check if expired
            if time.time() > entry["expiry"]:
                file_path.unlink()
                return None
            
            return entry["value"]
            
        except Exception as e:
            print(f"Disk load error: {e}")
            return None
    
    def _hash_key(self, key: str) -> str:
        """Create hash for disk filename"""
        return hashlib.md5(key.encode()).hexdigest()
    
    async def _cleanup_expired(self):
        """Background task to cleanup expired entries"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = time.time()
                expired_keys = []
                
                # Find expired memory entries
                for key, entry in self.memory_cache.items():
                    if current_time > entry["expiry"]:
                        expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    del self.memory_cache[key]
                
                # Cleanup disk cache
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        if current_time > data["entry"]["expiry"]:
                            cache_file.unlink()
                            
                    except Exception:
                        # Remove corrupted files
                        cache_file.unlink()
                
                self.stats["size"] = len(self.memory_cache)
                
            except Exception as e:
                print(f"Cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0
        total_requests = self.stats["hits"] + self.stats["misses"]
        
        if total_requests > 0:
            hit_rate = (self.stats["hits"] / total_requests) * 100
        
        return {
            **self.stats,
            "hit_rate": round(hit_rate, 2),
            "memory_usage": len(self.memory_cache),
            "max_memory": self.max_memory_items,
            "disk_files": len(list(self.cache_dir.glob("*.cache")))
        }
