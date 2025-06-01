
"""
Netflix-Level Rate Limiting
Advanced rate limiting with Redis backend
"""

import time
import logging
from typing import Tuple, Dict, Any, Optional
import asyncio

try:
    from redis.asyncio import Redis
except ImportError:
    Redis = None

logger = logging.getLogger(__name__)

class RateLimiter:
    """Netflix-level rate limiting with sliding window algorithm"""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis_client = redis_client
        self.local_limits = {}  # Fallback for when Redis is unavailable
    
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int = 60,
        burst_limit: Optional[int] = None
    ) -> Tuple[bool, int]:
        """
        Check if request is within rate limit
        Returns (is_allowed, remaining_requests)
        """
        try:
            if self.redis_client:
                return await self._redis_rate_limit(key, limit, window, burst_limit)
            else:
                return await self._local_rate_limit(key, limit, window, burst_limit)
                
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Allow request on error (fail open)
            return True, limit
    
    async def _redis_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int,
        burst_limit: Optional[int] = None
    ) -> Tuple[bool, int]:
        """Redis-based sliding window rate limiting"""
        now = time.time()
        pipeline = self.redis_client.pipeline()
        
        # Sliding window key
        window_key = f"rate_limit:{key}:{int(now // window)}"
        
        # Current window count
        pipeline.incr(window_key)
        pipeline.expire(window_key, window)
        
        # Previous window for sliding calculation
        prev_window_key = f"rate_limit:{key}:{int((now - window) // window)}"
        pipeline.get(prev_window_key)
        
        results = await pipeline.execute()
        
        current_count = results[0]
        prev_count = int(results[2]) if results[2] else 0
        
        # Calculate sliding window count
        window_progress = (now % window) / window
        sliding_count = int(prev_count * (1 - window_progress) + current_count)
        
        # Check burst limit if specified
        if burst_limit and current_count > burst_limit:
            return False, 0
        
        # Check normal limit
        if sliding_count <= limit:
            remaining = max(0, limit - sliding_count)
            return True, remaining
        else:
            return False, 0
    
    async def _local_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int,
        burst_limit: Optional[int] = None
    ) -> Tuple[bool, int]:
        """Local memory-based rate limiting (fallback)"""
        now = time.time()
        
        if key not in self.local_limits:
            self.local_limits[key] = []
        
        # Clean old entries
        cutoff = now - window
        self.local_limits[key] = [
            timestamp for timestamp in self.local_limits[key]
            if timestamp > cutoff
        ]
        
        current_count = len(self.local_limits[key])
        
        # Check burst limit
        if burst_limit:
            recent_requests = [
                timestamp for timestamp in self.local_limits[key]
                if timestamp > now - 10  # Last 10 seconds
            ]
            if len(recent_requests) > burst_limit:
                return False, 0
        
        # Check normal limit
        if current_count < limit:
            self.local_limits[key].append(now)
            return True, limit - current_count - 1
        else:
            return False, 0
    
    async def get_rate_limit_info(self, key: str, window: int = 60) -> Dict[str, Any]:
        """Get detailed rate limit information"""
        try:
            now = time.time()
            
            if self.redis_client:
                # Get current and previous window counts
                current_window_key = f"rate_limit:{key}:{int(now // window)}"
                prev_window_key = f"rate_limit:{key}:{int((now - window) // window)}"
                
                pipeline = self.redis_client.pipeline()
                pipeline.get(current_window_key)
                pipeline.get(prev_window_key)
                pipeline.ttl(current_window_key)
                
                results = await pipeline.execute()
                
                current_count = int(results[0]) if results[0] else 0
                prev_count = int(results[1]) if results[1] else 0
                ttl = results[2] if results[2] > 0 else window
                
                # Calculate sliding window
                window_progress = (now % window) / window
                sliding_count = int(prev_count * (1 - window_progress) + current_count)
                
                return {
                    "current_count": current_count,
                    "sliding_count": sliding_count,
                    "window_remaining": ttl,
                    "reset_time": now + ttl,
                    "backend": "redis"
                }
            else:
                # Local fallback
                if key in self.local_limits:
                    cutoff = now - window
                    valid_requests = [
                        timestamp for timestamp in self.local_limits[key]
                        if timestamp > cutoff
                    ]
                    return {
                        "current_count": len(valid_requests),
                        "sliding_count": len(valid_requests),
                        "window_remaining": window,
                        "reset_time": now + window,
                        "backend": "local"
                    }
                else:
                    return {
                        "current_count": 0,
                        "sliding_count": 0,
                        "window_remaining": window,
                        "reset_time": now + window,
                        "backend": "local"
                    }
                    
        except Exception as e:
            logger.error(f"Rate limit info error: {e}")
            return {
                "current_count": 0,
                "sliding_count": 0,
                "window_remaining": window,
                "reset_time": now + window,
                "backend": "error"
            }
    
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for a specific key"""
        try:
            if self.redis_client:
                # Delete all rate limit keys for this identifier
                pattern = f"rate_limit:{key}:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            # Clear local cache
            if key in self.local_limits:
                del self.local_limits[key]
            
            logger.info(f"Rate limit reset for key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Rate limit reset error: {e}")
            return False
    
    async def cleanup_expired(self):
        """Clean up expired local rate limit entries"""
        now = time.time()
        cutoff = now - 3600  # Clean entries older than 1 hour
        
        keys_to_remove = []
        for key, timestamps in self.local_limits.items():
            valid_timestamps = [ts for ts in timestamps if ts > cutoff]
            if valid_timestamps:
                self.local_limits[key] = valid_timestamps
            else:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.local_limits[key]
        
        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} expired rate limit entries")
"""
Rate Limiting Utilities
Netflix-level rate limiting implementation
"""

import time
from typing import Tuple, Optional
from collections import defaultdict

class RateLimiter:
    """Netflix-level rate limiter with sliding window"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_cache = defaultdict(list)
    
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int
    ) -> Tuple[bool, int]:
        """
        Check if request is within rate limit
        Returns (is_allowed, remaining_requests)
        """
        current_time = time.time()
        
        if self.redis_client:
            return await self._check_redis_rate_limit(key, limit, window, current_time)
        else:
            return self._check_local_rate_limit(key, limit, window, current_time)
    
    async def _check_redis_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int, 
        current_time: float
    ) -> Tuple[bool, int]:
        """Redis-based rate limiting"""
        try:
            # Use Redis for distributed rate limiting
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, current_time - window)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(key, window)
            
            results = await pipe.execute()
            
            current_count = results[1]
            
            if current_count < limit:
                return True, limit - current_count - 1
            else:
                # Remove the request we just added since it's over limit
                await self.redis_client.zrem(key, str(current_time))
                return False, 0
                
        except Exception:
            # Fallback to local rate limiting
            return self._check_local_rate_limit(key, limit, window, current_time)
    
    def _check_local_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int, 
        current_time: float
    ) -> Tuple[bool, int]:
        """Local memory-based rate limiting"""
        requests = self.local_cache[key]
        
        # Remove old requests
        cutoff_time = current_time - window
        self.local_cache[key] = [req_time for req_time in requests if req_time > cutoff_time]
        
        current_count = len(self.local_cache[key])
        
        if current_count < limit:
            self.local_cache[key].append(current_time)
            return True, limit - current_count - 1
        else:
            return False, 0
