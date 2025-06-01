"""
Rate Limiting System
Netflix-level rate limiting for API protection
"""

import time
import logging
from typing import Dict, Tuple, Optional
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RateLimiter:
    """Netflix-level rate limiting system"""

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.memory_store = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int,
        identifier: str = "default"
    ) -> Tuple[bool, int]:
        """
        Check if request is within rate limit
        Returns: (is_allowed, remaining_requests)
        """
        try:
            current_time = time.time()
            window_start = current_time - window

            # Clean up old entries periodically
            if current_time - self.last_cleanup > self.cleanup_interval:
                await self._cleanup_expired_entries()
                self.last_cleanup = current_time

            # Try Redis first
            if self.redis_client:
                return await self._check_redis_rate_limit(
                    key, limit, window, current_time
                )

            # Fallback to memory-based rate limiting
            return await self._check_memory_rate_limit(
                key, limit, window, current_time, window_start
            )

        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request on error
            return True, limit

    async def _check_redis_rate_limit(
        self,
        key: str,
        limit: int,
        window: int,
        current_time: float
    ) -> Tuple[bool, int]:
        """Redis-based rate limiting using sliding window"""
        try:
            pipe = self.redis_client.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, current_time - window)

            # Count current requests
            pipe.zcard(key)

            # Add current request
            pipe.zadd(key, {str(current_time): current_time})

            # Set expiration
            pipe.expire(key, window)

            results = await pipe.execute()
            current_count = results[1]

            remaining = max(0, limit - current_count - 1)
            is_allowed = current_count < limit

            return is_allowed, remaining

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fallback to memory-based
            return await self._check_memory_rate_limit(
                key, limit, window, current_time, current_time - window
            )

    async def _check_memory_rate_limit(
        self,
        key: str,
        limit: int,
        window: int,
        current_time: float,
        window_start: float
    ) -> Tuple[bool, int]:
        """Memory-based rate limiting using sliding window"""
        if key not in self.memory_store:
            self.memory_store[key] = []

        # Remove old requests
        self.memory_store[key] = [
            req_time for req_time in self.memory_store[key]
            if req_time > window_start
        ]

        current_count = len(self.memory_store[key])

        if current_count < limit:
            # Allow request
            self.memory_store[key].append(current_time)
            remaining = limit - current_count - 1
            return True, remaining
        else:
            # Deny request
            return False, 0

    async def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for a specific key"""
        try:
            if self.redis_client:
                await self.redis_client.delete(key)

            if key in self.memory_store:
                del self.memory_store[key]

            return True

        except Exception as e:
            logger.error(f"Rate limit reset error: {e}")
            return False

    async def get_rate_limit_info(
        self,
        key: str,
        window: int
    ) -> Dict[str, int]:
        """Get current rate limit information"""
        try:
            current_time = time.time()
            window_start = current_time - window

            if self.redis_client:
                try:
                    count = await self.redis_client.zcount(key, window_start, current_time)
                    return {"current_requests": count, "window_start": window_start}
                except Exception:
                    pass

            # Fallback to memory store
            if key in self.memory_store:
                requests = [
                    req_time for req_time in self.memory_store[key]
                    if req_time > window_start
                ]
                return {"current_requests": len(requests), "window_start": window_start}

            return {"current_requests": 0, "window_start": window_start}

        except Exception as e:
            logger.error(f"Rate limit info error: {e}")
            return {"current_requests": 0, "window_start": current_time - window}

    async def _cleanup_expired_entries(self) -> None:
        """Clean up expired entries from memory store"""
        try:
            current_time = time.time()
            cleanup_threshold = current_time - 3600  # 1 hour

            keys_to_remove = []
            for key, requests in self.memory_store.items():
                # Remove old requests
                self.memory_store[key] = [
                    req_time for req_time in requests
                    if req_time > cleanup_threshold
                ]

                # Mark empty keys for removal
                if not self.memory_store[key]:
                    keys_to_remove.append(key)

            # Remove empty keys
            for key in keys_to_remove:
                del self.memory_store[key]

            logger.info(f"Cleaned up {len(keys_to_remove)} expired rate limit entries")

        except Exception as e:
            logger.error(f"Rate limit cleanup error: {e}")

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics"""
        return {
            "active_keys": len(self.memory_store),
            "total_tracked_requests": sum(
                len(requests) for requests in self.memory_store.values()
            )
        }

class IPRateLimiter(RateLimiter):
    """IP-based rate limiting"""

    def __init__(self, redis_client=None):
        super().__init__(redis_client)
        self.default_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000
        }

    async def check_ip_limits(self, ip: str) -> Tuple[bool, str]:
        """Check all IP-based limits"""
        # Check per-minute limit
        allowed, remaining = await self.check_rate_limit(
            f"ip_minute:{ip}",
            self.default_limits["requests_per_minute"],
            60
        )

        if not allowed:
            return False, "Rate limit exceeded: too many requests per minute"

        # Check per-hour limit
        allowed, remaining = await self.check_rate_limit(
            f"ip_hour:{ip}",
            self.default_limits["requests_per_hour"],
            3600
        )

        if not allowed:
            return False, "Rate limit exceeded: too many requests per hour"

        # Check per-day limit
        allowed, remaining = await self.check_rate_limit(
            f"ip_day:{ip}",
            self.default_limits["requests_per_day"],
            86400
        )

        if not allowed:
            return False, "Rate limit exceeded: too many requests per day"

        return True, "OK"
"""
Netflix-Level Rate Limiting System
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Netflix-level rate limiting with sliding window and advanced features"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_storage = defaultdict(lambda: defaultdict(deque))
        self.blocked_ips = set()
        self.whitelist = set()
        
        # Rate limit configurations
        self.default_limits = {
            "requests_per_minute": 100,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
            "processing_per_hour": 50,
            "upload_per_hour": 20
        }
        
        # Burst protection
        self.burst_limits = {
            "requests_per_second": 10,
            "concurrent_processing": 5
        }
        
        # Cleanup task
        self._cleanup_task = None
        self.start_cleanup_task()
    
    def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_old_entries())
    
    async def _cleanup_old_entries(self):
        """Clean up old rate limit entries"""
        while True:
            try:
                await self._cleanup_local_storage()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int,
        limit_type: str = "requests"
    ) -> Tuple[bool, int]:
        """
        Check if request is within rate limit
        
        Args:
            key: Identifier (IP, user ID, etc.)
            limit: Maximum number of requests
            window: Time window in seconds
            limit_type: Type of rate limit
        
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        
        # Check if IP is blocked
        if key in self.blocked_ips:
            logger.warning(f"Blocked IP attempted request: {key}")
            return False, 0
        
        # Check if IP is whitelisted
        if key in self.whitelist:
            return True, limit
        
        current_time = time.time()
        
        if self.redis_client:
            return await self._check_rate_limit_redis(key, limit, window, current_time, limit_type)
        else:
            return await self._check_rate_limit_local(key, limit, window, current_time, limit_type)
    
    async def _check_rate_limit_redis(
        self, 
        key: str, 
        limit: int, 
        window: int, 
        current_time: float,
        limit_type: str
    ) -> Tuple[bool, int]:
        """Redis-based rate limiting with sliding window"""
        try:
            redis_key = f"rate_limit:{limit_type}:{key}:{window}"
            
            # Use Redis sliding window log
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(redis_key, 0, current_time - window)
            
            # Count current entries
            pipe.zcard(redis_key)
            
            # Add current request
            pipe.zadd(redis_key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(redis_key, window + 1)
            
            results = await pipe.execute()
            current_count = results[1]
            
            if current_count < limit:
                remaining = limit - current_count - 1
                return True, remaining
            else:
                # Remove the request we just added since it's over limit
                await self.redis_client.zrem(redis_key, str(current_time))
                return False, 0
                
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fallback to local rate limiting
            return await self._check_rate_limit_local(key, limit, window, current_time, limit_type)
    
    async def _check_rate_limit_local(
        self, 
        key: str, 
        limit: int, 
        window: int, 
        current_time: float,
        limit_type: str
    ) -> Tuple[bool, int]:
        """Local memory-based rate limiting"""
        
        # Clean old entries first
        cutoff_time = current_time - window
        
        requests = self.local_storage[limit_type][key]
        
        # Remove old entries
        while requests and requests[0] < cutoff_time:
            requests.popleft()
        
        # Check if under limit
        if len(requests) < limit:
            requests.append(current_time)
            remaining = limit - len(requests)
            return True, remaining
        else:
            return False, 0
    
    async def check_burst_protection(self, key: str, action_type: str = "request") -> bool:
        """Check burst protection limits"""
        current_time = time.time()
        
        if action_type == "request":
            # Check requests per second
            return await self.check_rate_limit(
                f"burst:{key}", 
                self.burst_limits["requests_per_second"], 
                1, 
                "burst_requests"
            )
        
        return True, 0
    
    async def check_concurrent_limit(self, key: str, current_count: int) -> bool:
        """Check concurrent processing limits"""
        max_concurrent = self.burst_limits["concurrent_processing"]
        return current_count < max_concurrent
    
    async def record_processing_start(self, key: str, task_id: str):
        """Record the start of a processing task"""
        if self.redis_client:
            try:
                redis_key = f"processing:{key}"
                await self.redis_client.sadd(redis_key, task_id)
                await self.redis_client.expire(redis_key, 3600)  # 1 hour expiration
            except Exception as e:
                logger.error(f"Redis processing record error: {e}")
        else:
            # Use local storage
            if "processing" not in self.local_storage:
                self.local_storage["processing"] = defaultdict(set)
            self.local_storage["processing"][key].add(task_id)
    
    async def record_processing_end(self, key: str, task_id: str):
        """Record the end of a processing task"""
        if self.redis_client:
            try:
                redis_key = f"processing:{key}"
                await self.redis_client.srem(redis_key, task_id)
            except Exception as e:
                logger.error(f"Redis processing end record error: {e}")
        else:
            # Use local storage
            if "processing" in self.local_storage and key in self.local_storage["processing"]:
                self.local_storage["processing"][key].discard(task_id)
    
    async def get_current_processing_count(self, key: str) -> int:
        """Get current processing count for a key"""
        if self.redis_client:
            try:
                redis_key = f"processing:{key}"
                return await self.redis_client.scard(redis_key)
            except Exception as e:
                logger.error(f"Redis processing count error: {e}")
                return 0
        else:
            return len(self.local_storage.get("processing", {}).get(key, set()))
    
    async def block_ip(self, ip: str, duration: Optional[int] = None):
        """Block an IP address"""
        self.blocked_ips.add(ip)
        
        if duration:
            # Schedule unblock
            asyncio.create_task(self._unblock_ip_after_delay(ip, duration))
        
        logger.warning(f"IP blocked: {ip} for {duration or 'indefinite'} seconds")
    
    async def _unblock_ip_after_delay(self, ip: str, delay: int):
        """Unblock IP after delay"""
        await asyncio.sleep(delay)
        self.blocked_ips.discard(ip)
        logger.info(f"IP unblocked: {ip}")
    
    async def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)
        logger.info(f"IP manually unblocked: {ip}")
    
    async def add_to_whitelist(self, ip: str):
        """Add IP to whitelist"""
        self.whitelist.add(ip)
        logger.info(f"IP whitelisted: {ip}")
    
    async def remove_from_whitelist(self, ip: str):
        """Remove IP from whitelist"""
        self.whitelist.discard(ip)
        logger.info(f"IP removed from whitelist: {ip}")
    
    async def get_rate_limit_info(self, key: str) -> Dict[str, Any]:
        """Get rate limit information for a key"""
        info = {
            "key": key,
            "is_blocked": key in self.blocked_ips,
            "is_whitelisted": key in self.whitelist,
            "current_usage": {}
        }
        
        # Check current usage for different limits
        for limit_type, config in self.default_limits.items():
            if "per_minute" in limit_type:
                window = 60
            elif "per_hour" in limit_type:
                window = 3600
            elif "per_day" in limit_type:
                window = 86400
            else:
                window = 60
            
            is_allowed, remaining = await self.check_rate_limit(
                key, config, window, limit_type
            )
            
            info["current_usage"][limit_type] = {
                "limit": config,
                "remaining": remaining,
                "window": window,
                "is_allowed": is_allowed
            }
        
        return info
    
    async def _cleanup_local_storage(self):
        """Clean up old entries from local storage"""
        current_time = time.time()
        
        for limit_type in list(self.local_storage.keys()):
            for key in list(self.local_storage[limit_type].keys()):
                if limit_type == "processing":
                    continue  # Don't clean processing sets automatically
                
                requests = self.local_storage[limit_type][key]
                
                # Clean entries older than 24 hours
                cutoff_time = current_time - 86400
                
                while requests and requests[0] < cutoff_time:
                    requests.popleft()
                
                # Remove empty deques
                if not requests:
                    del self.local_storage[limit_type][key]
            
            # Remove empty limit types
            if not self.local_storage[limit_type]:
                del self.local_storage[limit_type]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        stats = {
            "blocked_ips": len(self.blocked_ips),
            "whitelisted_ips": len(self.whitelist),
            "active_limits": len(self.local_storage),
            "total_tracked_keys": sum(len(limits) for limits in self.local_storage.values()),
            "default_limits": self.default_limits,
            "burst_limits": self.burst_limits
        }
        
        # Add Redis stats if available
        if self.redis_client:
            try:
                stats["redis_connected"] = await self.redis_client.ping()
            except Exception:
                stats["redis_connected"] = False
        else:
            stats["redis_connected"] = False
        
        return stats
