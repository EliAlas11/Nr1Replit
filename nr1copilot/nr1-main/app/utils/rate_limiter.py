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