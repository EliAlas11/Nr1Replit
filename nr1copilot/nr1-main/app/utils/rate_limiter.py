
"""
Netflix-Level Rate Limiting
"""

import time
from typing import Tuple, Optional
from redis.asyncio import Redis

class RateLimiter:
    """Netflix-level rate limiting with Redis backend"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def check_rate_limit(self, key: str, limit: int, window: int) -> Tuple[bool, int]:
        """
        Check if request is within rate limit
        
        Args:
            key: Unique identifier for the rate limit (e.g., IP address, user ID)
            limit: Maximum number of requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        try:
            current_time = int(time.time())
            pipe = self.redis.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(f"rate_limit:{key}", 0, current_time - window)
            
            # Count current requests
            pipe.zcard(f"rate_limit:{key}")
            
            # Add current request
            pipe.zadd(f"rate_limit:{key}", {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(f"rate_limit:{key}", window)
            
            results = await pipe.execute()
            current_requests = results[1]
            
            if current_requests < limit:
                return True, limit - current_requests - 1
            else:
                # Remove the request we just added since it's over the limit
                await self.redis.zrem(f"rate_limit:{key}", str(current_time))
                return False, 0
                
        except Exception as e:
            # In case of Redis failure, allow the request
            return True, limit
    
    async def get_rate_limit_info(self, key: str, window: int) -> dict:
        """Get current rate limit information"""
        try:
            current_time = int(time.time())
            
            # Clean expired entries
            await self.redis.zremrangebyscore(f"rate_limit:{key}", 0, current_time - window)
            
            # Get current count
            current_requests = await self.redis.zcard(f"rate_limit:{key}")
            
            # Get oldest request time
            oldest = await self.redis.zrange(f"rate_limit:{key}", 0, 0, withscores=True)
            
            reset_time = None
            if oldest:
                reset_time = int(oldest[0][1]) + window
            
            return {
                "current_requests": current_requests,
                "reset_time": reset_time,
                "reset_in_seconds": max(0, reset_time - current_time) if reset_time else 0
            }
            
        except Exception:
            return {"current_requests": 0, "reset_time": None, "reset_in_seconds": 0}
