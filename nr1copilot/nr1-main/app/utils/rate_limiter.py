
"""
Netflix-Level Rate Limiter
Production-grade rate limiting with multiple strategies
"""

import time
import asyncio
from typing import Dict, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class RateLimit:
    requests: int
    window: int  # seconds
    burst: Optional[int] = None

class RateLimiter:
    """Advanced rate limiter with multiple algorithms"""
    
    def __init__(self):
        # Rate limit configurations
        self.limits = {
            "upload": RateLimit(requests=10, window=300, burst=3),  # 10 uploads per 5min
            "processing": RateLimit(requests=20, window=3600, burst=5),  # 20 processing per hour
            "download": RateLimit(requests=100, window=300, burst=10),  # 100 downloads per 5min
            "websocket": RateLimit(requests=1000, window=3600, burst=50)  # 1000 WS messages per hour
        }
        
        # Sliding window counters
        self.sliding_windows: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        
        # Token bucket for burst handling
        self.token_buckets: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
        
        # Client tracking
        self.client_stats: Dict[str, Dict] = defaultdict(dict)
        
    async def check_rate_limit(self, operation: str, client_id: str) -> bool:
        """Check if client can perform operation"""
        
        if operation not in self.limits:
            return True
            
        limit = self.limits[operation]
        current_time = time.time()
        
        # Update sliding window
        await self._update_sliding_window(operation, client_id, current_time, limit)
        
        # Check rate limit
        window_requests = len(self.sliding_windows[operation][client_id])
        
        if window_requests >= limit.requests:
            # Check if burst is available
            if limit.burst and await self._check_burst_capacity(operation, client_id, limit):
                await self._consume_burst_token(operation, client_id)
                return True
            return False
            
        return True
    
    async def _update_sliding_window(self, operation: str, client_id: str, current_time: float, limit: RateLimit):
        """Update sliding window for client"""
        window = self.sliding_windows[operation][client_id]
        
        # Remove old entries outside the window
        while window and window[0] <= current_time - limit.window:
            window.popleft()
            
        # Add current request
        window.append(current_time)
    
    async def _check_burst_capacity(self, operation: str, client_id: str, limit: RateLimit) -> bool:
        """Check if burst tokens are available"""
        bucket_key = f"{operation}:{client_id}"
        bucket = self.token_buckets[operation][client_id]
        
        current_time = time.time()
        
        # Initialize bucket if not exists
        if not bucket:
            bucket.update({
                "tokens": limit.burst or 0,
                "last_refill": current_time,
                "capacity": limit.burst or 0
            })
        
        # Refill tokens based on time passed
        time_passed = current_time - bucket["last_refill"]
        refill_rate = (limit.burst or 0) / limit.window  # tokens per second
        
        tokens_to_add = int(time_passed * refill_rate)
        bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = current_time
        
        return bucket["tokens"] > 0
    
    async def _consume_burst_token(self, operation: str, client_id: str):
        """Consume a burst token"""
        bucket = self.token_buckets[operation][client_id]
        if bucket["tokens"] > 0:
            bucket["tokens"] -= 1
    
    async def get_rate_limit_status(self, operation: str, client_id: str) -> Dict[str, Any]:
        """Get current rate limit status for client"""
        if operation not in self.limits:
            return {"error": "Unknown operation"}
            
        limit = self.limits[operation]
        current_time = time.time()
        
        # Count current window requests
        window = self.sliding_windows[operation][client_id]
        
        # Remove expired entries
        while window and window[0] <= current_time - limit.window:
            window.popleft()
            
        window_requests = len(window)
        remaining = max(0, limit.requests - window_requests)
        
        # Get burst status
        bucket = self.token_buckets[operation][client_id]
        burst_tokens = bucket.get("tokens", limit.burst or 0) if bucket else (limit.burst or 0)
        
        reset_time = int(current_time + limit.window)
        
        return {
            "operation": operation,
            "limit": limit.requests,
            "remaining": remaining,
            "window_seconds": limit.window,
            "reset_time": reset_time,
            "burst_available": burst_tokens,
            "current_usage": window_requests
        }
    
    async def reset_client_limits(self, client_id: str):
        """Reset all limits for a client (admin function)"""
        for operation in self.sliding_windows:
            if client_id in self.sliding_windows[operation]:
                del self.sliding_windows[operation][client_id]
                
        for operation in self.token_buckets:
            if client_id in self.token_buckets[operation]:
                del self.token_buckets[operation][client_id]
                
        if client_id in self.client_stats:
            del self.client_stats[client_id]
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics"""
        stats = {
            "total_clients": set(),
            "operations": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for operation in self.sliding_windows:
            operation_stats = {
                "active_clients": len(self.sliding_windows[operation]),
                "total_requests": 0,
                "limit_config": {
                    "requests": self.limits[operation].requests,
                    "window": self.limits[operation].window,
                    "burst": self.limits[operation].burst
                }
            }
            
            # Count total requests in current windows
            current_time = time.time()
            for client_id, window in self.sliding_windows[operation].items():
                stats["total_clients"].add(client_id)
                
                # Count valid requests in window
                valid_requests = sum(1 for req_time in window 
                                   if req_time > current_time - self.limits[operation].window)
                operation_stats["total_requests"] += valid_requests
            
            stats["operations"][operation] = operation_stats
        
        stats["total_clients"] = len(stats["total_clients"])
        return stats
