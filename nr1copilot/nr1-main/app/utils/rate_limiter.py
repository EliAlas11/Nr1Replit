"""
Basic Rate Limiter
Provides request rate limiting functionality
"""

import time
import logging
from typing import Dict, Tuple
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter implementation"""

    def __init__(self):
        self.requests = defaultdict(deque)
        self.limits = {
            "default": (100, 3600),  # 100 requests per hour
            "upload": (10, 3600),    # 10 uploads per hour
            "processing": (20, 3600) # 20 processing requests per hour
        }

    def is_allowed(self, identifier: str, limit_type: str = "default") -> Tuple[bool, Dict[str, any]]:
        """Check if request is allowed under rate limit"""
        max_requests, window = self.limits.get(limit_type, self.limits["default"])

        now = time.time()
        request_times = self.requests[f"{identifier}:{limit_type}"]

        # Remove old requests outside the window
        while request_times and request_times[0] <= now - window:
            request_times.popleft()

        # Check if under limit
        if len(request_times) < max_requests:
            request_times.append(now)
            return True, {
                "allowed": True,
                "remaining": max_requests - len(request_times),
                "reset_time": now + window
            }
        else:
            return False, {
                "allowed": False,
                "remaining": 0,
                "reset_time": request_times[0] + window
            }

    def get_status(self, identifier: str, limit_type: str = "default") -> Dict[str, any]:
        """Get current rate limit status"""
        max_requests, window = self.limits.get(limit_type, self.limits["default"])
        now = time.time()
        request_times = self.requests[f"{identifier}:{limit_type}"]

        # Remove old requests
        while request_times and request_times[0] <= now - window:
            request_times.popleft()

        return {
            "limit": max_requests,
            "remaining": max_requests - len(request_times),
            "reset_time": now + window,
            "window": window
        }