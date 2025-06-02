
"""
Netflix-Level Rate Limiting System
Advanced rate limiting with sliding windows, distributed limits, and adaptive throttling
"""

import asyncio
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict, deque
import json

from ..config import settings
from ..logging_config import LoggerMixin
from .cache import cache_manager


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


class RateLimitScope(Enum):
    """Rate limit scope"""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    CUSTOM = "custom"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    name: str
    scope: RateLimitScope
    strategy: RateLimitStrategy
    limit: int
    window_seconds: int
    burst_limit: Optional[int] = None
    burst_window_seconds: Optional[int] = None
    enabled: bool = True
    priority: int = 0
    
    def __post_init__(self):
        if self.burst_limit is None:
            self.burst_limit = self.limit * 2
        if self.burst_window_seconds is None:
            self.burst_window_seconds = min(60, self.window_seconds // 4)


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[int] = None
    rule_name: Optional[str] = None
    current_usage: int = 0
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers"""
        headers = {
            "X-RateLimit-Limit": str(self.remaining + self.current_usage),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_time))
        }
        
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
        
        if self.rule_name:
            headers["X-RateLimit-Rule"] = self.rule_name
        
        return headers


class SlidingWindowCounter:
    """Sliding window rate limit counter"""
    
    def __init__(self, window_seconds: int, precision_buckets: int = 10):
        self.window_seconds = window_seconds
        self.precision_buckets = precision_buckets
        self.bucket_size = window_seconds / precision_buckets
        self.buckets = deque(maxlen=precision_buckets)
        self.last_update = time.time()
    
    def _current_bucket(self) -> int:
        """Get current bucket index"""
        return int(time.time() // self.bucket_size)
    
    def _cleanup_old_buckets(self):
        """Remove old buckets outside the window"""
        current_time = time.time()
        current_bucket = self._current_bucket()
        
        # Remove buckets older than the window
        while (self.buckets and 
               current_time - self.buckets[0]["timestamp"] > self.window_seconds):
            self.buckets.popleft()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter and return current total"""
        current_bucket = self._current_bucket()
        current_time = time.time()
        
        self._cleanup_old_buckets()
        
        # Find or create current bucket
        if not self.buckets or self.buckets[-1]["bucket"] != current_bucket:
            self.buckets.append({
                "bucket": current_bucket,
                "count": 0,
                "timestamp": current_time
            })
        
        self.buckets[-1]["count"] += amount
        return self.get_count()
    
    def get_count(self) -> int:
        """Get current count in the sliding window"""
        self._cleanup_old_buckets()
        return sum(bucket["count"] for bucket in self.buckets)


class TokenBucket:
    """Token bucket rate limiter"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket"""
        async with self.lock:
            current_time = time.time()
            
            # Refill bucket
            elapsed = current_time - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = current_time
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_tokens(self) -> float:
        """Get current token count"""
        current_time = time.time()
        elapsed = current_time - self.last_refill
        return min(self.capacity, self.tokens + elapsed * self.refill_rate)


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load"""
    
    def __init__(self, base_limit: int, min_limit: int = None, max_limit: int = None):
        self.base_limit = base_limit
        self.min_limit = min_limit or base_limit // 4
        self.max_limit = max_limit or base_limit * 2
        self.current_limit = base_limit
        
        self.error_rate_window = deque(maxlen=100)
        self.response_time_window = deque(maxlen=100)
        self.last_adjustment = time.time()
        self.adjustment_interval = 60  # seconds
    
    def record_request(self, response_time: float, is_error: bool):
        """Record request metrics for adaptation"""
        current_time = time.time()
        
        self.response_time_window.append({
            "time": current_time,
            "response_time": response_time,
            "is_error": is_error
        })
        
        if is_error:
            self.error_rate_window.append(current_time)
    
    def _calculate_adaptive_limit(self) -> int:
        """Calculate adaptive limit based on system metrics"""
        current_time = time.time()
        
        # Clean old data
        while (self.error_rate_window and 
               current_time - self.error_rate_window[0] > 300):  # 5 minutes
            self.error_rate_window.popleft()
        
        # Calculate error rate
        recent_errors = len([t for t in self.error_rate_window 
                           if current_time - t < 60])  # Last minute
        error_rate = recent_errors / 60 if recent_errors > 0 else 0
        
        # Calculate average response time
        recent_requests = [r for r in self.response_time_window 
                          if current_time - r["time"] < 60]
        avg_response_time = (sum(r["response_time"] for r in recent_requests) / 
                           len(recent_requests)) if recent_requests else 0
        
        # Adjust limit based on metrics
        adjustment_factor = 1.0
        
        # Reduce limit if high error rate
        if error_rate > 0.1:  # 10% error rate
            adjustment_factor *= 0.7
        elif error_rate > 0.05:  # 5% error rate
            adjustment_factor *= 0.85
        
        # Reduce limit if high response time
        if avg_response_time > 2.0:  # 2 seconds
            adjustment_factor *= 0.8
        elif avg_response_time > 1.0:  # 1 second
            adjustment_factor *= 0.9
        
        # Increase limit if system is healthy
        if error_rate < 0.01 and avg_response_time < 0.5:
            adjustment_factor *= 1.1
        
        new_limit = int(self.base_limit * adjustment_factor)
        return max(self.min_limit, min(self.max_limit, new_limit))
    
    def get_current_limit(self) -> int:
        """Get current adaptive limit"""
        current_time = time.time()
        
        if current_time - self.last_adjustment > self.adjustment_interval:
            self.current_limit = self._calculate_adaptive_limit()
            self.last_adjustment = current_time
        
        return self.current_limit


class NetflixLevelRateLimiter(LoggerMixin):
    """Netflix-level rate limiter with multiple strategies and scopes"""
    
    def __init__(self):
        self.rules: List[RateLimitRule] = []
        self.counters: Dict[str, Union[SlidingWindowCounter, TokenBucket]] = {}
        self.adaptive_limiters: Dict[str, AdaptiveRateLimiter] = {}
        self.lock = asyncio.Lock()
        
        # Default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default rate limiting rules"""
        self.rules = [
            RateLimitRule(
                name="global_requests",
                scope=RateLimitScope.GLOBAL,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                limit=settings.security.rate_limit_requests,
                window_seconds=settings.security.rate_limit_window,
                priority=1
            ),
            RateLimitRule(
                name="per_ip_requests",
                scope=RateLimitScope.IP,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                limit=100,
                window_seconds=3600,  # 1 hour
                burst_limit=200,
                burst_window_seconds=60,
                priority=2
            ),
            RateLimitRule(
                name="upload_requests",
                scope=RateLimitScope.ENDPOINT,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                limit=10,
                window_seconds=60,
                priority=3
            ),
            RateLimitRule(
                name="ai_analysis",
                scope=RateLimitScope.USER,
                strategy=RateLimitStrategy.ADAPTIVE,
                limit=50,
                window_seconds=3600,
                priority=4
            )
        ]
    
    def add_rule(self, rule: RateLimitRule):
        """Add a new rate limiting rule"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rate limiting rule"""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                return True
        return False
    
    async def is_allowed(
        self,
        identifier: str,
        scope: RateLimitScope = RateLimitScope.IP,
        endpoint: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> RateLimitResult:
        """Check if request is allowed"""
        
        # Find applicable rules
        applicable_rules = [
            rule for rule in self.rules
            if rule.enabled and (
                rule.scope == scope or
                rule.scope == RateLimitScope.GLOBAL or
                (rule.scope == RateLimitScope.ENDPOINT and endpoint)
            )
        ]
        
        # Check each rule
        for rule in applicable_rules:
            result = await self._check_rule(rule, identifier, endpoint, user_id)
            if not result.allowed:
                return result
        
        # All rules passed
        return RateLimitResult(
            allowed=True,
            remaining=0,  # Will be updated by the most restrictive rule
            reset_time=time.time() + 3600,
            rule_name="allowed"
        )
    
    async def _check_rule(
        self,
        rule: RateLimitRule,
        identifier: str,
        endpoint: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> RateLimitResult:
        """Check a specific rate limiting rule"""
        
        # Generate cache key
        key_parts = [rule.name]
        if rule.scope == RateLimitScope.GLOBAL:
            key_parts.append("global")
        elif rule.scope == RateLimitScope.IP:
            key_parts.append(f"ip:{identifier}")
        elif rule.scope == RateLimitScope.USER and user_id:
            key_parts.append(f"user:{user_id}")
        elif rule.scope == RateLimitScope.ENDPOINT and endpoint:
            key_parts.append(f"endpoint:{endpoint}")
        else:
            key_parts.append(f"custom:{identifier}")
        
        cache_key = ":".join(key_parts)
        
        async with self.lock:
            if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window(rule, cache_key)
            elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket(rule, cache_key)
            elif rule.strategy == RateLimitStrategy.ADAPTIVE:
                return await self._check_adaptive(rule, cache_key)
            else:
                return await self._check_fixed_window(rule, cache_key)
    
    async def _check_sliding_window(self, rule: RateLimitRule, cache_key: str) -> RateLimitResult:
        """Check sliding window rate limit"""
        if cache_key not in self.counters:
            self.counters[cache_key] = SlidingWindowCounter(rule.window_seconds)
        
        counter = self.counters[cache_key]
        current_count = counter.get_count()
        
        if current_count >= rule.limit:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=time.time() + rule.window_seconds,
                retry_after=rule.window_seconds,
                rule_name=rule.name,
                current_usage=current_count
            )
        
        # Increment counter
        new_count = counter.increment()
        
        return RateLimitResult(
            allowed=True,
            remaining=rule.limit - new_count,
            reset_time=time.time() + rule.window_seconds,
            rule_name=rule.name,
            current_usage=new_count
        )
    
    async def _check_token_bucket(self, rule: RateLimitRule, cache_key: str) -> RateLimitResult:
        """Check token bucket rate limit"""
        if cache_key not in self.counters:
            refill_rate = rule.limit / rule.window_seconds
            self.counters[cache_key] = TokenBucket(rule.limit, refill_rate)
        
        bucket = self.counters[cache_key]
        
        if await bucket.consume():
            remaining_tokens = int(bucket.get_tokens())
            return RateLimitResult(
                allowed=True,
                remaining=remaining_tokens,
                reset_time=time.time() + rule.window_seconds,
                rule_name=rule.name,
                current_usage=rule.limit - remaining_tokens
            )
        else:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=time.time() + (1 / (rule.limit / rule.window_seconds)),
                retry_after=int(1 / (rule.limit / rule.window_seconds)) + 1,
                rule_name=rule.name,
                current_usage=rule.limit
            )
    
    async def _check_adaptive(self, rule: RateLimitRule, cache_key: str) -> RateLimitResult:
        """Check adaptive rate limit"""
        if cache_key not in self.adaptive_limiters:
            self.adaptive_limiters[cache_key] = AdaptiveRateLimiter(rule.limit)
        
        adaptive_limiter = self.adaptive_limiters[cache_key]
        current_limit = adaptive_limiter.get_current_limit()
        
        # Use sliding window with adaptive limit
        temp_rule = RateLimitRule(
            name=rule.name,
            scope=rule.scope,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            limit=current_limit,
            window_seconds=rule.window_seconds
        )
        
        return await self._check_sliding_window(temp_rule, cache_key)
    
    async def _check_fixed_window(self, rule: RateLimitRule, cache_key: str) -> RateLimitResult:
        """Check fixed window rate limit"""
        current_time = time.time()
        window_start = int(current_time // rule.window_seconds) * rule.window_seconds
        
        # Get current count from cache
        window_key = f"{cache_key}:{window_start}"
        current_count = await cache_manager.get(window_key, 0)
        
        if current_count >= rule.limit:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=window_start + rule.window_seconds,
                retry_after=int(window_start + rule.window_seconds - current_time),
                rule_name=rule.name,
                current_usage=current_count
            )
        
        # Increment counter
        new_count = current_count + 1
        await cache_manager.set(window_key, new_count, rule.window_seconds)
        
        return RateLimitResult(
            allowed=True,
            remaining=rule.limit - new_count,
            reset_time=window_start + rule.window_seconds,
            rule_name=rule.name,
            current_usage=new_count
        )
    
    async def record_request_metrics(
        self,
        identifier: str,
        response_time: float,
        is_error: bool,
        scope: RateLimitScope = RateLimitScope.IP
    ):
        """Record request metrics for adaptive rate limiting"""
        for cache_key, limiter in self.adaptive_limiters.items():
            if identifier in cache_key:
                limiter.record_request(response_time, is_error)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        stats = {
            "rules": [
                {
                    "name": rule.name,
                    "scope": rule.scope.value,
                    "strategy": rule.strategy.value,
                    "limit": rule.limit,
                    "window_seconds": rule.window_seconds,
                    "enabled": rule.enabled
                }
                for rule in self.rules
            ],
            "active_counters": len(self.counters),
            "adaptive_limiters": len(self.adaptive_limiters)
        }
        
        # Add adaptive limiter details
        adaptive_stats = {}
        for key, limiter in self.adaptive_limiters.items():
            adaptive_stats[key] = {
                "current_limit": limiter.get_current_limit(),
                "base_limit": limiter.base_limit,
                "min_limit": limiter.min_limit,
                "max_limit": limiter.max_limit
            }
        
        stats["adaptive_limits"] = adaptive_stats
        return stats
    
    async def reset_limits(self, identifier: Optional[str] = None):
        """Reset rate limits for identifier or all"""
        async with self.lock:
            if identifier:
                # Reset specific identifier
                keys_to_remove = [key for key in self.counters.keys() if identifier in key]
                for key in keys_to_remove:
                    del self.counters[key]
                
                keys_to_remove = [key for key in self.adaptive_limiters.keys() if identifier in key]
                for key in keys_to_remove:
                    del self.adaptive_limiters[key]
            else:
                # Reset all
                self.counters.clear()
                self.adaptive_limiters.clear()


# Global rate limiter instance
rate_limiter = NetflixLevelRateLimiter()


# Convenience functions
async def check_rate_limit(
    identifier: str,
    scope: RateLimitScope = RateLimitScope.IP,
    endpoint: Optional[str] = None,
    user_id: Optional[str] = None
) -> RateLimitResult:
    """Check rate limit for identifier"""
    return await rate_limiter.is_allowed(identifier, scope, endpoint, user_id)


async def record_request(
    identifier: str,
    response_time: float,
    is_error: bool = False,
    scope: RateLimitScope = RateLimitScope.IP
):
    """Record request metrics"""
    await rate_limiter.record_request_metrics(identifier, response_time, is_error, scope)
