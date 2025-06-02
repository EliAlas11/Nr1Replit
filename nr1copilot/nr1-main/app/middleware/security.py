
"""
Enterprise Security Middleware
Comprehensive security controls with rate limiting, input validation, and threat protection.
"""

import time
import logging
import hashlib
from typing import Dict, Optional, List, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter with sliding window."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        user_requests = self.requests[identifier]
        while user_requests and user_requests[0] < window_start:
            user_requests.popleft()
        
        # Check if under limit
        if len(user_requests) < self.max_requests:
            user_requests.append(now)
            return True
        
        return False
    
    def get_reset_time(self, identifier: str) -> int:
        """Get time until rate limit resets."""
        user_requests = self.requests[identifier]
        if not user_requests:
            return 0
        
        oldest_request = user_requests[0]
        reset_time = oldest_request + self.window_seconds
        return max(0, int(reset_time - time.time()))


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enterprise-grade security middleware."""
    
    def __init__(
        self,
        app,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
        max_content_length: int = 50 * 1024 * 1024,  # 50MB
        blocked_user_agents: Optional[List[str]] = None,
        require_https: bool = False
    ):
        super().__init__(app)
        
        # Rate limiting
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        
        # Content limits
        self.max_content_length = max_content_length
        
        # Security settings
        self.blocked_user_agents = blocked_user_agents or [
            "curl", "wget", "python-requests", "bot", "crawler", "scanner"
        ]
        self.require_https = require_https
        
        # Suspicious activity tracking
        self.suspicious_ips: Dict[str, Dict] = defaultdict(lambda: {
            "violations": 0,
            "last_violation": 0,
            "blocked_until": 0
        })
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        logger.info(f"Security middleware initialized with rate limit: {rate_limit_requests}/{rate_limit_window}s")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security checks."""
        try:
            # Get client identifier
            client_ip = self._get_client_ip(request)
            client_id = self._get_client_identifier(request)
            
            # Security checks
            security_check = await self._perform_security_checks(request, client_ip, client_id)
            if security_check:
                return security_check
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return JSONResponse(
                {"error": "Security check failed"},
                status_code=500
            )
    
    async def _perform_security_checks(
        self, 
        request: Request, 
        client_ip: str, 
        client_id: str
    ) -> Optional[JSONResponse]:
        """Perform comprehensive security checks."""
        
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip):
            logger.warning(f"Blocked IP attempted access: {client_ip}")
            return JSONResponse(
                {"error": "Access denied"},
                status_code=403
            )
        
        # HTTPS enforcement
        if self.require_https and request.url.scheme != "https":
            logger.warning(f"HTTP request rejected from {client_ip}")
            return JSONResponse(
                {"error": "HTTPS required"},
                status_code=403
            )
        
        # Rate limiting
        if not self.rate_limiter.is_allowed(client_id):
            reset_time = self.rate_limiter.get_reset_time(client_id)
            logger.warning(f"Rate limit exceeded for {client_id}")
            
            # Track as violation
            self._record_violation(client_ip, "rate_limit")
            
            return JSONResponse(
                {
                    "error": "Rate limit exceeded",
                    "reset_in_seconds": reset_time
                },
                status_code=429,
                headers={"Retry-After": str(reset_time)}
            )
        
        # Content length check
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            logger.warning(f"Content length exceeded from {client_ip}: {content_length}")
            self._record_violation(client_ip, "content_length")
            
            return JSONResponse(
                {"error": "Request too large"},
                status_code=413
            )
        
        # User agent check
        user_agent = request.headers.get("user-agent", "").lower()
        if any(blocked in user_agent for blocked in self.blocked_user_agents):
            logger.warning(f"Blocked user agent from {client_ip}: {user_agent}")
            self._record_violation(client_ip, "user_agent")
            
            return JSONResponse(
                {"error": "Access denied"},
                status_code=403
            )
        
        # Path traversal check
        if self._has_path_traversal(str(request.url.path)):
            logger.warning(f"Path traversal attempt from {client_ip}: {request.url.path}")
            self._record_violation(client_ip, "path_traversal")
            
            return JSONResponse(
                {"error": "Invalid request"},
                status_code=400
            )
        
        # SQL injection basic check
        if self._has_sql_injection_patterns(request):
            logger.warning(f"SQL injection attempt from {client_ip}")
            self._record_violation(client_ip, "sql_injection")
            
            return JSONResponse(
                {"error": "Invalid request"},
                status_code=400
            )
        
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check forwarded headers (for proxy/load balancer scenarios)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _get_client_identifier(self, request: Request) -> str:
        """Generate unique client identifier for rate limiting."""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Create hash-based identifier
        identifier_string = f"{client_ip}:{user_agent}"
        return hashlib.md5(identifier_string.encode()).hexdigest()[:16]
    
    def _is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is currently blocked."""
        ip_data = self.suspicious_ips[client_ip]
        
        # Check if still blocked
        if ip_data["blocked_until"] > time.time():
            return True
        
        # Auto-unblock if block period expired
        if ip_data["blocked_until"] > 0 and ip_data["blocked_until"] <= time.time():
            ip_data["blocked_until"] = 0
            ip_data["violations"] = max(0, ip_data["violations"] - 1)
        
        return False
    
    def _record_violation(self, client_ip: str, violation_type: str):
        """Record security violation and potentially block IP."""
        ip_data = self.suspicious_ips[client_ip]
        ip_data["violations"] += 1
        ip_data["last_violation"] = time.time()
        
        # Block IP if too many violations
        if ip_data["violations"] >= 5:
            block_duration = min(3600, 60 * (2 ** (ip_data["violations"] - 5)))  # Exponential backoff, max 1 hour
            ip_data["blocked_until"] = time.time() + block_duration
            
            logger.warning(
                f"IP {client_ip} blocked for {block_duration}s after {ip_data['violations']} violations"
            )
    
    def _has_path_traversal(self, path: str) -> bool:
        """Check for path traversal attempts."""
        suspicious_patterns = ["../", "..\\", "%2e%2e", "....//", "..;/"]
        path_lower = path.lower()
        
        return any(pattern in path_lower for pattern in suspicious_patterns)
    
    def _has_sql_injection_patterns(self, request: Request) -> bool:
        """Basic SQL injection pattern detection."""
        sql_patterns = [
            "union select", "drop table", "insert into", "delete from",
            "update set", "exec xp_", "sp_executesql", "'; --", "' or '1'='1"
        ]
        
        # Check URL parameters
        query_string = str(request.url.query).lower()
        if any(pattern in query_string for pattern in sql_patterns):
            return True
        
        # Check headers for injection attempts
        for header_value in request.headers.values():
            if any(pattern in header_value.lower() for pattern in sql_patterns):
                return True
        
        return False
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        for header, value in self.security_headers.items():
            response.headers[header] = value
    
    def get_security_stats(self) -> Dict:
        """Get security middleware statistics."""
        total_violations = sum(
            data["violations"] for data in self.suspicious_ips.values()
        )
        
        blocked_ips = sum(
            1 for data in self.suspicious_ips.values()
            if data["blocked_until"] > time.time()
        )
        
        return {
            "total_ips_tracked": len(self.suspicious_ips),
            "total_violations": total_violations,
            "currently_blocked_ips": blocked_ips,
            "rate_limiter_active": True,
            "max_content_length_mb": self.max_content_length / (1024 * 1024)
        }
