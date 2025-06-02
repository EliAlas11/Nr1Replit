
"""
Netflix-Level Security Middleware
Comprehensive security protection and threat detection
"""

import asyncio
import hashlib
import logging
import time
import ipaddress
from typing import Callable, Dict, Set, List, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Netflix-level security with advanced threat detection"""
    
    def __init__(self, app):
        super().__init__(app)
        
        # Threat detection
        self.failed_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Security rules
        self.max_failed_attempts = 5
        self.ban_duration = 3600  # 1 hour
        self.rate_limit_window = 60  # 1 minute
        self.rate_limit_requests = 100
        
        # Suspicious patterns
        self.malicious_patterns = [
            "script", "alert", "onload", "onerror", "javascript:",
            "../", "..\\", "/etc/passwd", "/proc/", "cmd.exe",
            "union select", "drop table", "delete from", "insert into",
            "<script", "</script>", "eval(", "setTimeout(", "setInterval("
        ]
        
        # Known bot user agents
        self.bot_patterns = [
            "bot", "crawler", "spider", "scraper", "automated"
        ]
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Advanced security filtering"""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "").lower()
        
        # Add security context
        request.state.client_ip = client_ip
        request.state.security_score = 0
        
        try:
            # Security checks
            security_result = await self._perform_security_checks(request, client_ip, user_agent)
            
            if security_result["blocked"]:
                return self._security_block_response(security_result)
                
            # Add security headers to request context
            request.state.security_context = security_result
            
            response = await call_next(request)
            
            # Add security headers to response
            self._add_security_headers(response)
            
            # Track successful request
            self._track_successful_request(client_ip)
            
            return response
            
        except Exception as e:
            # Track failed request
            self._track_failed_request(client_ip, str(e))
            raise
            
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP considering proxies"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
            
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
            
        return request.client.host
        
    async def _perform_security_checks(self, request: Request, client_ip: str, user_agent: str) -> Dict[str, Any]:
        """Comprehensive security assessment"""
        security_result = {
            "blocked": False,
            "reason": None,
            "threat_level": "low",
            "checks": {}
        }
        
        # Check 1: IP blocking
        if self._is_ip_blocked(client_ip):
            security_result.update({
                "blocked": True,
                "reason": "IP blocked due to previous violations",
                "threat_level": "high"
            })
            return security_result
            
        # Check 2: Rate limiting
        rate_limit_result = self._check_rate_limit(client_ip)
        security_result["checks"]["rate_limit"] = rate_limit_result
        if rate_limit_result["exceeded"]:
            security_result.update({
                "blocked": True,
                "reason": "Rate limit exceeded",
                "threat_level": "medium"
            })
            return security_result
            
        # Check 3: Failed login attempts
        failed_attempts_result = self._check_failed_attempts(client_ip)
        security_result["checks"]["failed_attempts"] = failed_attempts_result
        if failed_attempts_result["blocked"]:
            security_result.update({
                "blocked": True,
                "reason": "Too many failed attempts",
                "threat_level": "high"
            })
            return security_result
            
        # Check 4: Malicious patterns
        malicious_result = await self._check_malicious_patterns(request)
        security_result["checks"]["malicious_patterns"] = malicious_result
        if malicious_result["detected"]:
            security_result.update({
                "blocked": True,
                "reason": "Malicious pattern detected",
                "threat_level": "critical"
            })
            return security_result
            
        # Check 5: Bot detection
        bot_result = self._check_bot_detection(user_agent, request)
        security_result["checks"]["bot_detection"] = bot_result
        
        # Check 6: Geographic restrictions (if configured)
        geo_result = self._check_geographic_restrictions(client_ip)
        security_result["checks"]["geographic"] = geo_result
        
        # Calculate overall threat level
        security_result["threat_level"] = self._calculate_threat_level(security_result["checks"])
        
        return security_result
        
    def _is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is currently blocked"""
        return client_ip in self.blocked_ips
        
    def _check_rate_limit(self, client_ip: str) -> Dict[str, Any]:
        """Check rate limiting for IP"""
        now = time.time()
        window_start = now - self.rate_limit_window
        
        # Clean old requests
        while self.rate_limits[client_ip] and self.rate_limits[client_ip][0] < window_start:
            self.rate_limits[client_ip].popleft()
            
        # Add current request
        self.rate_limits[client_ip].append(now)
        
        current_count = len(self.rate_limits[client_ip])
        
        return {
            "exceeded": current_count > self.rate_limit_requests,
            "current_count": current_count,
            "limit": self.rate_limit_requests,
            "window": self.rate_limit_window
        }
        
    def _check_failed_attempts(self, client_ip: str) -> Dict[str, Any]:
        """Check failed login attempts"""
        recent_failures = len(self.failed_attempts[client_ip])
        
        return {
            "blocked": recent_failures >= self.max_failed_attempts,
            "count": recent_failures,
            "max_allowed": self.max_failed_attempts
        }
        
    async def _check_malicious_patterns(self, request: Request) -> Dict[str, Any]:
        """Check for malicious patterns in request"""
        detected_patterns = []
        
        # Check URL path
        path = request.url.path.lower()
        for pattern in self.malicious_patterns:
            if pattern in path:
                detected_patterns.append(f"URL: {pattern}")
                
        # Check query parameters
        for key, value in request.query_params.items():
            value_lower = str(value).lower()
            for pattern in self.malicious_patterns:
                if pattern in value_lower:
                    detected_patterns.append(f"Query {key}: {pattern}")
                    
        # Check headers
        for header_name, header_value in request.headers.items():
            header_value_lower = header_value.lower()
            for pattern in self.malicious_patterns:
                if pattern in header_value_lower:
                    detected_patterns.append(f"Header {header_name}: {pattern}")
                    
        return {
            "detected": len(detected_patterns) > 0,
            "patterns": detected_patterns,
            "count": len(detected_patterns)
        }
        
    def _check_bot_detection(self, user_agent: str, request: Request) -> Dict[str, Any]:
        """Detect and classify bots"""
        is_bot = False
        bot_type = "unknown"
        
        # Check user agent patterns
        for pattern in self.bot_patterns:
            if pattern in user_agent:
                is_bot = True
                bot_type = pattern
                break
                
        # Check for missing common headers
        missing_headers = []
        common_headers = ["accept", "accept-language", "accept-encoding"]
        for header in common_headers:
            if header not in request.headers:
                missing_headers.append(header)
                
        # Suspicious if many headers missing
        if len(missing_headers) >= 2:
            is_bot = True
            bot_type = "suspicious_headers"
            
        return {
            "is_bot": is_bot,
            "bot_type": bot_type,
            "confidence": 0.8 if is_bot else 0.2,
            "missing_headers": missing_headers
        }
        
    def _check_geographic_restrictions(self, client_ip: str) -> Dict[str, Any]:
        """Check geographic restrictions (simplified)"""
        # In production, this would use a GeoIP service
        return {
            "restricted": False,
            "country": "unknown",
            "allowed": True
        }
        
    def _calculate_threat_level(self, checks: Dict[str, Any]) -> str:
        """Calculate overall threat level"""
        score = 0
        
        # Rate limiting
        if checks.get("rate_limit", {}).get("exceeded"):
            score += 2
            
        # Failed attempts
        failed_count = checks.get("failed_attempts", {}).get("count", 0)
        if failed_count > 0:
            score += min(failed_count, 5)
            
        # Malicious patterns
        if checks.get("malicious_patterns", {}).get("detected"):
            score += 10
            
        # Bot detection
        if checks.get("bot_detection", {}).get("is_bot"):
            score += 1
            
        # Classify threat level
        if score >= 10:
            return "critical"
        elif score >= 5:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"
            
    def _security_block_response(self, security_result: Dict[str, Any]) -> JSONResponse:
        """Return security block response"""
        logger.warning(
            f"Security block: {security_result['reason']}",
            extra={
                "threat_level": security_result["threat_level"],
                "checks": security_result["checks"]
            }
        )
        
        status_code = 403
        if security_result["threat_level"] == "critical":
            status_code = 418  # I'm a teapot (rate limiting)
        elif "rate limit" in security_result["reason"].lower():
            status_code = 429
            
        return JSONResponse(
            status_code=status_code,
            content={
                "error": True,
                "message": "Access denied",
                "reason": security_result["reason"],
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
            
    def _track_successful_request(self, client_ip: str):
        """Track successful request"""
        # Reduce failed attempt count on success
        if client_ip in self.failed_attempts and self.failed_attempts[client_ip]:
            self.failed_attempts[client_ip].popleft()
            
    def _track_failed_request(self, client_ip: str, error: str):
        """Track failed request"""
        self.failed_attempts[client_ip].append({
            "timestamp": datetime.utcnow(),
            "error": error
        })
        
        # Block IP if too many failures
        if len(self.failed_attempts[client_ip]) >= self.max_failed_attempts:
            self.blocked_ips.add(client_ip)
            logger.warning(f"IP blocked due to failed attempts: {client_ip}")
            
            # Schedule unblock
            asyncio.create_task(self._schedule_unblock(client_ip))
            
    async def _schedule_unblock(self, client_ip: str):
        """Schedule IP unblock after ban duration"""
        await asyncio.sleep(self.ban_duration)
        if client_ip in self.blocked_ips:
            self.blocked_ips.remove(client_ip)
            logger.info(f"IP unblocked: {client_ip}")
            
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary"""
        return {
            "blocked_ips": len(self.blocked_ips),
            "active_rate_limits": len(self.rate_limits),
            "failed_attempts": {ip: len(attempts) for ip, attempts in self.failed_attempts.items()},
            "total_suspicious_patterns": sum(self.suspicious_patterns.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
