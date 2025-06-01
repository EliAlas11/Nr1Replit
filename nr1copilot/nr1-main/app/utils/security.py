"""
Netflix-Level Security Manager v5.0
Comprehensive security features including authentication, authorization, and threat protection
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass
from enum import Enum
import jwt
import bcrypt
from fastapi import HTTPException, Request
import ipaddress

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(str, Enum):
    """Threat type enumeration"""
    BRUTE_FORCE = "brute_force"
    DOS_ATTACK = "dos_attack"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    MALICIOUS_PAYLOAD = "malicious_payload"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    timestamp: datetime
    client_ip: str
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    details: Dict[str, Any]
    blocked: bool = False


@dataclass
class UserSession:
    """User session data structure"""
    session_id: str
    user_id: str
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    permissions: List[str]
    metadata: Dict[str, Any]


class SecurityManager:
    """Netflix-level security manager with comprehensive protection"""

    def __init__(
        self,
        secret_key: str = None,
        token_expire_minutes: int = 1440,  # 24 hours
        max_login_attempts: int = 5,
        lockout_duration: int = 900,  # 15 minutes
        suspicious_threshold: int = 10
    ):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.token_expire_minutes = token_expire_minutes
        self.max_login_attempts = max_login_attempts
        self.lockout_duration = lockout_duration
        self.suspicious_threshold = suspicious_threshold

        # Security state
        self.active_sessions: Dict[str, UserSession] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        self.security_events: List[SecurityEvent] = []
        self.whitelist_ips: Set[str] = set()
        self.blacklist_ips: Set[str] = set()

        # Rate limiting per IP
        self.request_counts: Dict[str, List[datetime]] = {}
        self.rate_limit_window = 60  # 1 minute
        self.rate_limit_max_requests = 100

        # Malicious patterns
        self.malicious_patterns = [
            r"<script.*?>.*?</script>",  # XSS
            r"union\s+select",  # SQL injection
            r"drop\s+table",  # SQL injection
            r"exec\s*\(",  # Code injection
            r"eval\s*\(",  # Code injection
            r"\.\.\/",  # Path traversal
            r"\/etc\/passwd",  # File access
            r"cmd\.exe",  # Command injection
        ]

        # Initialize security components
        self._setup_security_rules()

    def _setup_security_rules(self):
        """Setup security rules and patterns"""
        # Add common attack IP ranges to blacklist
        dangerous_ranges = [
            "10.0.0.0/8",  # Private range (if not internal)
            "172.16.0.0/12",  # Private range
            "192.168.0.0/16",  # Private range
        ]

        # Note: In production, this would be configured differently
        # and private ranges might be whitelisted for internal use

    async def authenticate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )

            # Check token expiration
            if payload.get("exp", 0) < time.time():
                return None

            # Validate session
            session_id = payload.get("session_id")
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.last_activity = datetime.utcnow()
                return {
                    "user_id": session.user_id,
                    "session_id": session_id,
                    "permissions": session.permissions
                }

            return payload

        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token authentication error: {e}")
            return None

    async def create_token(
        self,
        user_id: str,
        permissions: List[str] = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> str:
        """Create authenticated JWT token"""
        try:
            session_id = secrets.token_urlsafe(32)
            now = datetime.utcnow()

            # Create session
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address or "unknown",
                user_agent=user_agent or "unknown",
                created_at=now,
                last_activity=now,
                permissions=permissions or ["read"],
                metadata={}
            )

            self.active_sessions[session_id] = session

            # Create token payload
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "permissions": permissions or ["read"],
                "iat": int(now.timestamp()),
                "exp": int((now + timedelta(minutes=self.token_expire_minutes)).timestamp()),
                "iss": "viralclip-pro-v5"
            }

            token = jwt.encode(
                payload,
                self.secret_key,
                algorithm="HS256"
            )

            logger.info(f"Token created for user {user_id}")
            return token

        except Exception as e:
            logger.error(f"Token creation error: {e}")
            raise

    async def validate_request(self, request: Request) -> Dict[str, Any]:
        """Comprehensive request validation"""
        client_ip = self._get_client_ip(request)

        # Check IP blacklist
        if await self._is_blacklisted(client_ip):
            await self._log_security_event(
                client_ip=client_ip,
                threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                severity=SecurityLevel.HIGH,
                description="Blacklisted IP access attempt",
                blocked=True
            )
            raise HTTPException(status_code=403, detail="Access denied")

        # Check rate limiting
        if not await self._check_rate_limit(client_ip):
            await self._log_security_event(
                client_ip=client_ip,
                threat_type=ThreatType.DOS_ATTACK,
                severity=SecurityLevel.MEDIUM,
                description="Rate limit exceeded",
                blocked=True
            )
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Check for malicious patterns
        await self._scan_request_for_threats(request, client_ip)

        return {
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", ""),
            "validation_passed": True
        }

    async def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host

    async def _is_blacklisted(self, ip: str) -> bool:
        """Check if IP is blacklisted"""
        # Check explicit blacklist
        if ip in self.blacklist_ips:
            return True

        # Check if IP is currently blocked
        if ip in self.blocked_ips:
            block_time = self.blocked_ips[ip]
            if datetime.utcnow() < block_time + timedelta(seconds=self.lockout_duration):
                return True
            else:
                # Remove expired block
                del self.blocked_ips[ip]

        # Check IP ranges
        try:
            ip_addr = ipaddress.ip_address(ip)
            for range_str in self.blacklist_ips:
                if "/" in range_str:  # CIDR notation
                    network = ipaddress.ip_network(range_str, strict=False)
                    if ip_addr in network:
                        return True
        except ValueError:
            pass

        return False

    async def _check_rate_limit(self, ip: str) -> bool:
        """Check rate limiting for IP"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.rate_limit_window)

        # Initialize or clean old requests
        if ip not in self.request_counts:
            self.request_counts[ip] = []

        # Remove old requests outside window
        self.request_counts[ip] = [
            req_time for req_time in self.request_counts[ip]
            if req_time > window_start
        ]

        # Check if within limit
        if len(self.request_counts[ip]) >= self.rate_limit_max_requests:
            return False

        # Add current request
        self.request_counts[ip].append(now)
        return True

    async def _scan_request_for_threats(self, request: Request, client_ip: str):
        """Scan request for malicious patterns"""
        import re

        # Get request data
        url = str(request.url)
        headers = dict(request.headers)

        # Scan URL for threats
        for pattern in self.malicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                await self._log_security_event(
                    client_ip=client_ip,
                    threat_type=ThreatType.MALICIOUS_PAYLOAD,
                    severity=SecurityLevel.HIGH,
                    description=f"Malicious pattern detected in URL: {pattern}",
                    details={"url": url, "pattern": pattern},
                    blocked=True
                )
                raise HTTPException(status_code=400, detail="Malicious request detected")

        # Scan headers for threats
        for header, value in headers.items():
            for pattern in self.malicious_patterns:
                if re.search(pattern, str(value), re.IGNORECASE):
                    await self._log_security_event(
                        client_ip=client_ip,
                        threat_type=ThreatType.MALICIOUS_PAYLOAD,
                        severity=SecurityLevel.HIGH,
                        description=f"Malicious pattern detected in header {header}",
                        details={"header": header, "value": value, "pattern": pattern},
                        blocked=True
                    )
                    raise HTTPException(status_code=400, detail="Malicious request detected")

    async def _log_security_event(
        self,
        client_ip: str,
        threat_type: ThreatType,
        severity: SecurityLevel,
        description: str,
        details: Dict[str, Any] = None,
        blocked: bool = False
    ):
        """Log security event"""
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            timestamp=datetime.utcnow(),
            client_ip=client_ip,
            threat_type=threat_type,
            severity=severity,
            description=description,
            details=details or {},
            blocked=blocked
        )

        self.security_events.append(event)

        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]

        # Log to system logger
        log_level = {
            SecurityLevel.LOW: logging.INFO,
            SecurityLevel.MEDIUM: logging.WARNING,
            SecurityLevel.HIGH: logging.ERROR,
            SecurityLevel.CRITICAL: logging.CRITICAL
        }.get(severity, logging.WARNING)

        logger.log(
            log_level,
            f"Security Event [{event.event_id}]: {description} - IP: {client_ip} - Type: {threat_type.value}"
        )

        # Auto-block for critical threats
        if severity == SecurityLevel.CRITICAL or blocked:
            await self._auto_block_ip(client_ip, event)

    async def _auto_block_ip(self, ip: str, event: SecurityEvent):
        """Automatically block IP for security violations"""
        # Add to blocked IPs
        self.blocked_ips[ip] = datetime.utcnow()

        logger.warning(f"Auto-blocked IP {ip} for {self.lockout_duration} seconds due to: {event.description}")

    async def record_failed_login(self, ip: str, user_id: str = None):
        """Record failed login attempt"""
        now = datetime.utcnow()

        if ip not in self.failed_attempts:
            self.failed_attempts[ip] = []

        self.failed_attempts[ip].append(now)

        # Clean old attempts (older than lockout duration)
        cutoff = now - timedelta(seconds=self.lockout_duration)
        self.failed_attempts[ip] = [
            attempt for attempt in self.failed_attempts[ip]
            if attempt > cutoff
        ]

        # Check if should block
        if len(self.failed_attempts[ip]) >= self.max_login_attempts:
            await self._log_security_event(
                client_ip=ip,
                threat_type=ThreatType.BRUTE_FORCE,
                severity=SecurityLevel.HIGH,
                description=f"Brute force attack detected - {len(self.failed_attempts[ip])} failed attempts",
                details={"user_id": user_id, "attempts": len(self.failed_attempts[ip])},
                blocked=True
            )

    async def record_successful_login(self, ip: str, user_id: str):
        """Record successful login"""
        # Clear failed attempts for this IP
        if ip in self.failed_attempts:
            del self.failed_attempts[ip]

        await self._log_security_event(
            client_ip=ip,
            threat_type=ThreatType.UNAUTHORIZED_ACCESS,  # Using as general category
            severity=SecurityLevel.LOW,
            description=f"Successful login for user {user_id}",
            details={"user_id": user_id, "event_type": "successful_login"}
        )

    async def revoke_session(self, session_id: str):
        """Revoke user session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            del self.active_sessions[session_id]

            logger.info(f"Session revoked: {session_id} for user {session.user_id}")

    async def cleanup_expired_sessions(self):
        """Cleanup expired sessions"""
        now = datetime.utcnow()
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            # Check if session is expired (no activity for 24 hours)
            if (now - session.last_activity).total_seconds() > (self.token_expire_minutes * 60):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self.revoke_session(session_id)

        logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)

        recent_events = [
            event for event in self.security_events
            if event.timestamp > last_24h
        ]

        threat_counts = {}
        for event in recent_events:
            threat_type = event.threat_type.value
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1

        return {
            "active_sessions": len(self.active_sessions),
            "blocked_ips": len(self.blocked_ips),
            "failed_attempts_ips": len(self.failed_attempts),
            "security_events_24h": len(recent_events),
            "threat_breakdown": threat_counts,
            "blacklisted_ips": len(self.blacklist_ips),
            "whitelisted_ips": len(self.whitelist_ips)
        }

    async def add_to_whitelist(self, ip: str):
        """Add IP to whitelist"""
        self.whitelist_ips.add(ip)
        logger.info(f"Added IP to whitelist: {ip}")

    async def add_to_blacklist(self, ip: str):
        """Add IP to blacklist"""
        self.blacklist_ips.add(ip)
        logger.info(f"Added IP to blacklist: {ip}")

    async def remove_from_blacklist(self, ip: str):
        """Remove IP from blacklist"""
        self.blacklist_ips.discard(ip)
        if ip in self.blocked_ips:
            del self.blocked_ips[ip]
        logger.info(f"Removed IP from blacklist: {ip}")

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)

    def compute_hmac(self, data: str, key: str = None) -> str:
        """Compute HMAC-SHA256"""
        key = key or self.secret_key
        return hmac.new(
            key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user data"""
        return await self.authenticate_token(token)


# Export main classes
__all__ = [
    "SecurityManager",
    "SecurityLevel",
    "ThreatType",
    "SecurityEvent",
    "UserSession"
]