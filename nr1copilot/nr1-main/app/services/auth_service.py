"""
Netflix-Level Authentication Service v10.0
Complete authentication system with JWT, 2FA, and enterprise features
"""

import asyncio
import logging
import secrets
import time
import pyotp
import qrcode
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import jwt
import bcrypt

from app.utils.security import SecurityManager, SecurityLevel, ThreatType
from app.utils.cache import CacheManager
from app.database.models import User, UserStatus

logger = logging.getLogger(__name__)


class AuthProvider(str, Enum):
    """Authentication provider types"""
    LOCAL = "local"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    REPLIT = "replit"


class SessionType(str, Enum):
    """Session type enumeration"""
    WEB = "web"
    API = "api"
    MOBILE = "mobile"
    ADMIN = "admin"


@dataclass
class AuthResult:
    """Authentication result"""
    success: bool
    user_id: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    requires_2fa: bool = False
    error_message: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class TwoFactorSetup:
    """2FA setup data"""
    secret: str
    qr_code: str
    backup_codes: List[str]


class LoginRequest(BaseModel):
    """Login request model"""
    email: EmailStr
    password: str
    remember_me: bool = False
    session_type: SessionType = SessionType.WEB
    two_factor_code: Optional[str] = None


class RegisterRequest(BaseModel):
    """Registration request model"""
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None


class NetflixLevelAuthService:
    """Netflix-level authentication service with enterprise features"""

    def __init__(self):
        self.security_manager = SecurityManager()
        self.cache_manager = CacheManager()
        self.bearer_security = HTTPBearer()

        # Authentication settings
        self.secret_key = secrets.token_urlsafe(64)
        self.access_token_expire = 15  # 15 minutes
        self.refresh_token_expire = 7 * 24 * 60  # 7 days
        self.remember_me_expire = 30 * 24 * 60  # 30 days

        # 2FA settings
        self.totp_issuer = "ViralClip Pro"
        self.backup_codes_count = 10

        # Session management
        self.active_sessions: Dict[str, Dict] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.blocked_accounts: Dict[str, datetime] = {}

        # Activity logging
        self.activity_log: List[Dict] = []

        logger.info("Netflix-Level Authentication Service initialized")

    async def register_user(
        self, 
        request: RegisterRequest,
        client_ip: str = None
    ) -> AuthResult:
        """Register new user with security validation"""
        try:
            # Validate request security
            await self._validate_registration_security(request, client_ip)

            # Check if user exists
            existing_user = await self._get_user_by_email(request.email)
            if existing_user:
                await self._log_activity(
                    event_type="registration_attempt",
                    email=request.email,
                    success=False,
                    details={"reason": "email_exists", "ip": client_ip}
                )
                return AuthResult(
                    success=False,
                    error_message="Email already registered"
                )

            # Validate password strength
            if not self._validate_password_strength(request.password):
                return AuthResult(
                    success=False,
                    error_message="Password does not meet security requirements"
                )

            # Create user
            user_id = secrets.token_urlsafe(16)
            hashed_password = self.security_manager.hash_password(request.password)

            # Store user (in production, this would be in database)
            user_data = {
                "id": user_id,
                "email": request.email,
                "username": request.username,
                "full_name": request.full_name,
                "password_hash": hashed_password,
                "status": UserStatus.ACTIVE,
                "created_at": datetime.utcnow(),
                "email_verified": False,
                "two_factor_enabled": False
            }

            await self._store_user(user_data)

            # Generate tokens
            access_token = await self._create_access_token(user_id, ["user"])
            refresh_token = await self._create_refresh_token(user_id)

            await self._log_activity(
                event_type="user_registration",
                user_id=user_id,
                email=request.email,
                success=True,
                details={"ip": client_ip}
            )

            return AuthResult(
                success=True,
                user_id=user_id,
                access_token=access_token,
                refresh_token=refresh_token
            )

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return AuthResult(
                success=False,
                error_message="Registration failed"
            )

    async def authenticate_user(
        self, 
        request: LoginRequest,
        client_ip: str = None
    ) -> AuthResult:
        """Authenticate user with comprehensive security"""
        try:
            # Check for account lockout
            if await self._is_account_locked(request.email):
                await self._log_activity(
                    event_type="login_attempt",
                    email=request.email,
                    success=False,
                    details={"reason": "account_locked", "ip": client_ip}
                )
                return AuthResult(
                    success=False,
                    error_message="Account temporarily locked due to security reasons"
                )

            # Get user
            user = await self._get_user_by_email(request.email)
            if not user:
                await self._record_failed_attempt(request.email, client_ip)
                return AuthResult(
                    success=False,
                    error_message="Invalid credentials"
                )

            # Verify password
            if not self.security_manager.verify_password(
                request.password, 
                user["password_hash"]
            ):
                await self._record_failed_attempt(request.email, client_ip)
                return AuthResult(
                    success=False,
                    error_message="Invalid credentials"
                )

            # Check 2FA if enabled
            if user.get("two_factor_enabled", False):
                if not request.two_factor_code:
                    return AuthResult(
                        success=False,
                        requires_2fa=True,
                        error_message="Two-factor authentication required"
                    )

                if not await self._verify_2fa_code(user["id"], request.two_factor_code):
                    await self._record_failed_attempt(request.email, client_ip)
                    return AuthResult(
                        success=False,
                        error_message="Invalid two-factor authentication code"
                    )

            # Generate session
            session_id = await self._create_session(
                user["id"],
                client_ip,
                request.session_type
            )

            # Generate tokens
            permissions = await self._get_user_permissions(user["id"])
            token_expiry = (
                self.remember_me_expire if request.remember_me 
                else self.access_token_expire
            )

            access_token = await self._create_access_token(
                user["id"], 
                permissions,
                expire_minutes=token_expiry
            )
            refresh_token = await self._create_refresh_token(user["id"])

            # Clear failed attempts
            await self._clear_failed_attempts(request.email)

            await self._log_activity(
                event_type="user_login",
                user_id=user["id"],
                email=request.email,
                success=True,
                details={
                    "ip": client_ip,
                    "session_type": request.session_type.value,
                    "remember_me": request.remember_me,
                    "two_factor_used": user.get("two_factor_enabled", False)
                }
            )

            return AuthResult(
                success=True,
                user_id=user["id"],
                access_token=access_token,
                refresh_token=refresh_token,
                session_id=session_id
            )

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            await self._record_failed_attempt(request.email, client_ip)
            return AuthResult(
                success=False,
                error_message="Authentication failed"
            )

    async def setup_two_factor(self, user_id: str) -> TwoFactorSetup:
        """Setup two-factor authentication for user"""
        try:
            # Generate secret
            secret = pyotp.random_base32()

            # Create TOTP
            totp = pyotp.TOTP(secret)

            # Generate QR code
            provisioning_uri = totp.provisioning_uri(
                name=user_id,
                issuer_name=self.totp_issuer
            )

            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            qr_code_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            qr_code_data_uri = f"data:image/png;base64,{qr_code_base64}"

            # Generate backup codes
            backup_codes = [
                secrets.token_hex(4).upper() 
                for _ in range(self.backup_codes_count)
            ]

            # Store 2FA setup (temporarily, until confirmed)
            await self.cache_manager.set(
                f"2fa_setup:{user_id}",
                {
                    "secret": secret,
                    "backup_codes": backup_codes,
                    "created_at": datetime.utcnow().isoformat()
                },
                expire=300  # 5 minutes
            )

            await self._log_activity(
                event_type="2fa_setup_initiated",
                user_id=user_id,
                success=True
            )

            return TwoFactorSetup(
                secret=secret,
                qr_code=qr_code_data_uri,
                backup_codes=backup_codes
            )

        except Exception as e:
            logger.error(f"2FA setup failed: {e}")
            raise HTTPException(status_code=500, detail="2FA setup failed")

    async def verify_and_enable_2fa(
        self, 
        user_id: str, 
        verification_code: str
    ) -> bool:
        """Verify 2FA setup and enable it"""
        try:
            # Get setup data
            setup_data = await self.cache_manager.get(f"2fa_setup:{user_id}")
            if not setup_data:
                return False

            # Verify code
            totp = pyotp.TOTP(setup_data["secret"])
            if not totp.verify(verification_code):
                return False

            # Enable 2FA for user
            await self._update_user_2fa(
                user_id, 
                enabled=True,
                secret=setup_data["secret"],
                backup_codes=setup_data["backup_codes"]
            )

            # Clear setup cache
            await self.cache_manager.delete(f"2fa_setup:{user_id}")

            await self._log_activity(
                event_type="2fa_enabled",
                user_id=user_id,
                success=True
            )

            return True

        except Exception as e:
            logger.error(f"2FA verification failed: {e}")
            return False

    async def validate_token(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> Dict[str, Any]:
        """Validate JWT token and return user data"""
        try:
            token = credentials.credentials

            # Decode token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )

            # Check expiration
            if payload.get("exp", 0) < time.time():
                raise HTTPException(status_code=401, detail="Token expired")

            # Validate session
            session_id = payload.get("session_id")
            if session_id and session_id not in self.active_sessions:
                raise HTTPException(status_code=401, detail="Session invalid")

            # Update session activity
            if session_id:
                self.active_sessions[session_id]["last_activity"] = datetime.utcnow()

            return {
                "user_id": payload["user_id"],
                "permissions": payload.get("permissions", []),
                "session_id": session_id,
                "session_type": payload.get("session_type", "web")
            }

        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")

    async def refresh_token(self, refresh_token: str) -> AuthResult:
        """Refresh access token using refresh token"""
        try:
            # Decode refresh token
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=["HS256"]
            )

            user_id = payload["user_id"]

            # Generate new access token
            permissions = await self._get_user_permissions(user_id)
            access_token = await self._create_access_token(user_id, permissions)

            await self._log_activity(
                event_type="token_refresh",
                user_id=user_id,
                success=True
            )

            return AuthResult(
                success=True,
                user_id=user_id,
                access_token=access_token,
                refresh_token=refresh_token
            )

        except jwt.InvalidTokenError:
            return AuthResult(
                success=False,
                error_message="Invalid refresh token"
            )

    async def logout(self, session_id: str, user_id: str) -> bool:
        """Logout user and invalidate session"""
        try:
            # Remove session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            await self._log_activity(
                event_type="user_logout",
                user_id=user_id,
                success=True,
                details={"session_id": session_id}
            )

            return True

        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False

    # Helper methods
    async def _create_access_token(
        self, 
        user_id: str, 
        permissions: List[str],
        expire_minutes: int = None
    ) -> str:
        """Create JWT access token"""
        expire_minutes = expire_minutes or self.access_token_expire
        expire = datetime.utcnow() + timedelta(minutes=expire_minutes)

        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "type": "access",
            "iat": int(datetime.utcnow().timestamp()),
            "exp": int(expire.timestamp()),
            "iss": "viralclip-pro-v10"
        }

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    async def _create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        expire = datetime.utcnow() + timedelta(minutes=self.refresh_token_expire)

        payload = {
            "user_id": user_id,
            "type": "refresh",
            "iat": int(datetime.utcnow().timestamp()),
            "exp": int(expire.timestamp()),
            "iss": "viralclip-pro-v10"
        }

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    async def _create_session(
        self, 
        user_id: str, 
        client_ip: str, 
        session_type: SessionType
    ) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)

        self.active_sessions[session_id] = {
            "user_id": user_id,
            "client_ip": client_ip,
            "session_type": session_type.value,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }

        return session_id

    async def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < 8:
            return False

        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

        return has_upper and has_lower and has_digit and has_special

    async def _verify_2fa_code(self, user_id: str, code: str) -> bool:
        """Verify 2FA code"""
        try:
            user = await self._get_user_by_id(user_id)
            if not user or not user.get("two_factor_secret")):
                return False

            # Try TOTP first
            totp = pyotp.TOTP(user["two_factor_secret"])
            if totp.verify(code):
                return True

            # Try backup codes
            backup_codes = user.get("backup_codes", [])
            if code.upper() in backup_codes:
                # Remove used backup code
                backup_codes.remove(code.upper())
                await self._update_user_backup_codes(user_id, backup_codes)
                return True

            return False

        except Exception as e:
            logger.error(f"2FA verification error: {e}")
            return False

    async def _record_failed_attempt(self, email: str, client_ip: str):
        """Record failed login attempt"""
        now = datetime.utcnow()

        if email not in self.failed_attempts:
            self.failed_attempts[email] = []

        self.failed_attempts[email].append(now)

        # Keep only recent attempts (last hour)
        cutoff = now - timedelta(hours=1)
        self.failed_attempts[email] = [
            attempt for attempt in self.failed_attempts[email]
            if attempt > cutoff
        ]

        # Check if should lock account
        if len(self.failed_attempts[email]) >= 5:
            self.blocked_accounts[email] = now + timedelta(minutes=30)

            await self._log_activity(
                event_type="account_locked",
                email=email,
                success=False,
                details={"reason": "too_many_failed_attempts", "ip": client_ip}
            )

    async def _is_account_locked(self, email: str) -> bool:
        """Check if account is locked"""
        if email not in self.blocked_accounts:
            return False

        if datetime.utcnow() > self.blocked_accounts[email]:
            del self.blocked_accounts[email]
            return False

        return True

    async def _clear_failed_attempts(self, email: str):
        """Clear failed attempts for email"""
        if email in self.failed_attempts:
            del self.failed_attempts[email]

    async def _log_activity(
        self,
        event_type: str,
        user_id: str = None,
        email: str = None,
        success: bool = True,
        details: Dict = None
    ):
        """Log authentication activity"""
        activity = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "email": email,
            "success": success,
            "details": details or {}
        }

        self.activity_log.append(activity)

        # Keep only recent activities (last 1000)
        if len(self.activity_log) > 1000:
            self.activity_log = self.activity_log[-1000:]

        # Log to system logger
        log_level = logging.INFO if success else logging.WARNING
        logger.log(
            log_level,
            f"Auth Activity: {event_type} - User: {user_id or email} - Success: {success}"
        )

    # Mock database methods (replace with actual database calls)
    async def _get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email (mock implementation)"""
        # In production, this would query the database
        return None

    async def _get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID (mock implementation)"""
        # In production, this would query the database
        return None

    async def _store_user(self, user_data: Dict):
        """Store user data (mock implementation)"""
        # In production, this would save to database
        pass

    async def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions (mock implementation)"""
        # In production, this would query user roles/permissions
        return ["user", "read", "write"]

    async def _update_user_2fa(
        self, 
        user_id: str, 
        enabled: bool, 
        secret: str = None, 
        backup_codes: List[str] = None
    ):
        """Update user 2FA settings (mock implementation)"""
        # In production, this would update database
        pass

    async def _update_user_backup_codes(self, user_id: str, backup_codes: List[str]):
        """Update user backup codes (mock implementation)"""
        # In production, this would update database
        pass

    async def _validate_registration_security(
        self, 
        request: RegisterRequest, 
        client_ip: str
    ):
        """Validate registration security"""
        # Add rate limiting, IP validation, etc.
        pass

    async def get_activity_log(
        self, 
        user_id: str = None, 
        limit: int = 100
    ) -> List[Dict]:
        """Get authentication activity log"""
        activities = self.activity_log

        if user_id:
            activities = [
                activity for activity in activities
                if activity.get("user_id") == user_id
            ]

        return activities[-limit:]

    async def get_active_sessions(self) -> Dict[str, Any]:
        """Get active sessions statistics"""
        now = datetime.utcnow()
        active_count = 0

        for session in self.active_sessions.values():
            last_activity = session["last_activity"]
            if (now - last_activity).total_seconds() < 3600:  # Active in last hour
                active_count += 1

        return {
            "total_sessions": len(self.active_sessions),
            "active_sessions": active_count,
            "session_types": {},
            "recent_logins": []
        }


# Global auth service instance
auth_service = NetflixLevelAuthService()