"""
Netflix-Level Authentication Middleware v10.0
Role-based access control and session management
"""

import logging
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.services.auth_service import auth_service
from app.utils.security import SecurityManager

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication and authorization middleware"""

    def __init__(self, app):
        super().__init__(app)
        self.security_manager = SecurityManager()

        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/health",
            "/api/v10/auth/login",
            "/api/v10/auth/register",
            "/api/v10/auth/refresh",
            "/api/docs",
            "/api/redoc",
            "/openapi.json",
            "/static",
            "/favicon.ico"
        }

        # Admin-only endpoints
        self.admin_endpoints = {
            "/api/v10/admin",
            "/api/v10/enterprise/admin",
            "/api/v10/system/admin"
        }

        logger.info("Authentication middleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through authentication checks"""
        try:
            # Skip authentication for public endpoints
            if self._is_public_endpoint(request.url.path):
                return await call_next(request)

            # Extract and validate token
            auth_result = await self._authenticate_request(request)
            if not auth_result["success"]:
                return JSONResponse(
                    {"error": auth_result["error"]},
                    status_code=auth_result["status_code"]
                )

            # Check role-based permissions
            permission_result = await self._check_permissions(request, auth_result["user_data"])
            if not permission_result["success"]:
                return JSONResponse(
                    {"error": permission_result["error"]},
                    status_code=permission_result["status_code"]
                )

            # Add user data to request state
            request.state.user = auth_result["user_data"]
            request.state.authenticated = True

            # Process request
            response = await call_next(request)

            # Add security headers
            self._add_auth_headers(response, auth_result["user_data"])

            return response

        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            return JSONResponse(
                {"error": "Authentication failed"},
                status_code=500
            )

    async def _authenticate_request(self, request: Request) -> Dict:
        """Authenticate request and extract user data"""
        try:
            # Extract token from Authorization header
            authorization = request.headers.get("authorization")
            if not authorization:
                return {
                    "success": False,
                    "error": "Authorization header required",
                    "status_code": 401
                }

            if not authorization.startswith("Bearer "):
                return {
                    "success": False,
                    "error": "Invalid authorization format",
                    "status_code": 401
                }

            token = authorization.split(" ")[1]

            # Validate token
            from fastapi.security import HTTPAuthorizationCredentials
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=token
            )

            user_data = await auth_service.validate_token(credentials)

            return {
                "success": True,
                "user_data": user_data
            }

        except HTTPException as e:
            return {
                "success": False,
                "error": e.detail,
                "status_code": e.status_code
            }
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return {
                "success": False,
                "error": "Invalid authentication",
                "status_code": 401
            }

    async def _check_permissions(self, request: Request, user_data: Dict) -> Dict:
        """Check role-based permissions"""
        try:
            path = request.url.path
            method = request.method
            user_permissions = user_data.get("permissions", [])

            # Admin endpoint check
            if any(path.startswith(endpoint) for endpoint in self.admin_endpoints):
                if "admin" not in user_permissions:
                    await self._log_access_denied(
                        user_data["user_id"],
                        path,
                        "admin_required"
                    )
                    return {
                        "success": False,
                        "error": "Administrator access required",
                        "status_code": 403
                    }

            # API endpoint permissions
            if path.startswith("/api/v10/"):
                required_permission = self._get_required_permission(path, method)
                if required_permission and required_permission not in user_permissions:
                    await self._log_access_denied(
                        user_data["user_id"],
                        path,
                        f"missing_permission_{required_permission}"
                    )
                    return {
                        "success": False,
                        "error": f"Permission '{required_permission}' required",
                        "status_code": 403
                    }

            # Rate limiting per user
            rate_limit_result = await self._check_user_rate_limit(
                user_data["user_id"],
                path
            )
            if not rate_limit_result["allowed"]:
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "status_code": 429
                }

            return {"success": True}

        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return {
                "success": False,
                "error": "Permission check failed",
                "status_code": 500
            }

    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public"""
        return any(
            path.startswith(endpoint) 
            for endpoint in self.public_endpoints
        )

    def _get_required_permission(self, path: str, method: str) -> Optional[str]:
        """Get required permission for endpoint"""
        # Video endpoints
        if path.startswith("/api/v10/video"):
            if method in ["GET"]:
                return "video_read"
            elif method in ["POST", "PUT", "PATCH"]:
                return "video_write"
            elif method in ["DELETE"]:
                return "video_delete"

        # Project endpoints
        elif path.startswith("/api/v10/project"):
            if method in ["GET"]:
                return "project_read"
            elif method in ["POST", "PUT", "PATCH"]:
                return "project_write"
            elif method in ["DELETE"]:
                return "project_delete"

        # Analytics endpoints
        elif path.startswith("/api/v10/analytics"):
            return "analytics_read"

        # Enterprise endpoints
        elif path.startswith("/api/v10/enterprise"):
            return "enterprise_access"

        # Default permission for API access
        return "api_access"

    async def _check_user_rate_limit(
        self, 
        user_id: str, 
        path: str
    ) -> Dict:
        """Check user-specific rate limiting"""
        try:
            # Different limits for different endpoint types
            if path.startswith("/api/v10/video/upload"):
                limit = 10  # 10 uploads per hour
                window = 3600
            elif path.startswith("/api/v10/ai/"):
                limit = 100  # 100 AI requests per hour
                window = 3600
            else:
                limit = 1000  # 1000 general requests per hour
                window = 3600

            # Use security manager's rate limiting
            from app.utils.rate_limiter import rate_limiter, RateLimitScope

            result = await rate_limiter.is_allowed(
                identifier=f"user:{user_id}",
                scope=RateLimitScope.USER,
                endpoint=path
            )

            return {
                "allowed": result.allowed,
                "remaining": result.remaining,
                "reset_time": result.reset_time
            }

        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return {"allowed": True}  # Fail open

    async def _log_access_denied(
        self, 
        user_id: str, 
        path: str, 
        reason: str
    ):
        """Log access denied events"""
        await auth_service._log_activity(
            event_type="access_denied",
            user_id=user_id,
            success=False,
            details={
                "path": path,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    def _add_auth_headers(self, response: Response, user_data: Dict):
        """Add authentication-related headers"""
        response.headers["X-User-ID"] = user_data["user_id"]
        response.headers["X-Session-Type"] = user_data.get("session_type", "web")
        response.headers["X-Auth-Time"] = str(int(time.time()))