"""
Authentication Controller: Stateless, auditable interface between FastAPI routes and authentication services.

- Delegates all business logic, validation, and error handling to the service layer.
- Ensures controllers remain thin, testable, and maintainable.
- Designed for world-class auditability and clarity (Stripe/Netflix standards).
- All controller actions are ready for audit logging and compliance review.

Compliance Note:
    All authentication events are logged for audit/compliance. See audit_log().
    TODO: Integrate with centralized logging and security event monitoring.
"""

from typing import Any
from app.schemas import UserCreate, UserLogin, Token, Message
from fastapi import HTTPException
import logging
from ..services.auth_service import (
    login_service,
    refresh_token_service,
    signup_service,
)

# Use Python's logging for audit and error logs
logger = logging.getLogger("auth_audit")

# Use Pydantic models for API responses (highest-impact change)
# Remove AuthResponse dict and use Token/Message from schemas.py

def audit_log(event: str, details: dict[str, Any], status: str = "success", context: dict[str, Any] = {}) -> None:
    """
    Audit logging for authentication events. Integrate with production audit/compliance system.
    Always include event, user (if available), status, and details for audit traceability.

    Args:
        event (str): Event name or type.
        details (dict[str, Any]): Event details for audit/compliance.
        status (str): 'success' or 'error'.
        context (dict[str, Any]): Optional request context (IP, user agent, etc.)
    """
    log_entry: dict[str, Any] = {
        "event": event,
        "status": status,
        "details": details,
        "context": context,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat() + 'Z',
    }
    logger.info(f"AUDIT: {log_entry}")
    # TODO: Integrate with SIEM, external audit log, or compliance system


def signup(user: UserCreate) -> Message:
    """
    Register a new user. Returns Message Pydantic model.
    Raises HTTPException on error for FastAPI consistency.
    """
    user_email = getattr(user, "email", None)
    try:
        result = signup_service(user)
        audit_log("user_signup", {"user": user_email, "result": result}, status="success")
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return Message(message=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        audit_log("user_signup_error", {"user": user_email, "error": str(e)}, status="error")
        raise HTTPException(status_code=500, detail="Signup failed. Please try again later.")


def login(user: UserLogin) -> Token:
    """
    Authenticate a user and return a JWT token. Returns Token Pydantic model.
    Raises HTTPException on error for FastAPI consistency.
    """
    user_email = getattr(user, "email", None)
    try:
        result = login_service(user)
        audit_log("user_login", {"user": user_email, "result": result}, status="success")
        if "error" in result:
            raise HTTPException(status_code=401, detail=result["error"])
        return Token(**result)
    except HTTPException:
        raise
    except Exception as e:
        audit_log("user_login_error", {"user": user_email, "error": str(e)}, status="error")
        raise HTTPException(status_code=500, detail="Login failed. Please try again later.")


def refresh_token(token: str) -> Token:
    """
    Refresh a JWT token for an authenticated user. Returns Token Pydantic model.
    Raises HTTPException on error for FastAPI consistency.
    """
    try:
        result = refresh_token_service(token)
        audit_log("refresh_token", {"token": token, "result": result}, status="success")
        if "error" in result:
            raise HTTPException(status_code=401, detail=result["error"])
        return Token(**result)
    except HTTPException:
        raise
    except Exception as e:
        audit_log("refresh_token_error", {"token": token, "error": str(e)}, status="error")
        raise HTTPException(status_code=500, detail="Token refresh failed. Please try again later.")
