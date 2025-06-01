"""
Authentication Service Layer

- Handles user signup, login, password hashing, and JWT token management.
- All business logic, validation, and error handling for authentication is centralized here.
- Designed for auditability, security, and testability (Stripe/Netflix standards).
- All functions are stateless and side-effect free except for user creation.
- TODO: Add audit logging and security hooks for compliance.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Dict
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.schemas import UserCreate, UserLogin
from app.services.user_service import get_user_by_email, create_user
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY: str = settings.JWT_SECRET
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a hashed password.

    Args:
        plain_password (str): The plaintext password.
        hashed_password (str): The hashed password.

    Returns:
        bool: True if the password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.

    Args:
        password (str): The plaintext password.

    Returns:
        str: The hashed password.
    """
    return pwd_context.hash(password)

def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user by email and password.

    Args:
        email (str): The user's email.
        password (str): The plaintext password.

    Returns:
        dict[str, Any] | None: User dict if authenticated, None otherwise.
    """
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data (dict): Data to encode in the token (should include 'sub').
        expires_delta (timedelta, optional): Expiry duration. Defaults to ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        str: Encoded JWT token.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def signup_service(user_in: UserCreate) -> dict[str, Any]:
    """
    Register a new user.

    Args:
        user_in (UserCreate): Signup data.

    Returns:
        dict[str, Any]: On success: {"message": str, "user": dict}. On error: {"error": str}.
    """
    # TODO: Add audit logging for signup events
    user = get_user_by_email(user_in.email)
    if user:
        return {"error": "User already exists"}
    hashed_password = get_password_hash(user_in.password)
    user = create_user(email=user_in.email, full_name=user_in.full_name, hashed_password=hashed_password)
    return {"message": "User signed up", "user": user}

def login_service(user_in: UserLogin) -> dict[str, Any]:
    """
    Authenticate a user and return a JWT token.

    Args:
        user_in (UserLogin): Login credentials.

    Returns:
        dict[str, Any]: On success: {"access_token": str, "token_type": str}. On error: {"error": str}.
    """
    # TODO: Add audit logging for login events
    user = authenticate_user(user_in.email, user_in.password)
    if not user:
        return {"error": "Invalid credentials"}
    access_token = create_access_token({"sub": user["id"]})
    return {"access_token": access_token, "token_type": "bearer"}

def refresh_token_service(token: str) -> dict[str, Any]:
    """
    Refresh a JWT token for an authenticated user.

    Args:
        token (str): The refresh token (JWT).

    Returns:
        dict[str, Any]: On success: {"access_token": str, "token_type": str}. On error: {"error": str}.
    """
    # TODO: Add audit logging for token refresh events
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            return {"error": "Invalid token"}
        new_token = create_access_token({"sub": user_id})
        return {"access_token": new_token, "token_type": "bearer"}
    except JWTError:
        return {"error": "Invalid token"}
