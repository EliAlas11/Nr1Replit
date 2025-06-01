"""
User Service Layer (Refactored)

- Handles user creation, retrieval, and lookup logic.
- All business logic, validation, and error handling for user management is centralized here.
- Designed for auditability, security, and testability (Stripe/Netflix standards).
- All functions are stateless and side-effect free except for user creation.
- Implements audit logging and custom exceptions for compliance.
- TODO: Replace in-memory store with persistent database for production.
"""

import logging
from typing import Optional, Dict, Any
from app.schemas import UserCreate, UserLogin, UserOut
from passlib.context import CryptContext

logger = logging.getLogger("user_service")

class UserError(Exception):
    """Custom exception for user service errors."""
    pass

_fake_users_db: dict[str, dict[str, Any]] = {}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    """
    Retrieve a user by email.

    Args:
        email (str): The user's email.

    Returns:
        dict[str, Any] | None: User dict if found, None otherwise.
    """
    user = next((u for u in _fake_users_db.values() if u["email"] == email), None)
    logger.info(f"AUDIT: get_user_by_email: {email} found={user is not None}")
    return user

def create_user(email: str, full_name: Optional[str], hashed_password: str) -> UserOut:
    """
    Create a new user and add to the in-memory store.

    Args:
        email (str): The user's email.
        full_name (str | None): The user's full name.
        hashed_password (str): The hashed password.

    Returns:
        UserOut: The created user object.
    """
    user_id = str(len(_fake_users_db) + 1)
    user = {"id": user_id, "email": email, "full_name": full_name or "", "hashed_password": hashed_password}
    _fake_users_db[user_id] = user
    logger.info(f"AUDIT: create_user: {email} id={user_id}")
    return UserOut(id=user_id, email=email, full_name=full_name or "")

def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    """
    Retrieve a user by user ID.

    Args:
        user_id (str): The user's ID.

    Returns:
        dict[str, Any] | None: User dict if found, None otherwise.
    """
    user = _fake_users_db.get(user_id)
    logger.info(f"AUDIT: get_user_by_id: {user_id} found={user is not None}")
    return user

def signup_service(user: UserCreate) -> UserOut:
    """
    Register a new user.

    Args:
        user (UserCreate): Signup data.

    Returns:
        UserOut: The registered user object.

    Raises:
        UserError: If the user already exists.
    """
    if get_user_by_email(user.email):
        logger.warning(f"AUDIT: signup_service: User already exists: {user.email}")
        raise UserError("User already exists")
    hashed_password = pwd_context.hash(user.password)
    out = create_user(user.email, user.full_name, hashed_password)
    logger.info(f"AUDIT: signup_service: User created: {user.email}")
    return out

def login_service(user: UserLogin) -> UserOut:
    """
    Authenticate a user and return user info.

    Args:
        user (UserLogin): Login credentials.

    Returns:
        UserOut: The authenticated user object.

    Raises:
        UserError: If the credentials are invalid.
    """
    db_user = get_user_by_email(user.email)
    if not db_user or not pwd_context.verify(user.password, db_user["hashed_password"]):
        logger.warning(f"AUDIT: login_service: Invalid credentials for {user.email}")
        raise UserError("Invalid credentials")
    logger.info(f"AUDIT: login_service: User authenticated: {user.email}")
    return UserOut(id=db_user["id"], email=db_user["email"], full_name=db_user["full_name"])

def get_user_service(user_id: str) -> UserOut:
    """
    Retrieve a user by user ID. Raises UserError if not found.

    Args:
        user_id (str): The user's ID.

    Returns:
        UserOut: The user object.

    Raises:
        UserError: If user is not found.
    """
    user = get_user_by_id(user_id)
    if not user:
        logger.warning(f"AUDIT: get_user_service: User not found: {user_id}")
        raise UserError(f"User with id {user_id} not found")
    logger.info(f"AUDIT: get_user_service: User found: {user_id}")
    return UserOut(id=user["id"], email=user["email"], full_name=user["full_name"])

# Test block for service sanity (not for production)
if __name__ == "__main__":
    try:
        test_user = UserCreate(email="test@example.com", full_name="Test User", password="password123")
        out = signup_service(test_user)
        assert out.email == "test@example.com"
        login_out = login_service(UserLogin(email="test@example.com", password="password123"))
        assert login_out.email == "test@example.com"
        print("User service test passed.")
    except Exception as e:
        print(f"Error: {e}")

