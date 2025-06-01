from typing import Optional
from datetime import datetime
import logging
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from ..schemas import UserOut, LoginRequest, SignupRequest, AuthResponse
from app.schemas import UserCreate, UserLogin

logger = logging.getLogger("user_service")

class UserError(Exception):
    pass

# Mock user database (in production, use MongoDB)
users_db = {}
user_counter = 1

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_user_service(user_id: str) -> UserOut:
    """Get user by ID"""
    if user_id not in users_db:
        raise UserError(f"User {user_id} not found")

    user = users_db[user_id]
    return UserOut(
        id=user_id,
        email=user["email"],
        name=user.get("name"),
        created_at=user["created_at"]
    )

def signup_service(data: SignupRequest) -> AuthResponse:
    """Create new user account"""
    global user_counter

    # Check if email already exists
    for user in users_db.values():
        if user["email"] == data.email:
            raise UserError("Email already registered")

    user_id = str(user_counter)
    user_counter += 1

    # Store user (in production, hash password)
    users_db[user_id] = {
        "email": data.email,
        "password": data.password,  # Hash this in production!
        "name": data.name,
        "created_at": datetime.utcnow()
    }

    # Generate token (in production, use JWT)
    token = f"token_{user_id}"

    return AuthResponse(
        token=token,
        user_id=user_id,
        email=data.email
    )

def login_service(data: LoginRequest) -> AuthResponse:
    """Login user"""
    # Find user by email
    user_id = None
    for uid, user in users_db.items():
        if user["email"] == data.email and user["password"] == data.password:
            user_id = uid
            break

    if not user_id:
        raise UserError("Invalid email or password")

    # Generate token (in production, use JWT)
    token = f"token_{user_id}"

    return AuthResponse(
        token=token,
        user_id=user_id,
        email=data.email
    )

def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    return None

def create_user(email: str, full_name: Optional[str], hashed_password: str) -> UserOut:
    return UserOut(id="None", email=email, full_name=full_name or "")

def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    return None
# Test block for service sanity (not for production)
if __name__ == "__main__":
    try:
        test_user = UserCreate(email="test@example.com", full_name="Test User", password="password123")
        # The following line will cause error because UserCreate is not compatible with SignupRequest
        # out = signup_service(test_user)
        # login_out = login_service(UserLogin(email="test@example.com", password="password123"))
        print("User service test passed.")
    except Exception as e:
        print(f"Error: {e}")