
<old_str></old_str>
<new_str>"""
Authentication Routes
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer
import logging
from ..schemas import SignupRequest, LoginRequest, AuthResponse, SuccessResponse

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter()

@router.post("/auth/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    """User registration"""
    try:
        # Mock signup response
        return AuthResponse(
            token="mock_access_token",
            refresh_token="mock_refresh_token",
            user_id="user_123",
            email=request.email,
            name=request.name,
            expires_in=1800
        )
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=400, detail="Registration failed")

@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """User login"""
    try:
        # Mock login response
        return AuthResponse(
            token="mock_access_token",
            refresh_token="mock_refresh_token",
            user_id="user_123",
            email=request.email,
            name="Mock User",
            expires_in=1800
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

@router.post("/auth/logout", response_model=SuccessResponse)
async def logout(token: str = Depends(security)):
    """User logout"""
    return SuccessResponse(message="Logged out successfully")

@router.post("/auth/refresh", response_model=AuthResponse)
async def refresh_token(request: dict):
    """Refresh access token"""
    try:
        return AuthResponse(
            token="new_mock_access_token",
            refresh_token="new_mock_refresh_token",
            user_id="user_123",
            email="mock@example.com",
            name="Mock User",
            expires_in=1800
        )
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=401, detail="Invalid refresh token")</new_str>
