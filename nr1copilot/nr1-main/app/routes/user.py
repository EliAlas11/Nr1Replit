
<old_str></old_str>
<new_str>"""
User Management Routes
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer
import logging
from ..schemas import UserOut, UserUpdate, PasswordChangeRequest, SuccessResponse

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter()

@router.get("/user/profile", response_model=UserOut)
async def get_profile(token: str = Depends(security)):
    """Get user profile"""
    try:
        # Mock user profile
        from datetime import datetime
        return UserOut(
            id="user_123",
            email="mock@example.com",
            name="Mock User",
            created_at=datetime.now(),
            is_active=True,
            subscription_tier="free"
        )
    except Exception as e:
        logger.error(f"Profile error: {e}")
        raise HTTPException(status_code=404, detail="User not found")

@router.put("/user/profile", response_model=SuccessResponse)
async def update_profile(update: UserUpdate, token: str = Depends(security)):
    """Update user profile"""
    try:
        return SuccessResponse(message="Profile updated successfully")
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(status_code=400, detail="Update failed")

@router.post("/user/change-password", response_model=SuccessResponse)
async def change_password(request: PasswordChangeRequest, token: str = Depends(security)):
    """Change user password"""
    try:
        return SuccessResponse(message="Password changed successfully")
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(status_code=400, detail="Password change failed")</new_str>
