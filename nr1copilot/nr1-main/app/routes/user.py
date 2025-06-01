from fastapi import APIRouter, HTTPException
from app.schemas import UserCreate, UserLogin, UserOut, Message
from ..controllers.user_controller import signup, login, get_user

router = APIRouter(prefix="/api/v1/user", tags=["User"])

@router.post("/signup", response_model=Message)
def signup_route(user: UserCreate):
    result = signup(user)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return Message(message=result["message"])

@router.post("/login", response_model=UserOut)
def login_route(user: UserLogin):
    result = login(user)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=401, detail=result["error"])
    return UserOut(**result["user"])

@router.get("/{user_id}", response_model=UserOut)
def get_user_route(user_id: str):
    result = get_user(user_id)
    if not result or not result.get("user"):
        raise HTTPException(status_code=404, detail="User not found")
    return UserOut(**result["user"])
