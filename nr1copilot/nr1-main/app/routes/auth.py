from fastapi import APIRouter, HTTPException, Body
from app.schemas import UserCreate, UserLogin, Token, Message
from ..controllers.auth_controller import signup, login, refresh_token

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])

@router.post("/signup", response_model=Message)
def signup_route(user: UserCreate):
    result = signup(user)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return Message(message=result["message"])

@router.post("/login", response_model=Token)
def login_route(user: UserLogin):
    result = login(user)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=401, detail=result["error"])
    return Token(**result)

@router.post("/refresh", response_model=Token)
def refresh_route(token: str = Body(..., embed=True)):
    result = refresh_token(token)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=401, detail=result["error"])
    return Token(**result)
