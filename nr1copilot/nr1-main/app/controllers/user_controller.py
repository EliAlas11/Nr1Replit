from fastapi import HTTPException
from app.schemas import UserCreate, UserLogin, UserOut
from typing import Any
from ..services.user_service import (
    signup_service,
    login_service,
    get_user_service,
    UserError,
)

def signup(user: UserCreate) -> UserOut:
    """
    Register a new user. Raises HTTPException on error.
    """
    try:
        return signup_service(user)
    except UserError as e:
        raise HTTPException(status_code=400, detail=str(e))

def login(user: UserLogin) -> UserOut:
    """
    Authenticate a user and return user info. Raises HTTPException on error.
    """
    try:
        return login_service(user)
    except UserError as e:
        raise HTTPException(status_code=401, detail=str(e))

from fastapi import HTTPException
from app.schemas import UserOut, LoginRequest, SignupRequest, AuthResponse
from ..services.user_service import (
    get_user_service,
    signup_service, 
    login_service,
    UserError,
)

def signup(data: SignupRequest) -> AuthResponse:
    """
    Create new user account. Raises HTTPException on error.
    """
    try:
        return signup_service(data)
    except UserError as e:
        raise HTTPException(status_code=400, detail=str(e))

def login(data: LoginRequest) -> AuthResponse:
    """
    Login user. Raises HTTPException on error.
    """
    try:
        return login_service(data)
    except UserError as e:
        raise HTTPException(status_code=401, detail=str(e))

def get_user(user_id: str) -> UserOut:
    """
    Retrieve a user by user ID. Raises HTTPException on error.
    """
    try:
        return get_user_service(user_id)
    except UserError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Minimal test/assertion to confirm endpoint signature compiles
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    # Sanity-check POST /api/v1/user/signup returns 200 and has id in response
    resp = client.post("/api/v1/user/signup", json={"email": "test@example.com", "password": "secret"})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data and "email" in data
    # Sanity-check login
    resp2 = client.post("/api/v1/user/login", json={"email": "test@example.com", "password": "secret"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "id" in data2 and "email" in data2
