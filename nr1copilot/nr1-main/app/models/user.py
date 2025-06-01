
"""
User database model and schemas
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String, DateTime, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pydantic import BaseModel, EmailStr, Field
import uuid

from app.db.session import Base


class User(Base):
    """User database model"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    video_quota = Column(Integer, default=10)  # Videos per month
    storage_used = Column(Integer, default=0)  # Bytes
    
    # Relationships
    videos = relationship("Video", back_populates="owner")
    feedback = relationship("Feedback", back_populates="user")


# Pydantic schemas
class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseModel):
    """User update schema"""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None


class UserOut(UserBase):
    """User output schema"""
    id: str
    is_active: bool
    is_verified: bool
    is_premium: bool
    created_at: datetime
    video_quota: int
    storage_used: int
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """User login schema"""
    email: EmailStr
    password: str


class Token(BaseModel):
    """JWT token schema"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data schema"""
    user_id: Optional[str] = None
