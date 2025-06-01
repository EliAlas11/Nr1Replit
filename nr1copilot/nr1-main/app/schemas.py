from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

# Base schemas
class BaseResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None

# Video schemas
class VideoProcessRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    start_time: Optional[float] = Field(None, description="Start time in seconds")
    end_time: Optional[float] = Field(None, description="End time in seconds")
    quality: Optional[str] = Field("720p", description="Video quality")

class VideoProcessResponse(BaseResponse):
    job_id: str
    status: str
    video_url: Optional[str] = None

# Analytics schemas
class AnalyticsRequest(BaseModel):
    event_type: str
    video_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class AnalyticsResponse(BaseResponse):
    event_id: str

# Feedback schemas
class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    video_id: Optional[str] = None

class FeedbackResponse(BaseResponse):
    feedback_id: str

# I18n schemas
class SetLanguageIn(BaseModel):
    language: str = Field(..., description="Language code (e.g., 'en', 'es')")

class SetLanguageOut(BaseResponse):
    language: str

class TranslationsOut(BaseResponse):
    translations: Dict[str, str]
    language: str

# Auth schemas
class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class AuthResponse(BaseResponse):
    token: str
    user_id: str
    email: str

# User schemas
class UserOut(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    created_at: datetime

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None