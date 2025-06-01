
"""
Pydantic Schemas
All request/response models for the API with proper validation
"""

from pydantic import BaseModel, EmailStr, field_validator, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# ============= Authentication Schemas =============

class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=1, max_length=100)
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    token: str
    refresh_token: str
    user_id: str
    email: str
    name: str
    expires_in: int

class TokenRefreshRequest(BaseModel):
    refresh_token: str

# ============= User Schemas =============

class UserOut(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime
    is_active: bool = True
    subscription_tier: str = "free"

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[EmailStr] = None

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)

# ============= Video Schemas =============

class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoQuality(str, Enum):
    LOW = "360p"
    MEDIUM = "720p"
    HIGH = "1080p"

class VideoProcessRequest(BaseModel):
    youtube_url: str = Field(..., pattern=r'^https?://(www\.)?(youtube\.com|youtu\.be)/.+')
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., gt=0)
    quality: VideoQuality = VideoQuality.MEDIUM
    audio_effect: Optional[str] = None
    
    @field_validator('end_time')
    @classmethod
    def validate_times(cls, v, info):
        if info.data.get('start_time') is not None and v <= info.data['start_time']:
            raise ValueError('End time must be greater than start time')
        if v - info.data.get('start_time', 0) > 300:  # Max 5 minutes
            raise ValueError('Clip duration cannot exceed 5 minutes')
        return v

class VideoOut(BaseModel):
    id: str
    user_id: str
    youtube_url: str
    title: Optional[str] = None
    description: Optional[str] = None
    start_time: float
    end_time: float
    duration: float
    quality: str
    status: VideoStatus
    file_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    file_size: Optional[int] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class VideoListResponse(BaseModel):
    videos: List[VideoOut]
    total: int
    page: int
    per_page: int
    total_pages: int

# ============= Analytics Schemas =============

class AnalyticsEvent(BaseModel):
    event_type: str = Field(..., pattern=r'^[a-z_]+$')
    properties: Optional[Dict[str, Any]] = {}
    video_id: Optional[str] = None

class AnalyticsOut(BaseModel):
    user_id: str
    total_videos: int
    total_duration: float
    most_used_quality: str
    avg_processing_time: float
    success_rate: float
    last_activity: datetime

# ============= Feedback Schemas =============

class FeedbackType(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    GENERAL = "general"
    COMPLAINT = "complaint"

class FeedbackRequest(BaseModel):
    type: FeedbackType
    subject: str = Field(..., min_length=5, max_length=200)
    message: str = Field(..., min_length=10, max_length=2000)
    email: Optional[EmailStr] = None

class FeedbackOut(BaseModel):
    id: str
    user_id: Optional[str] = None
    type: FeedbackType
    subject: str
    message: str
    email: Optional[str] = None
    status: str = "pending"
    created_at: datetime

# ============= I18n Schemas =============

class TranslationRequest(BaseModel):
    key: str
    language: str = Field(..., pattern=r'^[a-z]{2}$')

class TranslationOut(BaseModel):
    key: str
    language: str
    value: str

class LanguageOut(BaseModel):
    code: str
    name: str
    native_name: str
    is_active: bool

# ============= Response Schemas =============

class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Any] = None

class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_code: Optional[str] = None
    details: Optional[Any] = None

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]
    uptime: float

# ============= Job Schemas =============

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobOut(BaseModel):
    id: str
    type: str
    status: JobStatus
    progress: int = Field(..., ge=0, le=100)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
