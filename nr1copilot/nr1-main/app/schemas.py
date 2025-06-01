from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(UserBase):
    id: str

# Auth schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: str
    exp: int

# Generic response
class Message(BaseModel):
    message: str

# Feedback schemas
class FeedbackIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    email: Optional[EmailStr] = None

class FeedbackOut(BaseModel):
    id: str
    message: str
    email: Optional[EmailStr] = None
    created_at: Optional[str] = None

class FeedbackList(BaseModel):
    feedback: list[FeedbackOut]

# Video schemas
class VideoValidateIn(BaseModel):
    url: str = Field(..., min_length=10)

class VideoValidateOut(BaseModel):
    valid: bool
    video_id: Optional[str] = None
    message: Optional[str] = None

class VideoInfoOut(BaseModel):
    video_id: str
    title: str
    duration: int
    thumbnail_url: Optional[str] = None

class VideoProcessIn(BaseModel):
    video_id: str
    start_time: int
    end_time: int
    aspect_ratio: Optional[str] = None
    captions: Optional[str] = None

class VideoProcessOut(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None

class VideoJobStatusOut(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    result_url: Optional[str] = None

class VideoServeOut(BaseModel):
    video_url: str
    message: Optional[str] = None

# Analytics schemas
class AnalyticsIn(BaseModel):
    event: str = Field(..., min_length=1, max_length=100)
    user_id: Optional[str] = None
    metadata: Optional[dict] = None

class AnalyticsOut(BaseModel):
    id: str
    event: str
    user_id: Optional[str] = None
    metadata: Optional[dict] = None
    timestamp: Optional[str] = None

class AnalyticsList(BaseModel):
    analytics: list[AnalyticsOut]

# I18N schemas
class SetLanguageIn(BaseModel):
    language: str = Field(..., min_length=2, max_length=10)

class SetLanguageOut(BaseModel):
    message: str

class TranslationsOut(BaseModel):
    translations: dict[str, str]
