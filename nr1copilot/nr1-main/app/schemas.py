
"""
Pydantic schemas for ViralClip Pro
Netflix-level data validation and serialization
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class QualityEnum(str, Enum):
    """Video quality options"""
    LOW = "480p"
    MEDIUM = "720p"
    HIGH = "1080p"
    ULTRA = "1440p"
    HIGHEST = "4k"

class AspectRatioEnum(str, Enum):
    """Aspect ratio options"""
    VERTICAL = "9:16"
    HORIZONTAL = "16:9"
    SQUARE = "1:1"
    STORY = "4:5"

class PriorityEnum(str, Enum):
    """Processing priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

# Request schemas
class VideoProcessRequest(BaseModel):
    """Video processing request model"""
    youtube_url: str = Field(..., description="YouTube or video URL")
    start_time: float = Field(default=0, ge=0, description="Start time in seconds")
    end_time: float = Field(..., gt=0, description="End time in seconds")
    quality: QualityEnum = Field(default=QualityEnum.HIGH)
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.VERTICAL)
    enable_captions: bool = Field(default=True)
    enable_transitions: bool = Field(default=True)
    ai_editing: bool = Field(default=True)
    viral_optimization: bool = Field(default=True)
    language: str = Field(default="en", regex="^[a-z]{2}$")
    priority: PriorityEnum = Field(default=PriorityEnum.NORMAL)
    webhook_url: Optional[str] = Field(None, description="Callback URL")
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v
    
    @validator('youtube_url')
    def validate_url(cls, v):
        if not v or len(v) < 10:
            raise ValueError('Invalid URL provided')
        return v

class ClipSettings(BaseModel):
    """Individual clip settings"""
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., gt=0)
    title: str = Field(default="", max_length=100)
    description: str = Field(default="", max_length=500)
    tags: List[str] = Field(default=[])
    quality: QualityEnum = Field(default=QualityEnum.HIGH)
    aspect_ratio: AspectRatioEnum = Field(default=AspectRatioEnum.VERTICAL)

class AnalyzeRequest(BaseModel):
    """Video analysis request"""
    url: str = Field(..., description="Video URL to analyze")
    language: str = Field(default="en")
    viral_optimization: bool = Field(default=True)

# Response schemas
class SuccessResponse(BaseModel):
    """Standard success response"""
    success: bool = Field(default=True)
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = Field(default=False)
    error: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class VideoInfo(BaseModel):
    """Video information model"""
    title: str
    duration: int
    thumbnail: Optional[str] = None
    uploader: Optional[str] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    description: Optional[str] = None
    upload_date: Optional[str] = None
    categories: List[str] = Field(default=[])
    tags: List[str] = Field(default=[])

class AIInsights(BaseModel):
    """AI analysis insights"""
    viral_potential: int = Field(..., ge=0, le=100)
    engagement_prediction: int = Field(..., ge=0, le=100)
    best_clips: List[Dict[str, Any]] = Field(default=[])
    suggested_formats: List[str] = Field(default=[])
    recommended_captions: bool = Field(default=True)
    optimal_length: int = Field(..., gt=0)
    trending_topics: List[str] = Field(default=[])
    sentiment_analysis: str = Field(default="neutral")
    hook_moments: List[Dict[str, Any]] = Field(default=[])
    emotional_peaks: List[Dict[str, Any]] = Field(default=[])

class AnalysisResponse(BaseModel):
    """Video analysis response"""
    success: bool = Field(default=True)
    session_id: str
    video_info: VideoInfo
    ai_insights: AIInsights
    processing_time: float
    cache_hit: bool = Field(default=False)

class ProcessingResponse(BaseModel):
    """Processing initiation response"""
    success: bool = Field(default=True)
    task_id: str
    message: str
    estimated_time: int
    priority: str
    position_in_queue: int

class ClipResult(BaseModel):
    """Individual clip processing result"""
    clip_index: int
    file_path: str
    title: str
    duration: float
    viral_score: int = Field(..., ge=0, le=100)
    file_size: int
    thumbnail: Optional[str] = None
    ai_enhancements: List[str] = Field(default=[])
    optimization_applied: List[str] = Field(default=[])

class ProcessingStatus(BaseModel):
    """Processing status response"""
    success: bool = Field(default=True)
    data: Dict[str, Any]

class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(default="healthy")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="3.0.0")
    uptime: Optional[float] = None

class MetricsResponse(BaseModel):
    """Metrics response"""
    requests_total: int = Field(default=0)
    requests_per_minute: float = Field(default=0.0)
    average_response_time: float = Field(default=0.0)
    active_connections: int = Field(default=0)
    queue_size: int = Field(default=0)
    cache_hit_rate: float = Field(default=0.0)

# WebSocket message schemas
class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any]

class ProgressUpdate(BaseModel):
    """Progress update message"""
    progress: int = Field(..., ge=0, le=100)
    current_step: str
    message: str
    estimated_remaining: Optional[int] = None

# Database models (for future MongoDB integration)
class UserModel(BaseModel):
    """User model for database"""
    id: Optional[str] = None
    username: str
    email: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(default=True)
    subscription_tier: str = Field(default="free")

class VideoModel(BaseModel):
    """Video model for database"""
    id: Optional[str] = None
    user_id: str
    original_url: str
    title: str
    duration: int
    status: str = Field(default="processing")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default={})

# Export commonly used models
VideoOut = VideoInfo
