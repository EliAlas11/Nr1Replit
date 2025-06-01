"""
ViralClip Pro - Pydantic Schema Definitions
Comprehensive data models for API validation and documentation
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator

# Enums for controlled values
class VideoQuality(str, Enum):
    """Video quality options"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class ProcessingStatus(str, Enum):
    """Processing status options"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Platform(str, Enum):
    """Target platform options"""
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    GENERAL = "general"

# Base models
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class SuccessResponse(BaseResponse):
    """Success response model"""
    success: bool = True
    data: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Video-related models
class VideoInfo(BaseModel):
    """Video information model"""
    id: str
    title: str
    duration: float = Field(..., description="Duration in seconds")
    thumbnail: Optional[str] = None
    uploader: Optional[str] = None
    upload_date: Optional[str] = None
    view_count: Optional[int] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None

    @validator('duration')
    def validate_duration(cls, v):
        if v <= 0:
            raise ValueError('Duration must be positive')
        return v

class VideoValidationRequest(BaseModel):
    """Video URL validation request"""
    url: str = Field(..., min_length=1, max_length=2048)

    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class VideoAnalysisRequest(BaseModel):
    """Video analysis request model"""
    url: str = Field(..., description="Video URL to analyze")
    clip_duration: Optional[float] = Field(default=30.0, description="Target clip duration")
    suggested_formats: Optional[List[Platform]] = Field(default=[Platform.GENERAL])
    viral_optimization: bool = Field(default=False, description="Enable viral optimization")

    @validator('clip_duration')
    def validate_clip_duration(cls, v):
        if v and (v < 5 or v > 300):
            raise ValueError('Clip duration must be between 5 and 300 seconds')
        return v

class ClipDefinition(BaseModel):
    """Clip definition for processing"""
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., gt=0, description="End time in seconds")
    title: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    quality: VideoQuality = Field(default=VideoQuality.HIGH)
    platform: Platform = Field(default=Platform.GENERAL)

    @validator('end_time')
    def validate_times(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be greater than start time')
        if 'start_time' in values and (v - values['start_time']) > 300:
            raise ValueError('Clip duration cannot exceed 300 seconds')
        return v

class VideoProcessRequest(BaseModel):
    """Video processing request"""
    youtube_url: str = Field(..., description="YouTube video URL")
    clips: List[ClipDefinition] = Field(..., min_items=1, max_items=10)
    priority: str = Field(default="normal", description="Processing priority")

    @validator('youtube_url')
    def validate_youtube_url(cls, v):
        youtube_patterns = ['youtube.com', 'youtu.be']
        if not any(pattern in v.lower() for pattern in youtube_patterns):
            raise ValueError('Must be a valid YouTube URL')
        return v

class ProcessingResult(BaseModel):
    """Processing result for a single clip"""
    clip_index: int
    success: bool
    output_path: Optional[str] = None
    file_size: Optional[int] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

class TaskStatus(BaseModel):
    """Task status information"""
    task_id: str
    status: ProcessingStatus
    progress: float = Field(ge=0, le=100, description="Progress percentage")
    created_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None
    results: Optional[List[ProcessingResult]] = None

class VideoAnalysisResponse(BaseResponse):
    """Video analysis response"""
    success: bool = True
    session_id: str
    video_info: VideoInfo
    ai_insights: Dict[str, Any]
    suggested_clips: List[ClipDefinition]
    processing_time: float

class AIInsights(BaseModel):
    """AI analysis insights"""
    sentiment_score: Optional[float] = Field(ge=-1, le=1)
    engagement_score: Optional[float] = Field(ge=0, le=1)
    viral_potential: Optional[float] = Field(ge=0, le=1)
    key_moments: Optional[List[Dict[str, Any]]] = None
    transcription: Optional[str] = None
    topics: Optional[List[str]] = None
    emotions: Optional[Dict[str, float]] = None

    @validator('sentiment_score', 'engagement_score', 'viral_potential')
    def validate_scores(cls, v):
        if v is not None and not (-1 <= v <= 1):
            raise ValueError('Score must be between -1 and 1')
        return v

class UploadInfo(BaseModel):
    """File upload information"""
    filename: str
    file_size: int
    content_type: str
    upload_id: str
    session_id: str

    @validator('file_size')
    def validate_file_size(cls, v):
        if v <= 0:
            raise ValueError('File size must be positive')
        if v > 2 * 1024 * 1024 * 1024:  # 2GB
            raise ValueError('File size exceeds maximum limit')
        return v

class VideoOut(BaseModel):
    """Video output model"""
    id: str
    title: str
    duration: float
    status: ProcessingStatus
    created_at: datetime
    file_size: Optional[int] = None
    quality: Optional[VideoQuality] = None
    thumbnail: Optional[str] = None

class VideoListResponse(BaseResponse):
    """Video list response"""
    success: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)
    pagination: Optional[Dict[str, Any]] = None

class PaginationInfo(BaseModel):
    """Pagination information"""
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    has_more: bool

class HealthStatus(BaseModel):
    """Health status model"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime
    version: str
    environment: str
    system: Optional[Dict[str, Any]] = None
    services: Optional[Dict[str, Any]] = None

class MetricsData(BaseModel):
    """Metrics data model"""
    total_requests: int = Field(ge=0)
    active_connections: int = Field(ge=0)
    processing_queue_size: int = Field(ge=0)
    average_processing_time: float = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    uptime_seconds: float = Field(ge=0)

class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ProgressUpdate(BaseModel):
    """Progress update model"""
    task_id: str
    progress: float = Field(ge=0, le=100)
    stage: str
    message: Optional[str] = None
    estimated_remaining: Optional[float] = None

class SecurityEvent(BaseModel):
    """Security event model"""
    event_type: str
    severity: str
    details: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None

# Configuration models
class ProcessingSettings(BaseModel):
    """Processing configuration settings"""
    quality: VideoQuality = VideoQuality.HIGH
    max_duration: int = Field(default=300, ge=5, le=3600)
    enable_ai_enhancement: bool = Field(default=False)
    target_platforms: List[Platform] = Field(default=[Platform.GENERAL])
    viral_optimization: bool = Field(default=False)

class CacheSettings(BaseModel):
    """Cache configuration settings"""
    ttl: int = Field(default=3600, ge=60)
    max_size: int = Field(default=1000, ge=10)
    enabled: bool = Field(default=True)

# Error models
class ValidationError(BaseModel):
    """Validation error model"""
    field: str
    message: str
    value: Any

class APIError(BaseModel):
    """API error model"""
    error_code: str
    message: str
    details: Optional[List[ValidationError]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Batch processing models
class BatchProcessRequest(BaseModel):
    """Batch processing request"""
    videos: List[VideoProcessRequest] = Field(..., min_items=1, max_items=5)
    priority: str = Field(default="normal")
    callback_url: Optional[str] = None

class BatchProcessResponse(BaseResponse):
    """Batch processing response"""
    success: bool = True
    batch_id: str
    total_videos: int
    estimated_completion: datetime