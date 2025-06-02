
"""
Netflix-Level Schema Definitions
Comprehensive data validation and serialization models
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import conint, confloat, constr


class ProcessingStage(str, Enum):
    """Processing stages enumeration"""
    QUEUED = "queued"
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    EXTRACTING_FEATURES = "extracting_features"
    SCORING_SEGMENTS = "scoring_segments"
    GENERATING_TIMELINE = "generating_timeline"
    GENERATING_PREVIEW = "generating_preview"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UploadStatus(str, Enum):
    """Upload status enumeration"""
    QUEUED = "queued"
    UPLOADING = "uploading"
    PAUSED = "paused"
    RETRYING = "retrying"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityLevel(str, Enum):
    """Quality level enumeration"""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"
    ULTRA = "ultra"


class PlatformType(str, Enum):
    """Social media platform types"""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE_SHORTS = "youtube_shorts"
    TWITTER = "twitter"
    SNAPCHAT = "snapchat"
    AUTO_DETECT = "auto_detect"


# Base Models
class NetflixBaseModel(BaseModel):
    """Base model with Netflix-level standards"""
    
    class Config:
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TimestampMixin(NetflixBaseModel):
    """Mixin for timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# Request Models
class VideoUploadRequest(NetflixBaseModel):
    """Video upload request validation"""
    title: Optional[constr(min_length=1, max_length=200)] = None
    description: Optional[constr(max_length=1000)] = None
    upload_id: Optional[str] = None
    platform_optimization: Optional[PlatformType] = None
    quality_preference: QualityLevel = QualityLevel.STANDARD
    
    @validator("title")
    def validate_title(cls, v):
        if v is not None and v.strip() == "":
            raise ValueError("Title cannot be empty")
        return v


class ChunkUploadRequest(NetflixBaseModel):
    """Chunked upload request validation"""
    upload_id: constr(min_length=1)
    chunk_index: conint(ge=0)
    total_chunks: conint(ge=1)
    filename: constr(min_length=1, max_length=255)
    chunk_hash: Optional[str] = None
    
    @validator("chunk_index")
    def validate_chunk_index(cls, v, values):
        if "total_chunks" in values and v >= values["total_chunks"]:
            raise ValueError("Chunk index must be less than total chunks")
        return v


class AnalysisRequest(NetflixBaseModel):
    """AI analysis request validation"""
    session_id: constr(min_length=1)
    file_path: constr(min_length=1)
    title: Optional[str] = None
    description: Optional[str] = None
    target_platforms: List[PlatformType] = Field(default_factory=list)
    custom_prompts: Optional[List[str]] = None
    analysis_depth: constr(regex="^(fast|standard|deep)$") = "standard"
    
    @validator("target_platforms")
    def validate_platforms(cls, v):
        if len(v) > 5:
            raise ValueError("Maximum 5 target platforms allowed")
        return list(set(v))  # Remove duplicates


class PreviewRequest(NetflixBaseModel):
    """Preview generation request validation"""
    session_id: constr(min_length=1)
    start_time: confloat(ge=0)
    end_time: confloat(gt=0)
    quality: QualityLevel = QualityLevel.STANDARD
    platform_optimization: Optional[PlatformType] = None
    include_analysis: bool = True
    
    @validator("end_time")
    def validate_end_time(cls, v, values):
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("End time must be greater than start time")
        if v - values.get("start_time", 0) > 300:  # 5 minutes max
            raise ValueError("Preview duration cannot exceed 5 minutes")
        return v


class TimelineInteractionRequest(NetflixBaseModel):
    """Timeline interaction request validation"""
    session_id: constr(min_length=1)
    interaction_type: constr(regex="^(click|hover|selection|zoom)$")
    timestamp: confloat(ge=0)
    position: Dict[str, confloat(ge=0)] = Field(description="x, y coordinates")
    selection_range: Optional[Dict[str, confloat(ge=0)]] = None
    
    @validator("position")
    def validate_position(cls, v):
        required_keys = {"x", "y"}
        if not required_keys.issubset(v.keys()):
            raise ValueError(f"Position must contain keys: {required_keys}")
        return v


# Response Models
class UploadProgressResponse(NetflixBaseModel):
    """Upload progress response"""
    session_id: str
    status: UploadStatus
    progress: confloat(ge=0, le=100)
    uploaded_bytes: conint(ge=0)
    total_bytes: conint(ge=0)
    speed: confloat(ge=0)  # bytes per second
    eta: Optional[int] = None  # seconds
    chunks_completed: conint(ge=0)
    chunks_total: conint(ge=0)
    error_message: Optional[str] = None


class VideoUploadResponse(NetflixBaseModel):
    """Video upload response"""
    success: bool
    session_id: str
    file_path: Optional[str] = None
    file_size: Optional[conint(ge=0)] = None
    estimated_processing_time: Optional[confloat(ge=0)] = None
    processing_time: Optional[confloat(ge=0)] = None
    upload_id: Optional[str] = None
    error: Optional[str] = None


class ViralAnalysis(NetflixBaseModel):
    """Viral potential analysis"""
    viral_score: conint(ge=0, le=100)
    confidence: confloat(ge=0, le=1)
    factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    engagement_prediction: Optional[confloat(ge=0)] = None
    optimal_duration: Optional[confloat(ge=0)] = None
    best_segments: List[Dict[str, Any]] = Field(default_factory=list)


class TimelineData(NetflixBaseModel):
    """Timeline visualization data"""
    duration: confloat(ge=0)
    viral_heatmap: List[conint(ge=0, le=100)]
    engagement_peaks: List[Dict[str, Any]] = Field(default_factory=list)
    key_moments: List[Dict[str, Any]] = Field(default_factory=list)
    emotional_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    trend_analysis: Optional[Dict[str, Any]] = None


class AnalysisResponse(NetflixBaseModel):
    """AI analysis response"""
    success: bool
    session_id: str
    analysis: Optional[ViralAnalysis] = None
    timeline: Optional[TimelineData] = None
    processing_time: confloat(ge=0)
    cached: bool = False
    error: Optional[str] = None


class PreviewResponse(NetflixBaseModel):
    """Preview generation response"""
    success: bool
    session_id: str
    preview_url: Optional[str] = None
    viral_analysis: Optional[ViralAnalysis] = None
    suggestions: List[str] = Field(default_factory=list)
    processing_time: confloat(ge=0)
    generation_time: Optional[confloat(ge=0)] = None
    quality_achieved: Optional[QualityLevel] = None
    file_size: Optional[conint(ge=0)] = None
    error: Optional[str] = None


class ProcessingStatus(NetflixBaseModel):
    """Processing status response"""
    session_id: str
    stage: ProcessingStage
    progress: confloat(ge=0, le=100)
    message: str
    details: Optional[Dict[str, Any]] = None
    estimated_completion: Optional[datetime] = None
    processing_time: confloat(ge=0)
    queue_position: Optional[conint(ge=0)] = None
    error: Optional[str] = None


class SystemHealth(NetflixBaseModel):
    """System health status"""
    status: constr(regex="^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    uptime: confloat(ge=0)
    version: str
    performance_score: Optional[confloat(ge=0, le=100)] = None


class ErrorResponse(NetflixBaseModel):
    """Standardized error response"""
    error: bool = True
    error_id: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status_code: conint(ge=100, le=599)
    path: Optional[str] = None
    request_id: Optional[str] = None


class MetricsResponse(NetflixBaseModel):
    """System metrics response"""
    success: bool
    metrics: Dict[str, Any]
    collection_time: datetime = Field(default_factory=datetime.utcnow)
    performance_indicators: Optional[Dict[str, Any]] = None
    recommendations: List[str] = Field(default_factory=list)


class ClipRecommendation(NetflixBaseModel):
    """Viral clip recommendation"""
    clip_id: str
    start_time: confloat(ge=0)
    end_time: confloat(gt=0)
    viral_score: conint(ge=0, le=100)
    confidence: confloat(ge=0, le=1)
    platform_optimized: List[PlatformType] = Field(default_factory=list)
    title_suggestion: Optional[str] = None
    description_suggestion: Optional[str] = None
    hashtag_suggestions: List[str] = Field(default_factory=list)
    
    @validator("end_time")
    def validate_duration(cls, v, values):
        if "start_time" in values:
            duration = v - values["start_time"]
            if duration < 3:  # Minimum 3 seconds
                raise ValueError("Clip duration must be at least 3 seconds")
            if duration > 300:  # Maximum 5 minutes
                raise ValueError("Clip duration cannot exceed 5 minutes")
        return v


class WebSocketMessage(NetflixBaseModel):
    """WebSocket message format"""
    type: constr(min_length=1)
    session_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None


class CacheInfo(NetflixBaseModel):
    """Cache information response"""
    cache_key: str
    hit: bool
    ttl: Optional[int] = None
    size: Optional[int] = None
    created_at: Optional[datetime] = None
    access_count: Optional[int] = None


# Validation Helpers
def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension"""
    extension = filename.lower().split('.')[-1] if '.' in filename else ''
    return extension in [ext.lower().strip('.') for ext in allowed_extensions]


def validate_session_id_format(session_id: str) -> bool:
    """Validate session ID format"""
    import re
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, session_id)) and len(session_id) >= 8


def validate_viral_score_range(score: int) -> bool:
    """Validate viral score is in valid range"""
    return 0 <= score <= 100


def validate_timestamp_sequence(start: float, end: float) -> bool:
    """Validate timestamp sequence"""
    return start >= 0 and end > start


# Export all models
__all__ = [
    # Enums
    "ProcessingStage",
    "UploadStatus", 
    "QualityLevel",
    "PlatformType",
    
    # Base Models
    "NetflixBaseModel",
    "TimestampMixin",
    
    # Request Models
    "VideoUploadRequest",
    "ChunkUploadRequest",
    "AnalysisRequest",
    "PreviewRequest",
    "TimelineInteractionRequest",
    
    # Response Models
    "UploadProgressResponse",
    "VideoUploadResponse",
    "ViralAnalysis",
    "TimelineData",
    "AnalysisResponse",
    "PreviewResponse",
    "ProcessingStatus",
    "SystemHealth",
    "ErrorResponse",
    "MetricsResponse",
    "ClipRecommendation",
    "WebSocketMessage",
    "CacheInfo",
    
    # Validators
    "validate_file_extension",
    "validate_session_id_format",
    "validate_viral_score_range",
    "validate_timestamp_sequence",
]
