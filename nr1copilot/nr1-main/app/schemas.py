
"""
ViralClip Pro Enterprise v6.0 - Netflix-Level Data Schemas
Production-grade data models with comprehensive validation
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import uuid


class ProcessingStage(str, Enum):
    """Processing stages for video analysis"""
    INITIALIZING = "initializing"
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    EXTRACTING_FEATURES = "extracting_features"
    SCORING_SEGMENTS = "scoring_segments"
    GENERATING_TIMELINE = "generating_timeline"
    CREATING_PREVIEWS = "creating_previews"
    OPTIMIZING = "optimizing"
    COMPLETE = "complete"
    FAILED = "failed"


class UploadStatus(str, Enum):
    """Upload status enumeration"""
    QUEUED = "queued"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class QualityLevel(str, Enum):
    """Quality levels for processing"""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    NETFLIX = "netflix"
    PREMIUM = "premium"


class Platform(str, Enum):
    """Social media platforms"""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE_SHORTS = "youtube_shorts"
    TWITTER = "twitter"
    SNAPCHAT = "snapchat"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"


# Base Response Models
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = Field(..., description="Operation success status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")


class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = Field(default=False, description="Always false for errors")
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    trace_id: Optional[str] = Field(None, description="Error trace identifier")


# Upload Models
class FileValidationResult(BaseModel):
    """File validation result"""
    valid: bool = Field(..., description="Validation success status")
    error: Optional[str] = Field(None, description="Validation error message")
    file_size: int = Field(0, description="File size in bytes")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    format: Optional[str] = Field(None, description="File format")
    resolution: Optional[str] = Field(None, description="Video resolution")
    estimated_time: Optional[float] = Field(None, description="Estimated processing time")


class ChunkUploadRequest(BaseModel):
    """Chunk upload request model"""
    upload_id: str = Field(..., description="Upload session identifier")
    chunk_index: int = Field(..., ge=0, description="Chunk index starting from 0")
    total_chunks: int = Field(..., ge=1, description="Total number of chunks")
    chunk_hash: Optional[str] = Field(None, description="Chunk integrity hash")
    
    @validator('chunk_index')
    def validate_chunk_index(cls, v, values):
        if 'total_chunks' in values and v >= values['total_chunks']:
            raise ValueError('Chunk index cannot be greater than or equal to total chunks')
        return v


class UploadProgress(BaseModel):
    """Upload progress model"""
    upload_id: str = Field(..., description="Upload identifier")
    status: UploadStatus = Field(..., description="Current upload status")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage")
    bytes_uploaded: int = Field(..., ge=0, description="Bytes uploaded so far")
    total_bytes: int = Field(..., ge=0, description="Total bytes to upload")
    upload_speed: Optional[float] = Field(None, description="Upload speed in bytes/second")
    eta_seconds: Optional[float] = Field(None, description="Estimated time to completion")
    chunks_completed: int = Field(0, description="Number of chunks completed")
    total_chunks: int = Field(1, description="Total number of chunks")


class VideoUploadResponse(BaseResponse):
    """Video upload response model"""
    session_id: str = Field(..., description="Processing session identifier")
    file_path: str = Field(..., description="Uploaded file path")
    file_size: int = Field(..., description="File size in bytes")
    upload_time: float = Field(..., description="Upload processing time in seconds")
    estimated_processing_time: Optional[float] = Field(None, description="Estimated processing time")
    processing_started: bool = Field(default=False, description="Whether background processing started")


# Analysis Models
class AnalysisRequest(BaseModel):
    """Analysis request model"""
    session_id: str = Field(..., description="Session identifier")
    file_path: str = Field(..., description="Video file path")
    title: Optional[str] = Field(None, max_length=200, description="Video title")
    description: Optional[str] = Field(None, max_length=1000, description="Video description")
    target_platforms: List[Platform] = Field(default_factory=list, description="Target platforms")
    quality_level: QualityLevel = Field(default=QualityLevel.HIGH, description="Analysis quality level")
    custom_prompts: Optional[List[str]] = Field(None, description="Custom analysis prompts")
    enable_captions: bool = Field(default=True, description="Enable caption analysis")
    enable_audio_analysis: bool = Field(default=True, description="Enable audio analysis")


class ViralFactors(BaseModel):
    """Viral potential factors"""
    hook_strength: float = Field(..., ge=0, le=100, description="Opening hook strength score")
    visual_appeal: float = Field(..., ge=0, le=100, description="Visual appeal score")
    audio_quality: float = Field(..., ge=0, le=100, description="Audio quality score")
    engagement_peaks: int = Field(..., ge=0, description="Number of engagement peaks")
    trending_elements: List[str] = Field(default_factory=list, description="Trending elements detected")
    emotional_impact: float = Field(..., ge=0, le=100, description="Emotional impact score")
    shareability: float = Field(..., ge=0, le=100, description="Shareability score")


class TimelineSegment(BaseModel):
    """Timeline segment with viral scoring"""
    start_time: float = Field(..., ge=0, description="Segment start time in seconds")
    end_time: float = Field(..., gt=0, description="Segment end time in seconds")
    viral_score: float = Field(..., ge=0, le=100, description="Viral potential score")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in scoring")
    key_moments: List[str] = Field(default_factory=list, description="Key moments in segment")
    emotions: List[str] = Field(default_factory=list, description="Detected emotions")
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be greater than start time')
        return v


class AnalysisResult(BaseModel):
    """Comprehensive analysis result"""
    overall_viral_score: float = Field(..., ge=0, le=100, description="Overall viral potential score")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence level")
    viral_factors: ViralFactors = Field(..., description="Detailed viral factors")
    timeline_segments: List[TimelineSegment] = Field(..., description="Timeline analysis")
    key_moments: List[Dict[str, Any]] = Field(default_factory=list, description="Identified key moments")
    platform_scores: Dict[Platform, float] = Field(default_factory=dict, description="Platform-specific scores")
    optimization_suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    processing_time: float = Field(..., description="Analysis processing time")
    model_version: str = Field(..., description="AI model version used")


class AnalysisResponse(BaseResponse):
    """Analysis response model"""
    session_id: str = Field(..., description="Session identifier")
    analysis: AnalysisResult = Field(..., description="Analysis results")
    cached: bool = Field(default=False, description="Whether result was cached")


# Preview Models
class PreviewRequest(BaseModel):
    """Preview generation request"""
    session_id: str = Field(..., description="Session identifier")
    start_time: float = Field(..., ge=0, description="Preview start time in seconds")
    end_time: float = Field(..., gt=0, description="Preview end time in seconds")
    quality: QualityLevel = Field(default=QualityLevel.HIGH, description="Preview quality")
    platform_optimization: Optional[Platform] = Field(None, description="Platform-specific optimization")
    enable_effects: bool = Field(default=True, description="Enable visual effects")
    enable_captions: bool = Field(default=False, description="Enable captions overlay")
    
    @validator('end_time')
    def validate_preview_duration(cls, v, values):
        if 'start_time' in values:
            duration = v - values['start_time']
            if duration <= 0:
                raise ValueError('Preview duration must be positive')
            if duration > 60:  # Max 60 seconds
                raise ValueError('Preview duration cannot exceed 60 seconds')
        return v


class PreviewResponse(BaseResponse):
    """Preview generation response"""
    session_id: str = Field(..., description="Session identifier")
    preview_url: str = Field(..., description="Preview video URL")
    duration: float = Field(..., description="Preview duration in seconds")
    viral_analysis: Optional[Dict[str, Any]] = Field(None, description="Segment viral analysis")
    suggestions: List[str] = Field(default_factory=list, description="Optimization suggestions")
    processing_time: float = Field(..., description="Generation processing time")
    quality_metrics: Optional[Dict[str, float]] = Field(None, description="Quality metrics")


# Processing Status Models
class ProcessingMetrics(BaseModel):
    """Processing performance metrics"""
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, description="Memory usage in MB")
    gpu_usage: Optional[float] = Field(None, ge=0, le=100, description="GPU usage percentage")
    processing_speed: float = Field(..., description="Processing speed multiplier")
    queue_length: int = Field(..., ge=0, description="Processing queue length")


class ProcessingStatus(BaseResponse):
    """Processing status model"""
    session_id: str = Field(..., description="Session identifier")
    stage: ProcessingStage = Field(..., description="Current processing stage")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    current_task: str = Field(..., description="Current task description")
    tasks_completed: int = Field(..., ge=0, description="Number of completed tasks")
    total_tasks: int = Field(..., ge=1, description="Total number of tasks")
    metrics: Optional[ProcessingMetrics] = Field(None, description="Performance metrics")
    error_message: Optional[str] = Field(None, description="Error message if failed")


# System Health Models
class ServiceHealth(BaseModel):
    """Individual service health"""
    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    response_time: Optional[float] = Field(None, description="Response time in milliseconds")
    last_check: datetime = Field(..., description="Last health check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class SystemMetrics(BaseModel):
    """System performance metrics"""
    cpu_percent: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_percent: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_percent: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    network_io: Dict[str, int] = Field(default_factory=dict, description="Network I/O stats")
    active_connections: int = Field(..., ge=0, description="Active WebSocket connections")
    requests_per_minute: float = Field(..., ge=0, description="Requests per minute")


class SystemHealth(BaseResponse):
    """System health response"""
    status: str = Field(..., description="Overall system status")
    services: Dict[str, str] = Field(..., description="Service status mapping")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(default="6.0.0", description="Application version")


# WebSocket Models
class WebSocketMessage(BaseModel):
    """Base WebSocket message"""
    type: str = Field(..., description="Message type")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")


class UploadProgressMessage(WebSocketMessage):
    """Upload progress WebSocket message"""
    type: str = Field(default="upload_progress", description="Message type")
    progress: UploadProgress = Field(..., description="Upload progress data")


class ProcessingStatusMessage(WebSocketMessage):
    """Processing status WebSocket message"""
    type: str = Field(default="processing_status", description="Message type")
    status: ProcessingStatus = Field(..., description="Processing status data")


class TimelineUpdateMessage(WebSocketMessage):
    """Timeline update WebSocket message"""
    type: str = Field(default="timeline_update", description="Message type")
    timeline_data: Dict[str, Any] = Field(..., description="Timeline data")


# Configuration Models
class EnterpriseConfig(BaseModel):
    """Enterprise configuration model"""
    max_file_size: int = Field(default=2_147_483_648, description="Maximum file size in bytes")
    max_concurrent_uploads: int = Field(default=10, description="Maximum concurrent uploads")
    chunk_size: int = Field(default=5_242_880, description="Upload chunk size in bytes")
    supported_formats: List[str] = Field(
        default_factory=lambda: ["mp4", "avi", "mov", "webm", "mkv", "m4v", "3gp", "mp3", "wav"],
        description="Supported file formats"
    )
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    enable_gpu_acceleration: bool = Field(default=True, description="Enable GPU acceleration")
    quality_presets: Dict[QualityLevel, Dict[str, Any]] = Field(
        default_factory=dict, description="Quality preset configurations"
    )


# Validation Utilities
def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension against allowed list"""
    if not filename or '.' not in filename:
        return False
    
    extension = filename.split('.')[-1].lower()
    return extension in [ext.lower() for ext in allowed_extensions]


def validate_video_duration(duration: float) -> bool:
    """Validate video duration constraints"""
    return 0 < duration <= 3600  # Max 1 hour


def validate_viral_score(score: float) -> bool:
    """Validate viral score range"""
    return 0 <= score <= 100


# Export all models
__all__ = [
    # Enums
    "ProcessingStage",
    "UploadStatus", 
    "QualityLevel",
    "Platform",
    
    # Base Models
    "BaseResponse",
    "ErrorResponse",
    
    # Upload Models
    "FileValidationResult",
    "ChunkUploadRequest",
    "UploadProgress",
    "VideoUploadResponse",
    
    # Analysis Models
    "AnalysisRequest",
    "ViralFactors",
    "TimelineSegment",
    "AnalysisResult",
    "AnalysisResponse",
    
    # Preview Models
    "PreviewRequest",
    "PreviewResponse",
    
    # Processing Models
    "ProcessingMetrics",
    "ProcessingStatus",
    
    # Health Models
    "ServiceHealth",
    "SystemMetrics",
    "SystemHealth",
    
    # WebSocket Models
    "WebSocketMessage",
    "UploadProgressMessage",
    "ProcessingStatusMessage",
    "TimelineUpdateMessage",
    
    # Configuration
    "EnterpriseConfig",
    
    # Utilities
    "validate_file_extension",
    "validate_video_duration", 
    "validate_viral_score"
]
