"""
ViralClip Pro - API Schemas and Data Models
Comprehensive Pydantic models for request/response validation
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

# Enums
class VideoQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class AspectRatio(str, Enum):
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
    SQUARE = "1:1"
    CINEMA = "21:9"

class ProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ClipType(str, Enum):
    HOOK = "hook"
    MAIN = "main"
    CLIMAX = "climax"
    RESOLUTION = "resolution"
    COMPLETE = "complete"
    HIGHLIGHT = "highlight"

# Base Models
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = None

class SuccessResponse(BaseResponse):
    """Generic success response"""
    data: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Video Analysis Models
class VideoAnalysisRequest(BaseModel):
    """Request model for video analysis"""
    url: str = Field(..., description="Video URL to analyze")
    clip_duration: int = Field(default=60, ge=10, le=300, description="Target clip duration in seconds")
    output_format: str = Field(default="mp4", description="Output video format")
    resolution: str = Field(default="1080p", description="Output resolution")
    aspect_ratio: AspectRatio = Field(default=AspectRatio.PORTRAIT, description="Output aspect ratio")
    enable_captions: bool = Field(default=True, description="Generate captions")
    enable_transitions: bool = Field(default=True, description="Add transitions")
    ai_editing: bool = Field(default=True, description="Enable AI-powered editing")
    viral_optimization: bool = Field(default=True, description="Optimize for viral potential")
    language: str = Field(default="en", description="Content language")
    priority: str = Field(default="normal", description="Processing priority")
    suggested_formats: List[str] = Field(default=["tiktok", "instagram"], description="Target platforms")

    @validator('url')
    def validate_url(cls, v):
        if not v or not v.startswith(('http://', 'https://')):
            raise ValueError('Must be a valid HTTP(S) URL')
        return v

class VideoInfo(BaseModel):
    """Video information model"""
    id: str
    title: str
    description: Optional[str] = None
    duration: float
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    upload_date: Optional[str] = None
    uploader: Optional[str] = None
    thumbnail: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    webpage_url: str
    platform: str
    resolution: Optional[str] = None
    fps: Optional[int] = None
    format: Optional[str] = None

class AIInsights(BaseModel):
    """AI analysis insights model"""
    viral_potential: int = Field(ge=0, le=100, description="Viral potential score (0-100)")
    engagement_prediction: int = Field(ge=0, le=100, description="Predicted engagement score")
    content_type: str = Field(description="Detected content type")
    content_confidence: int = Field(ge=0, le=100, description="Content type confidence")
    optimal_length: int = Field(description="Optimal clip length in seconds")
    suggested_formats: List[str] = Field(description="Recommended platforms")
    best_moments: List[Dict[str, Any]] = Field(default_factory=list, description="Highlight moments")
    improvement_suggestions: List[str] = Field(default_factory=list, description="AI suggestions")
    trending_score: int = Field(ge=0, le=100, description="Trending potential score")
    audience_retention: int = Field(ge=0, le=100, description="Predicted retention rate")
    hook_quality: int = Field(ge=0, le=100, description="Opening hook quality score")
    emotional_arc: List[str] = Field(default_factory=list, description="Emotional progression")

class ClipDefinition(BaseModel):
    """Individual clip definition"""
    start_time: float = Field(ge=0, description="Clip start time in seconds")
    end_time: float = Field(gt=0, description="Clip end time in seconds")
    title: str = Field(..., max_length=200, description="Clip title")
    description: Optional[str] = Field(None, max_length=500, description="Clip description")
    viral_score: Optional[int] = Field(None, ge=0, le=100, description="Predicted viral score")
    recommended_platforms: List[str] = Field(default_factory=list, description="Recommended platforms")
    clip_type: Optional[ClipType] = Field(None, description="Type of clip")

    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v

class VideoAnalysisResponse(BaseResponse):
    """Response model for video analysis"""
    session_id: str = Field(description="Analysis session ID")
    video_info: VideoInfo
    ai_insights: AIInsights
    suggested_clips: List[ClipDefinition] = Field(default_factory=list)
    processing_time: float = Field(description="Analysis time in seconds")

# Video Processing Models
class VideoProcessRequest(BaseModel):
    """Request model for video processing"""
    session_id: str = Field(..., description="Analysis session ID")
    clips: List[ClipDefinition] = Field(..., min_items=1, description="Clips to process")
    priority: str = Field(default="normal", description="Processing priority")
    quality: VideoQuality = Field(default=VideoQuality.HIGH, description="Output quality")
    enable_stabilization: bool = Field(default=True, description="Enable video stabilization")
    enable_noise_reduction: bool = Field(default=True, description="Enable noise reduction")
    enable_color_enhancement: bool = Field(default=True, description="Enable color enhancement")
    custom_settings: Optional[Dict[str, Any]] = Field(None, description="Custom processing settings")

class ClipResult(BaseModel):
    """Individual clip processing result"""
    clip_index: int = Field(description="Index of the processed clip")
    success: bool = Field(description="Whether processing succeeded")
    output_path: Optional[str] = Field(None, description="Path to output file")
    thumbnail_path: Optional[str] = Field(None, description="Path to thumbnail")
    file_size: Optional[int] = Field(None, description="Output file size in bytes")
    duration: Optional[float] = Field(None, description="Actual clip duration")
    viral_score: Optional[int] = Field(None, description="AI-calculated viral score")
    ai_enhancements: List[str] = Field(default_factory=list, description="Applied AI enhancements")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class ProcessingProgressUpdate(BaseModel):
    """Processing progress update"""
    current: int = Field(description="Current clip being processed")
    total: int = Field(description="Total clips to process")
    percentage: float = Field(ge=0, le=100, description="Overall progress percentage")
    current_clip: Optional[ClipDefinition] = Field(None, description="Currently processing clip")
    stage: str = Field(description="Current processing stage")
    message: Optional[str] = Field(None, description="Progress message")
    estimated_time_remaining: Optional[int] = Field(None, description="ETA in seconds")

class ProcessingStatusResponse(BaseResponse):
    """Processing status response"""
    task_id: str = Field(description="Processing task ID")
    status: ProcessingStatus = Field(description="Current processing status")
    progress: Optional[ProcessingProgressUpdate] = Field(None, description="Progress information")
    results: List[ClipResult] = Field(default_factory=list, description="Completed results")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(description="Task creation time")
    updated_at: datetime = Field(description="Last update time")

# File Upload Models
class FileUploadResponse(BaseResponse):
    """File upload response"""
    session_id: str = Field(description="Upload session ID")
    filename: str = Field(description="Original filename")
    file_size: int = Field(description="File size in bytes")
    file_path: str = Field(description="Server file path")
    thumbnail: Optional[str] = Field(None, description="Generated thumbnail path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Video metadata")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="File validation results")

class UploadProgressUpdate(BaseModel):
    """Upload progress update"""
    upload_id: str = Field(description="Upload identifier")
    stage: str = Field(description="Upload stage")
    progress: float = Field(ge=0, le=100, description="Upload progress percentage")
    loaded: int = Field(description="Bytes uploaded")
    total: int = Field(description="Total bytes")
    speed: Optional[float] = Field(None, description="Upload speed in bytes/sec")
    message: Optional[str] = Field(None, description="Progress message")

# WebSocket Message Models
class WebSocketMessage(BaseModel):
    """Base WebSocket message"""
    type: str = Field(description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")

class ProcessingWebSocketMessage(WebSocketMessage):
    """Processing WebSocket message"""
    task_id: str = Field(description="Processing task ID")

class UploadWebSocketMessage(WebSocketMessage):
    """Upload WebSocket message"""
    upload_id: str = Field(description="Upload identifier")

# Health and Status Models
class HealthStatus(BaseModel):
    """Health check response"""
    status: str = Field(description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(description="Application version")
    uptime: float = Field(description="Uptime in seconds")
    dependencies: Dict[str, bool] = Field(default_factory=dict, description="Dependency status")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")

class SystemMetrics(BaseModel):
    """System metrics model"""
    cpu_usage: float = Field(ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(ge=0, le=100, description="Memory usage percentage")
    disk_usage: float = Field(ge=0, le=100, description="Disk usage percentage")
    active_connections: int = Field(ge=0, description="Active WebSocket connections")
    processing_tasks: int = Field(ge=0, description="Active processing tasks")
    queue_size: int = Field(ge=0, description="Processing queue size")
    cache_hit_ratio: float = Field(ge=0, le=1, description="Cache hit ratio")

# Configuration Models
class ProcessingSettings(BaseModel):
    """Processing configuration"""
    max_concurrent_tasks: int = Field(default=3, ge=1, le=10)
    default_quality: VideoQuality = Field(default=VideoQuality.HIGH)
    enable_gpu_acceleration: bool = Field(default=True)
    temp_dir: str = Field(default="temp")
    output_dir: str = Field(default="output")
    cleanup_after_hours: int = Field(default=24, ge=1)

class SecuritySettings(BaseModel):
    """Security configuration"""
    rate_limit_per_minute: int = Field(default=60, ge=1)
    rate_limit_per_hour: int = Field(default=1000, ge=1)
    max_file_size: int = Field(default=2147483648)  # 2GB
    allowed_file_types: List[str] = Field(default=["mp4", "mov", "avi", "mkv", "webm"])
    enable_csrf_protection: bool = Field(default=True)
    session_timeout: int = Field(default=3600, ge=300)  # 1 hour

# Validation Models
class ValidationResult(BaseModel):
    """File/content validation result"""
    valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extracted metadata")
    recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")

# Analytics Models
class AnalyticsEvent(BaseModel):
    """Analytics event model"""
    event_type: str = Field(description="Event type")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: str = Field(description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    properties: Dict[str, Any] = Field(default_factory=dict, description="Event properties")
    context: Dict[str, Any] = Field(default_factory=dict, description="Event context")

class UsageStatistics(BaseModel):
    """Usage statistics model"""
    total_videos_processed: int = Field(ge=0)
    total_clips_created: int = Field(ge=0)
    average_processing_time: float = Field(ge=0)
    top_platforms: List[Dict[str, Any]] = Field(default_factory=list)
    success_rate: float = Field(ge=0, le=1)
    popular_content_types: List[Dict[str, Any]] = Field(default_factory=list)
    user_engagement_metrics: Dict[str, Any] = Field(default_factory=dict)

# Export Models (for external integrations)
class ExportRequest(BaseModel):
    """Export request model"""
    task_id: str = Field(description="Processing task ID")
    format: str = Field(default="mp4", description="Export format")
    quality: VideoQuality = Field(default=VideoQuality.HIGH)
    include_metadata: bool = Field(default=True)
    watermark: bool = Field(default=False)
    destination: Optional[str] = Field(None, description="Export destination")

class ExportResponse(BaseResponse):
    """Export response model"""
    export_id: str = Field(description="Export identifier")
    download_url: str = Field(description="Download URL")
    expires_at: datetime = Field(description="Download link expiration")
    file_size: int = Field(description="Exported file size")
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    limit: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Sort order")

class PaginatedResponse(BaseModel):
    """Paginated response model"""
    items: List[Any] = Field(description="Page items")
    total: int = Field(ge=0, description="Total items")
    page: int = Field(ge=1, description="Current page")
    limit: int = Field(ge=1, description="Items per page")
    pages: int = Field(ge=0, description="Total pages")
    has_next: bool = Field(description="Has next page")
    has_prev: bool = Field(description="Has previous page")

# Legacy compatibility (for gradual migration)
class VideoOut(BaseModel):
    """Legacy video output model"""
    id: str
    title: str
    duration: float
    status: str
    created_at: datetime
    thumbnail: Optional[str] = None
    file_size: Optional[int] = None