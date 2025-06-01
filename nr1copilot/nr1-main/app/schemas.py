
"""
Netflix-Level Pydantic Schemas v5.0
Comprehensive data validation and serialization
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl


class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    QUEUED = "queued"
    INITIALIZING = "initializing"
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    EXTRACTING_FEATURES = "extracting_features"
    CALCULATING_SCORES = "calculating_scores"
    GENERATING_TIMELINE = "generating_timeline"
    CREATING_PREVIEWS = "creating_previews"
    OPTIMIZING = "optimizing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityLevel(str, Enum):
    """Video quality levels"""
    INSTANT = "instant"
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"
    ULTRA = "ultra"


class Platform(str, Enum):
    """Supported platforms"""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE_SHORTS = "youtube_shorts"
    TWITTER = "twitter"
    SNAPCHAT = "snapchat"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"


class AnalysisType(str, Enum):
    """Analysis types"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    VIRAL_OPTIMIZATION = "viral_optimization"
    PLATFORM_SPECIFIC = "platform_specific"
    CUSTOM = "custom"


# Base response models
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = None
    error_id: Optional[str] = None
    message: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


# Video processing models
class VideoMetadata(BaseModel):
    """Video metadata structure"""
    duration: float = Field(..., description="Video duration in seconds")
    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    fps: float = Field(..., description="Frames per second")
    format: str = Field(..., description="Video format")
    bitrate: Optional[int] = Field(None, description="Video bitrate")
    audio_channels: Optional[int] = Field(None, description="Audio channels")
    audio_sample_rate: Optional[int] = Field(None, description="Audio sample rate")
    file_size: int = Field(..., description="File size in bytes")
    codec: Optional[str] = Field(None, description="Video codec")
    
    @validator('duration')
    def validate_duration(cls, v):
        if v <= 0:
            raise ValueError('Duration must be positive')
        if v > 600:  # 10 minutes max
            raise ValueError('Duration too long (max 10 minutes)')
        return v
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v < 1 or v > 8192:
            raise ValueError('Invalid video dimensions')
        return v


class ViralAnalysis(BaseModel):
    """Viral potential analysis"""
    viral_score: float = Field(..., ge=0, le=100, description="Viral score 0-100")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level 0-1")
    factors: List[str] = Field(default_factory=list, description="Contributing factors")
    recommendations: List[str] = Field(default_factory=list, description="Improvement suggestions")
    platform_scores: Dict[Platform, float] = Field(default_factory=dict, description="Platform-specific scores")
    emotion_analysis: Optional[Dict[str, Any]] = Field(None, description="Emotion analysis data")
    trend_alignment: Optional[float] = Field(None, ge=0, le=100, description="Trend alignment score")
    engagement_prediction: Optional[Dict[str, float]] = Field(None, description="Predicted engagement metrics")


class KeyMoment(BaseModel):
    """Key moment in video timeline"""
    timestamp: float = Field(..., description="Timestamp in seconds")
    type: str = Field(..., description="Moment type (hook, peak, climax, etc.)")
    description: str = Field(..., description="Moment description")
    viral_score: float = Field(..., ge=0, le=100, description="Viral score for this moment")
    emotional_intensity: Optional[float] = Field(None, ge=0, le=1, description="Emotional intensity")
    engagement_likelihood: Optional[float] = Field(None, ge=0, le=1, description="Engagement likelihood")


class TimelineData(BaseModel):
    """Interactive timeline data"""
    duration: float = Field(..., description="Total duration in seconds")
    viral_heatmap: List[float] = Field(..., description="Viral scores across timeline")
    key_moments: List[KeyMoment] = Field(default_factory=list, description="Key moments")
    energy_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Energy levels")
    emotion_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Emotion timeline")
    engagement_peaks: List[Dict[str, Any]] = Field(default_factory=list, description="Engagement peaks")
    recommended_clips: List[Dict[str, Any]] = Field(default_factory=list, description="Recommended clip segments")


class PreviewData(BaseModel):
    """Preview generation data"""
    preview_url: str = Field(..., description="Preview video URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail image URL")
    duration: float = Field(..., description="Preview duration")
    quality: QualityLevel = Field(..., description="Preview quality level")
    viral_analysis: Optional[ViralAnalysis] = Field(None, description="Viral analysis for preview")
    generation_time: float = Field(..., description="Generation time in milliseconds")
    cache_hit: bool = Field(False, description="Whether result was cached")


class ProcessingJob(BaseModel):
    """Processing job status"""
    job_id: str = Field(..., description="Unique job identifier")
    session_id: str = Field(..., description="Session identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress: float = Field(0, ge=0, le=100, description="Progress percentage")
    stage: str = Field("", description="Current processing stage")
    message: str = Field("", description="Status message")
    start_time: datetime = Field(..., description="Job start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    processing_time: Optional[float] = Field(None, description="Total processing time")
    error_message: Optional[str] = Field(None, description="Error message if failed")


# Request models
class VideoUploadRequest(BaseModel):
    """Video upload request"""
    title: Optional[str] = Field(None, max_length=200, description="Video title")
    description: Optional[str] = Field(None, max_length=2000, description="Video description")
    tags: List[str] = Field(default_factory=list, max_items=20, description="Video tags")
    target_platforms: List[Platform] = Field(default_factory=list, description="Target platforms")
    analysis_type: AnalysisType = Field(AnalysisType.COMPREHENSIVE, description="Analysis type")
    enable_ai_enhancement: bool = Field(True, description="Enable AI enhancement")
    enable_viral_optimization: bool = Field(True, description="Enable viral optimization")
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom processing settings")


class AnalysisRequest(BaseModel):
    """Video analysis request"""
    session_id: str = Field(..., description="Session identifier")
    file_path: str = Field(..., description="Video file path")
    title: Optional[str] = Field(None, max_length=200, description="Video title")
    description: Optional[str] = Field(None, max_length=2000, description="Video description")
    target_platforms: List[Platform] = Field(default_factory=list, description="Target platforms")
    analysis_type: AnalysisType = Field(AnalysisType.COMPREHENSIVE, description="Analysis type")
    custom_prompts: List[str] = Field(default_factory=list, description="Custom analysis prompts")
    enable_deep_analysis: bool = Field(True, description="Enable deep AI analysis")


class PreviewRequest(BaseModel):
    """Preview generation request"""
    session_id: str = Field(..., description="Session identifier")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    quality: QualityLevel = Field(QualityLevel.STANDARD, description="Preview quality")
    platform_optimizations: List[Platform] = Field(default_factory=list, description="Platform optimizations")
    enable_viral_enhancements: bool = Field(True, description="Enable viral enhancements")
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be greater than start time')
        return v


class ClipGenerationRequest(BaseModel):
    """Clip generation request"""
    session_id: str = Field(..., description="Session identifier")
    clips: List[Dict[str, Any]] = Field(..., description="Clip specifications")
    quality: QualityLevel = Field(QualityLevel.HIGH, description="Output quality")
    target_platforms: List[Platform] = Field(default_factory=list, description="Target platforms")
    enable_viral_optimization: bool = Field(True, description="Enable viral optimization")
    enable_ai_enhancement: bool = Field(True, description="Enable AI enhancement")
    output_format: str = Field("mp4", description="Output format")
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom settings")


# Response models
class VideoUploadResponse(BaseResponse):
    """Video upload response"""
    session_id: str = Field(..., description="Session identifier")
    file_path: str = Field(..., description="Uploaded file path")
    file_size: int = Field(..., description="File size in bytes")
    metadata: Optional[VideoMetadata] = Field(None, description="Video metadata")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Initial analysis")
    preview: Optional[Dict[str, Any]] = Field(None, description="Initial preview data")


class AnalysisResponse(BaseResponse):
    """Video analysis response"""
    session_id: str = Field(..., description="Session identifier")
    analysis: ViralAnalysis = Field(..., description="Comprehensive analysis")
    timeline: Optional[TimelineData] = Field(None, description="Timeline data")
    metadata: Optional[VideoMetadata] = Field(None, description="Video metadata")
    cached: bool = Field(False, description="Whether result was cached")


class PreviewResponse(BaseResponse):
    """Preview generation response"""
    session_id: str = Field(..., description="Session identifier")
    preview_url: str = Field(..., description="Preview video URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    viral_analysis: Optional[ViralAnalysis] = Field(None, description="Viral analysis")
    suggestions: List[str] = Field(default_factory=list, description="Optimization suggestions")
    quality: QualityLevel = Field(..., description="Preview quality")


class ClipGenerationResponse(BaseResponse):
    """Clip generation response"""
    task_id: str = Field(..., description="Processing task identifier")
    session_id: str = Field(..., description="Session identifier")
    estimated_completion: datetime = Field(..., description="Estimated completion time")
    clips_count: int = Field(..., description="Number of clips to generate")


class ProcessingStatusResponse(ProcessingJob):
    """Processing status response"""
    pass


# System models
class SystemHealth(BaseModel):
    """System health status"""
    status: str = Field(..., description="Overall system status")
    services: Dict[str, str] = Field(..., description="Service statuses")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    version: str = Field("5.0.0", description="Application version")
    uptime: Optional[float] = Field(None, description="System uptime in seconds")
    load: Optional[Dict[str, float]] = Field(None, description="System load metrics")


class MetricsData(BaseModel):
    """System metrics data"""
    requests_total: int = Field(0, description="Total requests processed")
    requests_per_second: float = Field(0, description="Current requests per second")
    active_sessions: int = Field(0, description="Active sessions count")
    processing_queue_size: int = Field(0, description="Processing queue size")
    cache_hit_rate: float = Field(0, ge=0, le=1, description="Cache hit rate")
    average_response_time: float = Field(0, description="Average response time in ms")
    error_rate: float = Field(0, ge=0, le=1, description="Error rate")
    memory_usage: Optional[Dict[str, float]] = Field(None, description="Memory usage metrics")
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="CPU usage percentage")


# WebSocket models
class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str = Field(..., description="Message type")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")


class UploadProgress(WebSocketMessage):
    """Upload progress message"""
    type: str = Field("upload_progress", const=True)
    progress: float = Field(..., ge=0, le=100, description="Upload progress percentage")
    uploaded_size: int = Field(..., description="Uploaded bytes")
    total_size: int = Field(..., description="Total file size")
    speed: Optional[float] = Field(None, description="Upload speed in bytes/second")


class ProcessingProgress(WebSocketMessage):
    """Processing progress message"""
    type: str = Field("processing_progress", const=True)
    status: ProcessingStatus = Field(..., description="Processing status")
    progress: float = Field(..., ge=0, le=100, description="Processing progress")
    stage: str = Field(..., description="Current stage")
    message: str = Field("", description="Status message")
    entertaining_fact: Optional[str] = Field(None, description="Entertaining fact")


class ViralScoreUpdate(WebSocketMessage):
    """Viral score update message"""
    type: str = Field("viral_score_update", const=True)
    viral_score: float = Field(..., ge=0, le=100, description="Updated viral score")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level")
    factors: List[str] = Field(default_factory=list, description="Contributing factors")


class TimelineUpdate(WebSocketMessage):
    """Timeline update message"""
    type: str = Field("timeline_update", const=True)
    timeline_data: TimelineData = Field(..., description="Updated timeline data")


class PreviewUpdate(WebSocketMessage):
    """Preview update message"""
    type: str = Field("preview_update", const=True)
    preview_data: PreviewData = Field(..., description="Preview data")


# Configuration models
class AppConfig(BaseModel):
    """Application configuration"""
    debug: bool = Field(False, description="Debug mode")
    max_file_size: int = Field(100 * 1024 * 1024, description="Max file size in bytes")
    upload_path: str = Field("uploads", description="Upload directory path")
    output_path: str = Field("output", description="Output directory path")
    temp_path: str = Field("temp", description="Temporary files path")
    cache_path: str = Field("cache", description="Cache directory path")
    log_level: str = Field("INFO", description="Logging level")
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    enable_caching: bool = Field(True, description="Enable response caching")
    require_auth: bool = Field(False, description="Require authentication")
    allowed_origins: List[str] = Field(["*"], description="CORS allowed origins")
    rate_limit_requests: int = Field(100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(60, description="Rate limit window in seconds")


# Export all models
__all__ = [
    # Enums
    "ProcessingStatus", "QualityLevel", "Platform", "AnalysisType",
    
    # Base models
    "BaseResponse", "ErrorResponse",
    
    # Data models
    "VideoMetadata", "ViralAnalysis", "KeyMoment", "TimelineData", 
    "PreviewData", "ProcessingJob",
    
    # Request models
    "VideoUploadRequest", "AnalysisRequest", "PreviewRequest", "ClipGenerationRequest",
    
    # Response models
    "VideoUploadResponse", "AnalysisResponse", "PreviewResponse", 
    "ClipGenerationResponse", "ProcessingStatusResponse",
    
    # System models
    "SystemHealth", "MetricsData",
    
    # WebSocket models
    "WebSocketMessage", "UploadProgress", "ProcessingProgress", 
    "ViralScoreUpdate", "TimelineUpdate", "PreviewUpdate",
    
    # Configuration
    "AppConfig"
]
