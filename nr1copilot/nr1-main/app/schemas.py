"""
Netflix-Level Data Schemas
Pydantic models for request/response validation
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class VideoFormat(str, Enum):
    """Supported video formats"""
    MP4 = "video/mp4"
    MOV = "video/mov"
    AVI = "video/avi"
    WEBM = "video/webm"
    QUICKTIME = "video/quicktime"


class Platform(str, Enum):
    """Supported social media platforms"""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE_SHORTS = "youtube_shorts"
    TWITTER = "twitter"


class ProcessingStatus(str, Enum):
    """Video processing status"""
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class QualityLevel(str, Enum):
    """Video quality levels"""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"


# Request Schemas
class UploadRequest(BaseModel):
    """Video upload request schema"""
    upload_id: str = Field(..., description="Unique upload identifier")
    quality_preference: QualityLevel = Field(default=QualityLevel.HIGH, description="Quality preference")
    target_platforms: List[Platform] = Field(default=[Platform.TIKTOK], description="Target platforms")


class PreviewRequest(BaseModel):
    """Live preview generation request"""
    session_id: str = Field(..., description="Session identifier")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., gt=0, description="End time in seconds")
    quality: QualityLevel = Field(default=QualityLevel.HIGH, description="Preview quality")
    platform_optimizations: Optional[List[Platform]] = Field(None, description="Platform optimizations")

    @validator("end_time")
    def validate_end_time(cls, v, values):
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("End time must be greater than start time")
        return v


class ClipGenerationRequest(BaseModel):
    """Clip generation request"""
    session_id: str = Field(..., description="Session identifier")
    clips: List[Dict[str, Any]] = Field(..., description="Clip definitions")
    quality: QualityLevel = Field(default=QualityLevel.HIGH, description="Output quality")
    platforms: List[Platform] = Field(..., description="Target platforms")


# Response Schemas
class ViralAnalysis(BaseModel):
    """Viral potential analysis"""
    viral_score: int = Field(..., ge=0, le=100, description="Viral score (0-100)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")
    factors: List[str] = Field(..., description="Contributing factors")
    platform_scores: Dict[Platform, int] = Field(default_factory=dict, description="Platform-specific scores")


class KeyMoment(BaseModel):
    """Key moment in video"""
    timestamp: float = Field(..., ge=0, description="Timestamp in seconds")
    moment_type: str = Field(..., description="Type of moment (hook, climax, reveal)")
    description: str = Field(..., description="Moment description")
    viral_score: int = Field(..., ge=0, le=100, description="Viral score for this moment")


class TimelineData(BaseModel):
    """Timeline visualization data"""
    duration: float = Field(..., gt=0, description="Video duration in seconds")
    viral_heatmap: List[int] = Field(..., description="Viral score heatmap")
    key_moments: List[KeyMoment] = Field(..., description="Key moments")
    segments: List[Dict[str, Any]] = Field(..., description="Video segments")


class UploadResponse(BaseModel):
    """Upload response schema"""
    success: bool = Field(..., description="Upload success status")
    session_id: str = Field(..., description="Session identifier")
    upload_id: str = Field(..., description="Upload identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    preview: Optional[Dict[str, Any]] = Field(None, description="Instant preview data")
    message: str = Field(..., description="Status message")


class PreviewResponse(BaseModel):
    """Preview generation response"""
    success: bool = Field(..., description="Generation success status")
    preview_url: str = Field(..., description="Preview video URL")
    viral_analysis: ViralAnalysis = Field(..., description="Viral analysis")
    suggestions: List[str] = Field(..., description="Optimization suggestions")
    duration: float = Field(..., gt=0, description="Preview duration")
    quality: QualityLevel = Field(..., description="Preview quality")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: float = Field(..., description="Check timestamp")
    services: Dict[str, str] = Field(..., description="Service statuses")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    path: str = Field(..., description="Request path")
    timestamp: int = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


# WebSocket Message Schemas
class WebSocketMessage(BaseModel):
    """Base WebSocket message"""
    type: str = Field(..., description="Message type")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Message timestamp")


class UploadProgressMessage(WebSocketMessage):
    """Upload progress WebSocket message"""
    type: str = Field(default="upload_progress", const=True)
    upload_id: str = Field(..., description="Upload identifier")
    bytes_received: int = Field(..., ge=0, description="Bytes received")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    status: ProcessingStatus = Field(..., description="Upload status")


class ProcessingStatusMessage(WebSocketMessage):
    """Processing status WebSocket message"""
    type: str = Field(default="processing_status", const=True)
    session_id: str = Field(..., description="Session identifier")
    stage: str = Field(..., description="Processing stage")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    message: str = Field(..., description="Status message")


class ViralScoreUpdateMessage(WebSocketMessage):
    """Viral score update WebSocket message"""
    type: str = Field(default="viral_score_update", const=True)
    session_id: str = Field(..., description="Session identifier")
    viral_score: int = Field(..., ge=0, le=100, description="Updated viral score")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level")
    factors: List[str] = Field(..., description="Contributing factors")


class TimelineUpdateMessage(WebSocketMessage):
    """Timeline update WebSocket message"""
    type: str = Field(default="timeline_update", const=True)
    session_id: str = Field(..., description="Session identifier")
    viral_heatmap: List[int] = Field(..., description="Viral score heatmap")
    key_moments: List[KeyMoment] = Field(..., description="Key moments")
    duration: float = Field(..., gt=0, description="Video duration")


class ErrorMessage(WebSocketMessage):
    """Error WebSocket message"""
    type: str = Field(default="error", const=True)
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")