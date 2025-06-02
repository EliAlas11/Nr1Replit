"""
Pydantic schemas for API validation and serialization
Netflix-level data validation and type safety
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid


class StatusEnum(str, Enum):
    """Status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"


class PlatformEnum(str, Enum):
    """Social media platforms"""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"


class VideoRequest(BaseModel):
    """Video processing request schema"""
    file_url: Optional[str] = Field(None, description="URL of the video file")
    file_name: Optional[str] = Field(None, description="Name of the uploaded file")
    session_id: str = Field(..., description="Session ID for tracking")
    platforms: List[PlatformEnum] = Field(default=[], description="Target platforms")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")

    @validator('session_id')
    def validate_session_id(cls, v):
        if not v or len(v) < 8:
            raise ValueError('Session ID must be at least 8 characters')
        return v


class AnalysisResponse(BaseModel):
    """Video analysis response schema"""
    success: bool = Field(..., description="Success status")
    session_id: str = Field(..., description="Session ID")
    analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis results")
    viral_score: Optional[float] = Field(None, ge=0, le=100, description="Viral potential score")
    processing_time: Optional[float] = Field(None, ge=0, description="Processing time in seconds")
    recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = Field(False, description="Success status")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    uptime: float = Field(..., ge=0, description="Uptime in seconds")
    version: str = Field(..., description="Application version")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Health metrics")
    services: Dict[str, str] = Field(default_factory=dict, description="Service statuses")


class TemplateRequest(BaseModel):
    """Template request schema"""
    category: Optional[str] = Field(None, description="Template category")
    platform: Optional[PlatformEnum] = Field(None, description="Target platform")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of templates")
    search: Optional[str] = Field(None, description="Search query")


class TemplateResponse(BaseModel):
    """Template response schema"""
    success: bool = Field(..., description="Success status")
    templates: List[Dict[str, Any]] = Field(default_factory=list, description="Template list")
    total: int = Field(default=0, ge=0, description="Total number of templates")
    page: int = Field(default=1, ge=1, description="Current page")
    has_more: bool = Field(default=False, description="More templates available")


class CollaborationRequest(BaseModel):
    """Collaboration request schema"""
    workspace_id: str = Field(..., description="Workspace ID")
    project_id: str = Field(..., description="Project ID")
    user_id: str = Field(..., description="User ID")
    action: str = Field(..., description="Collaboration action")
    data: Dict[str, Any] = Field(default_factory=dict, description="Action data")


class AIRequest(BaseModel):
    """AI processing request schema"""
    brand_id: str = Field(..., description="Brand ID")
    content_type: str = Field(..., description="Content type")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context data")
    customization_level: str = Field(default="high", description="Customization level")


class BatchJobRequest(BaseModel):
    """Batch job request schema"""
    job_type: str = Field(..., description="Job type")
    input_data: Dict[str, Any] = Field(..., description="Input data")
    priority: str = Field(default="normal", description="Job priority")
    callback_url: Optional[str] = Field(None, description="Callback URL")


class BatchJobResponse(BaseModel):
    """Batch job response schema"""
    success: bool = Field(..., description="Success status")
    job_id: str = Field(..., description="Job ID")
    status: StatusEnum = Field(..., description="Job status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion")


class MetricsResponse(BaseModel):
    """Metrics response schema"""
    success: bool = Field(..., description="Success status")
    metrics: Dict[str, Any] = Field(..., description="Metrics data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    period: str = Field(..., description="Metrics period")


class WebSocketMessage(BaseModel):
    """WebSocket message schema"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    user_id: Optional[str] = Field(None, description="User ID")


class PaginationParams(BaseModel):
    """Pagination parameters schema"""
    page: int = Field(default=1, ge=1, description="Page number")
    limit: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order")

    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v not in ['asc', 'desc']:
            raise ValueError('Sort order must be "asc" or "desc"')
        return v


class APIResponse(BaseModel):
    """Generic API response schema"""
    success: bool = Field(..., description="Success status")
    data: Optional[Any] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    meta: Optional[Dict[str, Any]] = Field(None, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


# Custom validators
def validate_uuid(v: str) -> str:
    """Validate UUID format"""
    try:
        uuid.UUID(v)
        return v
    except ValueError:
        raise ValueError('Invalid UUID format')


def validate_platform_list(v: List[str]) -> List[PlatformEnum]:
    """Validate platform list"""
    return [PlatformEnum(platform) for platform in v]


# Export all schemas
__all__ = [
    'StatusEnum',
    'PlatformEnum',
    'VideoRequest',
    'AnalysisResponse',
    'ErrorResponse',
    'HealthResponse',
    'TemplateRequest',
    'TemplateResponse',
    'CollaborationRequest',
    'AIRequest',
    'BatchJobRequest',
    'BatchJobResponse',
    'MetricsResponse',
    'WebSocketMessage',
    'PaginationParams',
    'APIResponse',
    'validate_uuid',
    'validate_platform_list'
]