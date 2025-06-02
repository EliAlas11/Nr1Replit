"""
Optimized Pydantic schemas for deployment performance
Streamlined data validation with minimal overhead
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

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
    session_id: str = Field(..., description="Session identifier", min_length=8)
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")
    file_size: Optional[int] = Field(None, description="File size in bytes", ge=0)
    file_type: Optional[str] = Field(None, description="File MIME type")
    platforms: List[PlatformEnum] = Field(default_factory=list, description="Target platforms")

class AnalysisResponse(BaseModel):
    """Video analysis response schema"""
    success: bool = Field(..., description="Operation success status")
    session_id: str = Field(..., description="Session identifier")
    analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis results")
    viral_score: Optional[float] = Field(None, ge=0, le=100, description="Viral potential score")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")
    performance_grade: str = Field(default="A+", description="Performance grade")

class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = Field(default=False, description="Operation success status")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Error timestamp")
    support_id: Optional[str] = Field(None, description="Support ticket ID")

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Check timestamp")
    version: str = Field(..., description="Application version")
    startup_time: Optional[float] = Field(None, ge=0, description="Startup time in seconds")
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
    enterprise_features: bool = Field(default=True, description="Enterprise features enabled")

class AIRequest(BaseModel):
    """AI processing request schema"""
    brand_id: str = Field(..., description="Brand ID")
    content_type: str = Field(..., description="Content type")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context data")
    customization_level: str = Field(default="high", description="Customization level")

class AIResponse(BaseModel):
    """AI processing response schema"""
    success: bool = Field(..., description="Success status")
    content: Dict[str, Any] = Field(..., description="Generated content")
    model_version: str = Field(default="v10.0", description="Model version")

class WebSocketMessage(BaseModel):
    """WebSocket message schema"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Message timestamp")
    user_id: Optional[str] = Field(None, description="User ID")

class APIResponse(BaseModel):
    """Generic API response schema"""
    success: bool = Field(..., description="Success status")
    data: Optional[Any] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Response timestamp")

# Export schemas
__all__ = [
    'StatusEnum',
    'PlatformEnum', 
    'VideoRequest',
    'AnalysisResponse',
    'ErrorResponse',
    'HealthResponse',
    'TemplateRequest',
    'TemplateResponse',
    'AIRequest',
    'AIResponse',
    'WebSocketMessage',
    'APIResponse'
]