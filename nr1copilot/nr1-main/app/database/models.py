
"""
Netflix-Level Database Models
Complete schema for video editing platform
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class UserStatus(str, Enum):
    """User status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class VideoStatus(str, Enum):
    """Video processing status"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class ProjectStatus(str, Enum):
    """Project status enumeration"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class PublicationStatus(str, Enum):
    """Social media publication status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Database Models
class User(BaseModel):
    """User model"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    email: str = Field(..., unique=True, index=True)
    username: str = Field(..., unique=True, index=True)
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    subscription_tier: str = Field(default="free")
    status: UserStatus = Field(default=UserStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    # Settings and preferences
    preferences: Dict[str, Any] = Field(default_factory=dict)
    storage_used: int = Field(default=0)
    storage_limit: int = Field(default=1073741824)  # 1GB default
    
    class Config:
        table_name = "users"
        indexes = ["email", "username", "status", "created_at"]


class Video(BaseModel):
    """Video model"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(..., foreign_key="users.id", index=True)
    title: str = Field(..., max_length=255)
    description: Optional[str] = None
    
    # File information
    filename: str
    file_path: str
    file_size: int
    duration: Optional[float] = None
    resolution: Optional[str] = None
    format: Optional[str] = None
    
    # Processing information
    status: VideoStatus = Field(default=VideoStatus.UPLOADING)
    upload_progress: float = Field(default=0.0)
    processing_progress: float = Field(default=0.0)
    
    # AI Analysis results
    ai_analysis: Dict[str, Any] = Field(default_factory=dict)
    viral_score: Optional[float] = None
    engagement_prediction: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    thumbnail_url: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    
    class Config:
        table_name = "videos"
        indexes = ["user_id", "status", "created_at", "viral_score"]


class Project(BaseModel):
    """Video editing project model"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(..., foreign_key="users.id", index=True)
    title: str = Field(..., max_length=255)
    description: Optional[str] = None
    
    # Project configuration
    template_id: Optional[UUID] = None
    brand_kit_id: Optional[UUID] = None
    
    # Project data
    timeline_data: Dict[str, Any] = Field(default_factory=dict)
    assets: List[Dict[str, Any]] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Status and progress
    status: ProjectStatus = Field(default=ProjectStatus.DRAFT)
    progress: float = Field(default=0.0)
    
    # Collaboration
    collaborators: List[UUID] = Field(default_factory=list)
    permissions: Dict[str, Any] = Field(default_factory=dict)
    
    # Export information
    output_video_id: Optional[UUID] = None
    export_settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_edited: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        table_name = "projects"
        indexes = ["user_id", "status", "created_at", "updated_at"]


class SocialPublication(BaseModel):
    """Social media publication model"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(..., foreign_key="users.id", index=True)
    project_id: Optional[UUID] = Field(None, foreign_key="projects.id")
    video_id: Optional[UUID] = Field(None, foreign_key="videos.id")
    
    # Platform information
    platform: str = Field(..., max_length=50)
    platform_account_id: str
    
    # Content
    title: str = Field(..., max_length=255)
    description: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)
    
    # Publication settings
    scheduled_for: Optional[datetime] = None
    status: PublicationStatus = Field(default=PublicationStatus.PENDING)
    
    # Results
    platform_post_id: Optional[str] = None
    published_url: Optional[str] = None
    engagement_stats: Dict[str, Any] = Field(default_factory=dict)
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = Field(default=0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    
    class Config:
        table_name = "social_publications"
        indexes = ["user_id", "platform", "status", "scheduled_for", "created_at"]


class Template(BaseModel):
    """Video template model"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    creator_id: Optional[UUID] = Field(None, foreign_key="users.id")
    
    # Template information
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    category: str = Field(..., max_length=100)
    
    # Template data
    template_data: Dict[str, Any] = Field(default_factory=dict)
    preview_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    
    # Usage statistics
    usage_count: int = Field(default=0)
    rating: float = Field(default=0.0)
    viral_performance: Dict[str, Any] = Field(default_factory=dict)
    
    # Availability
    is_public: bool = Field(default=True)
    is_premium: bool = Field(default=False)
    price: Optional[float] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        table_name = "templates"
        indexes = ["category", "is_public", "is_premium", "usage_count", "rating"]


class Analytics(BaseModel):
    """Analytics events model"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: Optional[UUID] = Field(None, foreign_key="users.id", index=True)
    
    # Event information
    event_type: str = Field(..., max_length=100)
    event_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Context
    session_id: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Related entities
    video_id: Optional[UUID] = None
    project_id: Optional[UUID] = None
    template_id: Optional[UUID] = None
    
    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        table_name = "analytics"
        indexes = ["user_id", "event_type", "created_at", "session_id"]


class SystemHealth(BaseModel):
    """System health monitoring model"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Health check information
    service_name: str = Field(..., max_length=100)
    status: str = Field(..., max_length=50)
    response_time_ms: float
    
    # Metrics
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    
    # Details
    details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        table_name = "system_health"
        indexes = ["service_name", "status", "created_at"]
