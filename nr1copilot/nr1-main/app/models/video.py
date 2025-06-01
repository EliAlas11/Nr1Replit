
"""
Video database model and schemas
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Integer, Float, Boolean, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, HttpUrl
import uuid
from enum import Enum

from app.db.session import Base


class VideoStatus(str, Enum):
    """Video processing status"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoQuality(str, Enum):
    """Video quality options"""
    LOW = "360p"
    MEDIUM = "720p"
    HIGH = "1080p"
    ULTRA = "4k"


class Video(Base):
    """Video database model"""
    __tablename__ = "videos"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    youtube_url = Column(String, nullable=False)
    youtube_id = Column(String, nullable=False, index=True)
    
    # File paths
    original_path = Column(String, nullable=True)
    processed_path = Column(String, nullable=True)
    thumbnail_path = Column(String, nullable=True)
    
    # Video properties
    duration = Column(Float, nullable=True)  # seconds
    quality = Column(String, default=VideoQuality.MEDIUM.value)
    file_size = Column(Integer, nullable=True)  # bytes
    
    # Processing
    status = Column(String, default=VideoStatus.PENDING.value)
    progress = Column(Float, default=0.0)  # 0-100
    error_message = Column(Text, nullable=True)
    
    # Clip settings
    start_time = Column(Float, default=0.0)
    end_time = Column(Float, nullable=True)
    audio_effects = Column(JSONB, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="videos")


# Pydantic schemas
class VideoBase(BaseModel):
    """Base video schema"""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    youtube_url: HttpUrl


class VideoCreate(VideoBase):
    """Video creation schema"""
    quality: VideoQuality = VideoQuality.MEDIUM
    start_time: float = Field(default=0.0, ge=0)
    end_time: Optional[float] = Field(None, gt=0)
    audio_effects: Optional[Dict[str, Any]] = None


class VideoUpdate(BaseModel):
    """Video update schema"""
    title: Optional[str] = None
    description: Optional[str] = None
    quality: Optional[VideoQuality] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class VideoOut(VideoBase):
    """Video output schema"""
    id: str
    youtube_id: str
    status: VideoStatus
    progress: float
    duration: Optional[float]
    quality: str
    file_size: Optional[int]
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True


class VideoProcessRequest(BaseModel):
    """Video processing request"""
    video_id: str
    priority: int = Field(default=1, ge=1, le=5)
