
"""
Application Configuration
Centralized configuration management for all environment variables
"""

from pydantic import BaseSettings, validator
from typing import List, Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_SECRET_KEY: str = "jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Database
    MONGODB_URI: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "viral_clips"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # AWS S3
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None
    
    # YouTube API
    YOUTUBE_API_KEY: Optional[str] = None
    
    # FFmpeg
    FFMPEG_PATH: str = "ffmpeg"
    
    # File handling
    MAX_FILE_SIZE_MB: int = 500
    ALLOWED_VIDEO_FORMATS: List[str] = ["mp4", "avi", "mov", "mkv", "webm"]
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    @validator("DEBUG")
    def validate_debug(cls, v, values):
        if values.get("ENVIRONMENT") == "production" and v:
            return False
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
