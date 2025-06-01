"""
ViralClip Pro - Configuration Management
Production-ready configuration with proper Pydantic v2 support
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application Settings
    APP_NAME: str = "ViralClip Pro"
    APP_VERSION: str = "3.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=5000, env="PORT")

    # Security Settings
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    JWT_SECRET: str = Field(default="dev-jwt-secret", env="JWT_SECRET")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440

    # CORS Settings
    ALLOWED_HOSTS: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")

    # Database Settings
    DATABASE_URL: str = Field(default="sqlite:///./viralclip.db", env="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")

    # Video Processing Settings
    MAX_VIDEO_DURATION: int = Field(default=3600, env="MAX_VIDEO_DURATION")  # 1 hour
    MAX_FILE_SIZE: int = Field(default=2147483648, env="MAX_FILE_SIZE")  # 2GB
    MAX_CLIP_DURATION: int = Field(default=300, env="MAX_CLIP_DURATION")  # 5 minutes
    MIN_CLIP_DURATION: int = Field(default=5, env="MIN_CLIP_DURATION")
    CONCURRENT_PROCESSING: int = Field(default=4, env="CONCURRENT_PROCESSING")
    ALLOWED_VIDEO_FORMATS: List[str] = Field(
        default=["mp4", "mov", "avi", "mkv", "webm", "flv"],
        env="ALLOWED_VIDEO_FORMATS"
    )

    # Storage Settings
    UPLOAD_PATH: str = Field(default="uploads", env="UPLOAD_PATH")
    TEMP_PATH: str = Field(default="temp", env="TEMP_PATH")
    OUTPUT_PATH: str = Field(default="output", env="OUTPUT_PATH")
    VIDEO_STORAGE_PATH: str = Field(default="videos", env="VIDEO_STORAGE_PATH")
    LOG_PATH: str = Field(default="logs", env="LOG_PATH")

    # Cache Settings
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    ENABLE_CACHE: bool = Field(default=True, env="ENABLE_CACHE")

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")

    # Feature Flags
    ENABLE_WEBSOCKETS: bool = Field(default=True, env="ENABLE_WEBSOCKETS")
    ENABLE_FILE_UPLOAD: bool = Field(default=True, env="ENABLE_FILE_UPLOAD")
    ENABLE_BATCH_PROCESSING: bool = Field(default=True, env="ENABLE_BATCH_PROCESSING")
    ENABLE_SOCIAL_SHARING: bool = Field(default=True, env="ENABLE_SOCIAL_SHARING")
    ENABLE_ANALYTICS: bool = Field(default=True, env="ENABLE_ANALYTICS")

    # External APIs
    YOUTUBE_API_KEY: Optional[str] = Field(default=None, env="YOUTUBE_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    # Performance Settings
    WORKER_PROCESSES: int = Field(default=1, env="WORKER_PROCESSES")
    WORKER_CONNECTIONS: int = Field(default=1000, env="WORKER_CONNECTIONS")
    KEEPALIVE_TIMEOUT: int = Field(default=65, env="KEEPALIVE_TIMEOUT")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    return settings

# Environment-specific configurations
def is_development() -> bool:
    """Check if running in development mode"""
    return settings.ENVIRONMENT.lower() in ["development", "dev"]

def is_production() -> bool:
    """Check if running in production mode"""
    return settings.ENVIRONMENT.lower() in ["production", "prod"]

def is_testing() -> bool:
    """Check if running in test mode"""
    return settings.ENVIRONMENT.lower() in ["testing", "test"]