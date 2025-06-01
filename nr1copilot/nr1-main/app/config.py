"""
Production-grade configuration management with comprehensive settings
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with validation and environment variable support"""

    # Application
    app_name: str = "Viral Clip Generator API"
    version: str = "2.0.0"
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=5000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")

    # Security
    secret_key: str = Field(env="SECRET_KEY", default="your-secret-key-change-in-production")
    jwt_secret: str = Field(env="JWT_SECRET", default="your-jwt-secret-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 7 days

    # Database
    mongodb_uri: str = Field(env="MONGODB_URI", default="mongodb://localhost:27017/viral_clips")
    database_name: str = Field(env="DATABASE_NAME", default="viral_clips")

    # Redis
    redis_url: str = Field(env="REDIS_URL", default="redis://localhost:6379/0")

    # File Storage
    upload_path: str = "uploads"
    video_path: str = "videos"
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    allowed_video_types: List[str] = ["mp4", "avi", "mov", "mkv", "webm"]

    # AWS S3 (Optional)
    aws_access_key_id: Optional[str] = Field(env="AWS_ACCESS_KEY_ID", default=None)
    aws_secret_access_key: Optional[str] = Field(env="AWS_SECRET_ACCESS_KEY", default=None)
    aws_region: str = Field(env="AWS_REGION", default="us-east-1")
    s3_bucket: Optional[str] = Field(env="S3_BUCKET", default=None)

    # Video Processing
    max_video_duration: int = 3600  # 1 hour
    output_formats: List[str] = ["mp4", "webm"]
    video_quality: str = "720p"

    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 100

    # CORS
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )

    # Trusted Hosts
    allowed_hosts: List[str] = Field(
        default=["*"],
        env="ALLOWED_HOSTS"
    )

    # API Keys
    youtube_api_key: Optional[str] = Field(env="YOUTUBE_API_KEY", default=None)
    openai_api_key: Optional[str] = Field(env="OPENAI_API_KEY", default=None)

    # Monitoring
    sentry_dsn: Optional[str] = Field(env="SENTRY_DSN", default=None)
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()

# Global settings instance
settings = get_settings()