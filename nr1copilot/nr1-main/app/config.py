"""
ViralClip Pro - Configuration Management
Netflix-level configuration with environment support
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # App settings
    app_name: str = "ViralClip Pro"
    version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=5000, env="PORT")

    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    frontend_url: str = Field(default="http://localhost:5000", env="FRONTEND_URL")

    # File upload settings
    max_file_size: int = Field(default=2 * 1024 * 1024 * 1024, env="MAX_FILE_SIZE")  # 2GB
    max_video_duration: int = Field(default=3600, env="MAX_VIDEO_DURATION")  # 1 hour
    allowed_video_formats: List[str] = Field(
        default=["mp4", "mov", "avi", "mkv", "webm"],
        env="ALLOWED_VIDEO_FORMATS"
    )

    # Directory settings
    upload_path: str = Field(default="uploads", env="UPLOAD_PATH")
    output_path: str = Field(default="output", env="OUTPUT_PATH")
    temp_path: str = Field(default="temp", env="TEMP_PATH")

    # Database (if needed)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")

    # Redis (for caching and sessions)
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # External services
    youtube_api_key: Optional[str] = Field(default=None, env="YOUTUBE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    # Processing settings
    max_concurrent_processes: int = Field(default=3, env="MAX_CONCURRENT_PROCESSES")
    default_video_quality: str = Field(default="high", env="DEFAULT_VIDEO_QUALITY")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/viralclip.log", env="LOG_FILE")

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# Environment-specific configurations
def get_development_settings() -> Settings:
    """Get development-specific settings"""
    settings = get_settings()
    settings.debug = True
    settings.log_level = "DEBUG"
    return settings

def get_production_settings() -> Settings:
    """Get production-specific settings"""
    settings = get_settings()
    settings.debug = False
    settings.log_level = "WARNING"
    return settings