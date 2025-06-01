"""
Configuration management for ViralClip Pro
Netflix-level configuration with environment variables
"""
import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with Netflix-level configuration"""

    # Application
    app_name: str = "ViralClip Pro"
    version: str = "3.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=5000, env="PORT")

    # Security
    secret_key: str = Field(default="viralclip-pro-secret-key-change-in-production", env="SECRET_KEY")
    jwt_secret: str = Field(default="jwt-secret-key", env="JWT_SECRET")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")

    # Database
    mongodb_uri: str = Field(default="mongodb://localhost:27017", env="MONGODB_URI")
    database_name: str = Field(default="viralclip_pro", env="DATABASE_NAME")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")

    # Processing limits
    max_video_duration: int = Field(default=1800, env="MAX_VIDEO_DURATION")  # 30 minutes
    max_file_size: int = Field(default=536870912, env="MAX_FILE_SIZE")  # 512MB
    max_concurrent_jobs: int = Field(default=10, env="MAX_CONCURRENT_JOBS")

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")

    # File paths
    upload_path: str = Field(default="uploads", env="UPLOAD_PATH")
    video_storage_path: str = Field(default="videos", env="VIDEO_STORAGE_PATH")
    temp_path: str = Field(default="temp", env="TEMP_PATH")

    # Cache settings
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour

    # AI settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def get_database_url() -> str:
    """Get formatted database URL"""
    settings = get_settings()
    return f"{settings.mongodb_uri}/{settings.database_name}"

def is_production() -> bool:
    """Check if running in production"""
    return get_settings().environment == "production"

def is_development() -> bool:
    """Check if running in development"""
    return get_settings().environment == "development"