"""
ViralClip Pro - Netflix-Level Configuration Management
Advanced configuration with environment validation and security
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Netflix-level configuration management with validation"""

    # Application settings
    app_name: str = Field(default="ViralClip Pro", description="Application name")
    app_version: str = Field(default="3.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment (development/staging/production)")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")

    # Security settings
    secret_key: str = Field(default="fallback-secret-key-change-in-production", min_length=32)
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed hosts")
    cors_origins: List[str] = Field(default=["*"], description="CORS origins")
    csrf_protection: bool = Field(default=True, description="Enable CSRF protection")

    # Database and caching
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, description="Rate limit per minute")
    rate_limit_per_hour: int = Field(default=1000, description="Rate limit per hour")
    rate_limit_burst: int = Field(default=20, description="Rate limit burst")

    # File handling
    upload_path: str = Field(default="uploads", description="Upload directory")
    video_storage_path: str = Field(default="videos", description="Video storage directory")
    temp_path: str = Field(default="temp", description="Temporary files directory")
    output_path: str = Field(default="output", description="Output directory")
    max_file_size: int = Field(default=2147483648, description="Max file size in bytes (2GB)")
    allowed_video_formats: List[str] = Field(
        default=["mp4", "mov", "avi", "mkv", "webm", "flv"],
        description="Allowed video formats"
    )

    # Processing settings
    max_video_duration: int = Field(default=3600, description="Max video duration in seconds")
    max_concurrent_jobs: int = Field(default=5, description="Max concurrent processing jobs")
    processing_timeout: int = Field(default=1800, description="Processing timeout in seconds")
    queue_timeout: int = Field(default=3600, description="Queue timeout in seconds")

    # AI and ML settings
    ai_model_path: str = Field(default="models", description="AI model directory")
    enable_ai_analysis: bool = Field(default=True, description="Enable AI analysis")
    ai_confidence_threshold: float = Field(default=0.8, description="AI confidence threshold")
    viral_score_threshold: int = Field(default=70, description="Viral score threshold")

    # External services
    youtube_api_key: Optional[str] = Field(default=None, description="YouTube API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    aws_access_key: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_key: Optional[str] = Field(default=None, description="AWS secret key")
    aws_region: str = Field(default="us-east-1", description="AWS region")

    # Monitoring and logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")

    # Performance settings
    enable_compression: bool = Field(default=True, description="Enable response compression")
    compression_level: int = Field(default=6, description="Compression level")
    enable_caching: bool = Field(default=True, description="Enable caching")
    cache_middleware: bool = Field(default=True, description="Enable cache middleware")

    # Feature flags
    enable_websockets: bool = Field(default=True, description="Enable WebSocket support")
    enable_file_upload: bool = Field(default=True, description="Enable file upload")
    enable_social_sharing: bool = Field(default=True, description="Enable social media sharing")
    enable_analytics: bool = Field(default=True, description="Enable analytics")
    enable_notifications: bool = Field(default=True, description="Enable notifications")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of: {valid_envs}')
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

    @validator('ai_confidence_threshold')
    def validate_ai_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('AI confidence threshold must be between 0.0 and 1.0')
        return v

    @validator('viral_score_threshold')
    def validate_viral_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Viral score threshold must be between 0 and 100')
        return v

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"

    def get_cors_settings(self) -> dict:
        """Get CORS settings based on environment"""
        if self.is_production():
            return {
                "allow_origins": self.cors_origins,
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "PUT", "DELETE"],
                "allow_headers": ["*"],
            }
        else:
            return {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            }

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    try:
        settings = Settings()
        logger.info(f"Configuration loaded for environment: {settings.environment}")
        return settings
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Return minimal settings for fallback
        return Settings()

def is_production() -> bool:
    """Check if running in production"""
    return get_settings().is_production()

def is_development() -> bool:
    """Check if running in development"""
    return get_settings().is_development()

# Export commonly used settings
settings = get_settings()