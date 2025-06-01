
"""
Configuration management for ViralClip Pro
Netflix-level configuration with environment-based settings
"""

import os
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Netflix-level application settings"""
    
    # Application Settings
    app_name: str = Field(default="ViralClip Pro", description="Application name")
    app_version: str = Field(default="3.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")
    host: str = Field(default="0.0.0.0", description="Host address")
    port: int = Field(default=5000, description="Port number")
    
    # Security Settings
    secret_key: str = Field(default="your-super-secret-key-change-in-production", description="Secret key for sessions")
    jwt_secret: str = Field(default="your-jwt-secret-key", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=1440, description="JWT expiration in minutes")  # 24 hours
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed hosts for CORS")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Database Settings
    database_url: str = Field(default="sqlite:///./viralclip.db", description="Database URL")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    
    # Video Processing Settings
    max_video_duration: int = Field(default=3600, description="Maximum video duration in seconds")  # 1 hour
    max_file_size: int = Field(default=2147483648, description="Maximum file size in bytes")  # 2GB
    max_clip_duration: int = Field(default=300, description="Maximum clip duration in seconds")  # 5 minutes
    min_clip_duration: int = Field(default=5, description="Minimum clip duration in seconds")
    supported_formats: List[str] = Field(default=["mp4", "avi", "mov", "mkv", "webm"], description="Supported video formats")
    
    # Processing Settings
    concurrent_processing: int = Field(default=4, description="Number of concurrent processing tasks")
    processing_timeout: int = Field(default=1800, description="Processing timeout in seconds")  # 30 minutes
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed processing")
    queue_timeout: int = Field(default=3600, description="Queue timeout in seconds")  # 1 hour
    
    # Storage Settings
    upload_path: str = Field(default="uploads", description="Upload directory path")
    temp_path: str = Field(default="temp", description="Temporary files directory")
    output_path: str = Field(default="output", description="Output files directory")
    video_storage_path: str = Field(default="videos", description="Video storage directory")
    log_path: str = Field(default="logs", description="Log files directory")
    
    # Cache Settings
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")  # 1 hour
    cache_max_size: int = Field(default=1000, description="Maximum cache entries")
    enable_cache: bool = Field(default=True, description="Enable caching")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=100, description="Rate limit per minute per IP")
    rate_limit_per_hour: int = Field(default=1000, description="Rate limit per hour per IP")
    rate_limit_per_day: int = Field(default=10000, description="Rate limit per day per IP")
    
    # AI Settings
    ai_model_name: str = Field(default="gpt-3.5-turbo", description="AI model for analysis")
    ai_api_key: Optional[str] = Field(default=None, description="AI API key")
    ai_max_tokens: int = Field(default=2048, description="Maximum AI tokens")
    ai_temperature: float = Field(default=0.7, description="AI temperature")
    enable_ai_analysis: bool = Field(default=True, description="Enable AI analysis")
    
    # Monitoring & Logging
    log_level: str = Field(default="INFO", description="Logging level")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics endpoint port")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    
    # External Services
    webhook_timeout: int = Field(default=30, description="Webhook timeout in seconds")
    youtube_api_key: Optional[str] = Field(default=None, description="YouTube API key")
    social_media_apis: dict = Field(default_factory=dict, description="Social media API configurations")
    
    # Feature Flags
    enable_websockets: bool = Field(default=True, description="Enable WebSocket support")
    enable_file_upload: bool = Field(default=True, description="Enable file upload")
    enable_batch_processing: bool = Field(default=True, description="Enable batch processing")
    enable_social_sharing: bool = Field(default=True, description="Enable social media sharing")
    enable_analytics: bool = Field(default=True, description="Enable analytics")
    
    # Performance Settings
    worker_processes: int = Field(default=1, description="Number of worker processes")
    worker_connections: int = Field(default=1000, description="Worker connections")
    keepalive_timeout: int = Field(default=65, description="Keep-alive timeout")
    max_request_size: int = Field(default=104857600, description="Maximum request size in bytes")  # 100MB
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed_envs = ['development', 'staging', 'production', 'testing']
        if v not in allowed_envs:
            raise ValueError(f'Environment must be one of: {allowed_envs}')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f'Log level must be one of: {allowed_levels}')
        return v.upper()
    
    @validator('port')
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('max_file_size', 'max_request_size')
    def validate_file_sizes(cls, v):
        if v <= 0:
            raise ValueError('File size must be positive')
        return v
    
    @validator('ai_temperature')
    def validate_ai_temperature(cls, v):
        if not (0.0 <= v <= 2.0):
            raise ValueError('AI temperature must be between 0.0 and 2.0')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable prefixes
        env_prefix = ""
        
        # Field aliases for environment variables
        fields = {
            'app_name': {'env': 'APP_NAME'},
            'app_version': {'env': 'APP_VERSION'},
            'debug': {'env': 'DEBUG'},
            'environment': {'env': 'ENVIRONMENT'},
            'host': {'env': 'HOST'},
            'port': {'env': 'PORT'},
            'secret_key': {'env': 'SECRET_KEY'},
            'jwt_secret': {'env': 'JWT_SECRET'},
            'database_url': {'env': 'DATABASE_URL'},
            'redis_url': {'env': 'REDIS_URL'},
            'ai_api_key': {'env': 'AI_API_KEY'},
            'youtube_api_key': {'env': 'YOUTUBE_API_KEY'},
        }

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def is_production() -> bool:
    """Check if running in production environment"""
    return get_settings().environment == "production"

def is_development() -> bool:
    """Check if running in development environment"""
    return get_settings().environment == "development"

def is_testing() -> bool:
    """Check if running in testing environment"""
    return get_settings().environment == "testing"

# Development-specific overrides
if is_development():
    # Override some settings for development
    settings = get_settings()
    settings.debug = True
    settings.log_level = "DEBUG"
    settings.allowed_hosts = ["*"]
    settings.cors_origins = ["*"]

# Production-specific security
if is_production():
    settings = get_settings()
    settings.debug = False
    settings.log_level = "INFO"
    # Ensure secure settings are properly configured
    if settings.secret_key == "your-super-secret-key-change-in-production":
        raise ValueError("SECRET_KEY must be changed in production!")
    if settings.jwt_secret == "your-jwt-secret-key":
        raise ValueError("JWT_SECRET must be changed in production!")
