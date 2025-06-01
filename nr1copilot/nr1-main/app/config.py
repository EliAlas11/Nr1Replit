
"""
Application Configuration - Netflix-Level Implementation
Handles all environment variables and settings with proper validation
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "ViralClip Pro - SendShort.ai Killer"
    version: str = "3.0.0"
    debug: bool = Field(default=False, alias="DEBUG")
    environment: str = Field(default="production", alias="ENVIRONMENT")
    
    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=5000, alias="PORT")
    
    # Security - Netflix-level security
    secret_key: str = Field(default="ultra-secure-netflix-level-key-2024", alias="SECRET_KEY")
    jwt_secret: str = Field(default="jwt-ultra-secure-netflix-2024", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database - Production ready
    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    database_name: str = Field(default="viralclip_pro", alias="DATABASE_NAME")
    
    # Redis for caching and queues
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    
    # Storage - Cloud-native
    video_storage_path: str = Field(default="./videos", alias="VIDEO_STORAGE_PATH")
    upload_path: str = Field(default="./uploads", alias="UPLOAD_PATH")
    temp_path: str = Field(default="./temp", alias="TEMP_PATH")
    
    # External APIs - AI Services
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    google_ai_key: str = Field(default="", alias="GOOGLE_AI_KEY")
    
    # Cloud Storage - Netflix-level scalability
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    s3_bucket: str = Field(default="viralclip-pro-storage", alias="S3_BUCKET")
    
    # CDN Configuration
    cloudflare_zone_id: str = Field(default="", alias="CLOUDFLARE_ZONE_ID")
    cloudflare_api_token: str = Field(default="", alias="CLOUDFLARE_API_TOKEN")
    
    # CORS - Production ready
    cors_origins: List[str] = Field(
        default=["https://viralclippro.com", "https://*.viralclippro.com"], 
        alias="CORS_ORIGINS"
    )
    
    # Security
    allowed_hosts: List[str] = Field(
        default=["viralclippro.com", "*.viralclippro.com"], 
        alias="ALLOWED_HOSTS"
    )
    
    # Rate limiting - Netflix-level performance
    rate_limit_per_minute: int = Field(default=120, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=200, alias="RATE_LIMIT_BURST")
    
    # Video processing - Advanced limits
    max_video_duration: int = Field(default=3600, alias="MAX_VIDEO_DURATION")  # 1 hour
    max_file_size: int = Field(default=2 * 1024 * 1024 * 1024, alias="MAX_FILE_SIZE")  # 2GB
    max_concurrent_processes: int = Field(default=10, alias="MAX_CONCURRENT_PROCESSES")
    
    # AI Processing
    ai_processing_timeout: int = Field(default=300, alias="AI_PROCESSING_TIMEOUT")  # 5 minutes
    enable_gpu_acceleration: bool = Field(default=True, alias="ENABLE_GPU_ACCELERATION")
    
    # Monitoring - Netflix-level observability
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")
    
    # Caching
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")  # 1 hour
    enable_redis_cache: bool = Field(default=True, alias="ENABLE_REDIS_CACHE")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('allowed_hosts', pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(',')]
        return v

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
