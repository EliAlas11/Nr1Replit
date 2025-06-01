
<old_str></old_str>
<new_str>"""
Application Configuration
Handles all environment variables and settings with proper validation
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "Viral Clip Generator API"
    version: str = "2.0.0"
    debug: bool = Field(default=False, alias="DEBUG")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    
    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=5000, alias="PORT")
    
    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production", alias="SECRET_KEY")
    jwt_secret: str = Field(default="jwt-secret-key-change-in-production", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database
    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    database_name: str = Field(default="viral_clips", alias="DATABASE_NAME")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    
    # Storage
    video_storage_path: str = Field(default="./videos", alias="VIDEO_STORAGE_PATH")
    upload_path: str = Field(default="./uploads", alias="UPLOAD_PATH")
    
    # External APIs
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    s3_bucket: str = Field(default="", alias="S3_BUCKET")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["*"], 
        alias="CORS_ORIGINS"
    )
    
    # Security
    allowed_hosts: List[str] = Field(
        default=["*"], 
        alias="ALLOWED_HOSTS"
    )
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")
    
    # Video processing
    max_video_duration: int = Field(default=300, alias="MAX_VIDEO_DURATION")  # 5 minutes
    max_file_size: int = Field(default=100 * 1024 * 1024, alias="MAX_FILE_SIZE")  # 100MB
    
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
    return _settings</new_str>
