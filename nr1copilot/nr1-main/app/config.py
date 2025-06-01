
"""
Production-grade configuration management with environment validation
Supports multiple deployment environments with secure defaults
"""

import os
from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    """Application settings with validation and environment support"""
    
    # Environment
    environment: str = Field(default="development", env="ENV")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=5000, env="PORT")
    
    # Security
    jwt_secret: str = Field(..., env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # Database
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    database_name: str = Field(default="viralclip", env="DATABASE_NAME")
    
    # Redis
    redis_url: str = Field(..., env="REDIS_URL")
    
    # AWS S3
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    s3_bucket_name: Optional[str] = Field(default=None, env="S3_BUCKET_NAME")
    
    # File storage
    video_storage_path: str = Field(default="videos", env="VIDEO_STORAGE_PATH")
    upload_storage_path: str = Field(default="uploads", env="UPLOAD_STORAGE_PATH")
    max_file_size: int = Field(default=500 * 1024 * 1024, env="MAX_FILE_SIZE")  # 500MB
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    
    # External APIs
    youtube_api_key: Optional[str] = Field(default=None, env="YOUTUBE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Processing
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    processing_timeout: int = Field(default=300, env="PROCESSING_TIMEOUT")  # 5 minutes
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("allowed_hosts", pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v
    
    @validator("port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Validate critical settings on import
try:
    settings = get_settings()
    print(f"✅ Configuration loaded successfully for {settings.environment} environment")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    raise
