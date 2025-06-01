
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
    """Application settings with environment variable support and validation"""
    
    # Application
    APP_NAME: str = "Viral Clip Generator"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENV")
    DEBUG: bool = Field(default=False)
    
    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=5000, env="PORT")
    
    # Security
    JWT_SECRET: str = Field(..., env="JWT_SECRET")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    MONGODB_URI: Optional[str] = Field(None, env="MONGODB_URI")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # File Storage
    VIDEO_STORAGE_PATH: str = Field(default="./videos", env="VIDEO_STORAGE_PATH")
    MAX_FILE_SIZE: int = Field(default=500 * 1024 * 1024)  # 500MB
    ALLOWED_VIDEO_FORMATS: List[str] = ["mp4", "avi", "mov", "mkv"]
    
    # AWS S3
    AWS_ACCESS_KEY_ID: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    AWS_S3_BUCKET: Optional[str] = Field(None, env="AWS_S3_BUCKET")
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    
    # External APIs
    YOUTUBE_API_KEY: Optional[str] = Field(None, env="YOUTUBE_API_KEY")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE_PATH: Optional[str] = Field(None, env="LOG_FILE_PATH")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    
    # Video Processing
    MAX_VIDEO_DURATION: int = Field(default=3600)  # 1 hour
    DEFAULT_VIDEO_QUALITY: str = "720p"
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        allowed_envs = ["development", "production", "testing"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of {allowed_envs}")
        return v
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    @property
    def database_url_async(self) -> str:
        """Convert sync database URL to async"""
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        return self.DATABASE_URL
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
