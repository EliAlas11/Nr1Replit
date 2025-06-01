
"""
Enhanced Configuration Management
Handles all application settings with environment variable support
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Enhanced application settings with comprehensive configuration"""
    
    # Basic app settings
    app_name: str = "ViralClip Pro"
    version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=5000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Security settings
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # File upload settings
    max_file_size: int = Field(default=2 * 1024 * 1024 * 1024, env="MAX_FILE_SIZE")  # 2GB
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    output_dir: str = Field(default="output", env="OUTPUT_DIR")
    temp_dir: str = Field(default="temp", env="TEMP_DIR")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_path: str = Field(default="logs", env="LOG_PATH")
    
    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Database settings (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # AI/ML settings
    ai_model_path: str = Field(default="models/", env="AI_MODEL_PATH")
    enable_gpu: bool = Field(default=False, env="ENABLE_GPU")
    
    # Video processing settings
    ffmpeg_path: str = Field(default="ffmpeg", env="FFMPEG_PATH")
    max_processing_time: int = Field(default=300, env="MAX_PROCESSING_TIME")  # 5 minutes
    quality_profiles: dict = Field(default={
        "draft": {"crf": 28, "preset": "ultrafast"},
        "standard": {"crf": 23, "preset": "medium"},
        "high": {"crf": 18, "preset": "slow"},
        "premium": {"crf": 15, "preset": "veryslow"}
    })
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    # Monitoring and logging
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_analytics: bool = Field(default=False, env="ENABLE_ANALYTICS")
    
    # External services
    youtube_api_key: Optional[str] = Field(default=None, env="YOUTUBE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    @validator('cors_origins', 'allowed_hosts', pre=True)
    def split_list_strings(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level. Must be one of: {valid_levels}')
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing)"""
    global _settings
    _settings = None
    return get_settings()
