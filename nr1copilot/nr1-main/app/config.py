"""
Netflix-Level Configuration Management
Environment-based configuration with validation
"""

import os
from functools import lru_cache
from typing import Optional, List
from pydantic import BaseSettings, Field, validator
from pathlib import Path


class Settings(BaseSettings):
    """Netflix-level application settings with validation"""

    # Application
    app_name: str = Field(default="ViralClip Pro v4.0", env="APP_NAME")
    app_version: str = Field(default="4.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=5000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")

    # Security
    secret_key: str = Field(default="netflix-level-secret-key", env="SECRET_KEY")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")

    # File Processing
    max_file_size: int = Field(default=2 * 1024 * 1024 * 1024, env="MAX_FILE_SIZE")  # 2GB
    allowed_video_types: List[str] = Field(
        default=["video/mp4", "video/mov", "video/avi", "video/webm", "video/quicktime"],
        env="ALLOWED_VIDEO_TYPES"
    )
    upload_path: str = Field(default="uploads", env="UPLOAD_PATH")
    output_path: str = Field(default="output", env="OUTPUT_PATH")
    temp_path: str = Field(default="temp", env="TEMP_PATH")

    # WebSocket
    websocket_heartbeat_interval: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")
    websocket_timeout: int = Field(default=300, env="WS_TIMEOUT")

    # Performance
    max_concurrent_uploads: int = Field(default=10, env="MAX_CONCURRENT_UPLOADS")
    processing_timeout: int = Field(default=600, env="PROCESSING_TIMEOUT")
    chunk_size: int = Field(default=8192, env="CHUNK_SIZE")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: bool = Field(default=True, env="LOG_FILE")

    # Monitoring
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    health_check_interval: int = Field(default=60, env="HEALTH_CHECK_INTERVAL")

    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    def get_upload_dir(self) -> Path:
        """Get upload directory path"""
        path = Path(self.upload_path)
        path.mkdir(exist_ok=True)
        return path

    def get_output_dir(self) -> Path:
        """Get output directory path"""
        path = Path(self.output_path)
        path.mkdir(exist_ok=True)
        return path

    def get_temp_dir(self) -> Path:
        """Get temporary directory path"""
        path = Path(self.temp_path)
        path.mkdir(exist_ok=True)
        return path

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()


# Global settings instance
settings = get_settings()