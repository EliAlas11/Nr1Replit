"""
Netflix-Level Configuration Management
Enterprise-grade configuration with validation and hot-reloading
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import lru_cache
from pydantic import BaseSettings, validator, Field
from pydantic_settings import SettingsConfigDict


class NetflixLevelSettings(BaseSettings):
    """Netflix-level configuration with comprehensive validation"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="forbid"
    )

    # Application Settings
    app_name: str = Field(default="ViralClip Pro v6.0", description="Application name")
    app_version: str = Field(default="6.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    testing: bool = Field(default=False, description="Testing mode")

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5000, ge=1024, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=8, description="Number of workers")

    # Security Settings
    secret_key: str = Field(default_factory=lambda: os.urandom(32).hex(), description="Secret key")
    allowed_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    require_auth: bool = Field(default=False, description="Require authentication")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, ge=1, description="JWT expiration minutes")

    # Performance Settings
    max_file_size: int = Field(default=2_147_483_648, ge=1, description="Max file size (2GB)")
    chunk_size: int = Field(default=5_242_880, ge=1024, description="Chunk size (5MB)")
    max_concurrent_uploads: int = Field(default=10, ge=1, le=50, description="Max concurrent uploads")
    cache_ttl: int = Field(default=3600, ge=60, description="Cache TTL seconds")
    rate_limit_requests: int = Field(default=100, ge=1, description="Rate limit requests per minute")

    # Database Configuration (for future use)
    database_url: Optional[str] = Field(default=None, description="Database URL")
    redis_url: Optional[str] = Field(default=None, description="Redis URL")

    # AI/ML Configuration
    ai_model_path: Optional[str] = Field(default=None, description="AI model path")
    ai_batch_size: int = Field(default=32, ge=1, le=256, description="AI batch size")
    ai_timeout: int = Field(default=300, ge=30, description="AI processing timeout")

    # Storage Paths
    base_path: Path = Field(default=Path("nr1copilot/nr1-main"), description="Base application path")

    @property
    def upload_path(self) -> Path:
        return self.base_path / "uploads"

    @property
    def output_path(self) -> Path:
        return self.base_path / "output"

    @property
    def temp_path(self) -> Path:
        return self.base_path / "temp"

    @property
    def cache_path(self) -> Path:
        return self.base_path / "cache"

    @property
    def logs_path(self) -> Path:
        return self.base_path / "logs"

    # Monitoring and Observability
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    log_level: str = Field(default="INFO", description="Logging level")
    structured_logging: bool = Field(default=True, description="Enable structured logging")

    # Feature Flags
    enable_preview_generation: bool = Field(default=True, description="Enable preview generation")
    enable_real_time_updates: bool = Field(default=True, description="Enable real-time updates")
    enable_ai_analysis: bool = Field(default=True, description="Enable AI analysis")
    enable_cloud_processing: bool = Field(default=False, description="Enable cloud processing")

    # Supported File Formats
    supported_video_formats: List[str] = Field(
        default=["mp4", "avi", "mov", "webm", "mkv", "m4v", "3gp"],
        description="Supported video formats"
    )
    supported_audio_formats: List[str] = Field(
        default=["mp3", "wav", "m4a", "aac", "flac"],
        description="Supported audio formats"
    )

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @validator("base_path")
    def validate_base_path(cls, v):
        if not isinstance(v, Path):
            v = Path(v)
        return v.resolve()

    @validator("supported_video_formats", "supported_audio_formats")
    def validate_formats(cls, v):
        return [fmt.lower().strip('.') for fmt in v]

    def get_all_supported_formats(self) -> List[str]:
        """Get all supported file formats"""
        return self.supported_video_formats + self.supported_audio_formats

    def is_supported_format(self, filename: str) -> bool:
        """Check if file format is supported"""
        extension = Path(filename).suffix.lower().strip('.')
        return extension in self.get_all_supported_formats()

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        return {
            "max_file_size": self.max_file_size,
            "chunk_size": self.chunk_size,
            "max_concurrent_uploads": self.max_concurrent_uploads,
            "cache_ttl": self.cache_ttl,
            "rate_limit_requests": self.rate_limit_requests,
            "ai_batch_size": self.ai_batch_size,
            "ai_timeout": self.ai_timeout
        }

    def get_security_config(self) -> Dict[str, Any]:
        """Get security-related configuration"""
        return {
            "require_auth": self.require_auth,
            "jwt_algorithm": self.jwt_algorithm,
            "jwt_expire_minutes": self.jwt_expire_minutes,
            "allowed_origins": self.allowed_origins,
            "rate_limit_requests": self.rate_limit_requests
        }

    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags"""
        return {
            "enable_preview_generation": self.enable_preview_generation,
            "enable_real_time_updates": self.enable_real_time_updates,
            "enable_ai_analysis": self.enable_ai_analysis,
            "enable_cloud_processing": self.enable_cloud_processing,
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing
        }

    class Config:
        """Pydantic configuration"""
        env_prefix = "VIRALCLIP_"
        case_sensitive = False
        validate_assignment = True


@lru_cache()
def get_settings() -> NetflixLevelSettings:
    """Get cached settings instance"""
    return NetflixLevelSettings()


# Global settings instance
settings = get_settings()

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "structured": {
            "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
        }
    },
    "handlers": {
        "console": {
            "level": settings.log_level,
            "class": "logging.StreamHandler",
            "formatter": "standard" if not settings.structured_logging else "structured",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "structured",
            "filename": str(settings.logs_path / "app.log"),
            "maxBytes": 10_485_760,  # 10MB
            "backupCount": 5
        },
        "error_file": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "structured",
            "filename": str(settings.logs_path / "errors.log"),
            "maxBytes": 10_485_760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": settings.log_level,
            "propagate": False
        },
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False
        },
        "uvicorn.error": {
            "handlers": ["console", "error_file"],
            "level": "INFO",
            "propagate": False
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# Export commonly used settings
__all__ = [
    "settings",
    "get_settings",
    "NetflixLevelSettings",
    "LOGGING_CONFIG"
]