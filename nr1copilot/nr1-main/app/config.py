
"""
Netflix-Level Configuration Management v6.0
Comprehensive settings with validation and environment-specific configs
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from functools import lru_cache

from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.types import PositiveInt, PositiveFloat

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseSettings):
    """Database configuration with connection pooling"""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: PositiveInt = Field(default=5432, env="DB_PORT")
    name: str = Field(default="viralclip", env="DB_NAME")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    
    # Connection pool settings
    min_connections: PositiveInt = Field(default=5, env="DB_MIN_CONNECTIONS")
    max_connections: PositiveInt = Field(default=20, env="DB_MAX_CONNECTIONS")
    connection_timeout: PositiveFloat = Field(default=30.0, env="DB_CONNECTION_TIMEOUT")
    command_timeout: PositiveFloat = Field(default=60.0, env="DB_COMMAND_TIMEOUT")
    
    # SSL settings
    ssl_mode: str = Field(default="prefer", env="DB_SSL_MODE")
    ssl_cert_path: Optional[str] = Field(default=None, env="DB_SSL_CERT_PATH")
    
    @property
    def connection_string(self) -> str:
        """Get complete database connection string"""
        password_part = f":{self.password}" if self.password else ""
        ssl_part = f"?sslmode={self.ssl_mode}" if self.ssl_mode != "disable" else ""
        return f"postgresql://{self.user}{password_part}@{self.host}:{self.port}/{self.name}{ssl_part}"


class CacheConfig(BaseSettings):
    """Cache configuration for Redis/Memory"""
    
    type: str = Field(default="memory", env="CACHE_TYPE")  # memory, redis
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: PositiveInt = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Cache settings
    default_ttl: PositiveInt = Field(default=3600, env="CACHE_DEFAULT_TTL")
    max_memory_mb: PositiveInt = Field(default=512, env="CACHE_MAX_MEMORY_MB")
    
    # Connection pool
    max_connections: PositiveInt = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    connection_timeout: PositiveFloat = Field(default=5.0, env="REDIS_CONNECTION_TIMEOUT")
    
    @validator("type")
    def validate_cache_type(cls, v):
        if v not in ["memory", "redis"]:
            raise ValueError("Cache type must be 'memory' or 'redis'")
        return v
    
    @property
    def redis_connection_string(self) -> Optional[str]:
        """Get Redis connection string"""
        if self.type != "redis":
            return None
            
        if self.redis_url:
            return self.redis_url
            
        password_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{password_part}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class SecurityConfig(BaseSettings):
    """Security configuration"""
    
    # JWT settings
    jwt_secret_key: str = Field(default="your-super-secret-key", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: PositiveInt = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # Rate limiting
    rate_limit_requests_per_minute: PositiveInt = Field(default=100, env="RATE_LIMIT_RPM")
    rate_limit_burst: PositiveInt = Field(default=200, env="RATE_LIMIT_BURST")
    
    # Security features
    require_auth: bool = Field(default=False, env="REQUIRE_AUTH")
    enable_csrf_protection: bool = Field(default=True, env="ENABLE_CSRF")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    # Encryption
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    hash_rounds: PositiveInt = Field(default=12, env="HASH_ROUNDS")
    
    @validator("allowed_origins", pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class AIConfig(BaseSettings):
    """AI and ML configuration"""
    
    # Model settings
    model_path: str = Field(default="models", env="AI_MODEL_PATH")
    enable_gpu: bool = Field(default=False, env="AI_ENABLE_GPU")
    batch_size: PositiveInt = Field(default=4, env="AI_BATCH_SIZE")
    max_concurrent_analyses: PositiveInt = Field(default=5, env="AI_MAX_CONCURRENT")
    
    # API keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # Model-specific settings
    sentiment_model: str = Field(default="cardiffnlp/twitter-roberta-base-sentiment-latest", env="SENTIMENT_MODEL")
    object_detection_model: str = Field(default="yolov5s", env="OBJECT_DETECTION_MODEL")
    
    # Performance settings
    model_cache_size: PositiveInt = Field(default=3, env="AI_MODEL_CACHE_SIZE")
    inference_timeout: PositiveFloat = Field(default=30.0, env="AI_INFERENCE_TIMEOUT")


class VideoConfig(BaseSettings):
    """Video processing configuration"""
    
    # File size limits
    max_file_size_mb: PositiveInt = Field(default=500, env="MAX_FILE_SIZE_MB")
    max_duration_seconds: PositiveInt = Field(default=600, env="MAX_DURATION_SECONDS")
    
    # Supported formats
    supported_formats: List[str] = Field(
        default=["mp4", "avi", "mov", "mkv", "webm"],
        env="SUPPORTED_FORMATS"
    )
    
    # Processing settings
    default_quality: str = Field(default="high", env="DEFAULT_QUALITY")
    max_concurrent_processing: PositiveInt = Field(default=3, env="MAX_CONCURRENT_PROCESSING")
    
    # Output settings
    output_format: str = Field(default="mp4", env="OUTPUT_FORMAT")
    output_quality: str = Field(default="1080p", env="OUTPUT_QUALITY")
    audio_bitrate: str = Field(default="128k", env="AUDIO_BITRATE")
    video_bitrate: str = Field(default="2000k", env="VIDEO_BITRATE")
    
    @validator("supported_formats", pre=True)
    def parse_supported_formats(cls, v):
        if isinstance(v, str):
            return [fmt.strip().lower() for fmt in v.split(",")]
        return [fmt.lower() for fmt in v]
    
    @validator("default_quality")
    def validate_quality(cls, v):
        valid_qualities = ["low", "medium", "high", "ultra"]
        if v not in valid_qualities:
            raise ValueError(f"Quality must be one of {valid_qualities}")
        return v


class Settings(BaseSettings):
    """Main application settings with Netflix-level configuration"""
    
    # Application settings
    app_name: str = Field(default="ViralClip Pro v6.0", env="APP_NAME")
    app_version: str = Field(default="6.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: PositiveInt = Field(default=5000, env="PORT")
    workers: PositiveInt = Field(default=1, env="WORKERS")
    
    # Directory settings
    base_path: Path = Field(default=Path("nr1copilot/nr1-main"), env="BASE_PATH")
    upload_path: Path = Field(default=Path("nr1copilot/nr1-main/uploads"), env="UPLOAD_PATH")
    output_path: Path = Field(default=Path("nr1copilot/nr1-main/output"), env="OUTPUT_PATH")
    temp_path: Path = Field(default=Path("nr1copilot/nr1-main/temp"), env="TEMP_PATH")
    cache_path: Path = Field(default=Path("nr1copilot/nr1-main/cache"), env="CACHE_PATH")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    enable_structured_logging: bool = Field(default=True, env="ENABLE_STRUCTURED_LOGGING")
    log_file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    log_rotation_size: str = Field(default="10MB", env="LOG_ROTATION_SIZE")
    log_retention_days: PositiveInt = Field(default=30, env="LOG_RETENTION_DAYS")
    
    # Performance settings
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_retention_hours: PositiveInt = Field(default=24, env="METRICS_RETENTION_HOURS")
    enable_health_checks: bool = Field(default=True, env="ENABLE_HEALTH_CHECKS")
    health_check_interval: PositiveInt = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Feature flags
    enable_ai_analysis: bool = Field(default=True, env="ENABLE_AI_ANALYSIS")
    enable_realtime_processing: bool = Field(default=True, env="ENABLE_REALTIME_PROCESSING")
    enable_cloud_processing: bool = Field(default=False, env="ENABLE_CLOUD_PROCESSING")
    enable_websocket: bool = Field(default=True, env="ENABLE_WEBSOCKET")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    cache: CacheConfig = CacheConfig()
    security: SecurityConfig = SecurityConfig()
    ai: AIConfig = AIConfig()
    video: VideoConfig = VideoConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @root_validator
    def validate_paths(cls, values):
        """Ensure all paths are absolute and exist"""
        path_fields = ["base_path", "upload_path", "output_path", "temp_path", "cache_path"]
        
        for field in path_fields:
            if field in values and values[field]:
                path = Path(values[field])
                if not path.is_absolute():
                    # Make relative paths absolute from current working directory
                    values[field] = Path.cwd() / path
                else:
                    values[field] = path
        
        return values
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    @property
    def max_file_size(self) -> int:
        """Get max file size in bytes"""
        return self.video.max_file_size_mb * 1024 * 1024
    
    @property
    def cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": self.security.allowed_origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["*"],
            "max_age": 600
        }
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get comprehensive logging configuration"""
        handlers = {
            "console": {
                "level": self.log_level,
                "class": "logging.StreamHandler",
                "formatter": "structured" if self.enable_structured_logging else "default",
            }
        }
        
        # Add file handler if path specified
        if self.log_file_path:
            handlers["file"] = {
                "level": self.log_level,
                "class": "logging.handlers.RotatingFileHandler",
                "filename": self.log_file_path,
                "maxBytes": self._parse_size(self.log_rotation_size),
                "backupCount": self.log_retention_days,
                "formatter": "structured" if self.enable_structured_logging else "default",
            }
        
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": self.log_format,
                },
                "structured": {
                    "format": "%(message)s",
                },
            },
            "handlers": handlers,
            "loggers": {
                "": {
                    "level": self.log_level,
                    "handlers": list(handlers.keys()),
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "fastapi": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes"""
        size_str = size_str.upper()
        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging"""
        return {
            "app_name": self.app_name,
            "version": self.app_version,
            "environment": self.environment,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "features": {
                "ai_analysis": self.enable_ai_analysis,
                "realtime_processing": self.enable_realtime_processing,
                "cloud_processing": self.enable_cloud_processing,
                "websocket": self.enable_websocket,
                "metrics": self.enable_metrics,
                "health_checks": self.enable_health_checks
            },
            "cache_type": self.cache.type,
            "database_host": self.database.host,
            "max_file_size_mb": self.video.max_file_size_mb,
            "log_level": self.log_level
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()

# Validate critical settings on import
if settings.is_production and settings.debug:
    logger.warning("⚠️ Debug mode is enabled in production!")

if settings.security.jwt_secret_key == "your-super-secret-key":
    logger.warning("⚠️ Using default JWT secret key - please change in production!")
