
"""
Netflix-Level Configuration Management v5.0
Environment-based settings with comprehensive validation
"""

import os
import secrets
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseSettings, Field, validator, HttpUrl


class NetflixLevelSettings(BaseSettings):
    """Netflix-level application settings with comprehensive validation"""
    
    # Application settings
    app_name: str = Field("ViralClip Pro v5.0", description="Application name")
    app_version: str = Field("5.0.0", description="Application version")
    debug: bool = Field(False, description="Debug mode")
    environment: str = Field("production", description="Environment (dev/staging/production)")
    
    # Server settings
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(5000, description="Server port")
    workers: int = Field(1, description="Number of worker processes")
    
    # Security settings
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="Secret key")
    require_auth: bool = Field(False, description="Require authentication")
    auth_token_expire_minutes: int = Field(1440, description="Auth token expiration in minutes")
    allowed_origins: List[str] = Field(["*"], description="CORS allowed origins")
    
    # File handling
    max_file_size: int = Field(500 * 1024 * 1024, description="Max file size (500MB)")
    allowed_video_formats: List[str] = Field(
        ["mp4", "avi", "mov", "mkv", "webm", "flv"],
        description="Allowed video formats"
    )
    max_video_duration: int = Field(600, description="Max video duration in seconds (10 minutes)")
    
    # Directory paths
    base_path: Path = Field(Path("nr1copilot/nr1-main"), description="Base application path")
    upload_path: Path = Field(Path("nr1copilot/nr1-main/uploads"), description="Upload directory")
    output_path: Path = Field(Path("nr1copilot/nr1-main/output"), description="Output directory")
    temp_path: Path = Field(Path("nr1copilot/nr1-main/temp"), description="Temporary files directory")
    cache_path: Path = Field(Path("nr1copilot/nr1-main/cache"), description="Cache directory")
    static_path: Path = Field(Path("nr1copilot/nr1-main/static"), description="Static files directory")
    public_path: Path = Field(Path("nr1copilot/nr1-main/public"), description="Public files directory")
    
    # Logging settings
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    enable_structured_logging: bool = Field(True, description="Enable structured JSON logging")
    log_file_path: Optional[Path] = Field(
        Path("nr1copilot/nr1-main/logs/app.log"),
        description="Log file path"
    )
    log_rotation: str = Field("1 day", description="Log rotation interval")
    log_retention: str = Field("30 days", description="Log retention period")
    
    # Performance settings
    enable_caching: bool = Field(True, description="Enable response caching")
    cache_ttl: int = Field(3600, description="Default cache TTL in seconds")
    cache_max_size: int = Field(1000, description="Maximum cache entries")
    
    # Rate limiting
    enable_rate_limiting: bool = Field(True, description="Enable rate limiting")
    rate_limit_requests: int = Field(100, description="Rate limit requests per window")
    rate_limit_window: int = Field(60, description="Rate limit window in seconds")
    rate_limit_storage: str = Field("memory", description="Rate limit storage backend")
    
    # Metrics and monitoring
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    metrics_endpoint: str = Field("/api/v5/metrics", description="Metrics endpoint path")
    health_endpoint: str = Field("/api/v5/health", description="Health check endpoint path")
    enable_performance_monitoring: bool = Field(True, description="Enable performance monitoring")
    
    # AI/ML settings
    ai_model_path: Optional[Path] = Field(None, description="AI model directory path")
    enable_gpu_acceleration: bool = Field(False, description="Enable GPU acceleration")
    max_concurrent_analysis: int = Field(5, description="Max concurrent AI analysis tasks")
    analysis_timeout: int = Field(300, description="Analysis timeout in seconds")
    
    # Video processing settings
    enable_hardware_acceleration: bool = Field(False, description="Enable hardware acceleration")
    ffmpeg_path: Optional[str] = Field(None, description="FFmpeg binary path")
    max_concurrent_processing: int = Field(3, description="Max concurrent video processing")
    processing_quality_default: str = Field("standard", description="Default processing quality")
    
    # WebSocket settings
    websocket_ping_interval: int = Field(30, description="WebSocket ping interval in seconds")
    websocket_ping_timeout: int = Field(10, description="WebSocket ping timeout in seconds")
    max_websocket_connections: int = Field(1000, description="Maximum WebSocket connections")
    
    # Database settings (if needed in future)
    database_url: Optional[str] = Field(None, description="Database connection URL")
    database_pool_size: int = Field(10, description="Database connection pool size")
    
    # Cloud storage settings (if needed)
    cloud_storage_enabled: bool = Field(False, description="Enable cloud storage")
    cloud_storage_bucket: Optional[str] = Field(None, description="Cloud storage bucket name")
    cloud_storage_region: Optional[str] = Field(None, description="Cloud storage region")
    
    # Redis settings (if needed)
    redis_url: Optional[str] = Field(None, description="Redis connection URL")
    redis_ttl: int = Field(3600, description="Redis default TTL")
    
    # Feature flags
    enable_preview_generation: bool = Field(True, description="Enable preview generation")
    enable_real_time_analysis: bool = Field(True, description="Enable real-time analysis")
    enable_viral_optimization: bool = Field(True, description="Enable viral optimization")
    enable_multi_platform_export: bool = Field(True, description="Enable multi-platform export")
    enable_advanced_analytics: bool = Field(True, description="Enable advanced analytics")
    
    # Development settings
    enable_debug_toolbar: bool = Field(False, description="Enable debug toolbar")
    enable_auto_reload: bool = Field(False, description="Enable auto-reload in development")
    mock_ai_responses: bool = Field(False, description="Use mock AI responses for testing")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production", "test"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator("max_file_size")
    def validate_max_file_size(cls, v):
        max_allowed = 1024 * 1024 * 1024  # 1GB
        if v > max_allowed:
            raise ValueError(f"Max file size cannot exceed {max_allowed} bytes")
        return v
    
    @validator("workers")
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError("Workers must be at least 1")
        if v > 16:
            raise ValueError("Workers should not exceed 16")
        return v
    
    @validator("upload_path", "output_path", "temp_path", "cache_path", pre=True)
    def validate_paths(cls, v):
        if isinstance(v, str):
            v = Path(v)
        return v
    
    def setup_directories(self):
        """Setup required directories"""
        directories = [
            self.upload_path,
            self.output_path,
            self.temp_path,
            self.cache_path,
            self.base_path / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == "development" or self.debug
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == "production"
    
    @property
    def database_config(self) -> Optional[dict]:
        """Get database configuration"""
        if not self.database_url:
            return None
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size
        }
    
    @property
    def redis_config(self) -> Optional[dict]:
        """Get Redis configuration"""
        if not self.redis_url:
            return None
        return {
            "url": self.redis_url,
            "ttl": self.redis_ttl
        }
    
    @property
    def cors_config(self) -> dict:
        """Get CORS configuration"""
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        }
    
    @property
    def logging_config(self) -> dict:
        """Get logging configuration"""
        config = {
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
            "handlers": {
                "console": {
                    "level": self.log_level,
                    "class": "logging.StreamHandler",
                    "formatter": "structured" if self.enable_structured_logging else "default",
                },
            },
            "loggers": {
                "": {
                    "level": self.log_level,
                    "handlers": ["console"],
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
        
        # Add file handler if log file path is specified
        if self.log_file_path:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            config["handlers"]["file"] = {
                "level": self.log_level,
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(self.log_file_path),
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
                "formatter": "structured" if self.enable_structured_logging else "default",
            }
            config["loggers"][""]["handlers"].append("file")
        
        return config
    
    def get_uvicorn_config(self) -> dict:
        """Get Uvicorn server configuration"""
        return {
            "host": self.host,
            "port": self.port,
            "reload": self.is_development and self.enable_auto_reload,
            "workers": 1 if self.is_development else self.workers,
            "log_level": self.log_level.lower(),
            "access_log": True,
            "use_colors": self.is_development,
        }


# Create global settings instance
settings = NetflixLevelSettings()

# Setup directories on import
settings.setup_directories()

# Environment-specific configurations
if settings.is_development:
    # Development overrides
    settings.enable_debug_toolbar = True
    settings.enable_auto_reload = True
    settings.mock_ai_responses = True
    settings.log_level = "DEBUG"

elif settings.is_production:
    # Production optimizations
    settings.enable_caching = True
    settings.enable_rate_limiting = True
    settings.enable_metrics = True
    settings.require_auth = True

# Export settings
__all__ = ["settings", "NetflixLevelSettings"]
