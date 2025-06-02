
"""
Netflix-Level Configuration Management
Production-ready settings with environment-based configuration
"""

import os
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings
import logging


class DatabaseSettings(PydanticBaseSettings):
    """Database configuration settings"""
    url: str = Field(default="sqlite:///./viralclip.db", env="DATABASE_URL")
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    echo: bool = Field(default=False, env="DB_ECHO")


class RedisSettings(PydanticBaseSettings):
    """Redis configuration for caching and sessions"""
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(default=30, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(default=30, env="REDIS_CONNECT_TIMEOUT")
    retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    decode_responses: bool = Field(default=True, env="REDIS_DECODE_RESPONSES")


class SecuritySettings(PydanticBaseSettings):
    """Security configuration"""
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration: int = Field(default=3600, env="JWT_EXPIRATION")  # 1 hour
    bcrypt_rounds: int = Field(default=12, env="BCRYPT_ROUNDS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    trusted_hosts: List[str] = Field(default=["*"], env="TRUSTED_HOSTS")
    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")


class UploadSettings(PydanticBaseSettings):
    """Upload configuration"""
    max_file_size: int = Field(default=2 * 1024 * 1024 * 1024, env="MAX_FILE_SIZE")  # 2GB
    chunk_size: int = Field(default=8 * 1024 * 1024, env="CHUNK_SIZE")  # 8MB
    max_concurrent_uploads: int = Field(default=5, env="MAX_CONCURRENT_UPLOADS")
    max_concurrent_chunks: int = Field(default=10, env="MAX_CONCURRENT_CHUNKS")
    upload_timeout: int = Field(default=3600, env="UPLOAD_TIMEOUT")  # 1 hour
    supported_formats: List[str] = Field(
        default=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp", ".mp3", ".wav", ".m4a", ".flac", ".aac"],
        env="SUPPORTED_FORMATS"
    )


class ProcessingSettings(PydanticBaseSettings):
    """Video processing configuration"""
    ffmpeg_threads: int = Field(default=4, env="FFMPEG_THREADS")
    processing_timeout: int = Field(default=1800, env="PROCESSING_TIMEOUT")  # 30 minutes
    output_formats: List[str] = Field(default=["mp4", "webm"], env="OUTPUT_FORMATS")
    quality_presets: List[str] = Field(default=["draft", "standard", "high", "premium"], env="QUALITY_PRESETS")
    thumbnail_count: int = Field(default=5, env="THUMBNAIL_COUNT")


class AISettings(PydanticBaseSettings):
    """AI processing configuration"""
    model_name: str = Field(default="gpt-4", env="AI_MODEL_NAME")
    max_tokens: int = Field(default=4000, env="AI_MAX_TOKENS")
    temperature: float = Field(default=0.7, env="AI_TEMPERATURE")
    timeout: int = Field(default=120, env="AI_TIMEOUT")
    max_retries: int = Field(default=3, env="AI_MAX_RETRIES")
    api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")


class MonitoringSettings(PydanticBaseSettings):
    """Monitoring and observability configuration"""
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    structured_logging: bool = Field(default=True, env="STRUCTURED_LOGGING")
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    jaeger_enabled: bool = Field(default=False, env="JAEGER_ENABLED")


class PerformanceSettings(PydanticBaseSettings):
    """Performance optimization settings"""
    worker_count: int = Field(default=4, env="WORKER_COUNT")
    max_requests_per_child: int = Field(default=10000, env="MAX_REQUESTS_PER_CHILD")
    keepalive_timeout: int = Field(default=2, env="KEEPALIVE_TIMEOUT")
    max_concurrent_requests: int = Field(default=1000, env="MAX_CONCURRENT_REQUESTS")
    memory_limit_mb: int = Field(default=2048, env="MEMORY_LIMIT_MB")
    cpu_limit_percent: int = Field(default=80, env="CPU_LIMIT_PERCENT")


class Settings(PydanticBaseSettings):
    """Main application settings"""
    
    # Application metadata
    app_name: str = Field(default="ViralClip Pro", env="APP_NAME")
    app_version: str = Field(default="7.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=5000, env="PORT")
    
    # Directory paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    upload_path: Path = Field(default_factory=lambda: Path("uploads"))
    output_path: Path = Field(default_factory=lambda: Path("output"))
    temp_path: Path = Field(default_factory=lambda: Path("temp"))
    cache_path: Path = Field(default_factory=lambda: Path("cache"))
    logs_path: Path = Field(default_factory=lambda: Path("logs"))
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    upload: UploadSettings = Field(default_factory=UploadSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    ai: AISettings = Field(default_factory=AISettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    
    # Feature flags
    enable_websockets: bool = Field(default=True, env="ENABLE_WEBSOCKETS")
    enable_ai_analysis: bool = Field(default=True, env="ENABLE_AI_ANALYSIS")
    enable_social_publishing: bool = Field(default=True, env="ENABLE_SOCIAL_PUBLISHING")
    enable_batch_processing: bool = Field(default=True, env="ENABLE_BATCH_PROCESSING")
    enable_real_time_updates: bool = Field(default=True, env="ENABLE_REAL_TIME_UPDATES")
    
    # External service URLs
    webhook_url: Optional[str] = Field(default=None, env="WEBHOOK_URL")
    cdn_url: Optional[str] = Field(default=None, env="CDN_URL")
    analytics_url: Optional[str] = Field(default=None, env="ANALYTICS_URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"
    
    @validator("upload_path", "output_path", "temp_path", "cache_path", "logs_path", pre=True)
    def ensure_absolute_paths(cls, v, values):
        """Ensure all paths are absolute"""
        if isinstance(v, str):
            v = Path(v)
        
        if not v.is_absolute():
            base_dir = values.get("base_dir", Path(__file__).parent.parent)
            v = base_dir / v
        
        return v
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting"""
        valid_environments = ["development", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    def ensure_directories(self):
        """Create necessary directories"""
        directories = [
            self.upload_path,
            self.output_path,
            self.temp_path,
            self.cache_path,
            self.logs_path,
            self.upload_path / "chunks",
            self.output_path / "thumbnails",
            self.output_path / "previews",
            self.temp_path / "processing",
            self.cache_path / "redis_backup"
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o755)
            except Exception as e:
                logging.warning(f"Failed to create directory {directory}: {e}")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    @property
    def is_staging(self) -> bool:
        """Check if running in staging"""
        return self.environment == "staging"
    
    def get_database_url(self) -> str:
        """Get formatted database URL"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get formatted Redis URL"""
        return self.redis.url
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as list"""
        if isinstance(self.security.cors_origins, str):
            return [origin.strip() for origin in self.security.cors_origins.split(",")]
        return self.security.cors_origins
    
    def get_trusted_hosts(self) -> List[str]:
        """Get trusted hosts as list"""
        if isinstance(self.security.trusted_hosts, str):
            return [host.strip() for host in self.security.trusted_hosts.split(",")]
        return self.security.trusted_hosts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "app": {
                "name": self.app_name,
                "version": self.app_version,
                "environment": self.environment,
                "debug": self.debug
            },
            "server": {
                "host": self.host,
                "port": self.port
            },
            "features": {
                "websockets": self.enable_websockets,
                "ai_analysis": self.enable_ai_analysis,
                "social_publishing": self.enable_social_publishing,
                "batch_processing": self.enable_batch_processing,
                "real_time_updates": self.enable_real_time_updates
            },
            "performance": {
                "worker_count": self.performance.worker_count,
                "max_concurrent_requests": self.performance.max_concurrent_requests,
                "memory_limit_mb": self.performance.memory_limit_mb
            }
        }


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()


# Configuration utilities
class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, settings_instance: Settings):
        self.settings = settings_instance
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate all configuration settings"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required directories
        for path_name in ["upload_path", "output_path", "temp_path", "cache_path", "logs_path"]:
            path = getattr(self.settings, path_name)
            if not path.exists():
                validation_results["warnings"].append(f"Directory does not exist: {path}")
        
        # Check external dependencies
        if self.settings.enable_ai_analysis and not self.settings.ai.api_key:
            validation_results["warnings"].append("AI analysis enabled but no API key provided")
        
        # Validate memory limits
        if self.settings.performance.memory_limit_mb < 512:
            validation_results["warnings"].append("Memory limit is very low, may cause performance issues")
        
        # Check file size limits
        if self.settings.upload.max_file_size > 5 * 1024 * 1024 * 1024:  # 5GB
            validation_results["warnings"].append("Very large file size limit may impact performance")
        
        return validation_results
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags"""
        return {
            "websockets": self.settings.enable_websockets,
            "ai_analysis": self.settings.enable_ai_analysis,
            "social_publishing": self.settings.enable_social_publishing,
            "batch_processing": self.settings.enable_batch_processing,
            "real_time_updates": self.settings.enable_real_time_updates
        }
    
    def update_runtime_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration at runtime"""
        try:
            for key, value in updates.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
            return True
        except Exception as e:
            logging.error(f"Failed to update runtime config: {e}")
            return False


# Global config manager
config_manager = ConfigManager(settings)
