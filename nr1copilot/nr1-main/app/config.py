"""
Enterprise Configuration Management
Netflix-level configuration with environment-specific settings and validation
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, validator, Field
from enum import Enum
import logging


class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SecuritySettings(BaseSettings):
    """Security configuration with validation"""
    secret_key: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"),
        description="Secret key for JWT and session encryption"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration: int = Field(default=3600, ge=300, le=86400, description="JWT expiration in seconds")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    rate_limit_requests: int = Field(default=1000, ge=10, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=3600, ge=60, description="Rate limit window in seconds")
    max_upload_size: int = Field(default=500 * 1024 * 1024, description="Max upload size in bytes")

    @validator("secret_key")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v

    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Database configuration with connection pooling"""
    url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./viralclip.db"),
        description="Database connection URL"
    )
    pool_size: int = Field(default=20, ge=5, le=100, description="Connection pool size")
    max_overflow: int = Field(default=30, ge=10, le=200, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=5, le=300, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, ge=300, description="Pool recycle time in seconds")
    echo: bool = Field(default=False, description="Echo SQL queries")

    class Config:
        env_prefix = "DB_"


class CacheSettings(BaseSettings):
    """Cache configuration with Redis support"""
    redis_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("REDIS_URL"),
        description="Redis connection URL"
    )
    default_ttl: int = Field(default=3600, ge=60, description="Default TTL in seconds")
    max_memory_mb: int = Field(default=512, ge=64, description="Max memory usage in MB")
    compression_enabled: bool = Field(default=True, description="Enable cache compression")
    cluster_mode: bool = Field(default=False, description="Enable Redis cluster mode")

    class Config:
        env_prefix = "CACHE_"


class AISettings(BaseSettings):
    """AI/ML configuration with model management"""
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key"
    )
    huggingface_token: Optional[str] = Field(
        default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN"),
        description="HuggingFace API token"
    )
    model_cache_dir: str = Field(default="./models", description="Model cache directory")
    max_video_size_mb: int = Field(default=500, ge=10, le=2000, description="Max video size in MB")
    processing_timeout: int = Field(default=300, ge=30, le=1800, description="Processing timeout in seconds")
    batch_size: int = Field(default=32, ge=1, le=128, description="AI model batch size")
    gpu_enabled: bool = Field(default=False, description="Enable GPU acceleration")

    class Config:
        env_prefix = "AI_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: int = Field(default=60, ge=10, description="Metrics collection interval")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    structured_logging: bool = Field(default=True, description="Enable structured logging")
    performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    health_check_interval: int = Field(default=30, ge=5, description="Health check interval")

    class Config:
        env_prefix = "MONITORING_"


class PerformanceSettings(BaseSettings):
    """Performance optimization settings"""
    enable_gzip: bool = Field(default=True, description="Enable gzip compression")
    gzip_min_size: int = Field(default=1000, ge=100, description="Minimum size for gzip")
    max_request_size: int = Field(default=100 * 1024 * 1024, description="Max request size")
    connection_timeout: int = Field(default=60, ge=5, description="Connection timeout")
    read_timeout: int = Field(default=300, ge=30, description="Read timeout")
    worker_processes: int = Field(default=4, ge=1, le=32, description="Worker processes")
    async_pool_size: int = Field(default=100, ge=10, le=1000, description="Async pool size")
    cache_max_entries: int = Field(default=10000, ge=1000, description="Max cache entries")

    class Config:
        env_prefix = "PERFORMANCE_"


class Settings(BaseSettings):
    """Main application settings with comprehensive validation"""

    # Core settings
    app_name: str = Field(default="ViralClip Pro", description="Application name")
    app_version: str = Field(default="10.0.0", description="Application version")
    # Environment-specific optimizations for deployment
    environment: Environment = Field(
        default_factory=lambda: Environment(os.getenv("ENV", "production").lower()),
        description="Environment"
    )

    # Render.com specific settings
    port: int = Field(
        default_factory=lambda: int(os.getenv("PORT", "5000")),
        ge=1000, le=65535, description="Port number"
    )
    debug: bool = Field(default=False, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Host address")

    # Paths with validation
    upload_path: str = Field(default="./uploads", description="Upload directory")
    temp_path: str = Field(default="./temp", description="Temporary files directory")
    static_path: str = Field(default="./static", description="Static files directory")
    log_path: str = Field(default="./logs", description="Log files directory")

    # Feature flags for performance optimization
    enable_analytics: bool = Field(default=True, description="Enable analytics")
    enable_collaboration: bool = Field(default=True, description="Enable collaboration")
    enable_realtime: bool = Field(default=True, description="Enable real-time features")
    enable_ai_processing: bool = Field(default=True, description="Enable AI processing")
    enable_social_publishing: bool = Field(default=True, description="Enable social publishing")
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_compression: bool = Field(default=True, description="Enable compression")

    # Sub-configurations
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    ai: AISettings = Field(default_factory=AISettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)

    @validator("debug", pre=True)
    def debug_from_env(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v

    @validator("environment", pre=True)
    def environment_from_env(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @validator("upload_path", "temp_path", "static_path", "log_path")
    def create_directories(cls, v):
        """Ensure directories exist"""
        os.makedirs(v, exist_ok=True)
        return v

    def get_database_url(self) -> str:
        """Get database URL with fallback"""
        return self.database.url

    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL with fallback"""
        return self.cache.redis_url

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins based on environment"""
        if self.is_production():
            return ["https://*.replit.app", "https://*.replit.dev"]
        return ["*"]

    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            "level": self.monitoring.log_level.value,
            "structured": self.monitoring.structured_logging,
            "path": self.log_path
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            "async_pool_size": self.performance.async_pool_size,
            "cache_max_entries": self.performance.cache_max_entries,
            "worker_processes": self.performance.worker_processes,
            "enable_compression": self.enable_compression,
            "enable_caching": self.enable_caching
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True


# Global settings instance with environment-specific optimizations
settings = Settings()

# Environment-specific optimizations
if settings.is_production():
    settings.debug = False
    settings.monitoring.log_level = LogLevel.WARNING
    settings.security.cors_origins = settings.get_cors_origins()
    settings.cache.default_ttl = 7200
    settings.performance.worker_processes = 4
    settings.performance.async_pool_size = 200
elif settings.is_development():
    settings.debug = True
    settings.monitoring.log_level = LogLevel.DEBUG
    settings.security.cors_origins = ["*"]
    settings.performance.worker_processes = 1
    settings.performance.async_pool_size = 50

# Logging configuration
logging.basicConfig(
    level=getattr(logging, settings.monitoring.log_level.value),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"Configuration loaded for environment: {settings.environment.value}")