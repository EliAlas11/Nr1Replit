"""
Netflix-Grade Configuration Management v12.0
Optimized configuration with advanced validation and performance tuning
"""

import os
import tempfile
import logging
from enum import Enum
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, ConfigDict
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Application environment enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Optimized database configuration"""
    model_config = ConfigDict(env_prefix="DB_", env_file_encoding="utf-8")

    # Connection settings
    url: Optional[str] = Field(default=None, description="Database URL")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1024, le=65535, description="Database port")
    name: str = Field(default="viralclip", description="Database name")
    username: str = Field(default="postgres", description="Database username")
    password: SecretStr = Field(default=SecretStr(""), description="Database password")

    # Connection pooling
    pool_size: int = Field(default=20, ge=5, le=100, description="Connection pool size")
    max_overflow: int = Field(default=30, ge=0, le=200, description="Pool overflow")
    pool_timeout: int = Field(default=30, ge=5, le=300, description="Pool timeout")
    pool_recycle: int = Field(default=3600, ge=300, description="Connection recycle time")

    # Performance settings
    query_timeout: int = Field(default=30, ge=1, le=300, description="Query timeout")
    statement_timeout: int = Field(default=60, ge=1, le=600, description="Statement timeout")

    @property
    def connection_url(self) -> str:
        """Generate connection URL"""
        if self.url:
            return self.url

        password = self.password.get_secret_value() if self.password else ""
        auth = f"{self.username}:{password}@" if password else f"{self.username}@"

        return f"postgresql+asyncpg://{auth}{self.host}:{self.port}/{self.name}"


class SecuritySettings(BaseSettings):
    """Enhanced security configuration"""
    model_config = ConfigDict(env_prefix="SECURITY_", env_file_encoding="utf-8")

    # JWT Configuration
    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.urandom(32).hex()),
        description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, ge=1, le=168, description="JWT expiration")

    # Password security
    bcrypt_rounds: int = Field(default=12, ge=10, le=15, description="BCrypt rounds")
    password_min_length: int = Field(default=8, ge=6, le=128, description="Min password length")

    # Rate limiting
    rate_limit_requests: int = Field(default=1000, ge=10, description="Requests per minute")
    rate_limit_burst: int = Field(default=100, ge=5, description="Burst limit")

    # 2FA Configuration
    enable_2fa: bool = Field(default=True, description="Enable 2FA")
    totp_issuer: str = Field(default="ViralClip Pro", description="TOTP issuer")

    # Security headers
    enable_security_headers: bool = Field(default=True, description="Enable security headers")
    csrf_protection: bool = Field(default=True, description="Enable CSRF protection")


class PerformanceSettings(BaseSettings):
    """Netflix-grade performance configuration"""
    model_config = ConfigDict(env_prefix="PERF_", env_file_encoding="utf-8")

    # Application workers
    worker_processes: int = Field(default=1, ge=1, le=16, description="Worker processes")
    worker_connections: int = Field(default=1000, ge=100, le=10000, description="Worker connections")
    worker_timeout: int = Field(default=30, ge=10, le=300, description="Worker timeout")

    # Request handling
    max_upload_size: int = Field(default=500 * 1024 * 1024, description="Max upload size")
    request_timeout: int = Field(default=30, ge=5, le=300, description="Request timeout")
    max_concurrent_requests: int = Field(default=1000, ge=10, description="Max concurrent requests")

    # Caching
    enable_caching: bool = Field(default=True, description="Enable caching")
    cache_ttl_short: int = Field(default=300, ge=60, description="Short cache TTL")
    cache_ttl_medium: int = Field(default=1800, ge=300, description="Medium cache TTL")
    cache_ttl_long: int = Field(default=3600, ge=1800, description="Long cache TTL")

    # Compression
    enable_gzip: bool = Field(default=True, description="Enable GZIP")
    gzip_minimum_size: int = Field(default=1000, ge=100, description="GZIP min size")
    gzip_compression_level: int = Field(default=6, ge=1, le=9, description="GZIP level")

    # Memory management
    max_memory_usage_mb: int = Field(default=2048, ge=512, description="Max memory MB")
    gc_threshold: float = Field(default=0.8, ge=0.5, le=0.95, description="GC threshold")


class MonitoringSettings(BaseSettings):
    """Comprehensive monitoring configuration"""
    model_config = ConfigDict(env_prefix="MONITOR_", env_file_encoding="utf-8")

    # Health monitoring
    enable_health_checks: bool = Field(default=True, description="Enable health checks")
    health_check_interval: int = Field(default=30, ge=10, le=300, description="Health check interval")

    # Metrics collection
    enable_metrics: bool = Field(default=True, description="Enable metrics")
    metrics_interval: int = Field(default=60, ge=10, le=600, description="Metrics interval")

    # Logging
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")
    log_retention_days: int = Field(default=30, ge=1, le=90, description="Log retention")

    # Alerting thresholds
    cpu_alert_threshold: float = Field(default=80.0, ge=50.0, le=95.0, description="CPU alert %")
    memory_alert_threshold: float = Field(default=85.0, ge=50.0, le=95.0, description="Memory alert %")
    response_time_threshold: float = Field(default=2.0, ge=0.5, le=10.0, description="Response time alert")


class Settings(BaseSettings):
    """Main application configuration"""
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application identity
    app_name: str = Field(default="ViralClip Pro v12.0", description="Application name")
    app_version: str = Field(default="12.0.0", description="Application version")
    app_description: str = Field(
        default="Netflix-Grade Video Editing & AI-Powered Social Platform",
        description="Application description"
    )

    # Environment
    environment: Environment = Field(default=Environment.PRODUCTION, description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    testing: bool = Field(default=False, description="Testing mode")

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5000, ge=1024, le=65535, description="Server port")

    # CORS configuration
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "https://*.replit.app",
            "https://*.replit.dev",
            "https://*.replit.com"
        ],
        description="CORS origins"
    )

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    log_format: str = Field(default="json", description="Log format")

    # Feature flags
    enable_ai_analysis: bool = Field(default=True, description="Enable AI features")
    enable_video_processing: bool = Field(default=True, description="Enable video processing")
    enable_social_publishing: bool = Field(default=True, description="Enable social publishing")
    enable_real_time_processing: bool = Field(default=True, description="Enable real-time processing")
    enable_enterprise_features: bool = Field(default=True, description="Enable enterprise features")
    enable_websockets: bool = Field(default=True, description="Enable WebSocket support")

    # Directory configuration
    base_dir: Path = Field(default_factory=lambda: Path.cwd(), description="Base directory")
    upload_path: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "viralclip" / "uploads"
    )
    temp_path: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "viralclip" / "temp"
    )
    output_path: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "viralclip" / "output"
    )
    cache_path: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "viralclip" / "cache"
    )
    logs_path: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "viralclip" / "logs"
    )

    # Nested configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @field_validator('environment', mode='before')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting"""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                logger.warning(f"Invalid environment '{v}', defaulting to production")
                return Environment.PRODUCTION
        return v

    @field_validator('debug')
    @classmethod
    def validate_debug_mode(cls, v, info):
        """Ensure debug is disabled in production"""
        if hasattr(info, 'data') and info.data:
            environment = info.data.get('environment')
            if environment == Environment.PRODUCTION and v:
                logger.warning("Debug mode disabled in production")
                return False
        return v

    def ensure_directories(self) -> None:
        """Ensure all directories exist"""
        directories = [
            self.upload_path,
            self.temp_path,
            self.output_path,
            self.cache_path,
            self.logs_path
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                directory.chmod(0o755)
                logger.debug(f"Ensured directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                fallback = Path(tempfile.gettempdir()) / directory.name
                fallback.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Using fallback directory: {fallback}")

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_staging(self) -> bool:
        return self.environment == Environment.STAGING

    @property
    def is_testing(self) -> bool:
        return self.environment == Environment.TESTING

    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database.connection_url

    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags"""
        return {
            "ai_analysis": self.enable_ai_analysis,
            "video_processing": self.enable_video_processing,
            "social_publishing": self.enable_social_publishing,
            "real_time_processing": self.enable_real_time_processing,
            "enterprise_features": self.enable_enterprise_features,
            "websockets": self.enable_websockets
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            "workers": self.performance.worker_processes,
            "connections": self.performance.worker_connections,
            "timeout": self.performance.worker_timeout,
            "max_upload_size": self.performance.max_upload_size,
            "caching_enabled": self.performance.enable_caching,
            "gzip_enabled": self.performance.enable_gzip,
            "max_memory_mb": self.performance.max_memory_usage_mb
        }

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            "jwt_algorithm": self.security.jwt_algorithm,
            "jwt_expiration_hours": self.security.jwt_expiration_hours,
            "2fa_enabled": self.security.enable_2fa,
            "rate_limiting": self.security.rate_limit_requests,
            "security_headers": self.security.enable_security_headers,
            "csrf_protection": self.security.csrf_protection
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    try:
        settings = Settings()
        settings.ensure_directories()

        logger.info(f"Configuration loaded for {settings.environment.value}")
        return settings

    except Exception as e:
        logger.error(f"Configuration loading failed: {e}")
        minimal_settings = Settings()
        minimal_settings.ensure_directories()
        return minimal_settings


def reload_settings() -> Settings:
    """Force reload settings"""
    get_settings.cache_clear()
    return get_settings()


# Export settings instance
settings = get_settings()

# Validate on import
try:
    test_settings = get_settings()
    logger.info(f"✅ Configuration validated for {test_settings.environment.value}")
    logger.info(f"🚀 Application: {test_settings.app_name} v{test_settings.app_version}")
except Exception as e:
    logger.error(f"❌ Configuration validation failed: {e}")