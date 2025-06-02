
"""
Enterprise Configuration Management v10.0
Modern, secure, and scalable configuration system with validation and environment management
"""

import os
import tempfile
import logging
from enum import Enum
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic_settings import BaseSettings as PydanticBaseSettings

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Application environment enumeration with clear definitions"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SecuritySettings(PydanticBaseSettings):
    """Enterprise security configuration with best practices"""
    
    # Authentication & Encryption
    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.urandom(32).hex()),
        description="Secret key for JWT tokens and encryption"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiration_hours: int = Field(default=24, ge=1, le=720, description="JWT token expiration in hours")
    bcrypt_rounds: int = Field(default=12, ge=10, le=15, description="BCrypt hashing rounds")
    
    # Security Headers
    enable_security_headers: bool = Field(default=True, description="Enable security headers")
    enable_cors: bool = Field(default=True, description="Enable CORS middleware")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests per minute per IP")
    
    class Config:
        env_prefix = "SECURITY_"
        env_file_encoding = "utf-8"


class DatabaseSettings(PydanticBaseSettings):
    """Enterprise database configuration with connection pooling"""
    
    # Connection settings
    url: Optional[str] = Field(default=None, description="Database connection URL")
    pool_size: int = Field(default=10, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Pool overflow limit")
    pool_timeout: int = Field(default=30, ge=1, le=300, description="Pool checkout timeout")
    pool_recycle: int = Field(default=3600, ge=300, description="Pool connection recycle time")
    
    # Query settings
    query_timeout: int = Field(default=30, ge=1, le=300, description="Query timeout in seconds")
    enable_query_logging: bool = Field(default=False, description="Enable SQL query logging")
    
    # Health check settings
    health_check_interval: int = Field(default=30, ge=10, le=300, description="Health check interval")
    
    @validator('url')
    def validate_database_url(cls, v):
        """Validate and enhance database URL for Replit PostgreSQL"""
        if v and 'postgresql://' in v and '-pooler' not in v:
            # Auto-configure for Replit PostgreSQL pooler
            v = v.replace('@', '-pooler@')
            logger.info("Enhanced database URL for connection pooling")
        return v
    
    class Config:
        env_prefix = "DB_"
        env_file_encoding = "utf-8"


class PerformanceSettings(PydanticBaseSettings):
    """Enterprise performance and resource optimization"""
    
    # Worker configuration
    worker_processes: int = Field(default=1, ge=1, le=8, description="Number of worker processes")
    worker_connections: int = Field(default=1000, ge=100, le=10000, description="Connections per worker")
    
    # Request handling
    max_upload_size: int = Field(default=100 * 1024 * 1024, description="Maximum upload size (100MB)")
    request_timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    keepalive_timeout: int = Field(default=65, ge=1, le=300, description="Keep-alive timeout")
    
    # Caching
    enable_caching: bool = Field(default=True, description="Enable application caching")
    cache_ttl: int = Field(default=3600, ge=60, description="Default cache TTL in seconds")
    cache_max_size: int = Field(default=1000, ge=100, description="Maximum cache entries")
    
    # Optimization features
    enable_gzip: bool = Field(default=True, description="Enable GZIP compression")
    gzip_minimum_size: int = Field(default=1000, ge=100, description="Minimum size for GZIP")
    enable_response_caching: bool = Field(default=True, description="Enable HTTP response caching")
    
    class Config:
        env_prefix = "PERF_"
        env_file_encoding = "utf-8"


class MonitoringSettings(PydanticBaseSettings):
    """Enterprise monitoring and observability configuration"""
    
    # Health monitoring
    enable_health_checks: bool = Field(default=True, description="Enable health check endpoints")
    health_check_interval: int = Field(default=30, ge=10, le=300, description="Health check interval")
    
    # Metrics collection
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: int = Field(default=60, ge=10, le=600, description="Metrics collection interval")
    
    # Logging
    enable_structured_logging: bool = Field(default=True, description="Enable structured JSON logging")
    enable_request_logging: bool = Field(default=True, description="Enable HTTP request logging")
    log_retention_days: int = Field(default=30, ge=1, le=90, description="Log retention period")
    
    # Alerting
    enable_alerting: bool = Field(default=True, description="Enable system alerting")
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0
        },
        description="Alert threshold percentages"
    )
    
    class Config:
        env_prefix = "MONITOR_"
        env_file_encoding = "utf-8"


class Settings(PydanticBaseSettings):
    """Main enterprise application configuration"""
    
    # Application identity
    app_name: str = Field(default="ViralClip Pro v10.0", description="Application name")
    app_version: str = Field(default="10.0.0", description="Application version")
    app_description: str = Field(
        default="Netflix-Grade Video Editing & Social Automation Platform",
        description="Application description"
    )
    
    # Environment configuration
    environment: Environment = Field(default=Environment.PRODUCTION, description="Application environment")
    debug: bool = Field(default=False, description="Debug mode flag")
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=5000, ge=1024, le=65535, description="Server port")
    
    # CORS configuration
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "https://*.replit.app",
            "https://*.replit.dev",
            "https://*.replit.com",
            "http://localhost:3000",
            "http://localhost:5000"
        ],
        description="Allowed CORS origins"
    )
    
    # Logging configuration
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    
    # Feature flags
    enable_ai_analysis: bool = Field(default=True, description="Enable AI analysis features")
    enable_real_time_processing: bool = Field(default=True, description="Enable real-time processing")
    enable_enterprise_features: bool = Field(default=True, description="Enable enterprise features")
    enable_video_processing: bool = Field(default=True, description="Enable video processing")
    enable_social_publishing: bool = Field(default=True, description="Enable social media publishing")
    
    # Directory configuration with automatic creation
    base_dir: Path = Field(default_factory=lambda: Path.cwd(), description="Base application directory")
    upload_path: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()) / "viralclip" / "uploads")
    temp_path: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()) / "viralclip" / "temp")
    output_path: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()) / "viralclip" / "output")
    cache_path: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()) / "viralclip" / "cache")
    logs_path: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()) / "viralclip" / "logs")
    
    # Nested configuration objects
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        """Validate and normalize environment setting"""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                logger.warning(f"Invalid environment '{v}', defaulting to production")
                return Environment.PRODUCTION
        return v
    
    @validator('debug')
    def validate_debug_mode(cls, v, values):
        """Ensure debug is disabled in production"""
        environment = values.get('environment')
        if environment == Environment.PRODUCTION and v:
            logger.warning("Debug mode disabled in production environment")
            return False
        return v
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v, values):
        """Validate CORS origins based on environment"""
        environment = values.get('environment')
        if environment == Environment.PRODUCTION:
            # Filter out localhost origins in production
            return [origin for origin in v if 'localhost' not in origin]
        return v
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist with proper permissions"""
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
                # Set appropriate permissions (755)
                directory.chmod(0o755)
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                # Fall back to system temp directory
                fallback = Path(tempfile.gettempdir()) / directory.name
                fallback.mkdir(parents=True, exist_ok=True)
                setattr(self, f"{directory.name}_path", fallback)
    
    def get_database_url(self) -> Optional[str]:
        """Get database URL with environment variable fallback"""
        return (
            self.database.url or
            os.getenv('DATABASE_URL') or
            os.getenv('POSTGRES_URL') or
            os.getenv('REPLIT_DB_URL')
        )
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_staging(self) -> bool:
        """Check if running in staging mode"""
        return self.environment == Environment.STAGING
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags as a dictionary"""
        return {
            "ai_analysis": self.enable_ai_analysis,
            "real_time_processing": self.enable_real_time_processing,
            "enterprise_features": self.enable_enterprise_features,
            "video_processing": self.enable_video_processing,
            "social_publishing": self.enable_social_publishing
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration summary"""
        return {
            "workers": self.performance.worker_processes,
            "connections": self.performance.worker_connections,
            "caching": self.performance.enable_caching,
            "gzip": self.performance.enable_gzip,
            "upload_limit": self.performance.max_upload_size
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable mapping
        fields = {
            'app_name': {'env': 'APP_NAME'},
            'app_version': {'env': 'APP_VERSION'},
            'environment': {'env': 'ENVIRONMENT'},
            'debug': {'env': 'DEBUG'},
            'host': {'env': 'HOST'},
            'port': {'env': 'PORT'},
            'log_level': {'env': 'LOG_LEVEL'}
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings with directory initialization.
    Uses LRU cache for performance optimization.
    """
    try:
        settings = Settings()
        settings.ensure_directories()
        
        logger.info(f"Configuration loaded for {settings.environment.value} environment")
        logger.debug(f"Feature flags: {settings.get_feature_flags()}")
        
        return settings
        
    except Exception as e:
        logger.error(f"Configuration loading failed: {e}")
        # Return minimal safe configuration
        minimal_settings = Settings()
        minimal_settings.ensure_directories()
        return minimal_settings


def reload_settings() -> Settings:
    """Force reload settings (clears cache)"""
    get_settings.cache_clear()
    return get_settings()


# Export commonly used settings
settings = get_settings()

# Configuration validation on module import
if __name__ != "__main__":
    try:
        # Validate configuration on import
        test_settings = get_settings()
        logger.info(f"✅ Configuration validated for {test_settings.environment.value}")
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
