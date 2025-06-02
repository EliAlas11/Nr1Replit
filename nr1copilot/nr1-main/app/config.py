"""
Netflix-Grade Enterprise Configuration Management
Ultra-optimized configuration system with singleton pattern and dependency injection
"""

import os
import logging
import functools
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Application environments with Netflix-grade definitions"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration with connection pooling settings"""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass(frozen=True)
class RedisConfig:
    """Redis configuration for caching and sessions"""
    url: Optional[str]
    ttl: int = 3600
    max_connections: int = 100

@dataclass(frozen=True)
class SecurityConfig:
    """Security configuration with enterprise standards"""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

@dataclass(frozen=True)
class PerformanceConfig:
    """Performance tuning configuration"""
    max_upload_size: int = 500 * 1024 * 1024  # 500MB
    request_timeout: int = 300
    worker_processes: int = 1
    max_connections: int = 1000
    keepalive_timeout: int = 75

class NetflixGradeSettings:
    """Netflix-grade settings with enterprise patterns"""

    def __init__(self):
        # Core application settings
        self.app_name = "ViralClip Pro v10.0"
        self.app_version = "10.0.0"
        self.environment = Environment(os.getenv("ENV", "production").lower())

        # Server configuration
        self.host = "0.0.0.0"
        self.port = int(os.getenv("PORT", "5000"))
        self.debug = self._get_debug_mode()

        # CORS configuration
        self.cors_origins = self._get_cors_origins()

        # Component configurations
        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite:///./viralclip.db"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "30"))
        )

        self.redis = RedisConfig(
            url=os.getenv("REDIS_URL"),
            ttl=int(os.getenv("CACHE_TTL", "3600")),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
        )

        self.security = SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", self._generate_secret_key()),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        )

        self.performance = PerformanceConfig(
            max_upload_size=int(os.getenv("MAX_UPLOAD_SIZE", str(500 * 1024 * 1024))),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "300")),
            worker_processes=int(os.getenv("WORKER_PROCESSES", "1")),
            max_connections=int(os.getenv("MAX_CONNECTIONS", "1000"))
        )

        # Feature flags
        self.features = {
            "analytics": self._get_bool_env("ENABLE_ANALYTICS", True),
            "collaboration": self._get_bool_env("ENABLE_COLLABORATION", True),
            "ai_processing": self._get_bool_env("ENABLE_AI_PROCESSING", True),
            "caching": self._get_bool_env("ENABLE_CACHING", True),
            "real_time": self._get_bool_env("ENABLE_REAL_TIME", True),
            "monitoring": self._get_bool_env("ENABLE_MONITORING", True)
        }

        # AI configuration
        self.ai_config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "max_video_size_mb": int(os.getenv("MAX_VIDEO_SIZE_MB", "500")),
            "processing_timeout": int(os.getenv("PROCESSING_TIMEOUT", "300")),
            "model_cache_size": int(os.getenv("MODEL_CACHE_SIZE", "100"))
        }

        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

        # Initialize directories and logging
        self._initialize_system()

    def _get_debug_mode(self) -> bool:
        """Get debug mode with environment consideration"""
        if self.environment == Environment.PRODUCTION:
            return False
        return os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    def _get_cors_origins(self) -> List[str]:
        """Get CORS origins with security considerations"""
        if self.environment == Environment.PRODUCTION:
            return [
                "https://*.replit.app",
                "https://*.replit.dev",
                "https://*.onrender.com",
                "https://viralclip.pro"
            ]
        elif self.environment == Environment.STAGING:
            return [
                "https://*.replit.app",
                "https://*.replit.dev",
                "http://localhost:*",
                "http://127.0.0.1:*"
            ]
        return ["*"]

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Safely get boolean environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key for production"""
        import secrets
        return secrets.token_urlsafe(32)

    def _initialize_system(self):
        """Initialize system directories and logging"""
        # Create necessary directories
        directories = [
            "./uploads", "./temp", "./logs", "./static", 
            "./cache", "./models", "./backups"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=self.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("./logs/application.log")
            ]
        )

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_staging(self) -> bool:
        """Check if running in staging"""
        return self.environment == Environment.STAGING

    def get_database_url(self) -> str:
        """Get database URL with connection parameters"""
        return self.database.url

    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL for caching"""
        return self.redis.url

    def get_feature_flag(self, feature: str) -> bool:
        """Get feature flag status"""
        return self.features.get(feature, False)

    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI configuration"""
        return self.ai_config.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for serialization"""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment.value,
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "features": self.features,
            "database_url": self.database.url,
            "redis_available": self.redis.url is not None
        }

# Singleton pattern for settings
@functools.lru_cache(maxsize=1)
def get_settings() -> NetflixGradeSettings:
    """Get cached settings instance (Singleton pattern)"""
    return NetflixGradeSettings()

# Backward compatibility
settings = get_settings()

logger.info(f"Netflix-grade configuration loaded for environment: {settings.environment.value}")