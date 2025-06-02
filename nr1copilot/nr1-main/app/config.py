"""
Enterprise Configuration Management
Netflix-level configuration with environment-specific settings
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from enum import Enum


class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SecuritySettings(BaseSettings):
    """Security configuration"""
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    cors_origins: List[str] = ["*"]
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600

    class Config:
        env_prefix = "SECURITY_"


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    url: str = os.getenv("DATABASE_URL", "sqlite:///./viralclip.db")
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600

    class Config:
        env_prefix = "DB_"


class CacheSettings(BaseSettings):
    """Cache configuration"""
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    default_ttl: int = 3600
    max_memory_mb: int = 512

    class Config:
        env_prefix = "CACHE_"


class AISettings(BaseSettings):
    """AI/ML configuration"""
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    model_cache_dir: str = "./models"
    max_video_size_mb: int = 500
    processing_timeout: int = 300

    class Config:
        env_prefix = "AI_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    metrics_interval: int = 60
    log_level: str = "INFO"
    structured_logging: bool = True

    class Config:
        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Main application settings"""

    # Core settings
    app_name: str = "ViralClip Pro"
    app_version: str = "10.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 5000

    # Paths
    upload_path: str = "./uploads"
    temp_path: str = "./temp"
    static_path: str = "./static"

    # Feature flags
    enable_analytics: bool = True
    enable_collaboration: bool = True
    enable_realtime: bool = True
    enable_ai_processing: bool = True

    # Sub-configurations
    security: SecuritySettings = SecuritySettings()
    database: DatabaseSettings = DatabaseSettings()
    cache: CacheSettings = CacheSettings()
    ai: AISettings = AISettings()
    monitoring: MonitoringSettings = MonitoringSettings()

    @validator("debug", pre=True)
    def debug_from_env(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return v

    @validator("environment", pre=True)
    def environment_from_env(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Environment-specific optimizations
if settings.environment == Environment.PRODUCTION:
    settings.debug = False
    settings.monitoring.log_level = "WARNING"
    settings.security.cors_origins = ["https://*.replit.app"]
    settings.cache.default_ttl = 7200
elif settings.environment == Environment.DEVELOPMENT:
    settings.debug = True
    settings.monitoring.log_level = "DEBUG"
    settings.security.cors_origins = ["*"]