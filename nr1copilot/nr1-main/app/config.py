
"""
ViralClip Pro v8.0 - Netflix-Level Configuration System
Enterprise-grade configuration with comprehensive settings management
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class SecuritySettings(BaseSettings):
    """Security configuration with Netflix-level standards"""
    
    # Authentication & Authorization
    secret_key: str = Field(default_factory=lambda: os.urandom(32).hex())
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # CORS Configuration
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5000",
        "https://your-domain.com"
    ]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_headers: List[str] = ["*"]
    
    # Rate Limiting
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600  # 1 hour
    rate_limit_burst: int = 100
    
    # Security Headers
    enable_security_headers: bool = True
    content_security_policy: str = "default-src 'self'; script-src 'self' 'unsafe-inline'"
    
    # Encryption
    encryption_key: Optional[str] = None
    password_hash_rounds: int = 12
    
    class Config:
        env_prefix = "SECURITY_"


class DatabaseSettings(BaseSettings):
    """Database configuration for enterprise scalability"""
    
    # Primary Database
    database_url: str = "sqlite:///./viralclip_pro.db"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    
    # Redis Cache
    redis_url: str = "redis://localhost:6379/0"
    redis_pool_size: int = 10
    redis_timeout: int = 5
    
    # Cache Configuration
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 10000
    cache_compression: bool = True
    
    class Config:
        env_prefix = "DATABASE_"


class AISettings(BaseSettings):
    """AI/ML configuration for Netflix-level performance"""
    
    # Model Configuration
    ai_model_path: str = "./models"
    ai_model_cache_size: int = 100
    ai_batch_size: int = 32
    ai_max_workers: int = 4
    
    # Performance Optimization
    enable_gpu_acceleration: bool = False
    model_quantization: bool = True
    inference_timeout: int = 30
    
    # API Keys (should be set via environment)
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    # Viral Analysis
    viral_threshold: float = 0.8
    sentiment_confidence_threshold: float = 0.7
    trending_factor_weight: float = 0.3
    
    class Config:
        env_prefix = "AI_"


class PerformanceSettings(BaseSettings):
    """Performance optimization configuration"""
    
    # Threading & Concurrency
    max_workers: int = 8
    worker_timeout: int = 300
    async_pool_size: int = 100
    
    # Memory Management
    memory_limit_mb: int = 2048
    gc_threshold: int = 700
    memory_monitoring_interval: int = 60
    
    # Caching
    cache_backend: str = "redis"  # redis, memory, filesystem
    cache_compression_level: int = 6
    cache_serialization: str = "pickle"  # pickle, json, msgpack
    
    # File Processing
    max_file_size_mb: int = 500
    chunk_size: int = 8192
    temp_dir: str = "/tmp/viralclip"
    
    # Monitoring
    metrics_enabled: bool = True
    metrics_interval: int = 30
    health_check_interval: int = 60
    
    class Config:
        env_prefix = "PERFORMANCE_"


class MediaSettings(BaseSettings):
    """Media processing configuration"""
    
    # Video Processing
    video_quality_levels: List[str] = ["720p", "1080p", "4K"]
    default_video_quality: str = "1080p"
    video_compression_level: int = 23  # CRF value for H.264
    
    # Audio Processing
    audio_sample_rate: int = 44100
    audio_bitrate: int = 128
    audio_channels: int = 2
    
    # Image Processing
    image_quality: int = 85
    thumbnail_size: tuple = (320, 180)
    preview_size: tuple = (640, 360)
    
    # Format Support
    supported_video_formats: List[str] = ["mp4", "mov", "avi", "mkv"]
    supported_audio_formats: List[str] = ["mp3", "wav", "aac", "ogg"]
    supported_image_formats: List[str] = ["jpg", "jpeg", "png", "webp"]
    
    # Storage
    storage_backend: str = "filesystem"  # filesystem, s3, gcs
    storage_path: str = "./storage"
    cdn_url: Optional[str] = None
    
    class Config:
        env_prefix = "MEDIA_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json, text
    log_file: str = "./logs/viralclip.log"
    log_rotation: str = "100MB"
    log_retention: int = 30  # days
    
    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Tracing
    tracing_enabled: bool = False
    jaeger_endpoint: Optional[str] = None
    trace_sample_rate: float = 0.1
    
    # Health Checks
    health_check_enabled: bool = True
    health_check_path: str = "/health"
    deep_health_checks: bool = True
    
    # Alerts
    alert_webhook_url: Optional[str] = None
    alert_email: Optional[str] = None
    error_threshold: int = 10
    latency_threshold_ms: int = 1000
    
    class Config:
        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Main application settings with Netflix-level configuration"""
    
    # Application
    app_name: str = "ViralClip Pro v8.0"
    app_version: str = "8.0.0"
    debug: bool = False
    environment: str = "production"  # development, staging, production
    
    # Server
    host: str = "0.0.0.0"
    port: int = 5000
    workers: int = 1
    reload: bool = False
    
    # Feature Flags
    enable_analytics: bool = True
    enable_realtime: bool = True
    enable_ai_features: bool = True
    enable_social_publishing: bool = True
    enable_batch_processing: bool = True
    enable_enterprise_features: bool = True
    
    # Nested Settings
    security: SecuritySettings = SecuritySettings()
    database: DatabaseSettings = DatabaseSettings()
    ai: AISettings = AISettings()
    performance: PerformanceSettings = PerformanceSettings()
    media: MediaSettings = MediaSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    # API Configuration
    api_version: str = "v8"
    api_prefix: str = "/api/v8"
    docs_url: Optional[str] = "/docs"
    redoc_url: Optional[str] = "/redoc"
    openapi_url: Optional[str] = "/openapi.json"
    
    @validator("environment")
    def validate_environment(cls, v):
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    @validator("debug")
    def validate_debug(cls, v, values):
        if values.get("environment") == "production" and v:
            logger.warning("Debug mode should not be enabled in production")
        return v
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "json": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json" if self.monitoring.log_format == "json" else "default",
                    "level": self.monitoring.log_level
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": self.monitoring.log_file,
                    "maxBytes": 100 * 1024 * 1024,  # 100MB
                    "backupCount": 5,
                    "formatter": "json" if self.monitoring.log_format == "json" else "default",
                    "level": self.monitoring.log_level
                }
            },
            "root": {
                "level": self.monitoring.log_level,
                "handlers": ["console", "file"]
            }
        }
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == "production"
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": self.security.cors_origins,
            "allow_methods": self.security.cors_methods,
            "allow_headers": self.security.cors_headers,
            "allow_credentials": True,
            "max_age": 86400
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()


# Environment-specific overrides
if settings.environment == "development":
    settings.debug = True
    settings.reload = True
    settings.monitoring.log_level = "DEBUG"
elif settings.environment == "production":
    settings.debug = False
    settings.reload = False
    settings.monitoring.log_level = "WARNING"
    settings.docs_url = None
    settings.redoc_url = None
    settings.openapi_url = None


# Validation
def validate_settings():
    """Validate configuration settings"""
    issues = []
    
    # Security validation
    if settings.environment == "production":
        if not settings.security.secret_key or len(settings.security.secret_key) < 32:
            issues.append("Production requires a strong secret key")
        
        if settings.debug:
            issues.append("Debug mode should be disabled in production")
    
    # Performance validation
    if settings.performance.max_workers > 16:
        issues.append("Too many workers may cause resource contention")
    
    # Database validation
    if settings.database.database_pool_size < 5:
        issues.append("Database pool size is too small for production")
    
    if issues:
        logger.warning(f"Configuration issues found: {issues}")
    
    return len(issues) == 0


# Initialize validation
if not validate_settings():
    logger.warning("Configuration validation failed")
