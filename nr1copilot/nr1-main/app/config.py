"""
Enterprise Configuration Management
Modern configuration system with validation, environment management, and security.
"""

import os
from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class Environment(str, Enum):
    """Application environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SecuritySettings(PydanticBaseSettings):
    """Security-related configuration."""
    secret_key: str = Field(
        default_factory=lambda: os.urandom(32).hex(),
        description="Secret key for JWT tokens and encryption"
    )
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    bcrypt_rounds: int = 12

    class Config:
        env_prefix = "SECURITY_"


class PerformanceSettings(PydanticBaseSettings):
    """Performance and resource configuration."""
    worker_processes: int = Field(default=1, ge=1, le=8)
    max_upload_size: int = Field(default=50 * 1024 * 1024, description="50MB default")
    request_timeout: int = Field(default=30, ge=1, le=300)
    keepalive_timeout: int = Field(default=65, ge=1, le=300)
    max_connections: int = Field(default=1000, ge=1)

    class Config:
        env_prefix = "PERF_"


class DatabaseSettings(PydanticBaseSettings):
    """Database configuration."""
    url: Optional[str] = None
    pool_size: int = Field(default=5, ge=1, le=20)
    max_overflow: int = Field(default=10, ge=0, le=50)
    pool_timeout: int = Field(default=30, ge=1)

    class Config:
        env_prefix = "DB_"


class Settings(PydanticBaseSettings):
    """Main application configuration."""

    # Application basics
    app_name: str = "ViralClip Pro v10.0"
    app_version: str = "10.0.0"
    environment: Environment = Environment.PRODUCTION
    debug: bool = False

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 5000

    # CORS configuration
    cors_origins: List[str] = Field(default_factory=lambda: [
        "https://*.replit.app",
        "https://*.replit.dev", 
        "http://localhost:3000",
        "http://localhost:5000"
    ])

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Nested settings
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    @validator('environment', pre=True)
    def validate_environment(cls, v):
        """Validate and normalize environment."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @validator('debug')
    def validate_debug(cls, v, values):
        """Debug should be False in production."""
        environment = values.get('environment')
        if environment == Environment.PRODUCTION and v:
            return False
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            return "INFO"
        return v.upper()

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_staging(self) -> bool:
        """Check if running in staging mode."""
        return self.environment == Environment.STAGING

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
            'log_level': {'env': 'LOG_LEVEL'},
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    Uses LRU cache for performance optimization.
    """
    return Settings()


# Export commonly used settings
settings = get_settings()