"""
Enterprise Configuration Management - Deployment Optimized
Streamlined configuration for maximum deployment reliability and performance
"""

import os
import logging
from typing import List, Optional
from enum import Enum

# Simple configuration without heavy dependencies
class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Settings:
    """Streamlined settings for deployment reliability"""

    def __init__(self):
        # Core deployment settings
        self.app_name = "ViralClip Pro v10.0"
        self.app_version = "10.0.0"
        self.environment = Environment(os.getenv("ENV", "production").lower())

        # Render.com optimized settings
        self.port = int(os.getenv("PORT", "5000"))
        self.host = "0.0.0.0"
        self.debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

        # Security settings
        self.secret_key = os.getenv("SECRET_KEY", "production-secret-key-change-me")
        self.cors_origins = self._get_cors_origins()

        # Performance settings
        self.max_upload_size = 500 * 1024 * 1024  # 500MB
        self.request_timeout = 300  # 5 minutes
        self.worker_processes = 1  # Single worker for Render

        # Feature flags
        self.enable_analytics = True
        self.enable_collaboration = True
        self.enable_ai_processing = True
        self.enable_caching = True

        # Database settings
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./viralclip.db")

        # Cache settings
        self.redis_url = os.getenv("REDIS_URL")
        self.cache_ttl = 3600

        # AI settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_video_size_mb = 500
        self.processing_timeout = 300

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Create necessary directories
        self._create_directories()

    def _get_cors_origins(self) -> List[str]:
        """Get CORS origins based on environment"""
        if self.is_production():
            return [
                "https://*.replit.app",
                "https://*.replit.dev", 
                "https://*.onrender.com"
            ]
        return ["*"]

    def _create_directories(self):
        """Create necessary directories"""
        directories = ["./uploads", "./temp", "./logs", "./static"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT

    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database_url

    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL"""
        return self.redis_url

# Global settings instance
settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"Configuration loaded for environment: {settings.environment.value}")