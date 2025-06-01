"""
Configuration module for Viral Clip Generator backend.
- Loads environment variables and settings using Pydantic.
- Provides robust CORS origins parsing for Render.com compatibility.
"""
import os
import json
import logging
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List

# Load environment variables from .env if present
load_dotenv()

def parse_cors_origins(value: str) -> List[str]:
    """Parse CORS_ORIGINS env var as JSON array, comma-separated, or wildcard."""
    # If the value is empty or None, allow all origins
    if not value or value.strip() == "":
        return ["*"]
    value = value.strip()
    # Try to parse as JSON array (Render.com sometimes sets env as '["*"]')
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    # Fallback: comma-separated string
    return [v.strip() for v in value.split(",") if v.strip()]

class Settings(BaseSettings):
    """App settings loaded from environment variables."""
    ENV: str = Field(default=os.getenv("ENV", "production"), description="App environment")
    PORT: int = Field(default=int(os.getenv("PORT", 5000)), description="App port")
    MONGODB_URI: str = Field(default=os.getenv("MONGODB_URI", ""), description="MongoDB connection URI")
    REDIS_URL: str = Field(default=os.getenv("REDIS_URL", ""), description="Redis connection URL")
    JWT_SECRET: str = Field(default=os.getenv("JWT_SECRET", ""), description="JWT secret key")
    AWS_S3_BUCKET: str = Field(default=os.getenv("AWS_S3_BUCKET", ""), description="AWS S3 bucket name")
    AWS_REGION: str = Field(default=os.getenv("AWS_REGION", ""), description="AWS region")
    AWS_ACCESS_KEY_ID: str = Field(default=os.getenv("AWS_ACCESS_KEY_ID", ""), description="AWS access key ID")
    AWS_SECRET_ACCESS_KEY: str = Field(default=os.getenv("AWS_SECRET_ACCESS_KEY", ""), description="AWS secret access key")
    VIDEO_STORAGE_PATH: str = Field(default=os.getenv("VIDEO_STORAGE_PATH", "/opt/render/project/src/nr1-main/videos"), description="Video storage path")
    LOG_FILE_PATH: str = Field(default=os.getenv("LOG_FILE_PATH", ""), description="Log file path")
    CORS_ORIGINS_RAW: str = Field(default=os.getenv("CORS_ORIGINS", "*"), description="Raw CORS origins env var")

    @property
    def CORS_ORIGINS(self) -> List[str]:
        """Return parsed CORS origins as a list."""
        return parse_cors_origins(self.CORS_ORIGINS_RAW)

settings = Settings()

# Startup logging for diagnostics
logging.basicConfig(level=logging.INFO)
logging.info(f"ENV: {settings.ENV}")
logging.info(f"PORT: {settings.PORT}")
logging.info(f"CORS_ORIGINS: {settings.CORS_ORIGINS}")
logging.info(f"VIDEO_STORAGE_PATH: {settings.VIDEO_STORAGE_PATH}")
