"""
Health check utilities for FastAPI application.
Provides health and dependency status endpoints.
"""

from . import runtime
import os
from typing import Any, Dict

def health_check() -> Dict[str, Any]:
    """Return basic health status and environment info."""
    return {
        "status": "ok",
        "environment": runtime.get_env(),
        "version": runtime.get_version(),
        "timestamp": runtime.get_timestamp(),
        "env_vars": {
            "MONGODB_URI": bool(os.getenv("MONGODB_URI")),
            "REDIS_URL": bool(os.getenv("REDIS_URL")),
            "JWT_SECRET": bool(os.getenv("JWT_SECRET")),
            "AWS_S3_BUCKET": bool(os.getenv("AWS_S3_BUCKET")),
            "AWS_REGION": bool(os.getenv("AWS_REGION")),
            "AWS_ACCESS_KEY_ID": bool(os.getenv("AWS_ACCESS_KEY_ID")),
            "AWS_SECRET_ACCESS_KEY": bool(os.getenv("AWS_SECRET_ACCESS_KEY")),
        }
    }

def dependencies_check() -> Dict[str, Any]:
    """Return status of external dependencies (stubbed for now)."""
    # TODO: Add real checks for MongoDB, Redis, FFmpeg, etc.
    return {
        "status": "ok",
        "dependencies": {
            "mongodb": "ok",
            "redis": "ok",
            "ffmpeg": "ok"
        }
    }
