"""
Health check utilities for FastAPI application.
Provides health and dependency status endpoints.
"""

import time
import os
import sys
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def health_check() -> Dict[str, Any]:
    """Basic health check for the application"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    }

async def dependencies_check() -> Dict[str, Any]:
    """Check external dependencies status"""
    checks = {
        "mongodb": await _check_mongodb(),
        "redis": await _check_redis(),
        "storage": _check_storage(),
        "ffmpeg": _check_ffmpeg()
    }
    
    all_healthy = all(check["status"] == "healthy" for check in checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": time.time(),
        "dependencies": checks
    }

async def _check_mongodb() -> Dict[str, Any]:
    """Check MongoDB connectivity"""
    try:
        # Import here to avoid circular imports
        from ..db.session import get_database
        db = get_database()
        await db.admin.command('ping')
        return {"status": "healthy", "message": "MongoDB connected"}
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        return {"status": "unhealthy", "message": str(e)}

async def _check_redis() -> Dict[str, Any]:
    """Check Redis connectivity"""
    try:
        import redis.asyncio as redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url)
        await r.ping()
        await r.close()
        return {"status": "healthy", "message": "Redis connected"}
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {"status": "unhealthy", "message": str(e)}

def _check_storage() -> Dict[str, Any]:
    """Check storage directories"""
    try:
        directories = ["videos", "uploads", "logs"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            # Test write access
            test_file = os.path.join(directory, ".health_check")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        
        return {"status": "healthy", "message": "Storage accessible"}
    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
        return {"status": "unhealthy", "message": str(e)}

def _check_ffmpeg() -> Dict[str, Any]:
    """Check FFmpeg availability"""
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            return {"status": "healthy", "message": "FFmpeg available"}
        else:
            return {"status": "unhealthy", "message": "FFmpeg not working"}
    except Exception as e:
        logger.error(f"FFmpeg health check failed: {e}")
        return {"status": "unhealthy", "message": str(e)}