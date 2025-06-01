"""
Health Check Utilities
Netflix-level health monitoring and status checks
"""

import psutil
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

"""
Health check utilities for ViralClip Pro
Netflix-level health monitoring
"""

import logging
import os
from typing import Dict, Any
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "service": "ViralClip Pro"
    }

async def detailed_health_check(redis_client=None) -> Dict[str, Any]:
    """Comprehensive health check with system metrics"""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0",
            "service": "ViralClip Pro",
            "checks": {}
        }

        # System health
        try:
            if psutil:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                health_data["system"] = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "disk_percent": disk.percent,
                    "disk_free_gb": round(disk.free / (1024**3), 2)
                }
            else:
                health_data["system"] = {
                    "cpu_percent": 0,
                    "memory_percent": 0,
                    "memory_available_gb": 0,
                    "disk_percent": 0,
                    "disk_free_gb": 0,
                    "note": "psutil not available"
                }

            health_data["checks"]["system"] = "healthy"

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            health_data["checks"]["system"] = f"unhealthy: {e}"

        # Redis health check
        if redis_client:
            try:
                await redis_client.ping()
                health_data["checks"]["redis"] = "healthy"
            except Exception as e:
                health_data["checks"]["redis"] = f"unhealthy: {e}"
        else:
            health_data["checks"]["redis"] = "not_configured"

        # File system checks
        try:
            required_dirs = ["uploads", "videos", "temp", "output"]
            for directory in required_dirs:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
            health_data["checks"]["filesystem"] = "healthy"
        except Exception as e:
            health_data["checks"]["filesystem"] = f"unhealthy: {e}"

        # Overall status
        failed_checks = [k for k, v in health_data["checks"].items() if "unhealthy" in str(v)]
        if failed_checks:
            health_data["status"] = "degraded"
            health_data["failed_checks"] = failed_checks

        return health_data

    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "service": "ViralClip Pro"
        }

        # Redis health
        if redis_client:
            try:
                await redis_client.ping()
                health_data["checks"]["redis"] = "healthy"
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")
                health_data["checks"]["redis"] = f"unhealthy: {e}"
        else:
            health_data["checks"]["redis"] = "not_configured"

        # Disk space check
        try:
            temp_space = psutil.disk_usage('/tmp')
            health_data["storage"] = {
                "temp_free_gb": round(temp_space.free / (1024**3), 2),
                "temp_percent": temp_space.percent
            }

            if temp_space.percent > 90:
                health_data["checks"]["storage"] = "warning: low disk space"
            else:
                health_data["checks"]["storage"] = "healthy"

        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            health_data["checks"]["storage"] = f"unhealthy: {e}"

        # FFmpeg availability
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                health_data["checks"]["ffmpeg"] = "healthy"
            else:
                health_data["checks"]["ffmpeg"] = "unhealthy: not available"

        except Exception as e:
            logger.error(f"FFmpeg health check failed: {e}")
            health_data["checks"]["ffmpeg"] = f"unhealthy: {e}"

        # Overall status
        unhealthy_checks = [
            check for check, status in health_data["checks"].items()
            if status.startswith("unhealthy")
        ]

        if unhealthy_checks:
            health_data["status"] = "degraded"
            health_data["issues"] = unhealthy_checks

        return health_data

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

async def check_dependencies() -> Dict[str, str]:
    """Check external dependencies"""
    dependencies = {}

    # Check yt-dlp
    try:
        import yt_dlp
        dependencies["yt-dlp"] = "available"
    except ImportError:
        dependencies["yt-dlp"] = "missing"

    # Check FFmpeg
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        dependencies["ffmpeg"] = "available" if process.returncode == 0 else "error"
    except FileNotFoundError:
        dependencies["ffmpeg"] = "missing"
    except Exception:
        dependencies["ffmpeg"] = "error"

    return dependencies