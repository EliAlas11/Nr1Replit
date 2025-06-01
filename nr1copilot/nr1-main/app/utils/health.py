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
"""
Health Check Utilities
Netflix-level health monitoring
"""

import asyncio
import logging
import os
import psutil
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0",
        "uptime": get_uptime()
    }

async def detailed_health_check(redis_client: Optional[Any] = None) -> Dict[str, Any]:
    """Detailed health check with system metrics"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Redis health
        redis_status = "unknown"
        if redis_client:
            try:
                await redis_client.ping()
                redis_status = "healthy"
            except Exception:
                redis_status = "unhealthy"
        
        # Check disk space
        disk_health = "healthy" if disk.percent < 90 else "warning"
        
        # Check memory usage
        memory_health = "healthy" if memory.percent < 90 else "warning"
        
        # Overall status
        overall_status = "healthy"
        if disk_health == "warning" or memory_health == "warning":
            overall_status = "warning"
        if redis_status == "unhealthy":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "version": "3.0.0",
            "uptime": get_uptime(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "disk_percent": disk.percent,
                "disk_free": disk.free
            },
            "services": {
                "redis": redis_status,
                "ffmpeg": check_ffmpeg_availability()
            },
            "health_indicators": {
                "disk": disk_health,
                "memory": memory_health,
                "cpu": "healthy" if cpu_percent < 80 else "warning"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

def get_uptime() -> float:
    """Get system uptime in seconds"""
    try:
        return time.time() - psutil.boot_time()
    except Exception:
        return 0.0

def check_ffmpeg_availability() -> str:
    """Check if ffmpeg is available"""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        return "available" if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"
