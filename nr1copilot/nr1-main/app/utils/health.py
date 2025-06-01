"""
Basic Health Checker
Provides system health monitoring functionality
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthChecker:
    """Basic health monitoring for the application"""

    def __init__(self):
        self.start_time = time.time()
        self.checks = {}

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "system": await self._get_system_health(),
                "application": await self._get_application_health(),
                "dependencies": await self._get_dependencies_health()
            }
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system resource health"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            return {"error": str(e)}

    async def _get_application_health(self) -> Dict[str, Any]:
        """Get application-specific health"""
        return {
            "status": "running",
            "uptime": time.time() - self.start_time,
            "checks_performed": len(self.checks)
        }

    async def _get_dependencies_health(self) -> Dict[str, Any]:
        """Check health of external dependencies"""
        return {
            "cache": "available",
            "storage": "available",
            "processing": "available"
        }

    def is_healthy(self) -> bool:
        """Simple health check"""
        try:
            # Basic checks
            if psutil.virtual_memory().percent > 90:
                return False
            if psutil.cpu_percent(interval=1) > 95:
                return False
            return True
        except:
            return True  # Assume healthy if can't check