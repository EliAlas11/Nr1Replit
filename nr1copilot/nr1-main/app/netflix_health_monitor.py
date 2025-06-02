
"""
Netflix-Grade Health Monitor
Comprehensive system health monitoring and diagnostics
"""

import time
import logging
import asyncio
import psutil
import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: Any
    status: HealthStatus
    timestamp: float = None
    message: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class NetflixHealthMonitor:
    """Netflix-tier comprehensive health monitoring system"""

    def __init__(self):
        self.start_time = time.time()
        self.metrics: Dict[str, HealthMetric] = {}
        self.error_counts = defaultdict(int)
        self.health_history: deque = deque(maxlen=100)
        self.last_check_time = 0
        self.check_interval = 30  # seconds
        self._initialized = False

    async def initialize(self):
        """Initialize async components"""
        if self._initialized:
            return
        try:
            self._initialized = True
            logger.info("🏥 NetflixHealthMonitor initialized")
        except Exception as e:
            logger.error(f"NetflixHealthMonitor initialization failed: {e}")
            raise

    def get_uptime(self) -> timedelta:
        """Get application uptime"""
        return timedelta(seconds=time.time() - self.start_time)

    def update_status(self, status: str):
        """Update overall health status"""
        self.metrics["overall_status"] = HealthMetric(
            name="overall_status",
            value=status,
            status=HealthStatus.HEALTHY if status == "healthy" else HealthStatus.DEGRADED
        )

    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            check_start = time.time()
            
            # Perform all health checks
            await self._check_system_health()
            await self._check_application_health()
            await self._check_dependencies()

            # Calculate overall status
            overall_status = self._calculate_overall_status()
            
            # Store health history
            health_record = {
                "timestamp": check_start,
                "status": overall_status,
                "check_duration": time.time() - check_start
            }
            self.health_history.append(health_record)

            # Build comprehensive response
            return {
                "status": overall_status,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime": {
                    "seconds": self.get_uptime().total_seconds(),
                    "human_readable": str(self.get_uptime())
                },
                "system": await self._get_system_metrics(),
                "application": await self._get_application_metrics(),
                "dependencies": await self._get_dependency_status(),
                "performance": {
                    "check_duration": time.time() - check_start,
                    "last_check": self.last_check_time
                },
                "netflix_tier": "Enterprise AAA+",
                "version": "10.0.0"
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "netflix_tier": "Emergency Mode"
            }

    async def _check_system_health(self):
        """Check system-level health metrics"""
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_status = HealthStatus.HEALTHY
            if cpu_percent > 90:
                cpu_status = HealthStatus.CRITICAL
            elif cpu_percent > 75:
                cpu_status = HealthStatus.DEGRADED

            self.metrics["cpu"] = HealthMetric(
                name="cpu",
                value=cpu_percent,
                status=cpu_status
            )

            # Memory check
            memory = psutil.virtual_memory()
            memory_status = HealthStatus.HEALTHY
            if memory.percent > 90:
                memory_status = HealthStatus.CRITICAL
            elif memory.percent > 80:
                memory_status = HealthStatus.DEGRADED

            self.metrics["memory"] = HealthMetric(
                name="memory",
                value=memory.percent,
                status=memory_status
            )

            # Disk check
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = HealthStatus.HEALTHY
            if disk_percent > 95:
                disk_status = HealthStatus.CRITICAL
            elif disk_percent > 85:
                disk_status = HealthStatus.DEGRADED

            self.metrics["disk"] = HealthMetric(
                name="disk",
                value=disk_percent,
                status=disk_status
            )

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            self.metrics["system"] = HealthMetric(
                name="system",
                value="error",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )

    async def _check_application_health(self) -> Dict[str, Any]:
        """Check application-level health"""
        try:
            # Check critical modules
            critical_modules = [
                "app.config",
                "app.middleware.security",
                "app.middleware.performance",
                "app.middleware.error_handler"
            ]
            
            module_status = {}
            failed_modules = []
            
            for module in critical_modules:
                try:
                    if module in sys.modules:
                        module_status[module] = "loaded"
                    else:
                        # Try to import
                        __import__(module)
                        module_status[module] = "healthy"
                except ImportError as e:
                    module_status[module] = f"failed: {str(e)}"
                    failed_modules.append(module)
                except Exception as e:
                    module_status[module] = f"error: {str(e)}"
                    failed_modules.append(module)

            # Check essential directories
            essential_dirs = ["./uploads", "./temp", "./logs", "./cache"]
            directory_status = {}
            
            for directory in essential_dirs:
                if os.path.exists(directory):
                    directory_status[directory] = "exists"
                else:
                    try:
                        os.makedirs(directory, exist_ok=True)
                        directory_status[directory] = "created"
                    except Exception as e:
                        directory_status[directory] = f"failed: {str(e)}"

            app_status = HealthStatus.HEALTHY
            if failed_modules:
                app_status = HealthStatus.DEGRADED if len(failed_modules) < 2 else HealthStatus.CRITICAL

            self.metrics["application"] = HealthMetric(
                name="application",
                value={
                    "modules": module_status,
                    "directories": directory_status,
                    "failed_modules": failed_modules
                },
                status=app_status
            )

            return {
                "modules": module_status,
                "directories": directory_status,
                "failed_modules": failed_modules,
                "status": app_status.value,
                "error_count": sum(self.error_counts.values()),
                "middleware_active": len([m for m in module_status.values() if "healthy" in str(m) or "loaded" in str(m)]) > 0
            }

        except Exception as e:
            logger.error(f"Application health check failed: {e}")
            self.metrics["application"] = HealthMetric(
                name="application",
                value="error",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )
            return {"status": "error", "error": str(e)}

    async def _check_dependencies(self):
        """Check external dependencies"""
        try:
            # For now, mark dependencies as healthy
            # In a real implementation, you would check database, Redis, APIs, etc.
            self.metrics["dependencies"] = HealthMetric(
                name="dependencies",
                value="available",
                status=HealthStatus.HEALTHY
            )

        except Exception as e:
            logger.error(f"Dependencies check failed: {e}")
            self.metrics["dependencies"] = HealthMetric(
                name="dependencies",
                value="error",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )

    def _calculate_overall_status(self) -> str:
        """Calculate overall health status from all metrics"""
        if not self.metrics:
            return "unknown"

        statuses = [metric.status for metric in self.metrics.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return "critical"
        elif HealthStatus.UNHEALTHY in statuses:
            return "unhealthy"
        elif HealthStatus.DEGRADED in statuses:
            return "degraded"
        else:
            return "healthy"

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            return {
                "cpu_percent": self.metrics.get("cpu", HealthMetric("cpu", 0, HealthStatus.HEALTHY)).value,
                "memory_percent": self.metrics.get("memory", HealthMetric("memory", 0, HealthStatus.HEALTHY)).value,
                "disk_percent": self.metrics.get("disk", HealthMetric("disk", 0, HealthStatus.HEALTHY)).value,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
        except Exception:
            return {"error": "system_metrics_unavailable"}

    async def _get_application_metrics(self) -> Dict[str, Any]:
        """Get current application metrics"""
        app_metric = self.metrics.get("application")
        if app_metric:
            return app_metric.value if isinstance(app_metric.value, dict) else {"status": str(app_metric.value)}
        return {"status": "unknown"}

    async def _get_dependency_status(self) -> Dict[str, Any]:
        """Get dependency status"""
        dep_metric = self.metrics.get("dependencies")
        if dep_metric:
            return {"status": dep_metric.value, "health": dep_metric.status.value}
        return {"status": "unknown"}

    async def get_detailed_diagnostics(self) -> Dict[str, Any]:
        """Get detailed system diagnostics"""
        try:
            diagnostics = {
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "pid": os.getpid(),
                    "working_directory": os.getcwd()
                },
                "resource_usage": await self._get_system_metrics(),
                "application_state": await self._get_application_metrics(),
                "health_history": list(self.health_history)[-10:],  # Last 10 checks
                "error_summary": dict(self.error_counts),
                "uptime": {
                    "seconds": self.get_uptime().total_seconds(),
                    "human_readable": str(self.get_uptime())
                }
            }

            return diagnostics

        except Exception as e:
            logger.error(f"Diagnostics collection failed: {e}")
            return {"error": str(e)}

    def record_error(self, error_type: str):
        """Record an error occurrence"""
        self.error_counts[error_type] += 1

# Global health monitor instance
health_monitor = NetflixHealthMonitor()
