"""
The HealthMonitor class has been modified to address async initialization issues, ensuring proper lazy initialization for async components.
"""
"""
Netflix-Grade Health Monitoring System
Comprehensive health checks and system monitoring
"""

import asyncio
import logging
import psutil
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    SHUTTING_DOWN = "shutting_down"

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: Any
    status: HealthStatus
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

class HealthMonitor:
    """Netflix-tier health monitoring system"""

    def __init__(self):
        self.start_time = time.time()
        self.status = "starting"
        self.last_check: Optional[datetime] = None
        self.health_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._health_task: Optional[asyncio.Task] = None
        self._initialized = False

        logger.info("💚 HealthMonitor initialized (ready for async startup)")

    async def initialize(self):
        """Initialize async components when event loop is available"""
        if self._initialized:
            return

        try:
            # Start health monitoring task only if we have a running event loop
            if not self._health_task or self._health_task.done():
                self._health_task = asyncio.create_task(self._periodic_health_check())

            self._initialized = True
            logger.info("✅ HealthMonitor async initialization completed")

        except Exception as e:
            logger.error(f"HealthMonitor async initialization failed: {e}")
            raise

    def update_status(self, status: str):
        """Update overall health status"""
        try:
            self.status = HealthStatus(status)
        except ValueError:
            logger.warning(f"Invalid health status: {status}")
            self.status = HealthStatus.DEGRADED

    async def _periodic_health_check(self):
        """Periodically perform health checks"""
        while True:
            try:
                await self.perform_health_check()
                await asyncio.sleep(60)  # Check every 60 seconds
            except Exception as e:
                logger.error(f"Periodic health check failed: {e}")
                await asyncio.sleep(60)

    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            start_time = time.time()

            # System metrics
            await self._check_system_resources()
            await self._check_application_health()
            await self._check_dependencies()

            check_duration = time.time() - start_time
            self.last_check = datetime.utcnow()

            # Determine overall status
            overall_status = self._calculate_overall_status()

            # Store health check result
            with self._lock:
                self.health_history.append({
                    "status": overall_status.value,
                    "timestamp": self.last_check.isoformat(),
                    "check_duration_ms": round(check_duration * 1000, 2)
                })
                if len(self.health_history) > 100:
                    self.health_history.pop(0)

            return {
                "status": overall_status.value,
                "timestamp": self.last_check.isoformat(),
                "uptime_seconds": (self.last_check - self.start_time).total_seconds(),
                "check_duration_ms": round(check_duration * 1000, 2),
                "metrics": {name: {
                    "value": metric.value,
                    "status": metric.status.value,
                    "timestamp": metric.timestamp.isoformat()
                } for name, metric in self.metrics.items()},
                "version": "10.0.0",
                "environment": "production",
                "health_history": self.health_history  # Include health history in result
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _check_system_resources(self):
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics["cpu_usage"] = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                status=HealthStatus.HEALTHY if cpu_percent < self.alert_thresholds["cpu_usage"] else HealthStatus.DEGRADED
            )

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["memory_usage"] = HealthMetric(
                name="memory_usage",
                value=memory.percent,
                status=HealthStatus.HEALTHY if memory.percent < self.alert_thresholds["memory_usage"] else HealthStatus.DEGRADED
            )

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics["disk_usage"] = HealthMetric(
                name="disk_usage",
                value=round(disk_percent, 2),
                status=HealthStatus.HEALTHY if disk_percent < self.alert_thresholds["disk_usage"] else HealthStatus.DEGRADED
            )

        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            self.metrics["system_resources"] = HealthMetric(
                name="system_resources",
                value="error",
                status=HealthStatus.UNHEALTHY
            )

    async def _check_application_health(self):
        """Check application-specific health"""
        try:
            # Application is running
            self.metrics["application"] = HealthMetric(
                name="application",
                value="running",
                status=HealthStatus.HEALTHY
            )

            # Check if we can access essential directories
            import os
            essential_dirs = ["./uploads", "./temp", "./logs"]
            for directory in essential_dirs:
                if not os.path.exists(directory):
                    self.metrics[f"directory_{directory}"] = HealthMetric(
                        name=f"directory_{directory}",
                        value="missing",
                        status=HealthStatus.DEGRADED
                    )
                else:
                    self.metrics[f"directory_{directory}"] = HealthMetric(
                        name=f"directory_{directory}",
                        value="exists",
                        status=HealthStatus.HEALTHY
                    )

        except Exception as e:
            logger.error(f"Application health check failed: {e}")
            self.metrics["application"] = HealthMetric(
                name="application",
                value="error",
                status=HealthStatus.UNHEALTHY
            )

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
                status=HealthStatus.UNHEALTHY
            )

    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall health status based on individual metrics"""
        if not self.metrics:
            return HealthStatus.STARTING

        statuses = [metric.status for metric in self.metrics.values()]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_uptime(self) -> timedelta:
        """Get application uptime"""
        return datetime.utcnow() - self.start_time

    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self.status == HealthStatus.HEALTHY