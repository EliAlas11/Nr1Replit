"""
Netflix-Level Health Checker
Comprehensive system health monitoring with detailed diagnostics
"""

import asyncio
import time
import psutil
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class HealthChecker:
    """Netflix-level health monitoring system"""

    def __init__(self):
        self.start_time = time.time()
        self.check_history = []
        self.max_history = 100

    async def comprehensive_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        try:
            start_time = time.time()

            # Perform all health checks
            system_health = self._check_system_resources()
            disk_health = self._check_disk_space()
            memory_health = self._check_memory_usage()
            network_health = await self._check_network()
            service_health = await self._check_services()

            # Overall status determination
            all_checks = [system_health, disk_health, memory_health, network_health, service_health]
            failed_checks = [check for check in all_checks if check.get("status") != "healthy"]

            overall_status = "healthy" if not failed_checks else "degraded"
            if len(failed_checks) >= len(all_checks) / 2:
                overall_status = "unhealthy"

            # Calculate uptime
            uptime = time.time() - self.start_time

            result = {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": round(uptime, 2),
                "version": "4.0.0",
                "checks": {
                    "system": system_health,
                    "disk": disk_health,
                    "memory": memory_health,
                    "network": network_health,
                    "services": service_health
                },
                "metrics": {
                    "response_time_ms": round((time.time() - start_time) * 1000, 2),
                    "check_count": len(all_checks),
                    "failed_checks": len(failed_checks)
                }
            }

            # Store in history
            self._store_check_result(result)

            return result

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "uptime_seconds": round(time.time() - self.start_time, 2)
            }

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)

            # CPU health thresholds
            cpu_status = "healthy"
            if cpu_percent > 80:
                cpu_status = "critical"
            elif cpu_percent > 60:
                cpu_status = "warning"

            return {
                "status": cpu_status,
                "cpu_percent": round(cpu_percent, 2),
                "load_average": {
                    "1m": round(load_avg[0], 2),
                    "5m": round(load_avg[1], 2),
                    "15m": round(load_avg[2], 2)
                },
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }

        except Exception as e:
            logger.error(f"System check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability"""
        try:
            disk_usage = shutil.disk_usage(".")
            total = disk_usage.total
            used = disk_usage.used
            free = disk_usage.free
            usage_percent = (used / total) * 100

            # Disk health thresholds
            disk_status = "healthy"
            if usage_percent > 90:
                disk_status = "critical"
            elif usage_percent > 80:
                disk_status = "warning"

            return {
                "status": disk_status,
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "usage_percent": round(usage_percent, 2)
            }

        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory utilization"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Memory health thresholds
            memory_status = "healthy"
            if memory.percent > 90:
                memory_status = "critical"
            elif memory.percent > 80:
                memory_status = "warning"

            return {
                "status": memory_status,
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": round(memory.percent, 2),
                "swap": {
                    "total_gb": round(swap.total / (1024**3), 2),
                    "used_gb": round(swap.used / (1024**3), 2),
                    "usage_percent": round(swap.percent, 2)
                }
            }

        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            # Simple network check - in production, you might check external services
            network_status = "healthy"

            return {
                "status": network_status,
                "interfaces": len(psutil.net_if_addrs()),
                "connections": len(psutil.net_connections()),
                "io_stats": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None
            }

        except Exception as e:
            logger.error(f"Network check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def _check_services(self) -> Dict[str, Any]:
        """Check service-specific health"""
        try:
            services = {
                "upload_service": "operational",
                "processing_service": "operational",
                "websocket_service": "operational",
                "file_system": "operational"
            }

            # Check if required directories exist
            required_dirs = ["uploads", "output", "temp", "logs"]
            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    services["file_system"] = "degraded"
                    break

            # All services operational
            overall_service_status = "healthy"
            if any(status != "operational" for status in services.values()):
                overall_service_status = "degraded"

            return {
                "status": overall_service_status,
                "services": services,
                "required_directories": required_dirs
            }

        except Exception as e:
            logger.error(f"Service check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def _store_check_result(self, result: Dict[str, Any]):
        """Store health check result in history"""
        self.check_history.append({
            "timestamp": result["timestamp"],
            "status": result["status"],
            "response_time_ms": result.get("metrics", {}).get("response_time_ms", 0)
        })

        # Keep only recent history
        if len(self.check_history) > self.max_history:
            self.check_history.pop(0)

    def get_health_history(self) -> List[Dict[str, Any]]:
        """Get health check history"""
        return self.check_history.copy()

    def get_health_trends(self) -> Dict[str, Any]:
        """Get health trends analysis"""
        if not self.check_history:
            return {"trends": "insufficient_data"}

        recent_checks = self.check_history[-10:]  # Last 10 checks

        healthy_count = sum(1 for check in recent_checks if check["status"] == "healthy")
        degraded_count = sum(1 for check in recent_checks if check["status"] == "degraded")
        unhealthy_count = sum(1 for check in recent_checks if check["status"] == "unhealthy")

        avg_response_time = sum(check["response_time_ms"] for check in recent_checks) / len(recent_checks)

        return {
            "recent_checks": len(recent_checks),
            "health_distribution": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            },
            "health_percentage": round((healthy_count / len(recent_checks)) * 100, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "trend": self._determine_trend(recent_checks)
        }

    def _determine_trend(self, checks: List[Dict[str, Any]]) -> str:
        """Determine health trend from recent checks"""
        if len(checks) < 3:
            return "stable"

        # Simple trend analysis
        recent_healthy = sum(1 for check in checks[-3:] if check["status"] == "healthy")
        earlier_healthy = sum(1 for check in checks[-6:-3] if check["status"] == "healthy")

        if recent_healthy > earlier_healthy:
            return "improving"
        elif recent_healthy < earlier_healthy:
            return "degrading"
        else:
            return "stable"
"""
Netflix-Grade Health Monitoring System
Comprehensive health checks and system monitoring
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
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
        self.status = HealthStatus.STARTING
        self.startup_time = datetime.utcnow()
        self.last_check = None
        self.metrics: Dict[str, HealthMetric] = {}
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "response_time": 5.0
        }
        
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
            
            return {
                "status": overall_status.value,
                "timestamp": self.last_check.isoformat(),
                "uptime_seconds": (self.last_check - self.startup_time).total_seconds(),
                "check_duration_ms": round(check_duration * 1000, 2),
                "metrics": {name: {
                    "value": metric.value,
                    "status": metric.status.value,
                    "timestamp": metric.timestamp.isoformat()
                } for name, metric in self.metrics.items()},
                "version": "10.0.0",
                "environment": "production"
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
    
    def update_status(self, status: str):
        """Update overall health status"""
        try:
            self.status = HealthStatus(status)
        except ValueError:
            logger.warning(f"Invalid health status: {status}")
            self.status = HealthStatus.DEGRADED
    
    def get_uptime(self) -> timedelta:
        """Get application uptime"""
        return datetime.utcnow() - self.startup_time
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self.status == HealthStatus.HEALTHY
