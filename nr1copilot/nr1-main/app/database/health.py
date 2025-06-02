
"""
Netflix-Level Database Health Monitoring
Real-time database performance and health tracking
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from app.database.connection import db_manager
from app.database.repositories import repositories

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Health metric data"""
    name: str
    value: float
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    message: str = ""


class DatabaseHealthMonitor:
    """Netflix-level database health monitoring system"""
    
    def __init__(self):
        self.health_checks = {}
        self.metrics_history: Dict[str, List[Dict]] = {}
        self.alerts_sent = set()
        self.monitoring_active = False
        
        # Health thresholds
        self.thresholds = {
            "connection_pool_usage": {"warning": 70.0, "critical": 90.0},
            "avg_query_time": {"warning": 100.0, "critical": 500.0},  # ms
            "active_connections": {"warning": 15, "critical": 18},
            "failed_queries": {"warning": 5.0, "critical": 10.0},  # per minute
            "cache_hit_rate": {"warning": 80.0, "critical": 60.0},  # lower is worse
            "disk_usage": {"warning": 80.0, "critical": 95.0},
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0}
        }
    
    async def start_monitoring(self, interval: int = 60):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("🏥 Starting database health monitoring...")
        
        asyncio.create_task(self._monitoring_loop(interval))
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        logger.info("🛑 Database health monitoring stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self.perform_health_check()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start_time = time.time()
        
        try:
            # Collect all health metrics
            metrics = await self._collect_all_metrics()
            
            # Determine overall health status
            overall_status = self._determine_overall_status(metrics)
            
            # Store health record
            health_record = {
                "timestamp": datetime.utcnow(),
                "status": overall_status.value,
                "metrics": {metric.name: {
                    "value": metric.value,
                    "status": metric.status.value,
                    "unit": metric.unit,
                    "message": metric.message
                } for metric in metrics},
                "check_duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Store in database
            await self._store_health_record(health_record)
            
            # Update history
            self._update_metrics_history(metrics)
            
            # Check for alerts
            await self._check_alerts(metrics)
            
            logger.debug(f"Health check completed: {overall_status.value}")
            return health_record
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "timestamp": datetime.utcnow(),
                "status": HealthStatus.CRITICAL.value,
                "error": str(e),
                "check_duration_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    async def _collect_all_metrics(self) -> List[HealthMetric]:
        """Collect all health metrics"""
        metrics = []
        
        # Database connection metrics
        metrics.extend(await self._collect_connection_metrics())
        
        # Query performance metrics
        metrics.extend(await self._collect_performance_metrics())
        
        # Repository metrics
        metrics.extend(await self._collect_repository_metrics())
        
        # System resource metrics
        metrics.extend(await self._collect_system_metrics())
        
        return metrics
    
    async def _collect_connection_metrics(self) -> List[HealthMetric]:
        """Collect database connection metrics"""
        metrics = []
        
        try:
            pool_stats = await db_manager.get_pool_stats()
            
            if pool_stats.get("status") == "not_initialized":
                metrics.append(HealthMetric(
                    name="database_status",
                    value=0,
                    status=HealthStatus.CRITICAL,
                    threshold_warning=1,
                    threshold_critical=0,
                    message="Database not initialized"
                ))
                return metrics
            
            # Pool usage percentage
            pool_size = pool_stats.get("pool_size", 0)
            max_size = pool_stats.get("pool_max_size", 1)
            pool_usage = (pool_size / max_size) * 100 if max_size > 0 else 0
            
            metrics.append(HealthMetric(
                name="connection_pool_usage",
                value=pool_usage,
                status=self._get_status(pool_usage, "connection_pool_usage"),
                threshold_warning=self.thresholds["connection_pool_usage"]["warning"],
                threshold_critical=self.thresholds["connection_pool_usage"]["critical"],
                unit="%",
                message=f"Pool: {pool_size}/{max_size} connections"
            ))
            
            # Active connections
            active_connections = pool_stats.get("stats", {}).get("active_connections", 0)
            metrics.append(HealthMetric(
                name="active_connections",
                value=active_connections,
                status=self._get_status(active_connections, "active_connections"),
                threshold_warning=self.thresholds["active_connections"]["warning"],
                threshold_critical=self.thresholds["active_connections"]["critical"],
                message=f"{active_connections} active connections"
            ))
            
            # Failed connections
            failed_connections = pool_stats.get("stats", {}).get("failed_connections", 0)
            metrics.append(HealthMetric(
                name="failed_connections",
                value=failed_connections,
                status=HealthStatus.HEALTHY if failed_connections == 0 else HealthStatus.WARNING,
                threshold_warning=1,
                threshold_critical=5,
                message=f"{failed_connections} failed connections"
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect connection metrics: {e}")
            metrics.append(HealthMetric(
                name="connection_error",
                value=1,
                status=HealthStatus.CRITICAL,
                threshold_warning=0,
                threshold_critical=1,
                message=str(e)
            ))
        
        return metrics
    
    async def _collect_performance_metrics(self) -> List[HealthMetric]:
        """Collect database performance metrics"""
        metrics = []
        
        try:
            pool_stats = await db_manager.get_pool_stats()
            stats = pool_stats.get("stats", {})
            
            # Average query time
            avg_query_time_ms = stats.get("avg_query_time", 0) * 1000
            metrics.append(HealthMetric(
                name="avg_query_time",
                value=avg_query_time_ms,
                status=self._get_status(avg_query_time_ms, "avg_query_time"),
                threshold_warning=self.thresholds["avg_query_time"]["warning"],
                threshold_critical=self.thresholds["avg_query_time"]["critical"],
                unit="ms",
                message=f"Average query time: {avg_query_time_ms:.2f}ms"
            ))
            
            # Total queries
            total_queries = stats.get("total_queries", 0)
            metrics.append(HealthMetric(
                name="total_queries",
                value=total_queries,
                status=HealthStatus.HEALTHY,
                threshold_warning=float('inf'),
                threshold_critical=float('inf'),
                message=f"{total_queries} total queries"
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
        
        return metrics
    
    async def _collect_repository_metrics(self) -> List[HealthMetric]:
        """Collect repository performance metrics"""
        metrics = []
        
        try:
            repo_stats = repositories.get_all_stats()
            
            total_cache_hit_rate = 0
            repo_count = 0
            
            for repo_name, stats in repo_stats.items():
                cache_hit_rate = stats.get("cache_hit_rate", 0) * 100
                if cache_hit_rate > 0:
                    total_cache_hit_rate += cache_hit_rate
                    repo_count += 1
            
            if repo_count > 0:
                avg_cache_hit_rate = total_cache_hit_rate / repo_count
                metrics.append(HealthMetric(
                    name="cache_hit_rate",
                    value=avg_cache_hit_rate,
                    status=self._get_status(avg_cache_hit_rate, "cache_hit_rate", reverse=True),
                    threshold_warning=self.thresholds["cache_hit_rate"]["warning"],
                    threshold_critical=self.thresholds["cache_hit_rate"]["critical"],
                    unit="%",
                    message=f"Average cache hit rate: {avg_cache_hit_rate:.1f}%"
                ))
            
        except Exception as e:
            logger.error(f"Failed to collect repository metrics: {e}")
        
        return metrics
    
    async def _collect_system_metrics(self) -> List[HealthMetric]:
        """Collect system resource metrics"""
        metrics = []
        
        try:
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            metrics.append(HealthMetric(
                name="cpu_usage",
                value=cpu_usage,
                status=self._get_status(cpu_usage, "cpu_usage"),
                threshold_warning=self.thresholds["cpu_usage"]["warning"],
                threshold_critical=self.thresholds["cpu_usage"]["critical"],
                unit="%",
                message=f"CPU usage: {cpu_usage:.1f}%"
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            metrics.append(HealthMetric(
                name="memory_usage",
                value=memory_usage,
                status=self._get_status(memory_usage, "memory_usage"),
                threshold_warning=self.thresholds["memory_usage"]["warning"],
                threshold_critical=self.thresholds["memory_usage"]["critical"],
                unit="%",
                message=f"Memory usage: {memory_usage:.1f}%"
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            metrics.append(HealthMetric(
                name="disk_usage",
                value=disk_usage,
                status=self._get_status(disk_usage, "disk_usage"),
                threshold_warning=self.thresholds["disk_usage"]["warning"],
                threshold_critical=self.thresholds["disk_usage"]["critical"],
                unit="%",
                message=f"Disk usage: {disk_usage:.1f}%"
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    def _get_status(self, value: float, metric_name: str, reverse: bool = False) -> HealthStatus:
        """Determine health status based on value and thresholds"""
        thresholds = self.thresholds.get(metric_name, {"warning": 0, "critical": 0})
        warning_threshold = thresholds["warning"]
        critical_threshold = thresholds["critical"]
        
        if reverse:  # For metrics where lower is worse (like cache hit rate)
            if value <= critical_threshold:
                return HealthStatus.CRITICAL
            elif value <= warning_threshold:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
        else:  # For metrics where higher is worse
            if value >= critical_threshold:
                return HealthStatus.CRITICAL
            elif value >= warning_threshold:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
    
    def _determine_overall_status(self, metrics: List[HealthMetric]) -> HealthStatus:
        """Determine overall health status from all metrics"""
        if any(metric.status == HealthStatus.CRITICAL for metric in metrics):
            return HealthStatus.CRITICAL
        elif any(metric.status == HealthStatus.WARNING for metric in metrics):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    async def _store_health_record(self, health_record: Dict[str, Any]):
        """Store health record in database"""
        try:
            await repositories.system_health.create({
                "service_name": "database",
                "status": health_record["status"],
                "response_time_ms": health_record["check_duration_ms"],
                "details": health_record["metrics"]
            })
        except Exception as e:
            logger.error(f"Failed to store health record: {e}")
    
    def _update_metrics_history(self, metrics: List[HealthMetric]):
        """Update metrics history for trending analysis"""
        timestamp = time.time()
        
        for metric in metrics:
            if metric.name not in self.metrics_history:
                self.metrics_history[metric.name] = []
            
            self.metrics_history[metric.name].append({
                "timestamp": timestamp,
                "value": metric.value,
                "status": metric.status.value
            })
            
            # Keep only last 24 hours of history
            cutoff_time = timestamp - (24 * 3600)
            self.metrics_history[metric.name] = [
                entry for entry in self.metrics_history[metric.name]
                if entry["timestamp"] > cutoff_time
            ]
    
    async def _check_alerts(self, metrics: List[HealthMetric]):
        """Check for alert conditions"""
        critical_metrics = [m for m in metrics if m.status == HealthStatus.CRITICAL]
        warning_metrics = [m for m in metrics if m.status == HealthStatus.WARNING]
        
        if critical_metrics:
            alert_key = f"critical_{len(critical_metrics)}"
            if alert_key not in self.alerts_sent:
                logger.error(f"🚨 CRITICAL DATABASE ALERTS: {[m.name for m in critical_metrics]}")
                self.alerts_sent.add(alert_key)
        
        if warning_metrics and not critical_metrics:
            alert_key = f"warning_{len(warning_metrics)}"
            if alert_key not in self.alerts_sent:
                logger.warning(f"⚠️ DATABASE WARNINGS: {[m.name for m in warning_metrics]}")
                self.alerts_sent.add(alert_key)
        
        # Clear alerts if everything is healthy
        if not critical_metrics and not warning_metrics:
            self.alerts_sent.clear()
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        health_check = await self.perform_health_check()
        
        return {
            "overall_status": health_check["status"],
            "last_check": health_check["timestamp"].isoformat(),
            "check_duration_ms": health_check["check_duration_ms"],
            "metrics_count": len(health_check.get("metrics", {})),
            "critical_issues": len([
                m for m in health_check.get("metrics", {}).values()
                if m["status"] == "critical"
            ]),
            "warning_issues": len([
                m for m in health_check.get("metrics", {}).values()
                if m["status"] == "warning"
            ])
        }
    
    async def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information"""
        return await self.perform_health_check()


# Global health monitor instance
health_monitor = DatabaseHealthMonitor()
