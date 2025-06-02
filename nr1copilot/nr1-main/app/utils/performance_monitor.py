
"""
ViralClip Pro v6.0 - Netflix-Level Performance Monitoring
Real-time performance tracking and optimization
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    response_times: List[float]
    throughput: float
    error_rate: float
    cache_hit_rate: float


class NetflixLevelPerformanceMonitor:
    """Enterprise-grade performance monitoring system"""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.performance_alerts = []
        self.monitoring_active = False
        self.thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 85.0,
            "memory_warning": 80.0,
            "memory_critical": 90.0,
            "response_time_warning": 2.0,
            "response_time_critical": 5.0,
            "error_rate_warning": 0.05,
            "error_rate_critical": 0.10
        }

    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Check thresholds and generate alerts
                await self._check_performance_thresholds(metrics)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system metrics"""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network_io = psutil.net_io_counters()
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            network_io={
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv
            },
            active_connections=len(psutil.net_connections()),
            response_times=[],  # Will be populated by request middleware
            throughput=0.0,     # Will be calculated from request data
            error_rate=0.0,     # Will be calculated from error logs
            cache_hit_rate=0.0  # Will be populated by cache service
        )

    async def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and generate alerts"""
        
        alerts = []
        
        # CPU alerts
        if metrics.cpu_usage >= self.thresholds["cpu_critical"]:
            alerts.append({
                "level": "critical",
                "metric": "cpu_usage",
                "value": metrics.cpu_usage,
                "message": f"Critical CPU usage: {metrics.cpu_usage:.1f}%"
            })
        elif metrics.cpu_usage >= self.thresholds["cpu_warning"]:
            alerts.append({
                "level": "warning", 
                "metric": "cpu_usage",
                "value": metrics.cpu_usage,
                "message": f"High CPU usage: {metrics.cpu_usage:.1f}%"
            })
        
        # Memory alerts
        if metrics.memory_usage >= self.thresholds["memory_critical"]:
            alerts.append({
                "level": "critical",
                "metric": "memory_usage", 
                "value": metrics.memory_usage,
                "message": f"Critical memory usage: {metrics.memory_usage:.1f}%"
            })
        elif metrics.memory_usage >= self.thresholds["memory_warning"]:
            alerts.append({
                "level": "warning",
                "metric": "memory_usage",
                "value": metrics.memory_usage, 
                "message": f"High memory usage: {metrics.memory_usage:.1f}%"
            })
        
        # Store alerts
        for alert in alerts:
            alert["timestamp"] = datetime.utcnow()
            alert["id"] = str(uuid.uuid4())
            self.performance_alerts.append(alert)
            
            # Log critical alerts
            if alert["level"] == "critical":
                logger.critical(alert["message"])
            else:
                logger.warning(alert["message"])

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest = self.metrics_history[-1]
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "cpu_usage": latest.cpu_usage,
            "memory_usage": latest.memory_usage,
            "disk_usage": latest.disk_usage,
            "active_connections": latest.active_connections,
            "status": self._get_overall_status(latest),
            "performance_score": self._calculate_performance_score(latest)
        }

    async def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over specified time period"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"status": "insufficient_data"}
        
        # Calculate trends
        cpu_trend = [m.cpu_usage for m in recent_metrics]
        memory_trend = [m.memory_usage for m in recent_metrics]
        
        return {
            "time_period_hours": hours,
            "data_points": len(recent_metrics),
            "cpu_trend": {
                "values": cpu_trend,
                "average": sum(cpu_trend) / len(cpu_trend),
                "max": max(cpu_trend),
                "min": min(cpu_trend)
            },
            "memory_trend": {
                "values": memory_trend,
                "average": sum(memory_trend) / len(memory_trend),
                "max": max(memory_trend),
                "min": min(memory_trend)
            },
            "performance_summary": self._generate_performance_summary(recent_metrics)
        }

    def _get_overall_status(self, metrics: PerformanceMetrics) -> str:
        """Determine overall system status"""
        
        if (metrics.cpu_usage >= self.thresholds["cpu_critical"] or 
            metrics.memory_usage >= self.thresholds["memory_critical"]):
            return "critical"
        elif (metrics.cpu_usage >= self.thresholds["cpu_warning"] or 
              metrics.memory_usage >= self.thresholds["memory_warning"]):
            return "warning"
        else:
            return "healthy"

    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score (0-100)"""
        
        cpu_score = max(0, 100 - metrics.cpu_usage)
        memory_score = max(0, 100 - metrics.memory_usage)
        disk_score = max(0, 100 - metrics.disk_usage)
        
        # Weighted average
        overall_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
        
        return round(overall_score, 1)

    def _generate_performance_summary(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Generate performance summary for the time period"""
        
        if not metrics_list:
            return {}
        
        cpu_values = [m.cpu_usage for m in metrics_list]
        memory_values = [m.memory_usage for m in metrics_list]
        
        return {
            "avg_cpu": round(sum(cpu_values) / len(cpu_values), 1),
            "avg_memory": round(sum(memory_values) / len(memory_values), 1),
            "peak_cpu": max(cpu_values),
            "peak_memory": max(memory_values),
            "stability_score": self._calculate_stability_score(cpu_values, memory_values),
            "recommendation": self._generate_performance_recommendation(cpu_values, memory_values)
        }

    def _calculate_stability_score(self, cpu_values: List[float], memory_values: List[float]) -> float:
        """Calculate system stability score based on variance"""
        
        import statistics
        
        cpu_variance = statistics.variance(cpu_values) if len(cpu_values) > 1 else 0
        memory_variance = statistics.variance(memory_values) if len(memory_values) > 1 else 0
        
        # Lower variance = higher stability
        cpu_stability = max(0, 100 - cpu_variance)
        memory_stability = max(0, 100 - memory_variance)
        
        return round((cpu_stability + memory_stability) / 2, 1)

    def _generate_performance_recommendation(self, cpu_values: List[float], memory_values: List[float]) -> str:
        """Generate performance optimization recommendation"""
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        avg_memory = sum(memory_values) / len(memory_values)
        
        if avg_cpu > 80 and avg_memory > 80:
            return "Consider scaling up resources - both CPU and memory are under pressure"
        elif avg_cpu > 80:
            return "CPU-bound workload detected - consider CPU optimization or scaling"
        elif avg_memory > 80:
            return "Memory-intensive workload - consider memory optimization or increase allocation"
        elif avg_cpu < 30 and avg_memory < 50:
            return "System resources are underutilized - consider cost optimization"
        else:
            return "System performance is optimal"

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("📊 Performance monitoring stopped")


# Global performance monitor instance
performance_monitor = NetflixLevelPerformanceMonitor()
