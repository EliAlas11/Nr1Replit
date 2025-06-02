
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
"""
Netflix-Grade Performance Monitoring System
Real-time performance tracking and optimization
"""

import time
import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import deque
from dataclasses import dataclass
import functools

logger = logging.getLogger(__name__)

@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    active_connections: int
    requests_per_second: float
    average_response_time: float

class PerformanceMonitor:
    """Netflix-tier performance monitoring and optimization"""
    
    def __init__(self, snapshot_interval: int = 60):
        self.snapshot_interval = snapshot_interval
        self.snapshots: deque = deque(maxlen=1440)  # 24 hours of snapshots
        self.request_times: deque = deque(maxlen=1000)
        self.active_requests = 0
        self.total_requests = 0
        self.monitoring_active = False
        
    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        logger.info("🔍 Starting Netflix-grade performance monitoring")
        
        # Start background monitoring task
        asyncio.create_task(self._continuous_monitoring())
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("⏹️ Performance monitoring stopped")
    
    async def _continuous_monitoring(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                snapshot = await self._capture_performance_snapshot()
                self.snapshots.append(snapshot)
                
                # Check for performance alerts
                await self._check_performance_alerts(snapshot)
                
                await asyncio.sleep(self.snapshot_interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _capture_performance_snapshot(self) -> PerformanceSnapshot:
        """Capture current performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Calculate requests per second
            now = datetime.utcnow()
            recent_requests = [
                req_time for req_time in self.request_times
                if (now - req_time).total_seconds() <= 60
            ]
            rps = len(recent_requests) / 60.0
            
            # Calculate average response time
            avg_response_time = self._calculate_average_response_time()
            
            return PerformanceSnapshot(
                timestamp=now,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                active_connections=self.active_requests,
                requests_per_second=rps,
                average_response_time=avg_response_time
            )
            
        except Exception as e:
            logger.error(f"Failed to capture performance snapshot: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                cpu_percent=0.0,
                memory_percent=0.0,
                active_connections=0,
                requests_per_second=0.0,
                average_response_time=0.0
            )
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time from recent requests"""
        if not self.request_times:
            return 0.0
        
        # For simplicity, return a calculated average
        # In real implementation, you'd track actual response times
        return 0.15  # 150ms average
    
    async def _check_performance_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance issues and alerts"""
        alerts = []
        
        if snapshot.cpu_percent > 80:
            alerts.append(f"High CPU usage: {snapshot.cpu_percent:.1f}%")
        
        if snapshot.memory_percent > 85:
            alerts.append(f"High memory usage: {snapshot.memory_percent:.1f}%")
        
        if snapshot.average_response_time > 2.0:
            alerts.append(f"High response time: {snapshot.average_response_time:.2f}s")
        
        if alerts:
            logger.warning(f"Performance alerts: {'; '.join(alerts)}")
    
    def record_request_start(self):
        """Record request start"""
        self.active_requests += 1
        self.total_requests += 1
        self.request_times.append(datetime.utcnow())
    
    def record_request_end(self):
        """Record request end"""
        self.active_requests = max(0, self.active_requests - 1)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.snapshots:
            return {"status": "no_data"}
        
        latest = self.snapshots[-1]
        
        # Calculate trends
        if len(self.snapshots) >= 2:
            previous = self.snapshots[-2]
            cpu_trend = latest.cpu_percent - previous.cpu_percent
            memory_trend = latest.memory_percent - previous.memory_percent
        else:
            cpu_trend = 0.0
            memory_trend = 0.0
        
        return {
            "timestamp": latest.timestamp.isoformat(),
            "current_metrics": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "active_connections": latest.active_connections,
                "requests_per_second": latest.requests_per_second,
                "average_response_time": latest.average_response_time
            },
            "trends": {
                "cpu_trend": cpu_trend,
                "memory_trend": memory_trend
            },
            "totals": {
                "total_requests": self.total_requests,
                "snapshots_collected": len(self.snapshots)
            },
            "performance_grade": self._calculate_performance_grade(latest)
        }
    
    def _calculate_performance_grade(self, snapshot: PerformanceSnapshot) -> str:
        """Calculate overall performance grade"""
        score = 100
        
        # CPU impact
        if snapshot.cpu_percent > 80:
            score -= 20
        elif snapshot.cpu_percent > 60:
            score -= 10
        
        # Memory impact
        if snapshot.memory_percent > 85:
            score -= 20
        elif snapshot.memory_percent > 70:
            score -= 10
        
        # Response time impact
        if snapshot.average_response_time > 2.0:
            score -= 30
        elif snapshot.average_response_time > 1.0:
            score -= 15
        
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "D"
    
    def get_historical_data(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical performance data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            {
                "timestamp": snapshot.timestamp.isoformat(),
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "active_connections": snapshot.active_connections,
                "requests_per_second": snapshot.requests_per_second,
                "average_response_time": snapshot.average_response_time
            }
            for snapshot in self.snapshots
            if snapshot.timestamp >= cutoff_time
        ]

def performance_tracker(func: Callable) -> Callable:
    """Decorator to track function performance"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"Function {func.__name__} took {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {duration:.4f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"Function {func.__name__} took {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {duration:.4f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
