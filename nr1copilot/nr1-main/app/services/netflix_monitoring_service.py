
"""
Netflix Monitoring Service v10.0
Comprehensive system monitoring with predictive analytics and real-time insights
"""

import asyncio
import logging
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import weakref

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics being monitored"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: Any
    timestamp: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    """System alert definition"""
    id: str
    metric_name: str
    level: AlertLevel
    threshold: float
    condition: str
    message: str
    timestamp: float
    resolved: bool = False


@dataclass
class PerformanceSnapshot:
    """Complete system performance snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    response_times: List[float]
    error_count: int
    active_connections: int


class NetflixMonitoringService:
    """Netflix-grade monitoring service with comprehensive metrics and alerting"""
    
    def __init__(self):
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.performance_snapshots: deque = deque(maxlen=1000)
        
        # Alert system
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        self.alert_history: deque = deque(maxlen=1000)
        
        # Monitoring configuration
        self.monitoring_intervals = {
            MetricType.SYSTEM: 30,      # 30 seconds
            MetricType.APPLICATION: 60,  # 1 minute
            MetricType.BUSINESS: 300,   # 5 minutes
            MetricType.CUSTOM: 60       # 1 minute
        }
        
        # Thresholds
        self.thresholds = {
            "cpu_percent": {"warning": 70.0, "critical": 90.0},
            "memory_percent": {"warning": 80.0, "critical": 95.0},
            "disk_percent": {"warning": 85.0, "critical": 95.0},
            "response_time_ms": {"warning": 1000.0, "critical": 5000.0},
            "error_rate": {"warning": 0.05, "critical": 0.10}
        }
        
        # Background tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._monitoring_active = False
        
        # Performance tracking
        self.start_time = time.time()
        self._request_times: deque = deque(maxlen=1000)
        self._error_count = 0
        
        logger.info("📊 Netflix Monitoring Service v10.0 initialized")
    
    async def start_monitoring(self) -> None:
        """Start all monitoring background tasks"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Start monitoring tasks
        monitoring_tasks = [
            self._system_metrics_loop(),
            self._application_metrics_loop(),
            self._performance_analysis_loop(),
            self._alert_processing_loop(),
            self._cleanup_old_metrics_loop()
        ]
        
        for task_coro in monitoring_tasks:
            task = asyncio.create_task(task_coro)
            self._monitoring_tasks.append(task)
        
        logger.info("🚀 Netflix monitoring started - all systems operational")
    
    async def stop_monitoring(self) -> None:
        """Stop all monitoring tasks"""
        self._monitoring_active = False
        
        # Cancel all tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        self._monitoring_tasks.clear()
        logger.info("⏹️ Netflix monitoring stopped")
    
    async def _system_metrics_loop(self) -> None:
        """Continuous system metrics collection"""
        while self._monitoring_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.monitoring_intervals[MetricType.SYSTEM])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _application_metrics_loop(self) -> None:
        """Continuous application metrics collection"""
        while self._monitoring_active:
            try:
                await self._collect_application_metrics()
                await asyncio.sleep(self.monitoring_intervals[MetricType.APPLICATION])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Application metrics collection error: {e}")
                await asyncio.sleep(120)
    
    async def _performance_analysis_loop(self) -> None:
        """Continuous performance analysis and prediction"""
        while self._monitoring_active:
            try:
                await self._analyze_performance_trends()
                await asyncio.sleep(300)  # Every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
                await asyncio.sleep(600)
    
    async def _alert_processing_loop(self) -> None:
        """Continuous alert processing and notification"""
        while self._monitoring_active:
            try:
                await self._process_alerts()
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_old_metrics_loop(self) -> None:
        """Cleanup old metrics to prevent memory growth"""
        while self._monitoring_active:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(3600)  # Every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(1800)
    
    async def _collect_system_metrics(self) -> None:
        """Collect comprehensive system metrics"""
        try:
            current_time = time.time()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            await self.record_metric("cpu_percent", cpu_percent, MetricType.SYSTEM, unit="%")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.record_metric("memory_percent", memory.percent, MetricType.SYSTEM, unit="%")
            await self.record_metric("memory_available_gb", memory.available / (1024**3), MetricType.SYSTEM, unit="GB")
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self.record_metric("disk_percent", disk_percent, MetricType.SYSTEM, unit="%")
            await self.record_metric("disk_free_gb", disk.free / (1024**3), MetricType.SYSTEM, unit="GB")
            
            # Network metrics
            network = psutil.net_io_counters()
            await self.record_metric("network_bytes_sent", network.bytes_sent, MetricType.SYSTEM, unit="bytes")
            await self.record_metric("network_bytes_recv", network.bytes_recv, MetricType.SYSTEM, unit="bytes")
            
            # Load average (if available)
            if hasattr(psutil, 'getloadavg'):
                try:
                    load_avg = psutil.getloadavg()
                    await self.record_metric("load_average_1m", load_avg[0], MetricType.SYSTEM)
                    await self.record_metric("load_average_5m", load_avg[1], MetricType.SYSTEM)
                    await self.record_metric("load_average_15m", load_avg[2], MetricType.SYSTEM)
                except Exception:
                    pass
            
            # Create performance snapshot
            snapshot = PerformanceSnapshot(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk_percent,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                response_times=list(self._request_times)[-100:],  # Last 100 response times
                error_count=self._error_count,
                active_connections=0  # Would be populated from actual connection count
            )
            
            self.performance_snapshots.append(snapshot)
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    async def _collect_application_metrics(self) -> None:
        """Collect application-specific metrics"""
        try:
            current_time = time.time()
            
            # Application uptime
            uptime_seconds = current_time - self.start_time
            await self.record_metric("uptime_seconds", uptime_seconds, MetricType.APPLICATION, unit="seconds")
            
            # Request metrics
            if self._request_times:
                avg_response_time = sum(self._request_times) / len(self._request_times) * 1000  # Convert to ms
                await self.record_metric("avg_response_time_ms", avg_response_time, MetricType.APPLICATION, unit="ms")
                
                # 95th percentile response time
                sorted_times = sorted(self._request_times)
                p95_index = int(len(sorted_times) * 0.95)
                p95_response_time = sorted_times[p95_index] * 1000 if sorted_times else 0
                await self.record_metric("p95_response_time_ms", p95_response_time, MetricType.APPLICATION, unit="ms")
            
            # Error rate
            total_requests = len(self._request_times)
            error_rate = (self._error_count / total_requests) if total_requests > 0 else 0
            await self.record_metric("error_rate", error_rate, MetricType.APPLICATION, unit="ratio")
            
            # Requests per minute
            minute_ago = current_time - 60
            recent_requests = sum(1 for t in self._request_times if t > minute_ago)
            await self.record_metric("requests_per_minute", recent_requests, MetricType.APPLICATION, unit="req/min")
            
        except Exception as e:
            logger.error(f"Application metrics collection failed: {e}")
    
    async def _analyze_performance_trends(self) -> None:
        """Analyze performance trends and predict issues"""
        try:
            if len(self.performance_snapshots) < 5:
                return
            
            recent_snapshots = list(self.performance_snapshots)[-10:]  # Last 10 snapshots
            
            # Calculate trends
            cpu_trend = self._calculate_trend([s.cpu_percent for s in recent_snapshots])
            memory_trend = self._calculate_trend([s.memory_percent for s in recent_snapshots])
            
            # Record trend metrics
            await self.record_metric("cpu_trend", cpu_trend["slope"], MetricType.APPLICATION)
            await self.record_metric("memory_trend", memory_trend["slope"], MetricType.APPLICATION)
            
            # Predictive alerting
            if cpu_trend["slope"] > 5.0:  # CPU increasing rapidly
                await self._create_alert(
                    "cpu_trend_warning",
                    "cpu_percent",
                    AlertLevel.WARNING,
                    90.0,
                    "increasing",
                    f"CPU usage trending upward (slope: {cpu_trend['slope']:.2f})"
                )
            
            if memory_trend["slope"] > 3.0:  # Memory increasing
                await self._create_alert(
                    "memory_trend_warning",
                    "memory_percent",
                    AlertLevel.WARNING,
                    95.0,
                    "increasing",
                    f"Memory usage trending upward (slope: {memory_trend['slope']:.2f})"
                )
            
        except Exception as e:
            logger.error(f"Performance trend analysis failed: {e}")
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend slope and direction for a series of values"""
        if len(values) < 2:
            return {"slope": 0.0, "direction": "stable"}
        
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * v for i, v in enumerate(values))
        x2_sum = sum(i ** 2 for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum ** 2) if (n * x2_sum - x_sum ** 2) != 0 else 0
        
        direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        
        return {"slope": slope, "direction": direction}
    
    async def _process_alerts(self) -> None:
        """Process and check all alert conditions"""
        try:
            for metric_name, thresholds in self.thresholds.items():
                if metric_name in self.metrics and self.metrics[metric_name]:
                    latest_metric = self.metrics[metric_name][-1]
                    value = latest_metric.value
                    
                    # Check critical threshold
                    if value >= thresholds["critical"]:
                        await self._create_alert(
                            f"{metric_name}_critical",
                            metric_name,
                            AlertLevel.CRITICAL,
                            thresholds["critical"],
                            ">=",
                            f"{metric_name} is critically high: {value:.2f}"
                        )
                    # Check warning threshold
                    elif value >= thresholds["warning"]:
                        await self._create_alert(
                            f"{metric_name}_warning",
                            metric_name,
                            AlertLevel.WARNING,
                            thresholds["warning"],
                            ">=",
                            f"{metric_name} is high: {value:.2f}"
                        )
                    else:
                        # Resolve alert if value is back to normal
                        await self._resolve_alert(f"{metric_name}_warning")
                        await self._resolve_alert(f"{metric_name}_critical")
            
        except Exception as e:
            logger.error(f"Alert processing failed: {e}")
    
    async def _create_alert(self, alert_id: str, metric_name: str, level: AlertLevel, 
                          threshold: float, condition: str, message: str) -> None:
        """Create or update an alert"""
        if alert_id not in self.alerts or self.alerts[alert_id].resolved:
            alert = Alert(
                id=alert_id,
                metric_name=metric_name,
                level=level,
                threshold=threshold,
                condition=condition,
                message=message,
                timestamp=time.time(),
                resolved=False
            )
            
            self.alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Trigger alert handlers
            for handler in self.alert_handlers[level]:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
            
            logger.warning(f"🚨 ALERT [{level.value.upper()}]: {message}")
    
    async def _resolve_alert(self, alert_id: str) -> None:
        """Resolve an active alert"""
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            self.alerts[alert_id].resolved = True
            logger.info(f"✅ RESOLVED: Alert {alert_id}")
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics to prevent memory growth"""
        try:
            current_time = time.time()
            cutoff_time = current_time - 86400  # 24 hours ago
            
            for metric_name, metric_deque in self.metrics.items():
                # Remove metrics older than 24 hours
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()
            
            # Clean up old performance snapshots
            while self.performance_snapshots and self.performance_snapshots[0].timestamp < cutoff_time:
                self.performance_snapshots.popleft()
            
            # Clean up old alerts
            while self.alert_history and self.alert_history[0].timestamp < cutoff_time:
                self.alert_history.popleft()
            
            logger.debug("🧹 Old metrics cleaned up")
            
        except Exception as e:
            logger.error(f"Metrics cleanup failed: {e}")
    
    # Public API methods
    async def record_metric(self, name: str, value: Any, metric_type: MetricType, 
                          tags: Dict[str, str] = None, unit: str = "") -> None:
        """Record a new metric value"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            metric_type=metric_type,
            tags=tags or {},
            unit=unit
        )
        
        self.metrics[name].append(metric)
    
    def record_request_time(self, response_time: float) -> None:
        """Record request response time"""
        self._request_times.append(response_time)
    
    def record_error(self) -> None:
        """Record an error occurrence"""
        self._error_count += 1
    
    def add_alert_handler(self, level: AlertLevel, handler: Callable) -> None:
        """Add alert handler for specific alert level"""
        self.alert_handlers[level].append(handler)
    
    def get_metrics(self, metric_name: str, limit: int = 100) -> List[Metric]:
        """Get recent metrics for a specific metric name"""
        if metric_name in self.metrics:
            return list(self.metrics[metric_name])[-limit:]
        return []
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        current_time = time.time()
        
        # Get latest metrics
        latest_metrics = {}
        for metric_name, metric_deque in self.metrics.items():
            if metric_deque:
                latest_metrics[metric_name] = {
                    "value": metric_deque[-1].value,
                    "timestamp": metric_deque[-1].timestamp,
                    "unit": metric_deque[-1].unit
                }
        
        # Active alerts
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        
        # Performance summary
        if self.performance_snapshots:
            latest_snapshot = self.performance_snapshots[-1]
            performance_summary = {
                "cpu_percent": latest_snapshot.cpu_percent,
                "memory_percent": latest_snapshot.memory_percent,
                "disk_percent": latest_snapshot.disk_percent,
                "error_count": latest_snapshot.error_count
            }
        else:
            performance_summary = {}
        
        return {
            "timestamp": current_time,
            "uptime_seconds": current_time - self.start_time,
            "monitoring_active": self._monitoring_active,
            "latest_metrics": latest_metrics,
            "active_alerts_count": len(active_alerts),
            "active_alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
                for alert in active_alerts
            ],
            "performance_summary": performance_summary,
            "total_metrics_collected": sum(len(deque) for deque in self.metrics.values()),
            "total_alerts_generated": len(self.alert_history)
        }
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        if not self.performance_snapshots:
            return {"error": "No performance data available"}
        
        recent_snapshots = list(self.performance_snapshots)[-50:]  # Last 50 snapshots
        
        return {
            "timestamps": [s.timestamp for s in recent_snapshots],
            "cpu_usage": [s.cpu_percent for s in recent_snapshots],
            "memory_usage": [s.memory_percent for s in recent_snapshots],
            "disk_usage": [s.disk_percent for s in recent_snapshots],
            "error_counts": [s.error_count for s in recent_snapshots],
            "active_connections": [s.active_connections for s in recent_snapshots],
            "current_time": time.time()
        }


# Global monitoring service instance
monitoring_service = NetflixMonitoringService()
