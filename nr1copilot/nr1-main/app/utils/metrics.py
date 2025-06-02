
"""
Netflix-Grade Enterprise Metrics Collector v10.0
Comprehensive performance monitoring and analytics with real-time insights
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import weakref

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot"""
    timestamp: float
    cpu_percent: float
    memory_usage_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    response_time_avg_ms: float
    request_count: int
    error_count: int


class MetricsCollector:
    """Netflix-grade metrics collection and analysis"""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, Deque[MetricPoint]] = defaultdict(lambda: deque(maxlen=max_points))
        self.snapshots: Deque[PerformanceSnapshot] = deque(maxlen=1000)
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Performance tracking
        self.start_time = time.time()
        self.last_collection_time = 0
        self.collection_interval = 60  # seconds
        self._collection_task: Optional[asyncio.Task] = None
        self._initialized = False
        
        # Request tracking
        self.request_metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "response_times": deque(maxlen=1000),
            "status_codes": defaultdict(int)
        }
        
        logger.info("📊 Metrics Collector v10.0 initialized")
    
    async def start(self) -> None:
        """Start metrics collection"""
        if self._initialized:
            return
        
        try:
            # Start background collection
            self._collection_task = asyncio.create_task(self._background_collection())
            self._initialized = True
            
            logger.info("✅ Metrics collection started")
            
        except Exception as e:
            logger.error(f"Metrics collection startup failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown metrics collection"""
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Metrics collection shutdown complete")
    
    async def _background_collection(self) -> None:
        """Background metrics collection task"""
        while True:
            try:
                await asyncio.sleep(self.collection_interval)
                await self._collect_system_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background metrics collection error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self) -> None:
        """Collect comprehensive system metrics"""
        try:
            # System resources
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            # Create performance snapshot
            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_usage_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                disk_io_read_mb=(disk_io.read_bytes if disk_io else 0) / (1024 * 1024),
                disk_io_write_mb=(disk_io.write_bytes if disk_io else 0) / (1024 * 1024),
                network_bytes_sent=network_io.bytes_sent if network_io else 0,
                network_bytes_recv=network_io.bytes_recv if network_io else 0,
                active_connections=len(psutil.net_connections()),
                response_time_avg_ms=self._calculate_avg_response_time(),
                request_count=self.request_metrics["total_requests"],
                error_count=self.request_metrics["total_errors"]
            )
            
            self.snapshots.append(snapshot)
            
            # Record individual metrics
            self.record_metric("system.cpu_percent", cpu_percent)
            self.record_metric("system.memory_percent", memory.percent)
            self.record_metric("system.memory_usage_mb", memory.used / (1024 * 1024))
            
            if disk_io:
                self.record_metric("system.disk_read_mb", disk_io.read_bytes / (1024 * 1024))
                self.record_metric("system.disk_write_mb", disk_io.write_bytes / (1024 * 1024))
            
            if network_io:
                self.record_metric("system.network_sent_mb", network_io.bytes_sent / (1024 * 1024))
                self.record_metric("system.network_recv_mb", network_io.bytes_recv / (1024 * 1024))
            
            self.last_collection_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        try:
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            
            self.metrics[name].append(metric_point)
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric"""
        self.counters[name] += value
    
    def record_timer(self, name: str, duration: float) -> None:
        """Record a timing metric"""
        self.timers[name].append(duration)
        
        # Keep only recent timings
        if len(self.timers[name]) > 1000:
            self.timers[name] = self.timers[name][-1000:]
    
    def record_request(self, method: str, path: str, status_code: int, duration_ms: float) -> None:
        """Record HTTP request metrics"""
        self.request_metrics["total_requests"] += 1
        self.request_metrics["response_times"].append(duration_ms)
        self.request_metrics["status_codes"][status_code] += 1
        
        if status_code >= 400:
            self.request_metrics["total_errors"] += 1
        
        # Record as individual metrics
        self.record_metric("http.requests_total", 1, {"method": method, "status": str(status_code)})
        self.record_metric("http.response_time_ms", duration_ms, {"method": method, "path": path})
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent requests"""
        response_times = list(self.request_metrics["response_times"])
        if not response_times:
            return 0.0
        
        # Use recent 100 requests for average
        recent_times = response_times[-100:] if len(response_times) > 100 else response_times
        return sum(recent_times) / len(recent_times)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive metrics statistics"""
        return {
            "collection_info": {
                "uptime_seconds": time.time() - self.start_time,
                "last_collection": self.last_collection_time,
                "collection_interval": self.collection_interval,
                "total_metrics": len(self.metrics),
                "total_snapshots": len(self.snapshots)
            },
            "system_summary": await self._get_system_summary(),
            "request_summary": self._get_request_summary(),
            "performance_summary": self._get_performance_summary(),
            "counters": dict(self.counters),
            "timing_summary": self._get_timing_summary()
        }
    
    async def _get_system_summary(self) -> Dict[str, Any]:
        """Get system metrics summary"""
        if not self.snapshots:
            return {"status": "no_data"}
        
        latest = self.snapshots[-1]
        
        # Calculate averages from recent snapshots
        recent_snapshots = list(self.snapshots)[-10:] if len(self.snapshots) >= 10 else list(self.snapshots)
        
        avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
        avg_memory = sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots)
        
        return {
            "current": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_usage_mb": latest.memory_usage_mb,
                "active_connections": latest.active_connections
            },
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2)
            },
            "timestamp": latest.timestamp
        }
    
    def _get_request_summary(self) -> Dict[str, Any]:
        """Get HTTP request metrics summary"""
        response_times = list(self.request_metrics["response_times"])
        
        summary = {
            "total_requests": self.request_metrics["total_requests"],
            "total_errors": self.request_metrics["total_errors"],
            "error_rate_percent": 0.0,
            "status_codes": dict(self.request_metrics["status_codes"])
        }
        
        if self.request_metrics["total_requests"] > 0:
            summary["error_rate_percent"] = round(
                (self.request_metrics["total_errors"] / self.request_metrics["total_requests"]) * 100, 2
            )
        
        if response_times:
            summary["response_time_stats"] = {
                "avg_ms": round(sum(response_times) / len(response_times), 2),
                "min_ms": min(response_times),
                "max_ms": max(response_times),
                "recent_avg_ms": round(self._calculate_avg_response_time(), 2)
            }
        
        return summary
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from snapshots"""
        if len(self.snapshots) < 2:
            return {"status": "insufficient_data"}
        
        recent = list(self.snapshots)[-5:]  # Last 5 snapshots
        
        return {
            "trend_analysis": {
                "cpu_trend": self._calculate_trend([s.cpu_percent for s in recent]),
                "memory_trend": self._calculate_trend([s.memory_percent for s in recent]),
                "request_trend": self._calculate_trend([s.request_count for s in recent])
            },
            "performance_score": self._calculate_performance_score(recent[-1])
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "unknown"
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change = second_avg - first_avg
        
        if change > 5:
            return "increasing"
        elif change < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_performance_score(self, snapshot: PerformanceSnapshot) -> float:
        """Calculate performance score (0-10)"""
        score = 10.0
        
        # Deduct points for high resource usage
        if snapshot.cpu_percent > 80:
            score -= 3
        elif snapshot.cpu_percent > 60:
            score -= 1
        
        if snapshot.memory_percent > 90:
            score -= 3
        elif snapshot.memory_percent > 75:
            score -= 1
        
        # Deduct points for slow response times
        if snapshot.response_time_avg_ms > 1000:
            score -= 2
        elif snapshot.response_time_avg_ms > 500:
            score -= 1
        
        return max(0.0, score)
    
    def _get_timing_summary(self) -> Dict[str, Any]:
        """Get timing metrics summary"""
        summary = {}
        
        for name, times in self.timers.items():
            if times:
                summary[name] = {
                    "count": len(times),
                    "avg_ms": round(sum(times) / len(times), 2),
                    "min_ms": min(times),
                    "max_ms": max(times)
                }
        
        return summary
    
    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for monitoring dashboards"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "collection_stats": {
                "total_metrics": len(self.metrics),
                "total_snapshots": len(self.snapshots),
                "last_collection": self.last_collection_time
            },
            "latest_snapshot": self.snapshots[-1].__dict__ if self.snapshots else None,
            "system_health": await self._assess_system_health()
        }
    
    async def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health based on metrics"""
        if not self.snapshots:
            return {"status": "unknown", "reason": "no_data"}
        
        latest = self.snapshots[-1]
        
        health_issues = []
        
        if latest.cpu_percent > 90:
            health_issues.append("High CPU usage")
        if latest.memory_percent > 95:
            health_issues.append("High memory usage")
        if latest.response_time_avg_ms > 2000:
            health_issues.append("Slow response times")
        
        if health_issues:
            return {"status": "degraded", "issues": health_issues}
        elif latest.cpu_percent > 70 or latest.memory_percent > 80:
            return {"status": "warning", "message": "Resource usage elevated"}
        else:
            return {"status": "healthy", "message": "All systems normal"}


# Create alias for backward compatibility
NetflixEnterpriseMetricsCollector = MetricsCollector

# Global metrics collector instance
metrics_collector = MetricsCollector()
