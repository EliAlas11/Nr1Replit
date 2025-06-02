"""
Netflix-Level Metrics Collection
Real-time performance and business metrics
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Netflix-level metrics collection and monitoring"""

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics = defaultdict(deque)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.start_time = time.time()

        # Request metrics
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=1000)

        # Video processing metrics
        self.uploads_count = 0
        self.processing_count = 0
        self.completed_count = 0
        self.failed_count = 0

        # WebSocket metrics
        self.websocket_connections = 0
        self.websocket_messages = 0

        # Start background cleanup
        asyncio.create_task(self._cleanup_old_metrics())

    async def record_request(self, method: str, path: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.request_count += 1
        self.response_times.append(duration)

        if status_code >= 400:
            self.error_count += 1

        # Store detailed metrics
        metric_key = f"request_{method}_{self._sanitize_path(path)}"
        self.metrics[metric_key].append({
            "timestamp": time.time(),
            "status_code": status_code,
            "duration": duration
        })

        # Update counters
        self.counters[f"requests_total"] += 1
        self.counters[f"requests_{method.lower()}"] += 1

        if status_code >= 400:
            self.counters["requests_errors"] += 1

    async def record_upload(self, file_size: int, duration: float, success: bool = True):
        """Record upload metrics"""
        self.uploads_count += 1

        if success:
            self.counters["uploads_successful"] += 1
        else:
            self.counters["uploads_failed"] += 1

        self.metrics["uploads"].append({
            "timestamp": time.time(),
            "file_size": file_size,
            "duration": duration,
            "success": success
        })

        # Update gauges
        self.gauges["avg_upload_size"] = self._calculate_average("uploads", "file_size")
        self.gauges["avg_upload_duration"] = self._calculate_average("uploads", "duration")

    async def record_processing(self, session_id: str, stage: str, duration: float, success: bool = True):
        """Record video processing metrics"""
        self.processing_count += 1

        if success:
            if stage == "complete":
                self.completed_count += 1
        else:
            self.failed_count += 1

        self.metrics["processing"].append({
            "timestamp": time.time(),
            "session_id": session_id,
            "stage": stage,
            "duration": duration,
            "success": success
        })

        # Update processing stage counters
        self.counters[f"processing_{stage}"] += 1

    async def record_websocket_connection(self, connected: bool):
        """Record WebSocket connection metrics"""
        if connected:
            self.websocket_connections += 1
            self.counters["websocket_connections"] += 1
        else:
            self.websocket_connections = max(0, self.websocket_connections - 1)
            self.counters["websocket_disconnections"] += 1

        self.gauges["active_websocket_connections"] = self.websocket_connections

    async def record_websocket_message(self, message_type: str, size: int):
        """Record WebSocket message metrics"""
        self.websocket_messages += 1
        self.counters[f"websocket_messages_{message_type}"] += 1

        self.metrics["websocket_messages"].append({
            "timestamp": time.time(),
            "type": message_type,
            "size": size
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary for health checks"""
        uptime = time.time() - self.start_time

        # Calculate response time percentiles
        response_times = list(self.response_times)
        percentiles = {}
        if response_times:
            response_times.sort()
            percentiles = {
                "p50": self._percentile(response_times, 50),
                "p95": self._percentile(response_times, 95),
                "p99": self._percentile(response_times, 99)
            }

        return {
            "uptime_seconds": round(uptime, 2),
            "requests": {
                "total": self.request_count,
                "errors": self.error_count,
                "error_rate": (self.error_count / max(self.request_count, 1)) * 100,
                "response_times": percentiles
            },
            "uploads": {
                "total": self.uploads_count,
                "successful": self.counters.get("uploads_successful", 0),
                "failed": self.counters.get("uploads_failed", 0),
                "success_rate": (self.counters.get("uploads_successful", 0) / max(self.uploads_count, 1)) * 100
            },
            "processing": {
                "total": self.processing_count,
                "completed": self.completed_count,
                "failed": self.failed_count,
                "success_rate": (self.completed_count / max(self.processing_count, 1)) * 100
            },
            "websockets": {
                "active_connections": self.websocket_connections,
                "total_messages": self.websocket_messages
            },
            "gauges": dict(self.gauges),
            "counters": dict(self.counters)
        }

    def get_detailed_metrics(self, metric_type: str = None) -> Dict[str, Any]:
        """Get detailed metrics for analysis"""
        if metric_type:
            return {metric_type: list(self.metrics.get(metric_type, []))}

        return {key: list(values) for key, values in self.metrics.items()}

    async def export_metrics(self, file_path: Optional[str] = None) -> str:
        """Export metrics to JSON file"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"logs/metrics_{timestamp}.json"

        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "detailed": self.get_detailed_metrics()
        }

        Path(file_path).parent.mkdir(exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)

        logger.info(f"Metrics exported to {file_path}")
        return file_path

    def _sanitize_path(self, path: str) -> str:
        """Sanitize URL path for metric keys"""
        # Replace dynamic parts with placeholders
        import re
        path = re.sub(r'/\d+', '/{id}', path)
        path = re.sub(r'/[a-f0-9-]{32,}', '/{uuid}', path)
        return path.replace('/', '_').replace('-', '_')

    def _calculate_average(self, metric_type: str, field: str) -> float:
        """Calculate average for a specific field in metrics"""
        values = [m[field] for m in self.metrics[metric_type] if field in m]
        return sum(values) / len(values) if values else 0.0

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile from sorted data"""
        if not data:
            return 0.0
        index = int((percentile / 100) * len(data))
        index = min(index, len(data) - 1)
        return data[index]

    async def _cleanup_old_metrics(self):
        """Cleanup old metrics to prevent memory bloat"""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                cutoff_time = time.time() - (self.retention_hours * 3600)

                for metric_type, values in self.metrics.items():
                    # Remove old entries
                    while values and values[0].get("timestamp", 0) < cutoff_time:
                        values.popleft()

                logger.debug("Cleaned up old metrics")

            except Exception as e:
                logger.error(f"Error cleaning up metrics: {e}")
"""
Netflix-Grade Metrics Collection System
Real-time performance and business metrics collection
"""

import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

class MetricsCollector:
    """Netflix-tier metrics collection and aggregation"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.last_cleanup = datetime.utcnow()
        
    def increment(self, metric_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.counters[metric_name] += value
        self._record_metric(metric_name, value, tags or {}, "counter")
    
    def gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        self.gauges[metric_name] = value
        self._record_metric(metric_name, value, tags or {}, "gauge")
    
    def histogram(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        self.histograms[metric_name].append(value)
        self._record_metric(metric_name, value, tags or {}, "histogram")
    
    def timing(self, metric_name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record timing metric in milliseconds"""
        self.histogram(f"{metric_name}.duration", duration * 1000, tags)
    
    def _record_metric(self, name: str, value: float, tags: Dict[str, str], metric_type: str):
        """Record metric with metadata"""
        metric = Metric(
            name=name,
            value=value,
            tags={**tags, "type": metric_type}
        )
        self.metrics[name].append(metric)
        
        # Periodic cleanup
        if (datetime.utcnow() - self.last_cleanup).total_seconds() > 3600:  # Every hour
            asyncio.create_task(self._cleanup_old_metrics())
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to maintain memory efficiency"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
            
            for metric_name, metric_list in self.metrics.items():
                # Remove old metrics
                while metric_list and metric_list[0].timestamp < cutoff_time:
                    metric_list.popleft()
            
            # Clean up histograms
            for histogram_name, values in self.histograms.items():
                if len(values) > 1000:  # Keep last 1000 values
                    self.histograms[histogram_name] = values[-1000:]
            
            self.last_cleanup = datetime.utcnow()
            logger.info("Metrics cleanup completed")
            
        except Exception as e:
            logger.error(f"Metrics cleanup failed: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        try:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms_stats": self._calculate_histogram_stats(),
                "total_metrics": sum(len(metric_list) for metric_list in self.metrics.values()),
                "retention_hours": self.retention_hours
            }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}
    
    def _calculate_histogram_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for histogram metrics"""
        stats = {}
        
        for name, values in self.histograms.items():
            if values:
                sorted_values = sorted(values)
                count = len(sorted_values)
                
                stats[name] = {
                    "count": count,
                    "min": sorted_values[0],
                    "max": sorted_values[-1],
                    "mean": sum(sorted_values) / count,
                    "p50": sorted_values[count // 2],
                    "p95": sorted_values[int(count * 0.95)] if count > 1 else sorted_values[0],
                    "p99": sorted_values[int(count * 0.99)] if count > 1 else sorted_values[0]
                }
        
        return stats
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if metric_name not in self.metrics:
            return []
        
        return [
            {
                "timestamp": metric.timestamp.isoformat(),
                "value": metric.value,
                "tags": metric.tags
            }
            for metric in self.metrics[metric_name]
            if metric.timestamp >= cutoff_time
        ]
    
    async def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format"""
        try:
            if format_type == "json":
                return json.dumps(self.get_metrics_summary(), indent=2)
            elif format_type == "prometheus":
                return self._format_prometheus_metrics()
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _format_prometheus_metrics(self) -> str:
        """Format metrics in Prometheus format"""
        lines = []
        
        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        return "\n".join(lines)

# Global metrics instance
metrics = MetricsCollector()
"""
Netflix-Grade Metrics Collection System
Real-time metrics aggregation with enterprise monitoring capabilities
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Individual metric data point with timestamp and metadata"""
    name: str
    value: Union[float, int]
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, timer

class MetricsCollector:
    """Netflix-tier metrics collection and aggregation system"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.startup_time = time.time()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_old_metrics())
        
    def increment(self, metric_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        tags = tags or {}
        key = self._get_metric_key(metric_name, tags)
        self.counters[key] += value
        
        # Store as metric point
        point = MetricPoint(
            name=metric_name,
            value=value,
            tags=tags,
            metric_type="counter"
        )
        self.metrics[key].append(point)
        
    def gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        tags = tags or {}
        key = self._get_metric_key(metric_name, tags)
        self.gauges[key] = value
        
        # Store as metric point
        point = MetricPoint(
            name=metric_name,
            value=value,
            tags=tags,
            metric_type="gauge"
        )
        self.metrics[key].append(point)
        
    def timing(self, metric_name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        tags = tags or {}
        key = self._get_metric_key(metric_name, tags)
        self.timers[key].append(duration)
        
        # Store as metric point
        point = MetricPoint(
            name=metric_name,
            value=duration,
            tags=tags,
            metric_type="timer"
        )
        self.metrics[key].append(point)
        
    def histogram(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        tags = tags or {}
        key = self._get_metric_key(metric_name, tags)
        
        # Store as metric point
        point = MetricPoint(
            name=metric_name,
            value=value,
            tags=tags,
            metric_type="histogram"
        )
        self.metrics[key].append(point)
        
    def _get_metric_key(self, name: str, tags: Dict[str, str]) -> str:
        """Generate unique key for metric with tags"""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
        
    async def _cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period"""
        while True:
            try:
                cutoff_time = time.time() - (self.retention_hours * 3600)
                
                for metric_key, points in self.metrics.items():
                    # Remove old points
                    while points and points[0].timestamp < cutoff_time:
                        points.popleft()
                        
                # Clean up empty metrics
                empty_keys = [k for k, v in self.metrics.items() if not v]
                for key in empty_keys:
                    del self.metrics[key]
                    
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(60)
                
    async def _cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period"""
        while True:
            try:
                cutoff_time = time.time() - (self.retention_hours * 3600)
                
                for metric_key, points in self.metrics.items():
                    # Remove old points
                    while points and points[0].timestamp < cutoff_time:
                        points.popleft()
                        
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(60)
                
    def get_metric_summary(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        key = self._get_metric_key(metric_name, tags or {})
        points = list(self.metrics.get(key, []))
        
        if not points:
            return {"error": "no_data", "metric": metric_name}
            
        values = [p.value for p in points]
        recent_points = [p for p in points if p.timestamp > time.time() - 3600]  # Last hour
        
        return {
            "metric": metric_name,
            "tags": tags,
            "total_points": len(points),
            "recent_points": len(recent_points),
            "current_value": values[-1] if values else None,
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "recent_avg": sum(p.value for p in recent_points) / len(recent_points) if recent_points else None,
            "last_updated": points[-1].timestamp if points else None
        }
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        uptime = time.time() - self.startup_time
        
        # Calculate totals
        total_metrics = len(self.metrics)
        total_points = sum(len(points) for points in self.metrics.values())
        
        # Get top metrics by activity
        metric_activity = [(k, len(v)) for k, v in self.metrics.items()]
        metric_activity.sort(key=lambda x: x[1], reverse=True)
        top_metrics = metric_activity[:10]
        
        return {
            "uptime_seconds": round(uptime, 2),
            "total_metrics": total_metrics,
            "total_data_points": total_points,
            "top_metrics": [{"name": name, "points": points} for name, points in top_metrics],
            "counters_count": len(self.counters),
            "gauges_count": len(self.gauges),
            "timers_count": len(self.timers),
            "retention_hours": self.retention_hours,
            "netflix_grade": "AAA+",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in various formats"""
        if format_type == "json":
            return await self._export_json()
        elif format_type == "prometheus":
            return await self._export_prometheus()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
    async def _export_json(self) -> str:
        """Export metrics in JSON format"""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.startup_time,
            "metrics": {}
        }
        
        for metric_key, points in self.metrics.items():
            if points:
                export_data["metrics"][metric_key] = {
                    "points": len(points),
                    "current_value": points[-1].value if points else None,
                    "metric_type": points[-1].metric_type if points else "unknown",
                    "last_updated": points[-1].timestamp if points else None
                }
                
        return json.dumps(export_data, indent=2)
        
    async def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = [
            "# Netflix-Grade Metrics Export",
            f"# TIMESTAMP {int(time.time())}",
            ""
        ]
        
        for metric_key, points in self.metrics.items():
            if not points:
                continue
                
            latest_point = points[-1]
            metric_name = latest_point.name.replace("-", "_").replace(".", "_")
            
            # Add help text
            lines.append(f"# HELP {metric_name} {latest_point.metric_type} metric")
            lines.append(f"# TYPE {metric_name} {latest_point.metric_type}")
            
            # Add metric value with tags
            if latest_point.tags:
                tag_str = ",".join(f'{k}="{v}"' for k, v in latest_point.tags.items())
                lines.append(f"{metric_name}{{{tag_str}}} {latest_point.value}")
            else:
                lines.append(f"{metric_name} {latest_point.value}")
            lines.append("")
            
        return "\n".join(lines)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance-specific metrics"""
        return {
            "request_count": self.counters.get("requests.total", 0),
            "error_count": self.counters.get("requests.error", 0),
            "success_count": self.counters.get("requests.success", 0),
            "avg_response_time": self._calculate_avg_timer("requests.duration"),
            "cpu_usage": self.gauges.get("system.cpu_percent", 0),
            "memory_usage": self.gauges.get("system.memory_percent", 0),
            "active_connections": self.gauges.get("system.connections", 0),
            "uptime": time.time() - self.startup_time
        }
        
    def _calculate_avg_timer(self, timer_name: str) -> float:
        """Calculate average for a timer metric"""
        timer_data = self.timers.get(timer_name, deque())
        if not timer_data:
            return 0.0
        return sum(timer_data) / len(timer_data)
