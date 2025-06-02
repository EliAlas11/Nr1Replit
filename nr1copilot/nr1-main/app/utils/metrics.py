"""
Netflix-Level Metrics Collection
Real-time performance and business metrics with enterprise-grade reliability
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Netflix-tier metrics collection with real-time analytics"""

    def __init__(self):
        self.metrics: Dict[str, Any] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timings: deque = deque(maxlen=1000)
        self.tags: Dict[str, Dict[str, str]] = {}
        self._lock = threading.RLock()
        self._initialized = False
        self.start_time = time.time()
        
        logger.info("📊 MetricsCollector initialized (ready for async startup)")

    async def initialize(self):
        """Initialize async components"""
        if self._initialized:
            return
            
        try:
            self._initialized = True
            logger.info("✅ MetricsCollector async initialization completed")
        except Exception as e:
            logger.error(f"MetricsCollector async initialization failed: {e}")
            raise

    def increment(self, metric_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        try:
            with self._lock:
                self.counters[metric_name] += value
                if tags:
                    self.tags[metric_name] = tags
        except Exception as e:
            logger.error(f"Failed to increment metric {metric_name}: {e}")

    def gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        try:
            with self._lock:
                self.gauges[metric_name] = value
                if tags:
                    self.tags[metric_name] = tags
        except Exception as e:
            logger.error(f"Failed to set gauge {metric_name}: {e}")

    def timing(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        try:
            with self._lock:
                timing_data = {
                    "name": metric_name,
                    "value": value,
                    "timestamp": time.time(),
                    "tags": tags or {}
                }
                self.timings.append(timing_data)
        except Exception as e:
            logger.error(f"Failed to record timing {metric_name}: {e}")

    async def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format"""
        try:
            with self._lock:
                metrics_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "uptime_seconds": time.time() - self.start_time,
                    "counters": dict(self.counters),
                    "gauges": dict(self.gauges),
                    "timings": list(self.timings)[-100:],  # Last 100 timings
                    "tags": dict(self.tags)
                }
            
            if format_type == "json":
                return json.dumps(metrics_data, indent=2)
            else:
                return str(metrics_data)
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return json.dumps({"error": str(e)})

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of metrics"""
        try:
            with self._lock:
                return {
                    "total_counters": len(self.counters),
                    "total_gauges": len(self.gauges),
                    "total_timings": len(self.timings),
                    "uptime_seconds": time.time() - self.start_time,
                    "sample_counters": dict(list(self.counters.items())[:5]),
                    "sample_gauges": dict(list(self.gauges.items())[:5])
                }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}

    async def _cleanup_old_metrics(self):
        """Cleanup old metrics data"""
        try:
            with self._lock:
                # Keep only recent data
                cutoff_time = time.time() - 3600  # 1 hour
                self.timings = deque([
                    t for t in self.timings if t.get("timestamp", 0) > cutoff_time
                ], maxlen=1000)
            
            logger.debug("Cleaned up old metrics data")
            
        except Exception as e:
            logger.error(f"Failed to cleanup metrics: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


class MetricsCollector:
    """Netflix-level metrics collection and monitoring"""

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.timers: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
        self.start_time = time.time()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False

        logger.info("📊 MetricsCollector initialized (ready for async startup)")

    async def initialize(self):
        """Initialize async components when event loop is available"""
        if self._initialized:
            return

        try:
            # Start cleanup task only if we have a running event loop
            if not self._cleanup_task or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())

            self._initialized = True
            logger.info("✅ MetricsCollector async initialization completed")

        except Exception as e:
            logger.error(f"MetricsCollector async initialization failed: {e}")
            raise

    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self._lock:
            key = self._get_metric_key(name, tags)
            self.counters[key] = self.counters.get(key, 0.0) + value

            # Store as metric point
            self._record_metric(name, value, tags, "counter")

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        with self._lock:
            key = self._get_metric_key(name, tags)
            self.gauges[key] = value

            # Store as metric point
            self._record_metric(name, value, tags, "gauge")

    def timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        with self._lock:
            key = self._get_metric_key(name, tags)
            if key not in self.timers:
                self.timers[key] = []
            self.timers[key].append(duration)

            # Store as metric point
            self._record_metric(name, duration, tags, "timer")

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        with self._lock:
            key = self._get_metric_key(name, tags)
            if key not in self.histograms:
                self.histograms[key] = []
            self.histograms[key].append(value)

            # Store as metric point
            self._record_metric(name, value, tags, "histogram")

    def _get_metric_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Generate unique key for metric with tags"""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def _record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]], metric_type: str):
        """Record metric data point"""
        timestamp = time.time()
        metric_key = self._get_metric_key(name, tags)
        metric_data = {
            "name": name,
            "value": value,
            "timestamp": timestamp,
            "tags": tags,
            "metric_type": metric_type
        }

        if metric_key not in self.metrics:
            self.metrics[metric_key] = []
        self.metrics[metric_key].append(metric_data)

    async def _cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period"""
        while True:
            try:
                cutoff_time = time.time() - (self.retention_hours * 3600)

                with self._lock:
                    for metric_key, points in list(self.metrics.items()):
                        self.metrics[metric_key] = [p for p in points if p["timestamp"] >= cutoff_time]
                        if not self.metrics[metric_key]:
                            del self.metrics[metric_key]

                await asyncio.sleep(300)  # Clean every 5 minutes

            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(60)

    def get_metric_summary(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        key = self._get_metric_key(name, tags)
        with self._lock:
            points = self.metrics.get(key, [])

            if not points:
                return {"error": "no_data", "metric": name}

            values = [p["value"] for p in points]
            recent_points = [p for p in points if p["timestamp"] > time.time() - 3600]  # Last hour

            return {
                "metric": name,
                "tags": tags,
                "total_points": len(points),
                "recent_points": len(recent_points),
                "current_value": values[-1] if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "avg": sum(values) / len(values) if values else None,
                "recent_avg": sum(p["value"] for p in recent_points) / len(recent_points) if recent_points else None,
                "last_updated": points[-1]["timestamp"] if points else None
            }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            uptime = time.time() - self.start_time

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
                "timestamp": datetime.utcnow().isoformat()
            }

    async def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in various formats"""
        if format_type == "json":
            return self._export_json()
        elif format_type == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _export_json(self) -> str:
        """Export metrics in JSON format"""
        with self._lock:
            export_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "metrics": {}
            }

            for metric_key, points in self.metrics.items():
                if points:
                    export_data["metrics"][metric_key] = {
                        "points": len(points),
                        "current_value": points[-1]["value"] if points else None,
                        "metric_type": points[-1]["metric_type"] if points else "unknown",
                        "last_updated": points[-1]["timestamp"] if points else None
                    }

            return json.dumps(export_data, indent=2)

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        with self._lock:
            lines = [
                "# Netflix-Grade Metrics Export",
                f"# TIMESTAMP {int(time.time())}",
                ""
            ]

            for metric_key, points in self.metrics.items():
                if not points:
                    continue

                latest_point = points[-1]
                metric_name = latest_point["name"].replace("-", "_").replace(".", "_")

                # Add help text
                lines.append(f"# HELP {metric_name} {latest_point['metric_type']} metric")
                lines.append(f"# TYPE {metric_name} {latest_point['metric_type']}")

                # Add metric value with tags
                if latest_point["tags"]:
                    tag_str = ",".join(f'{k}="{v}"' for k, v in latest_point["tags"].items())
                    lines.append(f"{metric_name}{{{tag_str}}} {latest_point['value']}")
                else:
                    lines.append(f"{metric_name} {latest_point['value']}")
                lines.append("")

            return "\n".join(lines)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance-specific metrics"""
        with self._lock:
            return {
                "request_count": self.counters.get("requests.total", 0),
                "error_count": self.counters.get("requests.error", 0),
                "success_count": self.counters.get("requests.success", 0),
                "avg_response_time": self._calculate_avg_timer("requests.duration"),
                "cpu_usage": self.gauges.get("system.cpu_percent", 0),
                "memory_usage": self.gauges.get("system.memory_percent", 0),
                "active_connections": self.gauges.get("system.connections", 0),
                "uptime": time.time() - self.start_time
            }

    def _calculate_avg_timer(self, timer_name: str) -> float:
        """Calculate average for a timer metric"""
        with self._lock:
            timer_data = self.timers.get(timer_name, [])
            if not timer_data:
                return 0.0
            return sum(timer_data) / len(timer_data)