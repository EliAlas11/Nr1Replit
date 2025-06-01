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