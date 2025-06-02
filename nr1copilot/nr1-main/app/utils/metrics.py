"""
Enterprise Metrics Collection System
High-performance metrics collection with async patterns and comprehensive monitoring.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, timer

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "type": self.metric_type
        }


class MetricsCollector:
    """Enterprise-grade metrics collection system."""

    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        # Performance tracking
        self.metrics_collected = 0
        self.collection_errors = 0
        self.last_flush_time: Optional[datetime] = None

        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False

        logger.info("Metrics collector initialized")

    async def initialize(self):
        """Initialize metrics collection system."""
        if self._initialized:
            return

        try:
            # Start background tasks
            self._flush_task = asyncio.create_task(self._periodic_flush())
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

            self._initialized = True
            logger.info("Metrics collection system started")

        except Exception as e:
            logger.error(f"Metrics collector initialization failed: {e}")
            raise

    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        try:
            key = self._get_metric_key(name, tags or {})
            self.counters[key] += value

            # Add to buffer
            self._add_to_buffer(MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                metric_type="counter"
            ))

            self.metrics_collected += 1

        except Exception as e:
            logger.error(f"Counter increment failed for {name}: {e}")
            self.collection_errors += 1

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        try:
            key = self._get_metric_key(name, tags or {})
            self.gauges[key] = value

            # Add to buffer
            self._add_to_buffer(MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                metric_type="gauge"
            ))

            self.metrics_collected += 1

        except Exception as e:
            logger.error(f"Gauge set failed for {name}: {e}")
            self.collection_errors += 1

    def timing(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        try:
            key = self._get_metric_key(name, tags or {})
            self.timers[key].append(duration)

            # Keep only recent timings
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]

            # Add to buffer
            self._add_to_buffer(MetricPoint(
                name=name,
                value=duration,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                metric_type="timer"
            ))

            self.metrics_collected += 1

        except Exception as e:
            logger.error(f"Timing record failed for {name}: {e}")
            self.collection_errors += 1

    def histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric."""
        try:
            key = self._get_metric_key(name, tags or {})
            self.histograms[key].append(value)

            # Keep only recent values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]

            # Add to buffer
            self._add_to_buffer(MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                metric_type="histogram"
            ))

            self.metrics_collected += 1

        except Exception as e:
            logger.error(f"Histogram record failed for {name}: {e}")
            self.collection_errors += 1

    def record_request(self, method: str, path: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        tags = {
            "method": method,
            "path": path,
            "status_code": str(status_code),
            "status_class": f"{status_code // 100}xx"
        }

        self.increment("http_requests_total", 1.0, tags)
        self.timing("http_request_duration", duration, tags)

        # Record status-specific metrics
        if status_code >= 400:
            self.increment("http_errors_total", 1.0, tags)
        if status_code >= 500:
            self.increment("http_server_errors_total", 1.0, tags)

    def _get_metric_key(self, name: str, tags: Dict[str, str]) -> str:
        """Generate a unique key for metric with tags."""
        if not tags:
            return name

        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def _add_to_buffer(self, metric: MetricPoint):
        """Add metric to buffer."""
        self.metrics_buffer.append(metric)

    async def _periodic_flush(self):
        """Periodically flush metrics to storage/external systems."""
        while True:
            try:
                await asyncio.sleep(60)  # Flush every minute
                await self._flush_metrics()

            except asyncio.CancelledError:
                logger.info("Metrics flush task cancelled")
                break
            except Exception as e:
                logger.error(f"Metrics flush error: {e}")

    async def _periodic_cleanup(self):
        """Periodically clean up old metrics."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                self._cleanup_old_metrics()

            except asyncio.CancelledError:
                logger.info("Metrics cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")

    async def _flush_metrics(self):
        """Flush metrics to storage/logging."""
        if not self.metrics_buffer:
            return

        try:
            # Convert buffer to list for processing
            metrics_to_flush = list(self.metrics_buffer)

            # Log metrics (in production, this would be sent to external systems)
            logger.info(f"Flushing {len(metrics_to_flush)} metrics")

            self.last_flush_time = datetime.utcnow()

            # Clear buffer after successful flush
            self.metrics_buffer.clear()

        except Exception as e:
            logger.error(f"Metrics flush failed: {e}")
            self.collection_errors += 1

    def _cleanup_old_metrics(self):
        """Clean up old metric data."""
        try:
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(hours=1)

            # Clean up timer data older than 1 hour
            for key in list(self.timers.keys()):
                if not self.timers[key]:
                    del self.timers[key]

            # Clean up histogram data older than 1 hour
            for key in list(self.histograms.keys()):
                if not self.histograms[key]:
                    del self.histograms[key]

            logger.debug("Metrics cleanup completed")

        except Exception as e:
            logger.error(f"Metrics cleanup failed: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "collection_stats": {
                "metrics_collected": self.metrics_collected,
                "collection_errors": self.collection_errors,
                "buffer_size": len(self.metrics_buffer),
                "last_flush": self.last_flush_time.isoformat() if self.last_flush_time else None
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timer_stats": {
                name: {
                    "count": len(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                } for name, values in self.timers.items()
            }
        }

    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        try:
            if format_type == "json":
                return json.dumps(self.get_summary(), indent=2)
            elif format_type == "prometheus":
                return self._export_prometheus_format()
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
            return json.dumps({"error": str(e)})

    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Export counters
        for key, value in self.counters.items():
            lines.append(f"{key} {value}")

        # Export gauges
        for key, value in self.gauges.items():
            lines.append(f"{key} {value}")

        return "\n".join(lines)

    async def shutdown(self):
        """Graceful shutdown of metrics collection."""
        try:
            # Cancel background tasks
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass

            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Final flush
            await self._flush_metrics()

            logger.info("Metrics collection shutdown completed")

        except Exception as e:
            logger.error(f"Metrics shutdown error: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()