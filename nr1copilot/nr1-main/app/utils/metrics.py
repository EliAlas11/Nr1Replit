"""
ViralClip Pro - Netflix-Level Metrics Collection
Advanced performance monitoring and analytics
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metric_type: MetricType

class MetricsCollector:
    """Netflix-level metrics collection with real-time analytics"""

    def __init__(self, max_points: int = 10000, retention_hours: int = 24):
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)

        self.retention_hours = retention_hours
        self.max_points = max_points
        self.start_time = datetime.utcnow()

        # Request tracking
        self.request_counts = defaultdict(int)
        self.request_durations = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.status_codes = defaultdict(int)

        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.queue_lengths = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=1000)

        # User activity tracking
        self.active_sessions = set()
        self.user_actions = defaultdict(int)
        self.feature_usage = defaultdict(int)

        # Video processing metrics
        self.video_metrics = {
            'uploads_total': 0,
            'processing_total': 0,
            'successful_clips': 0,
            'failed_clips': 0,
            'total_processing_time': 0,
            'avg_file_size': 0,
            'viral_scores': []
        }

        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """Periodically clean old metrics"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_metrics()
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")

    async def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)

        for metric_name, points in self.metrics.items():
            # Remove old points
            while points and points[0].timestamp < cutoff_time:
                points.popleft()

        # Clean histograms and timers
        for timer_list in self.timers.values():
            timer_list[:] = [t for t in timer_list if t['timestamp'] > cutoff_time]

        for hist_list in self.histograms.values():
            hist_list[:] = [h for h in hist_list if h['timestamp'] > cutoff_time]

    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        tags = tags or {}
        self.counters[name] += value

        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags,
            metric_type=MetricType.COUNTER
        )

        self.metrics[name].append(metric_point)

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        tags = tags or {}
        self.gauges[name] = value

        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags,
            metric_type=MetricType.GAUGE
        )

        self.metrics[name].append(metric_point)

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value"""
        tags = tags or {}
        timestamp = datetime.utcnow()

        self.histograms[name].append({
            'value': value,
            'timestamp': timestamp,
            'tags': tags
        })

        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp,
            tags=tags,
            metric_type=MetricType.HISTOGRAM
        )

        self.metrics[name].append(metric_point)

    def start_timer(self, name: str, tags: Dict[str, str] = None) -> Callable:
        """Start a timer and return a function to stop it"""
        tags = tags or {}
        start_time = time.time()

        def stop_timer():
            duration = time.time() - start_time
            self.record_timer(name, duration, tags)
            return duration

        return stop_timer

    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer duration"""
        tags = tags or {}
        timestamp = datetime.utcnow()

        self.timers[name].append({
            'duration': duration,
            'timestamp': timestamp,
            'tags': tags
        })

        metric_point = MetricPoint(
            name=name,
            value=duration,
            timestamp=timestamp,
            tags=tags,
            metric_type=MetricType.TIMER
        )

        self.metrics[name].append(metric_point)

    def time_function(self, name: str = None, tags: Dict[str, str] = None):
        """Decorator to time function execution"""
        def decorator(func):
            metric_name = name or f"function.{func.__name__}.duration"

            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    stop_timer = self.start_timer(metric_name, tags)
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        stop_timer()
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    stop_timer = self.start_timer(metric_name, tags)
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        stop_timer()
                return sync_wrapper

        return decorator

    async def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        user_id: str = None,
        ip_address: str = None
    ):
        """Record HTTP request metrics"""
        # Basic request metrics
        self.increment_counter("http.requests.total", tags={
            "method": method,
            "path": path,
            "status": str(status_code)
        })

        self.record_timer("http.request.duration", duration, tags={
            "method": method,
            "path": path
        })

        # Status code tracking
        self.status_codes[status_code] += 1

        # Error tracking
        if status_code >= 400:
            self.increment_counter("http.errors.total", tags={
                "method": method,
                "path": path,
                "status": str(status_code)
            })

        # Response time tracking
        self.request_durations[path].append(duration)
        if len(self.request_durations[path]) > 1000:
            self.request_durations[path] = self.request_durations[path][-1000:]

        # User activity tracking
        if user_id:
            self.active_sessions.add(user_id)
            self.user_actions[user_id] += 1

    def record_video_upload(self, file_size: int, duration: float, success: bool):
        """Record video upload metrics"""
        self.video_metrics['uploads_total'] += 1

        if success:
            self.increment_counter("video.uploads.successful")
            self.record_histogram("video.upload.file_size", file_size)
            self.record_timer("video.upload.duration", duration)
        else:
            self.increment_counter("video.uploads.failed")

    def record_video_processing(
        self,
        processing_time: float,
        clips_created: int,
        viral_score: float,
        success: bool
    ):
        """Record video processing metrics"""
        self.video_metrics['processing_total'] += 1
        self.video_metrics['total_processing_time'] += processing_time

        if success:
            self.video_metrics['successful_clips'] += clips_created
            self.video_metrics['viral_scores'].append(viral_score)

            self.increment_counter("video.processing.successful")
            self.record_timer("video.processing.duration", processing_time)
            self.record_histogram("video.clips_created", clips_created)
            self.record_histogram("video.viral_score", viral_score)
        else:
            self.video_metrics['failed_clips'] += 1
            self.increment_counter("video.processing.failed")

    def record_feature_usage(self, feature: str, user_id: str = None):
        """Record feature usage"""
        self.feature_usage[feature] += 1
        self.increment_counter("feature.usage", tags={"feature": feature})

        if user_id:
            self.increment_counter("user.feature_usage", tags={
                "user_id": user_id,
                "feature": feature
            })

    def record_performance_metrics(self, cpu_percent: float, memory_mb: float, queue_length: int):
        """Record system performance metrics"""
        self.set_gauge("system.cpu.percent", cpu_percent)
        self.set_gauge("system.memory.mb", memory_mb)
        self.set_gauge("system.queue.length", queue_length)

        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_mb)
        self.queue_lengths.append(queue_length)

    def get_histogram_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        values = []

        for hist_data in self.histograms[name]:
            if tags is None or all(hist_data['tags'].get(k) == v for k, v in tags.items()):
                values.append(hist_data['value'])

        if not values:
            return {}

        values.sort()
        n = len(values)

        return {
            'count': n,
            'min': values[0],
            'max': values[-1],
            'mean': sum(values) / n,
            'median': values[n // 2],
            'p95': values[int(n * 0.95)] if n > 0 else 0,
            'p99': values[int(n * 0.99)] if n > 0 else 0
        }

    def get_timer_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get timer statistics"""
        durations = []

        for timer_data in self.timers[name]:
            if tags is None or all(timer_data['tags'].get(k) == v for k, v in tags.items()):
                durations.append(timer_data['duration'])

        if not durations:
            return {}

        durations.sort()
        n = len(durations)

        return {
            'count': n,
            'total': sum(durations),
            'min': durations[0],
            'max': durations[-1],
            'mean': sum(durations) / n,
            'median': durations[n // 2],
            'p95': durations[int(n * 0.95)] if n > 0 else 0,
            'p99': durations[int(n * 0.99)] if n > 0 else 0
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        now = datetime.utcnow()
        uptime = (now - self.start_time).total_seconds()

        # Calculate averages and rates
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        avg_queue = sum(self.queue_lengths) / len(self.queue_lengths) if self.queue_lengths else 0

        # Request metrics
        total_requests = sum(self.request_counts.values())
        request_rate = total_requests / uptime if uptime > 0 else 0

        # Error rate
        total_errors = sum(count for status, count in self.status_codes.items() if status >= 400)
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

        # Video processing metrics
        avg_viral_score = (
            sum(self.video_metrics['viral_scores']) / len(self.video_metrics['viral_scores'])
            if self.video_metrics['viral_scores'] else 0
        )

        success_rate = (
            self.video_metrics['successful_clips'] / 
            (self.video_metrics['successful_clips'] + self.video_metrics['failed_clips']) * 100
            if (self.video_metrics['successful_clips'] + self.video_metrics['failed_clips']) > 0 else 0
        )

        return {
            'system': {
                'uptime_seconds': uptime,
                'start_time': self.start_time.isoformat(),
                'current_time': now.isoformat(),
                'cpu_usage_percent': avg_cpu,
                'memory_usage_mb': avg_memory,
                'queue_length': avg_queue
            },
            'requests': {
                'total': total_requests,
                'rate_per_second': request_rate,
                'error_rate_percent': error_rate,
                'status_codes': dict(self.status_codes),
                'response_times': {
                    endpoint: {
                        'count': len(durations),
                        'avg': sum(durations) / len(durations) if durations else 0,
                        'min': min(durations) if durations else 0,
                        'max': max(durations) if durations else 0
                    }
                    for endpoint, durations in self.request_durations.items()
                }
            },
            'users': {
                'active_sessions': len(self.active_sessions),
                'total_actions': sum(self.user_actions.values()),
                'feature_usage': dict(self.feature_usage)
            },
            'video_processing': {
                **self.video_metrics,
                'average_viral_score': avg_viral_score,
                'success_rate_percent': success_rate,
                'average_processing_time': (
                    self.video_metrics['total_processing_time'] / 
                    self.video_metrics['processing_total']
                    if self.video_metrics['processing_total'] > 0 else 0
                )
            },
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histogram_stats': {
                name: self.get_histogram_stats(name)
                for name in self.histograms.keys()
            },
            'timer_stats': {
                name: self.get_timer_stats(name)
                for name in self.timers.keys()
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on metrics"""
        health = {
            'status': 'healthy',
            'checks': {},
            'score': 100
        }

        # Check error rate
        total_requests = sum(self.request_counts.values())
        total_errors = sum(count for status, count in self.status_codes.items() if status >= 400)
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

        if error_rate > 10:
            health['status'] = 'unhealthy'
            health['score'] -= 30
        elif error_rate > 5:
            health['status'] = 'degraded'
            health['score'] -= 15

        health['checks']['error_rate'] = {
            'status': 'pass' if error_rate <= 5 else 'fail',
            'value': error_rate,
            'threshold': 5
        }

        # Check response time
        recent_durations = []
        for durations in self.request_durations.values():
            recent_durations.extend(durations[-10:])  # Last 10 requests per endpoint

        avg_response_time = sum(recent_durations) / len(recent_durations) if recent_durations else 0

        if avg_response_time > 5:
            health['status'] = 'unhealthy'
            health['score'] -= 20
        elif avg_response_time > 2:
            health['status'] = 'degraded'
            health['score'] -= 10

        health['checks']['response_time'] = {
            'status': 'pass' if avg_response_time <= 2 else 'fail',
            'value': avg_response_time,
            'threshold': 2
        }

        # Check memory usage
        current_memory = self.memory_usage[-1] if self.memory_usage else 0
        if current_memory > 8000:  # 8GB
            health['status'] = 'unhealthy'
            health['score'] -= 25
        elif current_memory > 4000:  # 4GB
            health['score'] -= 10

        health['checks']['memory_usage'] = {
            'status': 'pass' if current_memory <= 4000 else 'fail',
            'value': current_memory,
            'threshold': 4000
        }

        return health

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        # Histograms
        for name, values in self.histograms.items():
            if values:
                stats = self.get_histogram_stats(name)
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {stats.get('count', 0)}")
                lines.append(f"{name}_sum {sum(v['value'] for v in values)}")
                lines.append(f"{name}_bucket{{le=\"0.1\"}} {sum(1 for v in values if v['value'] <= 0.1)}")
                lines.append(f"{name}_bucket{{le=\"0.5\"}} {sum(1 for v in values if v['value'] <= 0.5)}")
                lines.append(f"{name}_bucket{{le=\"1.0\"}} {sum(1 for v in values if v['value'] <= 1.0)}")
                lines.append(f"{name}_bucket{{le=\"+Inf\"}} {len(values)}")

        return '\n'.join(lines)

    def reset_metrics(self):
        """Reset all metrics (for testing)"""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()
        self.request_counts.clear()
        self.request_durations.clear()
        self.error_counts.clear()
        self.status_codes.clear()
        self.active_sessions.clear()
        self.user_actions.clear()
        self.feature_usage.clear()
        self.video_metrics = {
            'uploads_total': 0,
            'processing_total': 0,
            'successful_clips': 0,
            'failed_clips': 0,
            'total_processing_time': 0,
            'avg_file_size': 0,
            'viral_scores': []
        }