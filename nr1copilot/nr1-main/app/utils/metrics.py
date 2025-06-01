"""
Metrics Collection and Monitoring
Netflix-level application monitoring and analytics
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import defaultdict, deque
import asyncio

from ..config import get_settings
from ..logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


class MetricsCollector:
    """Comprehensive metrics collection system"""

    def __init__(self):
        self.request_metrics = defaultdict(list)
        self.upload_metrics = defaultdict(list)
        self.processing_metrics = defaultdict(list)
        self.error_metrics = defaultdict(int)
        self.performance_metrics = defaultdict(deque)
        self.system_metrics = {}

        # Metric retention settings
        self.max_metric_age = timedelta(hours=24)
        self.max_metric_count = 10000

        # Start cleanup task
        self._cleanup_task = None

    async def initialize(self):
        """Initialize metrics collector"""
        logger.info("📊 Initializing metrics collector")

        # Start periodic cleanup
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def close(self):
        """Close metrics collector"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("📊 Metrics collector closed")

    async def record_request(
        self, 
        method: str, 
        path: str, 
        status_code: int, 
        duration: float
    ):
        """Record HTTP request metrics"""
        timestamp = datetime.now()

        metric = {
            "timestamp": timestamp,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            "success": 200 <= status_code < 400
        }

        self.request_metrics["all"].append(metric)
        self.request_metrics[f"{method}:{path}"].append(metric)

        # Update performance metrics
        self.performance_metrics["request_duration"].append(duration)
        self.performance_metrics["request_count"].append(timestamp)

        # Track errors
        if status_code >= 400:
            self.error_metrics[f"http_{status_code}"] += 1

        await self._cleanup_old_metrics()

    async def record_upload(
        self,
        file_size: int,
        processing_time: float,
        file_type: str,
        success: bool
    ):
        """Record upload metrics"""
        timestamp = datetime.now()

        metric = {
            "timestamp": timestamp,
            "file_size": file_size,
            "processing_time": processing_time,
            "file_type": file_type,
            "success": success,
            "throughput": file_size / processing_time if processing_time > 0 else 0
        }

        self.upload_metrics["all"].append(metric)
        self.upload_metrics[file_type].append(metric)

        # Update performance metrics
        self.performance_metrics["upload_size"].append(file_size)
        self.performance_metrics["upload_time"].append(processing_time)

        if not success:
            self.error_metrics["upload_failed"] += 1

    async def record_processing(
        self,
        task_id: str,
        clip_count: int,
        total_duration: float,
        success: bool
    ):
        """Record processing metrics"""
        timestamp = datetime.now()

        metric = {
            "timestamp": timestamp,
            "task_id": task_id,
            "clip_count": clipcount,
            "total_duration": total_duration,
            "success": success,
            "clips_per_second": clip_count / total_duration if total_duration > 0 else 0
        }

        self.processing_metrics["all"].append(metric)

        # Update performance metrics
        self.performance_metrics["processing_time"].append(total_duration)
        self.performance_metrics["clips_processed"].append(clip_count)

        if not success:
            self.error_metrics["processing_failed"] += 1

    async def record_error(self, error_type: str, error_message: str):
        """Record error metrics"""
        self.error_metrics[f"error_{error_type}"] += 1

        # Log significant errors
        if self.error_metrics[f"error_{error_type}"] % 10 == 0:
            logger.warning(f"⚠️ Error type '{error_type}' occurred 10 times: {error_message}")

    def get_request_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get request statistics"""
        if time_window is None:
            time_window = timedelta(hours=1)

        cutoff_time = datetime.now() - time_window
        recent_requests = [
            req for req in self.request_metrics["all"]
            if req["timestamp"] > cutoff_time
        ]

        if not recent_requests:
            return {"total_requests": 0}

        total_requests = len(recent_requests)
        successful_requests = sum(1 for req in recent_requests if req["success"])
        error_requests = total_requests - successful_requests

        durations = [req["duration"] for req in recent_requests]

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_requests": error_requests,
            "success_rate": (successful_requests / total_requests) * 100,
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "requests_per_minute": total_requests / (time_window.total_seconds() / 60)
        }

    def get_upload_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get upload statistics"""
        if time_window is None:
            time_window = timedelta(hours=1)

        cutoff_time = datetime.now() - time_window
        recent_uploads = [
            upload for upload in self.upload_metrics["all"]
            if upload["timestamp"] > cutoff_time
        ]

        if not recent_uploads:
            return {"total_uploads": 0}

        total_uploads = len(recent_uploads)
        successful_uploads = sum(1 for upload in recent_uploads if upload["success"])

        file_sizes = [upload["file_size"] for upload in recent_uploads]
        processing_times = [upload["processing_time"] for upload in recent_uploads]
        throughputs = [upload["throughput"] for upload in recent_uploads if upload["throughput"] > 0]

        return {
            "total_uploads": total_uploads,
            "successful_uploads": successful_uploads,
            "success_rate": (successful_uploads / total_uploads) * 100,
            "avg_file_size": sum(file_sizes) / len(file_sizes),
            "total_data_processed": sum(file_sizes),
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
            "uploads_per_hour": total_uploads / (time_window.total_seconds() / 3600)
        }

    def get_processing_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get processing statistics"""
        if time_window is None:
            time_window = timedelta(hours=1)

        cutoff_time = datetime.now() - time_window
        recent_processing = [
            proc for proc in self.processing_metrics["all"]
            if proc["timestamp"] > cutoff_time
        ]

        if not recent_processing:
            return {"total_processing_jobs": 0}

        total_jobs = len(recent_processing)
        successful_jobs = sum(1 for proc in recent_processing if proc["success"])

        durations = [proc["total_duration"] for proc in recent_processing]
        clip_counts = [proc["clip_count"] for proc in recent_processing]

        return {
            "total_processing_jobs": total_jobs,
            "successful_jobs": successful_jobs,
            "success_rate": (successful_jobs / total_jobs) * 100,
            "total_clips_created": sum(clip_counts),
            "avg_clips_per_job": sum(clip_counts) / len(clip_counts),
            "avg_processing_time": sum(durations) / len(durations),
            "total_processing_time": sum(durations)
        }

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = sum(self.error_metrics.values())

        # Sort errors by frequency
        sorted_errors = sorted(
            self.error_metrics.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "total_errors": total_errors,
            "error_types": len(self.error_metrics),
            "top_errors": sorted_errors[:10],
            "error_breakdown": dict(self.error_metrics)
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        request_stats = self.get_request_stats()
        upload_stats = self.get_upload_stats()
        processing_stats = self.get_processing_stats()
        error_stats = self.get_error_stats()

        return {
            "timestamp": datetime.now().isoformat(),
            "requests": request_stats,
            "uploads": upload_stats,
            "processing": processing_stats,
            "errors": error_stats,
            "system": {
                "uptime": time.time() - self._start_time if hasattr(self, '_start_time') else 0,
                "metrics_collected": (
                    len(self.request_metrics["all"]) +
                    len(self.upload_metrics["all"]) +
                    len(self.processing_metrics["all"])
                )
            }
        }

    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues"""
        cutoff_time = datetime.now() - self.max_metric_age

        # Clean request metrics
        for key in list(self.request_metrics.keys()):
            self.request_metrics[key] = [
                metric for metric in self.request_metrics[key]
                if metric["timestamp"] > cutoff_time
            ][-self.max_metric_count:]

        # Clean upload metrics
        for key in list(self.upload_metrics.keys()):
            self.upload_metrics[key] = [
                metric for metric in self.upload_metrics[key]
                if metric["timestamp"] > cutoff_time
            ][-self.max_metric_count:]

        # Clean processing metrics
        for key in list(self.processing_metrics.keys()):
            self.processing_metrics[key] = [
                metric for metric in self.processing_metrics[key]
                if metric["timestamp"] > cutoff_time
            ][-self.max_metric_count:]

        # Clean performance metrics
        for key in self.performance_metrics:
            while len(self.performance_metrics[key]) > self.max_metric_count:
                self.performance_metrics[key].popleft()

    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        self._start_time = time.time()

        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                await self._cleanup_old_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in metrics cleanup: {e}")