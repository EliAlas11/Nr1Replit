
"""
Netflix-Grade Performance Middleware v12.0
Advanced performance monitoring and optimization
"""

import asyncio
import logging
import time
import gc
from typing import Callable, Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import weakref

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import psutil

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Thread-safe performance metrics collector"""

    def __init__(self, max_samples: int = 10000):
        self._lock = threading.RLock()
        self.max_samples = max_samples
        
        # Request metrics
        self.request_times: deque = deque(maxlen=max_samples)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.endpoint_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0
        })
        
        # System metrics
        self.memory_usage: deque = deque(maxlen=1000)
        self.cpu_usage: deque = deque(maxlen=1000)
        
        # Performance thresholds
        self.slow_request_threshold = 2.0  # seconds
        self.memory_threshold = 1024  # MB
        
        # Start time
        self.start_time = time.time()

    def record_request(self, method: str, path: str, duration: float, status_code: int) -> None:
        """Record a request with thread safety"""
        with self._lock:
            # Record basic metrics
            self.request_times.append({
                'timestamp': time.time(),
                'duration': duration,
                'method': method,
                'path': path,
                'status_code': status_code
            })
            
            self.request_counts[f"{method} {path}"] += 1
            
            if status_code >= 400:
                self.error_counts[f"{method} {path}"] += 1

            # Update endpoint metrics
            endpoint_key = f"{method} {path}"
            metrics = self.endpoint_metrics[endpoint_key]
            metrics['count'] += 1
            metrics['total_time'] += duration
            metrics['min_time'] = min(metrics['min_time'], duration)
            metrics['max_time'] = max(metrics['max_time'], duration)
            
            if status_code >= 400:
                metrics['error_count'] += 1

    def record_system_metrics(self) -> None:
        """Record system performance metrics"""
        try:
            with self._lock:
                # Memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.memory_usage.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb
                })
                
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.cpu_usage.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent
                })
                
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with thread safety"""
        with self._lock:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Calculate request statistics
            recent_requests = [
                req for req in self.request_times 
                if current_time - req['timestamp'] <= 300  # Last 5 minutes
            ]
            
            total_requests = len(self.request_times)
            recent_request_count = len(recent_requests)
            
            # Calculate average response time
            if recent_requests:
                avg_response_time = sum(req['duration'] for req in recent_requests) / len(recent_requests)
                max_response_time = max(req['duration'] for req in recent_requests)
                min_response_time = min(req['duration'] for req in recent_requests)
            else:
                avg_response_time = max_response_time = min_response_time = 0.0

            # Calculate requests per second
            requests_per_second = recent_request_count / 300 if recent_request_count > 0 else 0.0

            # Get memory statistics
            if self.memory_usage:
                current_memory = self.memory_usage[-1]['memory_mb']
                avg_memory = sum(m['memory_mb'] for m in self.memory_usage) / len(self.memory_usage)
            else:
                current_memory = avg_memory = 0.0

            # Get CPU statistics
            if self.cpu_usage:
                current_cpu = self.cpu_usage[-1]['cpu_percent']
                avg_cpu = sum(c['cpu_percent'] for c in self.cpu_usage) / len(self.cpu_usage)
            else:
                current_cpu = avg_cpu = 0.0

            # Calculate error rate
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0

            # Slow requests
            slow_requests = [
                req for req in recent_requests 
                if req['duration'] > self.slow_request_threshold
            ]

            return {
                'uptime_seconds': uptime,
                'total_requests': total_requests,
                'recent_requests_5min': recent_request_count,
                'requests_per_second': round(requests_per_second, 2),
                'response_times': {
                    'average_ms': round(avg_response_time * 1000, 2),
                    'min_ms': round(min_response_time * 1000, 2),
                    'max_ms': round(max_response_time * 1000, 2)
                },
                'memory': {
                    'current_mb': round(current_memory, 2),
                    'average_mb': round(avg_memory, 2),
                    'threshold_mb': self.memory_threshold
                },
                'cpu': {
                    'current_percent': round(current_cpu, 2),
                    'average_percent': round(avg_cpu, 2)
                },
                'errors': {
                    'total_count': total_errors,
                    'error_rate_percent': round(error_rate, 2)
                },
                'slow_requests': {
                    'count': len(slow_requests),
                    'threshold_seconds': self.slow_request_threshold
                },
                'timestamp': datetime.utcnow().isoformat()
            }

    def get_endpoint_statistics(self) -> Dict[str, Any]:
        """Get detailed endpoint performance statistics"""
        with self._lock:
            endpoint_stats = {}
            
            for endpoint, metrics in self.endpoint_metrics.items():
                if metrics['count'] > 0:
                    avg_time = metrics['total_time'] / metrics['count']
                    error_rate = (metrics['error_count'] / metrics['count']) * 100
                    
                    endpoint_stats[endpoint] = {
                        'request_count': metrics['count'],
                        'average_time_ms': round(avg_time * 1000, 2),
                        'min_time_ms': round(metrics['min_time'] * 1000, 2),
                        'max_time_ms': round(metrics['max_time'] * 1000, 2),
                        'error_count': metrics['error_count'],
                        'error_rate_percent': round(error_rate, 2)
                    }
            
            return endpoint_stats


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Netflix-grade performance monitoring middleware"""

    def __init__(self, app):
        super().__init__(app)
        self.metrics = PerformanceMetrics()
        self._optimization_enabled = True
        self._gc_threshold = 0.8
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_monitoring()

    def _start_monitoring(self):
        """Start background performance monitoring"""
        try:
            self._monitoring_task = asyncio.create_task(self._background_monitoring())
        except RuntimeError:
            # No event loop running, will start later
            pass

    async def _background_monitoring(self):
        """Background task for continuous performance monitoring"""
        while True:
            try:
                # Record system metrics
                self.metrics.record_system_metrics()
                
                # Intelligent garbage collection
                if self._optimization_enabled:
                    await self._intelligent_gc()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(60)

    async def _intelligent_gc(self):
        """Intelligent garbage collection based on memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.metrics.memory_threshold * self._gc_threshold:
                # Record pre-GC memory
                pre_gc_memory = memory_mb
                
                # Force garbage collection
                collected = gc.collect()
                
                # Record post-GC memory
                post_gc_memory = process.memory_info().rss / 1024 / 1024
                freed_mb = pre_gc_memory - post_gc_memory
                
                if freed_mb > 10:  # Only log if significant memory was freed
                    logger.info(
                        f"🧠 Intelligent GC: freed {freed_mb:.1f}MB, "
                        f"collected {collected} objects, "
                        f"memory: {post_gc_memory:.1f}MB"
                    )
                    
        except Exception as e:
            logger.error(f"Intelligent GC failed: {e}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance monitoring"""
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = str(request.url.path)
        
        # Add performance headers
        request.state.performance_start = start_time
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            self.metrics.record_request(method, path, duration, response.status_code)
            
            # Add performance headers to response
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Performance-Grade"] = "Netflix-Grade"
            
            # Log slow requests
            if duration > self.metrics.slow_request_threshold:
                logger.warning(
                    f"🐌 Slow request: {method} {path} took {duration:.3f}s"
                )
            
            return response
            
        except Exception as e:
            # Record error
            duration = time.time() - start_time
            self.metrics.record_request(method, path, duration, 500)
            
            logger.error(f"Request error: {method} {path} - {e}")
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration_ms": round(duration * 1000, 2)
                }
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "summary": self.metrics.get_performance_summary(),
            "endpoints": self.metrics.get_endpoint_statistics(),
            "optimization": {
                "enabled": self._optimization_enabled,
                "gc_threshold": self._gc_threshold,
                "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done()
            }
        }

    async def shutdown(self):
        """Shutdown performance monitoring"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🔄 Performance middleware shutdown complete")


# Global instance for metrics access
performance_middleware = None


def get_performance_metrics() -> Optional[Dict[str, Any]]:
    """Get performance metrics from global middleware instance"""
    global performance_middleware
    if performance_middleware:
        return performance_middleware.get_performance_metrics()
    return None


def initialize_performance_middleware(app):
    """Initialize global performance middleware"""
    global performance_middleware
    performance_middleware = PerformanceMiddleware(app)
    return performance_middleware
