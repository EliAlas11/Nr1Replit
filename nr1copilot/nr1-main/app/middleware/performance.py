
"""
Netflix-Level Performance Monitoring Middleware
Real-time performance tracking and optimization
"""

import asyncio
import logging
import time
import psutil
from typing import Callable, Dict, Any
from collections import deque, defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Netflix-level performance monitoring with adaptive optimization"""
    
    def __init__(self, app):
        super().__init__(app)
        
        # Performance tracking
        self.response_times = deque(maxlen=1000)
        self.endpoint_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.slow_requests = deque(maxlen=50)
        
        # System metrics
        self.cpu_usage = deque(maxlen=60)  # Last 60 seconds
        self.memory_usage = deque(maxlen=60)
        
        # Adaptive thresholds
        self.slow_request_threshold = 1.0  # seconds
        self.cpu_threshold = 80.0  # percentage
        self.memory_threshold = 85.0  # percentage
        
        # Background task for system monitoring
        self._start_system_monitoring()
        
    def _start_system_monitoring(self):
        """Start background system monitoring"""
        asyncio.create_task(self._monitor_system_metrics())
        
    async def _monitor_system_metrics(self):
        """Monitor system metrics in background"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.append({
                    "value": cpu_percent,
                    "timestamp": datetime.utcnow()
                })
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.append({
                    "value": memory.percent,
                    "timestamp": datetime.utcnow()
                })
                
                # Adaptive threshold adjustment
                self._adjust_thresholds()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(5)
                
    def _adjust_thresholds(self):
        """Dynamically adjust performance thresholds"""
        if len(self.response_times) >= 100:
            # Calculate 95th percentile
            sorted_times = sorted([rt["duration"] for rt in self.response_times])
            p95_index = int(0.95 * len(sorted_times))
            p95_time = sorted_times[p95_index]
            
            # Adjust slow request threshold to 2x 95th percentile
            self.slow_request_threshold = max(0.5, p95_time * 2)
            
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enhanced performance monitoring"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Add performance context to request
        request.state.perf_start = start_time
        request.state.perf_memory_start = start_memory
        
        try:
            # Check system health before processing
            if self._is_system_overloaded():
                return self._overload_response()
                
            response = await call_next(request)
            
            # Calculate metrics
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss
            memory_delta = end_memory - start_memory
            
            # Track performance metrics
            self._track_performance(request, duration, memory_delta)
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"
            response.headers["X-Memory-Delta"] = f"{memory_delta / 1024 / 1024:.2f}MB"
            
            # Log slow requests
            if duration > self.slow_request_threshold:
                self._log_slow_request(request, duration, memory_delta)
                
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            self._track_error_performance(request, duration)
            raise
            
    def _is_system_overloaded(self) -> bool:
        """Check if system is overloaded"""
        if len(self.cpu_usage) < 5:
            return False
            
        # Check recent CPU usage
        recent_cpu = [m["value"] for m in list(self.cpu_usage)[-5:]]
        avg_cpu = sum(recent_cpu) / len(recent_cpu)
        
        # Check recent memory usage
        recent_memory = [m["value"] for m in list(self.memory_usage)[-5:]]
        avg_memory = sum(recent_memory) / len(recent_memory)
        
        return avg_cpu > self.cpu_threshold or avg_memory > self.memory_threshold
        
    def _overload_response(self) -> Response:
        """Return overload response"""
        logger.warning("System overloaded - rejecting request")
        
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={
                "error": True,
                "message": "System temporarily overloaded",
                "retry_after": 30,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    def _track_performance(self, request: Request, duration: float, memory_delta: int):
        """Track performance metrics"""
        endpoint = f"{request.method}:{request.url.path}"
        
        # Track overall response times
        perf_data = {
            "duration": duration,
            "memory_delta": memory_delta,
            "timestamp": datetime.utcnow(),
            "endpoint": endpoint,
            "status": "success"
        }
        self.response_times.append(perf_data)
        
        # Track per-endpoint metrics
        self.endpoint_metrics[endpoint].append(perf_data)
        
    def _track_error_performance(self, request: Request, duration: float):
        """Track error performance"""
        endpoint = f"{request.method}:{request.url.path}"
        
        perf_data = {
            "duration": duration,
            "timestamp": datetime.utcnow(),
            "endpoint": endpoint,
            "status": "error"
        }
        self.response_times.append(perf_data)
        
    def _log_slow_request(self, request: Request, duration: float, memory_delta: int):
        """Log slow request details"""
        slow_request_data = {
            "path": request.url.path,
            "method": request.method,
            "duration": duration,
            "memory_delta": memory_delta,
            "timestamp": datetime.utcnow(),
            "query_params": dict(request.query_params),
            "client_ip": request.client.host
        }
        
        self.slow_requests.append(slow_request_data)
        
        logger.warning(
            f"Slow request detected: {request.method} {request.url.path} took {duration:.2f}s",
            extra=slow_request_data
        )
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.response_times:
            return {"status": "no_data"}
            
        durations = [rt["duration"] for rt in self.response_times]
        
        return {
            "response_times": {
                "count": len(durations),
                "avg": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "p50": sorted(durations)[int(0.5 * len(durations))],
                "p95": sorted(durations)[int(0.95 * len(durations))],
                "p99": sorted(durations)[int(0.99 * len(durations))]
            },
            "system_metrics": {
                "current_cpu": self.cpu_usage[-1]["value"] if self.cpu_usage else 0,
                "current_memory": self.memory_usage[-1]["value"] if self.memory_usage else 0,
                "avg_cpu_1min": sum(m["value"] for m in list(self.cpu_usage)[-60:]) / min(60, len(self.cpu_usage)) if self.cpu_usage else 0,
                "avg_memory_1min": sum(m["value"] for m in list(self.memory_usage)[-60:]) / min(60, len(self.memory_usage)) if self.memory_usage else 0
            },
            "slow_requests": len(self.slow_requests),
            "adaptive_thresholds": {
                "slow_request_threshold": self.slow_request_threshold,
                "cpu_threshold": self.cpu_threshold,
                "memory_threshold": self.memory_threshold
            },
            "timestamp": datetime.utcnow().isoformat()
        }
