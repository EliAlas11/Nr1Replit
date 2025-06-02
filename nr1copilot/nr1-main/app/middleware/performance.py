
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
        """Get comprehensive Netflix-level performance summary"""
        if not self.response_times:
            return {"status": "no_data"}
            
        durations = [rt["duration"] for rt in self.response_times]
        
        # Calculate advanced percentiles
        sorted_durations = sorted(durations)
        percentiles = {}
        for p in [50, 75, 90, 95, 99, 99.9]:
            index = int(p/100 * len(sorted_durations))
            percentiles[f"p{p}"] = sorted_durations[min(index, len(sorted_durations)-1)]
        
        # Calculate throughput
        recent_requests = [rt for rt in self.response_times 
                          if (datetime.utcnow() - rt["timestamp"]).seconds < 60]
        throughput_per_minute = len(recent_requests)
        
        # Calculate error rate
        error_requests = [rt for rt in self.response_times if rt.get("status") == "error"]
        error_rate = len(error_requests) / len(self.response_times) if self.response_times else 0
        
        return {
            "response_times": {
                "count": len(durations),
                "avg": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                **percentiles
            },
            "system_metrics": {
                "current_cpu": self.cpu_usage[-1]["value"] if self.cpu_usage else 0,
                "current_memory": self.memory_usage[-1]["value"] if self.memory_usage else 0,
                "avg_cpu_1min": sum(m["value"] for m in list(self.cpu_usage)[-60:]) / min(60, len(self.cpu_usage)) if self.cpu_usage else 0,
                "avg_memory_1min": sum(m["value"] for m in list(self.memory_usage)[-60:]) / min(60, len(self.memory_usage)) if self.memory_usage else 0,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_io": dict(psutil.net_io_counters()._asdict()) if hasattr(psutil, 'net_io_counters') else {}
            },
            "performance_metrics": {
                "throughput_per_minute": throughput_per_minute,
                "error_rate": error_rate,
                "slow_requests": len(self.slow_requests),
                "requests_per_second": throughput_per_minute / 60
            },
            "adaptive_thresholds": {
                "slow_request_threshold": self.slow_request_threshold,
                "cpu_threshold": self.cpu_threshold,
                "memory_threshold": self.memory_threshold
            },
            "health_score": self._calculate_health_score(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0
        
        # Deduct for high CPU usage
        if self.cpu_usage:
            recent_cpu = [m["value"] for m in list(self.cpu_usage)[-10:]]
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            if avg_cpu > self.cpu_threshold:
                score -= min(30, (avg_cpu - self.cpu_threshold) * 2)
        
        # Deduct for high memory usage
        if self.memory_usage:
            recent_memory = [m["value"] for m in list(self.memory_usage)[-10:]]
            avg_memory = sum(recent_memory) / len(recent_memory)
            if avg_memory > self.memory_threshold:
                score -= min(25, (avg_memory - self.memory_threshold) * 2)
        
        # Deduct for slow requests
        if self.response_times:
            slow_ratio = len(self.slow_requests) / len(self.response_times)
            score -= min(25, slow_ratio * 100)
        
        # Deduct for errors
        if self.response_times:
            error_requests = [rt for rt in self.response_times if rt.get("status") == "error"]
            error_ratio = len(error_requests) / len(self.response_times)
            score -= min(20, error_ratio * 100)
        
        return max(0.0, score)
