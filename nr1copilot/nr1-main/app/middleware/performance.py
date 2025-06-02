
"""
Enterprise Performance Middleware
Advanced performance monitoring, optimization, and real-time metrics collection.
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Callable
from collections import deque, defaultdict
from datetime import datetime, timedelta

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Performance monitoring and statistics collection."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_times: deque = deque(maxlen=max_history)
        self.slow_requests: deque = deque(maxlen=100)
        self.endpoint_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "error_count": 0
        })
        
        # Performance thresholds
        self.slow_request_threshold = 2.0  # seconds
        self.critical_threshold = 5.0  # seconds
        
        # Real-time metrics
        self.active_requests = 0
        self.total_requests = 0
        self.total_errors = 0
        
        # Performance alerts
        self.performance_alerts: deque = deque(maxlen=50)
    
    def record_request(
        self, 
        method: str, 
        path: str, 
        duration: float, 
        status_code: int,
        response_size: int = 0
    ):
        """Record request performance metrics."""
        self.total_requests += 1
        self.request_times.append(duration)
        
        # Record endpoint-specific stats
        endpoint_key = f"{method} {path}"
        stats = self.endpoint_stats[endpoint_key]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)
        
        if status_code >= 400:
            stats["error_count"] += 1
            self.total_errors += 1
        
        # Check for slow requests
        if duration > self.slow_request_threshold:
            self.slow_requests.append({
                "method": method,
                "path": path,
                "duration": duration,
                "status_code": status_code,
                "timestamp": datetime.utcnow(),
                "response_size": response_size
            })
        
        # Performance alerts
        if duration > self.critical_threshold:
            self._create_performance_alert("critical_slow_request", {
                "endpoint": endpoint_key,
                "duration": duration,
                "threshold": self.critical_threshold
            })
    
    def _create_performance_alert(self, alert_type: str, data: Dict):
        """Create performance alert."""
        alert = {
            "type": alert_type,
            "timestamp": datetime.utcnow(),
            "data": data
        }
        self.performance_alerts.append(alert)
        
        # Log critical performance issues
        if alert_type == "critical_slow_request":
            logger.warning(
                f"Critical slow request: {data['endpoint']} took {data['duration']:.3f}s"
            )
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if not self.request_times:
            return {"message": "No performance data available"}
        
        # Calculate statistics
        avg_response_time = sum(self.request_times) / len(self.request_times)
        sorted_times = sorted(self.request_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        # Top slow endpoints
        slow_endpoints = sorted(
            [(k, v) for k, v in self.endpoint_stats.items()],
            key=lambda x: x[1]["total_time"] / x[1]["count"],
            reverse=True
        )[:5]
        
        return {
            "overview": {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": (self.total_errors / max(self.total_requests, 1)) * 100,
                "active_requests": self.active_requests,
                "avg_response_time": round(avg_response_time, 3),
                "performance_grade": self._calculate_performance_grade(avg_response_time, p95)
            },
            "response_times": {
                "average": round(avg_response_time, 3),
                "p50": round(p50, 3),
                "p95": round(p95, 3),
                "p99": round(p99, 3),
                "min": round(min(self.request_times), 3),
                "max": round(max(self.request_times), 3)
            },
            "slow_requests": {
                "count": len(self.slow_requests),
                "threshold": self.slow_request_threshold,
                "recent": list(self.slow_requests)[-5:]  # Last 5 slow requests
            },
            "top_slow_endpoints": [
                {
                    "endpoint": endpoint,
                    "avg_time": round(stats["total_time"] / stats["count"], 3),
                    "count": stats["count"],
                    "error_rate": round((stats["error_count"] / stats["count"]) * 100, 2)
                }
                for endpoint, stats in slow_endpoints
            ],
            "alerts": list(self.performance_alerts)[-10:]  # Last 10 alerts
        }
    
    def _calculate_performance_grade(self, avg_time: float, p95_time: float) -> str:
        """Calculate perfect performance grade based on response times."""
        if avg_time < 0.01 and p95_time < 0.05:
            return "PERFECT 10/10 ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐"
        elif avg_time < 0.05 and p95_time < 0.1:
            return "A+ QUANTUM-FAST"
        elif avg_time < 0.1 and p95_time < 0.2:
            return "A+ NETFLIX-GRADE"
        elif avg_time < 0.2 and p95_time < 0.5:
            return "A ENTERPRISE"
        elif avg_time < 0.5 and p95_time < 1.0:
            return "B+ PRODUCTION"
        elif avg_time < 1.0 and p95_time < 2.0:
            return "B STANDARD"
        else:
            return "C NEEDS OPTIMIZATION"


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Enterprise-grade performance monitoring middleware."""
    
    def __init__(
        self, 
        app, 
        max_request_time: float = 30.0,
        enable_detailed_logging: bool = True
    ):
        super().__init__(app)
        self.max_request_time = max_request_time
        self.enable_detailed_logging = enable_detailed_logging
        self.monitor = PerformanceMonitor()
        
        # Circuit breaker for performance issues
        self.circuit_breaker = {
            "failure_count": 0,
            "last_failure_time": 0,
            "state": "closed",  # closed, open, half-open
            "failure_threshold": 5,
            "recovery_timeout": 60
        }
        
        logger.info(f"Performance middleware initialized (max_time={max_request_time}s)")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance monitoring."""
        start_time = time.time()
        self.monitor.active_requests += 1
        
        try:
            # Check circuit breaker
            if self._is_circuit_open():
                return JSONResponse(
                    {"error": "Service temporarily unavailable due to performance issues"},
                    status_code=503
                )
            
            # Create timeout task
            timeout_task = asyncio.create_task(self._request_timeout(self.max_request_time))
            request_task = asyncio.create_task(call_next(request))
            
            # Wait for either completion or timeout
            done, pending = await asyncio.wait(
                [request_task, timeout_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Check if request completed or timed out
            if timeout_task in done:
                # Request timed out
                self._record_circuit_breaker_failure()
                logger.error(f"Request timeout: {request.method} {request.url.path}")
                
                return JSONResponse(
                    {"error": "Request timeout"},
                    status_code=504
                )
            
            # Request completed successfully
            response = request_task.result()
            processing_time = time.time() - start_time
            
            # Record performance metrics
            response_size = self._get_response_size(response)
            self.monitor.record_request(
                method=request.method,
                path=str(request.url.path),
                duration=processing_time,
                status_code=response.status_code,
                response_size=response_size
            )
            
            # Add performance headers
            self._add_performance_headers(response, processing_time)
            
            # Reset circuit breaker on success
            self._reset_circuit_breaker()
            
            # Log slow requests
            if self.enable_detailed_logging and processing_time > 1.0:
                logger.warning(
                    f"Slow request: {request.method} {request.url.path} - {processing_time:.3f}s"
                )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._record_circuit_breaker_failure()
            
            logger.error(f"Request processing error: {e}")
            
            # Record error metrics
            self.monitor.record_request(
                method=request.method,
                path=str(request.url.path),
                duration=processing_time,
                status_code=500
            )
            
            raise
            
        finally:
            self.monitor.active_requests -= 1
    
    async def _request_timeout(self, timeout: float):
        """Request timeout task."""
        await asyncio.sleep(timeout)
        return "timeout"
    
    def _get_response_size(self, response: Response) -> int:
        """Estimate response size."""
        try:
            if hasattr(response, 'body'):
                if isinstance(response.body, bytes):
                    return len(response.body)
                elif isinstance(response.body, str):
                    return len(response.body.encode('utf-8'))
            return 0
        except Exception:
            return 0
    
    def _add_performance_headers(self, response: Response, processing_time: float):
        """Add performance-related headers to response."""
        response.headers.update({
            "X-Response-Time": f"{processing_time:.6f}s",
            "X-Performance-Grade": self.monitor._calculate_performance_grade(processing_time, processing_time),
            "X-Server-Timing": f"total;dur={processing_time * 1000:.2f}"
        })
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_breaker["state"] == "open":
            # Check if recovery timeout has passed
            if time.time() - self.circuit_breaker["last_failure_time"] > self.circuit_breaker["recovery_timeout"]:
                self.circuit_breaker["state"] = "half-open"
                logger.info("Circuit breaker moved to half-open state")
                return False
            return True
        return False
    
    def _record_circuit_breaker_failure(self):
        """Record circuit breaker failure."""
        self.circuit_breaker["failure_count"] += 1
        self.circuit_breaker["last_failure_time"] = time.time()
        
        if self.circuit_breaker["failure_count"] >= self.circuit_breaker["failure_threshold"]:
            self.circuit_breaker["state"] = "open"
            logger.warning("Circuit breaker opened due to performance issues")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker on successful requests."""
        if self.circuit_breaker["state"] == "half-open":
            self.circuit_breaker["state"] = "closed"
            self.circuit_breaker["failure_count"] = 0
            logger.info("Circuit breaker closed - performance recovered")
        elif self.circuit_breaker["state"] == "closed":
            # Gradually reduce failure count on successful requests
            self.circuit_breaker["failure_count"] = max(0, self.circuit_breaker["failure_count"] - 1)
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        return {
            **self.monitor.get_performance_summary(),
            "circuit_breaker": {
                "state": self.circuit_breaker["state"],
                "failure_count": self.circuit_breaker["failure_count"],
                "last_failure": datetime.fromtimestamp(
                    self.circuit_breaker["last_failure_time"]
                ).isoformat() if self.circuit_breaker["last_failure_time"] else None
            },
            "middleware_config": {
                "max_request_time": self.max_request_time,
                "detailed_logging": self.enable_detailed_logging
            }
        }
