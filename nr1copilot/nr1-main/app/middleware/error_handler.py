"""
Netflix-Level Error Handler Middleware
Advanced error handling with circuit breakers, retry mechanisms, and comprehensive logging
"""

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import defaultdict, deque

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import psutil

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Netflix-style circuit breaker for fault tolerance"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class ErrorMetrics:
    """Comprehensive error metrics collection"""

    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_rates = defaultdict(lambda: deque(maxlen=100))
        self.response_times = defaultdict(lambda: deque(maxlen=1000))
        self.circuit_breakers = defaultdict(lambda: CircuitBreaker())

    def record_error(self, error_type: str, endpoint: str, response_time: float):
        """Record error with context"""
        self.error_counts[f"{endpoint}:{error_type}"] += 1
        self.error_rates[endpoint].append(time.time())
        self.response_times[endpoint].append(response_time)
        self.circuit_breakers[endpoint].record_failure()

    def record_success(self, endpoint: str, response_time: float):
        """Record successful request"""
        self.response_times[endpoint].append(response_time)
        self.circuit_breakers[endpoint].record_success()

    def get_error_rate(self, endpoint: str, window_seconds: int = 300) -> float:
        """Calculate error rate for endpoint"""
        now = time.time()
        recent_errors = [
            t for t in self.error_rates[endpoint] 
            if now - t <= window_seconds
        ]
        return len(recent_errors) / max(window_seconds / 60, 1)  # errors per minute


class NetflixLevelErrorHandler(BaseHTTPMiddleware):
    """Netflix-grade error handling middleware"""

    def __init__(self, app):
        super().__init__(app)
        self.metrics = ErrorMetrics()
        self.health_degradation_threshold = 0.1  # 10% error rate
        self.critical_endpoints = {
            "/api/v7/upload/",
            "/api/v7/analytics/",
            "/api/v7/social/",
            "/api/v7/captions/"
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with comprehensive error handling"""
        start_time = time.time()
        endpoint = self._normalize_endpoint(request.url.path)

        # Circuit breaker check
        circuit_breaker = self.metrics.circuit_breakers[endpoint]
        if not circuit_breaker.can_execute():
            return await self._handle_circuit_breaker_open(request, endpoint)

        try:
            # Execute request
            response = await call_next(request)

            # Record metrics
            response_time = time.time() - start_time
            if response.status_code < 400:
                self.metrics.record_success(endpoint, response_time)
            else:
                error_type = f"HTTP_{response.status_code}"
                self.metrics.record_error(error_type, endpoint, response_time)

            # Add performance headers
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Circuit-Breaker-State"] = circuit_breaker.state

            return response

        except Exception as e:
            response_time = time.time() - start_time
            error_type = type(e).__name__

            # Record error metrics
            self.metrics.record_error(error_type, endpoint, response_time)

            # Handle different error types
            if isinstance(e, asyncio.TimeoutError):
                return await self._handle_timeout_error(request, e, response_time)
            elif isinstance(e, ConnectionError):
                return await self._handle_connection_error(request, e, response_time)
            elif isinstance(e, MemoryError):
                return await self._handle_memory_error(request, e, response_time)
            else:
                return await self._handle_generic_error(request, e, response_time)

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint for metrics collection"""
        # Group similar endpoints
        if path.startswith("/api/v7/upload/"):
            return "/api/v7/upload/*"
        elif path.startswith("/api/v7/analytics/"):
            return "/api/v7/analytics/*"
        elif path.startswith("/api/v7/social/"):
            return "/api/v7/social/*"
        elif path.startswith("/api/v7/captions/"):
            return "/api/v7/captions/*"
        else:
            return path

    async def _handle_circuit_breaker_open(self, request: Request, endpoint: str) -> JSONResponse:
        """Handle circuit breaker open state"""
        logger.warning(f"Circuit breaker OPEN for endpoint: {endpoint}")

        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "service_temporarily_unavailable",
                "message": "Service is temporarily unavailable due to high error rate",
                "retry_after": 60,
                "circuit_breaker_state": "OPEN",
                "timestamp": datetime.utcnow().isoformat(),
                "fallback_available": endpoint in self.critical_endpoints
            }
        )

    async def _handle_timeout_error(self, request: Request, error: Exception, response_time: float) -> JSONResponse:
        """Handle timeout errors with retry suggestions"""
        logger.error(f"Timeout error on {request.url.path}: {error}")

        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "request_timeout",
                "message": "Request timed out - please try again",
                "response_time": f"{response_time:.3f}s",
                "retry_suggested": True,
                "retry_delay": 2,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def _handle_connection_error(self, request: Request, error: Exception, response_time: float) -> JSONResponse:
        """Handle connection errors with fallback options"""
        logger.error(f"Connection error on {request.url.path}: {error}")

        return JSONResponse(
            status_code=502,
            content={
                "success": False,
                "error": "connection_failed",
                "message": "Service connection failed",
                "response_time": f"{response_time:.3f}s",
                "fallback_available": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def _handle_memory_error(self, request: Request, error: Exception, response_time: float) -> JSONResponse:
        """Handle memory errors with resource optimization"""
        logger.critical(f"Memory error on {request.url.path}: {error}")

        # Trigger garbage collection
        import gc
        gc.collect()

        return JSONResponse(
            status_code=507,
            content={
                "success": False,
                "error": "insufficient_memory",
                "message": "Insufficient server memory - please reduce request size",
                "response_time": f"{response_time:.3f}s",
                "memory_usage": f"{psutil.virtual_memory().percent}%",
                "suggestions": [
                    "Reduce file size",
                    "Try processing in smaller batches",
                    "Contact support for large file processing"
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def _handle_generic_error(self, request: Request, error: Exception, response_time: float) -> JSONResponse:
        """Handle generic errors with comprehensive logging"""
        error_id = f"err_{int(time.time())}_{hash(str(error)) % 1000000}"

        # Comprehensive error logging
        error_context = {
            "error_id": error_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "endpoint": request.url.path,
            "method": request.method,
            "user_agent": request.headers.get("User-Agent", ""),
            "ip_address": request.client.host if request.client else "unknown",
            "response_time": response_time,
            "timestamp": datetime.utcnow().isoformat(),
            "traceback": traceback.format_exc(),
            "system_metrics": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }

        logger.error(f"Unhandled error {error_id}: {json.dumps(error_context, indent=2)}")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "error_id": error_id,
                "response_time": f"{response_time:.3f}s",
                "support_info": {
                    "contact": "Include error ID when contacting support",
                    "retry_recommended": True,
                    "retry_delay": 5
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        now = time.time()

        # Calculate system health metrics
        overall_error_rate = sum(
            len([t for t in errors if now - t <= 300])  # 5 minute window
            for errors in self.metrics.error_rates.values()
        ) / max(len(self.metrics.error_rates), 1)

        circuit_breaker_states = {
            endpoint: cb.state 
            for endpoint, cb in self.metrics.circuit_breakers.items()
        }

        avg_response_times = {
            endpoint: sum(times) / len(times) if times else 0
            for endpoint, times in self.metrics.response_times.items()
        }

        health_status = "healthy"
        if overall_error_rate > self.health_degradation_threshold:
            health_status = "degraded"
        if any(state == "OPEN" for state in circuit_breaker_states.values()):
            health_status = "critical"

        return {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "overall_error_rate": overall_error_rate,
                "circuit_breaker_states": circuit_breaker_states,
                "average_response_times": avg_response_times,
                "total_errors": sum(self.metrics.error_counts.values()),
                "system_resources": {
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage('/').percent
                }
            },
            "recommendations": self._generate_health_recommendations(overall_error_rate, circuit_breaker_states)
        }

    def _generate_health_recommendations(self, error_rate: float, circuit_states: Dict[str, str]) -> list:
        """Generate health improvement recommendations"""
        recommendations = []

        if error_rate > self.health_degradation_threshold:
            recommendations.append("High error rate detected - investigate failing endpoints")

        open_circuits = [endpoint for endpoint, state in circuit_states.items() if state == "OPEN"]
        if open_circuits:
            recommendations.append(f"Circuit breakers open for: {', '.join(open_circuits)}")

        if psutil.cpu_percent() > 80:
            recommendations.append("High CPU usage - consider scaling or optimization")

        if psutil.virtual_memory().percent > 85:
            recommendations.append("High memory usage - investigate memory leaks")

        return recommendations
"""
Netflix-Grade Error Handling Middleware
Comprehensive error handling with monitoring and recovery
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import json

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Netflix-tier error handling with comprehensive logging and recovery"""
    
    def __init__(self, app, enable_debug: bool = False):
        super().__init__(app)
        self.enable_debug = enable_debug
        self.error_counts: Dict[str, int] = {}
        
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with comprehensive error handling"""
        request_id = str(uuid.uuid4())
        
        try:
            # Add request ID to request state
            request.state.request_id = request_id
            
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except HTTPException as http_exc:
            # Handle HTTP exceptions (expected errors)
            return await self._handle_http_exception(http_exc, request_id)
            
        except Exception as exc:
            # Handle unexpected exceptions
            return await self._handle_general_exception(exc, request, request_id)
    
    async def _handle_http_exception(self, exc: HTTPException, request_id: str) -> JSONResponse:
        """Handle HTTP exceptions with proper logging"""
        error_type = f"HTTP_{exc.status_code}"
        self._increment_error_count(error_type)
        
        logger.warning(f"HTTP Exception [{request_id}]: {exc.status_code} - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "status_code": exc.status_code,
                "message": exc.detail,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "type": "http_exception"
            },
            headers={"X-Request-ID": request_id}
        )
    
    async def _handle_general_exception(self, exc: Exception, request: Request, request_id: str) -> JSONResponse:
        """Handle general exceptions with comprehensive logging"""
        error_type = type(exc).__name__
        self._increment_error_count(error_type)
        
        # Log the full error details
        error_details = {
            "request_id": request_id,
            "error_type": error_type,
            "error_message": str(exc),
            "request_method": request.method,
            "request_url": str(request.url),
            "request_headers": dict(request.headers),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.enable_debug:
            error_details["traceback"] = traceback.format_exc()
        
        logger.error(f"Unhandled Exception [{request_id}]: {error_details}")
        
        # Write to error log file
        await self._write_error_log(error_details)
        
        # Return user-friendly error response
        response_content = {
            "error": True,
            "status_code": 500,
            "message": "Internal server error",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "server_error"
        }
        
        if self.enable_debug:
            response_content["debug"] = {
                "error_type": error_type,
                "error_message": str(exc)
            }
        
        return JSONResponse(
            status_code=500,
            content=response_content,
            headers={"X-Request-ID": request_id}
        )
    
    def _increment_error_count(self, error_type: str):
        """Increment error count for monitoring"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    async def _write_error_log(self, error_details: Dict[str, Any]):
        """Write error details to structured log file"""
        try:
            log_entry = json.dumps(error_details) + "\n"
            
            # Write to error log file
            with open("./logs/errors.jsonl", "a") as f:
                f.write(log_entry)
                
        except Exception as log_exc:
            logger.error(f"Failed to write error log: {log_exc}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": dict(self.error_counts),
            "timestamp": datetime.utcnow().isoformat()
        }
