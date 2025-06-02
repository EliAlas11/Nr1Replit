"""
Netflix-Level Error Handler Middleware v10.0
Comprehensive error handling with graceful degradation
"""

import logging
import traceback
import time
from typing import Callable, Dict, Any
from datetime import datetime

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Netflix-grade error handling middleware"""

    def __init__(self, app):
        super().__init__(app)
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle all requests with comprehensive error recovery"""
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"

        try:
            # Add request ID for tracing
            request.state.request_id = request_id

            # Check circuit breaker
            if self._is_circuit_open(request.url.path):
                return JSONResponse(
                    content={
                        "error": "Service temporarily unavailable",
                        "request_id": request_id,
                        "retry_after": 30
                    },
                    status_code=503
                )

            # Process request
            response = await call_next(request)

            # Record success
            self._record_success(request.url.path)

            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{(time.time() - start_time) * 1000:.2f}ms"

            return response

        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return self._handle_http_exception(e, request_id, start_time)

        except Exception as e:
            # Handle unexpected errors
            return self._handle_unexpected_error(e, request, request_id, start_time)

    def _handle_http_exception(
        self, 
        exc: HTTPException, 
        request_id: str, 
        start_time: float
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions"""
        return JSONResponse(
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=exc.status_code,
            headers={
                "X-Request-ID": request_id,
                "X-Response-Time": f"{(time.time() - start_time) * 1000:.2f}ms"
            }
        )

    def _handle_unexpected_error(
        self, 
        exc: Exception, 
        request: Request, 
        request_id: str, 
        start_time: float
    ) -> JSONResponse:
        """Handle unexpected errors with fallback responses"""
        error_type = type(exc).__name__
        endpoint = request.url.path

        # Log detailed error
        logger.error(
            f"Unexpected error in {endpoint}: {error_type}: {str(exc)}",
            extra={
                "request_id": request_id,
                "endpoint": endpoint,
                "method": request.method,
                "traceback": traceback.format_exc()
            }
        )

        # Record failure for circuit breaker
        self._record_failure(endpoint)

        # Return graceful error response
        status_code = 500
        error_message = "Internal server error"

        # Specific error handling
        if "Connection" in error_type:
            status_code = 503
            error_message = "Service temporarily unavailable"
        elif "Timeout" in error_type:
            status_code = 408
            error_message = "Request timeout"
        elif "Permission" in error_type or "Forbidden" in error_type:
            status_code = 403
            error_message = "Access forbidden"

        return JSONResponse(
            content={
                "error": error_message,
                "error_type": error_type,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "support_message": "Please contact support if this persists"
            },
            status_code=status_code,
            headers={
                "X-Request-ID": request_id,
                "X-Response-Time": f"{(time.time() - start_time) * 1000:.2f}ms",
                "X-Error-Type": error_type
            }
        )

    def _record_failure(self, endpoint: str):
        """Record failure for circuit breaker logic"""
        if endpoint not in self.error_counts:
            self.error_counts[endpoint] = 0

        self.error_counts[endpoint] += 1

        # Open circuit breaker if too many failures
        if self.error_counts[endpoint] >= 5:
            self.circuit_breakers[endpoint] = {
                "opened_at": time.time(),
                "failure_count": self.error_counts[endpoint]
            }

    def _record_success(self, endpoint: str):
        """Record success and potentially close circuit breaker"""
        if endpoint in self.error_counts:
            self.error_counts[endpoint] = max(0, self.error_counts[endpoint] - 1)

        # Close circuit breaker on success
        if endpoint in self.circuit_breakers:
            del self.circuit_breakers[endpoint]

    def _is_circuit_open(self, endpoint: str) -> bool:
        """Check if circuit breaker is open for endpoint"""
        if endpoint not in self.circuit_breakers:
            return False

        # Auto-close circuit after 30 seconds
        opened_at = self.circuit_breakers[endpoint]["opened_at"]
        if time.time() - opened_at > 30:
            del self.circuit_breakers[endpoint]
            self.error_counts[endpoint] = 0
            return False

        return True