
"""
Netflix-Level Error Handling Middleware
Comprehensive error tracking and recovery
"""

import asyncio
import logging
import time
import uuid
from typing import Callable, Dict, Any
from datetime import datetime

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Netflix-level error handling with circuit breaker and recovery"""
    
    def __init__(self, app):
        super().__init__(app)
        self.error_counts: Dict[str, int] = {}
        self.circuit_breaker_threshold = 10
        self.recovery_time = 60  # seconds
        self.last_errors: Dict[str, datetime] = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enhanced error handling with circuit breaker pattern"""
        start_time = time.time()
        error_id = str(uuid.uuid4())
        
        # Add request ID for tracing
        request.state.request_id = error_id
        request.state.start_time = start_time
        
        try:
            # Check circuit breaker
            if self._should_circuit_break(request):
                return self._circuit_breaker_response(request)
                
            response = await call_next(request)
            
            # Track successful requests
            self._track_success(request)
            
            # Add performance headers
            response.headers["X-Request-ID"] = error_id
            response.headers["X-Response-Time"] = f"{(time.time() - start_time) * 1000:.2f}ms"
            
            return response
            
        except Exception as e:
            # Track error
            self._track_error(request, e)
            
            # Log comprehensive error info
            logger.error(
                f"Request failed {error_id}: {str(e)}",
                extra={
                    "request_id": error_id,
                    "path": request.url.path,
                    "method": request.method,
                    "client_ip": request.client.host,
                    "processing_time": time.time() - start_time,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            
            # Return standardized error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "error_id": error_id,
                    "message": "Internal server error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": request.url.path
                }
            )
            
    def _should_circuit_break(self, request: Request) -> bool:
        """Check if circuit breaker should be triggered"""
        endpoint = f"{request.method}:{request.url.path}"
        
        # Check error count
        error_count = self.error_counts.get(endpoint, 0)
        if error_count < self.circuit_breaker_threshold:
            return False
            
        # Check recovery time
        last_error = self.last_errors.get(endpoint)
        if last_error:
            time_since_error = (datetime.utcnow() - last_error).total_seconds()
            if time_since_error > self.recovery_time:
                # Reset circuit breaker
                self.error_counts[endpoint] = 0
                return False
                
        return True
        
    def _circuit_breaker_response(self, request: Request) -> JSONResponse:
        """Return circuit breaker response"""
        logger.warning(f"Circuit breaker triggered for {request.url.path}")
        
        return JSONResponse(
            status_code=503,
            content={
                "error": True,
                "message": "Service temporarily unavailable",
                "retry_after": self.recovery_time,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    def _track_error(self, request: Request, error: Exception):
        """Track error for circuit breaker"""
        endpoint = f"{request.method}:{request.url.path}"
        self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1
        self.last_errors[endpoint] = datetime.utcnow()
        
    def _track_success(self, request: Request):
        """Track successful request"""
        endpoint = f"{request.method}:{request.url.path}"
        
        # Gradually reduce error count on success
        if endpoint in self.error_counts and self.error_counts[endpoint] > 0:
            self.error_counts[endpoint] = max(0, self.error_counts[endpoint] - 1)
