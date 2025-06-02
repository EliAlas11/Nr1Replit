
"""
Netflix-Level Error Handling Middleware
Comprehensive error handling, monitoring, and recovery
"""

import asyncio
import logging
import traceback
import uuid
from datetime import datetime
from typing import Callable, Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas import ErrorResponse
from app.config import settings

logger = logging.getLogger(__name__)


class ErrorCategory:
    """Error categorization for better handling"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    RESOURCE_NOT_FOUND = "resource_not_found"
    INTERNAL_SERVER = "internal_server"
    EXTERNAL_SERVICE = "external_service"
    NETWORK = "network"
    TIMEOUT = "timeout"
    CAPACITY = "capacity"
    DATA_CORRUPTION = "data_corruption"


class ErrorSeverity:
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorContext:
    """Error context information"""
    
    def __init__(self, request: Request, error: Exception):
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.request = request
        self.error = error
        self.category = self._categorize_error(error)
        self.severity = self._assess_severity(error)
        self.user_id = getattr(request.state, 'user_id', None)
        self.session_id = getattr(request.state, 'session_id', None)
        self.client_ip = self._get_client_ip(request)
        self.user_agent = request.headers.get("user-agent", "")
        self.path = str(request.url.path)
        self.method = request.method
        self.query_params = dict(request.query_params)
        self.headers = dict(request.headers)
        self.additional_context = {}
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP with proxy support"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return getattr(request.client, "host", "unknown")
    
    def _categorize_error(self, error: Exception) -> str:
        """Categorize error type"""
        if isinstance(error, HTTPException):
            status_code = error.status_code
            if status_code == 400:
                return ErrorCategory.VALIDATION
            elif status_code == 401:
                return ErrorCategory.AUTHENTICATION
            elif status_code == 403:
                return ErrorCategory.AUTHORIZATION
            elif status_code == 404:
                return ErrorCategory.RESOURCE_NOT_FOUND
            elif status_code == 429:
                return ErrorCategory.RATE_LIMIT
            elif 500 <= status_code < 600:
                return ErrorCategory.INTERNAL_SERVER
        
        error_type = type(error).__name__.lower()
        if "timeout" in error_type:
            return ErrorCategory.TIMEOUT
        elif "connection" in error_type or "network" in error_type:
            return ErrorCategory.NETWORK
        elif "memory" in error_type or "capacity" in error_type:
            return ErrorCategory.CAPACITY
        
        return ErrorCategory.INTERNAL_SERVER
    
    def _assess_severity(self, error: Exception) -> str:
        """Assess error severity"""
        if isinstance(error, HTTPException):
            status_code = error.status_code
            if status_code < 400:
                return ErrorSeverity.LOW
            elif status_code < 500:
                return ErrorSeverity.MEDIUM
            else:
                return ErrorSeverity.HIGH
        
        # Critical errors
        critical_patterns = ["memory", "disk", "database", "security"]
        error_str = str(error).lower()
        if any(pattern in error_str for pattern in critical_patterns):
            return ErrorSeverity.CRITICAL
        
        return ErrorSeverity.HIGH
    
    def add_context(self, key: str, value: Any):
        """Add additional context"""
        self.additional_context[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "severity": self.severity,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "client_ip": self.client_ip,
            "path": self.path,
            "method": self.method,
            "query_params": self.query_params,
            "user_agent": self.user_agent,
            "additional_context": self.additional_context
        }


class ErrorMetrics:
    """Error metrics tracking"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_rates = {}
        self.recovery_attempts = {}
        self.circuit_breaker_states = {}
    
    def record_error(self, context: ErrorContext):
        """Record error occurrence"""
        key = f"{context.category}:{context.path}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Update rates (simplified)
        current_minute = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        rate_key = f"{key}:{current_minute}"
        self.error_rates[rate_key] = self.error_rates.get(rate_key, 0) + 1
    
    def should_circuit_break(self, path: str, threshold: int = 10) -> bool:
        """Determine if circuit breaker should activate"""
        current_minute = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        rate_key = f"{path}:{current_minute}"
        return self.error_rates.get(rate_key, 0) >= threshold


class ErrorRecovery:
    """Automatic error recovery mechanisms"""
    
    @staticmethod
    async def attempt_recovery(context: ErrorContext) -> Optional[Response]:
        """Attempt automatic recovery"""
        if context.category == ErrorCategory.TIMEOUT:
            return await ErrorRecovery._handle_timeout_recovery(context)
        elif context.category == ErrorCategory.RATE_LIMIT:
            return await ErrorRecovery._handle_rate_limit_recovery(context)
        elif context.category == ErrorCategory.EXTERNAL_SERVICE:
            return await ErrorRecovery._handle_external_service_recovery(context)
        
        return None
    
    @staticmethod
    async def _handle_timeout_recovery(context: ErrorContext) -> Optional[Response]:
        """Handle timeout recovery"""
        # Implement retry with exponential backoff
        logger.info(f"Attempting timeout recovery for {context.error_id}")
        return None
    
    @staticmethod
    async def _handle_rate_limit_recovery(context: ErrorContext) -> Optional[Response]:
        """Handle rate limit recovery"""
        # Return appropriate retry-after response
        return JSONResponse(
            status_code=429,
            content={
                "error": True,
                "error_id": context.error_id,
                "message": "Rate limit exceeded. Please try again later.",
                "retry_after": 60,
                "timestamp": context.timestamp.isoformat()
            },
            headers={"Retry-After": "60"}
        )
    
    @staticmethod
    async def _handle_external_service_recovery(context: ErrorContext) -> Optional[Response]:
        """Handle external service recovery"""
        # Implement fallback mechanisms
        logger.info(f"Attempting external service recovery for {context.error_id}")
        return None


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Netflix-level error handling middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = ErrorMetrics()
        self.error_patterns = {}
        self.alert_thresholds = {
            ErrorSeverity.CRITICAL: 1,
            ErrorSeverity.HIGH: 5,
            ErrorSeverity.MEDIUM: 20,
            ErrorSeverity.LOW: 100
        }
        self.recovery_enabled = True
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main error handling dispatch"""
        try:
            # Add request tracking
            request.state.start_time = datetime.utcnow()
            request.state.request_id = str(uuid.uuid4())
            
            response = await call_next(request)
            return response
            
        except Exception as error:
            return await self._handle_error(request, error)
    
    async def _handle_error(self, request: Request, error: Exception) -> Response:
        """Comprehensive error handling"""
        context = ErrorContext(request, error)
        
        # Add request-specific context
        context.add_context("request_id", getattr(request.state, 'request_id', None))
        context.add_context("processing_time", self._get_processing_time(request))
        
        # Log error with full context
        await self._log_error(context)
        
        # Record metrics
        self.metrics.record_error(context)
        
        # Check for circuit breaker activation
        if self.metrics.should_circuit_break(context.path):
            logger.warning(f"Circuit breaker activated for {context.path}")
            return await self._create_circuit_breaker_response(context)
        
        # Attempt automatic recovery
        if self.recovery_enabled:
            recovery_response = await ErrorRecovery.attempt_recovery(context)
            if recovery_response:
                logger.info(f"Successfully recovered from error {context.error_id}")
                return recovery_response
        
        # Send alerts if needed
        await self._send_alert_if_needed(context)
        
        # Create error response
        return await self._create_error_response(context)
    
    async def _log_error(self, context: ErrorContext):
        """Log error with appropriate level"""
        error_data = context.to_dict()
        
        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(
                f"CRITICAL ERROR {context.error_id}: {context.error}",
                extra={
                    "error_context": error_data,
                    "traceback": traceback.format_exc()
                }
            )
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(
                f"HIGH SEVERITY ERROR {context.error_id}: {context.error}",
                extra={"error_context": error_data}
            )
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(
                f"MEDIUM SEVERITY ERROR {context.error_id}: {context.error}",
                extra={"error_context": error_data}
            )
        else:
            logger.info(
                f"LOW SEVERITY ERROR {context.error_id}: {context.error}",
                extra={"error_context": error_data}
            )
    
    async def _send_alert_if_needed(self, context: ErrorContext):
        """Send alerts for high-severity errors"""
        if context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            # In production, this would integrate with alerting systems
            logger.warning(f"ALERT: {context.severity} error detected - {context.error_id}")
    
    async def _create_error_response(self, context: ErrorContext) -> Response:
        """Create standardized error response"""
        if isinstance(context.error, HTTPException):
            status_code = context.error.status_code
            message = context.error.detail
        elif isinstance(context.error, StarletteHTTPException):
            status_code = context.error.status_code
            message = context.error.detail
        else:
            status_code = 500
            message = "Internal server error" if not settings.debug else str(context.error)
        
        error_response = ErrorResponse(
            error_id=context.error_id,
            message=message,
            status_code=status_code,
            path=context.path,
            details={
                "category": context.category,
                "severity": context.severity,
                "request_id": getattr(context.request.state, 'request_id', None)
            } if settings.debug else None
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict(),
            headers={
                "X-Error-ID": context.error_id,
                "X-Request-ID": getattr(context.request.state, 'request_id', ''),
                "Content-Type": "application/json"
            }
        )
    
    async def _create_circuit_breaker_response(self, context: ErrorContext) -> Response:
        """Create circuit breaker response"""
        return JSONResponse(
            status_code=503,
            content={
                "error": True,
                "error_id": context.error_id,
                "message": "Service temporarily unavailable due to high error rate",
                "category": "circuit_breaker",
                "retry_after": 60,
                "timestamp": context.timestamp.isoformat()
            },
            headers={
                "Retry-After": "60",
                "X-Circuit-Breaker": "active"
            }
        )
    
    def _get_processing_time(self, request: Request) -> Optional[float]:
        """Calculate request processing time"""
        start_time = getattr(request.state, 'start_time', None)
        if start_time:
            return (datetime.utcnow() - start_time).total_seconds()
        return None
    
    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        total_errors = sum(self.metrics.error_counts.values())
        
        # Group by category
        category_stats = {}
        for key, count in self.metrics.error_counts.items():
            category = key.split(':')[0]
            category_stats[category] = category_stats.get(category, 0) + count
        
        # Calculate error rates
        current_minute = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        current_minute_errors = sum(
            count for key, count in self.metrics.error_rates.items()
            if key.endswith(current_minute)
        )
        
        return {
            "total_errors": total_errors,
            "errors_by_category": category_stats,
            "current_minute_errors": current_minute_errors,
            "circuit_breaker_states": self.metrics.circuit_breaker_states,
            "recovery_enabled": self.recovery_enabled
        }
    
    @asynccontextmanager
    async def error_context(self, operation_name: str):
        """Context manager for operation-specific error handling"""
        operation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            logger.debug(f"Starting operation {operation_name} [{operation_id}]")
            yield operation_id
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                f"Operation {operation_name} [{operation_id}] failed after {duration:.2f}s: {e}",
                extra={
                    "operation_name": operation_name,
                    "operation_id": operation_id,
                    "duration": duration,
                    "error_type": type(e).__name__
                }
            )
            raise
        else:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"Operation {operation_name} [{operation_id}] completed in {duration:.2f}s")


# Utility functions for error handling
def create_http_exception(
    status_code: int,
    message: str,
    error_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create standardized HTTP exception"""
    if not error_id:
        error_id = str(uuid.uuid4())
    
    return HTTPException(
        status_code=status_code,
        detail=message,
        headers={
            "X-Error-ID": error_id,
            "Content-Type": "application/json"
        }
    )


def handle_validation_error(error: Exception) -> HTTPException:
    """Handle validation errors consistently"""
    return create_http_exception(
        status_code=422,
        message="Validation failed",
        details={"validation_error": str(error)}
    )


def handle_not_found_error(resource: str, identifier: str) -> HTTPException:
    """Handle not found errors consistently"""
    return create_http_exception(
        status_code=404,
        message=f"{resource} not found",
        details={"resource": resource, "identifier": identifier}
    )


def handle_unauthorized_error(message: str = "Authentication required") -> HTTPException:
    """Handle unauthorized errors consistently"""
    return create_http_exception(
        status_code=401,
        message=message
    )


def handle_forbidden_error(message: str = "Access forbidden") -> HTTPException:
    """Handle forbidden errors consistently"""
    return create_http_exception(
        status_code=403,
        message=message
    )


__all__ = [
    "ErrorHandlerMiddleware",
    "ErrorContext",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorMetrics",
    "ErrorRecovery",
    "create_http_exception",
    "handle_validation_error",
    "handle_not_found_error",
    "handle_unauthorized_error",
    "handle_forbidden_error"
]
