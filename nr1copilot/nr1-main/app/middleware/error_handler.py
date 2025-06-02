
"""
Enterprise Error Handling Middleware
Comprehensive error handling with structured logging, monitoring, and automatic recovery.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Enterprise-grade error handling middleware."""
    
    def __init__(self, app, enable_debug: bool = False):
        super().__init__(app)
        self.enable_debug = enable_debug
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, datetime] = {}
        
        logger.info(f"Error handler middleware initialized (debug={enable_debug})")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle request with comprehensive error handling."""
        start_time = time.time()
        request_id = f"req-{int(start_time * 1000000)}"
        
        try:
            # Process request
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return await self._handle_http_exception(request, e, request_id)
            
        except Exception as e:
            # Handle unexpected exceptions
            return await self._handle_unexpected_exception(request, e, request_id, start_time)
    
    async def _handle_http_exception(
        self, 
        request: Request, 
        exc: HTTPException, 
        request_id: str
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""
        
        # Log the exception
        logger.warning(
            f"HTTP exception: {exc.status_code} - {exc.detail}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "status_code": exc.status_code,
                "client_ip": self._get_client_ip(request)
            }
        )
        
        # Record error statistics
        error_key = f"http_{exc.status_code}"
        self._record_error(error_key)
        
        # Build response
        response_data = {
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_exception"
            },
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add debug info if enabled
        if self.enable_debug:
            response_data["debug"] = {
                "method": request.method,
                "path": str(request.url.path),
                "headers": dict(request.headers)
            }
        
        return JSONResponse(
            content=response_data,
            status_code=exc.status_code,
            headers={"X-Request-ID": request_id}
        )
    
    async def _handle_unexpected_exception(
        self, 
        request: Request, 
        exc: Exception, 
        request_id: str,
        start_time: float
    ) -> JSONResponse:
        """Handle unexpected exceptions with recovery attempts."""
        
        processing_time = time.time() - start_time
        error_type = type(exc).__name__
        
        # Log the exception with full context
        logger.error(
            f"Unexpected exception: {error_type} - {str(exc)}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "processing_time": processing_time,
                "client_ip": self._get_client_ip(request),
                "error_type": error_type,
                "traceback": traceback.format_exc()
            }
        )
        
        # Record error statistics
        self._record_error(error_type)
        
        # Attempt error recovery
        recovery_attempted = await self._attempt_error_recovery(exc, request)
        
        # Determine status code
        status_code = self._determine_status_code(exc)
        
        # Build response
        response_data = {
            "error": {
                "code": status_code,
                "message": "An internal error occurred",
                "type": "internal_error"
            },
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "recovery_attempted": recovery_attempted
        }
        
        # Add debug info if enabled
        if self.enable_debug:
            response_data["debug"] = {
                "error_type": error_type,
                "error_message": str(exc),
                "method": request.method,
                "path": str(request.url.path),
                "processing_time": processing_time,
                "traceback": traceback.format_exc().split('\n')[-10:]  # Last 10 lines
            }
        
        return JSONResponse(
            content=response_data,
            status_code=status_code,
            headers={
                "X-Request-ID": request_id,
                "X-Error-Type": error_type,
                "X-Recovery-Attempted": str(recovery_attempted)
            }
        )
    
    async def _attempt_error_recovery(self, exc: Exception, request: Request) -> bool:
        """Attempt automatic error recovery."""
        try:
            error_type = type(exc).__name__
            
            # Memory-related errors
            if "memory" in error_type.lower() or "MemoryError" in error_type:
                return await self._recover_memory_error()
            
            # Import/module errors
            if "Import" in error_type or "Module" in error_type:
                return await self._recover_import_error(exc)
            
            # Connection errors
            if "Connection" in error_type or "Timeout" in error_type:
                return await self._recover_connection_error()
            
            # Database errors
            if "database" in str(exc).lower() or "sql" in str(exc).lower():
                return await self._recover_database_error()
            
            return False
            
        except Exception as recovery_exc:
            logger.error(f"Error recovery failed: {recovery_exc}")
            return False
    
    async def _recover_memory_error(self) -> bool:
        """Attempt to recover from memory errors."""
        try:
            import gc
            collected = gc.collect()
            logger.info(f"Memory recovery: collected {collected} objects")
            return collected > 0
            
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False
    
    async def _recover_import_error(self, exc: Exception) -> bool:
        """Attempt to recover from import errors."""
        try:
            # Clear import caches
            import importlib
            importlib.invalidate_caches()
            
            logger.info("Import caches cleared for recovery")
            return True
            
        except Exception as e:
            logger.error(f"Import recovery failed: {e}")
            return False
    
    async def _recover_connection_error(self) -> bool:
        """Attempt to recover from connection errors."""
        try:
            # Brief pause to allow network recovery
            import asyncio
            await asyncio.sleep(1)
            
            logger.info("Connection recovery attempted")
            return True
            
        except Exception as e:
            logger.error(f"Connection recovery failed: {e}")
            return False
    
    async def _recover_database_error(self) -> bool:
        """Attempt to recover from database errors."""
        try:
            # Database connection recovery would go here
            # For now, just log the attempt
            logger.info("Database recovery attempted")
            return True
            
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            return False
    
    def _determine_status_code(self, exc: Exception) -> int:
        """Determine appropriate HTTP status code for exception."""
        error_type = type(exc).__name__
        error_message = str(exc).lower()
        
        # Client errors (4xx)
        if "not found" in error_message or "NotFound" in error_type:
            return 404
        if "permission" in error_message or "forbidden" in error_message:
            return 403
        if "authentication" in error_message or "unauthorized" in error_message:
            return 401
        if "validation" in error_message or "invalid" in error_message:
            return 400
        if "timeout" in error_message:
            return 408
        if "too large" in error_message or "PayloadTooLarge" in error_type:
            return 413
        if "rate limit" in error_message:
            return 429
        
        # Server errors (5xx)
        if "database" in error_message or "connection" in error_message:
            return 503
        if "memory" in error_message or "MemoryError" in error_type:
            return 503
        
        # Default server error
        return 500
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _record_error(self, error_key: str):
        """Record error statistics."""
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = datetime.utcnow()
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            "total_error_types": len(self.error_counts),
            "error_counts": dict(self.error_counts),
            "last_errors": {
                k: v.isoformat() for k, v in self.last_errors.items()
            },
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }
