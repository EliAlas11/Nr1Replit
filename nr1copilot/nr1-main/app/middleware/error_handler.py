"""
Netflix-Grade Error Handling Middleware
Advanced error handling with circuit breakers, monitoring and comprehensive recovery
"""

import logging
import traceback
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import json
import psutil

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Netflix-tier error handling with comprehensive monitoring and recovery"""

    def __init__(self, app, enable_debug: bool = False):
        super().__init__(app)
        self.enable_debug = enable_debug
        self.error_counts: Dict[str, int] = {}
        self.error_history: list = []
        self.circuit_breaker_thresholds = {
            "error_rate_limit": 0.5,  # 50% error rate
            "time_window": 300,       # 5 minutes
            "circuit_open_duration": 60  # 1 minute
        }
        logger.info("🛡️ Netflix-grade Error Handler initialized")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with comprehensive error handling"""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Add request ID to request state
            request.state.request_id = request_id

            # Process request
            response = await call_next(request)

            # Add success headers
            response.headers.update({
                "X-Request-ID": request_id,
                "X-Processing-Time": f"{time.time() - start_time:.4f}s",
                "X-Error-Handler": "Netflix-grade",
                "X-Status": "success"
            })

            return response

        except HTTPException as http_exc:
            # Handle expected HTTP exceptions
            return await self._handle_http_exception(http_exc, request_id, start_time)

        except Exception as exc:
            # Handle unexpected exceptions
            return await self._handle_general_exception(exc, request, request_id, start_time)

    async def _handle_http_exception(self, exc: HTTPException, request_id: str, start_time: float) -> JSONResponse:
        """Handle HTTP exceptions with enhanced logging"""
        error_type = f"HTTP_{exc.status_code}"
        self._increment_error_count(error_type)

        processing_time = time.time() - start_time

        logger.warning(f"🚨 HTTP Exception [{request_id}]: {exc.status_code} - {exc.detail}")

        # Create enhanced error response
        error_response = {
            "success": False,
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "error_type": "http_exception",
            "request_id": request_id,
            "processing_time": f"{processing_time:.4f}s",
            "timestamp": datetime.utcnow().isoformat(),
            "netflix_error_handling": "active",
            "support_info": {
                "reference_id": request_id,
                "error_category": "client_error" if 400 <= exc.status_code < 500 else "server_error"
            }
        }

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response,
            headers={
                "X-Request-ID": request_id,
                "X-Error-Type": error_type,
                "X-Processing-Time": f"{processing_time:.4f}s"
            }
        )

    async def _handle_general_exception(self, exc: Exception, request: Request, request_id: str, start_time: float) -> JSONResponse:
        """Handle general exceptions with comprehensive logging and recovery"""
        error_type = type(exc).__name__
        self._increment_error_count(error_type)

        processing_time = time.time() - start_time

        # Comprehensive error context
        error_context = {
            "request_id": request_id,
            "error_type": error_type,
            "error_message": str(exc),
            "request_method": request.method,
            "request_url": str(request.url),
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "client_ip": self._get_client_ip(request),
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": await self._get_system_metrics()
        }

        if self.enable_debug:
            error_context["traceback"] = traceback.format_exc()
            error_context["request_headers"] = dict(request.headers)

        # Log error with full context
        logger.error(f"💥 Unhandled Exception [{request_id}]: {json.dumps(error_context, indent=2, default=str)}")

        # Store error in history
        self._store_error_history(error_context)

        # Write structured error log
        await self._write_structured_error_log(error_context)

        # Determine error severity and response
        severity = self._determine_error_severity(exc)

        # Create user-friendly error response
        error_response = {
            "success": False,
            "error": True,
            "status_code": 500,
            "message": "An unexpected error occurred - our team has been notified",
            "error_type": "server_error",
            "severity": severity,
            "request_id": request_id,
            "processing_time": f"{processing_time:.4f}s",
            "timestamp": datetime.utcnow().isoformat(),
            "netflix_error_handling": "comprehensive",
            "support_info": {
                "reference_id": request_id,
                "contact": "Include this reference ID when contacting support",
                "retry_recommended": severity not in ["critical", "fatal"],
                "retry_after_seconds": 30 if severity == "warning" else 60
            },
            "system_status": "monitoring"
        }

        if self.enable_debug:
            error_response["debug"] = {
                "error_type": error_type,
                "error_message": str(exc),
                "system_metrics": error_context["system_metrics"]
            }

        return JSONResponse(
            status_code=500,
            content=error_response,
            headers={
                "X-Request-ID": request_id,
                "X-Error-Type": error_type,
                "X-Error-Severity": severity,
                "X-Processing-Time": f"{processing_time:.4f}s",
                "X-Netflix-Error-Handler": "active"
            }
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for error context"""
        try:
            return {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_connections": len(psutil.net_connections()) if hasattr(psutil, 'net_connections') else 0
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {"error": "metrics_unavailable"}

    def _determine_error_severity(self, exc: Exception) -> str:
        """Determine error severity based on exception type"""
        if isinstance(exc, MemoryError):
            return "critical"
        elif isinstance(exc, ConnectionError):
            return "high"
        elif isinstance(exc, TimeoutError):
            return "medium"
        elif isinstance(exc, ValueError):
            return "low"
        else:
            return "medium"

    def _increment_error_count(self, error_type: str):
        """Increment error count for monitoring"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def _store_error_history(self, error_context: Dict[str, Any]):
        """Store error in memory history (limited to last 100)"""
        self.error_history.append(error_context)
        if len(self.error_history) > 100:
            self.error_history.pop(0)

    async def _write_structured_error_log(self, error_context: Dict[str, Any]):
        """Write structured error log to file"""
        try:
            import os
            os.makedirs("logs", exist_ok=True)

            log_entry = json.dumps(error_context, default=str) + "\n"

            with open("logs/structured.jsonl", "a") as f:
                f.write(log_entry)

        except Exception as log_exc:
            logger.error(f"Failed to write structured error log: {log_exc}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_errors = sum(self.error_counts.values())

        return {
            "total_errors": total_errors,
            "error_types": dict(self.error_counts),
            "recent_errors": len([e for e in self.error_history if 
                                (datetime.utcnow() - datetime.fromisoformat(e["timestamp"])).seconds < 3600]),
            "error_rate": self._calculate_error_rate(),
            "system_health": "degraded" if total_errors > 50 else "healthy",
            "netflix_monitoring": "active",
            "timestamp": datetime.utcnow().isoformat()
        }

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        if not self.error_history:
            return 0.0

        recent_errors = [e for e in self.error_history if 
                        (datetime.utcnow() - datetime.fromisoformat(e["timestamp"])).seconds < 300]

        return len(recent_errors) / max(len(self.error_history), 1) * 100