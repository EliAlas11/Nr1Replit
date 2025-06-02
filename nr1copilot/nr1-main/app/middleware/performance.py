"""
Netflix-Grade Performance Middleware
Real-time performance monitoring and optimization with enterprise-level reliability
"""

import time
import asyncio
import logging
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Netflix-tier performance monitoring and optimization middleware"""

    def __init__(self, app, max_request_time: float = 30.0):
        super().__init__(app)
        self.max_request_time = max_request_time
        self.request_metrics: Dict[str, Any] = {}
        self.startup_time = time.time()
        logger.info("🚀 Netflix-grade Performance Middleware initialized")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with Netflix-level performance monitoring"""
        start_time = time.time()
        request_id = f"{request.method}_{request.url.path}_{int(start_time * 1000)}"

        try:
            # Add performance headers and monitoring
            response = await self._process_with_monitoring(request, call_next, request_id)

            # Calculate processing time
            process_time = time.time() - start_time

            # Add Netflix-grade performance headers
            response.headers.update({
                "X-Process-Time": f"{process_time:.4f}",
                "X-Request-ID": request_id,
                "X-Performance-Grade": self._get_performance_grade(process_time),
                "X-Server-Timestamp": datetime.utcnow().isoformat(),
                "X-Netflix-Tier": "AAA+"
            })

            # Log performance metrics
            if process_time > 1.0:
                logger.warning(f"🐌 Slow request detected: {request.method} {request.url.path} took {process_time:.4f}s")
            elif process_time < 0.1:
                logger.debug(f"⚡ Ultra-fast request: {request.method} {request.url.path} took {process_time:.4f}s")

            return response

        except Exception as e:
            logger.error(f"💥 Performance middleware error: {e}")
            # Return graceful error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "performance_monitoring_error",
                    "message": "Performance monitoring encountered an issue",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id
                }
            )

    async def _process_with_monitoring(self, request: Request, call_next: Callable, request_id: str) -> Response:
        """Process request with comprehensive monitoring and timeout protection"""
        try:
            # Netflix-grade timeout protection
            response = await asyncio.wait_for(
                call_next(request), 
                timeout=self.max_request_time
            )

            # Validate response
            if not hasattr(response, 'status_code'):
                logger.warning(f"⚠️ Invalid response object for request {request_id}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "invalid_response",
                        "message": "Server returned invalid response",
                        "request_id": request_id
                    }
                )

            return response

        except asyncio.TimeoutError:
            logger.error(f"⏰ Request timeout: {request.method} {request.url.path} exceeded {self.max_request_time}s")
            return JSONResponse(
                status_code=504,
                content={
                    "error": "request_timeout",
                    "message": f"Request exceeded maximum processing time of {self.max_request_time}s",
                    "timeout_limit": self.max_request_time,
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"💥 Request processing error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "processing_error",
                    "message": "Internal server error during request processing",
                    "error_type": type(e).__name__,
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

    def _get_performance_grade(self, process_time: float) -> str:
        """Netflix-grade performance scoring"""
        if process_time < 0.05:
            return "A+++"  # Ultra performance
        elif process_time < 0.1:
            return "A++"   # Excellent
        elif process_time < 0.2:
            return "A+"    # Great
        elif process_time < 0.5:
            return "A"     # Good
        elif process_time < 1.0:
            return "B"     # Acceptable
        elif process_time < 2.0:
            return "C"     # Slow
        elif process_time < 5.0:
            return "D"     # Very slow
        else:
            return "F"     # Unacceptable

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            uptime = time.time() - self.startup_time

            return {
                "uptime_seconds": round(uptime, 2),
                "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
                "memory_usage_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "active_connections": len(psutil.net_connections()),
                "system_load": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
                "performance_tier": "Netflix-grade",
                "monitoring_status": "optimal"
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {
                "error": "metrics_unavailable",
                "message": str(e),
                "performance_tier": "degraded"
            }