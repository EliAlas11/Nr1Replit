
"""
Netflix-Grade API Response Handler
Centralized response formatting with consistent error handling
"""

import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class APIStatusCode(str, Enum):
    """Standardized API status codes"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"
    PROCESSING = "processing"


class ErrorCode(str, Enum):
    """Standardized error codes"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    MAINTENANCE_MODE = "MAINTENANCE_MODE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class APIResponse(BaseModel):
    """Standardized API response format"""
    success: bool = Field(..., description="Operation success status")
    status: APIStatusCode = Field(..., description="Response status")
    data: Optional[Any] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Human-readable message")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class APIError(BaseModel):
    """Standardized API error format"""
    code: ErrorCode = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    field: Optional[str] = Field(None, description="Field that caused the error")
    suggestion: Optional[str] = Field(None, description="Suggested fix")
    docs_url: Optional[str] = Field(None, description="Documentation URL")


class PaginationMeta(BaseModel):
    """Pagination metadata"""
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, le=100, description="Items per page")
    total: int = Field(..., ge=0, description="Total number of items")
    pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


class APIResponseBuilder:
    """Builder for creating standardized API responses"""
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "Operation completed successfully",
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> APIResponse:
        """Create a success response"""
        return APIResponse(
            success=True,
            status=APIStatusCode.SUCCESS,
            data=data,
            message=message,
            metadata=metadata or {},
            request_id=request_id
        )
    
    @staticmethod
    def error(
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        field: Optional[str] = None,
        suggestion: Optional[str] = None,
        docs_url: Optional[str] = None,
        request_id: Optional[str] = None,
        http_status: int = 400
    ) -> JSONResponse:
        """Create an error response"""
        error_obj = APIError(
            code=error_code,
            message=message,
            details=details,
            field=field,
            suggestion=suggestion,
            docs_url=docs_url
        )
        
        response = APIResponse(
            success=False,
            status=APIStatusCode.ERROR,
            error=error_obj.dict(),
            request_id=request_id
        )
        
        return JSONResponse(
            content=response.dict(),
            status_code=http_status,
            headers={
                "X-Error-Code": error_code.value,
                "X-Request-ID": request_id or "",
                "Content-Type": "application/json"
            }
        )
    
    @staticmethod
    def validation_error(
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """Create a validation error response"""
        return APIResponseBuilder.error(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
            field=field,
            details=details,
            suggestion="Please check the input parameters and try again",
            docs_url="https://docs.viralclip.pro/validation",
            request_id=request_id,
            http_status=400
        )
    
    @staticmethod
    def authentication_error(
        message: str = "Authentication required",
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """Create an authentication error response"""
        return APIResponseBuilder.error(
            error_code=ErrorCode.AUTHENTICATION_ERROR,
            message=message,
            suggestion="Please provide valid authentication credentials",
            docs_url="https://docs.viralclip.pro/auth",
            request_id=request_id,
            http_status=401
        )
    
    @staticmethod
    def rate_limit_error(
        retry_after: int,
        limit: int,
        window: int,
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """Create a rate limit error response"""
        details = {
            "limit": limit,
            "window_seconds": window,
            "retry_after_seconds": retry_after
        }
        
        response = APIResponseBuilder.error(
            error_code=ErrorCode.RATE_LIMIT_ERROR,
            message=f"Rate limit exceeded. Try again in {retry_after} seconds",
            details=details,
            suggestion="Reduce request frequency or upgrade your plan",
            docs_url="https://docs.viralclip.pro/rate-limits",
            request_id=request_id,
            http_status=429
        )
        
        # Add rate limit headers
        response.headers.update({
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time() + retry_after)),
            "Retry-After": str(retry_after)
        })
        
        return response
    
    @staticmethod
    def processing(
        message: str = "Request is being processed",
        data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> APIResponse:
        """Create a processing response"""
        return APIResponse(
            success=True,
            status=APIStatusCode.PROCESSING,
            data=data,
            message=message,
            metadata=metadata or {},
            request_id=request_id
        )
    
    @staticmethod
    def paginated_success(
        data: List[Any],
        pagination: PaginationMeta,
        message: str = "Data retrieved successfully",
        request_id: Optional[str] = None
    ) -> APIResponse:
        """Create a paginated success response"""
        metadata = {
            "pagination": pagination.dict()
        }
        
        return APIResponse(
            success=True,
            status=APIStatusCode.SUCCESS,
            data=data,
            message=message,
            metadata=metadata,
            request_id=request_id
        )


# Convenience functions
def success_response(data: Any = None, message: str = "Success", **kwargs) -> APIResponse:
    """Create a success response"""
    return APIResponseBuilder.success(data=data, message=message, **kwargs)


def error_response(error_code: ErrorCode, message: str, **kwargs) -> JSONResponse:
    """Create an error response"""
    return APIResponseBuilder.error(error_code=error_code, message=message, **kwargs)


def validation_error_response(message: str, **kwargs) -> JSONResponse:
    """Create a validation error response"""
    return APIResponseBuilder.validation_error(message=message, **kwargs)


def auth_error_response(message: str = "Authentication required", **kwargs) -> JSONResponse:
    """Create an authentication error response"""
    return APIResponseBuilder.authentication_error(message=message, **kwargs)


def rate_limit_response(retry_after: int, limit: int, window: int, **kwargs) -> JSONResponse:
    """Create a rate limit error response"""
    return APIResponseBuilder.rate_limit_error(
        retry_after=retry_after, 
        limit=limit, 
        window=window, 
        **kwargs
    )


# Export main classes and functions
__all__ = [
    'APIResponse',
    'APIError', 
    'APIStatusCode',
    'ErrorCode',
    'PaginationMeta',
    'APIResponseBuilder',
    'success_response',
    'error_response',
    'validation_error_response',
    'auth_error_response',
    'rate_limit_response'
]
