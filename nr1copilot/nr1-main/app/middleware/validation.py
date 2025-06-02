
"""
Netflix-Grade Input Validation Middleware
Comprehensive request validation with sanitization and security checks
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from pydantic import ValidationError

from ..utils.api_responses import APIResponseBuilder, ErrorCode

logger = logging.getLogger(__name__)


class SecurityPatterns:
    """Security patterns for input validation"""
    
    # SQL injection patterns
    SQL_INJECTION = [
        r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
        r"(\b(or|and)\s+\d+\s*=\s*\d+)",
        r"('|\"|;|--|\/\*|\*\/)",
        r"(\bxp_cmdshell\b|\bsp_executesql\b)"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",
        r"javascript\s*:",
        r"on\w+\s*=",
        r"<\s*iframe[^>]*>",
        r"<\s*object[^>]*>",
        r"<\s*embed[^>]*>"
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e/",
        r"%2e%2e\\",
        r"\.\.%2f",
        r"\.\.%5c"
    ]
    
    # Command injection patterns
    COMMAND_INJECTION = [
        r"[;&|`$(){}\[\]]",
        r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig)\b"
    ]


class InputValidator:
    """Input validation and sanitization"""
    
    def __init__(self):
        self.max_content_length = 50 * 1024 * 1024  # 50MB
        self.max_field_length = 10000
        self.max_array_length = 1000
        self.allowed_content_types = {
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain"
        }
    
    def validate_content_type(self, request: Request) -> bool:
        """Validate request content type"""
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "").split(";")[0].strip()
            return content_type in self.allowed_content_types
        return True
    
    def validate_content_length(self, request: Request) -> bool:
        """Validate request content length"""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                return length <= self.max_content_length
            except ValueError:
                return False
        return True
    
    def detect_security_threats(self, text: str) -> List[str]:
        """Detect security threats in input text"""
        threats = []
        text_lower = text.lower()
        
        # Check SQL injection
        for pattern in SecurityPatterns.SQL_INJECTION:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append("sql_injection")
                break
        
        # Check XSS
        for pattern in SecurityPatterns.XSS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append("xss")
                break
        
        # Check path traversal
        for pattern in SecurityPatterns.PATH_TRAVERSAL:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append("path_traversal")
                break
        
        # Check command injection
        for pattern in SecurityPatterns.COMMAND_INJECTION:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append("command_injection")
                break
        
        return threats
    
    def sanitize_string(self, text: str) -> str:
        """Sanitize string input"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Limit length
        if len(text) > self.max_field_length:
            text = text[:self.max_field_length]
        
        # Remove dangerous characters for basic sanitization
        text = re.sub(r'[<>"\']', '', text)
        
        return text.strip()
    
    def validate_json_structure(self, data: Any, max_depth: int = 10) -> bool:
        """Validate JSON structure depth and complexity"""
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                return False
            
            if isinstance(obj, dict):
                if len(obj) > 100:  # Limit object size
                    return False
                for value in obj.values():
                    if not check_depth(value, current_depth + 1):
                        return False
            elif isinstance(obj, list):
                if len(obj) > self.max_array_length:
                    return False
                for item in obj:
                    if not check_depth(item, current_depth + 1):
                        return False
            
            return True
        
        return check_depth(data)
    
    async def validate_request_data(self, request: Request) -> Optional[Dict[str, Any]]:
        """Validate and sanitize request data"""
        try:
            # Validate content type
            if not self.validate_content_type(request):
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported content type"
                )
            
            # Validate content length
            if not self.validate_content_length(request):
                raise HTTPException(
                    status_code=413,
                    detail="Request entity too large"
                )
            
            # Parse and validate JSON data
            if request.headers.get("content-type", "").startswith("application/json"):
                body = await request.body()
                if body:
                    try:
                        data = json.loads(body)
                        
                        # Validate JSON structure
                        if not self.validate_json_structure(data):
                            raise HTTPException(
                                status_code=400,
                                detail="JSON structure too complex or deep"
                            )
                        
                        # Security validation
                        threats = self._check_data_security(data)
                        if threats:
                            logger.warning(f"Security threats detected: {threats}")
                            raise HTTPException(
                                status_code=400,
                                detail=f"Security validation failed: {', '.join(threats)}"
                            )
                        
                        return data
                    except json.JSONDecodeError:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid JSON format"
                        )
            
            return None
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=400,
                detail="Request validation failed"
            )
    
    def _check_data_security(self, data: Any) -> List[str]:
        """Recursively check data for security threats"""
        threats = []
        
        if isinstance(data, str):
            threats.extend(self.detect_security_threats(data))
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str):
                    threats.extend(self.detect_security_threats(key))
                threats.extend(self._check_data_security(value))
        elif isinstance(data, list):
            for item in data:
                threats.extend(self._check_data_security(item))
        
        return list(set(threats))  # Remove duplicates


class ValidationMiddleware(BaseHTTPMiddleware):
    """Input validation middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.validator = InputValidator()
        self.skip_paths = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with validation"""
        
        # Skip validation for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            return await call_next(request)
        
        # Add request validation data to request state
        try:
            validated_data = await self.validator.validate_request_data(request)
            request.state.validated_data = validated_data
        except HTTPException as e:
            return APIResponseBuilder.validation_error(
                message=e.detail,
                details={"status_code": e.status_code},
                request_id=getattr(request.state, "request_id", None)
            )
        except Exception as e:
            logger.error(f"Validation middleware error: {e}")
            return APIResponseBuilder.error(
                error_code=ErrorCode.INTERNAL_ERROR,
                message="Request validation failed",
                request_id=getattr(request.state, "request_id", None),
                http_status=500
            )
        
        return await call_next(request)


# Export
__all__ = ['ValidationMiddleware', 'InputValidator', 'SecurityPatterns']
