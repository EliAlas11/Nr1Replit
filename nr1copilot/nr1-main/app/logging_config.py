
"""
Netflix-Level Logging Configuration
Structured logging with JSON format and performance monitoring
"""

import json
import logging
import logging.handlers
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import uuid


class StructuredFormatter(logging.Formatter):
    """Structured console formatter for development"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add structured fields
        record.timestamp = datetime.fromtimestamp(record.created).isoformat()
        record.level = record.levelname
        record.logger = record.name
        
        # Add request context if available
        if hasattr(record, 'request_id'):
            record.context = f"[{record.request_id}]"
        else:
            record.context = ""
        
        # Format message
        if record.levelno >= logging.ERROR:
            return f"🔴 {record.timestamp} [{record.level}] {record.context} {record.getMessage()}"
        elif record.levelno >= logging.WARNING:
            return f"🟡 {record.timestamp} [{record.level}] {record.context} {record.getMessage()}"
        elif record.levelno >= logging.INFO:
            return f"🟢 {record.timestamp} [{record.level}] {record.context} {record.getMessage()}"
        else:
            return f"🔵 {record.timestamp} [{record.level}] {record.context} {record.getMessage()}"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info else None
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info', 
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated', 
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


def setup_logging(log_level: str = "INFO", log_file: bool = True):
    """Setup Netflix-level logging configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with structured format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = StructuredFormatter()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    if log_file:
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "application.jsonl",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.jsonl",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    # Set third-party loggers to WARNING
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    logging.info(f"🔧 Logging configured - Level: {log_level}, File logging: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with Netflix-level configuration"""
    return logging.getLogger(name)


class PerformanceLogger:
    """Netflix-level performance logging"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)
    
    def log_request(self, request_id: str, method: str, path: str, 
                   status_code: int, duration: float, **kwargs):
        """Log HTTP request performance"""
        self.logger.info(
            f"Request completed: {method} {path} - {status_code} - {duration:.3f}s",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration": duration,
                "performance_type": "http_request",
                **kwargs
            }
        )
    
    def log_operation(self, operation: str, duration: float, 
                     success: bool = True, **kwargs):
        """Log operation performance"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"Operation {operation} {status} in {duration:.3f}s",
            extra={
                "operation": operation,
                "duration": duration,
                "success": success,
                "performance_type": "operation",
                **kwargs
            }
        )
    
    def log_video_processing(self, session_id: str, stage: str, 
                           duration: float, file_size: int = None, **kwargs):
        """Log video processing performance"""
        self.logger.info(
            f"Video processing {stage} completed in {duration:.3f}s",
            extra={
                "session_id": session_id,
                "stage": stage,
                "duration": duration,
                "file_size": file_size,
                "performance_type": "video_processing",
                **kwargs
            }
        )


# Context manager for performance logging
class LogPerformance:
    """Context manager for automatic performance logging"""
    
    def __init__(self, logger: PerformanceLogger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        self.logger.log_operation(
            self.operation, 
            duration, 
            success=success, 
            **self.kwargs
        )


# Application-specific loggers
def get_api_logger() -> logging.Logger:
    """Get API-specific logger"""
    return get_logger("api")


def get_video_logger() -> logging.Logger:
    """Get video processing logger"""
    return get_logger("video")


def get_websocket_logger() -> logging.Logger:
    """Get WebSocket logger"""
    return get_logger("websocket")


def get_security_logger() -> logging.Logger:
    """Get security logger"""
    return get_logger("security")


# Request ID context
class RequestContext:
    """Request context for logging"""
    
    def __init__(self):
        self._context = {}
    
    def set_request_id(self, request_id: str):
        """Set request ID for current context"""
        self._context["request_id"] = request_id
    
    def get_request_id(self) -> Optional[str]:
        """Get current request ID"""
        return self._context.get("request_id")
    
    def clear(self):
        """Clear current context"""
        self._context.clear()


# Global request context
request_context = RequestContext()
