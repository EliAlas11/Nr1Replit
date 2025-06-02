
"""
Netflix-Level Logging Configuration
Advanced logging with structured output, correlation tracking, and performance monitoring
"""

import json
import logging
import logging.config
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4
import asyncio
from contextvars import ContextVar
import threading

from .config import settings


# Context variables for request correlation
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


class CorrelationFormatter(logging.Formatter):
    """Enhanced formatter with correlation IDs and structured logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hostname = self._get_hostname()
        self.process_id = threading.get_ident()
    
    def _get_hostname(self) -> str:
        """Get hostname safely"""
        try:
            import socket
            return socket.gethostname()
        except Exception:
            return "unknown"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with correlation data"""
        
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": self.process_id,
            "hostname": self.hostname
        }
        
        # Add correlation IDs
        if request_id := request_id_var.get(None):
            log_data["request_id"] = request_id
        
        if user_id := user_id_var.get(None):
            log_data["user_id"] = user_id
        
        if session_id := session_id_var.get(None):
            log_data["session_id"] = session_id
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'asctime'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_data["extra"] = extra_fields
        
        # Add performance metrics if available
        if hasattr(record, 'duration'):
            log_data["performance"] = {
                "duration_ms": getattr(record, 'duration', 0),
                "operation": getattr(record, 'operation', 'unknown')
            }
        
        # Add request context if available
        if hasattr(record, 'request_method'):
            log_data["request"] = {
                "method": getattr(record, 'request_method'),
                "path": getattr(record, 'request_path', ''),
                "ip": getattr(record, 'client_ip', ''),
                "user_agent": getattr(record, 'user_agent', '')
            }
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance context to log records"""
        
        # Add memory usage
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = round(process.memory_info().rss / 1024 / 1024, 2)
            record.cpu_percent = process.cpu_percent()
        except Exception:
            pass
        
        # Add async context info
        try:
            if asyncio.current_task():
                record.async_task = True
                record.event_loop = str(asyncio.get_event_loop())
        except RuntimeError:
            record.async_task = False
        
        return True


class ErrorAggregationHandler(logging.Handler):
    """Handler to aggregate and report errors"""
    
    def __init__(self, max_errors: int = 100):
        super().__init__()
        self.max_errors = max_errors
        self.error_counts = {}
        self.recent_errors = []
        self.lock = threading.Lock()
    
    def emit(self, record: logging.LogRecord):
        """Handle error aggregation"""
        if record.levelno >= logging.ERROR:
            with self.lock:
                error_key = f"{record.module}.{record.funcName}.{record.lineno}"
                self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
                
                self.recent_errors.append({
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "location": error_key,
                    "count": self.error_counts[error_key]
                })
                
                # Keep only recent errors
                if len(self.recent_errors) > self.max_errors:
                    self.recent_errors = self.recent_errors[-self.max_errors:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for monitoring"""
        with self.lock:
            return {
                "total_unique_errors": len(self.error_counts),
                "total_error_count": sum(self.error_counts.values()),
                "error_counts": dict(sorted(
                    self.error_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]),  # Top 10 errors
                "recent_errors": self.recent_errors[-10:]  # Last 10 errors
            }


class AsyncFileHandler(logging.FileHandler):
    """Asynchronous file handler for better performance"""
    
    def __init__(self, filename, mode='a', encoding=None, delay=False, buffer_size=8192):
        super().__init__(filename, mode, encoding, delay)
        self.buffer_size = buffer_size
        self.buffer = []
        self.lock = threading.Lock()
    
    def emit(self, record):
        """Buffer log records for async writing"""
        try:
            msg = self.format(record)
            with self.lock:
                self.buffer.append(msg + '\n')
                
                if len(self.buffer) >= self.buffer_size:
                    self._flush_buffer()
        except Exception:
            self.handleError(record)
    
    def _flush_buffer(self):
        """Flush buffer to file"""
        if self.buffer:
            try:
                self.stream.writelines(self.buffer)
                self.stream.flush()
                self.buffer.clear()
            except Exception:
                pass
    
    def close(self):
        """Close handler and flush remaining buffer"""
        with self.lock:
            self._flush_buffer()
        super().close()


def setup_logging():
    """Setup comprehensive logging configuration"""
    
    # Ensure logs directory exists
    settings.logs_path.mkdir(parents=True, exist_ok=True)
    
    # Create error aggregation handler
    error_handler = ErrorAggregationHandler()
    
    # Define log configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": CorrelationFormatter,
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "filters": {
            "performance_filter": {
                "()": PerformanceFilter,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO" if settings.is_production else "DEBUG",
                "formatter": "simple" if settings.is_development else "structured",
                "stream": sys.stdout,
                "filters": ["performance_filter"]
            },
            "file_structured": {
                "()": AsyncFileHandler,
                "filename": str(settings.logs_path / "structured.jsonl"),
                "level": "INFO",
                "formatter": "structured",
                "encoding": "utf-8",
                "filters": ["performance_filter"]
            },
            "file_errors": {
                "()": AsyncFileHandler,
                "filename": str(settings.logs_path / "errors.jsonl"),
                "level": "ERROR",
                "formatter": "structured",
                "encoding": "utf-8"
            },
            "error_aggregation": {
                "()": lambda: error_handler,
                "level": "ERROR"
            }
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file_structured"],
                "propagate": False
            },
            "uvicorn.error": {
                "level": "WARNING",
                "handlers": ["console", "file_errors"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file_structured"],
                "propagate": False
            },
            "fastapi": {
                "level": "INFO",
                "handlers": ["console", "file_structured"],
                "propagate": False
            },
            "app": {
                "level": settings.monitoring.log_level,
                "handlers": ["console", "file_structured", "file_errors", "error_aggregation"],
                "propagate": False
            }
        },
        "root": {
            "level": settings.monitoring.log_level,
            "handlers": ["console", "file_structured", "error_aggregation"]
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Store error handler globally for monitoring
    global_error_handler = error_handler
    
    # Setup log rotation
    setup_log_rotation()
    
    # Log startup message
    logger = logging.getLogger("app.startup")
    logger.info("Netflix-level logging configured successfully", extra={
        "environment": settings.environment,
        "log_level": settings.monitoring.log_level,
        "structured_logging": settings.monitoring.structured_logging
    })


def setup_log_rotation():
    """Setup log file rotation"""
    from logging.handlers import RotatingFileHandler
    import os
    
    try:
        # Check log file sizes and rotate if needed
        for log_file in settings.logs_path.glob("*.jsonl"):
            if log_file.stat().st_size > 50 * 1024 * 1024:  # 50MB
                # Simple rotation - rename current file and create new one
                backup_name = f"{log_file.stem}_{int(time.time())}.jsonl"
                backup_path = log_file.parent / backup_name
                os.rename(log_file, backup_path)
                
                # Keep only last 5 backups
                backups = sorted(
                    log_file.parent.glob(f"{log_file.stem}_*.jsonl"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                for old_backup in backups[5:]:
                    old_backup.unlink()
                    
    except Exception as e:
        print(f"Log rotation failed: {e}")


class LoggerMixin:
    """Mixin to add logging capabilities to classes"""
    
    @property
    def logger(self):
        """Get logger for this class"""
        return logging.getLogger(f"app.{self.__class__.__module__}.{self.__class__.__name__}")


class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, threshold_ms: float = 1000):
        self.operation = operation
        self.logger = logger or logging.getLogger("app.performance")
        self.threshold_ms = threshold_ms
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            
            log_level = logging.WARNING if duration_ms > self.threshold_ms else logging.INFO
            
            self.logger.log(
                log_level,
                f"Operation completed: {self.operation}",
                extra={
                    "operation": self.operation,
                    "duration": duration_ms,
                    "slow_operation": duration_ms > self.threshold_ms
                }
            )


def set_correlation_id(request_id: str):
    """Set correlation ID for current request"""
    request_id_var.set(request_id)


def set_user_context(user_id: str, session_id: Optional[str] = None):
    """Set user context for logging"""
    user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return request_id_var.get(None)


def log_request_start(method: str, path: str, ip: str, user_agent: str = ""):
    """Log request start with context"""
    logger = logging.getLogger("app.requests")
    logger.info(
        f"Request started: {method} {path}",
        extra={
            "request_method": method,
            "request_path": path,
            "client_ip": ip,
            "user_agent": user_agent,
            "request_start": True
        }
    )


def log_request_end(method: str, path: str, status_code: int, duration_ms: float):
    """Log request completion"""
    logger = logging.getLogger("app.requests")
    log_level = logging.WARNING if status_code >= 400 or duration_ms > 1000 else logging.INFO
    
    logger.log(
        log_level,
        f"Request completed: {method} {path} - {status_code}",
        extra={
            "request_method": method,
            "request_path": path,
            "status_code": status_code,
            "duration": duration_ms,
            "request_end": True,
            "slow_request": duration_ms > 1000,
            "error_response": status_code >= 400
        }
    )


def get_error_summary() -> Dict[str, Any]:
    """Get error summary from error aggregation handler"""
    try:
        return global_error_handler.get_error_summary()
    except NameError:
        return {"error": "Error handler not initialized"}


# Global error handler reference
global_error_handler = None
