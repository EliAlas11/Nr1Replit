"""
Enterprise Logging Configuration
Netflix-level structured logging with performance monitoring
"""

import logging
import logging.config
import json
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Dict, Any, Optional
import os

# Context variables for request tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_start_time: ContextVar[Optional[float]] = ContextVar('request_start_time', default=None)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available
        if correlation_id.get():
            log_entry["correlation_id"] = correlation_id.get()

        # Add request timing if available
        if request_start_time.get():
            log_entry["request_duration"] = time.time() - request_start_time.get()

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """Logger for performance metrics"""

    def __init__(self):
        self.logger = logging.getLogger('performance')

    def log_request_metrics(self, method: str, path: str, status_code: int, 
                          duration: float, **kwargs):
        """Log request performance metrics"""
        self.logger.info(
            f"Request completed: {method} {path}",
            extra={
                'extra_fields': {
                    'http_method': method,
                    'http_path': path,
                    'http_status': status_code,
                    'duration_ms': round(duration * 1000, 2),
                    'performance_grade': self._calculate_grade(duration),
                    **kwargs
                }
            }
        )

    def _calculate_grade(self, duration: float) -> str:
        """Calculate performance grade based on response time"""
        if duration < 0.1:
            return "A+"
        elif duration < 0.2:
            return "A"
        elif duration < 0.5:
            return "B"
        elif duration < 1.0:
            return "C"
        else:
            return "D"


def setup_logging() -> logging.Logger:
    """Setup enterprise logging configuration"""

    # Ensure log directories exist
    os.makedirs("nr1copilot/nr1-main/logs", exist_ok=True)

    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": StructuredFormatter,
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "level": "INFO"
            },
            "structured_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "nr1copilot/nr1-main/logs/structured.jsonl",
                "formatter": "structured",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "INFO"
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "nr1copilot/nr1-main/logs/errors.jsonl",
                "formatter": "structured",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": "ERROR"
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "structured_file"],
                "level": "INFO",
                "propagate": False
            },
            "performance": {
                "handlers": ["structured_file"],
                "level": "INFO",
                "propagate": False
            },
            "security": {
                "handlers": ["structured_file", "error_file"],
                "level": "WARNING",
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            }
        }
    }

    logging.config.dictConfig(config)

    # Get main logger
    logger = logging.getLogger(__name__)
    logger.info("Enterprise logging initialized successfully")

    return logger


def set_correlation_id(cid: str = None) -> str:
    """Set correlation ID for request tracking"""
    if cid is None:
        cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def log_request_start(method: str, path: str) -> None:
    """Log request start and set timing"""
    request_start_time.set(time.time())
    logger = logging.getLogger('performance')
    logger.info(f"Request started: {method} {path}")


def log_request_end(method: str, path: str, status_code: int) -> None:
    """Log request completion with metrics"""
    start_time = request_start_time.get()
    if start_time:
        duration = time.time() - start_time
        performance_logger = PerformanceLogger()
        performance_logger.log_request_metrics(method, path, status_code, duration)


# Global performance logger instance
performance_logger = PerformanceLogger()