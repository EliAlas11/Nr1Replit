
"""
Netflix-Level Logging Configuration
Structured logging with comprehensive monitoring
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import traceback

def get_logger(name: str) -> logging.Logger:
    """Get configured logger instance"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        setup_logging()
    
    return logger

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
            log_dir / "structured.jsonl",
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

class StructuredFormatter(logging.Formatter):
    """Structured console formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Format level with emoji
        level_emojis = {
            'DEBUG': '🔍',
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        
        level_emoji = level_emojis.get(record.levelname, '📝')
        
        # Build message
        parts = [
            f"{timestamp}",
            f"{level_emoji} {record.levelname}",
            f"[{record.name}]",
            record.getMessage()
        ]
        
        message = " ".join(parts)
        
        # Add exception info if present
        if record.exc_info:
            message += f"\n{traceback.format_exception(*record.exc_info)}"
        
        return message

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured file logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, ensure_ascii=False)

# Performance logging
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
                **kwargs
            }
        )

# Application-specific loggers
def get_api_logger() -> logging.Logger:
    """Get API-specific logger"""
    return get_logger("api")

def get_video_logger() -> logging.Logger:
    """Get video processing logger"""
    return get_logger("video_processing")

def get_realtime_logger() -> logging.Logger:
    """Get real-time processing logger"""
    return get_logger("realtime")

def get_security_logger() -> logging.Logger:
    """Get security logger"""
    return get_logger("security")

# Initialize performance logger
performance_logger = PerformanceLogger()
