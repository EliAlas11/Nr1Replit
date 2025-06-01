
"""
World-class logging configuration for production deployment
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

def setup_logging():
    """Setup comprehensive logging for production environment"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    
    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Apply formatters
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(detailed_formatter)
    error_handler.setFormatter(detailed_formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    
    logging.info("🔧 Logging system initialized successfully")
    
    # Setup request ID middleware logging
    logging.getLogger("app.middleware").setLevel(logging.INFO)
    
    # Setup performance logging
    logging.getLogger("app.performance").setLevel(logging.INFO)
    
    # Setup security logging
    logging.getLogger("app.security").setLevel(logging.WARNING)
"""
Logging Configuration
Netflix-level structured logging
"""

import logging
import sys
from typing import Any

def setup_logging():
    """Setup Netflix-level logging configuration"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    try:
        file_handler = logging.FileHandler('logs/app.log')
        file_handler.setFormatter(formatter)
    except:
        file_handler = None
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    
    return root_logger
"""
Netflix-Level Logging Configuration
"""

import logging
import logging.config
import os
from datetime import datetime

def setup_logging():
    """Setup Netflix-level logging configuration"""
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Get current date for log file naming
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "function": "%(funcName)s", "line": %(lineno)d}',
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file_debug": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": f"logs/viralclip_debug_{current_date}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": f"logs/viralclip_error_{current_date}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10
            },
            "file_access": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": f"logs/viralclip_access_{current_date}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 7
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": "INFO",
                "handlers": ["console", "file_debug", "file_error"]
            },
            "app": {
                "level": "DEBUG",
                "handlers": ["console", "file_debug", "file_error"],
                "propagate": False
            },
            "app.main": {
                "level": "INFO",
                "handlers": ["console", "file_access", "file_error"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file_access"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file_access"],
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Set specific log levels for third-party libraries
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("🎬 ViralClip Pro logging system initialized")
"""
Netflix-Level Logging Configuration
Advanced logging with structured output and performance monitoring
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Dict, Any
import json

def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
):
    """Setup Netflix-level logging configuration"""
    
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    json_formatter = JSONFormatter()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "viralclip.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "errors.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # JSON file handler for structured logging
    json_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "structured.jsonl"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    json_handler.setLevel(log_level)
    json_handler.setFormatter(json_formatter)
    root_logger.addHandler(json_handler)
    
    # Performance logger
    perf_logger = logging.getLogger("performance")
    perf_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "performance.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    perf_handler.setFormatter(detailed_formatter)
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    # Security logger
    security_logger = logging.getLogger("security")
    security_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "security.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    security_handler.setFormatter(detailed_formatter)
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.WARNING)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("🎬 ViralClip Pro logging system initialized")
    logger.info(f"Log level: {level}")
    logger.info(f"Log directory: {log_dir}")

class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self):
        super().__init__(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        # Add color to level name
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the record
        formatted = super().format(record)
        
        return formatted

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'exc_info',
                          'exc_text', 'stack_info', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread',
                          'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)

class RequestLogger:
    """Request logging middleware"""
    
    def __init__(self):
        self.logger = logging.getLogger("requests")
    
    async def log_request(
        self,
        method: str,
        url: str,
        status_code: int,
        duration: float,
        user_agent: str = "",
        ip_address: str = "",
        request_id: str = ""
    ):
        """Log HTTP request details"""
        self.logger.info(
            "HTTP Request",
            extra={
                'method': method,
                'url': url,
                'status_code': status_code,
                'duration_ms': duration * 1000,
                'user_agent': user_agent,
                'ip_address': ip_address,
                'request_id': request_id,
                'event_type': 'http_request'
            }
        )

class PerformanceLogger:
    """Performance monitoring logger"""
    
    def __init__(self):
        self.logger = logging.getLogger("performance")
    
    async def log_processing_time(
        self,
        operation: str,
        duration: float,
        success: bool,
        details: Dict[str, Any] = None
    ):
        """Log processing performance"""
        self.logger.info(
            f"Processing: {operation}",
            extra={
                'operation': operation,
                'duration_seconds': duration,
                'success': success,
                'details': details or {},
                'event_type': 'processing_performance'
            }
        )
    
    async def log_memory_usage(self, operation: str, memory_mb: float):
        """Log memory usage"""
        self.logger.info(
            f"Memory usage: {operation}",
            extra={
                'operation': operation,
                'memory_mb': memory_mb,
                'event_type': 'memory_usage'
            }
        )

class SecurityLogger:
    """Security event logger"""
    
    def __init__(self):
        self.logger = logging.getLogger("security")
    
    async def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        ip_address: str = "",
        user_agent: str = ""
    ):
        """Log security events"""
        self.logger.warning(
            f"Security Event: {event_type}",
            extra={
                'event_type': event_type,
                'severity': severity,
                'details': details,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'security_event': True
            }
        )
    
    async def log_rate_limit_exceeded(
        self,
        ip_address: str,
        endpoint: str,
        limit: int
    ):
        """Log rate limit violations"""
        await self.log_security_event(
            "rate_limit_exceeded",
            "medium",
            {
                'endpoint': endpoint,
                'limit': limit,
                'action': 'request_blocked'
            },
            ip_address=ip_address
        )

# Global logger instances
request_logger = RequestLogger()
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()
