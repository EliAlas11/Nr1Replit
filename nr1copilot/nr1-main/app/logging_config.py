
"""
Netflix-Level Logging Configuration
Advanced logging with structured output, correlation tracking, and performance monitoring
"""

import logging
import logging.config
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
import uuid

# Correlation ID context for request tracking
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        correlation_id = correlation_id_var.get('')
        if correlation_id:
            log_entry['correlation_id'] = correlation_id
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add performance context if available
        if hasattr(record, 'duration'):
            record.performance = {
                'duration_ms': getattr(record, 'duration', 0) * 1000,
                'slow_query': getattr(record, 'duration', 0) > 1.0
            }
        return True


class SecurityFilter(logging.Filter):
    """Filter to sanitize sensitive information from logs"""
    
    SENSITIVE_KEYS = {'password', 'token', 'key', 'secret', 'auth', 'credential'}
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Sanitize message
        if hasattr(record, 'args') and record.args:
            record.args = self._sanitize_data(record.args)
        
        # Sanitize extra fields
        for key, value in record.__dict__.items():
            if any(sensitive in key.lower() for sensitive in self.SENSITIVE_KEYS):
                setattr(record, key, '[REDACTED]')
            elif isinstance(value, (dict, list, tuple)):
                setattr(record, key, self._sanitize_data(value))
        
        return True
    
    def _sanitize_data(self, data: Any) -> Any:
        """Recursively sanitize sensitive data"""
        if isinstance(data, dict):
            return {
                k: '[REDACTED]' if any(sensitive in k.lower() for sensitive in self.SENSITIVE_KEYS)
                else self._sanitize_data(v)
                for k, v in data.items()
            }
        elif isinstance(data, (list, tuple)):
            return [self._sanitize_data(item) for item in data]
        return data


def setup_logging() -> logging.Logger:
    """Setup enterprise-grade logging configuration"""
    
    # Ensure log directory exists
    log_dir = "nr1copilot/nr1-main/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'structured': {
                '()': StructuredFormatter,
            },
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            }
        },
        'filters': {
            'performance': {
                '()': PerformanceFilter,
            },
            'security': {
                '()': SecurityFilter,
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': sys.stdout,
                'filters': ['security']
            },
            'structured_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'structured',
                'filename': f'{log_dir}/structured.jsonl',
                'maxBytes': 50 * 1024 * 1024,  # 50MB
                'backupCount': 10,
                'filters': ['performance', 'security']
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': f'{log_dir}/errors.log',
                'maxBytes': 10 * 1024 * 1024,  # 10MB
                'backupCount': 5,
                'filters': ['security']
            },
            'access_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'structured',
                'filename': f'{log_dir}/access.jsonl',
                'maxBytes': 100 * 1024 * 1024,  # 100MB
                'backupCount': 20,
                'filters': ['performance']
            }
        },
        'loggers': {
            # Application loggers
            'app': {
                'level': 'DEBUG',
                'handlers': ['console', 'structured_file', 'error_file'],
                'propagate': False
            },
            'app.services': {
                'level': 'DEBUG',
                'handlers': ['structured_file'],
                'propagate': True
            },
            'app.middleware': {
                'level': 'INFO',
                'handlers': ['structured_file'],
                'propagate': True
            },
            'app.utils': {
                'level': 'INFO',
                'handlers': ['structured_file'],
                'propagate': True
            },
            # Access logs
            'uvicorn.access': {
                'level': 'INFO',
                'handlers': ['access_file'],
                'propagate': False
            },
            # Third-party loggers
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'fastapi': {
                'level': 'INFO',
                'handlers': ['structured_file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': ['console', 'error_file']
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Get main application logger
    logger = logging.getLogger('app')
    logger.info("Logging system initialized with Netflix-level configuration")
    
    return logger


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for request tracking"""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str:
    """Get current correlation ID"""
    return correlation_id_var.get('')


def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs):
    """Log performance metrics with structured data"""
    logger.info(
        f"Performance: {operation} completed",
        extra={
            'operation': operation,
            'duration': duration,
            'performance_metrics': kwargs
        }
    )


def log_security_event(logger: logging.Logger, event_type: str, details: Dict[str, Any]):
    """Log security events with proper sanitization"""
    logger.warning(
        f"Security event: {event_type}",
        extra={
            'security_event': event_type,
            'event_details': details,
            'correlation_id': get_correlation_id()
        }
    )


def log_business_metric(logger: logging.Logger, metric_name: str, value: Any, **metadata):
    """Log business metrics for analytics"""
    logger.info(
        f"Business metric: {metric_name}",
        extra={
            'metric_name': metric_name,
            'metric_value': value,
            'metric_metadata': metadata,
            'metric_timestamp': datetime.utcnow().isoformat()
        }
    )


# Export commonly used functions
__all__ = [
    'setup_logging',
    'set_correlation_id',
    'get_correlation_id',
    'log_performance',
    'log_security_event',
    'log_business_metric'
]
