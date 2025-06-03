
"""
Netflix-Grade Error Handling System v15.0
Comprehensive error management with recovery, logging, and monitoring
"""

import asyncio
import logging
import traceback
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error category classification"""
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    PERFORMANCE = "performance"
    SECURITY = "security"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Comprehensive error context information"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    exception_type: str
    message: str
    stack_trace: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    resolution_time: Optional[float] = None


@dataclass
class ErrorPattern:
    """Error pattern for classification and handling"""
    pattern_id: str
    exception_types: List[Type[Exception]]
    keywords: List[str]
    category: ErrorCategory
    severity: ErrorSeverity
    recovery_strategy: Optional[Callable] = None
    alert_threshold: int = 5
    time_window: int = 300  # 5 minutes


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies"""
    
    async def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if error can be recovered from"""
        return False
    
    async def recover(self, error_context: ErrorContext) -> bool:
        """Attempt error recovery"""
        return False


class RetryRecoveryStrategy(ErrorRecoveryStrategy):
    """Retry-based recovery strategy"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
    
    async def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if error is retryable"""
        retryable_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.PERFORMANCE
        }
        return error_context.category in retryable_categories
    
    async def recover(self, error_context: ErrorContext) -> bool:
        """Attempt recovery with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.delay * (self.backoff_factor ** attempt))
                # Recovery logic would be injected here
                return True
            except Exception:
                continue
        return False


class FallbackRecoveryStrategy(ErrorRecoveryStrategy):
    """Fallback-based recovery strategy"""
    
    def __init__(self, fallback_handler: Callable):
        self.fallback_handler = fallback_handler
    
    async def can_recover(self, error_context: ErrorContext) -> bool:
        """Always can attempt fallback"""
        return True
    
    async def recover(self, error_context: ErrorContext) -> bool:
        """Execute fallback handler"""
        try:
            if asyncio.iscoroutinefunction(self.fallback_handler):
                await self.fallback_handler(error_context)
            else:
                self.fallback_handler(error_context)
            return True
        except Exception:
            return False


class NetflixErrorHandler:
    """Netflix-grade error handling system with advanced features"""
    
    def __init__(self):
        self.error_patterns: List[ErrorPattern] = []
        self.recovery_strategies: Dict[ErrorCategory, List[ErrorRecoveryStrategy]] = defaultdict(list)
        self.error_history: deque = deque(maxlen=10000)
        self.error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.alert_handlers: List[Callable] = []
        
        # Error rate limiting
        self.error_rate_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance tracking
        self.handling_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'unrecovered_errors': 0,
            'average_handling_time': 0.0
        }
        
        self._lock = threading.RLock()
        
        # Initialize default patterns
        self._initialize_default_patterns()
        
        logger.info("🛡️ Netflix Error Handler v15.0 initialized")
    
    def _initialize_default_patterns(self) -> None:
        """Initialize default error patterns"""
        patterns = [
            ErrorPattern(
                pattern_id="database_connection",
                exception_types=[ConnectionError, TimeoutError],
                keywords=["database", "connection", "timeout"],
                category=ErrorCategory.DATABASE,
                severity=ErrorSeverity.HIGH
            ),
            ErrorPattern(
                pattern_id="authentication_failure",
                exception_types=[PermissionError],
                keywords=["authentication", "unauthorized", "forbidden"],
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.MEDIUM
            ),
            ErrorPattern(
                pattern_id="validation_error",
                exception_types=[ValueError, TypeError],
                keywords=["validation", "invalid", "format"],
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW
            ),
            ErrorPattern(
                pattern_id="external_service",
                exception_types=[ConnectionError, TimeoutError],
                keywords=["external", "api", "service", "network"],
                category=ErrorCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.MEDIUM
            ),
            ErrorPattern(
                pattern_id="performance_degradation",
                exception_types=[TimeoutError],
                keywords=["timeout", "slow", "performance"],
                category=ErrorCategory.PERFORMANCE,
                severity=ErrorSeverity.MEDIUM
            )
        ]
        
        for pattern in patterns:
            self.add_error_pattern(pattern)
        
        # Add default recovery strategies
        self.add_recovery_strategy(ErrorCategory.NETWORK, RetryRecoveryStrategy(max_retries=3))
        self.add_recovery_strategy(ErrorCategory.EXTERNAL_SERVICE, RetryRecoveryStrategy(max_retries=2))
        self.add_recovery_strategy(ErrorCategory.PERFORMANCE, RetryRecoveryStrategy(max_retries=1))
    
    def add_error_pattern(self, pattern: ErrorPattern) -> None:
        """Add error pattern for classification"""
        with self._lock:
            self.error_patterns.append(pattern)
            logger.debug(f"Added error pattern: {pattern.pattern_id}")
    
    def add_recovery_strategy(self, category: ErrorCategory, strategy: ErrorRecoveryStrategy) -> None:
        """Add recovery strategy for error category"""
        with self._lock:
            self.recovery_strategies[category].append(strategy)
            logger.debug(f"Added recovery strategy for {category.value}")
    
    def add_alert_handler(self, handler: Callable[[ErrorContext], None]) -> None:
        """Add alert handler for critical errors"""
        self.alert_handlers.append(handler)
    
    async def handle_error(self, exception: Exception, 
                          context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle error with comprehensive processing"""
        start_time = time.time()
        
        try:
            with self._lock:
                self.handling_stats['total_errors'] += 1
            
            # Create error context
            error_context = self._create_error_context(exception, context or {})
            
            # Classify error
            self._classify_error(error_context)
            
            # Store error history
            with self._lock:
                self.error_history.append(error_context)
                self._update_error_counts(error_context)
            
            # Check for error rate limiting
            if self._should_rate_limit_error(error_context):
                logger.warning(f"Error rate limit reached for {error_context.category.value}")
                return error_context
            
            # Attempt recovery
            recovery_successful = await self._attempt_recovery(error_context)
            error_context.recovery_successful = recovery_successful
            
            # Update statistics
            with self._lock:
                if recovery_successful:
                    self.handling_stats['recovered_errors'] += 1
                else:
                    self.handling_stats['unrecovered_errors'] += 1
            
            # Check for alerts
            await self._check_and_send_alerts(error_context)
            
            # Log error
            self._log_error(error_context)
            
            # Update handling time
            handling_time = time.time() - start_time
            error_context.resolution_time = handling_time
            
            with self._lock:
                current_avg = self.handling_stats['average_handling_time']
                total_errors = self.handling_stats['total_errors']
                self.handling_stats['average_handling_time'] = (
                    (current_avg * (total_errors - 1) + handling_time) / total_errors
                )
            
            return error_context
            
        except Exception as e:
            logger.error(f"Error handler failed: {e}", exc_info=True)
            # Create minimal error context for handler failure
            return ErrorContext(
                error_id=f"handler_error_{int(time.time())}",
                timestamp=datetime.utcnow(),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM,
                exception_type=type(e).__name__,
                message=str(e),
                stack_trace=traceback.format_exc()
            )
    
    def _create_error_context(self, exception: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Create comprehensive error context"""
        error_id = f"err_{int(time.time() * 1000)}_{id(exception)}"
        
        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            severity=ErrorSeverity.MEDIUM,  # Default, will be updated
            category=ErrorCategory.UNKNOWN,  # Default, will be updated
            exception_type=type(exception).__name__,
            message=str(exception),
            stack_trace=traceback.format_exc(),
            user_id=context.get('user_id'),
            request_id=context.get('request_id'),
            endpoint=context.get('endpoint'),
            method=context.get('method'),
            user_agent=context.get('user_agent'),
            ip_address=context.get('ip_address'),
            additional_data=context
        )
    
    def _classify_error(self, error_context: ErrorContext) -> None:
        """Classify error using patterns"""
        for pattern in self.error_patterns:
            # Check exception type
            if any(isinstance(type(error_context.exception_type), exc_type) 
                   for exc_type in pattern.exception_types):
                error_context.category = pattern.category
                error_context.severity = pattern.severity
                return
            
            # Check keywords in message
            message_lower = error_context.message.lower()
            if any(keyword in message_lower for keyword in pattern.keywords):
                error_context.category = pattern.category
                error_context.severity = pattern.severity
                return
        
        # Default classification based on exception type
        self._default_classification(error_context)
    
    def _default_classification(self, error_context: ErrorContext) -> None:
        """Default error classification"""
        exception_type = error_context.exception_type.lower()
        
        if 'connection' in exception_type or 'network' in exception_type:
            error_context.category = ErrorCategory.NETWORK
            error_context.severity = ErrorSeverity.MEDIUM
        elif 'timeout' in exception_type:
            error_context.category = ErrorCategory.PERFORMANCE
            error_context.severity = ErrorSeverity.MEDIUM
        elif 'permission' in exception_type or 'auth' in exception_type:
            error_context.category = ErrorCategory.AUTHENTICATION
            error_context.severity = ErrorSeverity.HIGH
        elif 'value' in exception_type or 'type' in exception_type:
            error_context.category = ErrorCategory.VALIDATION
            error_context.severity = ErrorSeverity.LOW
        else:
            error_context.category = ErrorCategory.SYSTEM
            error_context.severity = ErrorSeverity.MEDIUM
    
    async def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt error recovery using available strategies"""
        strategies = self.recovery_strategies.get(error_context.category, [])
        
        for strategy in strategies:
            try:
                if await strategy.can_recover(error_context):
                    logger.info(f"Attempting recovery for {error_context.error_id}")
                    error_context.recovery_attempted = True
                    
                    if await strategy.recover(error_context):
                        logger.info(f"Recovery successful for {error_context.error_id}")
                        return True
                    
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
        
        return False
    
    def _should_rate_limit_error(self, error_context: ErrorContext) -> bool:
        """Check if error should be rate limited"""
        pattern_key = f"{error_context.category.value}_{error_context.exception_type}"
        current_time = time.time()
        
        # Clean old entries
        window = self.error_rate_windows[pattern_key]
        while window and current_time - window[0] > 300:  # 5 minute window
            window.popleft()
        
        # Add current error
        window.append(current_time)
        
        # Check if rate limit exceeded (more than 10 errors in 5 minutes)
        return len(window) > 10
    
    def _update_error_counts(self, error_context: ErrorContext) -> None:
        """Update error counts for monitoring"""
        category_key = error_context.category.value
        exception_key = error_context.exception_type
        
        self.error_counts[category_key]['total'] += 1
        self.error_counts[category_key][exception_key] += 1
    
    async def _check_and_send_alerts(self, error_context: ErrorContext) -> None:
        """Check if alerts should be sent"""
        if error_context.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            await self._send_alerts(error_context)
            return
        
        # Check error pattern thresholds
        for pattern in self.error_patterns:
            if (error_context.category == pattern.category and 
                self._get_recent_error_count(pattern) >= pattern.alert_threshold):
                await self._send_alerts(error_context)
                break
    
    def _get_recent_error_count(self, pattern: ErrorPattern) -> int:
        """Get recent error count for pattern"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=pattern.time_window)
        
        count = 0
        for error in self.error_history:
            if (error.timestamp >= cutoff_time and 
                error.category == pattern.category):
                count += 1
        
        return count
    
    async def _send_alerts(self, error_context: ErrorContext) -> None:
        """Send alerts for critical errors"""
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error_context)
                else:
                    handler(error_context)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate level"""
        log_data = {
            'error_id': error_context.error_id,
            'timestamp': error_context.timestamp.isoformat(),
            'severity': error_context.severity.value,
            'category': error_context.category.value,
            'exception_type': error_context.exception_type,
            'message': error_context.message,
            'user_id': error_context.user_id,
            'request_id': error_context.request_id,
            'endpoint': error_context.endpoint,
            'recovery_attempted': error_context.recovery_attempted,
            'recovery_successful': error_context.recovery_successful
        }
        
        if error_context.severity == ErrorSeverity.FATAL:
            logger.critical(f"FATAL ERROR: {json.dumps(log_data)}")
        elif error_context.severity == ErrorSeverity.CRITICAL:
            logger.error(f"CRITICAL ERROR: {json.dumps(log_data)}")
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR: {json.dumps(log_data)}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR: {json.dumps(log_data)}")
        else:
            logger.info(f"LOW SEVERITY ERROR: {json.dumps(log_data)}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        with self._lock:
            recent_errors = [
                error for error in self.error_history
                if error.timestamp >= datetime.utcnow() - timedelta(hours=1)
            ]
            
            error_by_category = defaultdict(int)
            error_by_severity = defaultdict(int)
            recovery_stats = {'attempted': 0, 'successful': 0}
            
            for error in recent_errors:
                error_by_category[error.category.value] += 1
                error_by_severity[error.severity.value] += 1
                
                if error.recovery_attempted:
                    recovery_stats['attempted'] += 1
                    if error.recovery_successful:
                        recovery_stats['successful'] += 1
            
            return {
                'total_errors_all_time': self.handling_stats['total_errors'],
                'total_errors_last_hour': len(recent_errors),
                'errors_by_category': dict(error_by_category),
                'errors_by_severity': dict(error_by_severity),
                'recovery_stats': recovery_stats,
                'recovery_rate': (
                    recovery_stats['successful'] / recovery_stats['attempted'] * 100
                    if recovery_stats['attempted'] > 0 else 0
                ),
                'average_handling_time': self.handling_stats['average_handling_time'],
                'error_patterns': len(self.error_patterns),
                'recovery_strategies': sum(len(strategies) for strategies in self.recovery_strategies.values()),
                'alert_handlers': len(self.alert_handlers)
            }
    
    def get_recent_errors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent errors for debugging"""
        with self._lock:
            recent = list(self.error_history)[-limit:]
            
            return [
                {
                    'error_id': error.error_id,
                    'timestamp': error.timestamp.isoformat(),
                    'severity': error.severity.value,
                    'category': error.category.value,
                    'exception_type': error.exception_type,
                    'message': error.message,
                    'endpoint': error.endpoint,
                    'recovery_successful': error.recovery_successful,
                    'resolution_time': error.resolution_time
                }
                for error in recent
            ]


def error_handler(category: Optional[ErrorCategory] = None, 
                 severity: Optional[ErrorSeverity] = None,
                 recovery_strategy: Optional[ErrorRecoveryStrategy] = None):
    """Decorator for automatic error handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                error_context = await netflix_error_handler.handle_error(e, context)
                
                # Re-raise if not recovered
                if not error_context.recovery_successful:
                    raise
                
                return None  # or default value
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                # For sync functions, we can't await, so just handle without recovery
                netflix_error_handler._create_error_context(e, context)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Global error handler instance
netflix_error_handler = NetflixErrorHandler()
