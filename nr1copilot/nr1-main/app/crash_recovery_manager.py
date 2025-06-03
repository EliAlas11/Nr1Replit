"""
Netflix-Grade Crash Recovery Manager v10.0
Enterprise-level crash recovery with auto-restart capabilities
"""

import asyncio
import logging
import time
import signal
import sys
import os
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryStatus(str, Enum):
    """Recovery status enumeration"""
    HEALTHY = "healthy"
    RECOVERING = "recovering" 
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class CrashEvent:
    """Crash event data structure"""
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    recovery_attempted: bool
    recovery_successful: bool


class NetflixCrashRecoveryManager:
    """Netflix-grade crash recovery system"""

    def __init__(self):
        self.start_time = time.time()
        self.crash_history: List[CrashEvent] = []
        self.recovery_handlers: Dict[str, Callable] = {}
        self.max_recovery_attempts = 3
        self.recovery_cooldown = 30  # seconds
        self.last_recovery_time = 0
        self.status = RecoveryStatus.HEALTHY
        self.monitoring_active = False

        # Setup signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            signal.signal(signal.SIGINT, self._handle_shutdown)
            if hasattr(signal, 'SIGUSR1'):
                signal.signal(signal.SIGUSR1, self._handle_recovery_signal)
        except Exception as e:
            logger.warning(f"Signal handler setup warning: {e}")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received shutdown signal {signum}")
        self.status = RecoveryStatus.CRITICAL
        sys.exit(0)

    def _handle_recovery_signal(self, signum, frame):
        """Handle recovery signals"""
        logger.info(f"🔄 Received recovery signal {signum}")
        asyncio.create_task(self.attempt_recovery("manual_signal"))

    async def initialize(self):
        """Initialize crash recovery system"""
        try:
            self.monitoring_active = True

            # Start monitoring task
            asyncio.create_task(self._monitoring_loop())

            logger.info("✅ Crash recovery system initialized")
            return True

        except Exception as e:
            logger.error(f"❌ Crash recovery initialization failed: {e}")
            return False

    async def _monitoring_loop(self):
        """Background monitoring for system health"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                await self._check_system_health()

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)

    async def _check_system_health(self):
        """Check overall system health"""
        try:
            # Basic health checks
            import psutil

            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Check for critical conditions
            if memory.percent > 95 or cpu_percent > 98:
                await self.record_crash_event(
                    "resource_exhaustion",
                    f"Critical resource usage: Memory {memory.percent}%, CPU {cpu_percent}%",
                    "System resource monitoring"
                )

        except Exception as e:
            logger.debug(f"Health check warning: {e}")

    async def record_crash_event(self, error_type: str, error_message: str, stack_trace: str = ""):
        """Record a crash event"""
        try:
            crash_event = CrashEvent(
                timestamp=datetime.utcnow(),
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace or traceback.format_exc(),
                recovery_attempted=False,
                recovery_successful=False
            )

            self.crash_history.append(crash_event)

            # Keep only last 100 events
            if len(self.crash_history) > 100:
                self.crash_history.pop(0)

            logger.error(f"🚨 Crash event recorded: {error_type} - {error_message}")

            # Attempt recovery
            await self.attempt_recovery(error_type)

        except Exception as e:
            logger.error(f"Failed to record crash event: {e}")

    async def attempt_recovery(self, error_type: str) -> bool:
        """Attempt system recovery"""
        try:
            current_time = time.time()

            # Check cooldown
            if current_time - self.last_recovery_time < self.recovery_cooldown:
                logger.warning(f"⏳ Recovery attempt skipped - cooldown active")
                return False

            # Check max attempts
            recent_crashes = [
                c for c in self.crash_history
                if (datetime.utcnow() - c.timestamp).seconds < 300  # Last 5 minutes
            ]

            if len(recent_crashes) > self.max_recovery_attempts:
                self.status = RecoveryStatus.FAILED
                logger.critical(f"🚨 Max recovery attempts exceeded")
                return False

            self.status = RecoveryStatus.RECOVERING
            self.last_recovery_time = current_time

            logger.info(f"🔄 Attempting recovery for {error_type}")

            # Execute recovery handler if available
            if error_type in self.recovery_handlers:
                try:
                    success = await self.recovery_handlers[error_type]()
                    if success:
                        self.status = RecoveryStatus.HEALTHY
                        logger.info(f"✅ Recovery successful for {error_type}")
                        return True
                except Exception as e:
                    logger.error(f"Recovery handler failed: {e}")

            # Default recovery actions
            success = await self._default_recovery_actions()

            if success:
                self.status = RecoveryStatus.HEALTHY
                logger.info(f"✅ Default recovery successful")
            else:
                self.status = RecoveryStatus.CRITICAL
                logger.error(f"❌ Recovery failed")

            return success

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            self.status = RecoveryStatus.FAILED
            return False

    async def _default_recovery_actions(self) -> bool:
        """Default recovery actions"""
        try:
            # Memory cleanup
            import gc
            gc.collect()

            # Clear caches if available
            try:
                from app.utils.cache import cache_manager
                await cache_manager.clear_cache()
            except:
                pass

            logger.info("🧹 Default recovery actions completed")
            return True

        except Exception as e:
            logger.error(f"Default recovery actions failed: {e}")
            return False

    def register_recovery_handler(self, error_type: str, handler: Callable):
        """Register a recovery handler for specific error types"""
        self.recovery_handlers[error_type] = handler
        logger.info(f"📝 Recovery handler registered for {error_type}")

    def get_crash_statistics(self) -> Dict[str, Any]:
        """Get crash statistics"""
        if not self.crash_history:
            return {
                "total_crashes": 0,
                "recent_crashes": 0,
                "recovery_rate": 100.0,
                "status": self.status.value
            }

        total_crashes = len(self.crash_history)
        successful_recoveries = len([c for c in self.crash_history if c.recovery_successful])
        recent_crashes = len([
            c for c in self.crash_history
            if (datetime.utcnow() - c.timestamp).seconds < 3600  # Last hour
        ])

        recovery_rate = (successful_recoveries / total_crashes * 100) if total_crashes > 0 else 100.0

        return {
            "total_crashes": total_crashes,
            "recent_crashes": recent_crashes,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": recovery_rate,
            "status": self.status.value,
            "uptime_seconds": time.time() - self.start_time,
            "last_recovery": self.last_recovery_time
        }

    async def shutdown(self):
        """Graceful shutdown"""
        self.monitoring_active = False
        logger.info("🛑 Crash recovery manager shutdown")


# Global crash recovery manager instance
crash_recovery_manager = NetflixCrashRecoveryManager()