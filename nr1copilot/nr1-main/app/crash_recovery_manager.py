
"""
Netflix-Grade Crash Recovery Manager for Production Deployment
Handles application-level crash recovery and auto-restart logic
"""

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Dict, Any, Optional
from datetime import datetime

from app.netflix_recovery_system import recovery_system
from app.netflix_health_monitor import health_monitor

logger = logging.getLogger(__name__)

class CrashRecoveryManager:
    """Production-grade crash recovery manager"""
    
    def __init__(self):
        self.restart_count = 0
        self.max_restarts = 5
        self.restart_window = 300  # 5 minutes
        self.restart_history = []
        self.recovery_active = False
        
    async def initialize(self):
        """Initialize crash recovery system"""
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        # Start monitoring
        await recovery_system.start_monitoring()
        logger.info("🛡️ Crash recovery manager initialized")
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self._graceful_shutdown())
    
    async def _graceful_shutdown(self):
        """Perform graceful shutdown"""
        try:
            await recovery_system.stop_monitoring()
            await health_monitor.shutdown()
            logger.info("✅ Graceful shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)
    
    async def handle_critical_failure(self, error: Exception):
        """Handle critical application failures"""
        if self.recovery_active:
            return
            
        self.recovery_active = True
        current_time = time.time()
        
        try:
            # Log the failure
            logger.error(f"🚨 Critical failure detected: {error}")
            
            # Check restart limits
            if self._should_attempt_restart(current_time):
                logger.info("🔄 Attempting automatic recovery...")
                
                # Trigger recovery system
                from app.netflix_recovery_system import FailureType
                await recovery_system.recover_from_failure(FailureType.SERVICE_CRASH)
                
                self.restart_count += 1
                self.restart_history.append(current_time)
                
                logger.info("✅ Recovery attempt completed")
            else:
                logger.error("❌ Maximum restart attempts exceeded - manual intervention required")
                await self._emergency_shutdown()
                
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
        finally:
            self.recovery_active = False
    
    def _should_attempt_restart(self, current_time: float) -> bool:
        """Check if restart should be attempted"""
        # Clean old restart history
        self.restart_history = [
            t for t in self.restart_history 
            if current_time - t < self.restart_window
        ]
        
        return len(self.restart_history) < self.max_restarts
    
    async def _emergency_shutdown(self):
        """Emergency shutdown when recovery fails"""
        logger.error("🆘 Emergency shutdown initiated")
        os._exit(1)

# Global instance
crash_recovery_manager = CrashRecoveryManager()
