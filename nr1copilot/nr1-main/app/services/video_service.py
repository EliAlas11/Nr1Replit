
"""
ViralClip Pro v6.0 - Netflix-Level Video Service
Enterprise video processing with advanced performance, monitoring, and scalability
"""

import asyncio
import logging
import time
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import json
import aiofiles
import weakref
from concurrent.futures import ThreadPoolExecutor
import psutil
from enum import Enum
import traceback
from contextlib import asynccontextmanager

from fastapi import UploadFile, HTTPException

# ================================
# Core Data Models
# ================================

class UploadStatus(Enum):
    """Upload status enumeration for type safety"""
    INITIALIZING = "initializing"
    QUEUED = "queued"
    UPLOADING = "uploading"
    PAUSED = "paused"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingStage(Enum):
    """Processing stage enumeration"""
    VALIDATION = "validation"
    ASSEMBLY = "assembly"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    ENCODING = "encoding"
    FINALIZATION = "finalization"


@dataclass
class UploadMetrics:
    """Comprehensive upload metrics for monitoring"""
    total_uploads: int = 0
    successful_uploads: int = 0
    failed_uploads: int = 0
    cancelled_uploads: int = 0
    total_bytes_processed: int = 0
    average_upload_speed: float = 0.0
    average_processing_time: float = 0.0
    peak_concurrent_uploads: int = 0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class UploadSession:
    """Enhanced upload session with comprehensive tracking"""
    upload_id: str
    filename: str
    file_size: int
    total_chunks: int
    
    # Status tracking
    status: UploadStatus = UploadStatus.INITIALIZING
    received_chunks: int = 0
    processing_stage: Optional[ProcessingStage] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    # User context
    user_info: Dict[str, Any] = field(default_factory=dict)
    client_info: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    chunk_timings: List[float] = field(default_factory=list)
    retry_counts: Dict[int, int] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    file_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    chunk_size: int = 5 * 1024 * 1024
    max_retries: int = 3
    timeout_seconds: int = 3600

    def calculate_progress(self) -> float:
        """Calculate upload progress percentage"""
        if self.total_chunks == 0:
            return 0.0
        return (self.received_chunks / self.total_chunks) * 100

    def calculate_upload_speed(self) -> float:
        """Calculate current upload speed in bytes per second"""
        if not self.chunk_timings or not self.started_at:
            return 0.0
        
        elapsed_time = (datetime.utcnow() - self.started_at).total_seconds()
        if elapsed_time <= 0:
            return 0.0
        
        uploaded_bytes = self.received_chunks * self.chunk_size
        return uploaded_bytes / elapsed_time

    def estimate_time_remaining(self) -> float:
        """Estimate time remaining in seconds"""
        speed = self.calculate_upload_speed()
        if speed <= 0:
            return float('inf')
        
        remaining_bytes = (self.total_chunks - self.received_chunks) * self.chunk_size
        return remaining_bytes / speed

    def is_expired(self) -> bool:
        """Check if session has expired"""
        elapsed = (datetime.utcnow() - self.last_activity).total_seconds()
        return elapsed > self.timeout_seconds

    def add_error(self, error: Exception, context: str = ""):
        """Add error to history with context"""
        self.error_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(error),
            "context": context,
            "traceback": traceback.format_exc()
        })


# ================================
# Performance Monitor
# ================================

class PerformanceMonitor:
    """Advanced performance monitoring for upload operations"""
    
    def __init__(self):
        self.metrics = UploadMetrics()
        self.start_time = time.time()
        self.metric_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "error_rate": 0.1,  # 10%
            "avg_processing_time": 300,  # 5 minutes
            "memory_usage_mb": 1024,  # 1GB
            "disk_usage_percent": 90  # 90%
        }

    def record_upload_start(self):
        """Record upload start"""
        self.metrics.total_uploads += 1

    def record_upload_success(self, processing_time: float, file_size: int):
        """Record successful upload"""
        self.metrics.successful_uploads += 1
        self.metrics.total_bytes_processed += file_size
        self._update_average_processing_time(processing_time)
        self._update_error_rate()

    def record_upload_failure(self):
        """Record failed upload"""
        self.metrics.failed_uploads += 1
        self._update_error_rate()

    def record_upload_cancellation(self):
        """Record cancelled upload"""
        self.metrics.cancelled_uploads += 1

    def update_concurrent_uploads(self, count: int):
        """Update concurrent upload count"""
        self.metrics.peak_concurrent_uploads = max(
            self.metrics.peak_concurrent_uploads, count
        )

    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time"""
        total_successful = self.metrics.successful_uploads
        if total_successful == 1:
            self.metrics.average_processing_time = processing_time
        else:
            current_avg = self.metrics.average_processing_time
            self.metrics.average_processing_time = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )

    def _update_error_rate(self):
        """Update error rate calculation"""
        total = self.metrics.total_uploads
        if total > 0:
            self.metrics.error_rate = self.metrics.failed_uploads / total

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_usage_mb": memory_info.rss / (1024 * 1024),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": dict(psutil.disk_usage('/')),
            "network_io": dict(psutil.net_io_counters()),
            "timestamp": datetime.utcnow().isoformat()
        }

    def check_health(self) -> Dict[str, Any]:
        """Perform health check against thresholds"""
        system_metrics = self.get_system_metrics()
        health_status = {
            "healthy": True,
            "alerts": [],
            "metrics": system_metrics
        }

        # Check error rate
        if self.metrics.error_rate > self.alert_thresholds["error_rate"]:
            health_status["healthy"] = False
            health_status["alerts"].append({
                "type": "high_error_rate",
                "current": self.metrics.error_rate,
                "threshold": self.alert_thresholds["error_rate"]
            })

        # Check memory usage
        if system_metrics["memory_usage_mb"] > self.alert_thresholds["memory_usage_mb"]:
            health_status["healthy"] = False
            health_status["alerts"].append({
                "type": "high_memory_usage",
                "current": system_metrics["memory_usage_mb"],
                "threshold": self.alert_thresholds["memory_usage_mb"]
            })

        return health_status

    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external monitoring"""
        self.metrics.uptime_seconds = time.time() - self.start_time
        
        return {
            "metrics": self.metrics.__dict__,
            "system": self.get_system_metrics(),
            "history": self.metric_history[-100:],  # Last 100 entries
            "health": self.check_health()
        }


# ================================
# Circuit Breaker Pattern
# ================================

class CircuitBreaker:
    """Circuit breaker for handling service failures gracefully"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    @asynccontextmanager
    async def call(self):
        """Context manager for circuit breaker calls"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            yield
            if self.state == "HALF_OPEN":
                self._reset()
        except Exception as e:
            self._record_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _record_failure(self):
        """Record a failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def _reset(self):
        """Reset the circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"


# ================================
# Upload Session Manager
# ================================

class UploadSessionManager:
    """Advanced session management with lifecycle handling"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.sessions: Dict[str, UploadSession] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.performance_monitor = performance_monitor
        self.cleanup_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(f"{__name__}.SessionManager")

    async def start(self):
        """Start session manager background tasks"""
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

    async def stop(self):
        """Stop session manager and cleanup"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    async def create_session(
        self,
        upload_id: str,
        filename: str,
        file_size: int,
        total_chunks: int,
        user_info: Dict[str, Any],
        client_info: Dict[str, Any] = None
    ) -> UploadSession:
        """Create new upload session with validation"""
        
        if upload_id in self.sessions:
            raise ValueError(f"Session {upload_id} already exists")

        session = UploadSession(
            upload_id=upload_id,
            filename=filename,
            file_size=file_size,
            total_chunks=total_chunks,
            user_info=user_info or {},
            client_info=client_info or {}
        )

        # Create session lock
        self.session_locks[upload_id] = asyncio.Lock()
        
        # Store session
        self.sessions[upload_id] = session
        
        # Record metrics
        self.performance_monitor.record_upload_start()
        self.performance_monitor.update_concurrent_uploads(len(self.sessions))
        
        self.logger.info(f"Created upload session: {upload_id}")
        return session

    async def get_session(self, upload_id: str) -> Optional[UploadSession]:
        """Get session by ID"""
        return self.sessions.get(upload_id)

    async def update_session(
        self,
        upload_id: str,
        update_func: Callable[[UploadSession], None]
    ) -> bool:
        """Thread-safe session update"""
        session = self.sessions.get(upload_id)
        if not session:
            return False

        lock = self.session_locks.get(upload_id)
        if not lock:
            return False

        async with lock:
            # Refresh session reference
            session = self.sessions.get(upload_id)
            if session:
                update_func(session)
                session.last_activity = datetime.utcnow()
                return True
        
        return False

    async def complete_session(self, upload_id: str, success: bool = True):
        """Mark session as completed"""
        session = self.sessions.get(upload_id)
        if not session:
            return

        # Calculate processing time
        if session.started_at:
            processing_time = (datetime.utcnow() - session.started_at).total_seconds()
        else:
            processing_time = 0

        # Update session
        def update_session(s: UploadSession):
            s.completed_at = datetime.utcnow()
            s.status = UploadStatus.COMPLETED if success else UploadStatus.FAILED

        await self.update_session(upload_id, update_session)

        # Record metrics
        if success:
            self.performance_monitor.record_upload_success(processing_time, session.file_size)
        else:
            self.performance_monitor.record_upload_failure()

        self.logger.info(f"Completed upload session: {upload_id} (success: {success})")

    async def remove_session(self, upload_id: str):
        """Remove session and cleanup resources"""
        if upload_id in self.sessions:
            session = self.sessions.pop(upload_id)
            
            # Cleanup session lock
            if upload_id in self.session_locks:
                del self.session_locks[upload_id]
            
            # Update metrics
            self.performance_monitor.update_concurrent_uploads(len(self.sessions))
            
            self.logger.info(f"Removed upload session: {upload_id}")

    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions"""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_sessions = []

                for upload_id, session in self.sessions.items():
                    if session.is_expired():
                        expired_sessions.append(upload_id)

                for upload_id in expired_sessions:
                    self.logger.info(f"Cleaning up expired session: {upload_id}")
                    await self.remove_session(upload_id)

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def get_active_sessions(self) -> List[UploadSession]:
        """Get all active sessions"""
        return list(self.sessions.values())

    def get_session_count(self) -> int:
        """Get current session count"""
        return len(self.sessions)


# ================================
# File Validator
# ================================

class FileValidator:
    """Advanced file validation with comprehensive checks"""
    
    def __init__(self):
        self.config = {
            "max_file_size": 500 * 1024 * 1024,  # 500MB
            "supported_formats": {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.3gp', '.mp3', '.wav'},
            "max_filename_length": 255,
            "blocked_patterns": [r'[<>:"/\\|?*]'],  # Invalid filename characters
            "virus_scan_enabled": False  # Could integrate with antivirus
        }
        self.logger = logging.getLogger(f"{__name__}.FileValidator")

    async def validate_upload_request(
        self,
        filename: str,
        file_size: int,
        total_chunks: int,
        user_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive upload request validation"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }

        # File extension validation
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.config["supported_formats"]:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "code": "UNSUPPORTED_FORMAT",
                "message": f"Unsupported format {file_ext}. Supported: {', '.join(self.config['supported_formats'])}"
            })

        # File size validation
        if file_size > self.config["max_file_size"]:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "code": "FILE_TOO_LARGE",
                "message": f"File too large ({self._format_size(file_size)}). Maximum: {self._format_size(self.config['max_file_size'])}"
            })

        if file_size <= 0:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "code": "INVALID_FILE_SIZE",
                "message": "Invalid file size"
            })

        # Filename validation
        if len(filename) > self.config["max_filename_length"]:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "code": "FILENAME_TOO_LONG",
                "message": f"Filename too long (max {self.config['max_filename_length']} characters)"
            })

        # Check for invalid characters
        import re
        for pattern in self.config["blocked_patterns"]:
            if re.search(pattern, filename):
                validation_result["valid"] = False
                validation_result["errors"].append({
                    "code": "INVALID_FILENAME",
                    "message": "Filename contains invalid characters"
                })
                break

        # Chunk validation
        if total_chunks <= 0 or total_chunks > 10000:  # Reasonable limit
            validation_result["valid"] = False
            validation_result["errors"].append({
                "code": "INVALID_CHUNK_COUNT",
                "message": "Invalid chunk count"
            })

        # User-specific validation
        await self._validate_user_limits(user_info, file_size, validation_result)

        # Generate metadata
        if validation_result["valid"]:
            validation_result["metadata"] = {
                "estimated_processing_time": self._estimate_processing_time(file_size, file_ext),
                "estimated_duration": self._estimate_video_duration(file_size),
                "file_category": self._categorize_file(file_ext),
                "quality_recommendations": self._get_quality_recommendations(file_size)
            }

        return validation_result

    async def _validate_user_limits(
        self,
        user_info: Dict[str, Any],
        file_size: int,
        validation_result: Dict[str, Any]
    ):
        """Validate user-specific limits"""
        user_tier = user_info.get("tier", "free")
        
        # Tier-based limits
        tier_limits = {
            "free": {"max_size": 100 * 1024 * 1024, "uploads_per_hour": 5},
            "standard": {"max_size": 250 * 1024 * 1024, "uploads_per_hour": 20},
            "premium": {"max_size": 500 * 1024 * 1024, "uploads_per_hour": 100}
        }

        limits = tier_limits.get(user_tier, tier_limits["free"])
        
        if file_size > limits["max_size"]:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "code": "TIER_LIMIT_EXCEEDED",
                "message": f"File size exceeds {user_tier} tier limit ({self._format_size(limits['max_size'])})"
            })

        # Rate limiting would be implemented here
        # For now, we'll just add a warning for free tier users
        if user_tier == "free" and file_size > 50 * 1024 * 1024:
            validation_result["warnings"].append({
                "code": "LARGE_FILE_FREE_TIER",
                "message": "Large files may process slower on free tier"
            })

    def _estimate_processing_time(self, file_size: int, file_ext: str) -> float:
        """Estimate processing time based on file characteristics"""
        # Base processing rate: 10MB per second
        base_rate = 10 * 1024 * 1024
        
        # Format complexity multipliers
        complexity_multipliers = {
            '.mp4': 1.0,
            '.mov': 1.2,
            '.avi': 1.5,
            '.mkv': 1.3,
            '.webm': 1.1,
            '.m4v': 1.0,
            '.3gp': 0.8,
            '.mp3': 0.3,
            '.wav': 0.3
        }
        
        multiplier = complexity_multipliers.get(file_ext, 1.5)
        return (file_size / base_rate) * multiplier

    def _estimate_video_duration(self, file_size: int) -> float:
        """Estimate video duration from file size"""
        # Rough estimation: 1MB ≈ 1 second for typical video
        return max(1, file_size / (1024 * 1024))

    def _categorize_file(self, file_ext: str) -> str:
        """Categorize file by extension"""
        video_formats = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.3gp'}
        audio_formats = {'.mp3', '.wav'}
        
        if file_ext in video_formats:
            return "video"
        elif file_ext in audio_formats:
            return "audio"
        else:
            return "unknown"

    def _get_quality_recommendations(self, file_size: int) -> List[str]:
        """Get quality recommendations based on file size"""
        recommendations = []
        
        # Size-based recommendations
        if file_size < 10 * 1024 * 1024:  # < 10MB
            recommendations.append("Consider higher resolution for better quality")
        elif file_size > 200 * 1024 * 1024:  # > 200MB
            recommendations.append("Large file - excellent for high-quality content")
            
        return recommendations

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        units = ['B', 'KB', 'MB', 'GB']
        size = float(size_bytes)
        unit_index = 0
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
            
        return f"{size:.1f} {units[unit_index]}"


# ================================
# Main Video Service
# ================================

class NetflixLevelVideoService:
    """Netflix-level video service with enterprise features"""

    def __init__(self):
        # Core components
        self.performance_monitor = PerformanceMonitor()
        self.session_manager = UploadSessionManager(self.performance_monitor)
        self.file_validator = FileValidator()
        self.circuit_breaker = CircuitBreaker()
        
        # Configuration
        self.config = {
            "chunk_size": 5 * 1024 * 1024,  # 5MB
            "max_concurrent_uploads": 10,
            "max_retries_per_chunk": 3,
            "session_timeout": 3600,  # 1 hour
            "enable_detailed_logging": True
        }
        
        # Storage paths
        self.storage_paths = {
            "uploads": Path("nr1copilot/nr1-main/uploads"),
            "temp": Path("nr1copilot/nr1-main/temp"),
            "output": Path("nr1copilot/nr1-main/output")
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize storage directories"""
        for path_type, path in self.storage_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Initialized {path_type} storage: {path}")

    async def startup(self):
        """Startup service with all components"""
        try:
            # Start session manager
            await self.session_manager.start()
            
            # Start background monitoring
            monitor_task = asyncio.create_task(self._background_monitoring())
            self.background_tasks.append(monitor_task)
            
            self.logger.info("🚀 Netflix-level video service started successfully")
            
        except Exception as e:
            self.logger.error(f"Service startup failed: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown"""
        try:
            self.logger.info("🔄 Starting video service shutdown...")
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Stop session manager
            await self.session_manager.stop()
            
            self.logger.info("✅ Video service shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def validate_upload_request(
        self,
        filename: str,
        file_size: int,
        total_chunks: int,
        user_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate upload request with comprehensive checks"""
        
        async with self.circuit_breaker.call():
            return await self.file_validator.validate_upload_request(
                filename, file_size, total_chunks, user_info
            )

    async def create_upload_session(
        self,
        upload_id: str,
        filename: str,
        file_size: int,
        total_chunks: int,
        user_info: Dict[str, Any],
        client_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create upload session with enhanced tracking"""
        
        try:
            # Validate request first
            validation = await self.validate_upload_request(
                filename, file_size, total_chunks, user_info
            )
            
            if not validation["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Upload validation failed",
                        "errors": validation["errors"]
                    }
                )

            # Create session
            session = await self.session_manager.create_session(
                upload_id, filename, file_size, total_chunks, user_info, client_info
            )
            
            # Create upload directory
            upload_dir = self.storage_paths["temp"] / upload_id
            upload_dir.mkdir(exist_ok=True)
            
            # Prepare response
            response = {
                "success": True,
                "upload_id": upload_id,
                "session_config": {
                    "chunk_size": self.config["chunk_size"],
                    "max_retries": self.config["max_retries_per_chunk"],
                    "timeout": self.config["session_timeout"]
                },
                "metadata": validation.get("metadata", {}),
                "endpoints": {
                    "chunk_upload": f"/api/v6/upload/chunk",
                    "status": f"/api/v6/upload/status/{upload_id}",
                    "cancel": f"/api/v6/upload/cancel/{upload_id}"
                }
            }
            
            self.logger.info(f"Created upload session: {upload_id}")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to create upload session: {e}")
            raise HTTPException(status_code=500, detail="Session creation failed")

    async def process_chunk(
        self,
        file: UploadFile,
        upload_id: str,
        chunk_index: int,
        total_chunks: int,
        chunk_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process chunk with Netflix-level reliability"""
        
        start_time = time.time()
        
        try:
            # Get session
            session = await self.session_manager.get_session(upload_id)
            if not session:
                raise HTTPException(status_code=404, detail="Upload session not found")

            # Validate chunk
            if chunk_index < 0 or chunk_index >= total_chunks:
                raise HTTPException(status_code=400, detail="Invalid chunk index")

            # Check for duplicate chunk
            chunk_path = self.storage_paths["temp"] / upload_id / f"chunk_{chunk_index:04d}"
            if chunk_path.exists():
                return await self._handle_duplicate_chunk(session, chunk_index, chunk_path)

            # Process chunk
            result = await self._process_new_chunk(
                session, file, chunk_index, chunk_path, chunk_hash
            )
            
            # Update session
            processing_time = time.time() - start_time
            await self._update_session_progress(session, chunk_index, processing_time)
            
            # Check if upload is complete
            if session.received_chunks >= session.total_chunks:
                await self._finalize_upload(session)
                result["upload_complete"] = True
                result["message"] = "Upload completed successfully"

            return result
            
        except HTTPException:
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Chunk processing failed: {e}")
            
            # Record error in session
            session = await self.session_manager.get_session(upload_id)
            if session:
                session.add_error(e, f"chunk_{chunk_index}")
            
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Chunk processing failed",
                    "error": str(e),
                    "processing_time": processing_time
                }
            )

    async def _handle_duplicate_chunk(
        self,
        session: UploadSession,
        chunk_index: int,
        chunk_path: Path
    ) -> Dict[str, Any]:
        """Handle duplicate chunk upload"""
        
        existing_size = chunk_path.stat().st_size
        
        return {
            "success": True,
            "upload_id": session.upload_id,
            "chunk_index": chunk_index,
            "progress": session.calculate_progress(),
            "duplicate": True,
            "chunk_size": existing_size,
            "message": "Chunk already uploaded"
        }

    async def _process_new_chunk(
        self,
        session: UploadSession,
        file: UploadFile,
        chunk_index: int,
        chunk_path: Path,
        chunk_hash: Optional[str]
    ) -> Dict[str, Any]:
        """Process new chunk upload"""
        
        # Read chunk data
        chunk_data = await file.read()
        if not chunk_data:
            raise ValueError("Empty chunk data")

        # Verify size limits
        if len(chunk_data) > self.config["chunk_size"] * 2:
            raise ValueError(f"Chunk too large: {len(chunk_data)} bytes")

        # Verify hash if provided
        if chunk_hash:
            calculated_hash = hashlib.md5(chunk_data).hexdigest()
            if calculated_hash != chunk_hash:
                raise ValueError("Chunk integrity check failed")

        # Save chunk atomically
        await self._save_chunk_atomically(chunk_path, chunk_data)
        
        return {
            "success": True,
            "upload_id": session.upload_id,
            "chunk_index": chunk_index,
            "chunk_size": len(chunk_data),
            "duplicate": False
        }

    async def _save_chunk_atomically(self, chunk_path: Path, chunk_data: bytes):
        """Save chunk with atomic write operation"""
        temp_path = chunk_path.with_suffix('.tmp')
        
        try:
            # Write to temporary file
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(chunk_data)
            
            # Atomic move
            temp_path.rename(chunk_path)
            
        except Exception as e:
            # Cleanup on failure
            if temp_path.exists():
                temp_path.unlink()
            raise e

    async def _update_session_progress(
        self,
        session: UploadSession,
        chunk_index: int,
        processing_time: float
    ):
        """Update session progress atomically"""
        
        def update_func(s: UploadSession):
            s.received_chunks = max(s.received_chunks, chunk_index + 1)
            s.chunk_timings.append(processing_time)
            s.status = UploadStatus.UPLOADING
            
            if not s.started_at:
                s.started_at = datetime.utcnow()

        await self.session_manager.update_session(session.upload_id, update_func)

    async def _finalize_upload(self, session: UploadSession):
        """Finalize upload by assembling chunks"""
        
        try:
            upload_dir = self.storage_paths["temp"] / session.upload_id
            final_path = self.storage_paths["uploads"] / f"{session.upload_id}_{session.filename}"
            
            self.logger.info(f"Finalizing upload: {session.upload_id}")
            
            # Verify all chunks exist
            missing_chunks = []
            for i in range(session.total_chunks):
                chunk_path = upload_dir / f"chunk_{i:04d}"
                if not chunk_path.exists():
                    missing_chunks.append(i)
            
            if missing_chunks:
                raise Exception(f"Missing chunks: {missing_chunks}")

            # Assemble file
            await self._assemble_chunks(upload_dir, final_path, session.total_chunks)
            
            # Cleanup chunks
            await self._cleanup_chunks(upload_dir)
            
            # Update session
            def update_func(s: UploadSession):
                s.status = UploadStatus.COMPLETED
                s.processing_metadata["final_path"] = str(final_path)
                s.processing_metadata["assembly_completed"] = datetime.utcnow().isoformat()

            await self.session_manager.update_session(session.upload_id, update_func)
            await self.session_manager.complete_session(session.upload_id, success=True)
            
            self.logger.info(f"Upload finalized successfully: {session.upload_id}")
            
        except Exception as e:
            self.logger.error(f"Upload finalization failed: {e}")
            
            # Mark session as failed
            def update_func(s: UploadSession):
                s.status = UploadStatus.FAILED
                s.add_error(e, "finalization")

            await self.session_manager.update_session(session.upload_id, update_func)
            await self.session_manager.complete_session(session.upload_id, success=False)
            
            raise e

    async def _assemble_chunks(self, upload_dir: Path, final_path: Path, total_chunks: int):
        """Assemble chunks into final file"""
        
        async with aiofiles.open(final_path, 'wb') as output_file:
            for i in range(total_chunks):
                chunk_path = upload_dir / f"chunk_{i:04d}"
                
                async with aiofiles.open(chunk_path, 'rb') as chunk_file:
                    chunk_data = await chunk_file.read()
                    await output_file.write(chunk_data)

    async def _cleanup_chunks(self, upload_dir: Path):
        """Cleanup chunk files"""
        
        try:
            for chunk_file in upload_dir.glob("chunk_*"):
                chunk_file.unlink()
            
            upload_dir.rmdir()
            
        except Exception as e:
            self.logger.warning(f"Chunk cleanup failed: {e}")

    async def get_upload_status(self, upload_id: str) -> Dict[str, Any]:
        """Get comprehensive upload status"""
        
        session = await self.session_manager.get_session(upload_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")

        return {
            "upload_id": session.upload_id,
            "filename": session.filename,
            "status": session.status.value,
            "progress": session.calculate_progress(),
            "received_chunks": session.received_chunks,
            "total_chunks": session.total_chunks,
            "upload_speed": session.calculate_upload_speed(),
            "estimated_time_remaining": session.estimate_time_remaining(),
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "error_count": len(session.error_history),
            "metadata": session.file_metadata
        }

    async def cancel_upload(self, upload_id: str) -> Dict[str, Any]:
        """Cancel upload and cleanup resources"""
        
        session = await self.session_manager.get_session(upload_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")

        # Update session status
        def update_func(s: UploadSession):
            s.status = UploadStatus.CANCELLED

        await self.session_manager.update_session(upload_id, update_func)
        
        # Cleanup files
        upload_dir = self.storage_paths["temp"] / upload_id
        if upload_dir.exists():
            await self._cleanup_chunks(upload_dir)
        
        # Remove session
        await self.session_manager.remove_session(upload_id)
        
        # Update metrics
        self.performance_monitor.record_upload_cancellation()
        
        return {
            "success": True,
            "message": f"Upload {upload_id} cancelled successfully"
        }

    async def _background_monitoring(self):
        """Background monitoring task"""
        
        while True:
            try:
                # Perform health check
                health = self.performance_monitor.check_health()
                
                if not health["healthy"]:
                    self.logger.warning(f"Health check failed: {health['alerts']}")
                
                # Log metrics periodically
                if self.config["enable_detailed_logging"]:
                    metrics = self.performance_monitor.export_metrics()
                    self.logger.info(f"Performance metrics: {metrics['metrics']}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(60)

    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        
        return {
            "service_info": {
                "name": "Netflix-Level Video Service",
                "version": "6.0",
                "status": "operational"
            },
            "performance": self.performance_monitor.export_metrics(),
            "session_management": {
                "active_sessions": self.session_manager.get_session_count(),
                "sessions": [
                    {
                        "id": s.upload_id,
                        "status": s.status.value,
                        "progress": s.calculate_progress(),
                        "speed": s.calculate_upload_speed()
                    }
                    for s in self.session_manager.get_active_sessions()
                ]
            },
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        
        health_status = self.performance_monitor.check_health()
        
        # Add service-specific checks
        health_status["service_checks"] = {
            "storage_accessible": all(
                path.exists() and path.is_dir() 
                for path in self.storage_paths.values()
            ),
            "session_manager_active": self.session_manager.cleanup_task is not None,
            "circuit_breaker_state": self.circuit_breaker.state
        }
        
        return health_status
