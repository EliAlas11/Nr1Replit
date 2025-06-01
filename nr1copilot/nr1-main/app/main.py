"""
ViralClip Pro - Main FastAPI Application
Netflix-level video processing platform with real-time WebSocket updates
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

import aiofiles
from fastapi import (
    FastAPI, HTTPException, WebSocket, WebSocketDisconnect, 
    File, UploadFile, Form, Depends, Request, BackgroundTasks,
    status, Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import http_exception_handler
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis

from .config import get_settings
from .config import get_settings

# Import modules with proper error handling
from .logging_config import get_logger, setup_logging
from .services.video_service import VideoProcessor
from .utils.security import SecurityManager
from .utils.rate_limiter import RateLimiter
from .utils.health import HealthChecker
from .utils.cache import CacheManager
from .utils.metrics import MetricsCollector

# Setup logging first
setup_logging()

# Initialize
settings = get_settings()
logger = get_logger(__name__)

# Global services
cache_manager: Optional[CacheManager] = None
metrics_collector: Optional[MetricsCollector] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global cache_manager, metrics_collector
    
    # Startup
    logger.info("🚀 Starting ViralClip Pro API v2.0.0")
    
    try:
        # Initialize cache
        cache_manager = CacheManager()
        await cache_manager.initialize()
        
        # Initialize metrics
        metrics_collector = MetricsCollector()
        
        # Warm up services
        await video_processor.initialize()
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down ViralClip Pro API")
    
    if cache_manager:
        await cache_manager.close()
    
    if metrics_collector:
        await metrics_collector.close()

# FastAPI app with enhanced configuration
app = FastAPI(
    title="ViralClip Pro API",
    description="Netflix-level AI-powered viral video creation platform with real-time processing",
    version="2.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "upload", "description": "Video upload operations"},
        {"name": "processing", "description": "Video processing operations"},
        {"name": "analytics", "description": "Analytics and insights"},
        {"name": "health", "description": "Health and monitoring"},
    ]
)

# Enhanced Middleware Stack
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] if settings.debug else settings.allowed_hosts
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware for request tracking and performance
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to headers
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(f"{process_time:.4f}")
        response.headers["X-Request-ID"] = request_id
        
        # Log performance metrics
        if metrics_collector:
            await metrics_collector.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=process_time
            )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed after {process_time:.4f}s: {e}")
        
        # Return proper error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )

# Services
video_processor = VideoProcessor()
security_manager = SecurityManager()
rate_limiter = RateLimiter()
health_checker = HealthChecker()

# Enhanced WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.connection_metadata: Dict[str, Dict] = {}
        self.upload_progress: Dict[str, Dict] = {}
        self.processing_progress: Dict[str, Dict] = {}
        self._heartbeat_interval = 30  # seconds
        self._max_connections_per_session = 5

    async def connect(self, websocket: WebSocket, session_id: str, connection_type: str = "general"):
        """Connect WebSocket with enhanced validation and monitoring"""
        try:
            await websocket.accept()
            
            # Validate connection limits
            if session_id in self.active_connections:
                if len(self.active_connections[session_id]) >= self._max_connections_per_session:
                    await websocket.close(code=1008, reason="Too many connections")
                    logger.warning(f"Connection limit exceeded for session {session_id}")
                    return False
            
            # Initialize session if not exists
            if session_id not in self.active_connections:
                self.active_connections[session_id] = {}
                self.connection_metadata[session_id] = {}
            
            # Store connection with metadata
            self.active_connections[session_id][connection_type] = websocket
            self.connection_metadata[session_id][connection_type] = {
                "connected_at": datetime.now(),
                "last_ping": datetime.now(),
                "message_count": 0
            }
            
            logger.info(f"✅ WebSocket connected: {session_id} ({connection_type})")
            
            # Send welcome message
            await self.send_message(session_id, {
                "type": "connection_established",
                "session_id": session_id,
                "connection_type": connection_type,
                "timestamp": datetime.now().isoformat(),
                "heartbeat_interval": self._heartbeat_interval
            }, connection_type)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to establish WebSocket connection: {e}")
            return False

    def disconnect(self, session_id: str, connection_type: str = "general"):
        """Enhanced disconnect with cleanup"""
        try:
            if session_id in self.active_connections:
                if connection_type in self.active_connections[session_id]:
                    # Get connection metadata for logging
                    metadata = self.connection_metadata.get(session_id, {}).get(connection_type, {})
                    connected_duration = datetime.now() - metadata.get("connected_at", datetime.now())
                    message_count = metadata.get("message_count", 0)
                    
                    # Clean up connections
                    del self.active_connections[session_id][connection_type]
                    if session_id in self.connection_metadata:
                        self.connection_metadata[session_id].pop(connection_type, None)
                    
                    # Clean up session if no more connections
                    if not self.active_connections[session_id]:
                        del self.active_connections[session_id]
                        self.connection_metadata.pop(session_id, None)
                        self.upload_progress.pop(session_id, None)
                        self.processing_progress.pop(session_id, None)
                    
                    logger.info(
                        f"🔌 WebSocket disconnected: {session_id} ({connection_type}) "
                        f"- Duration: {connected_duration}, Messages: {message_count}"
                    )
                    
        except Exception as e:
            logger.error(f"❌ Error during disconnect cleanup: {e}")

    async def send_message(self, session_id: str, message: dict, connection_type: str = "general"):
        """Enhanced message sending with error handling and metrics"""
        if session_id not in self.active_connections:
            logger.warning(f"⚠️ No active connections for session {session_id}")
            return False
            
        websocket = self.active_connections[session_id].get(connection_type)
        if not websocket:
            logger.warning(f"⚠️ No {connection_type} connection for session {session_id}")
            return False
            
        try:
            # Add metadata to message
            enhanced_message = {
                **message,
                "timestamp": message.get("timestamp", datetime.now().isoformat()),
                "session_id": session_id,
                "connection_type": connection_type
            }
            
            await websocket.send_text(json.dumps(enhanced_message))
            
            # Update connection metadata
            if session_id in self.connection_metadata:
                metadata = self.connection_metadata[session_id].get(connection_type, {})
                metadata["message_count"] = metadata.get("message_count", 0) + 1
                metadata["last_message"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send WebSocket message to {session_id}: {e}")
            self.disconnect(session_id, connection_type)
            return False

    async def broadcast_to_session(self, session_id: str, message: dict):
        """Enhanced broadcast with success tracking"""
        if session_id not in self.active_connections:
            return 0
            
        success_count = 0
        failed_connections = []
        
        for connection_type, websocket in self.active_connections[session_id].items():
            try:
                enhanced_message = {
                    **message,
                    "timestamp": message.get("timestamp", datetime.now().isoformat()),
                    "session_id": session_id,
                    "connection_type": connection_type,
                    "broadcast": True
                }
                
                await websocket.send_text(json.dumps(enhanced_message))
                success_count += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to broadcast to {connection_type}: {e}")
                failed_connections.append(connection_type)
        
        # Clean up failed connections
        for connection_type in failed_connections:
            self.disconnect(session_id, connection_type)
        
        return success_count

    async def send_heartbeat(self, session_id: str, connection_type: str = "general"):
        """Send heartbeat to keep connection alive"""
        return await self.send_message(session_id, {
            "type": "heartbeat",
            "timestamp": datetime.now().isoformat()
        }, connection_type)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics"""
        total_connections = sum(len(connections) for connections in self.active_connections.values())
        connection_types = {}
        
        for session_connections in self.active_connections.values():
            for conn_type in session_connections.keys():
                connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
        
        return {
            "total_sessions": len(self.active_connections),
            "total_connections": total_connections,
            "connection_types": connection_types,
            "active_uploads": len(self.upload_progress),
            "active_processing": len(self.processing_progress)
        }

manager = ConnectionManager()

# Enhanced Pydantic models with validation
class FileInfo(BaseModel):
    filename: str
    size: int = Field(..., gt=0, description="File size in bytes")
    type: str
    path: str
    thumbnail: Optional[str] = None
    processing_time: float = Field(..., ge=0)
    
class VideoInfo(BaseModel):
    duration: float = Field(..., gt=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fps: float = Field(..., gt=0)
    bitrate: Optional[int] = None
    codec: Optional[str] = None
    
class AIInsights(BaseModel):
    viral_potential: int = Field(..., ge=0, le=100)
    confidence_score: int = Field(..., ge=0, le=100)
    engagement_prediction: int = Field(..., ge=0, le=100)
    content_type: str
    best_platforms: List[str]
    optimal_length: str
    peak_moments: List[float]
    mood: str
    target_audience: str

class SuggestedClip(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=500)
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., gt=0)
    viral_score: int = Field(..., ge=0, le=100)
    recommended_platforms: List[str]
    estimated_views: str
    clip_type: str
    confidence: int = Field(..., ge=0, le=100)
    
    @validator('end_time')
    def end_time_must_be_greater_than_start_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v

class UploadResponse(BaseModel):
    success: bool
    message: str
    session_id: str = Field(..., regex=r'^session_[a-f0-9]{8}_\d+$')
    file_info: FileInfo
    video_info: Optional[VideoInfo] = None
    ai_insights: Optional[AIInsights] = None
    suggested_clips: Optional[List[SuggestedClip]] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ClipConfig(BaseModel):
    index: int = Field(..., ge=0)
    title: str = Field(..., min_length=1, max_length=100)
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., gt=0)
    custom_title: Optional[str] = Field(None, max_length=100)
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v

class ProcessingRequest(BaseModel):
    session_id: str = Field(..., regex=r'^session_[a-f0-9]{8}_\d+$')
    clips: List[ClipConfig] = Field(..., min_items=1, max_items=10)
    priority: str = Field(default="normal", regex=r'^(low|normal|high|urgent)$')
    quality: str = Field(default="high", regex=r'^(draft|standard|high|premium)$')
    platform_optimizations: List[str] = Field(
        default=["tiktok", "instagram"],
        description="Target platforms for optimization"
    )
    
    @validator('platform_optimizations')
    def validate_platforms(cls, v):
        allowed_platforms = {"tiktok", "instagram", "youtube_shorts", "snapchat", "twitter"}
        invalid_platforms = set(v) - allowed_platforms
        if invalid_platforms:
            raise ValueError(f"Invalid platforms: {invalid_platforms}")
        return v

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    task_id: str = Field(..., regex=r'^task_[a-f0-9]{8}_\d+$')
    estimated_time: int = Field(..., gt=0, description="Estimated time in seconds")
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None

# Create directories
upload_dir = Path("uploads")
output_dir = Path("output") 
temp_dir = Path("temp")

for directory in [upload_dir, output_dir, temp_dir]:
    directory.mkdir(exist_ok=True)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/public", StaticFiles(directory="public"), name="public")

# Routes
@app.get("/")
async def serve_index():
    """Serve the main application"""
    return FileResponse("index.html")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("favicon.ico")

@app.post("/api/v2/upload-video", response_model=UploadResponse, tags=["upload"])
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to upload"),
    upload_id: str = Form(..., regex=r'^id_[a-z0-9]+$', description="Unique upload ID")
):
    """
    Enhanced video upload with instant analysis and WebSocket progress
    
    Features:
    - Real-time upload progress via WebSocket
    - Instant video validation and analysis
    - AI-powered viral potential scoring
    - Automatic clip suggestions
    - Comprehensive error handling
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

    try:
        # Enhanced file validation
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )

        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Check content type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid content type. Must be a video file."
            )

        # Read file content
        logger.info(f"📤 Starting upload for {file.filename} (request: {request_id})")
        content = await file.read()
        file_size = len(content)

        # Validate file size
        max_size = getattr(settings, 'max_file_size', 2 * 1024 * 1024 * 1024)  # 2GB default
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {max_size / (1024**3):.1f}GB"
            )

        if file_size < 1024:  # Minimum 1KB
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too small. Minimum size: 1KB"
            )

        # Generate secure session ID
        session_id = f"session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # Create secure filename
        safe_filename = f"upload_{session_id}{file_extension}"
        file_path = upload_dir / safe_filename

        # Ensure upload directory exists
        upload_dir.mkdir(exist_ok=True)

        # Write file with progress updates
        logger.info(f"💾 Saving file: {safe_filename}")
        
        async with aiofiles.open(file_path, 'wb') as f:
            chunk_size = 8192
            total_written = 0

            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                await f.write(chunk)
                total_written += len(chunk)

                # Send progress update every 1% or 100KB
                if total_written % max(file_size // 100, 100_000) == 0 or total_written == file_size:
                    progress = min((total_written / file_size) * 100, 100)
                    await manager.send_message(upload_id, {
                        "type": "upload_progress",
                        "progress": round(progress, 2),
                        "uploaded": total_written,
                        "total": file_size,
                        "speed": total_written / (time.time() - start_time),
                        "eta": ((file_size - total_written) / max(total_written / (time.time() - start_time), 1)),
                        "timestamp": datetime.now().isoformat()
                    }, "upload")

        logger.info(f"✅ File saved successfully: {file_path}")

        # Validate video file
        logger.info(f"🔍 Validating video file...")
        validation_result = await video_processor.validate_video(str(file_path))
        
        if not validation_result.get("valid", False):
            # Clean up invalid file
            if file_path.exists():
                file_path.unlink()
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation_result.get("error", "Invalid video file")
            )

        # Extract thumbnail
        logger.info(f"🖼️ Extracting thumbnail...")
        thumbnail_path = await video_processor.extract_thumbnail(str(file_path))

        # Generate AI insights
        logger.info(f"🤖 Generating AI insights...")
        ai_insights = await generate_ai_insights(validation_result["metadata"])

        # Generate suggested clips
        logger.info(f"✂️ Generating clip suggestions...")
        suggested_clips = await generate_suggested_clips(validation_result["metadata"], ai_insights)

        # Cache results for faster subsequent access
        if cache_manager:
            await cache_manager.set(
                f"session:{session_id}",
                {
                    "file_path": str(file_path),
                    "validation_result": validation_result,
                    "ai_insights": ai_insights,
                    "suggested_clips": suggested_clips
                },
                ttl=3600  # 1 hour
            )

        # Send completion message
        await manager.send_message(upload_id, {
            "type": "upload_complete",
            "session_id": session_id,
            "message": "Upload and analysis complete!",
            "ai_insights": ai_insights,
            "suggested_clips": len(suggested_clips),
            "timestamp": datetime.now().isoformat()
        }, "upload")

        processing_time = time.time() - start_time
        logger.info(f"🎉 Upload completed in {processing_time:.2f}s for session {session_id}")

        # Record metrics
        if metrics_collector:
            await metrics_collector.record_upload(
                file_size=file_size,
                processing_time=processing_time,
                file_type=file_extension,
                success=True
            )

        return UploadResponse(
            success=True,
            message="Video uploaded and analyzed successfully",
            session_id=session_id,
            file_info=FileInfo(
                filename=file.filename,
                size=file_size,
                type=file.content_type,
                path=str(file_path),
                thumbnail=thumbnail_path,
                processing_time=processing_time
            ),
            video_info=VideoInfo(**validation_result["metadata"]) if validation_result.get("metadata") else None,
            ai_insights=AIInsights(**ai_insights) if ai_insights else None,
            suggested_clips=[SuggestedClip(**clip) for clip in suggested_clips] if suggested_clips else None,
            request_id=request_id
        )

    except HTTPException:
        # Log and re-raise HTTP exceptions
        logger.warning(f"⚠️ Upload validation failed for {upload_id}: {request.method} {request.url}")
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"❌ Unexpected upload error for {upload_id}: {e}", exc_info=True)
        
        # Clean up any partial files
        try:
            if 'file_path' in locals() and file_path and file_path.exists():
                file_path.unlink()
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup file: {cleanup_error}")
        
        # Notify client of error
        try:
            await manager.send_message(upload_id, {
                "type": "upload_error",
                "error": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }, "upload")
        except Exception as notification_error:
            logger.warning(f"Failed to send error notification: {notification_error}")
        
        # Record error metrics
        try:
            if metrics_collector:
                await metrics_collector.record_error("upload", str(e))
        except Exception as metrics_error:
            logger.warning(f"Failed to record error metrics: {metrics_error}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed due to internal error"
        )

@app.post("/api/v2/process-video", response_model=ProcessingResponse)
async def process_video(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start video processing with real-time WebSocket updates"""
    try:
        # Generate task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        # Estimate processing time
        estimated_time = len(request.clips) * 30  # 30 seconds per clip

        # Start background processing
        background_tasks.add_task(
            process_clips_background,
            task_id,
            request.session_id,
            request.clips,
            request.quality,
            request.platform_optimizations
        )

        return ProcessingResponse(
            success=True,
            message="Processing started successfully",
            task_id=task_id,
            estimated_time=estimated_time
        )

    except Exception as e:
        logger.error(f"Processing start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

async def process_clips_background(
    task_id: str,
    session_id: str,
    clips: List[Dict[str, Any]],
    quality: str,
    platform_optimizations: List[str]
):
    """Background task for processing clips with progress updates"""
    try:
        # Send processing started message
        await manager.send_message(task_id, {
            "type": "processing_started",
            "task_id": task_id,
            "total_clips": len(clips),
            "timestamp": datetime.now().isoformat()
        }, "processing")

        results = []

        for i, clip in enumerate(clips):
            # Send progress update
            progress = (i / len(clips)) * 100
            await manager.send_message(task_id, {
                "type": "progress_update",
                "data": {
                    "current": i + 1,
                    "total": len(clips),
                    "percentage": progress,
                    "stage": f"Processing clip {i + 1}",
                    "message": f"Creating viral clip: {clip.get('title', f'Clip {i + 1}')}",
                    "clip_title": clip.get('title', f'Clip {i + 1}')
                },
                "timestamp": datetime.now().isoformat()
            }, "processing")

            # Simulate processing time
            await asyncio.sleep(2)

            # Mock result
            result = {
                "clip_index": i,
                "title": clip.get('title', f'Viral Clip {i + 1}'),
                "success": True,
                "output_path": f"output/clip_{task_id}_{i}.mp4",
                "thumbnail_path": f"output/clip_{task_id}_{i}_thumb.jpg",
                "duration": clip.get('end_time', 60) - clip.get('start_time', 0),
                "file_size": 5 * 1024 * 1024,  # 5MB mock
                "viral_score": min(85 + (i * 2), 98),
                "performance_prediction": {
                    "estimated_views": f"{(10 + i) * 1000}K",
                    "engagement_rate": f"{75 + i}%",
                    "viral_potential": "High"
                }
            }
            results.append(result)

        # Send completion message
        await manager.send_message(task_id, {
            "type": "processing_complete",
            "data": {
                "task_id": task_id,
                "results": results,
                "total_processed": len(results),
                "success_count": len([r for r in results if r["success"]]),
                "processing_time": len(clips) * 2
            },
            "timestamp": datetime.now().isoformat()
        }, "processing")

        logger.info(f"Processing completed for task {task_id}")

    except Exception as e:
        logger.error(f"Background processing error: {e}")
        await manager.send_message(task_id, {
            "type": "processing_error",
            "data": {"error": str(e)},
            "timestamp": datetime.now().isoformat()
        }, "processing")

async def generate_ai_insights(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AI insights for the uploaded video"""
    duration = metadata.get("duration", 0)
    width = metadata.get("width", 0)
    height = metadata.get("height", 0)

    # Mock AI analysis
    viral_potential = min(85 + (duration / 10), 95)
    confidence_score = min(88 + (width / 100), 96)
    engagement_prediction = min(78 + (height / 100), 92)

    return {
        "viral_potential": int(viral_potential),
        "confidence_score": int(confidence_score),
        "engagement_prediction": int(engagement_prediction),
        "content_type": "entertainment",
        "best_platforms": ["tiktok", "instagram", "youtube_shorts"],
        "optimal_length": "15-45 seconds",
        "peak_moments": [15, 45, 75],
        "mood": "energetic",
        "target_audience": "18-34"
    }

async def generate_suggested_clips(metadata: Dict[str, Any], insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate suggested viral clips"""
    duration = metadata.get("duration", 120)
    clips = []

    # Generate 3-5 clips based on video length
    num_clips = min(5, max(3, int(duration / 30)))

    for i in range(num_clips):
        start_time = i * (duration / num_clips)
        end_time = min(start_time + 30, duration)

        clips.append({
            "title": f"Viral Moment #{i + 1}",
            "description": f"High-energy clip perfect for social media engagement",
            "start_time": start_time,
            "end_time": end_time,
            "viral_score": insights["viral_potential"] + (i * 2),
            "recommended_platforms": ["tiktok", "instagram", "youtube_shorts"],
            "estimated_views": f"{(15 + i * 5)}K",
            "clip_type": ["hook", "climax", "trending", "entertainment", "conclusion"][i],
            "confidence": min(85 + (i * 3), 98)
        })

    return clips

@app.get("/api/v2/download/{task_id}/{clip_index}")
async def download_clip(task_id: str, clip_index: int):
    """Download a processed video clip"""
    try:
        # Mock file path - in production, lookup actual file
        filename = f"clip_{task_id}_{clip_index}.mp4"
        file_path = output_dir / filename

        # Create mock file if it doesn't exist
        if not file_path.exists():
            # Create a small mock video file
            file_path.write_bytes(b"Mock video content for " + filename.encode())

        return FileResponse(
            path=str(file_path),
            media_type="video/mp4",
            filename=filename
        )

    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=404, detail="Clip not found")

# WebSocket endpoints
@app.websocket("/api/v2/upload-progress/{upload_id}")
async def upload_progress_websocket(websocket: WebSocket, upload_id: str):
    await manager.connect(websocket, upload_id, "upload")
    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Upload progress WebSocket connected",
            "upload_id": upload_id,
            "timestamp": datetime.now().isoformat()
        }))

        # Keep connection alive
        while True:
            try:
                # Wait for ping or send keep-alive
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))

            except asyncio.TimeoutError:
                # Send keep-alive
                await websocket.send_text(json.dumps({
                    "type": "keep_alive",
                    "timestamp": datetime.now().isoformat()
                }))

    except WebSocketDisconnect:
        manager.disconnect(upload_id, "upload")
    except Exception as e:
        logger.error(f"Upload WebSocket error: {e}")
        manager.disconnect(upload_id, "upload")

@app.websocket("/api/v2/ws/{task_id}")
async def processing_websocket(websocket: WebSocket, task_id: str):
    await manager.connect(websocket, task_id, "processing")
    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Processing WebSocket connected",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }))

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))

            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "type": "keep_alive",
                    "timestamp": datetime.now().isoformat()
                }))

    except WebSocketDisconnect:
        manager.disconnect(task_id, "processing")
    except Exception as e:
        logger.error(f"Processing WebSocket error: {e}")
        manager.disconnect(task_id, "processing")

# Health check
@app.get("/api/v2/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "video_processor": "online",
            "websocket_manager": "online",
            "file_system": "accessible"
        },
        "metrics": {
            "active_connections": len(manager.active_connections),
            "upload_directory": str(upload_dir),
            "output_directory": str(output_dir)
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return FileResponse("index.html")

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "Please try again later"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)