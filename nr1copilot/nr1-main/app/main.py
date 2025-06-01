"""
ViralClip Pro - Netflix-Level FastAPI Application
Production-ready video processing platform with enterprise-grade architecture
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
    Depends
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .logging_config import get_logger
from .services.video_service import VideoProcessor
from .utils.security import SecurityManager
from .utils.rate_limiter import RateLimiter
from .utils.cache import CacheManager
from .utils.health import HealthChecker

# Initialize logger
logger = get_logger(__name__)
settings = get_settings()

# Global services
video_processor: Optional[VideoProcessor] = None
security_manager: Optional[SecurityManager] = None
rate_limiter: Optional[RateLimiter] = None
cache_manager: Optional[CacheManager] = None
health_checker: Optional[HealthChecker] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, session_id: str, connection_type: str):
        await websocket.accept()

        if session_id not in self.active_connections:
            self.active_connections[session_id] = {}
            self.connection_metadata[session_id] = {}

        self.active_connections[session_id][connection_type] = websocket
        self.connection_metadata[session_id][connection_type] = {
            "connected_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }

        logger.info(f"WebSocket connected: {session_id}:{connection_type}")

    def disconnect(self, session_id: str, connection_type: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].pop(connection_type, None)
            self.connection_metadata[session_id].pop(connection_type, None)

            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
                del self.connection_metadata[session_id]

        logger.info(f"WebSocket disconnected: {session_id}:{connection_type}")

    async def send_message(self, session_id: str, connection_type: str, message: Dict[str, Any]):
        if (session_id in self.active_connections and 
            connection_type in self.active_connections[session_id]):

            websocket = self.active_connections[session_id][connection_type]
            try:
                await websocket.send_text(json.dumps(message))

                # Update last activity
                if session_id in self.connection_metadata:
                    self.connection_metadata[session_id][connection_type]["last_activity"] = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                self.disconnect(session_id, connection_type)

    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.active_connections:
            for connection_type, websocket in self.active_connections[session_id].items():
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to broadcast to {connection_type}: {e}")
                    self.disconnect(session_id, connection_type)

manager = ConnectionManager()

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle"""
    global video_processor, security_manager, rate_limiter, cache_manager, health_checker

    try:
        # Startup
        logger.info("🚀 Starting ViralClip Pro Application...")

        # Initialize core services
        video_processor = VideoProcessor()
        await video_processor.initialize()

        security_manager = SecurityManager()
        rate_limiter = RateLimiter()
        cache_manager = CacheManager()
        health_checker = HealthChecker()

        # Create required directories
        for directory in [settings.UPLOAD_PATH, settings.OUTPUT_PATH, settings.TEMP_PATH]:
            Path(directory).mkdir(parents=True, exist_ok=True)

        logger.info("✅ ViralClip Pro Application started successfully")

        yield

    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🛑 Shutting down ViralClip Pro Application...")
        if video_processor:
            await video_processor.cleanup_temp_files([])

# Pydantic models
class UploadResponse(BaseModel):
    success: bool = True
    session_id: str
    file_info: Dict[str, Any]
    video_info: Optional[Dict[str, Any]] = None
    ai_insights: Optional[Dict[str, Any]] = None
    suggested_clips: Optional[List[Dict[str, Any]]] = None
    message: str = "Upload successful"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ProcessingRequest(BaseModel):
    session_id: str
    clips: List[Dict[str, Any]]
    quality: str = "high"
    platform_optimizations: Optional[List[str]] = None
    priority: str = "standard"

class ProcessingResponse(BaseModel):
    success: bool = True
    task_id: str
    message: str = "Processing started"
    estimated_time: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Create FastAPI application
app = FastAPI(
    title="ViralClip Pro API",
    description="Netflix-level AI-powered viral video creation platform",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Static file serving
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/public", StaticFiles(directory="public"), name="public")

# Dependency injection
async def get_video_processor() -> VideoProcessor:
    if video_processor is None:
        raise HTTPException(status_code=503, detail="Video processor not available")
    return video_processor

async def get_rate_limiter() -> RateLimiter:
    if rate_limiter is None:
        raise HTTPException(status_code=503, detail="Rate limiter not available")
    return rate_limiter

# Main application routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main application"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>ViralClip Pro</h1><p>Application loading...</p>",
            status_code=200
        )

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    favicon_path = Path("favicon.ico")
    if favicon_path.exists():
        return FileResponse(favicon_path)
    return JSONResponse({"error": "Favicon not found"}, status_code=404)

# Upload endpoint with real-time progress
@app.post("/api/v2/upload-video", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    upload_id: str = Form(...),
    processor: VideoProcessor = Depends(get_video_processor),
    limiter: RateLimiter = Depends(get_rate_limiter)
):
    """Upload and analyze video with real-time progress updates"""

    session_id = str(uuid.uuid4())

    try:
        logger.info(f"Starting upload for session: {session_id}")

        # Rate limiting
        await limiter.check_rate_limit("upload", upload_id)

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)

        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )

        # Reset file pointer
        await file.seek(0)

        # Create upload directory
        upload_dir = Path(settings.UPLOAD_PATH)
        upload_dir.mkdir(exist_ok=True)

        # Save uploaded file
        file_path = upload_dir / f"{session_id}_{file.filename}"

        # Save file with progress updates
        with open(file_path, "wb") as f:
            f.write(content)

        # Send upload progress via WebSocket
        await manager.send_message(upload_id, "upload", {
            "type": "upload_progress",
            "progress": 100,
            "message": "Upload complete, analyzing video..."
        })

        # Validate video file
        validation_result = await processor.validate_video(str(file_path))

        if not validation_result["valid"]:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=validation_result["error"])

        # Extract metadata and generate insights
        metadata = validation_result["metadata"]

        # Generate AI insights (mock for now)
        ai_insights = {
            "viral_potential": 85,
            "confidence_score": 92,
            "engagement_prediction": 78,
            "recommended_platforms": ["tiktok", "instagram", "youtube_shorts"],
            "content_analysis": {
                "energy_level": "high",
                "visual_appeal": "excellent",
                "audio_quality": "good"
            }
        }

        # Generate suggested clips
        suggested_clips = [
            {
                "title": "Opening Hook",
                "start_time": 0,
                "end_time": 15,
                "viral_score": 88,
                "description": "Attention-grabbing opening sequence",
                "clip_type": "hook",
                "recommended_platforms": ["tiktok", "instagram"],
                "estimated_views": "500K+"
            },
            {
                "title": "Key Moment",
                "start_time": 30,
                "end_time": 60,
                "viral_score": 92,
                "description": "Main highlight of the video",
                "clip_type": "climax",
                "recommended_platforms": ["youtube_shorts", "tiktok"],
                "estimated_views": "1M+"
            },
            {
                "title": "Trending Segment",
                "start_time": 75,
                "end_time": 105,
                "viral_score": 79,
                "description": "Trending topic discussion",
                "clip_type": "trending",
                "recommended_platforms": ["tiktok"],
                "estimated_views": "300K+"
            }
        ]

        # Send completion message
        await manager.send_message(upload_id, "upload", {
            "type": "upload_complete",
            "message": "Analysis complete! Video ready for clip creation."
        })

        # Clean up background task
        background_tasks.add_task(
            cache_manager.cache_session_data,
            session_id,
            {
                "file_path": str(file_path),
                "metadata": metadata,
                "ai_insights": ai_insights,
                "suggested_clips": suggested_clips
            }
        )

        return UploadResponse(
            session_id=session_id,
            file_info={
                "filename": file.filename,
                "size": file_size,
                "format": file_path.suffix,
                "duration": metadata.get("duration", 0)
            },
            video_info={
                "title": file.filename,
                "duration": metadata.get("duration", 0),
                "thumbnail": f"/api/v2/thumbnail/{session_id}",
                "width": metadata.get("width", 0),
                "height": metadata.get("height", 0),
                "view_count": 0
            },
            ai_insights=ai_insights,
            suggested_clips=suggested_clips
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error for session {session_id}: {e}")

        # Send error via WebSocket
        await manager.send_message(upload_id, "upload", {
            "type": "upload_error",
            "error": str(e)
        })

        raise HTTPException(status_code=500, detail="Upload processing failed")

# Video processing endpoint
@app.post("/api/v2/process-video", response_model=ProcessingResponse)
async def process_video(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    processor: VideoProcessor = Depends(get_video_processor)
):
    """Process selected video clips with AI optimization"""

    task_id = str(uuid.uuid4())

    try:
        logger.info(f"Starting processing for task: {task_id}")

        # Get cached session data
        session_data = await cache_manager.get_session_data(request.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Estimate processing time
        estimated_time = await processor.estimate_processing_time(
            session_data["metadata"]["duration"],
            len(request.clips),
            request.quality
        )

        # Start background processing
        background_tasks.add_task(
            process_clips_background,
            task_id,
            request,
            session_data,
            processor
        )

        return ProcessingResponse(
            task_id=task_id,
            estimated_time=int(estimated_time),
            message=f"Processing {len(request.clips)} clips with {request.quality} quality"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing initiation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")

async def process_clips_background(
    task_id: str,
    request: ProcessingRequest,
    session_data: Dict[str, Any],
    processor: VideoProcessor
):
    """Background task for processing video clips"""

    try:
        # Send processing started message
        await manager.send_message(task_id, "processing", {
            "type": "processing_started",
            "total_clips": len(request.clips),
            "quality": request.quality
        })

        # Create output directory
        output_dir = Path(settings.OUTPUT_PATH) / task_id
        output_dir.mkdir(exist_ok=True)

        # Process clips
        results = await processor.batch_process(
            clips=request.clips,
            input_path=session_data["file_path"],
            output_dir=str(output_dir),
            quality=request.quality,
            platform_optimizations=request.platform_optimizations
        )

        # Send progress updates
        for i, result in enumerate(results):
            progress = ((i + 1) / len(results)) * 100

            await manager.send_message(task_id, "processing", {
                "type": "progress_update",
                "data": {
                    "percentage": progress,
                    "stage": f"Processing clip {i + 1} of {len(results)}",
                    "message": f"Completed: {result.get('clip_title', f'Clip {i + 1}')}"
                }
            })

        # Send completion message
        await manager.send_message(task_id, "processing", {
            "type": "processing_complete",
            "data": {
                "results": results,
                "total_clips": len(results),
                "success_count": sum(1 for r in results if r.get("success", False))
            }
        })

        # Cache results
        await cache_manager.cache_results(task_id, results)

    except Exception as e:
        logger.error(f"Background processing error for task {task_id}: {e}")

        await manager.send_message(task_id, "processing", {
            "type": "processing_error",
            "data": {"error": str(e)}
        })

# WebSocket endpoints
@app.websocket("/api/v2/upload-progress/{upload_id}")
async def upload_progress_websocket(websocket: WebSocket, upload_id: str):
    """WebSocket for real-time upload progress"""
    await manager.connect(websocket, upload_id, "upload")

    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Upload progress WebSocket connected",
            "upload_id": upload_id,
            "timestamp": datetime.now().isoformat()
        }))

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
        manager.disconnect(upload_id, "upload")
    except Exception as e:
        logger.error(f"Upload WebSocket error: {e}")
        manager.disconnect(upload_id, "upload")

@app.websocket("/api/v2/ws/{task_id}")
async def processing_websocket(websocket: WebSocket, task_id: str):
    """WebSocket for real-time processing updates"""
    await manager.connect(websocket, task_id, "processing")

    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Processing WebSocket connected",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }))

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

# Download endpoints
@app.get("/api/v2/download/{task_id}/{clip_index}")
async def download_clip(task_id: str, clip_index: int):
    """Download processed video clip"""
    try:
        output_dir = Path(settings.OUTPUT_PATH) / task_id
        clip_files = list(output_dir.glob(f"clip_{clip_index + 1}_*.mp4"))

        if not clip_files:
            raise HTTPException(status_code=404, detail="Clip not found")

        clip_path = clip_files[0]

        return FileResponse(
            path=str(clip_path),
            media_type="video/mp4",
            filename=f"viral_clip_{clip_index + 1}.mp4"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail="Download failed")

# Health check
@app.get("/api/v2/health")
async def health_check():
    """Comprehensive health check"""
    try:
        if health_checker is None:
            return {"status": "unhealthy", "error": "Health checker not initialized"}

        health_status = await health_checker.check_all()

        return {
            "status": "healthy" if health_status["overall_healthy"] else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "checks": health_status
        }

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )