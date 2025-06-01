"""
ViralClip Pro - Netflix-Level FastAPI Application
Production-ready video processing platform with real-time features
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .logging_config import get_logger
from .services.video_service import VideoProcessor
from .services.realtime_engine import RealtimeEngine
from .utils.security import SecurityManager
from .utils.rate_limiter import RateLimiter
from .utils.cache import CacheManager
from .utils.health import HealthChecker

# Initialize logger and settings
logger = get_logger(__name__)
settings = get_settings()

# Global services
video_processor: Optional[VideoProcessor] = None
realtime_engine: Optional[RealtimeEngine] = None
security_manager: Optional[SecurityManager] = None
rate_limiter: Optional[RateLimiter] = None
cache_manager: Optional[CacheManager] = None
health_checker: Optional[HealthChecker] = None

# Pydantic models for real-time features
class RealtimePreviewRequest(BaseModel):
    session_id: str
    start_time: float
    end_time: float
    quality: str = "preview"

class ViralScoreUpdate(BaseModel):
    session_id: str
    timeline_position: float
    viral_score: int
    confidence: float
    factors: List[str]

class ProcessingStatusUpdate(BaseModel):
    task_id: str
    stage: str
    progress: float
    eta_seconds: Optional[int] = None
    message: str
    entertaining_fact: Optional[str] = None

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle"""
    global video_processor, realtime_engine, security_manager, rate_limiter, cache_manager, health_checker

    try:
        logger.info("🚀 Starting ViralClip Pro Netflix-Level Application...")

        # Initialize core services
        video_processor = VideoProcessor()
        await video_processor.initialize()

        realtime_engine = RealtimeEngine()
        await realtime_engine.initialize()

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
        logger.info("🛑 Shutting down ViralClip Pro Application...")
        if realtime_engine:
            await realtime_engine.cleanup()
        if video_processor:
            await video_processor.cleanup_temp_files([])

# Create FastAPI application
app = FastAPI(
    title="ViralClip Pro API",
    description="Netflix-level AI-powered viral video creation platform with real-time features",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enhanced middleware
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

async def get_realtime_engine() -> RealtimeEngine:
    if realtime_engine is None:
        raise HTTPException(status_code=503, detail="Realtime engine not available")
    return realtime_engine

# Main routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main application"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>ViralClip Pro</h1><p>Netflix-level application loading...</p>",
            status_code=200
        )

# Real-time upload with instant preview
@app.post("/api/v3/upload-video")
async def upload_video_realtime(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    upload_id: str = Form(...),
    processor: VideoProcessor = Depends(get_video_processor),
    engine: RealtimeEngine = Depends(get_realtime_engine)
):
    """Upload video with real-time analysis and instant preview generation"""
    session_id = str(uuid.uuid4())

    try:
        logger.info(f"Real-time upload started: {session_id}")

        # Validate and save file
        if not file.filename or file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Invalid file")

        upload_dir = Path(settings.UPLOAD_PATH)
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{session_id}_{file.filename}"

        # Stream file with progress updates
        with open(file_path, "wb") as f:
            async for chunk in file.stream():
                f.write(chunk)
                # Send progress via WebSocket
                await engine.broadcast_upload_progress(upload_id, {
                    "type": "upload_progress",
                    "progress": (f.tell() / file.size) * 100 if file.size else 0
                })

        # Start real-time analysis
        analysis_task = asyncio.create_task(
            engine.start_realtime_analysis(session_id, str(file_path))
        )

        # Generate instant preview clips
        preview_task = asyncio.create_task(
            engine.generate_instant_previews(session_id, str(file_path))
        )

        # Wait for initial analysis
        analysis_result = await analysis_task
        preview_result = await preview_task

        return {
            "success": True,
            "session_id": session_id,
            "analysis": analysis_result,
            "previews": preview_result,
            "realtime_endpoints": {
                "viral_scores": f"/api/v3/realtime/viral-scores/{session_id}",
                "timeline": f"/api/v3/realtime/timeline/{session_id}",
                "preview": f"/api/v3/realtime/preview/{session_id}"
            }
        }

    except Exception as e:
        logger.error(f"Real-time upload error: {e}")
        await engine.broadcast_upload_progress(upload_id, {
            "type": "upload_error",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail="Upload failed")

# Real-time clip generation with live preview
@app.post("/api/v3/generate-clip-preview")
async def generate_clip_preview(
    request: RealtimePreviewRequest,
    engine: RealtimeEngine = Depends(get_realtime_engine)
):
    """Generate real-time clip preview with viral score analysis"""
    try:
        preview_data = await engine.generate_live_preview(
            request.session_id,
            request.start_time,
            request.end_time,
            request.quality
        )

        return {
            "success": True,
            "preview_url": preview_data["preview_url"],
            "viral_analysis": preview_data["viral_analysis"],
            "optimization_suggestions": preview_data["suggestions"],
            "processing_time": preview_data["processing_time"]
        }

    except Exception as e:
        logger.error(f"Preview generation error: {e}")
        raise HTTPException(status_code=500, detail="Preview generation failed")

# Interactive timeline with viral score visualization
@app.get("/api/v3/timeline/{session_id}")
async def get_interactive_timeline(
    session_id: str,
    engine: RealtimeEngine = Depends(get_realtime_engine)
):
    """Get interactive timeline data with viral score visualization"""
    try:
        timeline_data = await engine.get_timeline_data(session_id)

        return {
            "success": True,
            "timeline": {
                "duration": timeline_data["duration"],
                "viral_heatmap": timeline_data["viral_scores"],
                "key_moments": timeline_data["highlights"],
                "engagement_peaks": timeline_data["peaks"],
                "recommended_clips": timeline_data["clips"]
            },
            "visualization": {
                "score_graph": timeline_data["score_visualization"],
                "emotion_tracking": timeline_data["emotions"],
                "energy_levels": timeline_data["energy"]
            }
        }

    except Exception as e:
        logger.error(f"Timeline error: {e}")
        raise HTTPException(status_code=500, detail="Timeline generation failed")

# Enhanced processing with entertaining status updates
@app.post("/api/v3/process-clips")
async def process_clips_enhanced(
    request: dict,
    background_tasks: BackgroundTasks,
    engine: RealtimeEngine = Depends(get_realtime_engine)
):
    """Process clips with real-time status and entertaining content"""
    task_id = str(uuid.uuid4())

    try:
        # Start enhanced processing
        background_tasks.add_task(
            engine.process_clips_with_entertainment,
            task_id,
            request["session_id"],
            request["clips"],
            request.get("options", {})
        )

        return {
            "success": True,
            "task_id": task_id,
            "realtime_status": f"/api/v3/realtime/processing/{task_id}",
            "entertainment_feed": f"/api/v3/realtime/entertainment/{task_id}"
        }

    except Exception as e:
        logger.error(f"Enhanced processing error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed to start")

# WebSocket endpoints for real-time features
@app.websocket("/api/v3/ws/upload/{upload_id}")
async def upload_websocket(websocket: WebSocket, upload_id: str):
    """WebSocket for real-time upload progress"""
    await realtime_engine.handle_upload_websocket(websocket, upload_id)

@app.websocket("/api/v3/ws/viral-scores/{session_id}")
async def viral_scores_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time viral score updates"""
    await realtime_engine.handle_viral_scores_websocket(websocket, session_id)

@app.websocket("/api/v3/ws/timeline/{session_id}")
async def timeline_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for interactive timeline updates"""
    await realtime_engine.handle_timeline_websocket(websocket, session_id)

@app.websocket("/api/v3/ws/processing/{task_id}")
async def processing_websocket(websocket: WebSocket, task_id: str):
    """WebSocket for entertaining processing updates"""
    await realtime_engine.handle_processing_websocket(websocket, task_id)

# Real-time preview streaming
@app.get("/api/v3/stream/preview/{session_id}")
async def stream_preview(
    session_id: str,
    start_time: float,
    end_time: float,
    engine: RealtimeEngine = Depends(get_realtime_engine)
):
    """Stream real-time preview video"""
    try:
        async def generate_preview_stream():
            async for chunk in engine.stream_preview(session_id, start_time, end_time):
                yield chunk

        return StreamingResponse(
            generate_preview_stream(),
            media_type="video/mp4",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    except Exception as e:
        logger.error(f"Preview streaming error: {e}")
        raise HTTPException(status_code=500, detail="Preview streaming failed")

# Health check with real-time metrics
@app.get("/api/v3/health")
async def health_check_enhanced():
    """Enhanced health check with real-time system metrics"""
    try:
        health_data = await health_checker.comprehensive_check()
        realtime_metrics = await realtime_engine.get_system_metrics()

        return {
            "status": "healthy" if health_data["overall_healthy"] else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0",
            "services": health_data,
            "realtime_metrics": realtime_metrics,
            "performance": {
                "active_sessions": len(realtime_engine.active_sessions),
                "processing_queue": realtime_engine.processing_queue_size,
                "websocket_connections": realtime_engine.websocket_count
            }
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
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
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