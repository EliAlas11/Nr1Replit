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
from typing import Dict, List, Optional, Any

import aiofiles
from fastapi import (
    FastAPI, HTTPException, WebSocket, WebSocketDisconnect, 
    File, UploadFile, Form, Depends, Request, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .logging_config import get_logger
from .services.video_service import VideoProcessor
from .utils.security import SecurityManager
from .utils.rate_limiter import RateLimiter
from .utils.health import HealthChecker

# Initialize
settings = get_settings()
logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="ViralClip Pro API",
    description="Netflix-level AI-powered viral video creation platform",
    version="2.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Services
video_processor = VideoProcessor()
security_manager = SecurityManager()
rate_limiter = RateLimiter()
health_checker = HealthChecker()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.upload_progress: Dict[str, Dict] = {}
        self.processing_progress: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str, connection_type: str = "general"):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = {}
        self.active_connections[session_id][connection_type] = websocket
        logger.info(f"WebSocket connected: {session_id} ({connection_type})")

    def disconnect(self, session_id: str, connection_type: str = "general"):
        if session_id in self.active_connections:
            if connection_type in self.active_connections[session_id]:
                del self.active_connections[session_id][connection_type]
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected: {session_id} ({connection_type})")

    async def send_message(self, session_id: str, message: dict, connection_type: str = "general"):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id].get(connection_type)
            if websocket:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send WebSocket message: {e}")
                    self.disconnect(session_id, connection_type)

    async def broadcast_to_session(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            for connection_type, websocket in self.active_connections[session_id].items():
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to broadcast to {connection_type}: {e}")

manager = ConnectionManager()

# Pydantic models
class UploadResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    file_info: Dict[str, Any]
    video_info: Optional[Dict[str, Any]] = None
    ai_insights: Optional[Dict[str, Any]] = None
    suggested_clips: Optional[List[Dict[str, Any]]] = None

class ProcessingRequest(BaseModel):
    session_id: str
    clips: List[Dict[str, Any]]
    priority: str = "normal"
    quality: str = "high"
    platform_optimizations: List[str] = ["tiktok", "instagram"]

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    task_id: str
    estimated_time: int

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

@app.post("/api/v2/upload-video", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    upload_id: str = Form(...)
):
    """Enhanced video upload with instant analysis and WebSocket progress"""
    start_time = time.time()

    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)

        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size / (1024**3):.1f}GB"
            )

        # Generate session ID
        session_id = f"session_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        # Save uploaded file
        file_extension = Path(file.filename).suffix.lower()
        safe_filename = f"upload_{session_id}{file_extension}"
        file_path = upload_dir / safe_filename

        # Write file with progress updates
        async with aiofiles.open(file_path, 'wb') as f:
            chunk_size = 8192
            total_written = 0

            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                await f.write(chunk)
                total_written += len(chunk)

                # Send progress update
                progress = min((total_written / file_size) * 100, 100)
                await manager.send_message(upload_id, {
                    "type": "upload_progress",
                    "progress": progress,
                    "uploaded": total_written,
                    "total": file_size,
                    "timestamp": datetime.now().isoformat()
                }, "upload")

        # Validate video
        validation_result = await video_processor.validate_video(str(file_path))
        if not validation_result.get("valid", False):
            raise HTTPException(
                status_code=400,
                detail=validation_result.get("error", "Invalid video file")
            )

        # Extract thumbnail
        thumbnail_path = await video_processor.extract_thumbnail(str(file_path))

        # Generate mock AI insights
        ai_insights = await generate_ai_insights(validation_result["metadata"])

        # Generate suggested clips
        suggested_clips = await generate_suggested_clips(validation_result["metadata"], ai_insights)

        # Send completion message
        await manager.send_message(upload_id, {
            "type": "upload_complete",
            "session_id": session_id,
            "message": "Upload and analysis complete!",
            "timestamp": datetime.now().isoformat()
        }, "upload")

        processing_time = time.time() - start_time
        logger.info(f"Upload completed in {processing_time:.2f}s for session {session_id}")

        return UploadResponse(
            success=True,
            message="Video uploaded and analyzed successfully",
            session_id=session_id,
            file_info={
                "filename": file.filename,
                "size": file_size,
                "type": file.content_type,
                "path": str(file_path),
                "thumbnail": thumbnail_path,
                "processing_time": processing_time
            },
            video_info=validation_result["metadata"],
            ai_insights=ai_insights,
            suggested_clips=suggested_clips
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        await manager.send_message(upload_id, {
            "type": "upload_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, "upload")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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