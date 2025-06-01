"""
ViralClip Pro - Netflix-Level Video Processing Platform
Advanced FastAPI application with real-time WebSocket updates
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .logging_config import setup_logging
from .schemas import (
    VideoAnalysisRequest, VideoAnalysisResponse, VideoProcessRequest,
    SuccessResponse, ProcessingStatus, ClipDefinition
)
from .services.video_service import VideoProcessor
from .services.ai_analyzer import AIAnalyzer
from .services.cloud_processor import CloudProcessor
from .utils.security import SecurityManager
from .utils.rate_limiter import RateLimiter
from .utils.metrics import MetricsCollector
from .utils.health import HealthChecker
from .utils.cache import CacheManager

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="ViralClip Pro",
    description="Netflix-level video processing platform for creating viral clips",
    version="2.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Initialize services
video_processor = VideoProcessor()
ai_analyzer = AIAnalyzer()
cloud_processor = CloudProcessor()
security_manager = SecurityManager()
rate_limiter = RateLimiter()
metrics = MetricsCollector()
health_checker = HealthChecker()
cache_manager = CacheManager()

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}
upload_connections: Dict[str, WebSocket] = {}

# Processing sessions
processing_sessions: Dict[str, Dict[str, Any]] = {}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)

    headers = security_manager.get_security_headers()
    for header, value in headers.items():
        response.headers[header] = value

    return response

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    client_ip = request.client.host

    # Check if IP is blocked
    if security_manager.is_ip_blocked(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "IP temporarily blocked due to suspicious activity"}
        )

    # Apply rate limiting
    endpoint = str(request.url.path)
    rate_check = security_manager.check_rate_limit(endpoint, client_ip)

    if not rate_check["allowed"]:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": rate_check["retry_after"]
            }
        )

    response = await call_next(request)

    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(rate_check["limit"])
    response.headers["X-RateLimit-Remaining"] = str(rate_check["remaining"])
    response.headers["X-RateLimit-Reset"] = str(int(rate_check["reset_time"]))

    return response

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/public", StaticFiles(directory="public"), name="public")

# Ensure required directories exist
for directory in ["uploads", "output", "videos", "logs"]:
    os.makedirs(directory, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 ViralClip Pro starting up...")

    # Initialize health checker
    await health_checker.initialize()

    # Start background cleanup task
    asyncio.create_task(cleanup_old_files())

    logger.info("✅ ViralClip Pro startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("🛑 ViralClip Pro shutting down...")

    # Close all WebSocket connections
    for connection in active_connections.values():
        await connection.close()

    for connection in upload_connections.values():
        await connection.close()

    logger.info("✅ ViralClip Pro shutdown complete")

# Serve main HTML file
@app.get("/")
async def serve_index():
    """Serve the main application"""
    return FileResponse("index.html")

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/v2/health")
async def detailed_health_check():
    """Detailed health check with system info"""
    health_status = await health_checker.get_detailed_status()
    return health_status

# Metrics endpoint
@app.get("/api/v2/metrics")
async def get_metrics():
    """Get application metrics"""
    return metrics.get_current_metrics()

# Video analysis endpoint
@app.post("/api/v2/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """Analyze video from URL with AI insights"""
    try:
        logger.info(f"Analyzing video: {request.url}")

        # Validate URL
        if not request.url or not request.url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL provided")

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Initialize session
        processing_sessions[session_id] = {
            "status": "analyzing",
            "url": request.url,
            "created_at": datetime.now(),
            "settings": request.dict()
        }

        # Get video information
        video_info = await ai_analyzer.get_video_info(request.url)

        if not video_info:
            raise HTTPException(status_code=404, detail="Video not found or invalid URL")

        # Perform AI analysis
        ai_insights = await ai_analyzer.analyze_content(request.url, {
            "clip_duration": request.clip_duration,
            "target_platforms": request.suggested_formats,
            "enable_viral_optimization": request.viral_optimization
        })

        # Update session
        processing_sessions[session_id].update({
            "status": "completed",
            "video_info": video_info,
            "ai_insights": ai_insights
        })

        return VideoAnalysisResponse(
            success=True,
            session_id=session_id,
            video_info=video_info,
            ai_insights=ai_insights,
            suggested_clips=ai_insights.get("suggested_clips", []),
            processing_time=2.5
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Enhanced file upload endpoint with WebSocket support
@app.post("/api/v2/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    upload_id: str = Form(None)
):
    """Handle direct video uploads with real-time progress"""
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Only video files are allowed")

        # Validate file size (2GB limit)
        file_size = 0
        content = await file.read()
        file_size = len(content)

        if file_size > 2 * 1024 * 1024 * 1024:  # 2GB
            raise HTTPException(status_code=413, detail="File size must be less than 2GB")

        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        # Generate unique identifiers
        file_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        # Create safe filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
        filename = f"upload_{file_id}_{safe_filename}"
        upload_path = f"uploads/{filename}"

        # Create uploads directory with proper permissions
        os.makedirs("uploads", exist_ok=True)

        # Save file with progress tracking
        async with aiofiles.open(upload_path, 'wb') as f:
            await f.write(content)

        # Validate uploaded video
        validation_result = await video_processor.validate_video(upload_path)

        if not validation_result["valid"]:
            # Clean up invalid file
            try:
                os.remove(upload_path)
            except:
                pass
            raise HTTPException(status_code=400, detail=validation_result["error"])

        # Extract thumbnail
        thumbnail_path = await video_processor.extract_thumbnail(upload_path)

        # Initialize processing session
        processing_sessions[session_id] = {
            "status": "uploaded",
            "file_path": upload_path,
            "filename": safe_filename,
            "file_size": file_size,
            "metadata": validation_result.get("metadata", {}),
            "thumbnail": thumbnail_path,
            "created_at": datetime.now(),
            "upload_id": upload_id
        }

        # Notify upload WebSocket if connected
        if upload_id and upload_id in upload_connections:
            await upload_connections[upload_id].send_json({
                "type": "upload_complete",
                "session_id": session_id,
                "file_info": {
                    "filename": safe_filename,
                    "size": file_size,
                    "thumbnail": thumbnail_path
                }
            })

        return {
            "success": True,
            "session_id": session_id,
            "filename": safe_filename,
            "file_size": file_size,
            "metadata": validation_result.get("metadata", {}),
            "thumbnail": thumbnail_path,
            "message": "File uploaded successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Video processing endpoint
@app.post("/api/v2/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    clips: str = Form(...),
    priority: str = Form("normal")
):
    """Process video clips with background task"""
    try:
        # Parse clips data
        try:
            clip_definitions = json.loads(clips)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid clips data format")

        # Validate session
        if session_id not in processing_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = processing_sessions[session_id]

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Update session with task info
        session.update({
            "status": "processing",
            "task_id": task_id,
            "clips": clip_definitions,
            "priority": priority,
            "processing_started": datetime.now()
        })

        # Start background processing
        background_tasks.add_task(
            process_clips_background,
            task_id,
            session_id,
            clip_definitions
        )

        return {
            "success": True,
            "task_id": task_id,
            "session_id": session_id,
            "clips_count": len(clip_definitions),
            "estimated_time": len(clip_definitions) * 30,  # 30 seconds per clip estimate
            "message": "Processing started"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing initiation error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Background processing function
async def process_clips_background(task_id: str, session_id: str, clip_definitions: List[Dict]):
    """Background task for processing video clips"""
    try:
        session = processing_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return

        input_path = session.get("file_path")
        if not input_path or not os.path.exists(input_path):
            await notify_websocket_error(task_id, "Input file not found")
            return

        # Progress callback function
        async def progress_callback(progress_data):
            await notify_websocket_progress(task_id, progress_data)

        # Process clips
        results = await video_processor.batch_process_clips(
            input_path=input_path,
            clip_definitions=clip_definitions,
            progress_callback=progress_callback
        )

        # Update session with results
        session.update({
            "status": "completed",
            "results": results,
            "processing_completed": datetime.now()
        })

        # Notify completion
        await notify_websocket_completion(task_id, results)

    except Exception as e:
        logger.error(f"Background processing error: {e}")
        await notify_websocket_error(task_id, str(e))

# WebSocket helper functions
async def notify_websocket_progress(task_id: str, progress_data: Dict):
    """Notify WebSocket clients of progress updates"""
    if task_id in active_connections:
        try:
            await active_connections[task_id].send_json({
                "type": "progress_update",
                "data": progress_data,
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass

async def notify_websocket_completion(task_id: str, results: List[Dict]):
    """Notify WebSocket clients of completion"""
    if task_id in active_connections:
        try:
            await active_connections[task_id].send_json({
                "type": "processing_complete",
                "data": {"results": results},
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass

async def notify_websocket_error(task_id: str, error: str):
    """Notify WebSocket clients of errors"""
    if task_id in active_connections:
        try:
            await active_connections[task_id].send_json({
                "type": "processing_error",
                "data": {"error": error},
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass

# WebSocket endpoint for processing updates
@app.websocket("/api/v2/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time processing updates"""
    try:
        await websocket.accept()
        active_connections[task_id] = websocket
        logger.info(f"WebSocket connected: {task_id}")

        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "message": "WebSocket connected successfully"
        })

        # Keep connection alive
        while True:
            try:
                # Wait for client messages or timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                # Handle client messages
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send keep-alive ping
                await websocket.send_json({
                    "type": "keep_alive",
                    "timestamp": datetime.now().isoformat()
                })
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up connection
        if task_id in active_connections:
            del active_connections[task_id]
        logger.info(f"WebSocket disconnected: {task_id}")

@app.websocket("/api/v2/upload-progress/{upload_id}")
async def upload_progress_websocket(websocket: WebSocket, upload_id: str):
    """WebSocket endpoint for upload progress updates"""
    try:
        await websocket.accept()
        upload_connections[upload_id] = websocket
        logger.info(f"Upload progress WebSocket connected: {upload_id}")

        # Send initial status
        await websocket.send_json({
            "type": "upload_started",
            "upload_id": upload_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Upload WebSocket connected"
        })

        # Keep connection alive
        while True:
            try:
                # Wait for client messages or timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                # Handle client messages
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send keep-alive ping
                await websocket.send_json({
                    "type": "keep_alive",
                    "timestamp": datetime.now().isoformat()
                })
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"Upload WebSocket error: {e}")
    finally:
        # Clean up connection
        if upload_id in upload_connections:
            del upload_connections[upload_id]
        logger.info(f"Upload WebSocket disconnected: {upload_id}")

# Download endpoint
@app.get("/api/v2/download/{task_id}/{clip_index}")
async def download_clip(task_id: str, clip_index: int):
    """Download processed clip"""
    try:
        # Find session by task_id
        session = None
        for sess in processing_sessions.values():
            if sess.get("task_id") == task_id:
                session = sess
                break

        if not session:
            raise HTTPException(status_code=404, detail="Task not found")

        results = session.get("results", [])
        if clip_index >= len(results):
            raise HTTPException(status_code=404, detail="Clip not found")

        result = results[clip_index]
        if not result.get("success"):
            raise HTTPException(status_code=400, detail="Clip processing failed")

        output_path = result.get("output_path")
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="Output file not found")

        # Generate download filename
        original_filename = session.get("filename", "video")
        name, ext = os.path.splitext(original_filename)
        download_filename = f"{name}_clip_{clip_index + 1}{ext}"

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=download_filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# Processing status endpoint
@app.get("/api/v2/status/{task_id}")
async def get_processing_status(task_id: str):
    """Get processing status for a task"""
    try:
        # Find session by task_id
        session = None
        for sess in processing_sessions.values():
            if sess.get("task_id") == task_id:
                session = sess
                break

        if not session:
            raise HTTPException(status_code=404, detail="Task not found")

        return {
            "task_id": task_id,
            "status": session.get("status", "unknown"),
            "created_at": session.get("created_at"),
            "clips_count": len(session.get("clips", [])),
            "results": session.get("results", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Cleanup function
async def cleanup_old_files():
    """Periodic cleanup of old files"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await video_processor.cleanup_old_files(max_age_hours=24)

            # Clean up old sessions
            now = datetime.now()
            expired_sessions = []

            for session_id, session in processing_sessions.items():
                if (now - session.get("created_at", now)).total_seconds() > 86400:  # 24 hours
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                del processing_sessions[session_id]

            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "Something went wrong"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=settings.debug,
        log_level="info"
    )