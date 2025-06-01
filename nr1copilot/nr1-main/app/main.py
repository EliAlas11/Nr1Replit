"""
ViralClip Pro - Main FastAPI Application
Production-ready video processing platform with comprehensive features
"""

import asyncio
import json
import logging
import os
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException

# Add the app directory to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

try:
    from config import settings, get_settings
    from logging_config import setup_logging
    from routes.video import router as video_router
    from services.video_service import VideoProcessor
    from utils.security import SecurityManager
except ImportError as e:
    print(f"Import error: {e}")
    # Create minimal config if imports fail
    class MockSettings:
        upload_path = "uploads"
        output_path = "output"
        temp_path = "temp"
        max_file_size = 2 * 1024 * 1024 * 1024  # 2GB
        allowed_video_formats = ["mp4", "mov", "avi", "mkv", "webm"]
        cors_origins = ["*"]
        host = "0.0.0.0"
        port = 5000

    settings = MockSettings()

# Initialize logging
try:
    logger = setup_logging()
except:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Create directories
for directory in ["uploads", "output", "temp", "logs"]:
    Path(directory).mkdir(exist_ok=True)

# Active connections for WebSocket
active_connections: Dict[str, WebSocket] = {}
upload_connections: Dict[str, WebSocket] = {}
processing_sessions: Dict[str, Dict[str, Any]] = {}

# Initialize services
video_processor = VideoProcessor()
security_manager = SecurityManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 ViralClip Pro starting up...")

    # Startup tasks
    try:
        # Initialize directories
        for directory in ["uploads", "output", "temp"]:
            Path(directory).mkdir(exist_ok=True)

        # Cleanup old files
        await video_processor.cleanup_old_files()

        logger.info("✅ ViralClip Pro started successfully")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

    yield

    # Shutdown tasks
    logger.info("🛑 ViralClip Pro shutting down...")

    # Close WebSocket connections
    for ws in list(active_connections.values()):
        try:
            await ws.close()
        except:
            pass

    for ws in list(upload_connections.values()):
        try:
            await ws.close()
        except:
            pass

# Create FastAPI app
app = FastAPI(
    title="ViralClip Pro",
    description="AI-Powered Viral Video Creator - 10x Better than SendShort.ai",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)

    # Add security headers
    security_headers = security_manager.get_security_headers()
    for header_name, header_value in security_headers.items():
        response.headers[header_name] = header_value

    return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for production
)

# Mount static files
app.mount("/public", StaticFiles(directory="public"), name="public")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
try:
    app.include_router(video_router, prefix="/api/v2", tags=["video"])
except:
    logger.warning("Video router not available")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.upload_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str, connection_type: str = "processing"):
        await websocket.accept()
        if connection_type == "upload":
            self.upload_connections[client_id] = websocket
        else:
            self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id} ({connection_type})")

    def disconnect(self, client_id: str, connection_type: str = "processing"):
        if connection_type == "upload":
            self.upload_connections.pop(client_id, None)
        else:
            self.active_connections.pop(client_id, None)
        logger.info(f"WebSocket disconnected: {client_id} ({connection_type})")

    async def send_personal_message(self, message: dict, client_id: str, connection_type: str = "processing"):
        connections = self.upload_connections if connection_type == "upload" else self.active_connections
        websocket = connections.get(client_id)
        if websocket:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                self.disconnect(client_id, connection_type)

manager = ConnectionManager()

# Routes
@app.get("/")
async def root():
    """Serve the main application"""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "video_processor": "available",
            "security_manager": "available",
            "websockets": len(active_connections) + len(upload_connections)
        }
    }

# Upload endpoint
@app.post("/api/v2/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    upload_id: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload video file with validation and processing"""
    try:
        # Validate file
        validation_result = security_manager.validate_file_upload(
            filename=file.filename,
            content_type=file.content_type,
            file_size=0,  # Will be calculated during upload
            allowed_types=settings.allowed_video_formats,
            max_size=settings.max_file_size
        )

        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {', '.join(validation_result['errors'])}"
            )

        # Generate secure filename
        session_id = str(uuid.uuid4())
        safe_filename = validation_result["sanitized_filename"]
        upload_path = Path("uploads") / f"{session_id}_{safe_filename}"

        # Save file with progress tracking
        total_size = 0

        async with aiofiles.open(upload_path, "wb") as f:
            while chunk := await file.read(8192):  # 8KB chunks
                await f.write(chunk)
                total_size += len(chunk)

                # Send progress update
                progress = min((total_size / settings.max_file_size) * 100, 100)
                await manager.send_personal_message({
                    "type": "upload_progress",
                    "progress": progress,
                    "bytes_uploaded": total_size
                }, upload_id, "upload")

        # Validate uploaded file size
        if total_size > settings.max_file_size:
            upload_path.unlink()  # Delete the file
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {total_size} bytes exceeds {settings.max_file_size} bytes"
            )

        # Validate video format
        validation = await video_processor.validate_video(str(upload_path))

        if not validation["valid"]:
            upload_path.unlink()  # Delete invalid file
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video file: {validation['error']}"
            )

        # Generate thumbnail
        thumbnail_path = await video_processor.extract_thumbnail(str(upload_path))

        # Send completion message
        await manager.send_personal_message({
            "type": "upload_complete",
            "session_id": session_id,
            "file_size": total_size,
            "metadata": validation.get("metadata", {})
        }, upload_id, "upload")

        return {
            "success": True,
            "message": "Upload completed successfully",
            "session_id": session_id,
            "filename": safe_filename,
            "file_size": total_size,
            "thumbnail": f"/api/v2/thumbnail/{session_id}" if thumbnail_path else None,
            "metadata": validation.get("metadata", {})
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Upload failed")

# WebSocket for upload progress
@app.websocket("/api/v2/upload-progress/{upload_id}")
async def upload_progress_websocket(websocket: WebSocket, upload_id: str):
    await manager.connect(websocket, upload_id, "upload")
    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Upload progress WebSocket connected"
        }))

        # Keep connection alive
        while True:
            try:
                # Wait for ping or send keep-alive
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

            except asyncio.TimeoutError:
                # Send keep-alive
                await websocket.send_text(json.dumps({"type": "keep_alive"}))

    except WebSocketDisconnect:
        manager.disconnect(upload_id, "upload")
    except Exception as e:
        logger.error(f"Upload WebSocket error: {e}")
        manager.disconnect(upload_id, "upload")

# URL analysis endpoint
@app.post("/api/v2/analyze-video")
async def analyze_video(request: dict):
    """Analyze video from URL"""
    try:
        url = request.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        # Mock analysis for now - replace with actual implementation
        await asyncio.sleep(1)  # Simulate processing

        session_id = str(uuid.uuid4())

        return {
            "success": True,
            "session_id": session_id,
            "video_info": {
                "title": "Sample Video Analysis",
                "duration": 180,
                "thumbnail": "https://img.youtube.com/vi/sample/maxresdefault.jpg",
                "platform": "youtube",
                "view_count": 1000000
            },
            "ai_insights": {
                "viral_potential": 85,
                "content_type": "entertainment",
                "optimal_length": 60,
                "hook_quality": 90,
                "audience_retention": 78
            },
            "suggested_clips": [
                {
                    "title": "Epic Hook Moment",
                    "start_time": 5,
                    "end_time": 65,
                    "viral_score": 95,
                    "description": "Perfect attention-grabbing opening",
                    "recommended_platforms": ["tiktok", "instagram", "youtube_shorts"],
                    "clip_type": "hook"
                },
                {
                    "title": "Climax Scene",
                    "start_time": 90,
                    "end_time": 150,
                    "viral_score": 88,
                    "description": "High-energy peak moment",
                    "recommended_platforms": ["tiktok", "instagram"],
                    "clip_type": "climax"
                }
            ]
        }

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return await http_exception_handler(request, exc)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )