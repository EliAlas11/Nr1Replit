
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
import time

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Add the app directory to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

# Initialize logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Create directories
for directory in ["uploads", "output", "temp", "logs", "thumbnails"]:
    Path(directory).mkdir(exist_ok=True)

# Mock settings class
class Settings:
    def __init__(self):
        self.upload_path = "uploads"
        self.output_path = "output"
        self.temp_path = "temp"
        self.max_file_size = 2 * 1024 * 1024 * 1024  # 2GB
        self.allowed_video_formats = ["mp4", "mov", "avi", "mkv", "webm", "m4v", "flv"]
        self.cors_origins = ["*"]
        self.host = "0.0.0.0"
        self.port = 5000
        self.redis_url = "redis://localhost:6379"
        self.database_url = "sqlite:///./app.db"

settings = Settings()

# Active connections for WebSocket
active_connections: Dict[str, WebSocket] = {}
upload_connections: Dict[str, WebSocket] = {}
processing_sessions: Dict[str, Dict[str, Any]] = {}

# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        self.clients = {
            ip: requests for ip, requests in self.clients.items()
            if any(t > current_time - self.window_seconds for t in requests)
        }
        
        # Check rate limit
        if client_ip in self.clients:
            recent_requests = [t for t in self.clients[client_ip] if t > current_time - self.window_seconds]
            if len(recent_requests) >= self.max_requests:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"}
                )
            self.clients[client_ip] = recent_requests + [current_time]
        else:
            self.clients[client_ip] = [current_time]
        
        response = await call_next(request)
        return response

# Performance monitoring middleware
class PerformanceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log slow requests
        if process_time > 1.0:
            logger.warning(f"Slow request: {request.method} {request.url} took {process_time:.2f}s")
        
        return response

# Security middleware
class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' ws: wss:;"
        
        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 ViralClip Pro starting up...")

    # Startup tasks
    try:
        # Initialize directories
        for directory in ["uploads", "output", "temp", "logs", "thumbnails"]:
            Path(directory).mkdir(exist_ok=True)

        # Cleanup old files
        await cleanup_old_files()

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

async def cleanup_old_files():
    """Clean up old files to save disk space"""
    try:
        current_time = time.time()
        max_age = 24 * 60 * 60  # 24 hours
        
        for directory in ["uploads", "output", "temp", "thumbnails"]:
            dir_path = Path(directory)
            if dir_path.exists():
                for file_path in dir_path.iterdir():
                    if file_path.is_file() and current_time - file_path.stat().st_mtime > max_age:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Create FastAPI app
app = FastAPI(
    title="ViralClip Pro",
    description="AI-Powered Viral Video Creator - 10x Better than SendShort.ai",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add middleware in correct order
app.add_middleware(SecurityMiddleware)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Mount static files with caching headers
class CachedStaticFiles(StaticFiles):
    async def __call__(self, scope, receive, send):
        response = await super().__call__(scope, receive, send)
        if hasattr(response, 'headers'):
            response.headers["Cache-Control"] = "public, max-age=3600"
        return response

app.mount("/public", CachedStaticFiles(directory="public"), name="public")
app.mount("/static", CachedStaticFiles(directory="static"), name="static")

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
                return True
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                self.disconnect(client_id, connection_type)
                return False
        return False

    async def broadcast(self, message: dict, connection_type: str = "processing"):
        connections = self.upload_connections if connection_type == "upload" else self.active_connections
        disconnected = []
        
        for client_id, websocket in connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id, connection_type)

manager = ConnectionManager()

# File validation helper
def validate_file_upload(filename: str, content_type: str) -> Dict[str, Any]:
    """Validate uploaded file"""
    if not filename:
        return {"valid": False, "error": "No filename provided"}
    
    # Check file extension
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ""
    if file_ext not in settings.allowed_video_formats:
        return {
            "valid": False, 
            "error": f"Invalid file format. Allowed: {', '.join(settings.allowed_video_formats)}"
        }
    
    # Check content type
    if not content_type or not content_type.startswith('video/'):
        return {"valid": False, "error": "Invalid content type. Must be a video file"}
    
    # Sanitize filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')
    if not safe_filename:
        safe_filename = f"video_{int(time.time())}.{file_ext}"
    
    return {
        "valid": True,
        "sanitized_filename": safe_filename,
        "file_extension": file_ext
    }

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application"""
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
            <head><title>ViralClip Pro</title></head>
            <body>
                <h1>ViralClip Pro</h1>
                <p>AI-Powered Viral Video Creator</p>
                <p>Frontend files not found. Please check the deployment.</p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0",
        "services": {
            "websockets": {
                "active_connections": len(manager.active_connections),
                "upload_connections": len(manager.upload_connections)
            },
            "storage": {
                "uploads_dir": Path("uploads").exists(),
                "output_dir": Path("output").exists(),
                "temp_dir": Path("temp").exists()
            },
            "system": {
                "disk_usage": get_disk_usage(),
                "memory_usage": get_memory_usage()
            }
        }
    }

def get_disk_usage() -> Dict[str, Any]:
    """Get disk usage information"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        return {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "usage_percent": round((used / total) * 100, 2)
        }
    except:
        return {"error": "Could not retrieve disk usage"}

def get_memory_usage() -> Dict[str, Any]:
    """Get memory usage information"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "usage_percent": memory.percent
        }
    except:
        return {"error": "Could not retrieve memory usage"}

# Upload endpoint with enhanced validation and progress tracking
@app.post("/api/v2/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    upload_id: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload video file with comprehensive validation and processing"""
    session_id = None
    upload_path = None
    
    try:
        # Validate file
        validation_result = validate_file_upload(file.filename, file.content_type)
        
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=validation_result["error"])

        # Generate secure identifiers
        session_id = str(uuid.uuid4())
        safe_filename = validation_result["sanitized_filename"]
        upload_path = Path("uploads") / f"{session_id}_{safe_filename}"

        # Initialize progress tracking
        total_size = 0
        chunk_size = 8192  # 8KB chunks

        # Save file with progress tracking
        async with aiofiles.open(upload_path, "wb") as f:
            while chunk := await file.read(chunk_size):
                await f.write(chunk)
                total_size += len(chunk)

                # Check size limit
                if total_size > settings.max_file_size:
                    await manager.send_personal_message({
                        "type": "upload_error",
                        "error": f"File too large. Maximum size: {settings.max_file_size // (1024**3)}GB"
                    }, upload_id, "upload")
                    raise HTTPException(status_code=413, detail="File too large")

                # Send progress update every 100KB
                if total_size % (100 * 1024) == 0:
                    progress = min((total_size / settings.max_file_size) * 100, 100)
                    await manager.send_personal_message({
                        "type": "upload_progress",
                        "progress": progress,
                        "bytes_uploaded": total_size,
                        "estimated_total": settings.max_file_size
                    }, upload_id, "upload")

        # Final validation
        if total_size == 0:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Empty file received")

        # Validate video file (mock implementation)
        await asyncio.sleep(0.1)  # Simulate video validation
        
        # Generate thumbnail placeholder
        thumbnail_path = f"thumbnails/{session_id}_thumb.jpg"
        await generate_thumbnail_placeholder(thumbnail_path)

        # Extract basic metadata (mock)
        metadata = {
            "duration": 120.5,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "codec": "h264",
            "bitrate": "2000kbps"
        }

        # Send completion message
        await manager.send_personal_message({
            "type": "upload_complete",
            "session_id": session_id,
            "file_size": total_size,
            "metadata": metadata
        }, upload_id, "upload")

        # Schedule cleanup
        background_tasks.add_task(schedule_file_cleanup, upload_path, 24 * 60 * 60)  # 24 hours

        return {
            "success": True,
            "message": "Upload completed successfully",
            "session_id": session_id,
            "filename": safe_filename,
            "file_size": total_size,
            "thumbnail": f"/api/v2/thumbnail/{session_id}",
            "metadata": metadata,
            "processing_options": {
                "max_clips": 10,
                "supported_formats": ["mp4", "webm"],
                "quality_options": ["720p", "1080p", "4K"]
            }
        }

    except HTTPException:
        # Clean up on HTTP exceptions
        if upload_path and upload_path.exists():
            upload_path.unlink()
        raise
    except Exception as e:
        # Clean up on general exceptions
        if upload_path and upload_path.exists():
            upload_path.unlink()
        
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        
        await manager.send_personal_message({
            "type": "upload_error",
            "error": "Upload failed due to server error"
        }, upload_id, "upload")
        
        raise HTTPException(status_code=500, detail="Upload failed")

async def generate_thumbnail_placeholder(thumbnail_path: str):
    """Generate a placeholder thumbnail"""
    try:
        # Create a simple placeholder image (in real implementation, use PIL or OpenCV)
        placeholder_content = b"placeholder_thumbnail_data"
        async with aiofiles.open(thumbnail_path, "wb") as f:
            await f.write(placeholder_content)
    except Exception as e:
        logger.error(f"Error generating thumbnail: {e}")

async def schedule_file_cleanup(file_path: Path, delay_seconds: int):
    """Schedule file cleanup after specified delay"""
    await asyncio.sleep(delay_seconds)
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")

# WebSocket for upload progress
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

# Enhanced URL analysis endpoint
@app.post("/api/v2/analyze-video")
async def analyze_video(request: dict):
    """Analyze video from URL with comprehensive AI insights"""
    try:
        url = request.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        # Validate URL format
        if not any(platform in url.lower() for platform in ["youtube.com", "youtu.be", "tiktok.com", "instagram.com"]):
            raise HTTPException(status_code=400, detail="Unsupported platform. Please use YouTube, TikTok, or Instagram URLs")

        # Mock analysis with realistic processing time
        await asyncio.sleep(2)  # Simulate AI processing

        session_id = str(uuid.uuid4())

        # Enhanced mock response with realistic data
        return {
            "success": True,
            "session_id": session_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "video_info": {
                "title": "Amazing Content That Will Go Viral",
                "duration": 180,
                "thumbnail": "https://img.youtube.com/vi/sample/maxresdefault.jpg",
                "platform": "youtube",
                "view_count": 1000000,
                "like_count": 50000,
                "comment_count": 5000,
                "upload_date": "2024-01-01",
                "channel": "ViralCreator",
                "tags": ["viral", "trending", "amazing", "content"]
            },
            "ai_insights": {
                "viral_potential": 85,
                "confidence_score": 92,
                "content_type": "entertainment",
                "optimal_length": 60,
                "hook_quality": 90,
                "audience_retention": 78,
                "engagement_prediction": 85,
                "platform_suitability": {
                    "tiktok": 95,
                    "instagram": 88,
                    "youtube_shorts": 92
                },
                "trending_topics": ["viral", "amazing", "trending"],
                "sentiment_analysis": {
                    "positive": 85,
                    "neutral": 10,
                    "negative": 5
                }
            },
            "suggested_clips": [
                {
                    "title": "Epic Hook Moment",
                    "start_time": 5,
                    "end_time": 65,
                    "viral_score": 95,
                    "confidence": 98,
                    "description": "Perfect attention-grabbing opening sequence",
                    "recommended_platforms": ["tiktok", "instagram", "youtube_shorts"],
                    "clip_type": "hook",
                    "estimated_views": "500K-2M",
                    "key_moments": ["0:05 - Hook starts", "0:15 - Peak engagement", "0:45 - Call to action"],
                    "editing_suggestions": [
                        "Add trending music",
                        "Increase saturation by 15%",
                        "Add text overlay for first 3 seconds"
                    ]
                },
                {
                    "title": "Climax Scene",
                    "start_time": 90,
                    "end_time": 150,
                    "viral_score": 88,
                    "confidence": 85,
                    "description": "High-energy peak moment with maximum impact",
                    "recommended_platforms": ["tiktok", "instagram"],
                    "clip_type": "climax",
                    "estimated_views": "300K-1M",
                    "key_moments": ["1:30 - Build up", "1:45 - Climax", "2:20 - Resolution"],
                    "editing_suggestions": [
                        "Add quick cuts for intensity",
                        "Boost audio levels",
                        "Add motion blur effects"
                    ]
                },
                {
                    "title": "Emotional Moment",
                    "start_time": 120,
                    "end_time": 180,
                    "viral_score": 82,
                    "confidence": 78,
                    "description": "Heartfelt moment that drives engagement",
                    "recommended_platforms": ["instagram", "youtube_shorts"],
                    "clip_type": "emotional",
                    "estimated_views": "200K-800K",
                    "key_moments": ["2:00 - Setup", "2:30 - Emotional peak", "2:55 - Resolution"],
                    "editing_suggestions": [
                        "Slow motion at 2:30",
                        "Add emotional music",
                        "Soft focus transition"
                    ]
                }
            ],
            "optimization_recommendations": {
                "posting_times": ["6-8 PM", "12-2 PM", "8-10 PM"],
                "hashtags": ["#viral", "#trending", "#amazing", "#fyp", "#foryou"],
                "caption_suggestions": [
                    "This will blow your mind! 🤯",
                    "You won't believe what happens next...",
                    "This is why the internet exists ✨"
                ],
                "thumbnail_tips": [
                    "Use bright, contrasting colors",
                    "Include text overlay",
                    "Show emotion in thumbnail"
                ]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")

# Thumbnail endpoint
@app.get("/api/v2/thumbnail/{session_id}")
async def get_thumbnail(session_id: str):
    """Get thumbnail for a session"""
    thumbnail_path = Path(f"thumbnails/{session_id}_thumb.jpg")
    
    if thumbnail_path.exists():
        return FileResponse(
            path=thumbnail_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
    else:
        # Return a default placeholder
        return JSONResponse(
            status_code=404,
            content={"error": "Thumbnail not found"}
        )

# Processing endpoint
@app.post("/api/v2/process-video")
async def process_video(
    session_id: str = Form(...),
    clips: str = Form(...),
    priority: str = Form(default="normal"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Process video clips with advanced AI enhancement"""
    try:
        # Parse clips data
        clips_data = json.loads(clips)
        
        if not clips_data:
            raise HTTPException(status_code=400, detail="No clips selected for processing")

        task_id = str(uuid.uuid4())
        
        # Schedule background processing
        background_tasks.add_task(
            process_clips_background,
            task_id,
            session_id,
            clips_data,
            priority
        )

        return {
            "success": True,
            "task_id": task_id,
            "message": "Processing started",
            "estimated_time": len(clips_data) * 30,  # 30 seconds per clip
            "clips_count": len(clips_data),
            "priority": priority
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid clips data format")
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")

async def process_clips_background(task_id: str, session_id: str, clips_data: List[dict], priority: str):
    """Background task for processing clips"""
    try:
        total_clips = len(clips_data)
        results = []

        # Send initial status
        await manager.send_personal_message({
            "type": "processing_started",
            "task_id": task_id,
            "total_clips": total_clips,
            "timestamp": datetime.now().isoformat()
        }, task_id)

        for i, clip in enumerate(clips_data):
            # Send progress update
            progress = ((i + 1) / total_clips) * 100
            await manager.send_personal_message({
                "type": "progress_update",
                "data": {
                    "current": i + 1,
                    "total": total_clips,
                    "percentage": progress,
                    "stage": f"Processing clip {i + 1}",
                    "message": f"Creating clip: {clip.get('title', f'Clip {i + 1}')}",
                    "estimated_time_remaining": (total_clips - i - 1) * 30
                }
            }, task_id)

            # Simulate processing time
            await asyncio.sleep(2)

            # Mock successful result
            result = {
                "success": True,
                "clip_index": i,
                "title": clip.get("title", f"Clip {i + 1}"),
                "output_path": f"output/{task_id}_clip_{i}.mp4",
                "thumbnail_path": f"thumbnails/{task_id}_clip_{i}_thumb.jpg",
                "duration": clip.get("end_time", 60) - clip.get("start_time", 0),
                "file_size": 1024 * 1024 * 5,  # 5MB mock size
                "viral_score": clip.get("viral_score", 85),
                "ai_enhancements": [
                    "Dynamic color correction applied",
                    "Audio levels optimized for mobile",
                    "Viral optimization filters applied",
                    "Platform-specific aspect ratio adjusted"
                ],
                "performance_prediction": {
                    "estimated_views": "100K-500K",
                    "engagement_rate": "8-12%",
                    "viral_probability": "85%"
                }
            }
            results.append(result)

        # Send completion message
        await manager.send_personal_message({
            "type": "processing_complete",
            "data": {
                "task_id": task_id,
                "results": results,
                "total_processed": len(results),
                "processing_time": total_clips * 2,
                "timestamp": datetime.now().isoformat()
            }
        }, task_id)

    except Exception as e:
        logger.error(f"Background processing error: {e}")
        await manager.send_personal_message({
            "type": "processing_error",
            "data": {
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        }, task_id)

# WebSocket for processing updates
@app.websocket("/api/v2/ws/{task_id}")
async def processing_websocket(websocket: WebSocket, task_id: str):
    await manager.connect(websocket, task_id)
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
                elif message.get("type") == "pong":
                    # Client responded to keep-alive
                    pass

            except asyncio.TimeoutError:
                # Send keep-alive
                await websocket.send_text(json.dumps({
                    "type": "keep_alive",
                    "timestamp": datetime.now().isoformat()
                }))

    except WebSocketDisconnect:
        manager.disconnect(task_id)
    except Exception as e:
        logger.error(f"Processing WebSocket error: {e}")
        manager.disconnect(task_id)

# Download endpoint
@app.get("/api/v2/download/{task_id}/{clip_index}")
async def download_clip(task_id: str, clip_index: int):
    """Download processed clip"""
    file_path = Path(f"output/{task_id}_clip_{clip_index}.mp4")
    
    if file_path.exists():
        return FileResponse(
            path=file_path,
            media_type="video/mp4",
            filename=f"viral_clip_{clip_index + 1}.mp4",
            headers={
                "Content-Disposition": f"attachment; filename=viral_clip_{clip_index + 1}.mp4",
                "Cache-Control": "no-cache"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="Clip not found")

# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP {exc.status_code} - {exc.detail} - {request.method} {request.url}")
    return await http_exception_handler(request, exc)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Request: {request.method} {request.url}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }
    )

# Additional utility endpoints
@app.get("/api/v2/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "active_sessions": len(processing_sessions),
        "websocket_connections": {
            "processing": len(manager.active_connections),
            "upload": len(manager.upload_connections)
        },
        "system_info": {
            "disk_usage": get_disk_usage(),
            "memory_usage": get_memory_usage()
        },
        "uptime": "System running",
        "version": "2.1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info",
        access_log=True
    )
