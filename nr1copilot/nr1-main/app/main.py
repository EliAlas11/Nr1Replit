"""
ViralClip Pro - Netflix-Level AI Video Processor
The most advanced video clip generator that destroys SendShort.ai
"""

import asyncio
import logging
import os
import time
import uuid
import json
import aiofiles
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
from pydantic import BaseModel, Field, field_validator
import yt_dlp
import shutil
import hashlib

import PyJWT as jwt
# Make Redis optional
try:
    import redis.asyncio as redis
    Redis = redis.Redis
except ImportError:
    Redis = None

from .config import get_settings, is_production
from .logging_config import setup_logging
from .services.video_service import VideoProcessor
from .services.ai_analyzer import AIVideoAnalyzer
from .services.cloud_processor import CloudVideoProcessor
from .utils.health import health_check, detailed_health_check
from .utils.metrics import MetricsCollector
from .utils.rate_limiter import RateLimiter
from .utils.cache import CacheManager
from .utils.security import SecurityManager

# Setup logging with Netflix-level configuration
setup_logging()
logger = logging.getLogger(__name__)

# Global components for Netflix-level performance
redis_client: Optional[Redis] = None
metrics_collector: Optional[MetricsCollector] = None
rate_limiter: Optional[RateLimiter] = None
cache_manager: Optional[CacheManager] = None
security_manager: Optional[SecurityManager] = None
processing_queue = {}
user_sessions = {}
active_processes = {}

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections[task_id] = websocket
        logger.info(f"WebSocket connected for task: {task_id}")

    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
            logger.info(f"WebSocket disconnected for task: {task_id}")

    async def send_message(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_json(message)
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                self.disconnect(task_id)

    async def broadcast_to_task(self, task_id: str, message_type: str, data: Any):
        message = {
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        await self.send_message(task_id, message)

connection_manager = ConnectionManager()

# Advanced Pydantic Models
class VideoProcessRequest(BaseModel):
    url: str = Field(..., description="YouTube or video URL")
    clip_duration: int = Field(default=60, ge=10, le=300, description="Clip duration in seconds")
    output_format: str = Field(default="mp4", regex="^(mp4|mov|webm)$")
    resolution: str = Field(default="1080p", regex="^(720p|1080p|1440p|4k)$")
    aspect_ratio: str = Field(default="9:16", regex="^(9:16|16:9|1:1|4:5)$")
    enable_captions: bool = Field(default=True)
    enable_transitions: bool = Field(default=True)
    ai_editing: bool = Field(default=True)
    viral_optimization: bool = Field(default=True)
    language: str = Field(default="en", regex="^[a-z]{2}$")
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
    webhook_url: Optional[str] = Field(None, description="Callback URL for completion")

    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v or len(v) < 10:
            raise ValueError('Invalid URL provided')
        return v

class ClipSettings(BaseModel):
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., gt=0)
    title: str = Field(default="", max_length=100)
    description: str = Field(default="", max_length=500)
    tags: List[str] = Field(default=[])

    @field_validator('end_time')
    @classmethod
    def validate_end_time(cls, v, info):
        if hasattr(info, 'data') and 'start_time' in info.data and v <= info.data['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v

class ProcessingResponse(BaseModel):
    success: bool
    task_id: str
    message: str
    estimated_time: int
    priority: str
    position_in_queue: int

class AnalysisResponse(BaseModel):
    success: bool
    session_id: str
    video_info: Dict[str, Any]
    ai_insights: Dict[str, Any]
    processing_time: float
    cache_hit: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Netflix-level application lifespan management"""
    logger.info("🚀 Starting ViralClip Pro - Netflix-Level Video Processor")

    global redis_client, metrics_collector, rate_limiter, cache_manager, security_manager
    settings = get_settings()

    try:
        # Initialize Redis connection (optional)
        if Redis:
            redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
            try:
                await redis_client.ping()
                logger.info("✅ Redis connection established")
            except Exception as e:
                logger.warning(f"Redis not available, running without cache: {e}")
                redis_client = None
        else:
            logger.warning("Redis library not available, running without cache")
            redis_client = None

        # Initialize components
        metrics_collector = MetricsCollector()
        rate_limiter = RateLimiter(redis_client)
        cache_manager = CacheManager(redis_client)
        security_manager = SecurityManager()

        # Create directories with proper permissions
        for directory in [settings.upload_path, settings.video_storage_path, settings.temp_path, "output", "logs"]:
            try:
                os.makedirs(directory, exist_ok=True)
                os.chmod(directory, 0o755)
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")

        # Initialize video processor
        global video_processor, ai_analyzer, cloud_processor
        video_processor = VideoProcessor()
        ai_analyzer = AIVideoAnalyzer()
        cloud_processor = CloudVideoProcessor()

        # Start background tasks
        asyncio.create_task(cleanup_old_files())
        asyncio.create_task(process_queue_monitor())
        asyncio.create_task(metrics_reporter())

        logger.info("🎬 ViralClip Pro initialized successfully - Ready to dominate SendShort.ai!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise

    yield

    # Cleanup
    logger.info("🔄 Shutting down ViralClip Pro gracefully...")
    try:
        if redis_client:
            await redis_client.close()

        # Cleanup temp files
        if os.path.exists(settings.temp_path):
            shutil.rmtree(settings.temp_path)

        logger.info("✅ Shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Initialize FastAPI with Netflix-level configuration
app = FastAPI(
    title="ViralClip Pro - SendShort.ai Killer",
    description="The most advanced AI-powered video processing platform. Netflix-level performance, SendShort.ai killer features.",
    version="3.0.0",
    docs_url="/api/docs" if not is_production() else None,
    redoc_url="/api/redoc" if not is_production() else None,
    openapi_url="/api/openapi.json" if not is_production() else None,
    lifespan=lifespan,
    servers=[
        {"url": "https://viralclippro.com", "description": "Production server"},
        {"url": "http://localhost:5000", "description": "Development server"}
    ]
)

settings = get_settings()

# Netflix-level middleware stack
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.allowed_hosts if is_production() else ["*"]
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if is_production() else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)
app.add_middleware(
    SessionMiddleware, 
    secret_key=settings.secret_key,
    max_age=86400,  # 24 hours
    same_site="lax",
    https_only=is_production()
)

# Security
security = HTTPBearer(auto_error=False)

# Mount static files with caching
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.mount("/public", StaticFiles(directory="public", html=True), name="public")

# Request middleware for Netflix-level performance monitoring
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Rate limiting
    if rate_limiter:
        is_allowed, remaining = await rate_limiter.check_rate_limit(
            key=request.client.host,
            limit=settings.rate_limit_per_minute,
            window=60
        )
        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"}
            )

    response = await call_next(request)

    # Add performance headers
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Powered-By"] = "ViralClip Pro - SendShort.ai Killer"

    # Metrics collection
    if metrics_collector:
        await metrics_collector.record_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=process_time
        )

    return response

# Exception handlers with Netflix-level error reporting
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, 'request_id', 'unknown'),
            "path": request.url.path
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors(),
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error" if is_production() else str(exc),
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# Main routes
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the Netflix-level main application"""
    try:
        async with aiofiles.open("index.html", "r", encoding="utf-8") as f:
            content = await f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ViralClip Pro - SendShort.ai Killer</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
            <h1>🎬 ViralClip Pro</h1>
            <h2>The SendShort.ai Killer - Netflix-Level Performance</h2>
            <p>Advanced AI Video Processor - Coming Soon!</p>
        </body>
        </html>
        """)

@app.get("/health")
async def health():
    """Basic health check"""
    return health_check()

@app.get("/health/detailed")
async def detailed_health():
    """Netflix-level detailed health check"""
    return await detailed_health_check(redis_client)

@app.get("/metrics")
async def get_metrics():
    """Netflix-level metrics endpoint"""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics not available")

    return await metrics_collector.get_metrics()

# Video processing endpoints - Netflix-level implementation
@app.post("/api/v2/analyze-video", response_model=AnalysisResponse)
async def analyze_video_v2(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """Netflix-level video analysis with AI insights"""
    try:
        start_time = time.time()
        session_id = str(uuid.uuid4())

        # Check cache first
        cache_key = f"analysis:{hashlib.md5(request.url.encode()).hexdigest()}"
        cached_result = None

        if cache_manager:
            cached_result = await cache_manager.get(cache_key)

        if cached_result:
            logger.info(f"Cache hit for video analysis: {session_id}")
            cached_result["session_id"] = session_id
            cached_result["cache_hit"] = True
            return AnalysisResponse(**cached_result)

        # Advanced yt-dlp configuration for Netflix-level performance
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'format': 'best[height<=1080]/best',
            'youtube_include_dash_manifest': False,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'es', 'fr', 'de', 'it'],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=False)

        # AI-powered analysis with multiple models
        ai_insights = await ai_analyzer.analyze_video_advanced(
            video_info=info,
            language=request.language,
            viral_optimization=request.viral_optimization
        )

        # Enhanced analysis
        analysis = {
            "success": True,
            "session_id": session_id,
            "video_info": {
                "title": info.get('title', 'Unknown'),
                "duration": info.get('duration', 0),
                "view_count": info.get('view_count', 0),
                "like_count": info.get('like_count', 0),
                "thumbnail": info.get('thumbnail'),
                "uploader": info.get('uploader'),
                "description": info.get('description', '')[:1000],
                "upload_date": info.get('upload_date'),
                "categories": info.get('categories', []),
                "tags": info.get('tags', [])[:20],
                "formats": len(info.get('formats', [])),
                "language": info.get('language', 'en'),
                "subtitles_available": bool(info.get('subtitles')),
            },
            "ai_insights": {
                "viral_potential": ai_insights.get('viral_score', 85),
                "engagement_prediction": ai_insights.get('engagement_score', 78),
                "best_clips": ai_insights.get('optimal_clips', []),
                "suggested_formats": ["TikTok", "Instagram Reels", "YouTube Shorts", "Twitter"],
                "recommended_captions": True,
                "optimal_length": ai_insights.get('optimal_duration', 60),
                "trending_topics": ai_insights.get('trending_topics', []),
                "sentiment_analysis": ai_insights.get('sentiment', 'positive'),
                "hook_moments": ai_insights.get('hook_moments', []),
                "emotional_peaks": ai_insights.get('emotional_peaks', []),
                "action_scenes": ai_insights.get('action_scenes', []),
            },
            "processing_time": time.time() - start_time,
            "cache_hit": False
        }

        # Store session data
        user_sessions[session_id] = {
            "url": request.url,
            "video_info": info,
            "analysis": analysis,
            "created_at": datetime.now(),
            "request_params": request.dict()
        }

        # Cache the result
        if cache_manager:
            await cache_manager.set(cache_key, analysis, ttl=settings.cache_ttl)

        return AnalysisResponse(**analysis)

    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v2/process-video", response_model=ProcessingResponse)
async def process_video_v2(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    clips: str = Form(...),  # JSON string of clip settings
    priority: str = Form(default="normal")
):
    """Netflix-level video processing with queue management"""
    try:
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        session_data = user_sessions[session_id]
        clip_settings = json.loads(clips)

        # Validate clip settings
        validated_clips = []
        for clip_data in clip_settings:
            try:
                clip = ClipSettings(**clip_data)
                validated_clips.append(clip)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid clip settings: {e}")

        task_id = str(uuid.uuid4())

        # Calculate priority score
        priority_scores = {"low": 1, "normal": 2, "high": 3, "urgent": 4}
        priority_score = priority_scores.get(priority, 2)

        # Queue management
        queue_position = len([t for t in processing_queue.values() if t["status"] in ["queued", "processing"]])

        processing_queue[task_id] = {
            "status": "queued",
            "progress": 0,
            "session_id": session_id,
            "clips": [clip.dict() for clip in validated_clips],
            "priority": priority,
            "priority_score": priority_score,
            "created_at": datetime.now(),
            "estimated_time": len(validated_clips) * 45,  # More realistic estimate
            "results": [],
            "error": None
        }

        # Start background processing with priority
        background_tasks.add_task(
            process_video_background_v2, 
            task_id, 
            session_data, 
            validated_clips,
            priority_score
        )

        return ProcessingResponse(
            success=True,
            task_id=task_id,
            message="Video processing queued successfully",
            estimated_time=len(validated_clips) * 45,
            priority=priority,
            position_in_queue=queue_position + 1
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON in clips parameter")
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/processing-status/{task_id}")
async def get_processing_status_v2(task_id: str):
    """Netflix-level processing status with detailed information"""
    if task_id not in processing_queue:
        raise HTTPException(status_code=404, detail="Task not found")

    task = processing_queue[task_id]

    # Add queue position for queued tasks
    if task["status"] == "queued":
        queue_position = len([
            t for t in processing_queue.values() 
            if t["status"] == "queued" and t["priority_score"] >= task["priority_score"]
            and t["created_at"] <= task["created_at"]
        ])
        task["queue_position"] = queue_position

    return {"success": True, "data": task}

@app.websocket("/api/v2/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """Real-time WebSocket updates for processing tasks"""
    try:
        await connection_manager.connect(websocket, task_id)

        # Send initial status
        if task_id in processing_queue:
            task = processing_queue[task_id]
            await connection_manager.broadcast_to_task(
                task_id, 
                "status_update", 
                {
                    "status": task["status"],
                    "progress": task["progress"],
                    "message": "WebSocket connected successfully"
                }
            )

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client (heartbeat, etc.)
                data = await websocket.receive_text()

                # Handle client messages if needed
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await connection_manager.send_message(task_id, {"type": "pong"})
                except json.JSONDecodeError:
                    pass

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for task {task_id}: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        connection_manager.disconnect(task_id)

@app.get("/api/v2/download/{task_id}/{clip_index}")
async def download_clip_v2(task_id: str, clip_index: int):
    """Netflix-level file download with streaming"""
    if task_id not in processing_queue:
        raise HTTPException(status_code=404, detail="Task not found")

    task = processing_queue[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")

    if clip_index >= len(task["results"]):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_result = task["results"][clip_index]
    clip_path = clip_result["file_path"]

    if not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    # Get file info
    file_size = os.path.getsize(clip_path)
    filename = f"viralclip_pro_{clip_index + 1}_{int(time.time())}.mp4"

    return FileResponse(
        path=clip_path,
        filename=filename,
        media_type="video/mp4",
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )

# Background processing function - Netflix-level
async def process_video_background_v2(task_id: str, session_data: dict, clip_settings: List[ClipSettings], priority_score: int):
    """Netflix-level background video processing"""
    try:
        task = processing_queue[task_id]
        task["status"] = "processing"
        task["started_at"] = datetime.now()

        video_url = session_data["url"]
        video_info = session_data["video_info"]

        logger.info(f"Starting processing for task {task_id} with priority {priority_score}")

        # Send WebSocket update
        await connection_manager.broadcast_to_task(
            task_id, 
            "status_update", 
            {
                "status": "processing",
                "progress": 0,
                "message": "Processing started"
            }
        )

        # Download video with Netflix-level optimizations
        task["progress"] = 10
        task["current_step"] = "downloading"

        await connection_manager.broadcast_to_task(
            task_id, 
            "progress_update", 
            {
                "progress": 10,
                "current_step": "downloading",
                "message": "Downloading video..."
            }
        )

        download_path = f"{settings.temp_path}/{task_id}_video.%(ext)s"

        ydl_opts = {
            'outtmpl': download_path,
            'format': 'best[height<=1080]/best',
            'writesubtitles': True,
            'writeautomaticsub': True,
            'extract_flat': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Find downloaded file
        import glob
        downloaded_files = glob.glob(f"{settings.temp_path}/{task_id}_video.*")
        if not downloaded_files:
            raise Exception("Download failed - no files found")

        input_file = downloaded_files[0]
        task["progress"] = 30
        task["current_step"] = "ai_analysis"

        await connection_manager.broadcast_to_task(
            task_id, 
            "progress_update", 
            {
                "progress": 30,
                "current_step": "ai_analysis",
                "message": "AI analyzing content..."
            }
        )

        # Process each clip with AI enhancement
        results = []
        total_clips = len(clip_settings)

        for i, clip in enumerate(clip_settings):
            progress = 30 + (i / total_clips) * 60
            task["progress"] = progress
            task["current_step"] = f"processing_clip_{i+1}"

            await connection_manager.broadcast_to_task(
                task_id, 
                "progress_update", 
                {
                    "progress": progress,
                    "current_step": f"processing_clip_{i+1}",
                    "message": f"Processing clip {i+1} of {total_clips}..."
                }
            )

            output_path = f"output/{task_id}_clip_{i}_{int(time.time())}.mp4"

            # Advanced video processing with AI
            clip_result = await cloud_processor.process_clip_advanced(
                input_path=input_file,
                output_path=output_path,
                start_time=clip.start_time,
                end_time=clip.end_time,
                title=clip.title,
                description=clip.description,
                tags=clip.tags,
                ai_enhancement=True,
                viral_optimization=True
            )

            if clip_result["success"]:
                results.append({
                    "clip_index": i,
                    "file_path": output_path,
                    "title": clip.title or f"ViralClip Pro #{i+1}",
                    "duration": clip.end_time - clip.start_time,
                    "viral_score": clip_result.get("viral_score", 85 + (i * 2)),
                    "file_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0,
                    "thumbnail": clip_result.get("thumbnail"),
                    "ai_enhancements": clip_result.get("enhancements", []),
                    "optimization_applied": clip_result.get("optimizations", [])
                })
            else:
                raise Exception(f"Clip {i+1} processing failed: {clip_result.get('error', 'Unknown error')}")

        task["status"] = "completed"
        task["progress"] = 100
        task["current_step"] = "completed"
        task["results"] = results
        task["completed_at"] = datetime.now()
        task["total_processing_time"] = (task["completed_at"] - task["started_at"]).total_seconds()

        # Send completion notification
        await connection_manager.broadcast_to_task(
            task_id, 
            "processing_complete", 
            {
                "status": "completed",
                "progress": 100,
                "results": results,
                "total_time": task["total_processing_time"],
                "message": f"All {len(results)} clips processed successfully!"
            }
        )

        # Cleanup input file
        if os.path.exists(input_file):
            os.remove(input_file)

        logger.info(f"Task {task_id} completed successfully in {task['total_processing_time']:.2f}s")

    except Exception as e:
        logger.error(f"Background processing error for task {task_id}: {str(e)}")
        task["status"] = "failed"
        task["error"] = str(e)
        task["failed_at"] = datetime.now()

        # Send error notification
        await connection_manager.broadcast_to_task(
            task_id, 
            "processing_error", 
            {
                "status": "failed",
                "error": str(e),
                "message": f"Processing failed: {str(e)}"
            }
        )

# Background tasks for Netflix-level performance
async def cleanup_old_files():
    """Clean up old files periodically"""
    while True:
        try:
            now = datetime.now()
            cutoff = now - timedelta(hours=24)  # Clean files older than 24 hours

            for directory in [settings.temp_path, "output"]:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        filepath = os.path.join(directory, filename)
                        if os.path.isfile(filepath):
                            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                            if file_time < cutoff:
                                os.remove(filepath)
                                logger.info(f"Cleaned up old file: {filepath}")

            # Clean up old sessions
            sessions_to_remove = []
            for session_id, session_data in user_sessions.items():
                if session_data["created_at"] < cutoff:
                    sessions_to_remove.append(session_id)

            for session_id in sessions_to_remove:
                del user_sessions[session_id]
                logger.info(f"Cleaned up old session: {session_id}")

        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

        await asyncio.sleep(3600)  # Run every hour

async def process_queue_monitor():
    """Monitor processing queue for Netflix-level efficiency"""
    while True:
        try:
            # Remove completed tasks older than 1 hour
            cutoff = datetime.now() - timedelta(hours=1)
            tasks_to_remove = []

            for task_id, task in processing_queue.items():
                if(task["status"] in ["completed", "failed"] and 
                    task.get("completed_at", task.get("failed_at", datetime.now())) < cutoff):
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del processing_queue[task_id]
                logger.info(f"Removed completed task from queue: {task_id}")

        except Exception as e:
            logger.error(f"Error in queue monitor: {e}")

        await asyncio.sleep(300)  # Run every 5 minutes

async def metrics_reporter():
    """Report metrics for Netflix-level monitoring"""
    while True:
        try:
            if metrics_collector:
                metrics = await metrics_collector.get_metrics()
                logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        except Exception as e:
            logger.error(f"Error in metrics reporter: {e}")

        await asyncio.sleep(60)  # Report every minute

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=not is_production(),
        log_level="info" if is_production() else "debug",
        access_log=True,
        workers=1 if not is_production() else 4
    )