
"""
ViralClip Pro v5.0 - Netflix-Level Architecture
Enterprise-grade video processing platform with real-time AI analysis
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import (
    FastAPI, 
    HTTPException, 
    Depends, 
    WebSocket, 
    WebSocketDisconnect,
    File, 
    UploadFile, 
    Form,
    BackgroundTasks,
    Request,
    Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import aiofiles

from app.config import settings
from app.logging_config import setup_logging
from app.schemas import (
    VideoUploadResponse,
    AnalysisRequest,
    AnalysisResponse,
    PreviewRequest,
    PreviewResponse,
    ProcessingStatus,
    SystemHealth,
    ErrorResponse
)
from app.services.video_service import VideoService
from app.services.ai_analyzer import AIAnalyzer
from app.services.realtime_engine import RealtimeEngine
from app.services.cloud_processor import CloudProcessor
from app.utils.metrics import MetricsCollector
from app.utils.health import HealthChecker
from app.utils.rate_limiter import RateLimiter
from app.utils.security import SecurityManager
from app.utils.cache import CacheManager

# Initialize logging
logger = setup_logging()

# Global services
video_service: Optional[VideoService] = None
ai_analyzer: Optional[AIAnalyzer] = None
realtime_engine: Optional[RealtimeEngine] = None
cloud_processor: Optional[CloudProcessor] = None
metrics_collector: Optional[MetricsCollector] = None
health_checker: Optional[HealthChecker] = None
rate_limiter: Optional[RateLimiter] = None
security_manager: Optional[SecurityManager] = None
cache_manager: Optional[CacheManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Netflix-level application lifecycle management"""
    global video_service, ai_analyzer, realtime_engine, cloud_processor
    global metrics_collector, health_checker, rate_limiter, security_manager, cache_manager
    
    logger.info("🚀 Starting ViralClip Pro v5.0 - Netflix-Level Platform")
    
    try:
        # Initialize core services
        video_service = VideoService()
        ai_analyzer = AIAnalyzer()
        realtime_engine = RealtimeEngine()
        cloud_processor = CloudProcessor()
        
        # Initialize utility services
        metrics_collector = MetricsCollector()
        health_checker = HealthChecker()
        rate_limiter = RateLimiter()
        security_manager = SecurityManager()
        cache_manager = CacheManager()
        
        # Setup directories
        await setup_directories()
        
        # Initialize services
        await ai_analyzer.initialize()
        await realtime_engine.initialize()
        await cloud_processor.initialize()
        await metrics_collector.initialize()
        await cache_manager.initialize()
        
        # Start background services
        await realtime_engine.start()
        await metrics_collector.start()
        
        logger.info("✅ All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    finally:
        # Graceful shutdown
        logger.info("🔄 Shutting down services...")
        
        if realtime_engine:
            await realtime_engine.stop()
        if cloud_processor:
            await cloud_processor.cleanup()
        if metrics_collector:
            await metrics_collector.stop()
        if cache_manager:
            await cache_manager.cleanup()
            
        logger.info("✅ Shutdown complete")


# Create FastAPI app with Netflix-level configuration
app = FastAPI(
    title="ViralClip Pro v5.0",
    description="Netflix-Level AI Video Processing Platform",
    version="5.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Security
security = HTTPBearer(auto_error=False)

# Middleware stack (Netflix-level)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="nr1copilot/nr1-main/static"), name="static")
app.mount("/public", StaticFiles(directory="nr1copilot/nr1-main/public"), name="public")


async def setup_directories():
    """Setup required directories"""
    directories = [
        settings.upload_path,
        settings.output_path,
        settings.temp_path,
        settings.cache_path,
        Path("nr1copilot/nr1-main/logs")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Netflix-level authentication"""
    if not credentials and settings.require_auth:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if credentials and security_manager:
        user = await security_manager.validate_token(credentials.credentials)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user
    
    return {"user_id": "anonymous", "permissions": ["read", "write"]}


async def check_rate_limit(request: Request):
    """Netflix-level rate limiting"""
    if not rate_limiter:
        return
        
    client_ip = request.client.host
    if not await rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main application"""
    try:
        async with aiofiles.open("nr1copilot/nr1-main/index.html", mode="r") as f:
            content = await f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Failed to serve root: {e}")
        return HTMLResponse("<h1>ViralClip Pro v5.0</h1><p>Service unavailable</p>", status_code=500)


# Health and monitoring endpoints
@app.get("/api/v5/health", response_model=SystemHealth)
async def health_check():
    """Netflix-level health monitoring"""
    try:
        health_data = await health_checker.get_system_health() if health_checker else {
            "status": "healthy",
            "services": {"core": "online"},
            "metrics": {"uptime": time.time()}
        }
        return SystemHealth(**health_data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemHealth(
            status="unhealthy",
            services={"core": "error"},
            metrics={"error": str(e)}
        )


@app.get("/api/v5/metrics")
async def get_metrics():
    """Netflix-level metrics endpoint"""
    try:
        return await metrics_collector.get_metrics() if metrics_collector else {"status": "metrics_disabled"}
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"error": str(e)}


# Video upload endpoint
@app.post("/api/v5/upload-video", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
    user=Depends(get_current_user),
    _=Depends(check_rate_limit)
):
    """Netflix-level video upload with real-time processing"""
    start_time = time.time()
    
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        if file.size and file.size > settings.max_file_size:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Generate session ID
        session_id = upload_id or f"session_{uuid.uuid4().hex[:16]}"
        
        # Save uploaded file
        file_path = settings.upload_path / f"{session_id}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"📁 File uploaded: {file_path} ({len(content)} bytes)")
        
        # Start real-time analysis
        analysis_result = await realtime_engine.start_realtime_analysis(
            session_id, str(file_path)
        )
        
        # Generate instant preview
        preview_result = await realtime_engine.generate_instant_previews(
            session_id, str(file_path)
        )
        
        # Track metrics
        if metrics_collector:
            await metrics_collector.track_upload(
                file_size=len(content),
                processing_time=time.time() - start_time,
                user_id=user.get("user_id", "anonymous")
            )
        
        return VideoUploadResponse(
            success=True,
            session_id=session_id,
            file_path=str(file_path),
            file_size=len(content),
            analysis=analysis_result,
            preview=preview_result,
            processing_time=time.time() - start_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# AI Analysis endpoint
@app.post("/api/v5/analyze", response_model=AnalysisResponse)
async def analyze_video(
    request: AnalysisRequest,
    user=Depends(get_current_user),
    _=Depends(check_rate_limit)
):
    """Netflix-level AI video analysis"""
    try:
        if not video_service or not ai_analyzer:
            raise HTTPException(status_code=503, detail="Services not available")
        
        # Validate video file
        if not Path(request.file_path).exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Perform comprehensive analysis
        analysis_result = await ai_analyzer.analyze_video_comprehensive(
            file_path=request.file_path,
            title=request.title,
            description=request.description,
            target_platforms=request.target_platforms,
            custom_prompts=request.custom_prompts
        )
        
        # Cache results
        if cache_manager:
            await cache_manager.set(f"analysis_{request.session_id}", analysis_result, ttl=3600)
        
        return AnalysisResponse(
            success=True,
            session_id=request.session_id,
            analysis=analysis_result,
            cached=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Preview generation endpoint
@app.post("/api/v5/generate-preview", response_model=PreviewResponse)
async def generate_preview(
    request: PreviewRequest,
    user=Depends(get_current_user),
    _=Depends(check_rate_limit)
):
    """Netflix-level instant preview generation"""
    try:
        if not realtime_engine:
            raise HTTPException(status_code=503, detail="Realtime engine not available")
        
        # Generate live preview
        preview_data = await realtime_engine.generate_live_preview(
            session_id=request.session_id,
            start_time=request.start_time,
            end_time=request.end_time,
            quality=request.quality
        )
        
        return PreviewResponse(
            success=True,
            session_id=request.session_id,
            preview_url=preview_data["preview_url"],
            viral_analysis=preview_data.get("viral_analysis", {}),
            suggestions=preview_data.get("suggestions", []),
            processing_time=preview_data.get("processing_time", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Preview generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


# Timeline data endpoint
@app.get("/api/v5/timeline/{session_id}")
async def get_timeline_data(
    session_id: str,
    user=Depends(get_current_user),
    _=Depends(check_rate_limit)
):
    """Get interactive timeline data"""
    try:
        if not realtime_engine:
            raise HTTPException(status_code=503, detail="Realtime engine not available")
        
        timeline_data = await realtime_engine.get_timeline_data(session_id)
        return {"success": True, "timeline": timeline_data}
        
    except Exception as e:
        logger.error(f"❌ Timeline data failed: {e}")
        raise HTTPException(status_code=500, detail=f"Timeline data failed: {str(e)}")


# Processing status endpoint
@app.get("/api/v5/status/{session_id}", response_model=ProcessingStatus)
async def get_processing_status(
    session_id: str,
    user=Depends(get_current_user)
):
    """Get processing status"""
    try:
        if realtime_engine and session_id in realtime_engine.active_sessions:
            session_data = realtime_engine.active_sessions[session_id]
            return ProcessingStatus(
                session_id=session_id,
                status=session_data.get("status", "unknown"),
                progress=session_data.get("progress", 0),
                stage=session_data.get("current_stage", "unknown"),
                message=session_data.get("message", ""),
                start_time=session_data.get("start_time"),
                estimated_completion=session_data.get("estimated_completion")
            )
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# WebSocket endpoints
@app.websocket("/api/v5/ws/app")
async def websocket_main(websocket: WebSocket):
    """Main application WebSocket"""
    connection_id = f"main_{uuid.uuid4().hex[:16]}"
    await websocket.accept()
    
    if realtime_engine:
        await realtime_engine.handle_main_websocket(websocket, connection_id)
    else:
        await websocket.close(code=1000, reason="Service unavailable")


@app.websocket("/api/v5/ws/upload/{upload_id}")
async def websocket_upload(websocket: WebSocket, upload_id: str):
    """Upload progress WebSocket"""
    await websocket.accept()
    
    if realtime_engine:
        await realtime_engine.handle_upload_websocket(websocket, upload_id)
    else:
        await websocket.close(code=1000, reason="Service unavailable")


# File serving endpoints
@app.get("/api/v5/preview/{session_id}/{timestamp}")
async def serve_preview(session_id: str, timestamp: str):
    """Serve preview video files"""
    try:
        # Construct file path (mock for now)
        file_path = settings.output_path / f"preview_{session_id}_{timestamp}.mp4"
        
        if file_path.exists():
            return FileResponse(
                file_path,
                media_type="video/mp4",
                headers={"Cache-Control": "max-age=3600"}
            )
        else:
            raise HTTPException(status_code=404, detail="Preview not found")
            
    except Exception as e:
        logger.error(f"❌ Preview serving failed: {e}")
        raise HTTPException(status_code=500, detail="Preview serving failed")


@app.get("/api/v5/thumbnail/{session_id}/{timestamp}")
async def serve_thumbnail(session_id: str, timestamp: str):
    """Serve thumbnail images"""
    try:
        # Construct file path (mock for now)
        file_path = settings.output_path / f"thumb_{session_id}_{timestamp}.jpg"
        
        if file_path.exists():
            return FileResponse(
                file_path,
                media_type="image/jpeg",
                headers={"Cache-Control": "max-age=7200"}
            )
        else:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
            
    except Exception as e:
        logger.error(f"❌ Thumbnail serving failed: {e}")
        raise HTTPException(status_code=500, detail="Thumbnail serving failed")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Netflix-level error handling"""
    error_id = str(uuid.uuid4())
    
    logger.error(f"HTTP Error {error_id}: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_id": error_id,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    error_id = str(uuid.uuid4())
    
    logger.error(f"Unexpected Error {error_id}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_id": error_id,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=True
    )
