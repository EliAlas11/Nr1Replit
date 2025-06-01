
"""
Netflix-Level ViralClip Pro v4.0 - Main Application
Enterprise-grade FastAPI application with real-time features
"""

import asyncio
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

# Local imports
from .config import get_settings
from .logging_config import setup_logging, get_api_logger, PerformanceLogger
from .utils.health import HealthChecker
from .utils.security import SecurityManager
from .utils.rate_limiter import RateLimiter
from .utils.cache import CacheManager
from .services.video_service import VideoService
from .services.realtime_engine import RealtimeEngine
from .services.ai_analyzer import AIAnalyzer
from .db.session import SessionManager

# Initialize settings and logging
settings = get_settings()
setup_logging(settings.log_level, log_file=True)
logger = get_api_logger()
perf_logger = PerformanceLogger()

# Global services
health_checker = None
security_manager = None
rate_limiter = None
cache_manager = None
video_service = None
realtime_engine = None
ai_analyzer = None
session_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Netflix-level application lifespan management"""
    global health_checker, security_manager, rate_limiter, cache_manager
    global video_service, realtime_engine, ai_analyzer, session_manager
    
    try:
        logger.info("🚀 Starting ViralClip Pro v4.0 - Netflix-level application")
        
        # Initialize core services
        health_checker = HealthChecker()
        security_manager = SecurityManager()
        rate_limiter = RateLimiter()
        cache_manager = CacheManager()
        session_manager = SessionManager()
        
        # Initialize video processing services
        video_service = VideoService()
        ai_analyzer = AIAnalyzer()
        realtime_engine = RealtimeEngine()
        
        # Store services in app state
        app.state.health_checker = health_checker
        app.state.security_manager = security_manager
        app.state.rate_limiter = rate_limiter
        app.state.cache_manager = cache_manager
        app.state.video_service = video_service
        app.state.realtime_engine = realtime_engine
        app.state.ai_analyzer = ai_analyzer
        app.state.session_manager = session_manager
        
        # Start background services
        await realtime_engine.start()
        
        # Perform startup health check
        health_status = await health_checker.comprehensive_check()
        if health_status["status"] != "healthy":
            logger.warning(f"Application started with warnings: {health_status}")
        else:
            logger.info("✅ All systems healthy - ViralClip Pro v4.0 ready")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        logger.info("🔄 Shutting down ViralClip Pro v4.0")
        if realtime_engine:
            await realtime_engine.stop()
        logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="ViralClip Pro v4.0",
    description="Netflix-level AI-powered viral video clip generator",
    version="4.0.0",
    docs_url="/api/docs" if settings.environment != "production" else None,
    redoc_url="/api/redoc" if settings.environment != "production" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
app.mount("/static", StaticFiles(directory="nr1copilot/nr1-main/static"), name="static")
app.mount("/public", StaticFiles(directory="nr1copilot/nr1-main/public"), name="public")


# Performance monitoring middleware
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Netflix-level performance monitoring"""
    start_time = asyncio.get_event_loop().time()
    request_id = f"req_{int(start_time * 1000000)}"
    
    try:
        response = await call_next(request)
        duration = (asyncio.get_event_loop().time() - start_time) * 1000
        
        perf_logger.log_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.2f}ms"
        
        return response
        
    except Exception as e:
        duration = (asyncio.get_event_loop().time() - start_time) * 1000
        perf_logger.log_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration=duration,
            error=str(e)
        )
        raise


# Main page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main application page"""
    try:
        html_path = Path("nr1copilot/nr1-main/index.html")
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="Main page not found")
        
        return FileResponse(html_path)
    except Exception as e:
        logger.error(f"Error serving main page: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Health check endpoint
@app.get("/api/v3/health")
async def health_check():
    """Netflix-level health check endpoint"""
    try:
        health_status = await app.state.health_checker.comprehensive_check()
        status_code = 200 if health_status["status"] == "healthy" else 503
        return JSONResponse(content=health_status, status_code=status_code)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503
        )


# Video upload endpoint
@app.post("/api/v3/upload-video")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    upload_id: str = Form(...)
):
    """Netflix-level video upload with real-time progress"""
    try:
        # Rate limiting
        client_ip = request.client.host
        if not await app.state.rate_limiter.check_rate_limit("upload", client_ip):
            raise HTTPException(status_code=429, detail="Upload rate limit exceeded")
        
        # Security validation
        if not app.state.security_manager.validate_file_upload(file):
            raise HTTPException(status_code=400, detail="Invalid file upload")
        
        # Process upload
        result = await app.state.video_service.process_upload(file, upload_id)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload processing failed")


# Live preview generation
@app.post("/api/v3/generate-live-preview")
async def generate_live_preview(request: Request):
    """Generate live preview for timeline segment"""
    try:
        data = await request.json()
        
        session_id = data.get("session_id")
        start_time = data.get("start_time", 0)
        end_time = data.get("end_time", 10)
        quality = data.get("quality", "preview")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID required")
        
        # Generate preview
        result = await app.state.video_service.generate_live_preview(
            session_id, start_time, end_time, quality
        )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live preview generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Preview generation failed")


# Process clips endpoint
@app.post("/api/v3/process-clips")
async def process_clips(request: Request):
    """Process selected clips with viral optimization"""
    try:
        data = await request.json()
        
        session_id = data.get("session_id")
        clips = data.get("clips", [])
        options = data.get("options", {})
        
        if not session_id or not clips:
            raise HTTPException(status_code=400, detail="Session ID and clips required")
        
        # Start clip processing
        result = await app.state.video_service.process_clips(session_id, clips, options)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clip processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Clip processing failed")


# WebSocket endpoints
@app.websocket("/api/v3/ws/app")
async def websocket_main(websocket: WebSocket):
    """Main application WebSocket connection"""
    await websocket.accept()
    connection_id = f"main_{int(asyncio.get_event_loop().time() * 1000000)}"
    
    try:
        await app.state.realtime_engine.add_connection(connection_id, websocket)
        logger.info(f"WebSocket connected: {connection_id}")
        
        while True:
            try:
                data = await websocket.receive_json()
                await app.state.realtime_engine.handle_message(connection_id, data)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "error": "Message processing failed"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await app.state.realtime_engine.remove_connection(connection_id)


@app.websocket("/api/v3/ws/upload/{upload_id}")
async def websocket_upload(websocket: WebSocket, upload_id: str):
    """Upload progress WebSocket"""
    await websocket.accept()
    connection_id = f"upload_{upload_id}"
    
    try:
        await app.state.realtime_engine.add_connection(connection_id, websocket, {"type": "upload", "upload_id": upload_id})
        logger.info(f"Upload WebSocket connected: {connection_id}")
        
        while True:
            try:
                await websocket.receive_text()  # Keep connection alive
                await asyncio.sleep(1)
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        logger.info(f"Upload WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Upload WebSocket error: {str(e)}")
    finally:
        await app.state.realtime_engine.remove_connection(connection_id)


@app.websocket("/api/v3/ws/viral-scores/{session_id}")
async def websocket_viral_scores(websocket: WebSocket, session_id: str):
    """Viral score updates WebSocket"""
    await websocket.accept()
    connection_id = f"viral_{session_id}"
    
    try:
        await app.state.realtime_engine.add_connection(connection_id, websocket, {"type": "viral_scores", "session_id": session_id})
        logger.info(f"Viral scores WebSocket connected: {connection_id}")
        
        while True:
            try:
                await websocket.receive_text()
                await asyncio.sleep(1)
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        logger.info(f"Viral scores WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Viral scores WebSocket error: {str(e)}")
    finally:
        await app.state.realtime_engine.remove_connection(connection_id)


@app.websocket("/api/v3/ws/timeline/{session_id}")
async def websocket_timeline(websocket: WebSocket, session_id: str):
    """Timeline updates WebSocket"""
    await websocket.accept()
    connection_id = f"timeline_{session_id}"
    
    try:
        await app.state.realtime_engine.add_connection(connection_id, websocket, {"type": "timeline", "session_id": session_id})
        logger.info(f"Timeline WebSocket connected: {connection_id}")
        
        while True:
            try:
                await websocket.receive_text()
                await asyncio.sleep(1)
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        logger.info(f"Timeline WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Timeline WebSocket error: {str(e)}")
    finally:
        await app.state.realtime_engine.remove_connection(connection_id)


# File download endpoints
@app.get("/api/v3/download/{clip_id}")
async def download_clip(clip_id: str):
    """Download processed clip"""
    try:
        file_path = await app.state.video_service.get_clip_file(clip_id)
        
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Clip not found")
        
        return FileResponse(
            file_path,
            media_type="video/mp4",
            filename=f"clip_{clip_id}.mp4"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Download failed")


@app.get("/api/v3/preview/{clip_id}")
async def preview_clip(clip_id: str):
    """Get preview URL for clip"""
    try:
        preview_url = await app.state.video_service.get_preview_url(clip_id)
        
        if not preview_url:
            raise HTTPException(status_code=404, detail="Preview not found")
        
        return JSONResponse(content={"preview_url": preview_url})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Preview failed")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Netflix-level HTTP exception handling"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path,
            "timestamp": int(asyncio.get_event_loop().time())
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Netflix-level global exception handling"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": request.url.path,
            "timestamp": int(asyncio.get_event_loop().time())
        }
    )


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
