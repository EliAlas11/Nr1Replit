"""
ViralClip Pro v4.0 - Netflix-Level Main Application
Enterprise-grade FastAPI application with real-time features
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .config import get_settings
from .logging_config import setup_logging, get_logger
from .schemas import *
from .services.video_service import VideoService
from .services.realtime_engine import RealtimeEngine
from .utils.health import HealthChecker
from .utils.security import SecurityManager
from .utils.metrics import MetricsCollector

# Initialize configuration and logging
settings = get_settings()
setup_logging(settings.log_level, settings.log_file)
logger = get_logger("main")

# Global services
video_service = VideoService()
realtime_engine = RealtimeEngine()
health_checker = HealthChecker()
security_manager = SecurityManager()
metrics_collector = MetricsCollector()

# Application state
upload_sessions: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}
processing_queue: asyncio.Queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Netflix-level application lifespan management"""
    try:
        logger.info("🚀 Starting ViralClip Pro v4.0 - Netflix-level application")

        # Initialize services
        await video_service.initialize()
        await realtime_engine.initialize()

        # Create required directories
        for directory in [settings.upload_path, settings.output_path, settings.temp_path, "cache", "logs"]:
            Path(directory).mkdir(exist_ok=True)

        # Start background tasks
        asyncio.create_task(process_upload_queue())
        asyncio.create_task(cleanup_old_files())

        logger.info("✅ All systems healthy - ViralClip Pro v4.0 ready")
        yield

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("🔄 Shutting down ViralClip Pro v4.0")
        await video_service.cleanup()
        await realtime_engine.cleanup()
        logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Netflix-level AI-powered viral video clip generator",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None
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
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/public", StaticFiles(directory="public"), name="public")


@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Netflix-level performance monitoring"""
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000000)}"

    try:
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.2f}ms"

        # Collect metrics
        await metrics_collector.record_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration
        )

        return response

    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"Request {request_id} failed after {duration:.2f}ms: {e}")
        raise


# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main application page"""
    try:
        html_path = Path("index.html")
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="Main page not found")
        return FileResponse(html_path)
    except Exception as e:
        logger.error(f"Error serving main page: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v4/health", response_model=HealthResponse)
async def health_check():
    """Netflix-level health check endpoint"""
    try:
        health_data = await health_checker.comprehensive_check()
        return JSONResponse(content=health_data, status_code=200 if health_data["status"] == "healthy" else 503)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503
        )


@app.post("/api/v4/upload-video", response_model=UploadResponse)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    upload_id: str = Form(...)
):
    """Netflix-level video upload with real-time progress"""
    try:
        # Security validation
        await security_manager.validate_upload(file, request)

        # Process upload
        result = await video_service.process_upload(file, upload_id, upload_sessions)

        # Broadcast progress
        await realtime_engine.broadcast_upload_progress(upload_id, {
            "type": "upload_complete",
            "session_id": result["session_id"],
            "status": "complete"
        })

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload processing failed")


@app.post("/api/v4/generate-preview", response_model=PreviewResponse)
async def generate_live_preview(request: PreviewRequest):
    """Generate live preview for timeline segment"""
    try:
        result = await video_service.generate_preview(
            session_id=request.session_id,
            start_time=request.start_time,
            end_time=request.end_time,
            quality=request.quality,
            platform_optimizations=request.platform_optimizations
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Preview generation failed")


@app.get("/api/v4/thumbnail/{session_id}")
async def get_thumbnail(session_id: str):
    """Get video thumbnail"""
    try:
        thumbnail_path = await video_service.get_thumbnail(session_id)
        if not thumbnail_path or not thumbnail_path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")

        return FileResponse(thumbnail_path, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Thumbnail error: {e}")
        raise HTTPException(status_code=500, detail="Thumbnail generation failed")


@app.websocket("/api/v4/ws/upload/{upload_id}")
async def websocket_upload(websocket: WebSocket, upload_id: str):
    """Upload progress WebSocket with Netflix-level real-time updates"""
    await websocket.accept()
    connection_id = f"upload_{upload_id}"
    websocket_connections[connection_id] = websocket

    try:
        logger.info(f"Upload WebSocket connected: {connection_id}")

        await websocket.send_json({
            "type": "connection_established",
            "upload_id": upload_id,
            "timestamp": time.time(),
            "server_version": settings.app_version
        })

        await realtime_engine.handle_upload_websocket(websocket, upload_id)

    except WebSocketDisconnect:
        logger.info(f"Upload WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Upload WebSocket error: {e}")
    finally:
        websocket_connections.pop(connection_id, None)


@app.websocket("/api/v4/ws/app")
async def websocket_main(websocket: WebSocket):
    """Main application WebSocket connection"""
    await websocket.accept()
    connection_id = f"main_{int(time.time() * 1000000)}"
    websocket_connections[connection_id] = websocket

    try:
        logger.info(f"WebSocket connected: {connection_id}")

        await websocket.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": time.time(),
            "server_version": settings.app_version
        })

        await realtime_engine.handle_main_websocket(websocket, connection_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        websocket_connections.pop(connection_id, None)


@app.get("/api/v4/download/{clip_id}")
async def download_clip(clip_id: str):
    """Download processed clip"""
    try:
        file_path = await video_service.get_clip(clip_id)
        if not file_path or not file_path.exists():
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


# Background tasks
async def process_upload_queue():
    """Background task to process uploaded videos"""
    while True:
        try:
            item = await processing_queue.get()
            await video_service.process_video_item(item, realtime_engine)
        except Exception as e:
            logger.error(f"Processing queue error: {e}")
            await asyncio.sleep(1)


async def cleanup_old_files():
    """Cleanup old files periodically"""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour
            await video_service.cleanup_old_files()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Netflix-level HTTP exception handling"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url.path}")

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            path=request.url.path,
            timestamp=int(time.time())
        ).dict()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Netflix-level global exception handling"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            status_code=500,
            path=request.url.path,
            timestamp=int(time.time())
        ).dict()
    )


# Development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )