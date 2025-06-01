` tags. I will pay close attention to indentation, structure, and completeness, and avoid any forbidden words or placeholders.

Here's the combined code:

```
<replit_final_file>
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

from config import settings, get_settings
from logging_config import setup_logging
from routes.video import router as video_router

# Initialize logging
logger = setup_logging()

# Active connections for WebSocket
active_connections: Dict[str, WebSocket] = {}
upload_connections: Dict[str, WebSocket] = {}
processing_sessions: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Host: {settings.HOST}:{settings.PORT}")

    # Create necessary directories
    directories = [
        settings.UPLOAD_PATH,
        settings.OUTPUT_PATH,
        settings.TEMP_PATH,
        settings.LOG_PATH,
        settings.VIDEO_STORAGE_PATH,
        "static",
        "public"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")

    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_old_files())

    logger.info("✅ ViralClip Pro startup complete")

    yield

    # Shutdown
    logger.info("🛑 Shutting down ViralClip Pro...")

    # Cancel background tasks
    cleanup_task.cancel()

    # Close all WebSocket connections
    for connection in active_connections.values():
        try:
            await connection.close()
        except:
            pass

    for connection in upload_connections.values():
        try:
            await connection.close()
        except:
            pass

    logger.info("✅ ViralClip Pro shutdown complete")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered viral video analysis and clipping platform with Netflix-level features",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
    openapi_url="/openapi.json" if settings.DEBUG else None
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)

    # Security headers
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    }

    if not settings.DEBUG:
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    for header, value in headers.items():
        response.headers[header] = value

    return response

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring"""
    start_time = datetime.now()

    response = await call_next(request)

    process_time = (datetime.now() - start_time).total_seconds()

    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s - "
        f"IP: {request.client.host}"
    )

    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware for production
if settings.ALLOWED_HOSTS != ["*"]:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# Include API routes
app.include_router(video_router, prefix="/api/v1", tags=["Video Processing"])

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

@app.get("/api/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information"""
    import psutil

    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Check directory existence
        directories_status = {}
        for directory in [settings.UPLOAD_PATH, settings.OUTPUT_PATH, settings.TEMP_PATH]:
            directories_status[directory] = {
                "exists": Path(directory).exists(),
                "writable": os.access(directory, os.W_OK) if Path(directory).exists() else False
            }

        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_free": disk.free
            },
            "directories": directories_status,
            "active_connections": {
                "websocket": len(active_connections),
                "upload": len(upload_connections)
            },
            "processing_sessions": len(processing_sessions)
        }

        # Determine overall health
        if cpu_percent > 90 or memory.percent > 90:
            health_data["status"] = "degraded"

        status_code = 200 if health_data["status"] == "healthy" else 503
        return JSONResponse(content=health_data, status_code=status_code)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            },
            status_code=503
        )

# Enhanced file upload endpoint
@app.post("/api/v1/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    upload_id: str = Form(None)
):
    """Enhanced file upload with validation and progress tracking"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail="Only video files are allowed"
            )

        # Read and validate file size
        content = await file.read()
        file_size = len(content)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )

        # Generate unique identifiers
        file_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        # Create safe filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_', '-')).strip()
        if not safe_filename:
            safe_filename = f"video_{file_id}"

        filename = f"upload_{file_id}_{safe_filename}"
        upload_path = Path(settings.UPLOAD_PATH) / filename

        # Save file
        async with aiofiles.open(upload_path, 'wb') as f:
            await f.write(content)

        # Create processing session
        processing_sessions[session_id] = {
            "status": "uploaded",
            "file_path": str(upload_path),
            "filename": safe_filename,
            "file_size": file_size,
            "upload_id": upload_id,
            "created_at": datetime.now(),
            "file_id": file_id
        }

        # Notify upload WebSocket if connected
        if upload_id and upload_id in upload_connections:
            await upload_connections[upload_id].send_json({
                "type": "upload_complete",
                "session_id": session_id,
                "file_info": {
                    "filename": safe_filename,
                    "size": file_size
                }
            })

        logger.info(f"File uploaded successfully: {filename} ({file_size} bytes)")

        return {
            "success": True,
            "session_id": session_id,
            "filename": safe_filename,
            "file_size": file_size,
            "message": "File uploaded successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication"""
    try:
        await websocket.accept()
        active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "message": "WebSocket connected successfully"
        })

        # Keep connection alive
        while True:
            try:
                # Wait for messages with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                try:
                    message = json.loads(data)
                    await handle_websocket_message(websocket, client_id, message)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format"
                    })

            except asyncio.TimeoutError:
                # Send keep-alive ping
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                })
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        # Clean up connection
        if client_id in active_connections:
            del active_connections[client_id]
        logger.info(f"WebSocket disconnected: {client_id}")

async def handle_websocket_message(websocket: WebSocket, client_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")

    if message_type == "ping":
        await websocket.send_json({
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    elif message_type == "subscribe":
        # Handle subscription to specific events
        await websocket.send_json({
            "type": "subscribed",
            "channel": message.get("channel"),
            "timestamp": datetime.now().isoformat()
        })
    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })

# Serve static files
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

if Path("public").exists():
    app.mount("/public", StaticFiles(directory="public"), name="static")

# Serve main application
@app.get("/")
async def read_root():
    """Serve the main application page"""
    index_path = Path("index.html")
    if index_path.exists():
        return FileResponse("index.html")
    else:
        return JSONResponse(
            content={
                "message": f"Welcome to {settings.APP_NAME} v{settings.APP_VERSION}",
                "docs": "/docs" if settings.DEBUG else "Documentation not available in production",
                "health": "/health"
            }
        )

# Background cleanup task
async def cleanup_old_files():
    """Periodic cleanup of old files and sessions"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour

            now = datetime.now()

            # Clean up old uploads (older than 24 hours)
            upload_path = Path(settings.UPLOAD_PATH)
            if upload_path.exists():
                for file_path in upload_path.iterdir():
                    if file_path.is_file():
                        file_age = now.timestamp() - file_path.stat().st_mtime
                        if file_age > 86400:  # 24 hours
                            try:
                                file_path.unlink()
                                logger.info(f"Cleaned up old file: {file_path}")
                            except Exception as e:
                                logger.error(f"Failed to clean up {file_path}: {e}")

            # Clean up old processing sessions
            expired_sessions = []
            for session_id, session in processing_sessions.items():
                session_age = (now - session.get("created_at", now)).total_seconds()
                if session_age > 86400:  # 24 hours
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                del processing_sessions[session_id]

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        except Exception as e:
            logger.error(f"Cleanup task error: {e}")

# Enhanced error handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Custom HTTP exception handler with detailed logging"""
    logger.warning(
        f"HTTP {exc.status_code} error for {request.method} {request.url.path}: {exc.detail}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "path": str(request.url.path),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    error_id = str(uuid.uuid4())

    logger.error(
        f"Unhandled exception [{error_id}] for {request.method} {request.url.path}: {exc}"
    )
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "message": "Internal server error",
            "error_id": error_id,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn

    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKER_PROCESSES
    )