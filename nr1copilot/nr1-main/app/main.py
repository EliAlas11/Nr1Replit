"""
ViralClip Pro v6.0 - Netflix-Level Architecture
Enterprise-grade video processing platform with advanced patterns
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

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
    Response,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
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
from fastapi import Body

# Initialize logging first
logger = setup_logging()

# Lazy imports for better startup performance
from app.services.dependency_container import DependencyContainer
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.middleware.performance import PerformanceMiddleware
from app.middleware.security import SecurityMiddleware


class ApplicationState:
    """Application state management with circuit breaker pattern"""

    def __init__(self):
        self.is_ready = False
        self.is_healthy = True
        self.startup_time = None
        self.last_health_check = None
        self.error_count = 0
        self.max_errors = 10

    def mark_ready(self):
        self.is_ready = True
        self.startup_time = datetime.utcnow()

    def mark_unhealthy(self):
        self.is_healthy = False
        self.error_count += 1

    def mark_healthy(self):
        self.is_healthy = True
        self.error_count = 0
        self.last_health_check = datetime.utcnow()

    @property
    def should_circuit_break(self) -> bool:
        return self.error_count >= self.max_errors


# Global application state
app_state = ApplicationState()

# Dependency container - Netflix-level dependency injection
container = DependencyContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Netflix-level application lifecycle with graceful startup/shutdown"""
    startup_start = time.time()

    try:
        logger.info("🎬 Starting ViralClip Pro v6.0 - Netflix Architecture")

        # Pre-startup validations
        await validate_environment()

        # Setup directories with proper permissions
        await setup_directories()

        # Initialize dependency container
        await container.initialize()

        # Warm up critical services
        await warm_up_services()

        # Mark application as ready
        app_state.mark_ready()

        startup_time = time.time() - startup_start
        logger.info(f"✅ Application ready in {startup_time:.2f}s")

        yield

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        app_state.mark_unhealthy()
        raise
    finally:
        logger.info("🔄 Shutting down gracefully...")
        await container.cleanup()
        logger.info("✅ Shutdown complete")


async def validate_environment():
    """Validate environment and dependencies"""
    required_dirs = [
        settings.upload_path.parent,
        settings.output_path.parent,
        settings.temp_path.parent,
        settings.cache_path.parent
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            raise RuntimeError(f"Required directory missing: {dir_path}")


async def setup_directories():
    """Setup directories with optimal permissions"""
    directories = [
        settings.upload_path,
        settings.output_path,
        settings.temp_path,
        settings.cache_path,
        Path("nr1copilot/nr1-main/logs")
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True, mode=0o755)
        logger.debug(f"Directory ready: {directory}")


async def warm_up_services():
    """Warm up critical services for better first-request performance"""
    try:
        # Warm up AI analyzer
        if container.ai_analyzer:
            await container.ai_analyzer.warm_up()

        # Warm up cache
        if container.cache_manager:
            await container.cache_manager.warm_up()

        # Pre-load critical data
        await container.preload_critical_data()

    except Exception as e:
        logger.warning(f"Service warm-up partially failed: {e}")


# Create FastAPI app with optimized configuration
app = FastAPI(
    title="ViralClip Pro v6.0",
    description="Netflix-Level AI Video Processing Platform",
    version="6.0.0",
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
    openapi_url="/api/openapi.json" if settings.debug else None,
    lifespan=lifespan,
    default_response_class=JSONResponse
)

# Security
security = HTTPBearer(auto_error=False)

# Netflix-level middleware stack (order matters!)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)

# Static files with aggressive caching
app.mount(
    "/static", 
    StaticFiles(directory="nr1copilot/nr1-main/static"), 
    name="static"
)
app.mount(
    "/public", 
    StaticFiles(directory="nr1copilot/nr1-main/public"), 
    name="public"
)


# Dependency injection functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Netflix-level authentication with caching"""
    if not credentials and settings.require_auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    if credentials and container.security_manager:
        user = await container.security_manager.validate_token(
            credentials.credentials
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return user

    return {"user_id": "anonymous", "permissions": ["read", "write"]}


async def check_rate_limit(request: Request):
    """Rate limiting with adaptive throttling"""
    if not container.rate_limiter:
        return

    client_ip = request.client.host
    if not await container.rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )


async def check_health():
    """Health check dependency"""
    if app_state.should_circuit_break:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )


# Root endpoint with caching
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main application with aggressive caching"""
    try:
        # Check cache first
        if container.cache_manager:
            cached_content = await container.cache_manager.get("root_html")
            if cached_content:
                return HTMLResponse(content=cached_content)

        # Read and cache
        async with aiofiles.open("nr1copilot/nr1-main/index.html", mode="r") as f:
            content = await f.read()

        if container.cache_manager:
            await container.cache_manager.set("root_html", content, ttl=3600)

        return HTMLResponse(content=content)

    except Exception as e:
        logger.error(f"Failed to serve root: {e}")
        return HTMLResponse(
            "<h1>ViralClip Pro v6.0</h1><p>Service temporarily unavailable</p>", 
            status_code=503
        )


# Health and monitoring endpoints
@app.get("/api/v6/health", response_model=SystemHealth)
async def health_check():
    """Comprehensive health monitoring with detailed metrics"""
    try:
        health_data = await container.health_checker.get_comprehensive_health()
        app_state.mark_healthy()
        return SystemHealth(**health_data)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        app_state.mark_unhealthy()
        return SystemHealth(
            status="unhealthy",
            services={"core": "error"},
            metrics={"error": str(e)},
            timestamp=datetime.utcnow().isoformat()
        )


@app.get("/api/v6/metrics")
async def get_metrics(user=Depends(get_current_user)):
    """Advanced metrics with business intelligence"""
    try:
        metrics = await container.metrics_collector.get_comprehensive_metrics()
        return {"success": True, "metrics": metrics}

    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Chunked upload endpoint for Netflix-level uploads
@app.post("/api/v6/upload/init")
async def init_upload(
    request: Request,
    data: dict = Body(...)
):
    """Initialize chunked upload with validation"""
    try:
        filename = data.get("filename")
        file_size = data.get("file_size", 0)
        total_chunks = data.get("total_chunks", 1)
        upload_id = data.get("upload_id")

        if not all([filename, file_size, upload_id]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required parameters"
            )

        # Validate file size
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {settings.max_file_size} bytes"
            )

        # Validate file type
        extension = filename.split('.')[-1].lower()
        if extension not in settings.allowed_video_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {extension}"
            )

        # Create upload directory
        upload_dir = settings.upload_path / upload_id
        upload_dir.mkdir(exist_ok=True)

        # Store upload metadata
        metadata = {
            "filename": filename,
            "file_size": file_size,
            "total_chunks": total_chunks,
            "upload_id": upload_id,
            "created_at": time.time(),
            "status": "initialized"
        }

        metadata_path = upload_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            import json
            json.dump(metadata, f)

        logger.info(f"📋 Upload initialized: {upload_id} - {filename}")

        return {
            "success": True,
            "upload_id": upload_id,
            "status": "initialized",
            "message": "Upload initialized successfully"
        }

    except Exception as e:
        logger.error(f"❌ Upload initialization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload initialization failed"
        )

@app.post("/api/v6/upload/chunk")
async def upload_chunk(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...)
):
    """Handle chunked video upload with Netflix-level reliability"""
    try:
        logger.info(f"📦 Chunk upload: {upload_id} - {chunk_index}/{total_chunks}")

        # Validate chunk
        if chunk_index >= total_chunks or chunk_index < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid chunk index"
            )

        # Verify upload directory exists
        upload_dir = settings.temp_path / upload_id
        if not upload_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload not initialized"
            )

        # Load metadata
        metadata_path = upload_dir / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"filename": file.filename}

        # Validate chunk size
        chunk_data = await file.read()
        max_chunk_size = 10 * 1024 * 1024  # 10MB
        if len(chunk_data) > max_chunk_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Chunk too large"
            )

        # Save chunk with integrity check
        chunk_path = upload_dir / f"chunk_{chunk_index:06d}"
        with open(chunk_path, "wb") as chunk_file:
            chunk_file.write(chunk_data)

        # Verify chunk was written correctly
        if chunk_path.stat().st_size != len(chunk_data):
            chunk_path.unlink()  # Remove corrupted chunk
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Chunk write verification failed"
            )

        # Check if all chunks received
        existing_chunks = sorted(upload_dir.glob("chunk_*"))
        chunks_received = len(existing_chunks)

        # Broadcast progress via WebSocket
        if hasattr(container, 'realtime_engine'):
            progress_data = {
                "type": "chunk_progress",
                "upload_id": upload_id,
                "chunks_received": chunks_received,
                "total_chunks": total_chunks,
                "progress": (chunks_received / total_chunks) * 100,
                "chunk_index": chunk_index
            }
            await container.realtime_engine.broadcast_upload_progress(upload_id, progress_data)

        if chunks_received == total_chunks:
            # Verify all chunks exist
            missing_chunks = []
            for i in range(total_chunks):
                chunk_path = upload_dir / f"chunk_{i:06d}"
                if not chunk_path.exists():
                    missing_chunks.append(i)

            if missing_chunks:
                return {
                    "success": False,
                    "upload_id": upload_id,
                    "status": "missing_chunks",
                    "missing_chunks": missing_chunks,
                    "chunks_received": chunks_received,
                    "total_chunks": total_chunks
                }

            # Reassemble file
            original_filename = metadata.get("filename", file.filename)
            final_path = settings.upload_path / f"{upload_id}_{original_filename}"

            try:
                with open(final_path, "wb") as final_file:
                    for i in range(total_chunks):
                        chunk_path = upload_dir / f"chunk_{i:06d}"
                        with open(chunk_path, "rb") as chunk_file:
                            chunk_data = chunk_file.read()
                            final_file.write(chunk_data)

                # Verify final file size
                expected_size = metadata.get("file_size", 0)
                actual_size = final_path.stat().st_size

                if expected_size > 0 and abs(actual_size - expected_size) > 1024:  # Allow 1KB tolerance
                    final_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"File size mismatch. Expected: {expected_size}, Got: {actual_size}"
                    )

                # Cleanup chunks
                for chunk_file in existing_chunks:
                    chunk_file.unlink()

                # Remove metadata and directory
                if metadata_path.exists():
                    metadata_path.unlink()
                upload_dir.rmdir()

                # Start processing
                background_tasks.add_task(
                    container.realtime_engine.process_upload_pipeline,
                    upload_id,
                    str(final_path),
                    None,  # title
                    None,  # description
                    None   # user
                )

                logger.info(f"✅ Upload completed: {upload_id} - {original_filename}")

                return {
                    "success": True,
                    "upload_id": upload_id,
                    "status": "complete",
                    "final_path": str(final_path),
                    "file_size": actual_size,
                    "message": "Upload complete, processing started"
                }

            except Exception as e:
                # Cleanup on error
                if final_path.exists():
                    final_path.unlink()
                raise e

        else:
            return {
                "success": True,
                "upload_id": upload_id,
                "status": "chunk_received",
                "chunks_received": chunks_received,
                "total_chunks": total_chunks,
                "progress": (chunks_received / total_chunks) * 100,
                "message": f"Chunk {chunk_index + 1} of {total_chunks} received"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chunk upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chunk upload failed"
        )

@app.post("/api/v6/upload/finalize")
async def finalize_upload(
    request: Request,
    data: dict = Body(...)
):
    """Finalize upload and verify integrity"""
    try:
        upload_id = data.get("upload_id")
        if not upload_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing upload_id"
            )

        # Check if upload exists
        upload_pattern = f"{upload_id}_*"
        upload_files = list(settings.upload_path.glob(upload_pattern))

        if not upload_files:
            # Check if still in progress
            upload_dir = settings.temp_path / upload_id
            if upload_dir.exists():
                return {
                    "success": False,
                    "status": "in_progress",
                    "message": "Upload still in progress"
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Upload not found"
                )

        upload_file = upload_files[0]

        # Perform final verification
        try:
            # Basic file verification
            if upload_file.stat().st_size == 0:
                raise ValueError("File is empty")

            # Quick format verification for video files
            if upload_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.webm']:
                # Basic header check
                with open(upload_file, 'rb') as f:
                    header = f.read(12)
                    if not header:
                        raise ValueError("Cannot read file header")

            return {
                "success": True,
                "upload_id": upload_id,
                "status": "verified",
                "file_path": str(upload_file),
                "file_size": upload_file.stat().st_size,
                "message": "Upload verified successfully"
            }

        except Exception as e:
            logger.error(f"Upload verification failed: {e}")
            return {
                "success": False,
                "status": "verification_failed",
                "error": str(e),
                "message": "Upload verification failed"
            }

    except Exception as e:
        logger.error(f"❌ Upload finalization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload finalization failed"
        )


# Video upload endpoint with advanced features
@app.post("/api/v6/upload-video", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
    user=Depends(get_current_user),
    _=Depends(check_rate_limit),
    _health=Depends(check_health)
):
    """Netflix-level video upload with intelligent processing"""
    start_time = time.time()
    session_id = upload_id or f"upload_{uuid.uuid4().hex[:16]}"

    try:
        # Advanced file validation
        validation_result = await container.video_service.validate_upload(
            file, user.get("permissions", [])
        )

        if not validation_result.valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation_result.error
            )

        # Secure file handling with atomic operations
        file_path = await container.video_service.save_upload_securely(
            file, session_id
        )

        # Start intelligent background processing
        background_tasks.add_task(
            container.realtime_engine.process_upload_pipeline,
            session_id,
            str(file_path),
            title,
            description,
            user
        )

        # Track comprehensive metrics
        await container.metrics_collector.track_upload_event(
            file_size=validation_result.file_size,
            processing_time=time.time() - start_time,
            user_id=user.get("user_id"),
            session_id=session_id
        )

        return VideoUploadResponse(
            success=True,
            session_id=session_id,
            file_path=str(file_path),
            file_size=validation_result.file_size,
            estimated_processing_time=validation_result.estimated_time,
            processing_time=time.time() - start_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed for session {session_id}: {e}", exc_info=True)
        app_state.mark_unhealthy()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload processing failed"
        )


# AI Analysis endpoint with caching
@app.post("/api/v6/analyze", response_model=AnalysisResponse)
async def analyze_video(
    request: AnalysisRequest,
    user=Depends(get_current_user),
    _=Depends(check_rate_limit),
    _health=Depends(check_health)
):
    """Advanced AI analysis with intelligent caching"""
    try:
        # Check cache first
        cache_key = f"analysis_{request.session_id}_{hash(str(request.dict()))}"
        if container.cache_manager:
            cached_result = await container.cache_manager.get(cache_key)
            if cached_result:
                return AnalysisResponse(
                    success=True,
                    session_id=request.session_id,
                    analysis=cached_result,
                    cached=True
                )

        # Validate request
        if not Path(request.file_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found"
            )

        # Perform advanced AI analysis
        analysis_result = await container.ai_analyzer.analyze_comprehensive(
            file_path=request.file_path,
            title=request.title,
            description=request.description,
            target_platforms=request.target_platforms,
            custom_prompts=request.custom_prompts
        )

        # Cache result with intelligent TTL
        if container.cache_manager:
            ttl = 3600 if analysis_result.get("confidence", 0) > 0.8 else 1800
            await container.cache_manager.set(cache_key, analysis_result, ttl=ttl)

        return AnalysisResponse(
            success=True,
            session_id=request.session_id,
            analysis=analysis_result,
            cached=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis processing failed"
        )


# Preview generation with adaptive quality
@app.post("/api/v6/generate-preview", response_model=PreviewResponse)
async def generate_preview(
    request: PreviewRequest,
    user=Depends(get_current_user),
    _=Depends(check_rate_limit),
    _health=Depends(check_health)
):
    """Intelligent preview generation with adaptive streaming"""
    try:
        preview_data = await container.realtime_engine.generate_adaptive_preview(
            session_id=request.session_id,
            start_time=request.start_time,
            end_time=request.end_time,
            quality=request.quality,
            user_preferences=user.get("preferences", {})
        )

        return PreviewResponse(
            success=True,
            session_id=request.session_id,
            preview_url=preview_data["preview_url"],
            viral_analysis=preview_data.get("viral_analysis", {}),
            suggestions=preview_data.get("suggestions", []),
            processing_time=preview_data.get("processing_time", 0)
        )

    except Exception as e:
        logger.error(f"❌ Preview generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Preview generation failed"
        )


# Advanced timeline endpoint
@app.get("/api/v6/timeline/{session_id}")
async def get_timeline_data(
    session_id: str,
    user=Depends(get_current_user),
    _=Depends(check_rate_limit)
):
    """Advanced interactive timeline with ML insights"""
    try:
        timeline_data = await container.realtime_engine.get_intelligent_timeline(
            session_id,
            user_preferences=user.get("preferences", {})
        )
        return {"success": True, "timeline": timeline_data}

    except Exception as e:
        logger.error(f"❌ Timeline data failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeline data failed"
        )


# Processing status with intelligent updates
@app.get("/api/v6/status/{session_id}", response_model=ProcessingStatus)
async def get_processing_status(
    session_id: str,
    user=Depends(get_current_user)
):
    """Intelligent processing status with predictive completion"""
    try:
        status_data = await container.realtime_engine.get_intelligent_status(
            session_id
        )

        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        return ProcessingStatus(**status_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Status check failed"
        )


# WebSocket endpoints with connection pooling
@app.websocket("/api/v6/ws/realtime/{session_id}")
async def websocket_realtime(websocket: WebSocket, session_id: str):
    """High-performance WebSocket with connection pooling"""
    connection_id = f"realtime_{session_id}_{uuid.uuid4().hex[:8]}"

    try:
        await websocket.accept()
        await container.realtime_engine.handle_realtime_connection(
            websocket, session_id, connection_id
        )

    except WebSocketDisconnect:
        logger.info(f"WebSocket {connection_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error {connection_id}: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason="Internal error")
        except:
            pass
    finally:
        await container.realtime_engine.cleanup_connection(connection_id)


# File serving with intelligent caching
@app.get("/api/v6/preview/{session_id}/{timestamp}")
async def serve_preview(
    session_id: str, 
    timestamp: str,
    user=Depends(get_current_user)
):
    """Serve preview files with adaptive streaming"""
    try:
        file_path = await container.video_service.get_preview_path(
            session_id, timestamp
        )

        if not file_path or not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preview not found"
            )

        return FileResponse(
            file_path,
            media_type="video/mp4",
            headers={
                "Cache-Control": "public, max-age=3600",
                "X-Content-Type-Options": "nosniff"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Preview serving failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Preview serving failed"
        )


# Advanced error handlers
@app.exception_handler(HTTPException)
async def enhanced_http_exception_handler(request: Request, exc: HTTPException):
    """Netflix-level error handling with context"""
    error_id = str(uuid.uuid4())

    logger.error(
        f"HTTP Error {error_id}: {exc.status_code} - {exc.detail}",
        extra={
            "error_id": error_id,
            "path": str(request.url),
            "method": request.method,
            "client_ip": request.client.host
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_id": error_id,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Enhanced validation error handling"""
    error_id = str(uuid.uuid4())

    logger.warning(
        f"Validation Error {error_id}: {exc.errors()}",
        extra={"error_id": error_id, "path": str(request.url)}
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "error_id": error_id,
            "message": "Validation failed",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Comprehensive error handling with monitoring"""
    error_id = str(uuid.uuid4())

    logger.error(
        f"Unexpected Error {error_id}: {str(exc)}",
        extra={
            "error_id": error_id,
            "path": str(request.url),
            "traceback": traceback.format_exc()
        },
        exc_info=True
    )

    app_state.mark_unhealthy()

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "error_id": error_id,
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Application startup
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=True,
        workers=1,
        loop="uvloop" if not settings.debug else "asyncio"
    )