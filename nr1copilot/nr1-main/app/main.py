"""
ViralClip Pro v6.0 - Netflix-Level Main Application
Enterprise-grade FastAPI application with comprehensive middleware and monitoring
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import psutil
import uvicorn
import hashlib
import shutil
import aiofiles
from datetime import datetime
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import sys
from pathlib import Path
from fastapi.middleware import TrustedHostMiddleware


# Import our modules
from app.services.video_service import NetflixLevelVideoService
from app.services.realtime_engine import EnterpriseRealtimeEngine
from app.middleware.performance import PerformanceMiddleware
from app.middleware.security import SecurityMiddleware
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.utils.metrics import MetricsCollector
from app.utils.health import HealthChecker
from app.logging_config import setup_logging
from app.config import settings
from app.schemas import *
from app.services.dependency_container import DependencyContainer
# Add these imports at the top
from app.services.captions_service import NetflixLevelCaptionService, JobType as CaptionJobType
from app.services.template_service import NetflixLevelTemplateService, TemplateCategory, PlatformType
from app.services.batch_processor import NetflixLevelBatchProcessor, JobType, JobPriority
from app.services.social_publisher import NetflixLevelSocialPublisher, SocialPlatform, OptimizationLevel
import aiofiles
import uuid
import traceback
from datetime import datetime
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import sys
from pathlib import Path


# ================================
# Pydantic Models
# ================================

class UploadInitRequest(BaseModel):
    """Request model for upload initialization"""
    filename: str = Field(..., min_length=1, max_length=255)
    file_size: int = Field(..., gt=0, le=500*1024*1024)  # Max 500MB
    total_chunks: int = Field(..., gt=0, le=10000)
    upload_id: str = Field(..., min_length=1, max_length=100)
    user_tier: str = Field(default="free", regex="^(free|standard|premium)$")


class UploadChunkRequest(BaseModel):
    """Request model for chunk upload"""
    upload_id: str = Field(..., min_length=1, max_length=100)
    chunk_index: int = Field(..., ge=0)
    total_chunks: int = Field(..., gt=0)
    chunk_hash: Optional[str] = Field(None, min_length=32, max_length=32)


class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    timestamp: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    checks: Dict[str, Any]


# ================================
# Application Factory
# ================================

class NetflixLevelApplication:
    """Netflix-level application factory with enterprise features"""

    def __init__(self):
        self.video_service: Optional[NetflixLevelVideoService] = None
        self.realtime_engine: Optional[EnterpriseRealtimeEngine] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.health_checker: Optional[HealthChecker] = None
        self.startup_time = time.time()
        self.logger = logging.getLogger(__name__)

    async def create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""

        # Setup logging first
        setup_logging()

        # Create app with lifespan management
        app = FastAPI(
            title="ViralClip Pro - Netflix-Level Video Service",
            description="Enterprise-grade AI video processing platform",
            version="6.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc",
            openapi_url="/api/openapi.json",
            lifespan=self.lifespan,
            default_response_class=JSONResponse,
            swagger_ui_parameters={
                "defaultModelsExpandDepth": -1,
                "displayRequestDuration": True,
                "persistAuthorization": True
            }
        )

        # Configure middleware
        await self._configure_middleware(app)

        # Configure routes
        await self._configure_routes(app)

        # Configure exception handlers
        await self._configure_exception_handlers(app)

        # Configure static files
        await self._configure_static_files(app)

        self.logger.info("🚀 Netflix-level application created successfully")
        return app

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Application lifespan management"""
        # Startup
        await self._startup()
        try:
            yield
        finally:
            # Shutdown
            await self._shutdown()

    async def _startup(self):
        """Application startup sequence"""
        try:
            self.logger.info("🔄 Starting Netflix-level video service...")

            # Initialize core services
            self.video_service = NetflixLevelVideoService()
            await self.video_service.startup()

            self.realtime_engine = EnterpriseRealtimeEngine()
            await self.realtime_engine.enterprise_warm_up()

            # Initialize monitoring
            self.metrics_collector = MetricsCollector()
            await self.metrics_collector.start()

            self.health_checker = HealthChecker([
                self.video_service,
                self.realtime_engine,
                self.metrics_collector
            ])

            self.logger.info("✅ All services started successfully")

        except Exception as e:
            self.logger.error(f"Startup failed: {e}")
            raise

    async def _shutdown(self):
        """Application shutdown sequence"""
        try:
            self.logger.info("🔄 Shutting down services...")

            # Shutdown in reverse order
            if self.health_checker:
                await self.health_checker.stop()

            if self.metrics_collector:
                await self.metrics_collector.stop()

            if self.realtime_engine:
                await self.realtime_engine.graceful_shutdown()

            if self.video_service:
                await self.video_service.shutdown()

            self.logger.info("✅ All services shut down successfully")

        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

    async def _configure_middleware(self, app: FastAPI):
        """Configure application middleware"""

        # Security middleware (first)
        app.add_middleware(SecurityMiddleware)

        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            max_age=86400,  # 24 hours
        )

        # Compression middleware
        app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)

        # Performance monitoring middleware
        app.add_middleware(PerformanceMiddleware)

        # Error handling middleware (last)
        app.add_middleware(ErrorHandlerMiddleware)
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

    async def _configure_routes(self, app: FastAPI):
        """Configure application routes"""

        # Enterprise dependency injection
        async def get_authenticated_user(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ) -> Dict[str, Any]:
            """Netflix-level authentication with enterprise caching and validation"""
            if not credentials and settings.require_auth:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"}
                )

            if credentials and container.security_manager:
                try:
                    user = await container.security_manager.validate_token_enterprise(
                        credentials.credentials
                    )
                    if not user:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"}
                        )
                    return user
                except Exception as e:
                    self.logger.warning(f"Authentication failed: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication failed"
                    )

            # Default user for development
            return {
                "user_id": "anonymous",
                "permissions": ["read", "write"],
                "tier": "standard"
            }

        security = HTTPBearer(auto_error=False)
        # Health check endpoints
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Comprehensive health check endpoint"""
            health_data = await self.health_checker.check_health()

            return HealthResponse(
                status="healthy" if health_data["healthy"] else "unhealthy",
                timestamp=health_data["timestamp"],
                version="6.0.0",
                uptime_seconds=time.time() - self.startup_time,
                checks=health_data["checks"]
            )

        @app.get("/health/readiness")
        async def readiness_check():
            """Kubernetes readiness probe"""
            if self.video_service and self.realtime_engine:
                return {"status": "ready"}
            raise HTTPException(status_code=503, detail="Service not ready")

        @app.get("/health/liveness")
        async def liveness_check():
            """Kubernetes liveness probe"""
            return {"status": "alive", "timestamp": time.time()}

        # Metrics endpoints
        @app.get("/metrics")
        async def get_metrics():
            """Prometheus-compatible metrics endpoint"""
            if self.metrics_collector:
                return await self.metrics_collector.export_prometheus_metrics()
            return {"error": "Metrics collector not available"}

        @app.get("/api/v6/metrics/detailed")
        async def get_detailed_metrics():
            """Detailed metrics for monitoring dashboard"""
            metrics = {}

            if self.video_service:
                metrics["video_service"] = await self.video_service.get_service_metrics()

            if self.realtime_engine:
                metrics["realtime_engine"] = await self.realtime_engine.get_realtime_stats()

            if self.metrics_collector:
                metrics["system"] = await self.metrics_collector.get_system_metrics()

            return APIResponse(
                success=True,
                message="Metrics retrieved successfully",
                data=metrics,
                timestamp=time.time()
            )

        # Upload API endpoints
        @app.post("/api/v6/upload/init")
        async def initialize_upload(
            request: UploadInitRequest,
            background_tasks: BackgroundTasks,
            req: Request,
            user=Depends(get_authenticated_user),
            _=Depends(check_enterprise_rate_limit),
            _health=Depends(check_enterprise_health)
        ):
            """Initialize upload session with enhanced validation"""

            # Extract user info (in production, this would come from authentication)
            user_info = {
                "user_id": req.headers.get("X-User-ID", "anonymous"),
                "tier": request.user_tier,
                "ip_address": req.client.host
            }

            client_info = {
                "user_agent": req.headers.get("User-Agent", ""),
                "accept_language": req.headers.get("Accept-Language", ""),
                "connection_type": req.headers.get("Connection", "")
            }

            try:
                result = await self.video_service.create_upload_session(
                    upload_id=request.upload_id,
                    filename=request.filename,
                    file_size=request.file_size,
                    total_chunks=request.total_chunks,
                    user_info=user_info,
                    client_info=client_info
                )

                # Schedule background analytics
                background_tasks.add_task(
                    self._track_upload_init,
                    request.upload_id,
                    user_info,
                    request.file_size
                )

                return APIResponse(
                    success=True,
                    message="Upload session initialized successfully",
                    data=result,
                    timestamp=time.time(),
                    request_id=req.headers.get("X-Request-ID")
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Upload initialization failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to initialize upload session"
                )

        @app.post("/api/v6/upload/chunk")
        async def upload_chunk(
            file: UploadFile = File(...),
            upload_id: str = Form(...),
            chunk_index: int = Form(...),
            total_chunks: int = Form(...),
            chunk_hash: Optional[str] = Form(None),
            req: Request = None,
            user=Depends(get_authenticated_user),
            _=Depends(check_enterprise_rate_limit)
        ):
            """Upload file chunk with enhanced error handling"""

            try:
                result = await self.video_service.process_chunk(
                    file=file,
                    upload_id=upload_id,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    chunk_hash=chunk_hash
                )

                # Broadcast progress via WebSocket
                if self.realtime_engine:
                    await self.realtime_engine.broadcast_enterprise_progress(
                        upload_id=upload_id,
                        progress_data=result,
                        user={"user_id": req.headers.get("X-User-ID", "anonymous")}
                    )

                return APIResponse(
                    success=True,
                    message="Chunk processed successfully",
                    data=result,
                    timestamp=time.time()
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Chunk upload failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "Chunk upload failed",
                        "upload_id": upload_id,
                        "chunk_index": chunk_index
                    }
                )

        @app.get("/api/v6/upload/status/{upload_id}")
        async def get_upload_status(upload_id: str):
            """Get upload status with comprehensive information"""

            try:
                status = await self.video_service.get_upload_status(upload_id)

                return APIResponse(
                    success=True,
                    message="Status retrieved successfully",
                    data=status,
                    timestamp=time.time()
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Status retrieval failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to retrieve upload status"
                )

        @app.delete("/api/v6/upload/cancel/{upload_id}")
        async def cancel_upload(upload_id: str):
            """Cancel upload and cleanup resources"""

            try:
                result = await self.video_service.cancel_upload(upload_id)

                return APIResponse(
                    success=True,
                    message="Upload cancelled successfully",
                    data=result,
                    timestamp=time.time()
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Upload cancellation failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to cancel upload"
                )

        # WebSocket endpoints
        @app.websocket("/api/v6/ws/enterprise/upload_manager")
        async def upload_manager_websocket(websocket: WebSocket):
            """Enterprise upload manager WebSocket"""

            await websocket.accept()
            connection_id = f"conn_{time.time()}_{id(websocket)}"



        # ================================
        # Smart Captions API Endpoints
        # ================================

        @app.post("/api/v6/captions/generate")
        async def generate_captions(
            request: Request,
            session_id: str = Form(...),
            audio_file: UploadFile = File(...),
            language: str = Form("en"),
            speaker_personalization: bool = Form(True),
            viral_optimization: str = Form("high"),
            user=Depends(get_authenticated_user)
        ):
            """Generate AI captions with viral optimization"""
            
            try:
                # Save uploaded audio file
                audio_path = f"temp/audio_{session_id}_{audio_file.filename}"
                async with aiofiles.open(audio_path, "wb") as f:
                    content = await audio_file.read()
                    await f.write(content)
                
                # Initialize caption service
                caption_service = NetflixLevelCaptionService()
                
                # Generate captions with personalization
                result = await caption_service.generate_captions(
                    session_id=session_id,
                    audio_file_path=audio_path,
                    language=language,
                    speaker_personalization=speaker_personalization,
                    viral_optimization_level=viral_optimization
                )
                
                return APIResponse(
                    success=True,
                    message="Captions generated successfully",
                    data={
                        "session_id": result.session_id,
                        "viral_score": result.overall_viral_score,
                        "processing_time": result.processing_time,
                        "segments_count": len(result.segments),
                        "speaker_count": result.speaker_count,
                        "viral_keywords": result.viral_keywords,
                        "optimization_suggestions": result.optimization_suggestions,
                        "segments": [
                            {
                                "start_time": seg.start_time,
                                "end_time": seg.end_time,
                                "text": seg.text,
                                "confidence": seg.confidence,
                                "viral_score": seg.viral_score,
                                "emotion": seg.emotion,
                                "engagement_potential": seg.engagement_potential
                            }
                            for seg in result.segments[:10]  # Limit for response size
                        ]
                    },
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Caption generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/v6/captions/export")
        async def export_captions(
            request: Request,
            session_id: str = Form(...),
            format_type: str = Form("srt"),
            user=Depends(get_authenticated_user)
        ):
            """Export captions in various formats"""
            
            try:
                caption_service = NetflixLevelCaptionService()
                
                # Get caption result (mock for demo)
                result = await caption_service.get_caption_result(session_id)
                
                if not result:
                    raise HTTPException(status_code=404, detail="Caption session not found")
                
                # Export captions
                export_result = await caption_service.export_captions(
                    result, format_type, platform_specific=True
                )
                
                return APIResponse(
                    success=export_result["success"],
                    message="Captions exported successfully",
                    data=export_result,
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Caption export failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ================================
        # Viral Templates API Endpoints
        # ================================

        @app.get("/api/v6/templates/library")
        async def get_template_library(
            category: Optional[str] = None,
            platform: Optional[str] = None,
            viral_score_min: Optional[float] = None,
            user=Depends(get_authenticated_user)
        ):
            """Get viral template library with filtering"""
            
            try:
                template_service = NetflixLevelTemplateService()
                
                # Get templates with filtering
                templates = await template_service.get_templates(
                    category=category,
                    platform=platform,
                    viral_score_min=viral_score_min
                )
                
                return APIResponse(
                    success=True,
                    message="Template library retrieved",
                    data={
                        "templates": [
                            {
                                "template_id": t.template_id,
                                "name": t.name,
                                "category": t.category.value,
                                "description": t.description,
                                "viral_score": t.viral_score,
                                "platform_optimized": [p.value for p in t.platform_optimized],
                                "duration_range": t.duration_range,
                                "engagement_predictors": t.engagement_predictors,
                                "trending_elements": t.trending_elements,
                                "usage_count": t.usage_count,
                                "customizable_elements": t.customizable_elements
                            }
                            for t in templates
                        ],
                        "total_templates": len(templates)
                    },
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Template library retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/v6/templates/brand-kit")
        async def create_brand_kit(
            request: Request,
            brand_config: dict,
            user=Depends(get_authenticated_user)
        ):
            """Create brand kit for template customization"""
            
            try:
                template_service = NetflixLevelTemplateService()
                
                brand_kit = await template_service.create_brand_kit(
                    user_id=user.get("user_id", "anonymous"),
                    name=brand_config.get("name", "New Brand Kit"),
                    brand_config=brand_config
                )
                
                return APIResponse(
                    success=True,
                    message="Brand kit created successfully",
                    data={
                        "brand_kit_id": brand_kit.kit_id,
                        "name": brand_kit.name,
                        "colors": {
                            "primary": brand_kit.primary_color,
                            "secondary": brand_kit.secondary_color,
                            "accent": brand_kit.accent_color,
                            "background": brand_kit.background_color,
                            "text": brand_kit.text_color
                        },
                        "typography": {
                            "primary_font": brand_kit.primary_font,
                            "secondary_font": brand_kit.secondary_font,
                            "heading_font": brand_kit.heading_font
                        }
                    },
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Brand kit creation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/v6/templates/apply")
        async def apply_template(
            request: Request,
            template_id: str = Form(...),
            brand_kit_id: Optional[str] = Form(None),
            customizations: str = Form("{}"),
            session_id: str = Form(...),
            user=Depends(get_authenticated_user)
        ):
            """Apply viral template to video session"""
            
            try:
                template_service = NetflixLevelTemplateService()
                
                # Parse customizations
                custom_data = json.loads(customizations)
                
                # Apply template
                if brand_kit_id:
                    result = await template_service.apply_brand_kit_to_template(
                        template_id, brand_kit_id
                    )
                else:
                    result = await template_service.apply_template(
                        template_id, session_id, custom_data
                    )
                
                return APIResponse(
                    success=True,
                    message="Template applied successfully",
                    data=result,
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Template application failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/v6/templates/analytics")
        async def get_template_analytics(
            template_id: Optional[str] = None,
            user=Depends(get_authenticated_user)
        ):
            """Get template analytics dashboard data"""
            
            try:
                template_service = NetflixLevelTemplateService()
                
                if template_id:
                    analytics = await template_service.get_template_analytics(template_id)
                    return APIResponse(
                        success=True,
                        message="Template analytics retrieved",
                        data=analytics.__dict__,
                        timestamp=datetime.utcnow().isoformat()
                    )
                else:
                    dashboard_data = await template_service.get_analytics_dashboard_data()
                    return APIResponse(
                        success=True,
                        message="Analytics dashboard data retrieved",
                        data=dashboard_data,
                        timestamp=datetime.utcnow().isoformat()
                    )
                
            except Exception as e:
                logger.error(f"Template analytics retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ================================
        # Enhanced Batch Processing API
        # ================================

        @app.post("/api/v6/batch/submit")
        async def submit_batch_job(
            request: Request,
            job_request: dict,
            priority: str = "normal",
            dependencies: Optional[List[str]] = None,
            retry_config: Optional[dict] = None,
            user=Depends(get_authenticated_user)
        ):
            """Submit job to batch processing queue with advanced options"""
            
            try:
                batch_processor = NetflixLevelBatchProcessor()
                
                # Convert priority string to enum
                priority_map = {
                    "critical": JobPriority.CRITICAL,
                    "high": JobPriority.HIGH,
                    "normal": JobPriority.NORMAL,
                    "low": JobPriority.LOW,
                    "background": JobPriority.BACKGROUND
                }
                
                job_priority = priority_map.get(priority.lower(), JobPriority.NORMAL)
                
                # Add user context to job
                job_request["user_id"] = user.get("user_id", "anonymous")
                job_request["user_tier"] = user.get("tier", "standard")
                
                # Submit job
                job_id = await batch_processor.submit_job(
                    job_request=job_request,
                    priority=job_priority,
                    dependencies=dependencies,
                    retry_config=retry_config
                )
                
                return APIResponse(
                    success=True,
                    message="Job submitted successfully",
                    data={
                        "job_id": job_id,
                        "priority": priority,
                        "estimated_start": "Within 30 seconds",
                        "queue_position": await batch_processor.get_queue_position(job_id)
                    },
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Batch job submission failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/v6/batch/status/{job_id}")
        async def get_batch_job_status(
            job_id: str,
            user=Depends(get_authenticated_user)
        ):
            """Get detailed batch job status"""
            
            try:
                batch_processor = NetflixLevelBatchProcessor()
                
                status = await batch_processor.get_job_status(job_id)
                
                if not status:
                    raise HTTPException(status_code=404, detail="Job not found")
                
                return APIResponse(
                    success=True,
                    message="Job status retrieved",
                    data=status,
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Job status retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/v6/batch/analytics")
        async def get_batch_analytics(
            user=Depends(get_authenticated_user)
        ):
            """Get advanced batch processing analytics"""
            
            try:
                batch_processor = NetflixLevelBatchProcessor()
                
                analytics = await batch_processor.get_advanced_analytics()
                
                return APIResponse(
                    success=True,
                    message="Batch analytics retrieved",
                    data=analytics,
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Batch analytics retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/v6/batch/optimize")
        async def optimize_batch_system(
            request: Request,
            optimization_config: dict = {},
            user=Depends(get_authenticated_user)
        ):
            """Apply optimization recommendations to batch system"""
            
            try:
                batch_processor = NetflixLevelBatchProcessor()
                
                # Apply optimizations
                result = await batch_processor.apply_optimizations(optimization_config)
                
                return APIResponse(
                    success=True,
                    message="Batch system optimized",
                    data=result,
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Batch optimization failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))


            try:
                # Get session info from query params or headers
                session_id = websocket.query_params.get("session_id", "default")
                user_info = {
                    "user_id": websocket.query_params.get("user_id", "anonymous"),
                    "connection_id": connection_id
                }

                await self.realtime_engine.connect_websocket(websocket, session_id, user_info)

                # Keep connection alive and handle messages
                while True:
                    try:
                        data = await websocket.receive_text()
                        message = json.loads(data)

                        # Handle client messages
                        if message.get("type") == "ping":
                            await websocket.send_text(json.dumps({
                                "type": "pong",
                                "timestamp": time.time()
                            }))

                    except WebSocketDisconnect:
                        break
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format"
                        }))
                    except Exception as e:
                        self.logger.error(f"WebSocket message error: {e}")

            except WebSocketDisconnect:
                pass
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}")
            finally:
                await self.realtime_engine.disconnect_websocket(websocket, session_id)

        # Main application route
        @app.get("/", response_class=HTMLResponse)
        async def serve_app():
            """Serve the main application"""
            try:
                async with aiofiles.open("nr1copilot/nr1-main/index.html", mode="r") as f:
                    return HTMLResponse(content=await f.read())
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Application not found")

    async def _configure_exception_handlers(self, app: FastAPI):
        """Configure custom exception handlers"""

        @app.exception_handler(HTTPException)
        async def custom_http_exception_handler(request: Request, exc: HTTPException):
            """Custom HTTP exception handler with enhanced logging"""

            self.logger.warning(
                f"HTTP Exception: {exc.status_code} - {exc.detail} - "
                f"Path: {request.url.path} - IP: {request.client.host}"
            )

            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "success": False,
                    "message": str(exc.detail),
                    "timestamp": time.time(),
                    "path": str(request.url.path)
                }
            )

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            """Custom validation exception handler"""

            self.logger.warning(f"Validation Error: {exc.errors()} - Path: {request.url.path}")

            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "Request validation failed",
                    "errors": exc.errors(),
                    "timestamp": time.time()
                }
            )

        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """General exception handler for unhandled exceptions"""

            self.logger.error(
                f"Unhandled Exception: {str(exc)} - Path: {request.url.path}",
                exc_info=True
            )

            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Internal server error",
                    "timestamp": time.time(),
                    "error_id": f"error_{int(time.time())}"
                }
            )

    async def _configure_static_files(self, app: FastAPI):
        """Configure static file serving"""

        # Mount static files
        app.mount("/static", StaticFiles(directory="nr1copilot/nr1-main/static", html=True), name="static")
        app.mount("/public", StaticFiles(directory="nr1copilot/nr1-main/public", html=True), name="public")

    async def _track_upload_init(self, upload_id: str, user_info: Dict[str, Any], file_size: int):
        """Background task to track upload initialization"""

        try:
            if self.metrics_collector:
                await self.metrics_collector.track_event("upload_initialized", {
                    "upload_id": upload_id,
                    "user_tier": user_info.get("tier", "free"),
                    "file_size": file_size,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.error(f"Analytics tracking failed: {e}")

async def check_enterprise_rate_limit(request: Request):
    """Enterprise rate limiting with adaptive throttling"""
    if not container.rate_limiter:
        return

    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")

    try:
        is_allowed, retry_after = await container.rate_limiter.is_allowed_enterprise(
            client_ip, user_agent, request.url.path
        )

        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded - Please try again later",
                headers={"Retry-After": str(retry_after)}
            )
    except Exception as e:
        logging.warning(f"Rate limiting error: {e}")
        # Fail open for availability

async def check_enterprise_health():
    """Enterprise health check with circuit breaker"""
    app_state = NetflixLevelApplicationState()
    if app_state.should_circuit_break:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable - Circuit breaker active",
            headers={"Retry-After": "60"}
        )

# ================================
# Application Instance
# ================================

# Create application factory
app_factory = NetflixLevelApplication()

# Create the FastAPI app (this will be used by uvicorn)
app = None

async def create_application():
    """Create the application instance"""
    global app
    if app is None:
        app = await app_factory.create_app()
    return app

# For uvicorn compatibility
async def get_application():
    """Get or create the application instance"""
    return await create_application()

# Create app for immediate use
import asyncio
app = asyncio.run(create_application())

# Global application state with enterprise monitoring
class NetflixLevelApplicationState:
    """Enterprise application state with circuit breaker and health monitoring"""

    def __init__(self):
        self.is_ready = False
        self.is_healthy = True
        self.startup_time = None
        self.last_health_check = None
        self.error_count = 0
        self.max_errors = 10
        self.circuit_breaker_open = False
        self.performance_metrics = {
            "total_requests": 0,
            "error_rate": 0.0,
            "avg_response_time": 0.0,
            "active_connections": 0
        }

    def mark_ready(self):
        self.is_ready = True
        self.startup_time = datetime.utcnow()
        logging.info(f"🚀 Application ready at {self.startup_time}")

    def mark_unhealthy(self, error: Exception = None):
        self.is_healthy = False
        self.error_count += 1
        if self.error_count >= self.max_errors:
            self.circuit_breaker_open = True
            logging.critical(f"🔴 Circuit breaker OPEN - Error threshold exceeded: {self.error_count}")

    def mark_healthy(self):
        self.is_healthy = True
        self.error_count = 0
        self.circuit_breaker_open = False
        self.last_health_check = datetime.utcnow()

    @property
    def should_circuit_break(self) -> bool:
        return self.circuit_breaker_open or self.error_count >= self.max_errors

    def update_metrics(self, response_time: float, is_error: bool = False):
        self.performance_metrics["total_requests"] += 1
        current_avg = self.performance_metrics["avg_response_time"]
        total_requests = self.performance_metrics["total_requests"]
        self.performance_metrics["avg_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        if is_error:
            self.performance_metrics["error_rate"] = (
                self.error_count / self.performance_metrics["total_requests"]
            )

container = DependencyContainer()

async def validate_enterprise_environment():
    """Validate enterprise environment and dependencies"""
    required_paths = [
        settings.upload_path.parent,
        settings.output_path.parent,
        settings.temp_path.parent,
        settings.cache_path.parent,
        settings.logs_path.parent
    ]

    for path in required_paths:
        if not path.exists():
            raise RuntimeError(f"Critical directory missing: {path}")

    # Validate system resources
    import psutil
    available_memory = psutil.virtual_memory().available
    if available_memory < 512 * 1024 * 1024:  # 512MB minimum
        logging.warning(f"Low memory available: {available_memory / 1024 / 1024:.1f}MB")


async def setup_enterprise_infrastructure():
    """Setup enterprise-grade infrastructure"""
    directories = [
        settings.upload_path,
        settings.output_path,
        settings.temp_path,
        settings.cache_path,
        settings.logs_path,
        Path("nr1copilot/nr1-main/metrics"),
        Path("nr1copilot/nr1-main/health"),
        Path("nr1copilot/nr1-main/templates"),
        Path("nr1copilot/nr1-main/captions"),
        Path("nr1copilot/nr1-main/batch_output")
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True, mode=0o755)
        logging.debug(f"🏗️ Infrastructure ready: {directory}")


async def warm_up_enterprise_services():
    """Warm up enterprise services with performance monitoring"""
    try:
        warm_up_tasks = []

        # AI analyzer warm-up
        if container.ai_analyzer:
            warm_up_tasks.append(container.ai_analyzer.enterprise_warm_up())

        # Cache warm-up
        if container.cache_manager:
            warm_up_tasks.append(container.cache_manager.enterprise_warm_up())

        # Video service warm-up
        if container.video_service:
            warm_up_tasks.append(container.video_service.enterprise_warm_up())

        # Execute warm-up tasks in parallel
        await asyncio.gather(*warm_up_tasks, return_exceptions=True)

        logging.info("🔥 Enterprise services warmed up successfully")

    except Exception as e:
        logging.warning(f"⚠️ Service warm-up partially failed: {e}")


async def setup_enterprise_monitoring():
    """Setup Netflix-level monitoring and observability"""
    try:
        if container.metrics_collector:
            await container.metrics_collector.start_collection()

        if container.health_checker:
            await container.health_checker.start_monitoring()

        logging.info("📊 Enterprise monitoring active")

    except Exception as e:
        logging.error(f"❌ Monitoring setup failed: {e}")

@asynccontextmanager
async def netflix_level_lifespan(app: FastAPI):
    """Netflix-level application lifecycle with comprehensive startup/shutdown"""
    startup_start = time.time()

    try:
        logging.info("🎬 Starting ViralClip Pro v6.0 - Netflix Enterprise Architecture")

        # Phase 1: Environment validation
        await validate_enterprise_environment()

        # Phase 2: Infrastructure setup
        await setup_enterprise_infrastructure()

        # Phase 3: Dependency injection initialization
        await container.initialize_enterprise_services()

        # Phase 4: Service warm-up with performance monitoring
        await warm_up_enterprise_services()

        # Phase 5: Health checks and monitoring setup
        await setup_enterprise_monitoring()

        # Mark application as production-ready
        app_state = NetflixLevelApplicationState()
        app_state.mark_ready()
        startup_time = time.time() - startup_start
        logging.info(f"✅ Netflix-level application ready in {startup_time:.2f}s")

        yield

    except Exception as e:
        logging.error(f"❌ Enterprise startup failed: {e}", exc_info=True)
        app_state = NetflixLevelApplicationState()
        app_state.mark_unhealthy(e)
        raise
    finally:
        logging.info("🔄 Initiating graceful enterprise shutdown...")
        await container.graceful_shutdown()
        logging.info("✅ Enterprise shutdown complete")

# ================================
# Main Entry Point
# ================================

if __name__ == "__main__":
    # Development server configuration
    uvicorn_config = {
        "app": "app.main:app",
        "host": "0.0.0.0",
        "port": 5000,
        "reload": True,
        "log_level": "info",
        "access_log": True,
        "workers": 1,
        "loop": "uvloop",
        "http": "httptools",
        "lifespan": "on",
        "server_header": False,
        "date_header": False
    }

    logging.info("🚀 Starting Netflix-level ViralClip Pro Enterprise")
    uvicorn.run(**uvicorn_config)

# The following code implements Netflix-level upload API endpoints and WebSocket for real-time updates.
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import time
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
import os
import aiofiles
from datetime import datetime

# Import our services
from app.services.video_service import VideoService
from app.services.ai_analyzer import AIAnalyzer
from app.services.realtime_engine import RealtimeEngine
from app.config import settings
from app.schemas import UploadSession, ProcessingResult
from app.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI with enhanced configuration
app = FastAPI(
    title="ViralClip Pro - Netflix-Level Video Processing Platform",
    description="Advanced AI-powered video processing with real-time analytics",
    version="7.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
video_service = VideoService()
ai_analyzer = AIAnalyzer()
realtime_engine = RealtimeEngine()

# Global storage for active sessions
active_sessions: Dict[str, Dict] = {}
active_websockets: Dict[str, WebSocket] = {}
upload_sessions: Dict[str, Dict] = {}
chunk_storage: Dict[str, Dict] = {}

# Create upload directories
UPLOAD_DIR = Path("uploads")
TEMP_DIR = Path("temp")
CHUNKS_DIR = Path("chunks")

for directory in [UPLOAD_DIR, TEMP_DIR, CHUNKS_DIR]:
    directory.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>ViralClip Pro</h1><p>Application not found</p>",
            status_code=404
        )

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "version": "7.0.0",
        "timestamp": time.time(),
        "services": {
            "video_service": "operational",
            "ai_analyzer": "operational",
            "realtime_engine": "operational"
        },
        "system": {
            "active_uploads": len(upload_sessions),
            "active_websockets": len(active_websockets),
            "disk_usage": _get_disk_usage()
        }
    }

def _get_disk_usage():
    """Get disk usage statistics"""
    try:
        total, used, free = shutil.disk_usage("/")
        return {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "usage_percent": round((used/total) * 100, 1)
        }
    except:
        return {"error": "Unable to get disk usage"}

# ================================
# Netflix-Level Upload API
# ================================

@app.post("/api/v7/upload/init")
async def initialize_upload(
    filename: str = Form(...),
    file_size: int = Form(...),
    total_chunks: int = Form(...),
    upload_id: str = Form(...),
    metadata: str = Form("{}")
):
    """Initialize a chunked upload session"""
    try:
        session_id = f"session_{uuid.uuid4().hex[:12]}"

        # Parse metadata
        try:
            parsed_metadata = json.loads(metadata)
        except:
            parsed_metadata = {}

        # Create session
        session_data = {
            "session_id": session_id,
            "upload_id": upload_id,
            "filename": filename,
            "file_size": file_size,
            "total_chunks": total_chunks,
            "received_chunks": set(),
            "metadata": parsed_metadata,
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
            "chunk_hashes": {}
        }

        upload_sessions[session_id] = session_data
        chunk_storage[session_id] = {}

        # Create session directory
        session_dir = CHUNKS_DIR / session_id
        session_dir.mkdir(exist_ok=True)

        logger.info(f"Upload session initialized: {session_id} for file: {filename}")

        return {
            "session_id": session_id,
            "status": "initialized",
            "upload_url": f"/api/v7/upload/chunk",
            "finalize_url": f"/api/v7/upload/finalize"
        }

    except Exception as e:
        logger.error(f"Upload initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload initialization failed: {str(e)}")

@app.post("/api/v7/upload/chunk")
async def upload_chunk(
    file: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk_hash: str = Form(...),
    session_id: str = Form(...)
):
    """Upload a file chunk with integrity verification"""
    try:
        # Validate session
        if session_id not in upload_sessions:
            raise HTTPException(status_code=404, detail="Upload session not found")

        session = upload_sessions[session_id]

        # Read chunk data
        chunk_data = await file.read()

        # Verify chunk hash
        calculated_hash = hashlib.sha256(chunk_data).hexdigest()
        if calculated_hash != chunk_hash:
            raise HTTPException(status_code=400, detail="Chunk integrity check failed")

        # Save chunk to disk
        chunk_path = CHUNKS_DIR / session_id / f"chunk_{chunk_index}"
        async with aiofiles.open(chunk_path, "wb") as f:
            await f.write(chunk_data)

        # Update session
        session["received_chunks"].add(chunk_index)
        session["chunk_hashes"][chunk_index] = chunk_hash
        chunk_storage[session_id][chunk_index] = {
            "path": str(chunk_path),
            "hash": chunk_hash,
            "size": len(chunk_data),
            "received_at": datetime.now().isoformat()
        }

        # Broadcast progress via WebSocket
        if session_id in active_websockets:
            progress = len(session["received_chunks"]) / session["total_chunks"] * 100
            await active_websockets[session_id].send_text(json.dumps({
                "type": "chunk_uploaded",
                "session_id": session_id,
                "chunk_index": chunk_index,
                "progress": progress,
                "chunks_received": len(session["received_chunks"]),
                "total_chunks": session["total_chunks"]
            }))

        logger.info(f"Chunk {chunk_index} uploaded for session {session_id}")

        return {
            "status": "success",
            "chunk_index": chunk_index,
            "chunks_received": len(session["received_chunks"]),
            "total_chunks": session["total_chunks"],
            "progress": len(session["received_chunks"]) / session["total_chunks"] * 100
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chunk upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk upload failed: {str(e)}")

@app.post("/api/v7/upload/finalize")
async def finalize_upload(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
    session_id: str = Form(...)
):
    """Finalize upload by assembling chunks and starting processing"""
    try:
        # Validate session
        if session_id not in upload_sessions:
            raise HTTPException(status_code=404, detail="Upload session not found")

        session = upload_sessions[session_id]

        # Verify all chunks received
        if len(session["received_chunks"]) != session["total_chunks"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing chunks. Received: {len(session['received_chunks'])}, Expected: {session['total_chunks']}"
            )

        # Assemble file
        final_path = UPLOAD_DIR / f"{session_id}_{session['filename']}"

        async with aiofiles.open(final_path, "wb") as output_file:
            for chunk_index in sorted(session["received_chunks"]):
                chunk_path = CHUNKS_DIR / session_id / f"chunk_{chunk_index}"
                async with aiofiles.open(chunk_path, "rb") as chunk_file:
                    chunk_data = await chunk_file.read()
                    await output_file.write(chunk_data)

        # Update session status
        session["status"] = "assembled"
        session["final_path"] = str(final_path)
        session["assembled_at"] = datetime.now().isoformat()

        # Start background processing
        background_tasks.add_task(process_uploaded_file, session_id, str(final_path))

        # Cleanup chunks
        background_tasks.add_task(cleanup_chunks, session_id)

        logger.info(f"Upload finalized for session {session_id}")

        return {
            "status": "success",
            "session_id": session_id,
            "file_path": str(final_path),
            "processing_started": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload finalization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload finalization failed: {str(e)}")

async def process_uploaded_file(session_id: str, file_path: str):
    """Background task to process uploaded file"""
    try:
        session = upload_sessions[session_id]
        session["status"] = "processing"

        # Notify via WebSocket
        if session_id in active_websockets:
            await active_websockets[session_id].send_text(json.dumps({
                "type": "processing_started",
                "session_id": session_id,
                "message": "AI analysis and processing started"
            }))

        # Start AI analysis (mock for now)
        await asyncio.sleep(2)  # Simulate processing time

        session["status"] = "completed"
        session["completed_at"] = datetime.now().isoformat()

        # Notify completion
        if session_id in active_websockets:
            await active_websockets[session_id].send_text(json.dumps({
                "type": "processing_completed",
                "session_id": session_id,
                "message": "Video processing completed successfully",
                "results": {
                    "viral_score": 87,
                    "processing_time": "2.3s",
                    "optimizations_applied": 5
                }
            }))

        logger.info(f"Processing completed for session {session_id}")

    except Exception as e:
        logger.error(f"Processing failed for session {session_id}: {e}")
        session["status"] = "failed"
        session["error"] = str(e)

async def cleanup_chunks(session_id: str):
    """Clean up temporary chunk files"""
    try:
        chunk_dir = CHUNKS_DIR / session_id
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)

        if session_id in chunk_storage:
            del chunk_storage[session_id]

        logger.info(f"Cleaned up chunks for session {session_id}")

    except Exception as e:
        logger.error(f"Chunk cleanup failed for session {session_id}: {e}")

@app.get("/api/v7/upload/status/{session_id}")
async def get_upload_status(session_id: str):
    """Get upload session status"""
    if session_id not in upload_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = upload_sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "progress": len(session["received_chunks"]) / session["total_chunks"] * 100,
        "chunks_received": len(session["received_chunks"]),
        "total_chunks": session["total_chunks"],
        "created_at": session["created_at"],
        "metadata": session.get("metadata", {})
    }

@app.post("/api/v7/metrics/batch")
async def receive_metrics_batch(metrics_data: dict):
    """Receive and process metrics batch from frontend"""
    try:
        metrics = metrics_data.get("metrics", [])

        # Process metrics (store in database, send to analytics service, etc.)
        logger.info(f"Received {len(metrics)} metrics")

        # For now, just log important metrics
        for metric in metrics:
            if metric.get("event", "").startswith("error:"):
                logger.error(f"Frontend error: {metric}")

        return {"status": "success", "processed": len(metrics)}

    except Exception as e:
        logger.error(f"Metrics processing failed: {e}")
        return {"status": "error", "message": str(e)}

# ================================
# WebSocket for Real-time Updates
# ================================

@app.websocket("/ws/upload/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time upload updates"""
    await websocket.accept()
    active_websockets[session_id] = websocket

    try:
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "message": "Real-time connection established"
        }))

        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": time.time()
                }))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        if session_id in active_websockets:
            del active_websockets[session_id]