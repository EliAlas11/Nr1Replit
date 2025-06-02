
"""
ViralClip Pro v11.0 - Netflix-Grade Video Editing Platform
Production-optimized with enterprise reliability and performance
"""

import asyncio
import logging
import time
import gc
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Core imports
from app.config import get_settings, Settings
from app.logging_config import setup_logging
from app.services.dependency_container import ServiceContainer
from app.middleware.performance import PerformanceMiddleware
from app.middleware.security import SecurityMiddleware
from app.middleware.error_handler import ErrorHandlerMiddleware

# Routes
from app.routes import auth, enterprise, websocket

# Application metadata
APPLICATION_INFO = {
    "name": "ViralClip Pro v11.0",
    "version": "11.0.0",
    "description": "Netflix-Grade Video Editing & AI-Powered Social Platform",
    "tier": "ENTERPRISE_NETFLIX_GRADE",
    "performance_grade": "A++",
    "architecture": "PRODUCTION_READY"
}

# Global instances
services = ServiceContainer()
settings = get_settings()

# Setup structured logging
setup_logging()
logger = logging.getLogger(__name__)


class GracefulShutdownHandler:
    """Handles graceful shutdown with signal management"""
    
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.trigger_shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def trigger_shutdown(self):
        """Trigger graceful shutdown"""
        self.shutdown_event.set()


shutdown_handler = GracefulShutdownHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Netflix-optimized application lifecycle with comprehensive error handling"""
    startup_start = time.time()
    
    try:
        logger.info("🚀 Starting ViralClip Pro v11.0 (Netflix-Grade Production Mode)")
        
        # Phase 1: Initialize core services
        logger.info("📊 Phase 1: Core Services Initialization")
        await services.initialize()
        
        # Phase 2: Initialize application services
        logger.info("🎬 Phase 2: Application Services Initialization")
        await _initialize_application_services(app)
        
        # Phase 3: Setup health monitoring
        logger.info("🏥 Phase 3: Health Monitoring Setup")
        health_monitor = services.get_health_monitor()
        await health_monitor.initialize()
        
        # Phase 4: Setup background tasks
        logger.info("⚙️ Phase 4: Background Tasks Setup")
        await _setup_background_tasks(app)
        
        # Phase 5: Memory optimization
        logger.info("🧠 Phase 5: Memory Optimization")
        gc.collect()
        
        startup_duration = time.time() - startup_start
        memory_usage = _get_memory_usage()
        
        logger.info(f"✅ ViralClip Pro v11.0 started successfully in {startup_duration:.2f}s")
        logger.info(f"📊 Memory usage: {memory_usage:.1f}MB")
        logger.info(f"🌐 Environment: {settings.environment.value}")
        logger.info(f"🔐 Security: Enterprise-grade with 2FA")
        logger.info(f"⚡ Performance: Netflix-level optimization active")
        logger.info("🎯 All systems operational - Ready for production traffic!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Critical startup failure: {e}", exc_info=True)
        raise
        
    finally:
        # Graceful shutdown
        shutdown_start = time.time()
        logger.info("🔄 Initiating Netflix-grade graceful shutdown...")
        
        try:
            # Stop background tasks
            if hasattr(app.state, 'background_tasks'):
                for task in app.state.background_tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait for tasks to complete
                await asyncio.gather(*app.state.background_tasks, return_exceptions=True)
            
            # Shutdown services
            await services.shutdown()
            
            # Final cleanup
            gc.collect()
            
            shutdown_duration = time.time() - shutdown_start
            logger.info(f"✅ Graceful shutdown completed in {shutdown_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"⚠️ Shutdown error: {e}")


async def _initialize_application_services(app: FastAPI) -> None:
    """Initialize core application services with error handling"""
    try:
        # Video processing services
        from app.services.video_service import NetflixLevelVideoService
        from app.services.video_pipeline import NetflixLevelVideoPipeline
        from app.services.ffmpeg_processor import NetflixLevelFFmpegProcessor
        
        app.state.video_service = NetflixLevelVideoService()
        app.state.video_pipeline = NetflixLevelVideoPipeline()
        app.state.ffmpeg_processor = NetflixLevelFFmpegProcessor()
        
        # AI and analytics services
        from app.services.ai_analyzer import AIVideoAnalyzer
        from app.services.analytics_engine import AnalyticsEngine
        from app.services.ai_production_engine import ai_production_engine
        
        app.state.ai_analyzer = AIVideoAnalyzer()
        app.state.analytics_engine = AnalyticsEngine()
        app.state.ai_production_engine = ai_production_engine
        
        # WebSocket engine
        if settings.enable_websockets:
            from app.services.websocket_engine import websocket_engine
            await websocket_engine.start_engine()
            app.state.websocket_engine = websocket_engine
        
        # Initialize all services
        for service_name in dir(app.state):
            if not service_name.startswith('_'):
                service = getattr(app.state, service_name)
                if hasattr(service, 'startup'):
                    await service.startup()
                    logger.debug(f"✅ {service_name} initialized")
        
        logger.info("🎬 All application services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise


async def _setup_background_tasks(app: FastAPI) -> None:
    """Setup background tasks for monitoring and maintenance"""
    app.state.background_tasks = []
    
    try:
        # Health monitoring task
        health_task = asyncio.create_task(_background_health_monitoring())
        app.state.background_tasks.append(health_task)
        
        # Performance monitoring task
        perf_task = asyncio.create_task(_background_performance_monitoring())
        app.state.background_tasks.append(perf_task)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(_background_cleanup())
        app.state.background_tasks.append(cleanup_task)
        
        logger.info(f"⚙️ {len(app.state.background_tasks)} background tasks started")
        
    except Exception as e:
        logger.error(f"❌ Background task setup failed: {e}")


async def _background_health_monitoring():
    """Background health monitoring task"""
    while not shutdown_handler.shutdown_event.is_set():
        try:
            health_monitor = services.get_health_monitor()
            await health_monitor.perform_health_check()
            await asyncio.sleep(30)  # Check every 30 seconds
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
            await asyncio.sleep(60)


async def _background_performance_monitoring():
    """Background performance monitoring task"""
    while not shutdown_handler.shutdown_event.is_set():
        try:
            # Collect performance metrics
            metrics_collector = services.get_metrics_collector()
            await metrics_collector.collect_metrics()
            
            # Trigger GC if memory usage is high
            memory_mb = _get_memory_usage()
            if memory_mb > settings.performance.max_memory_usage_mb * 0.8:
                gc.collect()
                logger.info(f"🧠 Triggered GC, memory: {memory_mb:.1f}MB")
            
            await asyncio.sleep(60)  # Check every minute
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            await asyncio.sleep(120)


async def _background_cleanup():
    """Background cleanup task for temporary files and caches"""
    while not shutdown_handler.shutdown_event.is_set():
        try:
            # Clean up temporary files
            temp_files_cleaned = await _cleanup_temp_files()
            if temp_files_cleaned > 0:
                logger.info(f"🧹 Cleaned up {temp_files_cleaned} temporary files")
            
            await asyncio.sleep(3600)  # Run every hour
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(1800)


async def _cleanup_temp_files() -> int:
    """Clean up temporary files older than 1 hour"""
    import os
    import time
    
    temp_path = settings.temp_path
    if not temp_path.exists():
        return 0
    
    cleaned_count = 0
    current_time = time.time()
    
    try:
        for file_path in temp_path.rglob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > 3600:  # 1 hour
                    file_path.unlink()
                    cleaned_count += 1
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")
    
    return cleaned_count


def _get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


# Create FastAPI application with Netflix-grade configuration
app = FastAPI(
    title=APPLICATION_INFO["name"],
    description=APPLICATION_INFO["description"],
    version=APPLICATION_INFO["version"],
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
    openapi_url="/api/openapi.json" if settings.debug else None,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "System health and monitoring"},
        {"name": "Video", "description": "Video processing and management"},
        {"name": "AI", "description": "AI analysis and optimization"},
        {"name": "Analytics", "description": "Performance and usage analytics"},
        {"name": "Enterprise", "description": "Enterprise features and management"},
        {"name": "Auth", "description": "Authentication and authorization"},
        {"name": "WebSocket", "description": "Real-time communication"}
    ]
)

# Netflix-grade middleware stack (order optimized for performance)

# 1. Trusted hosts (security boundary)
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "*.replit.app",
            "*.replit.dev",
            "*.replit.com",
            "localhost",
            "127.0.0.1"
        ]
    )

# 2. CORS (before compression)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    max_age=86400,  # Cache preflight for 24 hours
)

# 3. Compression (high priority)
app.add_middleware(
    GZipMiddleware,
    minimum_size=settings.performance.gzip_minimum_size,
    compresslevel=settings.performance.gzip_compression_level
)

# 4. Performance monitoring
app.add_middleware(PerformanceMiddleware)

# 5. Security middleware
app.add_middleware(SecurityMiddleware)

# 6. Error handling (last middleware)
app.add_middleware(ErrorHandlerMiddleware)

# Include routers
app.include_router(auth.router, prefix="/api/v1", tags=["Auth"])
app.include_router(enterprise.router, prefix="/api/v1", tags=["Enterprise"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])

# AI Production routes
try:
    from app.routes import ai_production
    app.include_router(ai_production.router, prefix="/api/v1", tags=["AI Production"])
except ImportError:
    logger.warning("AI production routes not available")


# Netflix-grade health endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Ultra-fast health check for load balancers (<10ms target)"""
    start_time = time.time()
    
    try:
        # Quick health verification
        status = "healthy"
        
        # Check critical services
        if not services.is_healthy():
            status = "degraded"
        
        response_time = round((time.time() - start_time) * 1000, 2)
        
        return Response(
            content=f'{{"status":"{status}","response_time_ms":{response_time},"version":"11.0.0"}}',
            media_type="application/json",
            status_code=200 if status == "healthy" else 503,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Health-Check": "netflix-grade",
                "X-Version": "11.0.0"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(
            content='{"status":"error","error":"health_check_failed"}',
            media_type="application/json",
            status_code=503
        )


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Comprehensive health check for monitoring systems"""
    start_time = time.time()
    
    try:
        health_monitor = services.get_health_monitor()
        health_data = await health_monitor.perform_health_check()
        
        health_data["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        health_data["application"] = APPLICATION_INFO
        health_data["memory_usage_mb"] = _get_memory_usage()
        health_data["configuration"] = {
            "environment": settings.environment.value,
            "performance_grade": APPLICATION_INFO["performance_grade"],
            "features": settings.get_feature_flags()
        }
        
        status_code = 200 if health_data.get("status") == "healthy" else 503
        return JSONResponse(content=health_data, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            content={
                "status": "critical",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            },
            status_code=503
        )


@app.get("/metrics", tags=["Analytics"])
async def performance_metrics():
    """Comprehensive performance metrics for monitoring"""
    try:
        metrics_collector = services.get_metrics_collector()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "application": APPLICATION_INFO,
            "system": await metrics_collector.get_system_metrics(),
            "performance": await metrics_collector.get_performance_metrics(),
            "services": services.get_service_status(),
            "memory_usage_mb": _get_memory_usage(),
            "configuration": settings.get_performance_config()
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics temporarily unavailable")


@app.get("/status", tags=["Health"])
async def application_status():
    """Application status and metadata"""
    return {
        "application": APPLICATION_INFO,
        "environment": settings.environment.value,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - services.start_time if hasattr(services, 'start_time') else 0,
        "services_healthy": services.is_healthy(),
        "features": settings.get_feature_flags(),
        "version": "11.0.0",
        "netflix_grade": True
    }


# Static file serving with optimized caching
try:
    static_path = settings.base_dir / "static"
    if static_path.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(static_path), html=True),
            name="static"
        )
        logger.info(f"📁 Static files mounted: {static_path}")
except Exception as e:
    logger.warning(f"Static files not available: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Optimized root endpoint with Netflix-grade dashboard"""
    try:
        # Try to serve custom index.html
        index_path = settings.base_dir / "index.html"
        if index_path.exists():
            content = index_path.read_text(encoding='utf-8')
            return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Failed to read index.html: {e}")
    
    # Fallback to generated Netflix-grade dashboard
    return HTMLResponse(content=_generate_netflix_dashboard())


def _generate_netflix_dashboard() -> str:
    """Generate Netflix-grade enterprise dashboard"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{APPLICATION_INFO['name']} - Enterprise Dashboard</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
                background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
                color: #ffffff;
                min-height: 100vh;
                overflow-x: hidden;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }}
            .header {{
                text-align: center;
                margin-bottom: 3rem;
                position: relative;
            }}
            .header::before {{
                content: '';
                position: absolute;
                top: -20px;
                left: 50%;
                transform: translateX(-50%);
                width: 100px;
                height: 4px;
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
                border-radius: 2px;
            }}
            h1 {{
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                animation: glow 2s ease-in-out infinite alternate;
            }}
            @keyframes glow {{
                from {{ filter: brightness(1); }}
                to {{ filter: brightness(1.2); }}
            }}
            .subtitle {{
                font-size: 1.2rem;
                color: #a0a0a0;
                margin-bottom: 0.5rem;
            }}
            .status {{
                display: inline-block;
                padding: 0.5rem 1rem;
                background: rgba(76, 175, 80, 0.2);
                border: 1px solid #4CAF50;
                border-radius: 20px;
                color: #4CAF50;
                font-weight: 600;
                margin-top: 1rem;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                margin-top: 3rem;
            }}
            .card {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 2rem;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            .card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
            }}
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                border-color: rgba(255, 255, 255, 0.2);
            }}
            .card-title {{
                font-size: 1.3rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: #ffffff;
            }}
            .card-content {{
                color: #b0b0b0;
                line-height: 1.6;
            }}
            .links {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 1rem;
                margin-top: 2rem;
            }}
            .link {{
                display: block;
                padding: 1rem;
                background: rgba(255, 255, 255, 0.1);
                color: #ffffff;
                text-decoration: none;
                border-radius: 10px;
                text-align: center;
                font-weight: 500;
                transition: all 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            .link:hover {{
                background: rgba(255, 255, 255, 0.2);
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            }}
            .badge {{
                display: inline-block;
                padding: 0.3rem 0.8rem;
                background: rgba(69, 183, 209, 0.2);
                color: #45b7d1;
                border-radius: 15px;
                font-size: 0.9rem;
                font-weight: 500;
                margin: 0.2rem;
            }}
            .version {{
                position: absolute;
                top: 1rem;
                right: 1rem;
                padding: 0.5rem 1rem;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 20px;
                font-size: 0.9rem;
                color: #a0a0a0;
            }}
        </style>
    </head>
    <body>
        <div class="version">v{APPLICATION_INFO['version']}</div>
        <div class="container">
            <div class="header">
                <h1>🌟 {APPLICATION_INFO['name']} 🌟</h1>
                <div class="subtitle">{APPLICATION_INFO['description']}</div>
                <div class="status">🚀 ENTERPRISE READY • {APPLICATION_INFO['performance_grade']} GRADE</div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <div class="card-title">🎬 Video Processing Engine</div>
                    <div class="card-content">
                        Netflix-grade video processing with AI optimization, real-time rendering, and multi-platform delivery.
                        <br><br>
                        <div class="badge">FFmpeg Integration</div>
                        <div class="badge">AI Enhancement</div>
                        <div class="badge">Batch Processing</div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title">🤖 AI Intelligence Hub</div>
                    <div class="card-content">
                        Advanced AI analysis for content optimization, viral prediction, and automated social media publishing.
                        <br><br>
                        <div class="badge">Content Analysis</div>
                        <div class="badge">Viral Prediction</div>
                        <div class="badge">Auto Publishing</div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title">📊 Analytics & Monitoring</div>
                    <div class="card-content">
                        Comprehensive performance monitoring with real-time metrics, health checks, and predictive analytics.
                        <br><br>
                        <div class="badge">Real-time Metrics</div>
                        <div class="badge">Health Monitoring</div>
                        <div class="badge">Predictive Analytics</div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title">🔐 Enterprise Security</div>
                    <div class="card-content">
                        Bank-level security with 2FA, JWT authentication, rate limiting, and comprehensive audit logging.
                        <br><br>
                        <div class="badge">2FA Authentication</div>
                        <div class="badge">Rate Limiting</div>
                        <div class="badge">Audit Logging</div>
                    </div>
                </div>
            </div>
            
            <div class="links">
                <a href="/health" class="link">🏥 Health Status</a>
                <a href="/metrics" class="link">📊 System Metrics</a>
                <a href="/status" class="link">📋 App Status</a>
                <a href="/api/docs" class="link">📚 API Documentation</a>
            </div>
        </div>
        
        <script>
            // Auto-refresh health status
            setInterval(async () => {{
                try {{
                    const response = await fetch('/health');
                    const data = await response.json();
                    console.log('Health check:', data);
                }} catch (e) {{
                    console.warn('Health check failed:', e);
                }}
            }}, 30000);
        </script>
    </body>
    </html>
    """


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource not found",
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Custom 500 handler"""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": id(request)
        }
    )


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.value.lower(),
        access_log=True,
        use_colors=True,
        loop="uvloop" if not settings.debug else "auto",
        http="httptools" if not settings.debug else "auto"
    )
