"""
ViralClip Pro v10.0 - Netflix-Grade Video Editing Platform
Optimized architecture with enterprise performance and maintainability
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Core imports
from app.config import get_settings
from app.logging_config import setup_logging
from app.services.dependency_container import ServiceContainer
from app.middleware.performance import PerformanceMiddleware
from app.middleware.security import SecurityMiddleware
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.utils.health import SystemHealthMonitor

# Services imports
from app.services.video_service import NetflixLevelVideoService
from app.services.ai_analyzer import AIVideoAnalyzer
from app.services.social_publisher import SocialMediaPublisher
from app.services.analytics_engine import AnalyticsEngine
from app.services.collaboration_engine import CollaborationEngine
from app.services.realtime_engine import RealtimeEngine
from app.services.template_service import TemplateService
from app.services.batch_processor import NetflixLevelBatchProcessor
from app.services.ultimate_perfection_engine import UltimatePerfectionEngine
from app.services.enterprise_manager import EnterpriseManager
from app.services.video_pipeline import NetflixLevelVideoPipeline
from app.services.ffmpeg_processor import NetflixLevelFFmpegProcessor

# Application metadata
APPLICATION_INFO = {
    "name": "ViralClip Pro v10.0",
    "version": "10.0.0",
    "description": "Netflix-Grade Video Editing & Social Platform",
    "tier": "ENTERPRISE",
    "performance_grade": "A+",
    "architecture": "MICROSERVICES_READY"
}

# Global service container
services = ServiceContainer()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized application lifecycle with error recovery"""
    startup_start = time.time()

    try:
        logger.info("🚀 Initializing ViralClip Pro v10.0...")

        # Initialize services
        await services.initialize()
        app.state.video_service = NetflixLevelVideoService()
        app.state.ai_analyzer = AIVideoAnalyzer()
        app.state.social_publisher = SocialMediaPublisher()
        app.state.analytics_engine = AnalyticsEngine()
        app.state.collaboration_engine = CollaborationEngine()
        app.state.realtime_engine = RealtimeEngine()
        app.state.template_service = TemplateService()
        app.state.batch_processor = NetflixLevelBatchProcessor()
        app.state.perfection_engine = UltimatePerfectionEngine()
        app.state.enterprise_manager = EnterpriseManager()
        app.state.video_pipeline = NetflixLevelVideoPipeline()
        app.state.ffmpeg_processor = NetflixLevelFFmpegProcessor()

        # Start health monitoring
        health_monitor = services.get_health_monitor()
        await health_monitor.start_monitoring()

        await app.state.video_service.startup()
        await app.state.ai_analyzer.warm_up()
        await app.state.social_publisher.startup()
        await app.state.analytics_engine.startup()
        await app.state.collaboration_engine.startup()
        await app.state.realtime_engine.startup()
        await app.state.template_service.startup()
        await app.state.batch_processor.startup()
        await app.state.perfection_engine.startup()
        await app.state.enterprise_manager.startup()
        await app.state.video_pipeline.startup()
        await app.state.ffmpeg_processor.startup()

        startup_duration = time.time() - startup_start
        logger.info(f"✅ Startup completed in {startup_duration:.2f}s")

        yield

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise
    finally:
        await app.state.video_service.shutdown()
        await app.state.social_publisher.shutdown()
        await app.state.analytics_engine.shutdown()
        await app.state.collaboration_engine.shutdown()
        await app.state.realtime_engine.shutdown()
        await app.state.template_service.shutdown()
        await app.state.batch_processor.shutdown()
        await app.state.perfection_engine.shutdown()
        await app.state.enterprise_manager.shutdown()
        await app.state.video_pipeline.shutdown()
        await app.state.ffmpeg_processor.shutdown()
        await services.shutdown()
        logger.info("✅ Graceful shutdown completed")


# Create FastAPI application
app = FastAPI(
    title=APPLICATION_INFO["name"],
    description=APPLICATION_INFO["description"],
    version=APPLICATION_INFO["version"],
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "System health and monitoring"},
        {"name": "Video", "description": "Video processing and management"},
        {"name": "AI", "description": "AI intelligence and automation"},
        {"name": "Analytics", "description": "Performance analytics"},
        {"name": "Enterprise", "description": "Enterprise features"}
    ]
)

# Middleware stack (order matters for performance)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(ErrorHandlerMiddleware)


# Health endpoints
@app.get("/health", tags=["Health"])
async def comprehensive_health_check():
    """Netflix-level comprehensive health check"""
    start_time = time.time()

    try:
        health_monitor = services.get_health_monitor()
        health_data = await health_monitor.get_comprehensive_health()

        health_data["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        health_data["application"] = APPLICATION_INFO

        status_code = 200 if health_data.get("overall_score", 0) >= 7.0 else 503
        return JSONResponse(content=health_data, status_code=status_code)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "critical",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            },
            status_code=503
        )


@app.get("/health/summary", tags=["Health"])
async def health_summary():
    """Quick health summary for load balancers"""
    try:
        health_monitor = services.get_health_monitor()
        summary = await health_monitor.get_health_summary()
        summary["application"] = APPLICATION_INFO["name"]
        return summary
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/metrics", tags=["Analytics"])
async def performance_metrics():
    """Comprehensive performance metrics"""
    try:
        metrics_collector = services.get_metrics_collector()
        database_manager = services.get_database_manager()
        cache_manager = services.get_cache_manager()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "application": APPLICATION_INFO,
            "database": await database_manager.get_pool_stats(),
            "cache": await cache_manager.get_stats(),
            "system": await metrics_collector.get_detailed_metrics()
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/status", tags=["Health"])
async def application_status():
    """Application status and metadata"""
    return {
        "application": APPLICATION_INFO,
        "environment": settings.environment,
        "timestamp": datetime.utcnow().isoformat(),
        "services": services.get_service_status(),
        "healthy": services.is_healthy()
    }


# Business logic endpoints
@app.get("/api/v1/videos", tags=["Video"])
async def list_videos():
    """List videos with enterprise features"""
    try:
        video_service = services.get_video_service()
        return await video_service.list_videos()
    except Exception as e:
        logger.error(f"Video listing failed: {e}")
        raise HTTPException(status_code=500, detail="Video service unavailable")


@app.get("/api/v1/ai/analyze", tags=["AI"])
async def ai_analyze():
    """AI analysis endpoint"""
    try:
        ai_engine = services.get_ai_engine()
        return await ai_engine.get_analysis_status()
    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        raise HTTPException(status_code=500, detail="AI service unavailable")


# Static file serving with error handling
try:
    static_path = "static"
    if settings.base_dir:
        static_path = str(settings.base_dir / "static")
    app.mount("/static", StaticFiles(directory=static_path), name="static")
except Exception as e:
    logger.warning(f"Static files not available: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Optimized root endpoint with fallback"""
    try:
        index_path = settings.base_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.error(f"Failed to read index.html: {e}")

    return HTMLResponse(content=_generate_enterprise_html())


def _generate_enterprise_html() -> str:
    """Generate optimized enterprise HTML"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>{APPLICATION_INFO['name']}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .container {{
                text-align: center;
                padding: 2rem;
                background: rgba(255,255,255,0.1);
                border-radius: 20px;
                backdrop-filter: blur(10px);
                max-width: 600px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }}
            h1 {{
                font-size: 2.5rem;
                margin-bottom: 1rem;
                background: linear-gradient(45deg, #FFD700, #FFA500);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .status {{
                font-size: 1.1rem;
                margin: 1rem 0;
                opacity: 0.9;
            }}
            .links {{
                margin-top: 2rem;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 1rem;
            }}
            .links a {{
                padding: 0.8rem 1rem;
                background: rgba(255,255,255,0.2);
                color: white;
                text-decoration: none;
                border-radius: 10px;
                transition: all 0.3s ease;
                font-weight: 500;
            }}
            .links a:hover {{
                background: rgba(255,255,255,0.3);
                transform: translateY(-2px);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌟 {APPLICATION_INFO['name']} 🌟</h1>
            <div class="status">
                <p>🏆 Grade: {APPLICATION_INFO['performance_grade']}</p>
                <p>⚡ Status: <strong>ENTERPRISE READY</strong></p>
            </div>
            <div class="links">
                <a href="/health">🏥 Health</a>
                <a href="/metrics">📊 Metrics</a>
                <a href="/status">📋 Status</a>
                <a href="/api/docs">📚 API Docs</a>
            </div>
        </div>
    </body>
    </html>
    """


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=settings.debug,
        log_level="info",
        access_log=True,
        use_colors=True
    )