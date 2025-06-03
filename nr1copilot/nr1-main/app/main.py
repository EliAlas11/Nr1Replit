"""
Netflix-Grade Video Production Platform - Main Application
Enterprise-level FastAPI application with comprehensive system architecture
"""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Core system imports
from app.config import get_settings
from app.logging_config import setup_logging
from app.startup_validator import startup_validator
from app.crash_recovery_manager import crash_recovery_manager
from app.netflix_health_monitor import health_monitor
from app.netflix_recovery_system import recovery_system

# Service imports
from app.services.video_service import NetflixLevelVideoService
from app.services.ai_production_engine import AIProductionEngine
from app.services.storage_service import NetflixStorageService
from app.services.ai_analyzer import NetflixLevelAIAnalyzer
from app.services.enterprise_manager import EnterpriseManager

# Route imports
from app.routes import auth, websocket, storage, enterprise, ai_production
from app.routes import health_endpoints

# Middleware imports
from app.middleware.performance import PerformanceMiddleware
from app.middleware.security import SecurityMiddleware
from app.middleware.error_handler import ErrorHandlerMiddleware

# Utils imports
from app.utils.metrics import metrics_collector
from app.utils.performance_monitor import performance_monitor

# Configure logging
logger = setup_logging()

# Global service instances
video_service: NetflixLevelVideoService = None
ai_service: AIProductionEngine = None
storage_service: NetflixStorageService = None
ai_analyzer: NetflixLevelAIAnalyzer = None
enterprise_manager: EnterpriseManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Production-grade application lifespan management"""
    startup_time = time.time()

    try:
        logger.info("🚀 Starting Netflix-Level Video Production Platform...")

        # Phase 1: System validation
        validation_result = await startup_validator.validate_system_startup()
        if not validation_result["is_valid"]:
            logger.error(f"System validation failed: {validation_result['errors']}")
            raise RuntimeError("System validation failed")

        # Phase 2: Initialize core services
        await initialize_core_services()

        # Phase 3: Start monitoring systems
        await start_monitoring_systems()

        # Phase 4: Register health checks
        await register_health_checks()

        startup_duration = time.time() - startup_time
        logger.info(f"✅ Platform started successfully in {startup_duration:.2f}s")

        # Store startup metrics
        await metrics_collector.record_startup_metrics({
            "startup_time": startup_duration,
            "services_initialized": 5,
            "health_checks_registered": 4,
            "status": "success"
        })

        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        await crash_recovery_manager.handle_startup_failure(e)
        raise

    finally:
        # Graceful shutdown
        logger.info("🔄 Initiating graceful shutdown...")
        await graceful_shutdown()
        logger.info("✅ Shutdown complete")


async def initialize_core_services():
    """Initialize all core services with dependency injection"""
    global video_service, ai_service, storage_service, ai_analyzer, enterprise_manager

    try:
        # Initialize services in dependency order
        storage_service = NetflixStorageService()
        await storage_service.initialize()

        video_service = NetflixLevelVideoService()
        await video_service.startup()

        ai_analyzer = NetflixLevelAIAnalyzer()
        await ai_analyzer.enterprise_warm_up()

        ai_service = AIProductionEngine()
        await ai_service.startup()

        enterprise_manager = EnterpriseManager()
        await enterprise_manager.enterprise_warm_up()

        logger.info("✅ All core services initialized successfully")

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        await crash_recovery_manager.handle_service_failure("initialization", e)
        raise


async def start_monitoring_systems():
    """Start comprehensive monitoring systems"""
    try:
        # Start health monitoring
        await health_monitor.start_monitoring()

        # Start recovery system
        await recovery_system.start_recovery_monitoring()

        # Start performance monitoring
        await performance_monitor.start_monitoring()

        logger.info("✅ Monitoring systems started")

    except Exception as e:
        logger.error(f"Monitoring system startup failed: {e}")
        raise


async def register_health_checks():
    """Register comprehensive health check endpoints"""
    health_checks = {
        "video_service": video_service.health_check,
        "ai_service": ai_service.health_check,
        "storage_service": storage_service.health_check,
        "ai_analyzer": ai_analyzer.health_check,
        "enterprise_manager": enterprise_manager.health_check
    }

    for service_name, health_check in health_checks.items():
        await health_monitor.register_service_health_check(service_name, health_check)

    logger.info("✅ Health checks registered for all services")


async def graceful_shutdown():
    """Perform graceful shutdown of all services"""
    shutdown_tasks = []

    # Shutdown services in reverse dependency order
    if enterprise_manager:
        shutdown_tasks.append(enterprise_manager.shutdown())

    if ai_service:
        shutdown_tasks.append(ai_service.shutdown())

    if ai_analyzer:
        shutdown_tasks.append(ai_analyzer.shutdown())

    if video_service:
        shutdown_tasks.append(video_service.shutdown())

    if storage_service:
        shutdown_tasks.append(storage_service.shutdown())

    # Stop monitoring systems
    shutdown_tasks.extend([
        health_monitor.stop_monitoring(),
        recovery_system.stop_recovery_monitoring(),
        performance_monitor.stop_monitoring()
    ])

    # Execute all shutdowns concurrently
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)


# Create FastAPI application with lifespan
app = FastAPI(
    title="Netflix-Level Video Production Platform",
    description="Enterprise-grade video editing and publishing platform",
    version="10.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware in correct order (last added = first executed)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(ErrorHandlerMiddleware)

# Include routers with proper prefixes
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(ai_production.router, prefix="/api/v1/ai", tags=["AI Production"])
app.include_router(enterprise.router, prefix="/api/v1/enterprise", tags=["Enterprise"])
app.include_router(storage.router, prefix="/api/v1/storage", tags=["Storage"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
app.include_router(health_endpoints.router, prefix="/api/v1", tags=["Health"])

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with system status"""
    try:
        health_status = await health_monitor.get_overall_health()
        system_metrics = await metrics_collector.get_current_metrics()

        return {
            "application_name": "Netflix-Level Video Production Platform",
            "version": "10.0.0",
            "status": "operational",
            "health": health_status,
            "metrics": system_metrics,
            "features": [
                "AI-Powered Video Analysis",
                "Real-time Collaboration",
                "Enterprise Security",
                "Auto-scaling Infrastructure",
                "Netflix-Grade Performance"
            ],
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "docs": "/docs",
                "api": "/api/v1/"
            }
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        raise HTTPException(status_code=500, detail="System error")


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_report = await health_monitor.perform_comprehensive_health_check()

        status_code = 200 if health_report["status"] == "healthy" else 503

        return JSONResponse(
            status_code=status_code,
            content=health_report
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """System metrics endpoint for monitoring"""
    try:
        metrics = await metrics_collector.get_comprehensive_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/status", response_model=Dict[str, Any])
async def system_status():
    """Detailed system status for operations"""
    try:
        return {
            "platform": "Netflix-Level Video Production",
            "version": "10.0.0",
            "environment": "production",
            "uptime": await metrics_collector.get_uptime(),
            "services": await health_monitor.get_service_statuses(),
            "performance": await performance_monitor.get_current_performance(),
            "last_updated": time.time()
        }
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Status unavailable")


# Signal handlers for graceful shutdown
def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    setup_signal_handlers()

    settings = get_settings()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        log_level="info",
        access_log=True,
        workers=1,
        loop="uvloop",
        http="httptools",
        reload=False  # Disable in production
    )