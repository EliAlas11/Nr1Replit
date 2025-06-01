"""
Production-grade FastAPI application for Viral Clip Generator
World-class implementation with comprehensive error handling, logging, and security
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from .config import get_settings
from .logging_config import setup_logging
from .routes import (
    analytics,
    auth,
    feedback,
    i18n,
    user,
    video
)
from .utils.health import health_check, dependencies_check

# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting Viral Clip Generator API...")

    # Startup checks
    settings = get_settings()
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    # Verify critical environment variables
    critical_vars = ["MONGODB_URI", "REDIS_URL", "JWT_SECRET"]
    missing_vars = [var for var in critical_vars if not os.getenv(var)]
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")

    # Create necessary directories
    os.makedirs("videos", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)

    yield

    logger.info("👋 Shutting down Viral Clip Generator API...")

# Initialize FastAPI app
app = FastAPI(
    title="Viral Clip Generator API",
    description="Professional video processing API for creating viral clips from long-form content",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Get settings
settings = get_settings()

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("public"):
    app.mount("/public", StaticFiles(directory="public"), name="public")

# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with structured responses"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.error(f"Validation error: {exc.errors()} - {request.url}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "message": "Validation error",
            "details": exc.errors(),
            "status_code": 422
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)} - {request.url}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500
        }
    )

# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_endpoint():
    """Basic health check endpoint"""
    return health_check()

@app.get("/health/dependencies", tags=["Health"])
async def dependencies_endpoint():
    """Check external dependencies status"""
    return dependencies_check()

# Root endpoint
@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Serve the main application page"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <head><title>Viral Clip Generator API</title></head>
                <body>
                    <h1>🎬 Viral Clip Generator API</h1>
                    <p>Professional video processing API is running!</p>
                    <ul>
                        <li><a href="/docs">API Documentation (Swagger)</a></li>
                        <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                        <li><a href="/health">Health Check</a></li>
                    </ul>
                </body>
            </html>
            """,
            status_code=200
        )

# Include API routes
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])
app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
app.include_router(i18n.router, prefix="/api/v1", tags=["Internationalization"])
app.include_router(user.router, prefix="/api/v1", tags=["Users"])
app.include_router(video.router, prefix="/api/v1", tags=["Videos"])

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )