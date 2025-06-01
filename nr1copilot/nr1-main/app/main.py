
"""
FastAPI Main Application
Production-ready viral clip generator backend with comprehensive features
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import os
from datetime import datetime

# Import all routes
from app.routes import video, auth, user, analytics, feedback, i18n
from app.config import get_settings
from app.logging_config import setup_logging
from app.utils.health import health_check
from app.db.session import connect_to_mongo, close_mongo_connection

# Setup logging
setup_logging()
logger = logging.getLogger("main")

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Viral Clip Generator API...")
    await connect_to_mongo()
    
    # Create necessary directories
    os.makedirs("videos/original", exist_ok=True)
    os.makedirs("videos/processed", exist_ok=True)
    os.makedirs("videos/thumbnails", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Viral Clip Generator API...")
    await close_mongo_connection()

# Create FastAPI app
app = FastAPI(
    title="Viral Clip Generator API",
    description="Professional video processing API for creating viral clips from YouTube videos",
    version="2.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    
    response = await call_next(request)
    
    process_time = (datetime.utcnow() - start_time).total_seconds()
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Include all routers
app.include_router(video.router, prefix="/api/v1/video", tags=["Video"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(user.router, prefix="/api/v1/user", tags=["User"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])
app.include_router(i18n.router, prefix="/api/v1/i18n", tags=["Internationalization"])

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint for monitoring"""
    return await health_check()

# Serve static files
app.mount("/public", StaticFiles(directory="public"), name="public")

# Serve the main HTML file
@app.get("/", response_class=FileResponse)
async def read_root():
    """Serve the main application"""
    return FileResponse("index.html")

@app.get("/favicon.ico", response_class=FileResponse)
async def favicon():
    """Serve favicon"""
    return FileResponse("favicon.ico")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception on {request.url.path}: {exc}")
    return HTTPException(
        status_code=500,
        detail="Internal server error. Please try again later."
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        reload=settings.ENVIRONMENT == "development"
    )
