"""
Main FastAPI application entrypoint for Viral Clip Generator.
- Configures CORS, static files, logging, and error handling.
- Mounts all API routers and health endpoints.
"""
import os
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .config import settings
from .logging_config import setup_logging
from .utils.health import health_check, dependencies_check

# Load environment variables from .env if present
load_dotenv()

# Setup logging based on environment
setup_logging()

app = FastAPI(title="Viral Clip Generator", version="1.0.0")

# Configure CORS for frontend/backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
app.mount("/public", StaticFiles(directory=STATIC_DIR), name="public")

# Health check endpoints
@app.get("/health", tags=["Health"])
def health():
    """Basic health check endpoint."""
    return health_check()

@app.get("/health/dependencies", tags=["Health"])
def health_dependencies():
    """Check status of external dependencies (DB, Redis, etc)."""
    return dependencies_check()

# Global error handler for robust API responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler for unhandled exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": str(exc), "path": str(request.url)},
    )

# Root route serves frontend index.html
@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "index.html")
    return FileResponse(index_path)

# Import and mount all API routers
from .routes import video, analytics, feedback, i18n, auth, user
app.include_router(video.router)
app.include_router(analytics.router)
app.include_router(feedback.router)
app.include_router(i18n.router)
app.include_router(auth.router)
app.include_router(user.router)
