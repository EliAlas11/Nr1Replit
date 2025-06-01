
"""
ViralClip Pro Configuration
Netflix-level production configuration with comprehensive settings
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    """Application configuration with validation"""
    
    # Application settings
    APP_NAME: str = "ViralClip Pro"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=5000, env="PORT")
    RELOAD: bool = Field(default=True, env="RELOAD")
    
    # File upload settings
    MAX_FILE_SIZE: int = Field(default=2 * 1024 * 1024 * 1024, env="MAX_FILE_SIZE")  # 2GB
    MAX_VIDEO_DURATION: int = Field(default=3600, env="MAX_VIDEO_DURATION")  # 1 hour
    MIN_CLIP_DURATION: int = Field(default=5, env="MIN_CLIP_DURATION")  # 5 seconds
    MAX_CLIP_DURATION: int = Field(default=300, env="MAX_CLIP_DURATION")  # 5 minutes
    
    # Allowed video formats
    ALLOWED_VIDEO_FORMATS: List[str] = Field(
        default=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"],
        env="ALLOWED_VIDEO_FORMATS"
    )
    
    # Directory paths
    BASE_PATH: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    UPLOAD_PATH: str = Field(default="uploads", env="UPLOAD_PATH")
    OUTPUT_PATH: str = Field(default="output", env="OUTPUT_PATH")
    TEMP_PATH: str = Field(default="temp", env="TEMP_PATH")
    CACHE_PATH: str = Field(default="cache", env="CACHE_PATH")
    LOGS_PATH: str = Field(default="logs", env="LOGS_PATH")
    
    # WebSocket settings
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")
    WS_RECONNECT_DELAY: int = Field(default=5, env="WS_RECONNECT_DELAY")
    WS_MAX_CONNECTIONS: int = Field(default=1000, env="WS_MAX_CONNECTIONS")
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    UPLOAD_RATE_LIMIT: int = Field(default=10, env="UPLOAD_RATE_LIMIT")  # per 5 minutes
    PROCESSING_RATE_LIMIT: int = Field(default=20, env="PROCESSING_RATE_LIMIT")  # per hour
    
    # Cache settings
    CACHE_ENABLED: bool = Field(default=True, env="CACHE_ENABLED")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    CACHE_MAX_SIZE: int = Field(default=1000, env="CACHE_MAX_SIZE")
    
    # Security settings
    SECURITY_ENABLED: bool = Field(default=True, env="SECURITY_ENABLED")
    FILE_SIGNATURE_CHECK: bool = Field(default=True, env="FILE_SIGNATURE_CHECK")
    CONTENT_POLICY_CHECK: bool = Field(default=True, env="CONTENT_POLICY_CHECK")
    
    # Performance settings
    WORKER_PROCESSES: int = Field(default=1, env="WORKER_PROCESSES")
    WORKER_THREADS: int = Field(default=4, env="WORKER_THREADS")
    PROCESSING_TIMEOUT: int = Field(default=1800, env="PROCESSING_TIMEOUT")  # 30 minutes
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="structured", env="LOG_FORMAT")
    LOG_FILE_ENABLED: bool = Field(default=True, env="LOG_FILE_ENABLED")
    LOG_ROTATION: str = Field(default="1 day", env="LOG_ROTATION")
    
    # Health check settings
    HEALTH_CHECK_ENABLED: bool = Field(default=True, env="HEALTH_CHECK_ENABLED")
    HEALTH_CHECK_INTERVAL: int = Field(default=60, env="HEALTH_CHECK_INTERVAL")
    
    # Video processing settings
    DEFAULT_VIDEO_QUALITY: str = Field(default="high", env="DEFAULT_VIDEO_QUALITY")
    FFMPEG_ENABLED: bool = Field(default=False, env="FFMPEG_ENABLED")
    GPU_ACCELERATION: bool = Field(default=False, env="GPU_ACCELERATION")
    
    # AI settings
    AI_ENABLED: bool = Field(default=True, env="AI_ENABLED")
    AI_MODEL_PATH: Optional[str] = Field(default=None, env="AI_MODEL_PATH")
    AI_CONFIDENCE_THRESHOLD: float = Field(default=0.8, env="AI_CONFIDENCE_THRESHOLD")
    
    # External services
    YOUTUBE_API_KEY: Optional[str] = Field(default=None, env="YOUTUBE_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Database settings (for future use)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # CORS settings
    CORS_ENABLED: bool = Field(default=True, env="CORS_ENABLED")
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Monitoring settings
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    ANALYTICS_ENABLED: bool = Field(default=False, env="ANALYTICS_ENABLED")
    
    @validator('UPLOAD_PATH', 'OUTPUT_PATH', 'TEMP_PATH', 'CACHE_PATH', 'LOGS_PATH')
    def validate_paths(cls, v, values):
        """Ensure paths are absolute"""
        if not os.path.isabs(v):
            base_path = values.get('BASE_PATH', Path(__file__).parent.parent)
            return str(base_path / v)
        return v
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    @validator('DEFAULT_VIDEO_QUALITY')
    def validate_video_quality(cls, v):
        """Validate video quality setting"""
        valid_qualities = ['draft', 'standard', 'high', 'premium']
        if v not in valid_qualities:
            raise ValueError(f'Video quality must be one of: {valid_qualities}')
        return v
    
    def create_directories(self):
        """Create required directories if they don't exist"""
        directories = [
            self.UPLOAD_PATH,
            self.OUTPUT_PATH,
            self.TEMP_PATH,
            self.CACHE_PATH,
            self.LOGS_PATH
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.create_directories()
    return _settings

# Export settings for convenience
settings = get_settings()
