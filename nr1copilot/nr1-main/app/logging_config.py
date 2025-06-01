"""
Logging configuration for Viral Clip Generator backend.
Sets up colorized and file logging based on environment.
"""

import logging
import colorlog
import os
from .config import settings

def setup_logging() -> None:
    """Configure logging for the application (console and file if set)."""
    log_level = logging.INFO if settings.ENV == "production" else logging.DEBUG
    log_format = "%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s"
    colorlog.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )
    # Reduce noise from uvicorn access logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # Add file logging if LOG_FILE_PATH is set
    if settings.LOG_FILE_PATH:
        file_handler = logging.FileHandler(settings.LOG_FILE_PATH)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
