"""
Logging configuration for ViralClip Pro
"""

import logging
import sys
from typing import Any, Dict
from pathlib import Path

def setup_logging():
    """Setup application logging"""

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/viralclip.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    return logging.getLogger(__name__)