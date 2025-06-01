
"""
ViralClip Pro v4.0 - Netflix-Level Application Package
Enterprise-grade video processing platform
"""

__version__ = "4.0.0"
__title__ = "ViralClip Pro"
__description__ = "Netflix-level AI-powered viral video clip generator"
__author__ = "ViralClip Team"

# Package-level imports
from .config import get_settings
from .main import app

__all__ = ["app", "get_settings", "__version__"]
