
"""
Netflix-Grade Video Production Platform v12.0
Enterprise-level application package initialization
"""

import logging
import sys
import os

# Add the app directory to Python path for proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    # Import main application
    from .main import app
    
    logger.info("✅ Netflix-Grade Video Production Platform initialized successfully")
    
except Exception as e:
    logger.error(f"❌ Application initialization failed: {e}")
    raise

__version__ = "12.0.0"
__app_name__ = "Netflix-Grade Video Production Platform"

# Export the main app
__all__ = ["app"]
