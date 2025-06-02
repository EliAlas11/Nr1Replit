"""
Lightweight logging configuration optimized for deployment
"""

import logging
import sys
from datetime import datetime

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup optimized logging for deployment"""

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Create application logger
    logger = logging.getLogger("viralclip")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    logger.info("Logging configured for deployment")
    return logger

# Export function
__all__ = ['setup_logging']