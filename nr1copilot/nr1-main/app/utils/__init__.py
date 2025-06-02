
"""
Netflix-Grade Utility Package
Core utilities for performance, monitoring, and optimization
"""

import logging

logger = logging.getLogger(__name__)

# Import cache with error handling
try:
    from .cache import cache, CacheManager, NetflixCacheManager
    cache_available = True
except Exception as e:
    logger.warning(f"Cache import failed: {e}")
    cache = None
    CacheManager = None
    NetflixCacheManager = None
    cache_available = False

# Import other utilities with error handling
try:
    from .health import HealthMonitor
    health_available = True
except Exception as e:
    logger.warning(f"Health monitor import failed: {e}")
    HealthMonitor = None
    health_available = False

try:
    from .metrics import MetricsCollector
    metrics_available = True
except Exception as e:
    logger.warning(f"Metrics collector import failed: {e}")
    MetricsCollector = None
    metrics_available = False

try:
    from .performance_monitor import PerformanceMonitor
    performance_available = True
    logger.info("✅ PerformanceMonitor imported successfully")
except Exception as e:
    logger.error(f"Performance monitor import failed: {e}")
    PerformanceMonitor = None
    performance_available = False

# Optional imports
try:
    from .security import SecurityManager
    security_available = True
except ImportError:
    SecurityManager = None
    security_available = False

try:
    from .rate_limiter import RateLimiter
    rate_limiter_available = True
except ImportError:
    RateLimiter = None
    rate_limiter_available = False

# Export all available utilities
__all__ = [
    "cache",
    "CacheManager", 
    "NetflixCacheManager",
    "HealthMonitor",
    "MetricsCollector",
    "PerformanceMonitor",
    "SecurityManager",
    "RateLimiter"
]

# Provide availability flags for dependency checking
COMPONENT_AVAILABILITY = {
    "cache": cache_available,
    "health": health_available, 
    "metrics": metrics_available,
    "performance": performance_available,
    "security": security_available,
    "rate_limiter": rate_limiter_available
}

def get_available_components():
    """Get list of available utility components"""
    return [name for name, available in COMPONENT_AVAILABILITY.items() if available]

async def initialize_async_components():
    """Initialize async components when event loop is available"""
    try:
        if cache and hasattr(cache, 'initialize'):
            await cache.initialize()
            logger.info("✅ Cache async components initialized")
        
        # Initialize other async components as needed
        logger.info("✅ All utility async components initialized")
        
    except Exception as e:
        logger.error(f"Async component initialization failed: {e}")
        raise

logger.info(f"🚀 Netflix utilities loaded: {get_available_components()}")
