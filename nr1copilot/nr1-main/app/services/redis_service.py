"""
Redis Service Layer (Refactored)

- Provides a centralized, auditable, and testable interface for Redis operations.
- Designed for world-class auditability, reliability, and maintainability (Stripe/Netflix standards).
- All Redis access should go through this service for compliance and observability.
- Implements audit logging and custom exceptions for compliance.
"""

import logging
from typing import Optional, Any
import redis
from redis import Redis
from app.config import settings

logger = logging.getLogger("redis_service")

class RedisServiceError(Exception):
    """Custom exception for RedisService errors."""
    pass

_redis_conn: Optional[Redis] = None

def get_redis_conn() -> Redis:
    """
    Get a Redis connection instance. Initializes if not already connected.

    Returns:
        Redis: Redis connection instance.

    Raises:
        RedisServiceError: If Redis connection fails.
    """
    global _redis_conn
    if _redis_conn is None:
        try:
            _redis_conn = redis.Redis.from_url(settings.REDIS_URL)
            _redis_conn.ping()
            logger.info("Connected to Redis successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise RedisServiceError(f"Failed to connect to Redis: {e}")
    return _redis_conn

def set_value(key: str, value: str, ex: int = 3600) -> bool:
    """
    Set a value in Redis with expiration.

    Args:
        key (str): Redis key.
        value (str): Value to set.
        ex (int): Expiration in seconds.

    Returns:
        bool: True if set, False otherwise.

    Raises:
        RedisServiceError: If operation fails.
    """
    try:
        conn = get_redis_conn()
        result = conn.set(key, value, ex=ex)
        logger.info(f"Set Redis key: {key}")
        return result
    except Exception as e:
        logger.error(f"Redis set error: {e}")
        raise RedisServiceError(f"Redis set error: {e}")

def get_value(key: str) -> Optional[str]:
    """
    Get a value from Redis by key.

    Args:
        key (str): Redis key.

    Returns:
        Optional[str]: Value if found, else None.

    Raises:
        RedisServiceError: If operation fails.
    """
    try:
        conn = get_redis_conn()
        value = conn.get(key)
        logger.info(f"Got Redis key: {key}")
        return value.decode() if value else None
    except Exception as e:
        logger.error(f"Redis get error: {e}")
        raise RedisServiceError(f"Redis get error: {e}")

# Test block for service sanity (not for production)
if __name__ == "__main__":
    import sys
    try:
        print(set_value("test_key", "test_value"))
        print(get_value("test_key"))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
