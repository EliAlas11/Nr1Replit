"""
Runtime utility functions for environment, version, and timestamp.
"""

import os
import datetime


def get_env() -> str:
    """Get the current environment (default: production)."""
    return os.getenv("ENV", "production")


def get_version() -> str:
    """Get the current application version (default: 1.0.0)."""
    # TODO: Read from a version file or env
    return "1.0.0"


def get_timestamp() -> str:
    """Get the current UTC timestamp as ISO8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
