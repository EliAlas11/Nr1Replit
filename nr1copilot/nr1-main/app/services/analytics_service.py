"""
Analytics Service Layer (Refactored)

- Handles analytics event submission and retrieval logic.
- All business logic, validation, and error handling for analytics is centralized here.
- Designed for auditability, security, and testability (Stripe/Netflix standards).
- All functions are stateless and side-effect free except for analytics submission.
- Implements audit logging and custom exceptions for compliance.
- TODO: Replace in-memory store with persistent database for production.
"""

import logging
from typing import List
from app.schemas import AnalyticsIn, AnalyticsOut
from datetime import datetime, timezone

logger = logging.getLogger("analytics_service")

class AnalyticsError(Exception):
    """Custom exception for analytics service errors."""
    pass

_fake_analytics_db: List[AnalyticsOut] = []

def submit_analytics_service(analytics: AnalyticsIn) -> AnalyticsOut:
    """
    Submit an analytics event.

    Args:
        analytics (AnalyticsIn): Analytics event data.

    Returns:
        AnalyticsOut: The created analytics object.

    Raises:
        AnalyticsError: If analytics event is invalid or cannot be saved.
    """
    if not analytics.event or len(analytics.event.strip()) == 0:
        logger.warning("Analytics event name cannot be empty.")
        raise AnalyticsError("Analytics event name cannot be empty.")
    analytics_id = str(len(_fake_analytics_db) + 1)
    analytics_obj = AnalyticsOut(
        id=analytics_id,
        event=analytics.event,
        user_id=analytics.user_id,
        metadata=analytics.metadata,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    _fake_analytics_db.append(analytics_obj)
    logger.info(f"AUDIT: Analytics event submitted: {analytics_obj}")
    return analytics_obj

def get_analytics_service() -> List[AnalyticsOut]:
    """
    Retrieve all analytics events.

    Returns:
        List[AnalyticsOut]: List of all analytics event objects.
    """
    logger.info(f"AUDIT: Analytics events retrieved: count={len(_fake_analytics_db)}")
    return _fake_analytics_db

# Test block for service sanity (not for production)
if __name__ == "__main__":
    try:
        test_event = AnalyticsIn(event="test", user_id="u1", metadata={"foo": "bar"})
        out = submit_analytics_service(test_event)
        assert out.event == "test"
        all_events = get_analytics_service()
        assert isinstance(all_events, list)
        print("Analytics service test passed.")
    except Exception as e:
        print(f"Error: {e}")
