"""
Feedback Controller: Stateless, auditable interface for feedback endpoints.

- Delegates all business logic and validation to the service layer.
- Ensures endpoints are thin, testable, and maintainable.
- All actions are logged for audit/compliance (see logger usage).
- Designed for Stripe/Netflix-level auditability and clarity.

Compliance Note:
    All feedback events are logged for audit/compliance. See logger usage.
    TODO: Integrate with centralized logging and security event monitoring.
"""

from fastapi import HTTPException
from typing import List
from app.schemas import FeedbackIn, FeedbackOut
from ..services.feedback_service import (
    submit_feedback_service,
    get_feedback_service,
    FeedbackError,
)
import logging

logger = logging.getLogger("feedback_audit")

def submit_feedback(feedback: FeedbackIn) -> FeedbackOut:
    """
    Submit user feedback. Raises HTTPException on error.

    Args:
        feedback (FeedbackIn): Feedback data from request.
    Returns:
        FeedbackOut: The created feedback object.
    Raises:
        HTTPException: On validation or service error.
    """
    try:
        result = submit_feedback_service(feedback)
        logger.info(f"AUDIT: Feedback submitted: {result}")
        return result
    except FeedbackError as e:
        logger.error(f"AUDIT: Feedback submission error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

def get_feedback() -> List[FeedbackOut]:
    """
    Retrieve all feedback entries. Raises HTTPException on error.

    Returns:
        List[FeedbackOut]: List of all feedback objects.
    Raises:
        HTTPException: On service error.
    """
    try:
        result = get_feedback_service()
        logger.info(f"AUDIT: Feedback retrieved: count={len(result)}")
        return result
    except FeedbackError as e:
        logger.error(f"AUDIT: Feedback retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Minimal test/assertion to confirm endpoint signature compiles
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    # Test POST
    resp = client.post("/api/v1/feedback", json={"message": "Test feedback"})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data and "message" in data
    # Test GET
    resp2 = client.get("/api/v1/feedback")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert isinstance(data2, list)
