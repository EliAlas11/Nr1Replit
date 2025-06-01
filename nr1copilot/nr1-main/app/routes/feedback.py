
<old_str></old_str>
<new_str>"""
Feedback Routes
"""

from fastapi import APIRouter, HTTPException
import logging
from ..schemas import FeedbackRequest, FeedbackOut, SuccessResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/feedback", response_model=SuccessResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback"""
    try:
        logger.info(f"Feedback received: {feedback.subject}")
        return SuccessResponse(
            message="Feedback submitted successfully",
            data={"feedback_id": "feedback_123"}
        )
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=400, detail="Failed to submit feedback")</new_str>
