"""
Feedback Service Layer (Production, Async, SQLAlchemy, Render.com Ready)

- All DB access is async and uses SQLAlchemy async ORM.
- No dev-only logic; all file paths/envs are from environment variables.
- Production-safe, deploy-ready, PostgreSQL assumed.
- Only exports functions used by controllers.
"""

import logging
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError
from app.models.feedback import Feedback
from app.schemas import FeedbackIn, FeedbackOut
from app.db.session import get_async_session

logger = logging.getLogger("feedback_service")

class FeedbackError(Exception):
    """Custom exception for feedback service errors."""
    pass

def _feedback_to_out(fb: Feedback) -> FeedbackOut:
    """Convert Feedback ORM to FeedbackOut schema."""
    return FeedbackOut(
        id=str(fb.id),
        message=str(getattr(fb, 'message', '')),
        email=str(getattr(fb, 'email', '')) if getattr(fb, 'email', None) else None,
        created_at=fb.created_at.isoformat() if getattr(fb, 'created_at', None) else None
    )

async def submit_feedback_service(feedback: FeedbackIn) -> FeedbackOut:
    """
    Submit user feedback (async, production-safe).

    Args:
        feedback (FeedbackIn): Feedback data.

    Returns:
        FeedbackOut: The created feedback object.

    Raises:
        FeedbackError: If feedback is invalid or cannot be saved.
    """
    if not feedback.message or len(feedback.message.strip()) == 0:
        logger.warning("Feedback message cannot be empty.")
        raise FeedbackError("Feedback message cannot be empty.")
    session_gen = get_async_session()
    session = await session_gen.__anext__()
    try:
        fb = Feedback(message=feedback.message, email=feedback.email)
        session.add(fb)
        await session.commit()
        await session.refresh(fb)
        logger.info(f"AUDIT: Feedback submitted: {fb.id}")
        return _feedback_to_out(fb)
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"AUDIT: Feedback DB error: {e}")
        raise FeedbackError("Database error while submitting feedback.")
    finally:
        await session_gen.aclose()

async def get_feedback_service() -> List[FeedbackOut]:
    """
    Retrieve all feedback entries (async, production-safe).

    Returns:
        List[FeedbackOut]: List of all feedback objects.

    Raises:
        FeedbackError: On DB error.
    """
    session_gen = get_async_session()
    session = await session_gen.__anext__()
    try:
        result = await session.execute(select(Feedback).order_by(Feedback.created_at.desc()))
        feedbacks = result.scalars().all()
        logger.info(f"AUDIT: Feedback retrieved: count={len(feedbacks)}")
        return [_feedback_to_out(fb) for fb in feedbacks]
    except SQLAlchemyError as e:
        logger.error(f"AUDIT: Feedback DB error: {e}")
        raise FeedbackError("Database error while retrieving feedback.")
    finally:
        await session_gen.aclose()

# Export only the functions used by controllers
__all__ = ["submit_feedback_service", "get_feedback_service", "FeedbackError"]
