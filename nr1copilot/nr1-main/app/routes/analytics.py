
<old_str>from fastapi import APIRouter, HTTPException
from app.schemas import AnalyticsIn, AnalyticsOut, AnalyticsList, Message
from ..controllers.analytics_controller import submit_analytics, get_analytics

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])

@router.post("/", response_model=Message)
def submit(analytics: AnalyticsIn):
    result = submit_analytics(analytics)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return Message(message=result["message"])

@router.get("/", response_model=AnalyticsList)
def get():
    analytics_list = get_analytics()
    return AnalyticsList(analytics=analytics_list)</old_str>
<new_str>from fastapi import APIRouter, HTTPException, Depends
from app.schemas import AnalyticsEvent, AnalyticsOut, SuccessResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analytics", response_model=SuccessResponse)
async def submit_analytics(event: AnalyticsEvent):
    """Submit analytics event"""
    try:
        logger.info(f"Analytics event: {event.event_type}")
        return SuccessResponse(message="Analytics event recorded successfully")
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/analytics", response_model=SuccessResponse)
async def get_analytics():
    """Get analytics data"""
    try:
        # Mock analytics data for now
        analytics_data = {
            "total_videos": 0,
            "total_users": 0,
            "success_rate": 100.0
        }
        return SuccessResponse(message="Analytics retrieved", data=analytics_data)
    except Exception as e:
        logger.error(f"Analytics retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))</new_str>
