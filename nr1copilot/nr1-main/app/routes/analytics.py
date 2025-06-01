from fastapi import APIRouter, HTTPException
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
    return AnalyticsList(analytics=analytics_list)
