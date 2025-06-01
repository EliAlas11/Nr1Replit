"""
Analytics controller for analytics endpoints.
- Handles analytics submission and retrieval using Pydantic models.
- Delegates business logic to the analytics service layer.
"""

from fastapi import HTTPException
from typing import List
from app.schemas import AnalyticsIn, AnalyticsOut
from ..services.analytics_service import submit_analytics_service, get_analytics_service, AnalyticsError

def submit_analytics(analytics: AnalyticsIn) -> AnalyticsOut:
    """
    Submit an analytics event. Raises HTTPException on error.
    """
    try:
        return submit_analytics_service(analytics)
    except AnalyticsError as e:
        raise HTTPException(status_code=400, detail=str(e))

def get_analytics() -> List[AnalyticsOut]:
    """
    Retrieve all analytics events.
    """
    return get_analytics_service()

# Minimal test/assertion to confirm endpoint signature compiles
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/api/v1/analytics")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
