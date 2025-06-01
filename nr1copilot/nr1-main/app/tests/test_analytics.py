import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.main import app
import pytest
from fastapi.testclient import TestClient

client = TestClient(app)

def test_submit_and_get_analytics():
    analytics_data = {"event": "page_view", "user_id": "1", "metadata": {"page": "home"}}
    resp = client.post("/api/v1/analytics/", json=analytics_data)
    assert resp.status_code == 200
    assert resp.json()["message"] == "Analytics submitted"

    resp = client.get("/api/v1/analytics/")
    assert resp.status_code == 200
    assert any(a["event"] == "page_view" for a in resp.json()["analytics"])
