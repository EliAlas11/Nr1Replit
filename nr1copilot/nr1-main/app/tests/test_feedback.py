import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_submit_and_get_feedback():
    feedback_data = {"message": "Great app!", "email": "user@example.com"}
    resp = client.post("/api/v1/feedback/", json=feedback_data)
    assert resp.status_code == 200
    assert resp.json()["message"] == "Feedback submitted"

    resp = client.get("/api/v1/feedback/")
    assert resp.status_code == 200
    assert any(f["message"] == "Great app!" for f in resp.json()["feedback"])
