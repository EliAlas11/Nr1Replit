import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_video_validate():
    data = {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    resp = client.post("/api/v1/videos/validate", json=data)
    assert resp.status_code == 200
    assert resp.json()["valid"] is True

def test_video_info():
    resp = client.get("/api/v1/videos/info/sampleid")
    assert resp.status_code == 200
    assert "title" in resp.json()
