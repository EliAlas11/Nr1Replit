import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_translations():
    resp = client.get("/api/v1/i18n/translations")
    assert resp.status_code == 200
    assert "translations" in resp.json()

def test_set_language():
    data = {"language": "en"}
    resp = client.post("/api/v1/i18n/set-language", json=data)
    assert resp.status_code == 200
    assert resp.json()["message"].startswith("Language set to")
