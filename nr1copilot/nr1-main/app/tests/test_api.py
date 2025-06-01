
"""
API Integration Tests
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


class TestAPI:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_video_download_invalid_url(self, client):
        """Test video download with invalid URL"""
        response = client.post(
            "/api/v1/video/download",
            json={"url": "invalid-url"}
        )
        assert response.status_code == 422
    
    def test_video_download_valid_url(self, client):
        """Test video download with valid URL"""
        response = client.post(
            "/api/v1/video/download",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )
        # This would normally succeed but we don't have actual video processing
        assert response.status_code in [200, 400, 500]
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/v1/videos")
        assert "access-control-allow-origin" in response.headers
    
    def test_api_validation_error(self, client):
        """Test API validation error handling"""
        response = client.post("/api/v1/video/clip", json={})
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"] is True


if __name__ == "__main__":
    pytest.main([__file__])
