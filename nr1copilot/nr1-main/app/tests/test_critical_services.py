
"""
Comprehensive test suite for critical service logic
Tests all essential services with mocking and fallbacks
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.video_service import NetflixLevelVideoService
from app.services.ai_production_engine import AIProductionEngine, ModelType
from app.services.storage_service import NetflixStorageService
from app.services.ai_analyzer import NetflixLevelAIAnalyzer
from app.utils.fallbacks import FallbackManager


class TestCriticalServiceLogic:
    """Test critical service implementations"""
    
    @pytest.fixture
    async def video_service(self):
        """Video service fixture"""
        service = NetflixLevelVideoService()
        await service.startup()
        yield service
        await service.shutdown()
    
    @pytest.fixture
    async def ai_engine(self):
        """AI production engine fixture"""
        engine = AIProductionEngine()
        await engine.startup()
        yield engine
        await engine.shutdown()
    
    @pytest.fixture
    async def storage_service(self):
        """Storage service fixture"""
        service = NetflixStorageService()
        await service.initialize()
        yield service
        await service.shutdown()
    
    @pytest.fixture
    async def ai_analyzer(self):
        """AI analyzer fixture"""
        analyzer = NetflixLevelAIAnalyzer()
        await analyzer.enterprise_warm_up()
        yield analyzer
        await analyzer.graceful_shutdown()
    
    @pytest.fixture
    async def fallback_manager(self):
        """Fallback manager fixture"""
        manager = FallbackManager()
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_video_service_complete_workflow(self, video_service):
        """Test complete video service workflow"""
        
        # Test session creation
        user_info = {"user_id": "test_user", "tier": "pro"}
        client_info = {"user_agent": "Test Agent"}
        
        session_result = await video_service.create_upload_session(
            upload_id="test_upload_001",
            filename="test_video.mp4",
            file_size=5000000,  # 5MB
            total_chunks=5,
            user_info=user_info,
            client_info=client_info
        )
        
        assert "session_id" in session_result
        assert "upload_url" in session_result
        session_id = session_result["session_id"]
        
        # Test chunk upload
        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=b"test chunk data" * 100)
        
        chunk_result = await video_service.process_chunk(
            file=mock_file,
            upload_id="test_upload_001",
            chunk_index=0,
            total_chunks=5,
            chunk_hash="abc123"
        )
        
        assert chunk_result["status"] == "success"
        assert chunk_result["chunk_index"] == 0
        
        # Test status retrieval
        status = await video_service.get_upload_status(session_id)
        assert "progress_percentage" in status
        assert status["chunks_uploaded"] == 1
    
    @pytest.mark.asyncio
    async def test_ai_engine_inference_workflow(self, ai_engine):
        """Test AI engine inference workflow"""
        
        # Test viral prediction
        inputs = {
            "content_type": "video",
            "duration": 30,
            "resolution": "1080p",
            "audio_present": True
        }
        
        result = await ai_engine.inference(
            ModelType.VIRAL_PREDICTOR,
            inputs
        )
        
        assert result.outputs is not None
        assert "viral_score" in result.outputs
        assert 0 <= result.outputs["viral_score"] <= 100
        assert result.confidence > 0
        
        # Test sentiment analysis
        sentiment_inputs = {"text": "This is an amazing video!"}
        
        sentiment_result = await ai_engine.inference(
            ModelType.SENTIMENT_ANALYZER,
            sentiment_inputs
        )
        
        assert "sentiment" in sentiment_result.outputs
        assert sentiment_result.outputs["sentiment"] in ["positive", "neutral", "negative"]
        
        # Test caching
        cached_result = await ai_engine.inference(
            ModelType.VIRAL_PREDICTOR,
            inputs,
            use_cache=True
        )
        
        assert cached_result.cache_hit is True
    
    @pytest.mark.asyncio
    async def test_storage_service_operations(self, storage_service):
        """Test storage service operations"""
        
        # Create test file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(b"test video content" * 1000)
            temp_file_path = temp_file.name
        
        try:
            # Test file storage
            storage_obj = await storage_service.store_file(
                file_path=temp_file_path,
                key="test/video.mp4",
                content_type="video/mp4",
                metadata={"resolution": "1080p"},
                expires_in=3600
            )
            
            assert storage_obj.key == "test/video.mp4"
            assert storage_obj.content_type == "video/mp4"
            assert storage_obj.size > 0
            
            # Test file retrieval
            retrieved_obj = await storage_service.retrieve_file("test/video.mp4")
            assert retrieved_obj is not None
            assert retrieved_obj.key == storage_obj.key
            
            # Test signed URL generation
            signed_url = await storage_service.generate_signed_url(
                "test/video.mp4",
                expires_in=3600
            )
            assert signed_url is not None
            assert "signature" in signed_url
            
            # Test file deletion
            delete_result = await storage_service.delete_file("test/video.mp4")
            assert delete_result is True
            
        finally:
            os.unlink(temp_file_path)
    
    @pytest.mark.asyncio
    async def test_ai_analyzer_comprehensive_analysis(self, ai_analyzer):
        """Test AI analyzer comprehensive analysis"""
        
        # Mock upload file
        mock_file = MagicMock()
        mock_file.filename = "test_video.mp4"
        mock_file.size = 5000000
        
        # Test quick assessment
        quick_result = await ai_analyzer.quick_viral_assessment(
            file=mock_file,
            session_id="test_session_001"
        )
        
        assert "viral_score" in quick_result
        assert "confidence" in quick_result
        assert "insights" in quick_result
        assert quick_result["analysis_type"] == "quick_assessment"
        
        # Test comprehensive analysis
        comprehensive_result = await ai_analyzer.analyze_video_comprehensive(
            file=mock_file,
            session_id="test_session_002",
            enable_realtime=True
        )
        
        assert comprehensive_result.viral_score >= 0
        assert comprehensive_result.confidence > 0
        assert len(comprehensive_result.insights) > 0
        assert len(comprehensive_result.trending_factors) > 0
        assert len(comprehensive_result.platform_recommendations) > 0
        
        # Test trending factors
        trending_factors = await ai_analyzer.get_trending_viral_factors()
        assert "factors" in trending_factors
        assert "platform_trends" in trending_factors
        assert "confidence" in trending_factors
    
    @pytest.mark.asyncio
    async def test_fallback_manager_failure_handling(self, fallback_manager):
        """Test fallback manager failure handling"""
        
        # Test service failure handling
        result = await fallback_manager.handle_service_failure(
            service="video",
            operation="analyze",
            error=Exception("Service temporarily down"),
            context={"file_type": "mp4"}
        )
        
        assert "viral_score" in result
        assert result["degraded_mode"] is True
        
        # Test circuit breaker
        for i in range(6):  # Exceed threshold
            await fallback_manager.handle_service_failure(
                service="ai",
                operation="inference",
                error=Exception(f"Failure {i}"),
                context={}
            )
        
        assert fallback_manager.is_circuit_open("ai") is True
        
        # Test health check
        health = await fallback_manager.health_check()
        assert "degraded_mode" in health
        assert "circuit_breakers" in health
    
    @pytest.mark.asyncio
    async def test_service_integration_with_fallbacks(self, video_service, fallback_manager):
        """Test service integration with fallback handling"""
        
        # Simulate AI service failure
        with patch('app.services.ai_analyzer.NetflixLevelAIAnalyzer.analyze_video_comprehensive') as mock_analyze:
            mock_analyze.side_effect = Exception("AI service down")
            
            # Video service should still work with fallbacks
            user_info = {"user_id": "test_user"}
            client_info = {"user_agent": "Test Agent"}
            
            session_result = await video_service.create_upload_session(
                upload_id="test_fallback",
                filename="test.mp4",
                file_size=1000000,
                total_chunks=1,
                user_info=user_info,
                client_info=client_info
            )
            
            assert "session_id" in session_result
            # Service should continue working despite AI failure
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, ai_engine):
        """Test service performance under concurrent load"""
        
        async def run_inference():
            return await ai_engine.inference(
                ModelType.QUALITY_SCORER,
                {"test": "data"}
            )
        
        # Run 10 concurrent inferences
        tasks = [run_inference() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 8  # Allow for some failures under load
        
        # Check performance metrics
        metrics = ai_engine.get_performance_metrics()
        assert metrics["total_requests"] >= 10
    
    @pytest.mark.asyncio
    async def test_data_validation_and_sanitization(self, video_service):
        """Test data validation and sanitization"""
        
        # Test invalid file size
        with pytest.raises(Exception):  # Should raise HTTPException
            await video_service.create_upload_session(
                upload_id="invalid_test",
                filename="test.mp4",
                file_size=-1,  # Invalid size
                total_chunks=1,
                user_info={"user_id": "test"},
                client_info={}
            )
        
        # Test invalid filename
        with pytest.raises(Exception):
            await video_service.create_upload_session(
                upload_id="invalid_test_2",
                filename="",  # Empty filename
                file_size=1000000,
                total_chunks=1,
                user_info={"user_id": "test"},
                client_info={}
            )
    
    @pytest.mark.asyncio
    async def test_memory_and_resource_management(self, ai_analyzer):
        """Test memory and resource management"""
        
        # Get initial metrics
        initial_metrics = await ai_analyzer.get_performance_metrics()
        
        # Run multiple analyses
        mock_file = MagicMock()
        mock_file.filename = "test.mp4"
        mock_file.size = 1000000
        
        for i in range(5):
            await ai_analyzer.quick_viral_assessment(
                file=mock_file,
                session_id=f"memory_test_{i}"
            )
        
        # Check that memory usage is reasonable
        final_metrics = await ai_analyzer.get_performance_metrics()
        assert final_metrics["memory_usage_mb"] < 1000  # Should be under 1GB
        
        # Cleanup expired sessions
        await ai_analyzer.cleanup_expired_sessions()
        
        # Cache should be managed
        assert final_metrics["cache_size"] <= ai_analyzer.max_cache_size


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
