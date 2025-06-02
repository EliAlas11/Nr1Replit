
"""
Comprehensive test suite for Netflix-level video service
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.video_service import (
    NetflixLevelVideoService,
    VideoProcessor,
    AIVideoAnalyzer,
    UploadSession,
    UploadStatus
)


class TestVideoProcessor:
    """Test suite for VideoProcessor"""
    
    @pytest.fixture
    def processor(self):
        return VideoProcessor()
    
    @pytest.fixture
    def temp_video_file(self):
        """Create temporary video file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b"fake video content")
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_validate_video_success(self, processor, temp_video_file):
        """Test successful video validation"""
        result = await processor.validate_video(temp_video_file)
        
        assert result["valid"] is True
        assert "metadata" in result
        assert "quality_score" in result
        assert result["quality_score"] >= 0
    
    @pytest.mark.asyncio
    async def test_validate_video_missing_file(self, processor):
        """Test validation with missing file"""
        result = await processor.validate_video("nonexistent.mp4")
        
        assert result["valid"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_validate_video_unsupported_format(self, processor):
        """Test validation with unsupported format"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"not a video")
            temp_file = f.name
        
        try:
            result = await processor.validate_video(temp_file)
            assert result["valid"] is False
            assert "Unsupported format" in result["error"]
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_analyze_video_content(self, processor, temp_video_file):
        """Test video content analysis"""
        result = await processor.analyze_video_content(temp_video_file)
        
        assert "analysis_available" in result
        if result["analysis_available"]:
            assert "brightness" in result
            assert "motion" in result
            assert "quality_score" in result
    
    @pytest.mark.asyncio
    async def test_create_video_clip(self, processor, temp_video_file):
        """Test video clip creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            
            result = await processor.create_video_clip(
                input_path=temp_video_file,
                output_path=output_path,
                start_time=0.0,
                end_time=10.0
            )
            
            assert result["success"] is True
            assert os.path.exists(result["output_path"])
            assert result["duration"] == 10.0
    
    @pytest.mark.asyncio
    async def test_create_video_clip_invalid_time_range(self, processor, temp_video_file):
        """Test clip creation with invalid time range"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            
            result = await processor.create_video_clip(
                input_path=temp_video_file,
                output_path=output_path,
                start_time=10.0,
                end_time=5.0  # Invalid: end < start
            )
            
            assert result["success"] is False
            assert "Invalid time range" in result["error"]


class TestAIVideoAnalyzer:
    """Test suite for AIVideoAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        return AIVideoAnalyzer()
    
    @pytest.fixture
    def temp_video_file(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b"fake video content")
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_warm_up(self, analyzer):
        """Test AI analyzer warm-up"""
        await analyzer.warm_up()
        assert analyzer.models_loaded is True
    
    @pytest.mark.asyncio
    async def test_analyze_clip_segment(self, analyzer, temp_video_file):
        """Test clip segment analysis"""
        result = await analyzer.analyze_clip_segment(
            temp_video_file, 0.0, 10.0, "Test Title", "Test Description"
        )
        
        assert "viral_potential" in result
        assert "confidence" in result
        assert "engagement_score" in result
        assert "sentiment" in result
        assert "platform_suitability" in result
        
        # Check value ranges
        assert 0 <= result["viral_potential"] <= 100
        assert 0 <= result["confidence"] <= 1


class TestUploadSession:
    """Test suite for UploadSession"""
    
    @pytest.fixture
    def upload_session(self):
        return UploadSession(
            id="test_session",
            filename="test_video.mp4",
            file_size=1000000,
            total_chunks=10
        )
    
    def test_upload_session_creation(self, upload_session):
        """Test upload session creation"""
        assert upload_session.id == "test_session"
        assert upload_session.filename == "test_video.mp4"
        assert upload_session.file_size == 1000000
        assert upload_session.total_chunks == 10
        assert upload_session.status == UploadStatus.INITIALIZING
    
    def test_progress_percentage(self, upload_session):
        """Test progress percentage calculation"""
        assert upload_session.progress_percentage == 0.0
        
        # Add some chunks
        upload_session.chunks_uploaded.add(0)
        upload_session.chunks_uploaded.add(1)
        upload_session.chunks_uploaded.add(2)
        
        assert upload_session.progress_percentage == 30.0
    
    def test_is_complete(self, upload_session):
        """Test completion check"""
        assert upload_session.is_complete is False
        
        # Add all chunks
        for i in range(upload_session.total_chunks):
            upload_session.chunks_uploaded.add(i)
        
        assert upload_session.is_complete is True
    
    def test_missing_chunks(self, upload_session):
        """Test missing chunks calculation"""
        upload_session.chunks_uploaded.add(0)
        upload_session.chunks_uploaded.add(2)
        upload_session.chunks_uploaded.add(4)
        
        missing = upload_session.missing_chunks
        expected = [1, 3, 5, 6, 7, 8, 9]
        assert missing == expected
    
    def test_update_progress(self, upload_session):
        """Test progress update functionality"""
        from app.services.video_service import ChunkInfo
        
        # Add chunk info
        chunk_info = ChunkInfo(
            index=0,
            size=100000,
            hash="test_hash",
            uploaded_at=datetime.utcnow(),
            path=Path("test_chunk")
        )
        upload_session.chunks_info[0] = chunk_info
        upload_session.chunks_uploaded.add(0)
        
        upload_session.update_progress()
        
        assert upload_session.bytes_uploaded == 100000
        assert upload_session.upload_speed >= 0
    
    def test_to_dict(self, upload_session):
        """Test dictionary conversion"""
        result = upload_session.to_dict()
        
        required_keys = [
            "id", "filename", "file_size", "total_chunks", 
            "chunks_uploaded", "status", "progress_percentage",
            "bytes_uploaded", "upload_speed", "created_at"
        ]
        
        for key in required_keys:
            assert key in result


class TestNetflixLevelVideoService:
    """Test suite for NetflixLevelVideoService"""
    
    @pytest.fixture
    async def video_service(self):
        service = NetflixLevelVideoService()
        await service.startup()
        yield service
        await service.shutdown()
    
    @pytest.fixture
    def mock_upload_file(self):
        mock_file = MagicMock()
        mock_file.filename = "test_video.mp4"
        mock_file.read = AsyncMock(return_value=b"test chunk data")
        return mock_file
    
    @pytest.mark.asyncio
    async def test_startup_shutdown(self):
        """Test service startup and shutdown"""
        service = NetflixLevelVideoService()
        
        # Test startup
        await service.startup()
        assert service.cleanup_task is not None
        
        # Test shutdown
        await service.shutdown()
        assert service.cleanup_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_create_upload_session(self, video_service):
        """Test upload session creation"""
        user_info = {
            "user_id": "test_user",
            "tier": "free",
            "ip_address": "127.0.0.1"
        }
        client_info = {
            "user_agent": "Test Agent"
        }
        
        result = await video_service.create_upload_session(
            upload_id="test_upload",
            filename="test_video.mp4",
            file_size=1000000,
            total_chunks=10,
            user_info=user_info,
            client_info=client_info
        )
        
        assert "session_id" in result
        assert "upload_url" in result
        assert "chunk_size" in result
        assert result["session_id"] in video_service.active_sessions
    
    @pytest.mark.asyncio
    async def test_create_upload_session_invalid_file_size(self, video_service):
        """Test upload session creation with invalid file size"""
        user_info = {"user_id": "test_user"}
        client_info = {"user_agent": "Test Agent"}
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await video_service.create_upload_session(
                upload_id="test_upload",
                filename="test_video.mp4",
                file_size=-1,  # Invalid size
                total_chunks=10,
                user_info=user_info,
                client_info=client_info
            )
    
    @pytest.mark.asyncio
    async def test_create_upload_session_unsupported_format(self, video_service):
        """Test upload session creation with unsupported format"""
        user_info = {"user_id": "test_user"}
        client_info = {"user_agent": "Test Agent"}
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await video_service.create_upload_session(
                upload_id="test_upload",
                filename="test_video.txt",  # Unsupported format
                file_size=1000000,
                total_chunks=10,
                user_info=user_info,
                client_info=client_info
            )
    
    @pytest.mark.asyncio
    async def test_process_chunk(self, video_service, mock_upload_file):
        """Test chunk processing"""
        # Create session first
        user_info = {"user_id": "test_user"}
        client_info = {"user_agent": "Test Agent"}
        
        session_result = await video_service.create_upload_session(
            upload_id="test_upload",
            filename="test_video.mp4",
            file_size=1000000,
            total_chunks=10,
            user_info=user_info,
            client_info=client_info
        )
        
        # Process chunk
        result = await video_service.process_chunk(
            file=mock_upload_file,
            upload_id="test_upload",
            chunk_index=0,
            total_chunks=10
        )
        
        assert result["status"] == "success"
        assert result["chunk_index"] == 0
        assert result["chunks_uploaded"] == 1
    
    @pytest.mark.asyncio
    async def test_process_chunk_duplicate(self, video_service, mock_upload_file):
        """Test processing duplicate chunk"""
        # Create session and process chunk
        user_info = {"user_id": "test_user"}
        client_info = {"user_agent": "Test Agent"}
        
        await video_service.create_upload_session(
            upload_id="test_upload",
            filename="test_video.mp4",
            file_size=1000000,
            total_chunks=10,
            user_info=user_info,
            client_info=client_info
        )
        
        # Process chunk first time
        result1 = await video_service.process_chunk(
            file=mock_upload_file,
            upload_id="test_upload",
            chunk_index=0,
            total_chunks=10
        )
        
        # Process same chunk again
        result2 = await video_service.process_chunk(
            file=mock_upload_file,
            upload_id="test_upload",
            chunk_index=0,
            total_chunks=10
        )
        
        assert result1["status"] == "success"
        assert result2["status"] == "success"
        assert result2["chunks_uploaded"] == 1  # Should still be 1
    
    @pytest.mark.asyncio
    async def test_get_upload_status(self, video_service):
        """Test getting upload status"""
        # Create session
        user_info = {"user_id": "test_user"}
        client_info = {"user_agent": "Test Agent"}
        
        session_result = await video_service.create_upload_session(
            upload_id="test_upload",
            filename="test_video.mp4",
            file_size=1000000,
            total_chunks=10,
            user_info=user_info,
            client_info=client_info
        )
        
        session_id = session_result["session_id"]
        
        # Get status
        status = await video_service.get_upload_status(session_id)
        
        assert status["id"] == session_id
        assert status["filename"] == "test_video.mp4"
        assert status["status"] == UploadStatus.INITIALIZING.value
    
    @pytest.mark.asyncio
    async def test_cancel_upload(self, video_service):
        """Test upload cancellation"""
        # Create session
        user_info = {"user_id": "test_user"}
        client_info = {"user_agent": "Test Agent"}
        
        session_result = await video_service.create_upload_session(
            upload_id="test_upload",
            filename="test_video.mp4",
            file_size=1000000,
            total_chunks=10,
            user_info=user_info,
            client_info=client_info
        )
        
        session_id = session_result["session_id"]
        
        # Cancel upload
        result = await video_service.cancel_upload(session_id)
        
        assert result["status"] == "cancelled"
        assert session_id not in video_service.active_sessions
    
    @pytest.mark.asyncio
    async def test_get_service_metrics(self, video_service):
        """Test getting service metrics"""
        metrics = await video_service.get_service_metrics()
        
        required_keys = [
            "total_uploads", "successful_uploads", "failed_uploads",
            "active_sessions", "processing_queue_size", "memory_usage"
        ]
        
        for key in required_keys:
            assert key in metrics
    
    @pytest.mark.asyncio
    async def test_enterprise_warm_up(self, video_service):
        """Test enterprise warm-up"""
        # Should complete without error
        await video_service.enterprise_warm_up()
        assert video_service.ai_analyzer.models_loaded is True


# Mock fixtures for testing
@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    with patch('app.services.video_service.settings') as mock:
        mock.upload_path = Path(tempfile.gettempdir())
        mock.temp_path = Path(tempfile.gettempdir())
        mock.output_path = Path(tempfile.gettempdir())
        mock.enable_ai_analysis = True
        mock.upload.chunk_size = 1024 * 1024
        mock.upload.max_file_size = 100 * 1024 * 1024
        mock.upload.supported_formats = ['.mp4', '.avi', '.mov']
        mock.performance.worker_count = 2
        yield mock


@pytest.fixture(autouse=True)
def setup_test_environment(mock_settings):
    """Setup test environment"""
    # Ensure test directories exist
    mock_settings.upload_path.mkdir(exist_ok=True)
    mock_settings.temp_path.mkdir(exist_ok=True)
    mock_settings.output_path.mkdir(exist_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
