
"""
Comprehensive tests for Video Service
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from app.services.video_service import VideoService, VideoServiceError


class TestVideoService:
    """Test cases for VideoService"""
    
    @pytest.fixture
    def video_service(self):
        """Create video service instance for testing"""
        return VideoService()
    
    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Create a minimal MP4 file (just headers for testing)
            f.write(b'\x00\x00\x00\x20ftypmp42')
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_analyze_highlights_success(self, video_service, temp_video_file):
        """Test successful video analysis"""
        with patch('ffmpeg.probe') as mock_probe:
            mock_probe.return_value = {
                'streams': [{'duration': '120.0'}]
            }
            
            highlights = await video_service.analyze_highlights(temp_video_file)
            
            assert isinstance(highlights, list)
            assert len(highlights) > 0
            assert all('start_time' in h and 'end_time' in h for h in highlights)
    
    @pytest.mark.asyncio
    async def test_analyze_highlights_file_not_found(self, video_service):
        """Test analysis with non-existent file"""
        with pytest.raises(VideoServiceError):
            await video_service.analyze_highlights('/nonexistent/file.mp4')
    
    @pytest.mark.asyncio
    async def test_create_clip_success(self, video_service, temp_video_file):
        """Test successful clip creation"""
        output_path = tempfile.mktemp(suffix='.mp4')
        
        with patch('ffmpeg.input') as mock_input, \
             patch('ffmpeg.output') as mock_output, \
             patch('ffmpeg.run') as mock_run, \
             patch('os.path.getsize', return_value=1024):
            
            mock_input.return_value = Mock()
            mock_output.return_value = Mock()
            
            result = await video_service.create_clip(
                temp_video_file, output_path, 10.0, 30.0, "720p"
            )
            
            assert result['output_path'] == output_path
            assert result['duration'] == 20.0
            assert result['quality'] == "720p"
            assert result['file_size'] == 1024
    
    @pytest.mark.asyncio
    async def test_create_clip_invalid_times(self, video_service, temp_video_file):
        """Test clip creation with invalid time parameters"""
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # End time before start time
        with pytest.raises(VideoServiceError, match="End time must be greater"):
            await video_service.create_clip(
                temp_video_file, output_path, 30.0, 10.0, "720p"
            )
        
        # Duration too long
        with pytest.raises(VideoServiceError, match="cannot exceed 5 minutes"):
            await video_service.create_clip(
                temp_video_file, output_path, 0.0, 400.0, "720p"
            )
    
    @pytest.mark.asyncio
    async def test_get_video_info_success(self, video_service, temp_video_file):
        """Test getting video information"""
        mock_probe_data = {
            'format': {
                'duration': '120.5',
                'bit_rate': '2000000',
                'size': '5000000'
            },
            'streams': [
                {
                    'codec_type': 'video',
                    'width': 1920,
                    'height': 1080,
                    'r_frame_rate': '30/1',
                    'codec_name': 'h264'
                },
                {
                    'codec_type': 'audio',
                    'codec_name': 'aac'
                }
            ]
        }
        
        with patch('ffmpeg.probe', return_value=mock_probe_data):
            info = await video_service.get_video_info(temp_video_file)
            
            assert info['duration'] == 120.5
            assert info['width'] == 1920
            assert info['height'] == 1080
            assert info['fps'] == 30.0
            assert info['has_audio'] is True
    
    @pytest.mark.asyncio
    async def test_get_video_info_no_video_stream(self, video_service, temp_video_file):
        """Test video info with no video stream"""
        mock_probe_data = {
            'format': {'duration': '120.5'},
            'streams': [{'codec_type': 'audio'}]
        }
        
        with patch('ffmpeg.probe', return_value=mock_probe_data):
            with pytest.raises(VideoServiceError, match="No video stream found"):
                await video_service.get_video_info(temp_video_file)


@pytest.mark.integration
class TestVideoServiceIntegration:
    """Integration tests for VideoService"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete video processing workflow"""
        # This would test with actual video files in a real environment
        pass


if __name__ == "__main__":
    pytest.main([__file__])
