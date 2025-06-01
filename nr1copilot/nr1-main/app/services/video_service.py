
"""
Professional video service with YouTube integration and processing
Handles video downloads, analysis, and clip generation
"""

import logging
import os
import asyncio
from typing import Dict, Any, List, Optional
import yt_dlp
from ..config import get_settings
from .ffmpeg_service import process_video, FFmpegServiceError

logger = logging.getLogger(__name__)

class VideoServiceError(Exception):
    """Custom exception for video service errors"""
    pass

class VideoService:
    """Professional video service implementation"""
    
    def __init__(self):
        self.settings = get_settings()
        self.download_path = self.settings.video_storage_path
        os.makedirs(self.download_path, exist_ok=True)
    
    async def download_youtube_video(self, url: str, video_id: str) -> Dict[str, Any]:
        """
        Download video from YouTube
        
        Args:
            url (str): YouTube video URL
            video_id (str): Unique video identifier
            
        Returns:
            dict: Download result with file path and metadata
            
        Raises:
            VideoServiceError: If download fails
        """
        try:
            logger.info(f"Downloading YouTube video: {url}")
            
            output_path = os.path.join(self.download_path, f"{video_id}")
            
            ydl_opts = {
                'format': 'best[height<=720]',
                'outtmpl': f'{output_path}/%(title)s.%(ext)s',
                'extractaudio': False,
                'audioformat': 'mp3',
                'ignoreerrors': True,
                'no_warnings': True,
            }
            
            # Run download in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._download_video_sync,
                url,
                ydl_opts
            )
            
            logger.info(f"✅ Video downloaded successfully: {result['filepath']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to download video: {e}")
            raise VideoServiceError(f"Download failed: {e}")
    
    def _download_video_sync(self, url: str, ydl_opts: dict) -> Dict[str, Any]:
        """Synchronous video download helper"""
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first
            info = ydl.extract_info(url, download=False)
            
            # Download the video
            ydl.download([url])
            
            # Find downloaded file
            output_template = ydl_opts['outtmpl']
            filename = ydl.prepare_filename(info)
            
            return {
                "filepath": filename,
                "title": info.get('title', 'Unknown'),
                "duration": info.get('duration', 0),
                "uploader": info.get('uploader', 'Unknown'),
                "view_count": info.get('view_count', 0),
                "description": info.get('description', ''),
                "upload_date": info.get('upload_date', ''),
                "thumbnail": info.get('thumbnail', ''),
                "format": info.get('ext', 'mp4')
            }
    
    async def analyze_video_highlights(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Analyze video to find potential highlights
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            list: List of highlight segments with timestamps
        """
        try:
            logger.info(f"Analyzing video highlights: {video_path}")
            
            # Simple implementation - in production, use AI/ML for better analysis
            highlights = []
            
            # Get video duration using ffmpeg
            import ffmpeg
            probe = ffmpeg.probe(video_path)
            duration = float(probe['streams'][0]['duration'])
            
            # Generate segments (every 60 seconds for demo)
            segment_length = 60
            num_segments = int(duration // segment_length)
            
            for i in range(min(num_segments, 10)):  # Limit to 10 segments
                start_time = i * segment_length
                highlights.append({
                    "start_time": start_time,
                    "end_time": min(start_time + segment_length, duration),
                    "score": 0.8 - (i * 0.1),  # Decreasing score
                    "reason": f"Segment {i + 1}",
                    "description": f"Potential highlight at {start_time}s"
                })
            
            logger.info(f"✅ Found {len(highlights)} potential highlights")
            return highlights
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze video: {e}")
            raise VideoServiceError(f"Analysis failed: {e}")
    
    async def create_clip(
        self,
        input_path: str,
        output_filename: str,
        start_time: int,
        duration: int = 60
    ) -> Dict[str, Any]:
        """
        Create a video clip from the original video
        
        Args:
            input_path (str): Path to source video
            output_filename (str): Output filename
            start_time (int): Start time in seconds
            duration (int): Clip duration in seconds
            
        Returns:
            dict: Clip creation result
        """
        try:
            output_path = os.path.join(self.download_path, "clips", output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logger.info(f"Creating clip: {start_time}s-{start_time + duration}s")
            
            # Use FFmpeg service
            result = process_video(
                input_path=input_path,
                output_path=output_path,
                start=start_time,
                duration=duration
            )
            
            if result.get("success"):
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "start_time": start_time,
                    "duration": duration,
                    "file_size": file_size,
                    "filename": output_filename
                }
            else:
                raise VideoServiceError("FFmpeg processing failed")
                
        except FFmpegServiceError as e:
            logger.error(f"❌ FFmpeg error in clip creation: {e}")
            raise VideoServiceError(f"Clip creation failed: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in clip creation: {e}")
            raise VideoServiceError(f"Clip creation failed: {e}")
    
    async def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information"""
        try:
            import ffmpeg
            
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise VideoServiceError("No video stream found")
            
            return {
                "duration": float(probe['format']['duration']),
                "size": int(probe['format']['size']),
                "bitrate": int(probe['format'].get('bit_rate', 0)),
                "width": int(video_stream['width']),
                "height": int(video_stream['height']),
                "fps": eval(video_stream['r_frame_rate']),
                "codec": video_stream['codec_name'],
                "format": probe['format']['format_name']
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get video info: {e}")
            raise VideoServiceError(f"Video info extraction failed: {e}")

# Service instance
video_service = VideoService()
