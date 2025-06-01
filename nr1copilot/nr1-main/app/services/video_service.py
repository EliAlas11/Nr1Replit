"""
Video Service Layer
"""

import os
import logging
from typing import Dict, Any, Optional
import yt_dlp
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class VideoServiceError(Exception):
    """Custom exception for video service errors"""
    pass

async def download_video(url: str, output_path: str) -> Dict[str, Any]:
    """
    Download video from YouTube URL

    Args:
        url: YouTube video URL
        output_path: Directory to save the video

    Returns:
        Dict with download information

    Raises:
        VideoServiceError: If download fails
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Configure yt-dlp options
        ydl_opts = {
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'format': 'best[height<=720]',  # Limit to 720p
            'extractaudio': False,
            'audioformat': 'mp3',
            'embed_subs': True,
            'writesubtitles': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)

            # Check duration limit
            if duration > settings.max_video_duration:
                raise VideoServiceError(f"Video too long: {duration}s (max: {settings.max_video_duration}s)")

            # Download the video
            ydl.download([url])

            logger.info(f"Successfully downloaded: {video_title}")

            return {
                'success': True,
                'title': video_title,
                'duration': duration,
                'file_path': os.path.join(output_path, f"{video_title}.mp4")
            }

    except yt_dlp.DownloadError as e:
        logger.error(f"yt-dlp download error: {e}")
        raise VideoServiceError(f"Download failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in download_video: {e}")
        raise VideoServiceError(f"Download failed: {e}")

async def get_video_info(url: str) -> Dict[str, Any]:
    """
    Get video information without downloading

    Args:
        url: YouTube video URL

    Returns:
        Dict with video information
    """
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)

            return {
                'success': True,
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'thumbnail': info.get('thumbnail', ''),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', '')
            }

    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        raise VideoServiceError(f"Failed to get video info: {e}")

async def validate_youtube_url(url: str) -> bool:
    """
    Validate if URL is a valid YouTube URL

    Args:
        url: URL to validate

    Returns:
        True if valid YouTube URL
    """
    try:
        # Basic URL validation
        youtube_domains = ['youtube.com', 'youtu.be', 'm.youtube.com']

        if not any(domain in url for domain in youtube_domains):
            return False

        # Try to extract info (without downloading)
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            ydl.extract_info(url, download=False)
            return True

    except Exception:
        return False

async def process_video_clip(input_path: str, output_path: str, start_time: float, end_time: float) -> Dict[str, Any]:
    """
    Create a clip from video using FFmpeg

    Args:
        input_path: Path to input video
        output_path: Path for output clip
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Dict with processing results
    """
    try:
        from .ffmpeg_service import process_video

        duration = end_time - start_time
        if duration <= 0:
            raise VideoServiceError("Invalid time range")

        if duration > 300:  # 5 minutes max
            raise VideoServiceError("Clip duration too long (max 5 minutes)")

        result = process_video(input_path, output_path, int(start_time), int(duration))

        if result.get('success'):
            return {
                'success': True,
                'output_path': result.get('output'),
                'duration': duration
            }
        else:
            raise VideoServiceError("FFmpeg processing failed")

    except Exception as e:
        logger.error(f"Video processing error: {e}")
        raise VideoServiceError(f"Processing failed: {e}")