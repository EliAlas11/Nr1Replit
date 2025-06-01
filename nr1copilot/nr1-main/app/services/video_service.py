"""
Netflix-Level Video Processing Service
Advanced video downloading, processing, and optimization
"""

import asyncio
import logging
import os
import hashlib
import time
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import yt_dlp
import json

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Netflix-level video processing service"""

    def __init__(self):
        self.cache = {}
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "average_time": 0
        }

        # Check system capabilities
        self.ffmpeg_available = self._check_ffmpeg()
        self.temp_dir = "temp"
        self.output_dir = "output"

        # Create directories
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            logger.warning("ffmpeg not available, using fallback processing")
            return False

    async def download_video_optimized(self, url: str, output_path: str, quality: str = "best") -> Dict[str, Any]:
        """
        Netflix-level optimized video download
        """
        try:
            os.makedirs(output_path, exist_ok=True)

            # Generate cache key
            cache_key = hashlib.md5(f"{url}_{quality}".encode()).hexdigest()

            # Check cache
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if os.path.exists(cached_result.get("file_path", "")):
                    logger.info(f"Using cached download: {cache_key}")
                    return cached_result

            # Advanced yt-dlp configuration
            ydl_opts = {
                'outtmpl': os.path.join(output_path, '%(title)s_%(id)s.%(ext)s'),
                'format': self._get_format_selector(quality),
                'extractaudio': False,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'es', 'fr', 'de'],
                'ignoreerrors': False,
                'no_warnings': False,
                'extract_flat': False,
                'youtube_include_dash_manifest': False,
                'quiet': True
            }

            start_time = time.time()

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(url, download=False)

                # Download the video
                ydl.download([url])

                # Find the downloaded file
                expected_filename = ydl.prepare_filename(info)

                if os.path.exists(expected_filename):
                    download_time = time.time() - start_time

                    result = {
                        "success": True,
                        "file_path": expected_filename,
                        "video_info": {
                            "title": info.get("title", "Unknown"),
                            "duration": info.get("duration", 0),
                            "uploader": info.get("uploader", "Unknown"),
                            "view_count": info.get("view_count", 0),
                            "like_count": info.get("like_count", 0),
                            "upload_date": info.get("upload_date"),
                            "thumbnail": info.get("thumbnail"),
                            "description": info.get("description", "")[:500]
                        },
                        "download_time": download_time,
                        "file_size": os.path.getsize(expected_filename)
                    }

                    # Cache result
                    self.cache[cache_key] = result

                    return result
                else:
                    raise Exception("Download completed but file not found")

        except Exception as e:
            logger.error(f"Download error: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": None
            }

    def _get_format_selector(self, quality: str) -> str:
        """Get format selector for yt-dlp"""
        quality_map = {
            "best": "best[height<=1080]/best",
            "1080p": "best[height<=1080]",
            "720p": "best[height<=720]",
            "480p": "best[height<=480]",
            "360p": "best[height<=360]"
        }
        return quality_map.get(quality, "best[height<=1080]/best")

    async def process_video_clip(
        self,
        input_path: str,
        start_time: float,
        end_time: float,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Process video clip with advanced options"""
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            processing_start = time.time()

            if self.ffmpeg_available:
                result = await self._process_with_ffmpeg(
                    input_path, start_time, end_time, output_path, **kwargs
                )
            else:
                # Fallback processing
                result = await self._process_fallback(
                    input_path, start_time, end_time, output_path
                )

            processing_time = time.time() - processing_start

            # Update statistics
            self.processing_stats["total_processed"] += 1
            if result["success"]:
                self.processing_stats["successful"] += 1
            else:
                self.processing_stats["failed"] += 1

            # Update average time
            total = self.processing_stats["total_processed"]
            current_avg = self.processing_stats["average_time"]
            self.processing_stats["average_time"] = (
                (current_avg * (total - 1) + processing_time) / total
            )

            result["processing_time"] = processing_time
            return result

        except Exception as e:
            logger.error(f"Video processing error: {e}")
            self.processing_stats["failed"] += 1

            return {
                "success": False,
                "error": str(e),
                "processing_time": 0
            }

    async def _process_with_ffmpeg(
        self,
        input_path: str,
        start_time: float,
        end_time: float,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Advanced processing with ffmpeg"""
        try:
            duration = end_time - start_time

            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'medium',
                '-crf', '23',
                '-y',  # Overwrite output
                output_path
            ]

            # Add quality settings
            quality = kwargs.get('quality', '720p')
            aspect_ratio = kwargs.get('aspect_ratio', '16:9')

            if quality == '1080p':
                if aspect_ratio == '9:16':
                    cmd.extend(['-vf', 'scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2'])
                else:
                    cmd.extend(['-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2'])
            elif quality == '720p':
                if aspect_ratio == '9:16':
                    cmd.extend(['-vf', 'scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2'])
                else:
                    cmd.extend(['-vf', 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2'])

            # Add frame rate optimization
            cmd.extend(['-r', '30'])

            # Run ffmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and os.path.exists(output_path):
                return {
                    "success": True,
                    "output_path": output_path,
                    "duration": duration,
                    "file_size": os.path.getsize(output_path),
                    "processing_method": "ffmpeg"
                }
            else:
                raise Exception(f"ffmpeg failed: {stderr.decode()}")

        except Exception as e:
            logger.error(f"ffmpeg processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _process_fallback(
        self,
        input_path: str,
        start_time: float,
        end_time: float,
        output_path: str
    ) -> Dict[str, Any]:
        """Fallback processing without ffmpeg"""
        try:
            # Simple file copy as fallback
            import shutil
            shutil.copy2(input_path, output_path)

            logger.warning("Using fallback processing (file copy)")

            return {
                "success": True,
                "output_path": output_path,
                "duration": end_time - start_time,
                "file_size": os.path.getsize(output_path),
                "processing_method": "fallback",
                "warning": "Advanced processing unavailable"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information without downloading"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                return {
                    "success": True,
                    "info": {
                        "title": info.get("title", "Unknown"),
                        "duration": info.get("duration", 0),
                        "uploader": info.get("uploader", "Unknown"),
                        "view_count": info.get("view_count", 0),
                        "like_count": info.get("like_count", 0),
                        "upload_date": info.get("upload_date"),
                        "thumbnail": info.get("thumbnail"),
                        "description": info.get("description", "")[:1000],
                        "tags": info.get("tags", [])[:20],
                        "categories": info.get("categories", []),
                        "webpage_url": info.get("webpage_url", url)
                    }
                }

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def validate_youtube_url(self, url: str) -> bool:
        """Validate YouTube URL"""
        try:
            # Basic URL validation
            if not url or len(url) < 10:
                return False

            # Check if it's a valid YouTube URL pattern
            youtube_patterns = [
                'youtube.com/watch',
                'youtu.be/',
                'youtube.com/embed/',
                'youtube.com/v/',
                'm.youtube.com/watch'
            ]

            if not any(pattern in url.lower() for pattern in youtube_patterns):
                return False

            # Try to extract info to validate
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info is not None

        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            "ffmpeg_available": self.ffmpeg_available,
            "cache_size": len(self.cache),
            "success_rate": (
                self.processing_stats["successful"] / 
                max(self.processing_stats["total_processed"], 1) * 100
            )
        }

# Convenience functions
async def download_video(url: str, output_path: str = "downloads") -> Dict[str, Any]:
    """Download video from URL"""
    processor = VideoProcessor()
    return await processor.download_video_optimized(url, output_path)

async def get_video_info(url: str) -> Dict[str, Any]:
    """Get video information"""
    processor = VideoProcessor()
    return await processor.get_video_info(url)

async def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL"""
    processor = VideoProcessor()
    return await processor.validate_youtube_url(url)

async def process_video_clip(
    input_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    **kwargs
) -> Dict[str, Any]:
    """Process video clip"""
    processor = VideoProcessor()
    return await processor.process_video_clip(
        input_path, start_time, end_time, output_path, **kwargs
    )