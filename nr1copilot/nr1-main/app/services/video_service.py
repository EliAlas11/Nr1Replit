
"""
Video Service Layer - Netflix Level Implementation
"""

import os
import logging
import asyncio
import aiofiles
from typing import Dict, Any, Optional, List
import yt_dlp
import hashlib
from datetime import datetime, timedelta
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class VideoServiceError(Exception):
    """Custom exception for video service errors"""
    pass

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
                'writethumbnail': True,
                'writeinfojson': True,
            }
            
            start_time = datetime.now()
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(url, download=False)
                
                # Validate video
                await self._validate_video(info)
                
                # Download
                ydl.download([url])
                
                video_title = info.get('title', 'Unknown')
                video_id = info.get('id', 'unknown')
                duration = info.get('duration', 0)
                
                # Find downloaded file
                import glob
                pattern = os.path.join(output_path, f"*{video_id}*")
                files = glob.glob(pattern)
                video_file = next((f for f in files if f.endswith(('.mp4', '.webm', '.mkv'))), None)
                
                if not video_file:
                    raise VideoServiceError("Downloaded file not found")
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                result = {
                    'success': True,
                    'title': video_title,
                    'duration': duration,
                    'file_path': video_file,
                    'file_size': os.path.getsize(video_file),
                    'quality': quality,
                    'processing_time': processing_time,
                    'video_info': info,
                    'subtitles': info.get('subtitles', {}),
                    'thumbnail': info.get('thumbnail'),
                    'uploader': info.get('uploader'),
                    'view_count': info.get('view_count', 0)
                }
                
                # Cache result
                self.cache[cache_key] = result
                
                # Update stats
                self._update_stats(processing_time, True)
                
                logger.info(f"Successfully downloaded: {video_title} ({processing_time:.2f}s)")
                return result
                
        except yt_dlp.DownloadError as e:
            logger.error(f"yt-dlp download error: {e}")
            self._update_stats(0, False)
            raise VideoServiceError(f"Download failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in download_video_optimized: {e}")
            self._update_stats(0, False)
            raise VideoServiceError(f"Download failed: {e}")
    
    async def get_video_info_advanced(self, url: str) -> Dict[str, Any]:
        """
        Get comprehensive video information
        """
        try:
            cache_key = f"info_{hashlib.md5(url.encode()).hexdigest()}"
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                result = {
                    'success': True,
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'thumbnail': info.get('thumbnail', ''),
                    'uploader': info.get('uploader', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'categories': info.get('categories', []),
                    'tags': info.get('tags', []),
                    'language': info.get('language', 'unknown'),
                    'fps': info.get('fps'),
                    'width': info.get('width'),
                    'height': info.get('height'),
                    'subtitles_available': bool(info.get('subtitles')),
                    'formats_available': len(info.get('formats', [])),
                    'is_live': info.get('is_live', False),
                    'availability': info.get('availability', 'unknown')
                }
                
                self.cache[cache_key] = result
                return result
                
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise VideoServiceError(f"Failed to get video info: {e}")
    
    async def validate_youtube_url_advanced(self, url: str) -> Dict[str, Any]:
        """
        Advanced YouTube URL validation
        """
        try:
            # Basic URL pattern validation
            import re
            youtube_patterns = [
                r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})',
                r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})'
            ]
            
            video_id = None
            for pattern in youtube_patterns:
                match = re.search(pattern, url)
                if match:
                    video_id = match.group(1)
                    break
            
            if not video_id:
                return {
                    'valid': False,
                    'error': 'Invalid YouTube URL format',
                    'supported_formats': ['youtube.com/watch?v=', 'youtu.be/', 'youtube.com/shorts/']
                }
            
            # Try to extract basic info
            try:
                info = await self.get_video_info_advanced(url)
                
                return {
                    'valid': True,
                    'video_id': video_id,
                    'accessible': True,
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'availability': info.get('availability')
                }
            except Exception as e:
                return {
                    'valid': True,
                    'video_id': video_id,
                    'accessible': False,
                    'error': str(e)
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    def _get_format_selector(self, quality: str) -> str:
        """Get yt-dlp format selector based on quality"""
        quality_formats = {
            'highest': 'best',
            'high': 'best[height<=1080]',
            'medium': 'best[height<=720]',
            'low': 'best[height<=480]',
            'audio_only': 'bestaudio'
        }
        return quality_formats.get(quality, 'best[height<=1080]')
    
    async def _validate_video(self, info: Dict[str, Any]) -> None:
        """Validate video before downloading"""
        duration = info.get('duration', 0)
        
        if duration > settings.max_video_duration:
            raise VideoServiceError(
                f"Video too long: {duration}s (max: {settings.max_video_duration}s)"
            )
        
        if info.get('is_live', False):
            raise VideoServiceError("Live streams are not supported")
        
        availability = info.get('availability', 'unknown')
        if availability not in ['public', 'unlisted', 'unknown']:
            raise VideoServiceError(f"Video not accessible: {availability}")
    
    def _update_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics"""
        self.processing_stats['total_processed'] += 1
        
        if success:
            self.processing_stats['successful'] += 1
            # Update average time
            current_avg = self.processing_stats['average_time']
            current_count = self.processing_stats['successful']
            self.processing_stats['average_time'] = (
                (current_avg * (current_count - 1) + processing_time) / current_count
            )
        else:
            self.processing_stats['failed'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total = self.processing_stats['total_processed']
        return {
            **self.processing_stats,
            'success_rate': (
                self.processing_stats['successful'] / total * 100 
                if total > 0 else 0
            ),
            'cache_size': len(self.cache)
        }
    
    async def cleanup_cache(self, max_age_hours: int = 24) -> None:
        """Clean up old cache entries"""
        # This is a simplified cleanup - in production, you'd track timestamps
        if len(self.cache) > 1000:  # Simple size-based cleanup
            # Keep only the most recent 500 entries
            keys_to_remove = list(self.cache.keys())[:-500]
            for key in keys_to_remove:
                del self.cache[key]
            logger.info(f"Cleaned up {len(keys_to_remove)} cache entries")

# Global video processor instance
video_processor = VideoProcessor()

# Convenience functions for backwards compatibility
async def download_video(url: str, output_path: str) -> Dict[str, Any]:
    """Download video using the global processor"""
    return await video_processor.download_video_optimized(url, output_path)

async def get_video_info(url: str) -> Dict[str, Any]:
    """Get video info using the global processor"""
    return await video_processor.get_video_info_advanced(url)

async def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL using the global processor"""
    result = await video_processor.validate_youtube_url_advanced(url)
    return result.get('valid', False)

async def process_video_clip(input_path: str, output_path: str, start_time: float, end_time: float) -> Dict[str, Any]:
    """Process video clip - placeholder for FFmpeg integration"""
    try:
        # This would integrate with the CloudVideoProcessor
        duration = end_time - start_time
        
        if duration <= 0:
            raise VideoServiceError("Invalid time range")
        
        if duration > 300:  # 5 minutes max
            raise VideoServiceError("Clip duration too long (max 5 minutes)")
        
        # Placeholder for actual processing
        return {
            'success': True,
            'output_path': output_path,
            'duration': duration,
            'message': 'Clip processing would happen here'
        }
        
    except Exception as e:
        logger.error(f"Video clip processing error: {e}")
        raise VideoServiceError(f"Processing failed: {e}")
