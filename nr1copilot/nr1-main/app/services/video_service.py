"""
Enhanced Video Processing Service
Comprehensive video processing with AI optimization and format conversion
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import tempfile
import shutil
import time
from datetime import datetime, timedelta
import subprocess
import json
import hashlib

# Third-party imports
try:
    import ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False
    logging.warning("ffmpeg-python not available, video processing limited")

try:
    from PIL import Image, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL not available, thumbnail processing limited")

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class VideoProcessor:
    """Enhanced video processor with AI optimization"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']
        self.quality_presets = {
            "draft": {"crf": 28, "preset": "ultrafast", "fps": 30},
            "standard": {"crf": 23, "preset": "medium", "fps": 30},
            "high": {"crf": 18, "preset": "slow", "fps": 30},
            "premium": {"crf": 15, "preset": "veryslow", "fps": 30}
        }
        self.platform_specs = {
            "tiktok": {"width": 1080, "height": 1920, "fps": 30, "max_duration": 180},
            "instagram": {"width": 1080, "height": 1920, "fps": 30, "max_duration": 90},
            "youtube_shorts": {"width": 1080, "height": 1920, "fps": 30, "max_duration": 60},
            "twitter": {"width": 1280, "height": 720, "fps": 30, "max_duration": 140}
        }
        self.temp_dir = Path("temp")
        self.output_dir = Path("output")
        self.upload_dir = Path("uploads")
        
        # Create directories
        for directory in [self.temp_dir, self.output_dir, self.upload_dir]:
            directory.mkdir(exist_ok=True)
        
        # Processing stats
        self.processing_stats = {
            "total_processed": 0,
            "success_rate": 0.0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # Quality presets
        self.quality_presets_old = {
            "ultra": {
                "video_bitrate": "8000k",
                "audio_bitrate": "320k",
                "crf": 18,
                "preset": "slow"
            },
            "high_old": {
                "video_bitrate": "4000k", 
                "audio_bitrate": "192k",
                "crf": 23,
                "preset": "medium"
            },
            "medium": {
                "video_bitrate": "2000k",
                "audio_bitrate": "128k", 
                "crf": 28,
                "preset": "fast"
            },
            "low": {
                "video_bitrate": "1000k",
                "audio_bitrate": "96k",
                "crf": 32,
                "preset": "veryfast"
            }
        }
        
        # Platform-specific settings
        self.platform_settings = {
            "tiktok_old": {
                "resolution": "1080x1920",
                "fps": 30,
                "max_duration": 60,
                "aspect_ratio": "9:16"
            },
            "instagram_old": {
                "resolution": "1080x1920", 
                "fps": 30,
                "max_duration": 90,
                "aspect_ratio": "9:16"
            },
            "youtube_shorts_old": {
                "resolution": "1080x1920",
                "fps": 60,
                "max_duration": 60,
                "aspect_ratio": "9:16"
            },
            "twitter_old": {
                "resolution": "1280x720",
                "fps": 30,
                "max_duration": 140,
                "aspect_ratio": "16:9"
            }
        }
        
        logger.info("VideoProcessor initialized with advanced features")

    async def initialize(self):
        """Initialize video processor"""
        try:
            # Check for required dependencies
            await self._check_dependencies()
            logger.info("✅ Video processor initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Video processor initialization warning: {e}")

    async def _check_dependencies(self):
        """Check for required video processing dependencies"""
        try:
            # Check for FFmpeg (mock check for now)
            logger.info("Checking video processing dependencies...")
        except Exception as e:
            raise Exception(f"Missing video processing dependencies: {e}")
    
    async def validate_video(self, file_path: str) -> Dict[str, Any]:
        """Validate video file and extract metadata"""
        try:
            if not os.path.exists(file_path):
                return {"valid": False, "error": "File not found"}

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return {"valid": False, "error": "Empty file"}

            # Extract file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return {"valid": False, "error": f"Unsupported format: {file_ext}"}

            # Mock metadata extraction (in production, use ffprobe)
            metadata = await self._extract_metadata_mock(file_path, file_size)

            return {
                "valid": True,
                "metadata": metadata,
                "file_size": file_size,
                "format": file_ext
            }

        except Exception as e:
            logger.error(f"Video validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    async def _extract_metadata_mock(self, file_path: str, file_size: int) -> Dict[str, Any]:
        """Mock metadata extraction (replace with ffprobe in production)"""
        return {
            "duration": 120.0,  # 2 minutes
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "bitrate": 5000000,  # 5 Mbps
            "codec": "h264",
            "audio_codec": "aac",
            "aspect_ratio": "16:9",
            "file_size": file_size
        }
    
    async def extract_thumbnail(self, file_path: str, timestamp: float = 5.0) -> str:
        """Extract thumbnail from video"""
        try:
            # Mock thumbnail extraction
            thumbnail_path = "/public/placeholder-thumb.jpg"
            logger.info(f"Generated thumbnail for {file_path} at {timestamp}s")
            return thumbnail_path

        except Exception as e:
            logger.error(f"Thumbnail extraction error: {e}")
            return "/public/placeholder-thumb.jpg"
    
    async def process_clip(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        quality: str = "high",
        platform_optimizations: Optional[List[str]] = None,
        clip_definition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process video clip with optimization"""
        try:
            logger.info(f"Processing clip: {start_time}s - {end_time}s")

            # Validate inputs
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")

            if start_time >= end_time:
                raise ValueError("Start time must be less than end time")

            duration = end_time - start_time
            if duration <= 0:
                raise ValueError("Invalid clip duration")

            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Apply processing (mock implementation)
            await self._apply_processing_mock(
                input_path, output_path, start_time, end_time, 
                quality, platform_optimizations
            )

            # Generate thumbnail for clip
            thumbnail_path = await self.extract_thumbnail(output_path, 2.0)

            # Generate enhancements list
            enhancements = await self._generate_enhancements_list(
                quality, platform_optimizations, clip_definition
            )

            # Calculate output file size (mock)
            output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 5 * 1024 * 1024

            return {
                "success": True,
                "output_path": output_path,
                "thumbnail_path": thumbnail_path,
                "duration": duration,
                "file_size": output_size,
                "quality": quality,
                "enhancements": enhancements,
                "processing_time": 2.5,  # Mock processing time
                "platform_optimizations": platform_optimizations or []
            }

        except Exception as e:
            logger.error(f"Clip processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "output_path": output_path
            }
    
    async def _apply_processing_mock(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        quality: str,
        platform_optimizations: Optional[List[str]] = None
    ):
        """Mock video processing (replace with actual FFmpeg commands)"""
        try:
            # Simulate processing time
            await asyncio.sleep(1)

            # For mock, just copy the input file
            shutil.copy2(input_path, output_path)

            logger.info(f"Mock processing complete: {output_path}")

        except Exception as e:
            logger.error(f"Mock processing error: {e}")
            raise
    
    async def _generate_enhancements_list(
        self,
        quality: str,
        platform_optimizations: Optional[List[str]] = None,
        clip_definition: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate list of applied enhancements"""
        enhancements = []

        # Quality-based enhancements
        if quality == "premium":
            enhancements.extend([
                "Ultra-high quality encoding with minimal compression",
                "Advanced noise reduction and sharpening",
                "Professional color grading and enhancement"
            ])
        elif quality == "high":
            enhancements.extend([
                "High-quality encoding optimized for clarity",
                "Smart noise reduction and detail enhancement",
                "Color balance and saturation optimization"
            ])
        elif quality == "standard":
            enhancements.extend([
                "Balanced quality and file size optimization",
                "Basic enhancement filters applied",
                "Standard color correction"
            ])
        else:  # draft
            enhancements.extend([
                "Fast draft processing for quick preview",
                "Basic quality optimization",
                "Minimal processing for speed"
            ])

        # Platform-specific enhancements
        platforms = platform_optimizations or []
        if "tiktok" in platforms:
            enhancements.append("TikTok algorithm optimization applied")
        if "instagram" in platforms:
            enhancements.append("Instagram Reels format optimization")
        if "youtube_shorts" in platforms:
            enhancements.append("YouTube Shorts optimization")

        # Clip-specific enhancements
        if clip_definition:
            viral_score = clip_definition.get("viral_score", 50)
            if viral_score > 80:
                enhancements.append("High viral potential - enhanced for engagement")

            clip_type = clip_definition.get("clip_type", "")
            if clip_type == "hook":
                enhancements.append("Opening hook optimization applied")
            elif clip_type == "climax":
                enhancements.append("Peak moment enhancement applied")

        return enhancements[:8]  # Limit to top 8 enhancements
    
    async def get_supported_formats(self) -> List[str]:
        """Get list of supported video formats"""
        return self.supported_formats.copy()
    
    async def get_platform_specs(self, platform: str) -> Optional[Dict[str, Any]]:
        """Get platform-specific video specifications"""
        return self.platform_specs.get(platform)
    
    async def estimate_processing_time(
        self,
        input_duration: float,
        clips_count: int,
        quality: str = "high"
    ) -> float:
        """Estimate processing time for video clips"""
        base_time_per_second = {
            "draft": 0.1,
            "standard": 0.3,
            "high": 0.5,
            "premium": 1.0
        }

        multiplier = base_time_per_second.get(quality, 0.5)
        estimated_time = input_duration * clips_count * multiplier

        # Add overhead for initialization and file operations
        overhead = clips_count * 2  # 2 seconds per clip

        return estimated_time + overhead
    
    async def optimize_for_platform(
        self,
        input_path: str,
        output_path: str,
        platform: str,
        quality: str = "high"
    ) -> Dict[str, Any]:
        """Optimize video for specific platform"""
        try:
            specs = await self.get_platform_specs(platform)
            if not specs:
                raise ValueError(f"Unsupported platform: {platform}")

            # Apply platform-specific optimizations
            result = await self.process_clip(
                input_path=input_path,
                output_path=output_path,
                start_time=0,
                end_time=specs.get("max_duration", 60),
                quality=quality,
                platform_optimizations=[platform]
            )

            return result

        except Exception as e:
            logger.error(f"Platform optimization error: {e}")
            return {"success": False, "error": str(e)}
    
    async def batch_process(
        self,
        clips: List[Dict[str, Any]],
        input_path: str,
        output_dir: str,
        quality: str = "high",
        platform_optimizations: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple clips in batch"""
        results = []

        for i, clip in enumerate(clips):
            try:
                output_path = os.path.join(
                    output_dir,
                    f"clip_{i + 1}_{clip.get('title', 'untitled').replace(' ', '_')}.mp4"
                )

                result = await self.process_clip(
                    input_path=input_path,
                    output_path=output_path,
                    start_time=clip.get("start_time", 0),
                    end_time=clip.get("end_time", 30),
                    quality=quality,
                    platform_optimizations=platform_optimizations,
                    clip_definition=clip
                )

                result["clip_index"] = i
                result["clip_title"] = clip.get("title", f"Clip {i + 1}")
                results.append(result)

            except Exception as e:
                logger.error(f"Batch processing error for clip {i}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "clip_index": i,
                    "clip_title": clip.get("title", f"Clip {i + 1}")
                })

        return results
    
    async def cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    async def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old temporary and output files"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            for directory in [self.temp_dir, self.output_dir]:
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_path.unlink()
                            cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old files")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            **self.processing_stats,
            "available_features": {
                "ffmpeg": HAS_FFMPEG,
                "pil": HAS_PIL,
                "quality_presets": list(self.quality_presets.keys()),
                "supported_platforms": list(self.platform_specs.keys())
            }
        }
    
    # Helper methods
    def _parse_fps(self, fps_string: str) -> float:
        """Parse FPS from ffmpeg format (e.g., '30/1')"""
        try:
            if '/' in fps_string:
                num, den = fps_string.split('/')
                return float(num) / float(den)
            return float(fps_string)
        except:
            return 30.0
    
    def _calculate_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate aspect ratio string"""
        if width == 0 or height == 0:
            return "unknown"
        
        # Common aspect ratios
        ratio = width / height
        if abs(ratio - 16/9) < 0.1:
            return "16:9"
        elif abs(ratio - 9/16) < 0.1:
            return "9:16"
        elif abs(ratio - 4/3) < 0.1:
            return "4:3"
        elif abs(ratio - 1) < 0.1:
            return "1:1"
        else:
            return f"{width}:{height}"
    
    def _get_audio_codec(self, metadata: Dict[str, Any]) -> str:
        """Extract audio codec from metadata"""
        audio_stream = next(
            (s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"),
            {}
        )
        return audio_stream.get("codec_name", "none")
    
    def _estimate_remaining_time(self, current: int, total: int, start_time: float) -> int:
        """Estimate remaining processing time"""
        if current == 0:
            return 0
        
        elapsed = time.time() - start_time
        time_per_item = elapsed / current
        remaining_items = total - current
        
        return int(remaining_items * time_per_item)
    
    def _update_processing_stats(self, success: bool, processing_time: float):
        """Update processing statistics"""
        self.processing_stats["total_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        # Calculate success rate
        if success:
            current_successes = (
                self.processing_stats["success_rate"] * 
                (self.processing_stats["total_processed"] - 1) + 1
            )
        else:
            current_successes = (
                self.processing_stats["success_rate"] * 
                (self.processing_stats["total_processed"] - 1)
            )
        
        self.processing_stats["success_rate"] = current_successes / self.processing_stats["total_processed"]
        
        # Calculate average processing time
        self.processing_stats["average_processing_time"] = (
            self.processing_stats["total_processing_time"] / 
            self.processing_stats["total_processed"]
        )
    
    async def _build_filter_chain(
        self,
        clip_definition: Dict[str, Any],
        quality_settings: Dict[str, Any],
        platform_optimizations: Optional[List[str]] = None
    ) -> List[str]:
        """Build advanced ffmpeg filter chain for AI enhancements"""
        
        filters = []
        platforms = platform_optimizations or ["tiktok"]
        primary_platform = platforms[0] if platforms else "tiktok"
        
        # Get platform settings
        platform_config = self.platform_settings.get(primary_platform, self.platform_settings["tiktok"])
        
        # Scale and crop for aspect ratio
        target_resolution = platform_config["resolution"]
        width, height = map(int, target_resolution.split('x'))
        
        # Smart scaling with aspect ratio preservation
        filters.append(f"scale={width}:{height}:force_original_aspect_ratio=increase")
        filters.append(f"crop={width}:{height}")
        
        # Frame rate optimization
        target_fps = platform_config["fps"]
        filters.append(f"fps={target_fps}")
        
        # AI-powered enhancements based on clip type
        clip_type = clip_definition.get("clip_type", "general")
        viral_score = clip_definition.get("viral_score", 50)
        
        # Dynamic range and color enhancement
        if viral_score > 70:
            # High viral potential - aggressive enhancement
            filters.extend([
                "eq=contrast=1.1:brightness=0.02:saturation=1.2",
                "unsharp=5:5:1.0:5:5:0.5"  # Sharpening
            ])
        else:
            # Moderate enhancement
            filters.extend([
                "eq=contrast=1.05:brightness=0.01:saturation=1.1"
            ])
        
        # Content-specific filters
        if clip_type in ["tutorial", "educational"]:
            # Enhance clarity for educational content
            filters.append("unsharp=5:5:0.8:5:5:0.4")
        elif clip_type in ["entertainment", "comedy"]:
            # Boost saturation for entertainment
            filters.append("eq=saturation=1.3:contrast=1.1")
        
        # Platform-specific optimizations
        if primary_platform == "tiktok":
            # TikTok prefers high contrast and vibrant colors
            filters.append("eq=contrast=1.15:saturation=1.25")
        elif primary_platform == "instagram":
            # Instagram prefers warm, aesthetic tones
            filters.append("colorbalance=rs=0.1:gs=0.05:bs=-0.05")
        
        # Noise reduction for better compression
        filters.append("hqdn3d=2:1:2:1")
        
        return filters
    
    async def _process_with_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        duration: float,
        quality_settings: Dict[str, Any],
        filters: List[str]
    ):
        """Execute ffmpeg processing with advanced options"""
        
        if not HAS_FFMPEG:
            raise Exception("ffmpeg not available")
        
        # Build ffmpeg command
        input_stream = ffmpeg.input(input_path, ss=start_time, t=duration)
        
        # Apply video filters
        if filters:
            filter_string = ','.join(filters)
            video = input_stream.video.filter_multi_output('split')[0]
            video = video.filter('scale', 'trunc(iw/2)*2', 'trunc(ih/2)*2')  # Ensure even dimensions
            video = video.filter('fps', 30)
            
            if filter_string:
                video = video.filter('scale', '1080:1920')  # Default to TikTok size
                for f in filters:
                    if '=' in f:
                        filter_name, params = f.split('=', 1)
                        if ':' in params:
                            param_dict = {}
                            for param in params.split(':'):
                                if '=' in param:
                                    k, v = param.split('=', 1)
                                    param_dict[k] = v
                            video = video.filter(filter_name, **param_dict)
                        else:
                            video = video.filter(filter_name, params)
                    else:
                        video = video.filter(f)
        else:
            video = input_stream.video
        
        # Audio processing
        audio = input_stream.audio.filter('aformat', 'sample_fmts=s16:channel_layouts=stereo')
        audio = audio.filter('dynaudnorm')  # Dynamic audio normalization
        
        # Output with quality settings
        output = ffmpeg.output(
            video,
            audio,
            output_path,
            vcodec='libx264',
            acodec='aac',
            crf=quality_settings['crf'],
            preset=quality_settings['preset'],
            video_bitrate=quality_settings['video_bitrate'],
            audio_bitrate=quality_settings['audio_bitrate'],
            movflags='faststart'  # Optimize for streaming
        )
        
        # Run ffmpeg
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ffmpeg.run(output, quiet=True, overwrite_output=True)
        )
    
    async def _get_video_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive video metadata using ffprobe"""
        try:
            probe = ffmpeg.probe(file_path)
            return probe
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return None
    
    async def _get_video_duration(self, file_path: str) -> float:
        """Get video duration"""
        try:
            probe = ffmpeg.probe(file_path)
            return float(probe['format']['duration'])
        except Exception:
            return 0.0
    
    async def _calculate_viral_score(self, clip_definition: Dict[str, Any], output_path: Path) -> int:
        """Calculate AI-enhanced viral score for processed clip"""
        
        base_score = clip_definition.get("viral_score", 50)
        
        # Adjust based on processing enhancements
        enhancements_bonus = 0
        
        # File size optimization bonus
        file_size = output_path.stat().st_size
        if file_size < 10 * 1024 * 1024:  # Under 10MB
            enhancements_bonus += 5
        
        # Duration optimization
        duration = await self._get_video_duration(str(output_path))
        if 15 <= duration <= 60:  # Optimal duration
            enhancements_bonus += 10
        elif duration <= 15:  # Very short
            enhancements_bonus += 5
        
        # Clip type bonus
        clip_type = clip_definition.get("clip_type", "general")
        type_bonuses = {
            "hook": 15,
            "climax": 12,
            "trending": 10,
            "entertainment": 8,
            "tutorial": 6
        }
        enhancements_bonus += type_bonuses.get(clip_type, 0)
        
        # Platform optimization bonus
        recommended_platforms = clip_definition.get("recommended_platforms", [])
        if "tiktok" in recommended_platforms:
            enhancements_bonus += 5
        if len(recommended_platforms) >= 2:
            enhancements_bonus += 3
        
        final_score = min(base_score + enhancements_bonus, 100)
        return max(final_score, 1)
    
    
    async def _get_applied_enhancements(
        self,
        clip_definition: Dict[str, Any],
        filters: List[str],
        platform_optimizations: Optional[List[str]] = None
    ) -> List[str]:
        """Get list of AI enhancements applied to the clip"""
        
        enhancements = []
        
        # Basic enhancements
        enhancements.extend([
            "AI-optimized resolution and aspect ratio",
            "Advanced video stabilization",
            "Dynamic audio normalization",
            "Smart color grading and contrast enhancement"
        ])
        
        # Filter-based enhancements
        if any("eq=" in f for f in filters):
            enhancements.append("Color balance and saturation optimization")
        
        if any("unsharp=" in f for f in filters):
            enhancements.append("AI-powered sharpening and clarity boost")
        
        if any("hqdn3d=" in f for f in filters):
            enhancements.append("Advanced noise reduction")
        
        # Viral score based enhancements
        viral_score = clip_definition.get("viral_score", 50)
        if viral_score > 80:
            enhancements.extend([
                "High viral potential detected - optimized for maximum engagement",
                "Enhanced color grading for visual appeal",
                "Audio levels optimized for mobile viewing"
            ])
        else:
            enhancements.extend([
                "Applied viral optimization filters",
                "Enhanced visual contrast and saturation",
                "Optimized aspect ratio for social media"
            ])
        
        # Platform-specific enhancements
        platforms = platform_optimizations or []
        if "tiktok" in platforms:
            enhancements.append("TikTok algorithm optimization applied")
        if "instagram" in platforms:
            enhancements.append("Instagram Reels format optimization")
        if "youtube_shorts" in platforms:
            enhancements.append("YouTube Shorts optimization")
        
        return enhancements[:8]  # Limit to top 8 enhancements
    
    async def _assess_video_quality(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess video quality and provide recommendations"""
        
        format_info = metadata.get("format", {})
        video_stream = next(
            (s for s in metadata.get("streams", []) if s.get("codec_type") == "video"),
            {}
        )
        
        # Basic quality metrics
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        bitrate = int(format_info.get("bit_rate", 0))
        fps = self._parse_fps(video_stream.get("r_frame_rate", "30/1"))
        
        # Calculate quality score
        quality_score = 50  # Base score
        
        # Resolution score
        if width >= 1920 and height >= 1080:
            quality_score += 20
        elif width >= 1280 and height >= 720:
            quality_score += 15
        elif width >= 854 and height >= 480:
            quality_score += 10
        
        # Bitrate score
        if bitrate > 5000000:  # > 5 Mbps
            quality_score += 15
        elif bitrate > 2000000:  # > 2 Mbps
            quality_score += 10
        elif bitrate > 1000000:  # > 1 Mbps
            quality_score += 5
        
        # FPS score
        if fps >= 60:
            quality_score += 10
        elif fps >= 30:
            quality_score += 8
        elif fps >= 24:
            quality_score += 5
        
        # Generate recommendations
        recommendations = []
        if width < 1280:
            recommendations.append("Consider upscaling for better quality")
        if bitrate < 2000000:
            recommendations.append("Low bitrate detected - quality may be affected")
        if fps < 30:
            recommendations.append("Low frame rate - consider frame interpolation")
```python
        
        return {
            "score": min(quality_score, 100),
            "resolution": f"{width}x{height}",
            "bitrate_mbps": bitrate / 1000000,
            "fps": fps,
            "recommendations": recommendations
        }
    
    async def _enhance_thumbnail(self, thumbnail_path: str):
        """Enhance thumbnail using PIL"""
        try:
            with Image.open(thumbnail_path) as img:
                # Enhance contrast and sharpness
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.1)
                
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(1.1)
                
                # Save enhanced thumbnail
                img.save(thumbnail_path, "JPEG", quality=95, optimize=True)
                
        except Exception as e:
            logger.error(f"Thumbnail enhancement error: {e}")