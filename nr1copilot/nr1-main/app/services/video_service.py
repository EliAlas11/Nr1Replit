
"""
ViralClip Pro - Netflix-Level Video Processing Service
Advanced video processing with AI enhancement and optimization
"""

import asyncio
import os
import logging
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path

try:
    import ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False
    logging.warning("ffmpeg-python not available, video processing limited")

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logging.warning("OpenCV not available, video analysis limited")

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class VideoProcessor:
    """Netflix-level video processing with advanced features"""
    
    def __init__(self):
        self.supported_formats = settings.allowed_video_formats
        self.max_file_size = settings.max_file_size
        self.max_duration = settings.max_video_duration
        self.temp_dir = settings.temp_path
        self.output_dir = settings.output_path
        
        # Ensure directories exist
        for directory in [self.temp_dir, self.output_dir]:
            os.makedirs(directory, exist_ok=True)
    
    async def validate_video(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive video validation with detailed analysis"""
        try:
            if not os.path.exists(file_path):
                return {"valid": False, "error": "File does not exist"}
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return {
                    "valid": False, 
                    "error": f"File size {file_size} exceeds limit {self.max_file_size}"
                }
            
            # Probe video with ffmpeg
            if HAS_FFMPEG:
                try:
                    probe = ffmpeg.probe(file_path)
                    video_info = next(
                        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), 
                        None
                    )
                    
                    if not video_info:
                        return {"valid": False, "error": "No video stream found"}
                    
                    duration = float(probe['format'].get('duration', 0))
                    if duration > self.max_duration:
                        return {
                            "valid": False, 
                            "error": f"Duration {duration}s exceeds limit {self.max_duration}s"
                        }
                    
                    # Extract comprehensive metadata
                    metadata = {
                        "duration": duration,
                        "width": int(video_info.get('width', 0)),
                        "height": int(video_info.get('height', 0)),
                        "fps": eval(video_info.get('r_frame_rate', '0/1')),
                        "codec": video_info.get('codec_name'),
                        "bitrate": int(probe['format'].get('bit_rate', 0)),
                        "file_size": file_size,
                        "format": probe['format'].get('format_name'),
                        "has_audio": any(s['codec_type'] == 'audio' for s in probe['streams'])
                    }
                    
                    return {
                        "valid": True,
                        "metadata": metadata,
                        "quality_score": self._calculate_quality_score(metadata)
                    }
                    
                except Exception as e:
                    logger.error(f"FFmpeg probe error: {e}")
                    return {"valid": False, "error": f"Video analysis failed: {str(e)}"}
            
            # Fallback validation
            return {"valid": True, "metadata": {"file_size": file_size}}
            
        except Exception as e:
            logger.error(f"Video validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def _calculate_quality_score(self, metadata: Dict) -> int:
        """Calculate video quality score (0-100)"""
        score = 50  # Base score
        
        # Resolution scoring
        width = metadata.get('width', 0)
        height = metadata.get('height', 0)
        pixels = width * height
        
        if pixels >= 1920 * 1080:  # 1080p+
            score += 25
        elif pixels >= 1280 * 720:  # 720p
            score += 15
        elif pixels >= 640 * 480:   # 480p
            score += 5
        
        # FPS scoring
        fps = metadata.get('fps', 0)
        if fps >= 60:
            score += 15
        elif fps >= 30:
            score += 10
        elif fps >= 24:
            score += 5
        
        # Bitrate scoring
        bitrate = metadata.get('bitrate', 0)
        if bitrate >= 5000000:  # 5 Mbps+
            score += 10
        elif bitrate >= 2000000:  # 2 Mbps+
            score += 5
        
        return min(score, 100)
    
    async def extract_thumbnail(self, input_path: str, timestamp: float = None) -> str:
        """Extract high-quality thumbnail from video"""
        try:
            if not HAS_FFMPEG:
                raise ValueError("FFmpeg not available for thumbnail extraction")
            
            # Generate unique filename
            file_hash = hashlib.md5(f"{input_path}_{timestamp}".encode()).hexdigest()
            thumbnail_path = os.path.join(self.output_dir, f"thumb_{file_hash}.jpg")
            
            # Use middle of video if no timestamp specified
            if timestamp is None:
                probe = ffmpeg.probe(input_path)
                duration = float(probe['format']['duration'])
                timestamp = duration / 2
            
            # Extract thumbnail with high quality
            (
                ffmpeg
                .input(input_path, ss=timestamp)
                .output(
                    thumbnail_path,
                    vframes=1,
                    format='image2',
                    vcodec='mjpeg',
                    q=2  # High quality
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Thumbnail extraction error: {e}")
            return None
    
    async def create_video_clip(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        settings: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create optimized video clip with advanced processing"""
        try:
            if not HAS_FFMPEG:
                raise ValueError("FFmpeg not available for video processing")
            
            # Default settings
            clip_settings = {
                "resolution": "1080p",
                "aspect_ratio": "9:16",
                "fps": 30,
                "quality": "high",
                "enable_stabilization": True,
                "enable_noise_reduction": True,
                "enable_color_enhancement": True,
                **(settings or {})
            }
            
            duration = end_time - start_time
            if duration <= 0:
                raise ValueError("Invalid time range")
            
            # Build FFmpeg command
            input_stream = ffmpeg.input(input_path, ss=start_time, t=duration)
            
            # Video processing pipeline
            video_filters = []
            
            # Resolution and aspect ratio
            if clip_settings["resolution"] == "1080p":
                target_height = 1080
            elif clip_settings["resolution"] == "720p":
                target_height = 720
            else:
                target_height = 1080
            
            if clip_settings["aspect_ratio"] == "9:16":
                target_width = int(target_height * 9 / 16)
                video_filters.append(f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase")
                video_filters.append(f"crop={target_width}:{target_height}")
            elif clip_settings["aspect_ratio"] == "16:9":
                target_width = int(target_height * 16 / 9)
                video_filters.append(f"scale={target_width}:{target_height}")
            
            # Video enhancement filters
            if clip_settings["enable_stabilization"]:
                video_filters.append("vidstabdetect=shakiness=10:accuracy=10")
            
            if clip_settings["enable_noise_reduction"]:
                video_filters.append("hqdn3d=4:3:6:4.5")
            
            if clip_settings["enable_color_enhancement"]:
                video_filters.append("eq=contrast=1.1:brightness=0.05:saturation=1.1")
            
            # Apply filters
            if video_filters:
                video = input_stream.video.filter_multi_output(*video_filters)[0]
            else:
                video = input_stream.video
            
            # Audio processing
            audio = input_stream.audio.filter('volume', '1.2')  # Slight volume boost
            
            # Output settings based on quality
            if clip_settings["quality"] == "high":
                video_bitrate = "5M"
                audio_bitrate = "320k"
                preset = "slow"
            elif clip_settings["quality"] == "medium":
                video_bitrate = "3M"
                audio_bitrate = "192k"
                preset = "medium"
            else:  # low
                video_bitrate = "1.5M"
                audio_bitrate = "128k"
                preset = "fast"
            
            # Generate output
            output = ffmpeg.output(
                video,
                audio,
                output_path,
                vcodec='libx264',
                acodec='aac',
                video_bitrate=video_bitrate,
                audio_bitrate=audio_bitrate,
                preset=preset,
                r=clip_settings["fps"],
                movflags='faststart'  # Optimize for web streaming
            )
            
            # Run processing
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: output.overwrite_output().run(quiet=True)
            )
            
            # Verify output
            if not os.path.exists(output_path):
                raise Exception("Output file was not created")
            
            output_size = os.path.getsize(output_path)
            if output_size == 0:
                raise Exception("Output file is empty")
            
            # Extract thumbnail for the clip
            thumbnail_path = await self.extract_thumbnail(output_path, duration / 2)
            
            return {
                "success": True,
                "output_path": output_path,
                "thumbnail_path": thumbnail_path,
                "file_size": output_size,
                "duration": duration,
                "settings_used": clip_settings,
                "processing_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Video clip creation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "output_path": output_path
            }
    
    async def analyze_video_content(self, input_path: str) -> Dict[str, Any]:
        """Advanced video content analysis using computer vision"""
        try:
            if not HAS_OPENCV:
                return {"analysis_available": False, "error": "OpenCV not available"}
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return {"analysis_available": False, "error": "Cannot open video"}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            # Sample frames for analysis
            sample_frames = []
            sample_interval = max(1, frame_count // 20)  # Sample 20 frames
            
            brightness_values = []
            motion_values = []
            prev_frame = None
            
            for i in range(0, frame_count, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Brightness analysis
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)
                    brightness_values.append(brightness)
                    
                    # Motion analysis
                    if prev_frame is not None:
                        diff = cv2.absdiff(gray, prev_frame)
                        motion = np.mean(diff)
                        motion_values.append(motion)
                    
                    prev_frame = gray.copy()
            
            cap.release()
            
            # Calculate analysis metrics
            avg_brightness = np.mean(brightness_values) if brightness_values else 0
            brightness_variance = np.var(brightness_values) if brightness_values else 0
            avg_motion = np.mean(motion_values) if motion_values else 0
            motion_variance = np.var(motion_values) if motion_values else 0
            
            # Scene change detection
            scene_changes = len([m for m in motion_values if m > avg_motion * 2])
            
            # Quality assessment
            is_too_dark = avg_brightness < 50
            is_too_bright = avg_brightness > 200
            has_good_motion = avg_motion > 10
            has_scene_variety = scene_changes > 3
            
            return {
                "analysis_available": True,
                "duration": duration,
                "frame_count": frame_count,
                "fps": fps,
                "brightness": {
                    "average": float(avg_brightness),
                    "variance": float(brightness_variance),
                    "is_too_dark": is_too_dark,
                    "is_too_bright": is_too_bright
                },
                "motion": {
                    "average": float(avg_motion),
                    "variance": float(motion_variance),
                    "has_good_motion": has_good_motion
                },
                "scenes": {
                    "changes_detected": scene_changes,
                    "has_variety": has_scene_variety
                },
                "quality_score": self._calculate_content_quality(
                    avg_brightness, motion_variance, scene_changes
                ),
                "recommendations": self._generate_content_recommendations(
                    is_too_dark, is_too_bright, has_good_motion, has_scene_variety
                )
            }
            
        except Exception as e:
            logger.error(f"Video content analysis error: {e}")
            return {
                "analysis_available": False,
                "error": str(e)
            }
    
    def _calculate_content_quality(self, brightness: float, motion_var: float, scene_changes: int) -> int:
        """Calculate content quality score (0-100)"""
        score = 50
        
        # Brightness scoring
        if 75 <= brightness <= 175:  # Good brightness range
            score += 20
        elif 50 <= brightness <= 200:  # Acceptable range
            score += 10
        
        # Motion variety scoring
        if motion_var > 100:  # Good motion variety
            score += 15
        elif motion_var > 50:
            score += 10
        
        # Scene variety scoring
        if scene_changes > 5:
            score += 15
        elif scene_changes > 2:
            score += 10
        
        return min(score, 100)
    
    def _generate_content_recommendations(
        self, 
        too_dark: bool, 
        too_bright: bool, 
        good_motion: bool, 
        scene_variety: bool
    ) -> List[str]:
        """Generate content improvement recommendations"""
        recommendations = []
        
        if too_dark:
            recommendations.append("Video appears too dark - consider brightness correction")
        if too_bright:
            recommendations.append("Video appears overexposed - consider exposure correction")
        if not good_motion:
            recommendations.append("Low motion detected - dynamic content may perform better")
        if not scene_variety:
            recommendations.append("Limited scene variety - more cuts/transitions could improve engagement")
        
        if not recommendations:
            recommendations.append("Content quality looks good for viral potential")
        
        return recommendations
    
    async def batch_process_clips(
        self,
        input_path: str,
        clip_definitions: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple clips with progress tracking"""
        results = []
        total_clips = len(clip_definitions)
        
        for i, clip_def in enumerate(clip_definitions):
            try:
                # Generate unique output path
                clip_hash = hashlib.md5(
                    f"{input_path}_{clip_def['start_time']}_{clip_def['end_time']}".encode()
                ).hexdigest()
                output_path = os.path.join(self.output_dir, f"clip_{clip_hash}.mp4")
                
                # Process clip
                result = await self.create_video_clip(
                    input_path=input_path,
                    output_path=output_path,
                    start_time=clip_def['start_time'],
                    end_time=clip_def['end_time'],
                    settings=clip_def.get('settings')
                )
                
                result['clip_index'] = i
                result['clip_definition'] = clip_def
                results.append(result)
                
                # Progress callback
                if progress_callback:
                    await progress_callback({
                        "current": i + 1,
                        "total": total_clips,
                        "percentage": ((i + 1) / total_clips) * 100,
                        "current_clip": clip_def,
                        "result": result
                    })
                
            except Exception as e:
                logger.error(f"Error processing clip {i}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "clip_index": i,
                    "clip_definition": clip_def
                })
        
        return results
    
    async def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old temporary and output files"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for directory in [self.temp_dir, self.output_dir]:
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        filepath = os.path.join(directory, filename)
                        if os.path.isfile(filepath):
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                            if file_mtime < cutoff_time:
                                os.remove(filepath)
                                logger.info(f"Cleaned up old file: {filepath}")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and system info"""
        try:
            stats = {
                "ffmpeg_available": HAS_FFMPEG,
                "opencv_available": HAS_OPENCV,
                "supported_formats": self.supported_formats,
                "max_file_size": self.max_file_size,
                "max_duration": self.max_duration,
                "temp_dir": self.temp_dir,
                "output_dir": self.output_dir
            }
            
            # Directory statistics
            for dir_name, dir_path in [("temp", self.temp_dir), ("output", self.output_dir)]:
                if os.path.exists(dir_path):
                    files = os.listdir(dir_path)
                    total_size = sum(
                        os.path.getsize(os.path.join(dir_path, f)) 
                        for f in files if os.path.isfile(os.path.join(dir_path, f))
                    )
                    stats[f"{dir_name}_files"] = len(files)
                    stats[f"{dir_name}_size"] = total_size
                else:
                    stats[f"{dir_name}_files"] = 0
                    stats[f"{dir_name}_size"] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}
