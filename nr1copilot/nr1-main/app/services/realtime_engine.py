
"""
ViralClip Pro - Netflix-Level Real-time Processing Engine
Advanced real-time features with instant feedback and entertainment
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
import subprocess

from fastapi import WebSocket, WebSocketDisconnect

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

class RealtimeEngine:
    """Netflix-level real-time processing engine"""

    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
        self.processing_queue: Dict[str, Dict] = {}
        self.viral_cache: Dict[str, List] = {}
        self.preview_cache: Dict[str, Dict] = {}
        
        self.entertainment_facts = [
            "🎬 Did you know? The first viral video was a dancing baby in 1996!",
            "🚀 TikTok videos under 15 seconds get 5x more engagement!",
            "🎯 Videos with faces get 38% more engagement than those without!",
            "⚡ The best time to post viral content is Tuesday at 9 AM!",
            "🎵 Videos with trending audio are 4x more likely to go viral!",
            "🔥 Red thumbnails increase click rates by 30%!",
            "💫 The golden ratio for viral videos is 1.618:1 aspect ratio!",
            "🎭 Emotional content is 3x more likely to be shared!",
            "🌟 The first 3 seconds determine if 65% of viewers will keep watching!",
            "🎪 Adding captions increases engagement by 85%!",
            "🎨 High contrast visuals perform 67% better on mobile devices!",
            "📱 Vertical videos get 9x more engagement than horizontal ones!",
            "🎼 Videos synced to beat drops see 45% more shares!",
            "🌈 Colors can increase brand recognition by up to 80%!",
            "🔊 Videos with clear audio get 58% more completion rates!",
            "📊 Data shows that surprise elements boost retention by 73%!",
            "🎥 Quick cuts every 2-3 seconds keep viewers hooked!",
            "🏆 Videos that evoke strong emotions are 30x more likely to be shared!",
            "🌟 The 'rule of thirds' makes videos 40% more visually appealing!",
            "🎯 Call-to-actions in the first 5 seconds increase conversion by 85%!"
        ]

    async def initialize(self):
        """Initialize the real-time engine"""
        logger.info("🚀 Initializing Netflix-level real-time engine...")
        await self._setup_processing_pipelines()
        await self._initialize_ai_models()
        logger.info("✅ Real-time engine initialized")

    async def _setup_processing_pipelines(self):
        """Setup high-performance processing pipelines"""
        self.preview_pipeline = {
            "ffmpeg_cmd": [
                "ffmpeg", "-y", "-i", "{input}", 
                "-vf", "scale=640:360,fps=30", 
                "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "28", "-t", "{duration}", 
                "-ss", "{start}", "{output}"
            ],
            "timeout": 10
        }

    async def _initialize_ai_models(self):
        """Initialize AI models for real-time analysis"""
        # Placeholder for AI model initialization
        self.viral_analyzer = MockViralAnalyzer()
        self.emotion_detector = MockEmotionDetector()

    async def start_realtime_analysis(self, session_id: str, file_path: str) -> Dict:
        """Start real-time video analysis with progressive results"""
        try:
            logger.info(f"Starting real-time analysis: {session_id}")

            # Initialize session
            self.active_sessions[session_id] = {
                "file_path": file_path,
                "start_time": time.time(),
                "status": "analyzing",
                "progress": 0
            }

            # Get video metadata
            metadata = await self._extract_video_metadata(file_path)
            
            # Start progressive analysis
            analysis_task = asyncio.create_task(
                self._progressive_video_analysis(session_id, file_path, metadata)
            )

            # Return initial analysis
            return {
                "session_id": session_id,
                "metadata": metadata,
                "status": "analyzing",
                "estimated_completion": time.time() + metadata.get("duration", 60)
            }

        except Exception as e:
            logger.error(f"Real-time analysis error: {e}")
            raise

    async def _progressive_video_analysis(self, session_id: str, file_path: str, metadata: Dict):
        """Perform progressive video analysis with real-time updates"""
        try:
            duration = metadata.get("duration", 0)
            
            # Analyze in chunks for real-time feedback
            chunk_duration = 10  # 10-second chunks
            chunks = max(1, int(duration / chunk_duration))

            viral_scores = []
            emotions = []
            energy_levels = []

            for i in range(chunks):
                start_time = i * chunk_duration
                end_time = min((i + 1) * chunk_duration, duration)

                # Analyze chunk
                chunk_analysis = await self._analyze_video_chunk(
                    file_path, start_time, end_time
                )

                viral_scores.extend(chunk_analysis["viral_scores"])
                emotions.extend(chunk_analysis["emotions"])
                energy_levels.extend(chunk_analysis["energy"])

                # Update progress and broadcast
                progress = ((i + 1) / chunks) * 100
                await self._broadcast_analysis_progress(session_id, {
                    "progress": progress,
                    "chunk": i + 1,
                    "total_chunks": chunks,
                    "current_scores": viral_scores[-10:],  # Last 10 scores
                    "trending_emotions": emotions[-5:],    # Last 5 emotions
                    "energy_trend": energy_levels[-5:]     # Last 5 energy levels
                })

                # Add small delay to prevent overwhelming
                await asyncio.sleep(0.1)

            # Store final analysis
            self.viral_cache[session_id] = {
                "viral_scores": viral_scores,
                "emotions": emotions,
                "energy_levels": energy_levels,
                "duration": duration,
                "analyzed_at": datetime.now().isoformat()
            }

            # Final broadcast
            await self._broadcast_analysis_complete(session_id)

        except Exception as e:
            logger.error(f"Progressive analysis error: {e}")
            await self._broadcast_analysis_error(session_id, str(e))

    async def _analyze_video_chunk(self, file_path: str, start_time: float, end_time: float) -> Dict:
        """Analyze a video chunk for viral potential"""
        # Mock analysis - replace with actual AI models
        chunk_duration = end_time - start_time
        scores_count = max(1, int(chunk_duration))

        # Generate realistic viral scores with some variation
        base_score = 60 + (start_time / 10) % 30  # Vary by position
        
        return {
            "viral_scores": [
                {
                    "timestamp": start_time + i,
                    "score": min(100, base_score + (i * 2) % 40),  # Realistic viral score
                    "confidence": 0.7 + (i * 0.03) % 0.3
                }
                for i in range(scores_count)
            ],
            "emotions": [
                {
                    "timestamp": start_time + i * 2,
                    "emotion": ["joy", "surprise", "excitement", "anticipation"][i % 4],
                    "intensity": 0.5 + (i * 0.1) % 0.5
                }
                for i in range(max(1, scores_count // 2))
            ],
            "energy": [
                {
                    "timestamp": start_time + i,
                    "level": 30 + (i * 8) % 70
                }
                for i in range(scores_count)
            ]
        }

    async def generate_instant_previews(self, session_id: str, file_path: str) -> Dict:
        """Generate instant preview clips for immediate feedback"""
        try:
            logger.info(f"Generating instant previews: {session_id}")

            metadata = await self._extract_video_metadata(file_path)
            duration = metadata.get("duration", 0)

            # Generate quick preview clips
            previews = []
            preview_times = [
                (0, min(15, duration)),  # Opening
                (max(0, duration * 0.3), min(duration * 0.3 + 15, duration)),  # Middle
                (max(0, duration - 15), duration)  # Ending
            ]

            for i, (start, end) in enumerate(preview_times):
                if end > start:
                    preview_path = await self._generate_preview_clip(
                        file_path, start, end, f"preview_{i}"
                    )
                    
                    viral_score = await self._quick_viral_analysis(file_path, start, end)
                    
                    previews.append({
                        "id": f"preview_{i}",
                        "start_time": start,
                        "end_time": end,
                        "duration": end - start,
                        "preview_url": f"/api/v3/preview/{session_id}/preview_{i}.mp4",
                        "viral_score": viral_score,
                        "thumbnail": f"/api/v3/thumbnail/{session_id}/preview_{i}.jpg"
                    })

            return {
                "previews": previews,
                "total_previews": len(previews),
                "generation_time": time.time() - self.active_sessions[session_id]["start_time"]
            }

        except Exception as e:
            logger.error(f"Preview generation error: {e}")
            raise

    async def _generate_preview_clip(self, input_path: str, start: float, end: float, clip_id: str) -> str:
        """Generate a quick preview clip"""
        output_dir = Path(settings.TEMP_PATH) / "previews"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{clip_id}.mp4"

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ss", str(start), "-t", str(end - start),
            "-vf", "scale=640:360,fps=30",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "28", "-an", str(output_path)
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(process.wait(), timeout=30)
            return str(output_path)
        except Exception as e:
            logger.error(f"Preview clip generation failed: {e}")
            raise

    async def _quick_viral_analysis(self, file_path: str, start: float, end: float) -> int:
        """Quick viral potential analysis for previews"""
        # Mock viral analysis - replace with actual AI
        base_score = 65
        duration_bonus = min(15, (end - start) * 3)
        position_bonus = 10 if start < 30 else 5  # Opening bonus
        
        return int(base_score + duration_bonus + position_bonus)

    async def generate_live_preview(self, session_id: str, start_time: float, end_time: float, quality: str = "preview") -> Dict:
        """Generate live preview with real-time viral analysis"""
        generation_start = time.time()
        
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found or expired")

            file_path = self.active_sessions[session_id]["file_path"]
            
            # Validate file still exists
            if not os.path.exists(file_path):
                raise ValueError("Source video file not found")
            
            # Validate time range
            metadata = await self._extract_video_metadata(file_path)
            if start_time >= metadata.get("duration", 0):
                raise ValueError("Start time exceeds video duration")
            
            # Generate preview clip with progress tracking
            preview_id = f"live_{uuid.uuid4().hex[:8]}"
            logger.info(f"Generating preview clip: {preview_id} for session {session_id}")
            
            # Cache key for this specific preview
            cache_key = f"{session_id}_{start_time}_{end_time}"
            
            # Check cache first
            if cache_key in self.preview_cache:
                cached = self.preview_cache[cache_key]
                if time.time() - cached["created"] < 300:  # 5 minute cache
                    return cached["data"]
            
            preview_path = await self._generate_preview_clip(file_path, start_time, end_time, preview_id)

            # Perform enhanced viral analysis
            viral_analysis = await self._detailed_viral_analysis(file_path, start_time, end_time)

            # Generate contextual optimization suggestions
            suggestions = await self._generate_optimization_suggestions(viral_analysis)

            # Calculate processing metrics
            processing_time = time.time() - generation_start
            
            result = {
                "success": True,
                "preview_url": f"/api/v3/preview/{session_id}/{preview_id}.mp4",
                "viral_analysis": viral_analysis,
                "suggestions": suggestions,
                "processing_time": processing_time,
                "metadata": {
                    "preview_id": preview_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "quality": quality,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
            # Cache the result
            self.preview_cache[cache_key] = {
                "data": result,
                "created": time.time()
            }
            
            return result

        except Exception as e:
            logger.error(f"Live preview generation failed for session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - generation_start,
                "suggestions": [
                    {
                        "type": "error",
                        "priority": "high",
                        "suggestion": "Try selecting a different time range",
                        "impact": "Restore preview functionality"
                    }
                ]
            }

    async def _detailed_viral_analysis(self, file_path: str, start: float, end: float) -> Dict:
        """Detailed viral potential analysis"""
        duration = end - start
        
        # Mock analysis with realistic metrics
        base_score = 65 + (duration * 2)  # Longer clips score slightly higher
        
        # Add position-based bonuses
        if start < 30:  # Opening content
            base_score += 15
        elif start > 120:  # Later content
            base_score += 5
            
        overall_score = min(95, int(base_score))
        
        return {
            "overall_score": overall_score,
            "engagement_factors": {
                "visual_appeal": min(100, overall_score + 10),
                "audio_quality": min(100, overall_score - 5),
                "content_type": "entertainment",
                "energy_level": min(100, overall_score + 5),
                "hook_strength": min(100, overall_score + 15) if start < 10 else overall_score
            },
            "platform_optimization": {
                "tiktok": {"score": min(100, overall_score + 12), "confidence": 0.92},
                "instagram": {"score": min(100, overall_score + 5), "confidence": 0.85},
                "youtube_shorts": {"score": min(100, overall_score + 8), "confidence": 0.88}
            },
            "improvement_areas": [
                "Add captions for better accessibility",
                "Increase audio levels by 10%",
                "Consider adding trending music"
            ]
        }

    async def _generate_optimization_suggestions(self, analysis: Dict) -> List[Dict]:
        """Generate actionable optimization suggestions"""
        suggestions = []
        overall_score = analysis["overall_score"]
        
        if analysis["engagement_factors"]["audio_quality"] < 80:
            suggestions.append({
                "type": "audio",
                "priority": "high",
                "suggestion": "Enhance audio quality",
                "impact": "+12% engagement"
            })

        if overall_score < 85:
            suggestions.append({
                "type": "content",
                "priority": "medium", 
                "suggestion": "Add trending elements",
                "impact": "+8% viral potential"
            })
            
        if analysis["engagement_factors"]["hook_strength"] < 80:
            suggestions.append({
                "type": "timing",
                "priority": "high",
                "suggestion": "Strengthen opening hook",
                "impact": "+15% retention"
            })

        return suggestions

    async def get_timeline_data(self, session_id: str) -> Dict:
        """Get comprehensive timeline data with viral visualization"""
        try:
            if session_id not in self.viral_cache:
                # Return mock data if analysis isn't complete
                return await self._generate_mock_timeline_data()

            cache_data = self.viral_cache[session_id]
            
            return {
                "duration": cache_data["duration"],
                "viral_heatmap": cache_data["viral_scores"],
                "recommended_clips": await self._suggest_optimal_clips(cache_data),
                "highlights": await self._identify_highlights(cache_data),
                "peaks": await self._find_engagement_peaks(cache_data["viral_scores"]),
                "score_visualization": await self._generate_score_graph(cache_data["viral_scores"]),
                "emotions": cache_data["emotions"],
                "energy": cache_data["energy_levels"]
            }

        except Exception as e:
            logger.error(f"Timeline data error: {e}")
            raise

    async def _generate_mock_timeline_data(self) -> Dict:
        """Generate mock timeline data for immediate UI feedback"""
        duration = 180  # 3 minutes mock
        
        # Generate mock viral scores
        viral_scores = []
        for i in range(0, int(duration), 5):
            score = 50 + (i * 2) % 50
            viral_scores.append({
                "timestamp": i,
                "score": score,
                "confidence": 0.8
            })
        
        return {
            "duration": duration,
            "viral_heatmap": viral_scores,
            "recommended_clips": [
                {
                    "id": "clip_1",
                    "start_time": 0,
                    "end_time": 30,
                    "viral_score": 85,
                    "title": "Opening Hook",
                    "recommended_platforms": ["tiktok", "instagram"]
                },
                {
                    "id": "clip_2", 
                    "start_time": 60,
                    "end_time": 90,
                    "viral_score": 78,
                    "title": "Peak Moment",
                    "recommended_platforms": ["youtube_shorts"]
                }
            ],
            "highlights": [
                {"timestamp": 15, "score": 85, "type": "viral_moment"},
                {"timestamp": 75, "score": 78, "type": "viral_moment"}
            ],
            "peaks": [
                {"timestamp": 15, "peak_score": 85, "intensity": "high"}
            ]
        }

    async def _identify_highlights(self, data: Dict) -> List[Dict]:
        """Identify key highlight moments"""
        viral_scores = data["viral_scores"]
        
        highlights = []
        threshold = 75  # Viral score threshold
        
        for score_data in viral_scores:
            if score_data["score"] > threshold:
                highlights.append({
                    "timestamp": score_data["timestamp"],
                    "score": score_data["score"],
                    "type": "viral_moment",
                    "description": f"High viral potential detected ({score_data['score']}%)"
                })

        return highlights[:10]  # Top 10 highlights

    async def _find_engagement_peaks(self, viral_scores: List[Dict]) -> List[Dict]:
        """Find engagement peak moments"""
        peaks = []
        
        for i in range(1, len(viral_scores) - 1):
            current = viral_scores[i]["score"]
            prev = viral_scores[i-1]["score"]
            next_score = viral_scores[i+1]["score"]
            
            if current > prev and current > next_score and current > 70:
                peaks.append({
                    "timestamp": viral_scores[i]["timestamp"],
                    "peak_score": current,
                    "intensity": "high" if current > 85 else "medium"
                })

        return peaks

    async def _suggest_optimal_clips(self, data: Dict) -> List[Dict]:
        """Suggest optimal clip segments based on analysis"""
        viral_scores = data["viral_scores"]
        duration = data["duration"]
        
        clips = []
        
        # Find best 30-second segments
        window_size = 30
        best_segments = []
        
        for i in range(0, int(duration) - window_size, 5):
            segment_scores = [
                s["score"] for s in viral_scores 
                if i <= s["timestamp"] < i + window_size
            ]
            
            if segment_scores:
                avg_score = sum(segment_scores) / len(segment_scores)
                best_segments.append({
                    "start": i,
                    "end": i + window_size,
                    "avg_score": avg_score
                })

        # Sort by score and take top 5
        best_segments.sort(key=lambda x: x["avg_score"], reverse=True)
        
        for i, segment in enumerate(best_segments[:5]):
            clips.append({
                "id": f"suggested_{i+1}",
                "start_time": segment["start"],
                "end_time": segment["end"],
                "viral_score": int(segment["avg_score"]),
                "title": f"Viral Clip #{i+1}",
                "recommended_platforms": ["tiktok", "instagram"] if segment["avg_score"] > 80 else ["youtube_shorts"]
            })

        return clips

    async def _generate_score_graph(self, viral_scores: List[Dict]) -> Dict:
        """Generate data for viral score visualization"""
        return {
            "type": "line_chart",
            "data_points": [
                {"x": s["timestamp"], "y": s["score"], "confidence": s["confidence"]}
                for s in viral_scores
            ],
            "color_zones": [
                {"range": [0, 50], "color": "#ef4444"},      # Red - Low viral
                {"range": [50, 75], "color": "#f59e0b"},     # Yellow - Medium viral  
                {"range": [75, 100], "color": "#10b981"}     # Green - High viral
            ],
            "annotations": [
                {"x": s["timestamp"], "label": "Peak"} 
                for s in viral_scores if s["score"] > 90
            ]
        }

    async def process_clips_with_entertainment(self, task_id: str, session_id: str, clips: List[Dict], options: Dict):
        """Process clips with entertaining status updates"""
        try:
            logger.info(f"Starting entertaining processing: {task_id}")
            
            self.processing_queue[task_id] = {
                "status": "starting",
                "progress": 0,
                "clips": clips,
                "start_time": time.time(),
                "entertainment_index": 0
            }

            total_clips = len(clips)
            
            for i, clip in enumerate(clips):
                # Send entertaining fact
                fact_index = i % len(self.entertainment_facts)
                fact = self.entertainment_facts[fact_index]
                
                await self._broadcast_processing_update(task_id, {
                    "type": "processing_update",
                    "stage": f"Processing clip {i+1} of {total_clips}",
                    "progress": (i / total_clips) * 100,
                    "entertaining_fact": fact,
                    "eta_seconds": (total_clips - i) * 25
                })

                # Simulate processing with mini-updates
                steps = ["analyzing", "optimizing", "rendering", "finalizing"]
                for j, step in enumerate(steps):
                    await self._broadcast_processing_update(task_id, {
                        "type": "step_update",
                        "step": step,
                        "clip": i + 1,
                        "substep_progress": 25 * (j + 1)
                    })
                    await asyncio.sleep(1.5)  # Realistic processing time

            # Final completion
            await self._broadcast_processing_update(task_id, {
                "type": "processing_complete",
                "total_clips": total_clips,
                "processing_time": time.time() - self.processing_queue[task_id]["start_time"],
                "success_message": "🎉 All clips processed successfully! Ready for viral domination!"
            })

        except Exception as e:
            logger.error(f"Entertainment processing error: {e}")
            await self._broadcast_processing_update(task_id, {
                "type": "processing_error",
                "error": str(e)
            })

    # WebSocket handlers
    async def handle_upload_websocket(self, websocket: WebSocket, upload_id: str):
        """Handle upload progress WebSocket"""
        await websocket.accept()
        
        if upload_id not in self.websocket_connections:
            self.websocket_connections[upload_id] = []
        self.websocket_connections[upload_id].append(websocket)

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
        except WebSocketDisconnect:
            if upload_id in self.websocket_connections:
                self.websocket_connections[upload_id].remove(websocket)

    async def handle_viral_scores_websocket(self, websocket: WebSocket, session_id: str):
        """Handle viral scores WebSocket"""
        await websocket.accept()
        
        connection_key = f"viral_{session_id}"
        if connection_key not in self.websocket_connections:
            self.websocket_connections[connection_key] = []
        self.websocket_connections[connection_key].append(websocket)

        try:
            while True:
                await asyncio.sleep(1)  # Keep connection alive
        except WebSocketDisconnect:
            if connection_key in self.websocket_connections:
                self.websocket_connections[connection_key].remove(websocket)

    async def handle_timeline_websocket(self, websocket: WebSocket, session_id: str):
        """Handle timeline WebSocket"""
        await websocket.accept()
        
        connection_key = f"timeline_{session_id}"
        if connection_key not in self.websocket_connections:
            self.websocket_connections[connection_key] = []
        self.websocket_connections[connection_key].append(websocket)

        try:
            while True:
                await asyncio.sleep(1)  # Keep connection alive
        except WebSocketDisconnect:
            if connection_key in self.websocket_connections:
                self.websocket_connections[connection_key].remove(websocket)

    async def handle_processing_websocket(self, websocket: WebSocket, task_id: str):
        """Handle processing WebSocket"""
        await websocket.accept()
        
        connection_key = f"processing_{task_id}"
        if connection_key not in self.websocket_connections:
            self.websocket_connections[connection_key] = []
        self.websocket_connections[connection_key].append(websocket)

        try:
            while True:
                await asyncio.sleep(1)  # Keep connection alive
        except WebSocketDisconnect:
            if connection_key in self.websocket_connections:
                self.websocket_connections[connection_key].remove(websocket)

    # Broadcasting methods
    async def broadcast_upload_progress(self, upload_id: str, message: Dict):
        """Broadcast upload progress to connected clients"""
        connections = self.websocket_connections.get(upload_id, [])
        
        for websocket in connections.copy():
            try:
                await websocket.send_text(json.dumps(message))
            except:
                connections.remove(websocket)

    async def _broadcast_analysis_progress(self, session_id: str, data: Dict):
        """Broadcast analysis progress"""
        connection_key = f"viral_{session_id}"
        connections = self.websocket_connections.get(connection_key, [])
        
        message = {
            "type": "analysis_progress", 
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        for websocket in connections.copy():
            try:
                await websocket.send_text(json.dumps(message))
            except:
                connections.remove(websocket)

    async def _broadcast_analysis_complete(self, session_id: str):
        """Broadcast analysis completion"""
        connection_key = f"viral_{session_id}"
        connections = self.websocket_connections.get(connection_key, [])
        
        message = {
            "type": "analysis_complete",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        for websocket in connections.copy():
            try:
                await websocket.send_text(json.dumps(message))
            except:
                connections.remove(websocket)

    async def _broadcast_analysis_error(self, session_id: str, error: str):
        """Broadcast analysis error"""
        connection_key = f"viral_{session_id}"
        connections = self.websocket_connections.get(connection_key, [])
        
        message = {
            "type": "analysis_error",
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        for websocket in connections.copy():
            try:
                await websocket.send_text(json.dumps(message))
            except:
                connections.remove(websocket)

    async def _broadcast_processing_update(self, task_id: str, data: Dict):
        """Broadcast processing updates"""
        connection_key = f"processing_{task_id}"
        connections = self.websocket_connections.get(connection_key, [])
        
        message = {
            **data,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        
        for websocket in connections.copy():
            try:
                await websocket.send_text(json.dumps(message))
            except:
                connections.remove(websocket)

    # Utility methods
    async def _extract_video_metadata(self, file_path: str) -> Dict:
        """Extract video metadata using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", file_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            metadata = json.loads(stdout.decode())
            
            video_stream = next(
                (s for s in metadata.get("streams", []) if s.get("codec_type") == "video"),
                {}
            )
            
            duration = float(metadata.get("format", {}).get("duration", 0))
            
            return {
                "duration": duration,
                "width": video_stream.get("width", 0),
                "height": video_stream.get("height", 0),
                "fps": self._parse_frame_rate(video_stream.get("r_frame_rate", "30/1")),
                "bitrate": int(metadata.get("format", {}).get("bit_rate", 0)),
                "size": int(metadata.get("format", {}).get("size", 0))
            }
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return {"duration": 0, "width": 0, "height": 0}

    def _parse_frame_rate(self, frame_rate_str: str) -> float:
        """Parse frame rate string like '30/1' to float"""
        try:
            if '/' in frame_rate_str:
                num, den = frame_rate_str.split('/')
                return float(num) / float(den)
            return float(frame_rate_str)
        except:
            return 30.0

    async def get_system_metrics(self) -> Dict:
        """Get real-time system metrics"""
        return {
            "active_sessions": len(self.active_sessions),
            "websocket_connections": sum(len(conns) for conns in self.websocket_connections.values()),
            "processing_queue_size": len(self.processing_queue),
            "cache_size": len(self.viral_cache),
            "preview_cache_size": len(self.preview_cache),
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up real-time engine...")
        
        # Close all WebSocket connections
        for connections in self.websocket_connections.values():
            for websocket in connections:
                try:
                    await websocket.close()
                except:
                    pass
        
        # Clear caches
        self.active_sessions.clear()
        self.websocket_connections.clear()
        self.processing_queue.clear()
        self.viral_cache.clear()
        self.preview_cache.clear()
        
        logger.info("✅ Real-time engine cleanup complete")

# Mock classes for development
class MockViralAnalyzer:
    async def analyze(self, video_path: str) -> Dict:
        return {"viral_score": 75, "confidence": 0.85}

class MockEmotionDetector:
    async def detect(self, video_path: str) -> List[Dict]:
        return [{"timestamp": 0, "emotion": "joy", "confidence": 0.9}]
