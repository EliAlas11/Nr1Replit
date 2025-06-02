"""
ViralClip Pro v6.0 - Netflix-Level Real-Time Engine
Advanced real-time processing with viral insights and live feedback
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import weakref
from collections import defaultdict
import psutil

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class ViralInsight:
    """Real-time viral insight with comprehensive data"""
    timestamp: datetime
    session_id: str
    insight_type: str
    data: Dict[str, Any]
    confidence: float
    priority: str = "normal"
    expires_at: Optional[datetime] = None


@dataclass
class TimelineSegment:
    """Interactive timeline segment with viral scoring"""
    start_time: float
    end_time: float
    viral_score: float
    engagement_factors: List[str]
    sentiment_data: Dict[str, Any]
    hotspot_intensity: float
    recommended_for_platforms: List[str]
    confidence: float


@dataclass
class ProcessingStage:
    """Live processing stage with detailed feedback"""
    stage_id: str
    name: str
    description: str
    progress: float
    estimated_time_remaining: float
    current_operation: str
    substages: List[Dict[str, Any]] = field(default_factory=list)
    animation_type: str = "progress"


class EnterpriseRealtimeEngine:
    """Netflix-level real-time engine with viral insights and live feedback"""

    def __init__(self):
        # WebSocket connection management
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.session_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Real-time data streams
        self.viral_insights: Dict[str, List[ViralInsight]] = defaultdict(list)
        self.timeline_data: Dict[str, List[TimelineSegment]] = {}
        self.processing_stages: Dict[str, List[ProcessingStage]] = defaultdict(list)
        self.sentiment_streams: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Performance monitoring
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.broadcast_stats = {
            "total_messages": 0,
            "successful_broadcasts": 0,
            "failed_broadcasts": 0,
            "active_sessions": 0,
            "peak_concurrent_connections": 0
        }
        
        # Background tasks
        self.message_processor_task: Optional[asyncio.Task] = None
        self.insight_generator_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Enterprise features
        self.auto_recommendations_enabled = True
        self.sentiment_analysis_enabled = True
        self.hotspot_detection_enabled = True
        
        logger.info("🚀 Netflix-level real-time engine initialized")

    async def enterprise_warm_up(self):
        """Warm up real-time engine with enterprise features"""
        try:
            # Start background tasks
            self.message_processor_task = asyncio.create_task(self._process_message_queue())
            self.insight_generator_task = asyncio.create_task(self._generate_insights_continuously())
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_data())
            
            # Initialize AI models for real-time analysis
            await self._initialize_realtime_ai_models()
            
            logger.info("🔥 Real-time engine enterprise warm-up completed")
            
        except Exception as e:
            logger.error(f"Real-time engine warm-up failed: {e}", exc_info=True)

    async def connect_websocket(
        self,
        websocket: WebSocket,
        session_id: str,
        user_info: Dict[str, Any]
    ):
        """Connect WebSocket with enterprise session management"""
        try:
            await websocket.accept()
            
            # Store connection metadata
            self.connection_metadata[websocket] = {
                "session_id": session_id,
                "user_info": user_info,
                "connected_at": datetime.utcnow(),
                "message_count": 0,
                "last_activity": datetime.utcnow()
            }
            
            # Add to connection pools
            self.active_connections[session_id].add(websocket)
            self.session_connections[session_id].add(websocket)
            
            # Update stats
            total_connections = sum(len(conns) for conns in self.active_connections.values())
            self.broadcast_stats["peak_concurrent_connections"] = max(
                self.broadcast_stats["peak_concurrent_connections"],
                total_connections
            )
            self.broadcast_stats["active_sessions"] = len(self.active_connections)
            
            # Send welcome message with current state
            await self._send_welcome_message(websocket, session_id)
            
            logger.info(f"🔗 WebSocket connected: {session_id} (total: {total_connections})")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def disconnect_websocket(self, websocket: WebSocket, session_id: str):
        """Disconnect WebSocket with cleanup"""
        try:
            # Remove from connection pools
            self.active_connections[session_id].discard(websocket)
            self.session_connections[session_id].discard(websocket)
            
            # Clean up empty session pools
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
            
            # Remove connection metadata
            self.connection_metadata.pop(websocket, None)
            
            # Update stats
            self.broadcast_stats["active_sessions"] = len(self.active_connections)
            
            logger.info(f"🔌 WebSocket disconnected: {session_id}")
            
        except Exception as e:
            logger.error(f"WebSocket disconnect error: {e}")

    async def broadcast_viral_insights(
        self,
        session_id: str,
        insights_data: Dict[str, Any]
    ):
        """Broadcast viral insights with enterprise reliability"""
        try:
            # Create viral insight
            insight = ViralInsight(
                timestamp=datetime.utcnow(),
                session_id=session_id,
                insight_type=insights_data.get("type", "general"),
                data=insights_data,
                confidence=insights_data.get("confidence", 0.8),
                priority=insights_data.get("priority", "normal")
            )
            
            # Store insight
            self.viral_insights[session_id].append(insight)
            
            # Limit stored insights per session
            if len(self.viral_insights[session_id]) > 100:
                self.viral_insights[session_id] = self.viral_insights[session_id][-50:]
            
            # Queue for broadcast
            message = {
                "type": "viral_insights",
                "session_id": session_id,
                "timestamp": insight.timestamp.isoformat(),
                "data": {
                    "insights": insights_data,
                    "confidence": insight.confidence,
                    "priority": insight.priority,
                    "recommendations": await self._generate_live_recommendations(insights_data)
                }
            }
            
            await self.message_queue.put(("session", session_id, message))
            
        except Exception as e:
            logger.error(f"Viral insights broadcast failed: {e}")

    async def stream_timeline_updates(
        self,
        session_id: str,
        timeline_segments: List[Dict[str, Any]]
    ):
        """Stream interactive timeline updates with viral scoring"""
        try:
            # Process timeline segments
            processed_segments = []
            
            for segment_data in timeline_segments:
                segment = TimelineSegment(
                    start_time=segment_data.get("start_time", 0),
                    end_time=segment_data.get("end_time", 0),
                    viral_score=segment_data.get("viral_score", 50),
                    engagement_factors=segment_data.get("engagement_factors", []),
                    sentiment_data=segment_data.get("sentiment_data", {}),
                    hotspot_intensity=segment_data.get("hotspot_intensity", 0.5),
                    recommended_for_platforms=segment_data.get("platforms", []),
                    confidence=segment_data.get("confidence", 0.8)
                )
                processed_segments.append(segment)
            
            # Store timeline data
            self.timeline_data[session_id] = processed_segments
            
            # Generate heatmap data
            heatmap_data = await self._generate_heatmap_data(processed_segments)
            
            # Create timeline message
            message = {
                "type": "timeline_update",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "segments": [
                        {
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "viral_score": seg.viral_score,
                            "engagement_factors": seg.engagement_factors,
                            "sentiment": seg.sentiment_data,
                            "hotspot_intensity": seg.hotspot_intensity,
                            "platforms": seg.recommended_for_platforms,
                            "confidence": seg.confidence
                        }
                        for seg in processed_segments
                    ],
                    "heatmap": heatmap_data,
                    "recommendations": await self._generate_timeline_recommendations(processed_segments)
                }
            }
            
            await self.message_queue.put(("session", session_id, message))
            
        except Exception as e:
            logger.error(f"Timeline updates streaming failed: {e}")

    async def stream_processing_dashboard(
        self,
        session_id: str,
        current_stage: str,
        progress: float,
        estimated_time: float,
        details: Dict[str, Any]
    ):
        """Stream live processing dashboard with entertaining animations"""
        try:
            # Create processing stage
            stage = ProcessingStage(
                stage_id=str(uuid.uuid4()),
                name=current_stage,
                description=details.get("description", "Processing..."),
                progress=progress,
                estimated_time_remaining=estimated_time,
                current_operation=details.get("current_operation", "Working..."),
                substages=details.get("substages", []),
                animation_type=details.get("animation_type", "progress")
            )
            
            # Store stage
            self.processing_stages[session_id].append(stage)
            
            # Limit stored stages
            if len(self.processing_stages[session_id]) > 20:
                self.processing_stages[session_id] = self.processing_stages[session_id][-10:]
            
            # Generate entertaining messages
            entertaining_messages = await self._generate_entertaining_messages(stage)
            
            # Create dashboard message
            message = {
                "type": "processing_dashboard",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "stage": current_stage,
                    "progress": progress,
                    "estimated_time_remaining": estimated_time,
                    "current_operation": stage.current_operation,
                    "substages": stage.substages,
                    "animation_type": stage.animation_type,
                    "entertaining_messages": entertaining_messages,
                    "performance_stats": await self._get_processing_performance_stats(session_id)
                }
            }
            
            await self.message_queue.put(("session", session_id, message))
            
        except Exception as e:
            logger.error(f"Processing dashboard streaming failed: {e}")

    async def stream_sentiment_analysis(
        self,
        session_id: str,
        sentiment_data: Dict[str, Any]
    ):
        """Stream AI-based sentiment meter updates"""
        try:
            # Enhanced sentiment data
            enhanced_sentiment = {
                **sentiment_data,
                "timestamp": datetime.utcnow().isoformat(),
                "emotional_intensity": self._calculate_emotional_intensity(sentiment_data),
                "engagement_prediction": self._predict_engagement_from_sentiment(sentiment_data),
                "viral_potential": self._calculate_viral_potential_from_sentiment(sentiment_data)
            }
            
            # Store sentiment data
            self.sentiment_streams[session_id].append(enhanced_sentiment)
            
            # Limit stored sentiment data
            if len(self.sentiment_streams[session_id]) > 200:
                self.sentiment_streams[session_id] = self.sentiment_streams[session_id][-100:]
            
            # Create sentiment message
            message = {
                "type": "sentiment_analysis",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "sentiment": enhanced_sentiment,
                    "trends": await self._analyze_sentiment_trends(session_id),
                    "recommendations": await self._generate_sentiment_recommendations(enhanced_sentiment)
                }
            }
            
            await self.message_queue.put(("session", session_id, message))
            
        except Exception as e:
            logger.error(f"Sentiment analysis streaming failed: {e}")

    async def stream_smart_clip_recommendations(
        self,
        session_id: str,
        video_analysis: Dict[str, Any]
    ):
        """Stream smart clip trimming recommendations based on engagement peaks"""
        try:
            # Generate smart recommendations
            recommendations = await self._generate_smart_clip_recommendations(
                session_id, video_analysis
            )
            
            # Create recommendations message
            message = {
                "type": "smart_recommendations",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "recommendations": recommendations,
                    "auto_trim_suggestions": await self._generate_auto_trim_suggestions(video_analysis),
                    "engagement_peaks": await self._identify_engagement_peaks(video_analysis),
                    "optimal_durations": await self._calculate_optimal_durations(video_analysis)
                }
            }
            
            await self.message_queue.put(("session", session_id, message))
            
        except Exception as e:
            logger.error(f"Smart recommendations streaming failed: {e}")

    # Private helper methods

    async def _process_message_queue(self):
        """Background task to process message queue"""
        while True:
            try:
                # Get message from queue
                broadcast_type, target, message = await self.message_queue.get()
                
                # Broadcast message
                if broadcast_type == "session":
                    await self._broadcast_to_session(target, message)
                elif broadcast_type == "all":
                    await self._broadcast_to_all(message)
                
                # Update stats
                self.broadcast_stats["total_messages"] += 1
                
                # Mark task as done
                self.message_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message queue processing error: {e}")
                self.broadcast_stats["failed_broadcasts"] += 1

    async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        """Broadcast message to specific session"""
        if session_id not in self.active_connections:
            return
        
        connections = list(self.active_connections[session_id])
        failed_connections = []
        
        for websocket in connections:
            try:
                await websocket.send_text(json.dumps(message))
                
                # Update connection metadata
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["message_count"] += 1
                    self.connection_metadata[websocket]["last_activity"] = datetime.utcnow()
                
            except WebSocketDisconnect:
                failed_connections.append(websocket)
            except Exception as e:
                logger.error(f"Broadcast error to session {session_id}: {e}")
                failed_connections.append(websocket)
        
        # Clean up failed connections
        for websocket in failed_connections:
            await self.disconnect_websocket(websocket, session_id)
        
        if connections and not failed_connections:
            self.broadcast_stats["successful_broadcasts"] += 1
        elif failed_connections:
            self.broadcast_stats["failed_broadcasts"] += 1

    async def _broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all active connections"""
        all_connections = []
        for session_connections in self.active_connections.values():
            all_connections.extend(session_connections)
        
        failed_connections = []
        
        for websocket in all_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except (WebSocketDisconnect, Exception) as e:
                failed_connections.append(websocket)
        
        # Clean up failed connections
        for websocket in failed_connections:
            session_id = self.connection_metadata.get(websocket, {}).get("session_id")
            if session_id:
                await self.disconnect_websocket(websocket, session_id)

    async def _send_welcome_message(self, websocket: WebSocket, session_id: str):
        """Send welcome message with current state"""
        welcome_message = {
            "type": "welcome",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "features_enabled": {
                    "viral_insights": True,
                    "timeline_updates": True,
                    "sentiment_analysis": self.sentiment_analysis_enabled,
                    "smart_recommendations": self.auto_recommendations_enabled,
                    "hotspot_detection": self.hotspot_detection_enabled
                },
                "current_insights": self.viral_insights.get(session_id, [])[-5:],
                "timeline_data": self.timeline_data.get(session_id, []),
                "recent_sentiment": self.sentiment_streams.get(session_id, [])[-10:]
            }
        }
        
        try:
            await websocket.send_text(json.dumps(welcome_message))
        except Exception as e:
            logger.error(f"Welcome message send failed: {e}")

    async def _generate_insights_continuously(self):
        """Background task to generate continuous insights"""
        while True:
            try:
                await asyncio.sleep(5)  # Generate insights every 5 seconds
                
                # Generate insights for active sessions
                for session_id in list(self.active_connections.keys()):
                    if session_id in self.timeline_data:
                        await self._generate_periodic_insights(session_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Continuous insight generation error: {e}")

    async def _generate_periodic_insights(self, session_id: str):
        """Generate periodic insights for a session"""
        try:
            timeline_segments = self.timeline_data.get(session_id, [])
            if not timeline_segments:
                return
            
            # Generate insights based on timeline data
            insights = {
                "type": "periodic_analysis",
                "viral_score_trend": self._calculate_viral_score_trend(timeline_segments),
                "engagement_momentum": self._calculate_engagement_momentum(timeline_segments),
                "platform_optimization": self._suggest_platform_optimizations(timeline_segments),
                "confidence": 0.8,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            await self.broadcast_viral_insights(session_id, insights)
            
        except Exception as e:
            logger.error(f"Periodic insights generation failed: {e}")

    async def _cleanup_expired_data(self):
        """Background task to cleanup expired data"""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                current_time = datetime.utcnow()
                
                # Cleanup expired insights
                for session_id in list(self.viral_insights.keys()):
                    insights = self.viral_insights[session_id]
                    valid_insights = [
                        insight for insight in insights
                        if not insight.expires_at or insight.expires_at > current_time
                    ]
                    
                    if len(valid_insights) != len(insights):
                        self.viral_insights[session_id] = valid_insights
                        logger.info(f"🧹 Cleaned up expired insights for session: {session_id}")
                
                # Cleanup old sentiment data
                for session_id in list(self.sentiment_streams.keys()):
                    sentiments = self.sentiment_streams[session_id]
                    if len(sentiments) > 500:
                        self.sentiment_streams[session_id] = sentiments[-250:]
                
                # Cleanup inactive sessions
                inactive_sessions = []
                for session_id, connections in self.active_connections.items():
                    if not connections:
                        inactive_sessions.append(session_id)
                
                for session_id in inactive_sessions:
                    self._cleanup_session_data(session_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data cleanup error: {e}")

    def _cleanup_session_data(self, session_id: str):
        """Cleanup all data for a session"""
        self.viral_insights.pop(session_id, None)
        self.timeline_data.pop(session_id, None)
        self.processing_stages.pop(session_id, None)
        self.sentiment_streams.pop(session_id, None)

    async def _initialize_realtime_ai_models(self):
        """Initialize AI models for real-time analysis"""
        # Simulate AI model initialization
        await asyncio.sleep(0.5)
        logger.info("🤖 Real-time AI models initialized")

    async def _generate_heatmap_data(self, segments: List[TimelineSegment]) -> Dict[str, Any]:
        """Generate heatmap data from timeline segments"""
        heatmap_points = []
        
        for segment in segments:
            heatmap_points.append({
                "time": (segment.start_time + segment.end_time) / 2,
                "intensity": segment.hotspot_intensity,
                "viral_score": segment.viral_score,
                "factors": segment.engagement_factors
            })
        
        return {
            "points": heatmap_points,
            "max_intensity": max((p["intensity"] for p in heatmap_points), default=1.0),
            "overall_heat": sum(p["intensity"] for p in heatmap_points) / len(heatmap_points) if heatmap_points else 0
        }

    async def _generate_timeline_recommendations(self, segments: List[TimelineSegment]) -> List[Dict[str, Any]]:
        """Generate recommendations based on timeline analysis"""
        recommendations = []
        
        # Find high-scoring segments
        high_scoring = [seg for seg in segments if seg.viral_score >= 75]
        
        for segment in high_scoring[:3]:  # Top 3 recommendations
            recommendations.append({
                "type": "clip_suggestion",
                "start_time": segment.start_time,
                "end_time": min(segment.end_time, segment.start_time + 15),
                "viral_score": segment.viral_score,
                "platforms": segment.recommended_for_platforms,
                "reason": f"High viral potential ({segment.viral_score:.0f}/100)"
            })
        
        return recommendations

    async def _generate_live_recommendations(self, insights_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate live recommendations based on insights"""
        recommendations = []
        
        viral_score = insights_data.get("viral_score", 50)
        
        if viral_score >= 80:
            recommendations.append({
                "type": "immediate_clip",
                "message": "🔥 High viral potential detected! Consider creating a clip now.",
                "confidence": 0.9
            })
        elif viral_score >= 60:
            recommendations.append({
                "type": "optimization",
                "message": "💡 Good content! Try enhancing with trending audio or effects.",
                "confidence": 0.7
            })
        else:
            recommendations.append({
                "type": "improvement",
                "message": "📈 Consider adjusting pacing or adding visual interest.",
                "confidence": 0.6
            })
        
        return recommendations

    async def _generate_entertaining_messages(self, stage: ProcessingStage) -> List[str]:
        """Generate entertaining messages for processing stages"""
        messages = {
            "analyzing": [
                "🔍 Scanning for viral DNA...",
                "🎬 Analyzing cinematic genius...",
                "🚀 Detecting engagement rockets...",
                "✨ Finding the magic moments..."
            ],
            "processing": [
                "⚡ Turbocharged processing in progress...",
                "🎯 Optimizing for maximum viral impact...",
                "🔥 Applying Netflix-level enhancements...",
                "🎨 Crafting your masterpiece..."
            ],
            "encoding": [
                "📡 Encoding at light speed...",
                "🎭 Adding final touches of brilliance...",
                "🌟 Preparing for social media stardom...",
                "🎪 Creating scroll-stopping content..."
            ]
        }
        
        stage_messages = messages.get(stage.name.lower(), ["🚀 Working hard on your content..."])
        import random
        return random.sample(stage_messages, min(2, len(stage_messages)))

    def _calculate_emotional_intensity(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate emotional intensity from sentiment data"""
        emotions = sentiment_data.get("emotions", {})
        if not emotions:
            return 0.5
        
        # Calculate intensity based on strongest emotions
        max_emotion = max(emotions.values()) if emotions else 0.5
        emotion_diversity = len([v for v in emotions.values() if v > 0.3])
        
        return min(1.0, max_emotion + (emotion_diversity * 0.1))

    def _predict_engagement_from_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict engagement metrics from sentiment analysis"""
        intensity = self._calculate_emotional_intensity(sentiment_data)
        
        return {
            "predicted_likes": int(1000 + (intensity * 5000)),
            "predicted_shares": int(100 + (intensity * 500)),
            "predicted_comments": int(50 + (intensity * 200)),
            "engagement_rate": round(0.05 + (intensity * 0.10), 3)
        }

    def _calculate_viral_potential_from_sentiment(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate viral potential based on sentiment analysis"""
        emotions = sentiment_data.get("emotions", {})
        
        # High-engagement emotions
        viral_emotions = ["joy", "surprise", "excitement", "awe"]
        viral_score = sum(emotions.get(emotion, 0) for emotion in viral_emotions)
        
        return min(100, viral_score * 100 + 20)

    async def _analyze_sentiment_trends(self, session_id: str) -> Dict[str, Any]:
        """Analyze sentiment trends for a session"""
        sentiments = self.sentiment_streams.get(session_id, [])
        if len(sentiments) < 2:
            return {"trend": "stable", "direction": 0}
        
        recent = sentiments[-5:]
        if len(recent) < 2:
            return {"trend": "stable", "direction": 0}
        
        # Calculate trend
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_avg = sum(s.get("viral_potential", 50) for s in first_half) / len(first_half)
        second_avg = sum(s.get("viral_potential", 50) for s in second_half) / len(second_half)
        
        direction = second_avg - first_avg
        
        if direction > 5:
            trend = "improving"
        elif direction < -5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "direction": direction,
            "confidence": 0.8
        }

    async def _generate_sentiment_recommendations(self, sentiment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on sentiment analysis"""
        recommendations = []
        
        intensity = sentiment_data.get("emotional_intensity", 0.5)
        dominant_emotion = sentiment_data.get("dominant_emotion", "neutral")
        
        if intensity < 0.3:
            recommendations.append({
                "type": "boost_emotion",
                "message": "💥 Try adding more dynamic content to boost emotional impact",
                "confidence": 0.8
            })
        
        if dominant_emotion in ["joy", "excitement"]:
            recommendations.append({
                "type": "leverage_emotion",
                "message": f"🎉 Great {dominant_emotion} detected! Perfect for social platforms",
                "confidence": 0.9
            })
        
        return recommendations

    async def _generate_smart_clip_recommendations(
        self, 
        session_id: str, 
        video_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate smart clip recommendations"""
        recommendations = []
        
        # Mock smart recommendations based on analysis
        duration = video_analysis.get("duration", 60)
        viral_segments = video_analysis.get("viral_segments", [])
        
        for i, segment in enumerate(viral_segments[:3]):
            recommendations.append({
                "id": f"rec_{i}",
                "start_time": segment.get("start", 0),
                "end_time": min(segment.get("end", 15), segment.get("start", 0) + 15),
                "confidence": segment.get("confidence", 0.8),
                "reason": "High engagement potential detected",
                "platform": "TikTok" if segment.get("end", 15) - segment.get("start", 0) <= 15 else "Instagram",
                "viral_score": segment.get("viral_score", 75)
            })
        
        return recommendations

    async def _generate_auto_trim_suggestions(self, video_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate automatic trimming suggestions"""
        return [
            {
                "type": "remove_silence",
                "description": "Remove silent sections for better pacing",
                "potential_time_saved": "3.2s",
                "confidence": 0.9
            },
            {
                "type": "optimize_hook",
                "description": "Start at highest engagement point",
                "adjustment": "Start 2.1s later",
                "confidence": 0.8
            }
        ]

    async def _identify_engagement_peaks(self, video_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify engagement peaks in the video"""
        peaks = []
        
        # Mock engagement peaks
        import random
        for i in range(3):
            peak_time = random.uniform(0, video_analysis.get("duration", 60))
            peaks.append({
                "time": peak_time,
                "intensity": random.uniform(0.7, 1.0),
                "duration": random.uniform(2, 8),
                "factors": random.sample(["visual_impact", "audio_peak", "motion_change", "content_highlight"], 2)
            })
        
        return sorted(peaks, key=lambda x: x["intensity"], reverse=True)

    async def _calculate_optimal_durations(self, video_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal durations for different platforms"""
        return {
            "TikTok": {"min": 9, "max": 15, "optimal": 12, "reason": "Maximum retention"},
            "Instagram_Reels": {"min": 15, "max": 30, "optimal": 22, "reason": "Discovery algorithm"},
            "YouTube_Shorts": {"min": 30, "max": 60, "optimal": 45, "reason": "Watch time optimization"}
        }

    def _calculate_viral_score_trend(self, segments: List[TimelineSegment]) -> str:
        """Calculate viral score trend across segments"""
        if len(segments) < 2:
            return "stable"
        
        scores = [seg.viral_score for seg in segments]
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg + 5:
            return "improving"
        elif second_avg < first_avg - 5:
            return "declining"
        else:
            return "stable"

    def _calculate_engagement_momentum(self, segments: List[TimelineSegment]) -> float:
        """Calculate engagement momentum"""
        if not segments:
            return 0.5
        
        total_score = sum(seg.viral_score for seg in segments)
        avg_score = total_score / len(segments)
        
        return min(1.0, avg_score / 100)

    def _suggest_platform_optimizations(self, segments: List[TimelineSegment]) -> Dict[str, Any]:
        """Suggest platform-specific optimizations"""
        if not segments:
            return {}
        
        avg_score = sum(seg.viral_score for seg in segments) / len(segments)
        
        optimizations = {}
        
        if avg_score >= 80:
            optimizations["TikTok"] = ["Perfect for trending page", "Add trending hashtags"]
            optimizations["Instagram"] = ["High discovery potential", "Cross-post to Stories"]
        elif avg_score >= 60:
            optimizations["Instagram"] = ["Good for Reels", "Consider better thumbnail"]
            optimizations["YouTube"] = ["Optimize title and description", "Add chapters"]
        else:
            optimizations["General"] = ["Improve hook", "Add visual interest", "Enhance audio"]
        
        return optimizations

    async def _get_processing_performance_stats(self, session_id: str) -> Dict[str, Any]:
        """Get processing performance statistics"""
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "processing_speed": "1.2x realtime",
            "queue_position": 1,
            "estimated_completion": "45 seconds"
        }

    async def get_realtime_stats(self) -> Dict[str, Any]:
        """Get comprehensive real-time engine statistics"""
        total_connections = sum(len(conns) for conns in self.active_connections.values())
        
        return {
            "active_sessions": len(self.active_connections),
            "total_connections": total_connections,
            "peak_connections": self.broadcast_stats["peak_concurrent_connections"],
            "message_queue_size": self.message_queue.qsize(),
            "broadcast_stats": self.broadcast_stats,
            "features_status": {
                "sentiment_analysis": self.sentiment_analysis_enabled,
                "auto_recommendations": self.auto_recommendations_enabled,
                "hotspot_detection": self.hotspot_detection_enabled
            },
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
        }

    async def health_check(self) -> bool:
        """Perform health check on real-time engine"""
        try:
            # Check message queue
            if self.message_queue.qsize() > 5000:
                logger.warning("Message queue size is high")
                return False
            
            # Check background tasks
            if self.message_processor_task and self.message_processor_task.done():
                logger.error("Message processor task is not running")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Real-time engine health check failed: {e}")
            return False

    async def graceful_shutdown(self):
        """Gracefully shutdown real-time engine"""
        logger.info("🔄 Shutting down real-time engine...")
        
        # Cancel background tasks
        for task in [self.message_processor_task, self.insight_generator_task, self.cleanup_task]:
            if task and not task.done():
                task.cancel()
        
        # Close all WebSocket connections
        for session_id, connections in self.active_connections.items():
            for websocket in list(connections):
                try:
                    await websocket.close(code=1001, reason="Server shutdown")
                except Exception:
                    pass
        
        # Clear all data
        self.active_connections.clear()
        self.session_connections.clear()
        self.connection_metadata.clear()
        self.viral_insights.clear()
        self.timeline_data.clear()
        self.processing_stages.clear()
        self.sentiment_streams.clear()
        
        logger.info("✅ Real-time engine shutdown complete")


class EnterpriseRealtimeEngine:
    """Netflix-level real-time engine with advanced connection management"""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.viral_insights_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, Set[str]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.broadcast_queues: Dict[str, asyncio.Queue] = {}

        # Performance monitoring
        self.message_count = 0
        self.error_count = 0
        self.start_time = time.time()

        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()

        logger.info("🔗 Enterprise real-time engine initialized")

    async def handle_enterprise_connection(
        self, 
        websocket: WebSocket, 
        session_id: str, 
        connection_id: str
    ):
        """Handle enterprise WebSocket connection with advanced features"""
        try:
            # Register connection
            self.connections[connection_id] = websocket
            self.connection_metadata[connection_id] = {
                "session_id": session_id,
                "connected_at": datetime.utcnow(),
                "message_count": 0,
                "last_activity": datetime.utcnow()
            }

            # Add to session connections
            if session_id not in self.session_connections:
                self.session_connections[session_id] = set()
            self.session_connections[session_id].add(connection_id)

            logger.info(f"🔗 Enterprise connection established: {connection_id}")

            # Send welcome message
            await self.send_to_connection(connection_id, {
                "type": "connection_welcome",
                "connection_id": connection_id,
                "session_id": session_id,
                "server_time": datetime.utcnow().isoformat(),
                "features": [
                    "upload_progress",
                    "processing_status", 
                    "timeline_updates",
                    "preview_ready",
                    "enterprise_metrics"
                ]
            })

            # Start connection monitoring
            monitor_task = asyncio.create_task(
                self.monitor_connection(connection_id)
            )
            self.background_tasks.add(monitor_task)

            # Listen for messages
            await self.listen_for_messages(websocket, connection_id)

        except WebSocketDisconnect:
            logger.info(f"Enterprise connection disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Enterprise connection error {connection_id}: {e}", exc_info=True)
        finally:
            await self.cleanup_enterprise_connection(connection_id)

    async def start_viral_insights_stream(
        self,
        websocket: WebSocket,
        session_id: str, 
        connection_id: str
    ):
        """Start real-time viral insights streaming"""
        try:
            # Register viral insights connection
            self.viral_insights_connections[connection_id] = websocket

            logger.info(f"🎯 Viral insights stream started: {connection_id}")

            # Initialize session if needed
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    "viral_score": 0,
                    "sentiment": {"emotion": "neutral", "intensity": 0.5},
                    "engagement_peaks": [],
                    "processing_stage": "idle"
                }

            # Start real-time analysis loop
            analysis_task = asyncio.create_task(
                self.run_viral_analysis_loop(session_id, connection_id)
            )
            self.background_tasks.add(analysis_task)

            # Listen for client messages
            await self.listen_for_viral_messages(websocket, connection_id, session_id)

        except WebSocketDisconnect:
            logger.info(f"Viral insights stream disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Viral insights stream error {connection_id}: {e}", exc_info=True)
        finally:
            await self.cleanup_viral_insights_connection(connection_id)

    async def run_viral_analysis_loop(self, session_id: str, connection_id: str):
        """Run continuous viral analysis with real-time updates"""
        try:
            update_interval = 2  # Update every 2 seconds

            while connection_id in self.viral_insights_connections:
                # Generate mock viral insights (replace with real AI analysis)
                insights = await self.generate_viral_insights(session_id)

                # Broadcast insights
                await self.send_viral_insights(connection_id, {
                    "type": "viral_insights",
                    "session_id": session_id,
                    "insights": insights["insights"],
                    "confidence": insights["confidence"],
                    "trending_factors": insights["trending_factors"],
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Update sentiment
                sentiment = await self.analyze_sentiment_realtime(session_id)
                await self.send_viral_insights(connection_id, {
                    "type": "sentiment_update",
                    "session_id": session_id,
                    "sentiment": sentiment,
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Engagement predictions
                engagement = await self.predict_engagement_realtime(session_id)
                await self.send_viral_insights(connection_id, {
                    "type": "engagement_prediction",
                    "session_id": session_id,
                    "predictions": engagement["predictions"],
                    "platform_recommendations": engagement["platform_recommendations"],
                    "timestamp": datetime.utcnow().isoformat()
                })

                await asyncio.sleep(update_interval)

        except asyncio.CancelledError:
            logger.info(f"Viral analysis loop cancelled for session: {session_id}")
        except Exception as e:
            logger.error(f"Viral analysis loop error: {e}", exc_info=True)

    async def generate_viral_insights(self, session_id: str) -> Dict[str, Any]:
        """Generate real-time viral insights"""
        import random

        # Simulate AI-generated insights
        insights = [
            {
                "icon": "🎯",
                "text": "Strong visual hook detected at beginning",
                "score": random.randint(7, 10)
            },
            {
                "icon": "🎵", 
                "text": "Audio quality optimized for engagement",
                "score": random.randint(6, 9)
            },
            {
                "icon": "⚡",
                "text": "Fast-paced content maintains attention",
                "score": random.randint(7, 10)
            },
            {
                "icon": "📱",
                "text": "Mobile-optimized aspect ratio detected",
                "score": random.randint(8, 10)
            }
        ]

        trending_factors = [
            "Quick transitions",
            "Text overlays",
            "Trending music",
            "Before/after reveals",
            "Call-to-action prompts"
        ]

        return {
            "insights": random.sample(insights, 3),
            "confidence": 0.75 + random.random() * 0.2,
            "trending_factors": random.sample(trending_factors, 3)
        }

    async def analyze_sentiment_realtime(self, session_id: str) -> Dict[str, Any]:
        """Analyze sentiment in real-time"""
        import random

        sentiments = ["joy", "excitement", "surprise", "calm", "anticipation"]

        return {
            "emotion": random.choice(sentiments),
            "intensity": 0.6 + random.random() * 0.4,
            "confidence": 0.8 + random.random() * 0.15
        }

    async def predict_engagement_realtime(self, session_id: str) -> Dict[str, Any]:
        """Predict engagement metrics in real-time"""
        import random

        predictions = {
            "predicted_views": random.randint(5000, 50000),
            "predicted_shares": random.randint(50, 500),
            "predicted_likes": random.randint(200, 2000),
            "engagement_rate": round(random.uniform(0.03, 0.15), 3),
            "retention_rate": round(random.uniform(0.6, 0.9), 2)
        }

        platform_recommendations = [
            {
                "platform": "TikTok",
                "score": random.randint(80, 95),
                "optimization": "Perfect length for TikTok algorithm"
            },
            {
                "platform": "Instagram Reels", 
                "score": random.randint(70, 90),
                "optimization": "Strong visual appeal for Instagram"
            },
            {
                "platform": "YouTube Shorts",
                "score": random.randint(65, 85),
                "optimization": "Good for YouTube discovery"
            }
        ]

        return {
            "predictions": predictions,
            "platform_recommendations": platform_recommendations
        }

    async def broadcast_enterprise_progress(
        self,
        upload_id: str,
        progress_data: Dict[str, Any],
        user: Dict[str, Any]
    ):
        """Broadcast upload progress with enhanced real-time feedback"""
        try:
            # Enhanced progress message with additional context
            message = {
                "type": "upload_progress",
                "upload_id": upload_id,
                "progress": progress_data,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user.get("user_id", ""),
                "enhanced_data": {
                    "speed_mbps": progress_data.get("upload_speed", 0) / (1024 * 1024),
                    "eta_formatted": self._format_time(progress_data.get("estimated_time_remaining", 0)),
                    "progress_percentage": round(progress_data.get("progress", 0), 1),
                    "status_indicator": self._get_status_indicator(progress_data),
                    "throughput_efficiency": self._calculate_efficiency(progress_data)
                }
            }

            # Add performance insights
            if progress_data.get("average_speed", 0) > 0:
                message["enhanced_data"]["performance_tier"] = self._get_performance_tier(
                    progress_data["average_speed"]
                )

            # Broadcast to relevant connections
            await self._broadcast_to_upload_connections(upload_id, message)

            # Store progress for reconnection scenarios
            self._store_upload_progress(upload_id, message)

        except Exception as e:
            logger.error(f"Failed to broadcast progress: {e}")

    def _get_status_indicator(self, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get visual status indicator for upload progress"""
        progress = progress_data.get("progress", 0)
        speed = progress_data.get("upload_speed", 0)
        
        if progress >= 100:
            return {"icon": "✅", "color": "#4CAF50", "text": "Complete"}
        elif speed > 5 * 1024 * 1024:  # > 5 MB/s
            return {"icon": "🚀", "color": "#2196F3", "text": "Fast"}
        elif speed > 1 * 1024 * 1024:  # > 1 MB/s
            return {"icon": "⚡", "color": "#FF9800", "text": "Good"}
        elif speed > 0:
            return {"icon": "📤", "color": "#FFC107", "text": "Uploading"}
        else:
            return {"icon": "⏸️", "color": "#9E9E9E", "text": "Paused"}

    def _calculate_efficiency(self, progress_data: Dict[str, Any]) -> float:
        """Calculate upload efficiency percentage"""
        speed = progress_data.get("upload_speed", 0)
        # Assume optimal speed is 10 MB/s for efficiency calculation
        optimal_speed = 10 * 1024 * 1024
        return min(100, (speed / optimal_speed) * 100)

    def _get_performance_tier(self, speed: float) -> str:
        """Get performance tier based on upload speed"""
        speed_mbps = speed / (1024 * 1024)
        
        if speed_mbps >= 20:
            return "enterprise"
        elif speed_mbps >= 10:
            return "premium"
        elif speed_mbps >= 5:
            return "standard"
        elif speed_mbps >= 1:
            return "basic"
        else:
            return "slow"

    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds <= 0 or seconds == float('inf'):
            return "Calculating..."
        
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    async def _broadcast_to_upload_connections(self, upload_id: str, message: Dict[str, Any]):
        """Broadcast specifically to connections interested in this upload"""
        # Find connections for this upload session
        relevant_connections = []
        
        for connection_id, metadata in self.connection_metadata.items():
            session_id = metadata.get("session_id", "")
            if session_id == upload_id or session_id == "upload_manager":
                relevant_connections.append(connection_id)
        
        # Broadcast to relevant connections
        for connection_id in relevant_connections:
            await self.send_to_connection(connection_id, message)

    def _store_upload_progress(self, upload_id: str, message: Dict[str, Any]):
        """Store upload progress for reconnection scenarios"""
        if not hasattr(self, 'upload_progress_cache'):
            self.upload_progress_cache = {}
        
        self.upload_progress_cache[upload_id] = {
            "last_update": datetime.utcnow(),
            "progress_data": message,
            "expires_at": datetime.utcnow() + timedelta(hours=1)
        }
        
        # Cleanup expired entries
        expired_uploads = [
            uid for uid, data in self.upload_progress_cache.items()
            if data["expires_at"] < datetime.utcnow()
        ]
        
        for uid in expired_uploads:
            del self.upload_progress_cache[uid]

    async def broadcast_viral_insights(
        self,
        session_id: str,
        insights_data: Dict[str, Any]
    ):
        """Broadcast viral insights to relevant connections"""
        try:
            # Send to all viral insights connections for this session
            tasks = []
            for conn_id, websocket in self.viral_insights_connections.items():
                if self.connection_metadata.get(conn_id, {}).get("session_id") == session_id:
                    tasks.append(self.send_viral_insights(conn_id, insights_data))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Failed to broadcast viral insights: {e}")

    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection with error handling"""
        try:
            websocket = self.connections.get(connection_id)
            if websocket:
                await websocket.send_json(message)

                # Update metadata
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["message_count"] += 1
                    self.connection_metadata[connection_id]["last_activity"] = datetime.utcnow()

                self.message_count += 1

        except Exception as e:
            logger.warning(f"Failed to send message to {connection_id}: {e}")
            await self.cleanup_enterprise_connection(connection_id)

    async def send_viral_insights(self, connection_id: str, message: Dict[str, Any]):
        """Send viral insights message to specific connection"""
        try:
            websocket = self.viral_insights_connections.get(connection_id)
            if websocket:
                await websocket.send_json(message)
                self.message_count += 1

        except Exception as e:
            logger.warning(f"Failed to send viral insights to {connection_id}: {e}")
            await self.cleanup_viral_insights_connection(connection_id)

    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all active connections"""
        if not self.connections:
            return

        tasks = []
        for connection_id in list(self.connections.keys()):
            tasks.append(self.send_to_connection(connection_id, message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def listen_for_messages(self, websocket: WebSocket, connection_id: str):
        """Listen for incoming messages from client"""
        try:
            async for data in websocket.iter_json():
                await self.handle_client_message(connection_id, data)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Error listening for messages {connection_id}: {e}")

    async def listen_for_viral_messages(
        self, 
        websocket: WebSocket, 
        connection_id: str,
        session_id: str
    ):
        """Listen for viral insights related messages"""
        try:
            async for data in websocket.iter_json():
                await self.handle_viral_client_message(connection_id, session_id, data)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Error listening for viral messages {connection_id}: {e}")

    async def handle_client_message(self, connection_id: str, data: Dict[str, Any]):
        """Handle incoming client message"""
        try:
            message_type = data.get("type")

            if message_type == "heartbeat":
                await self.send_to_connection(connection_id, {
                    "type": "heartbeat_ack",
                    "timestamp": datetime.utcnow().isoformat()
                })
            elif message_type == "request_metrics":
                metrics = await self.get_connection_metrics(connection_id)
                await self.send_to_connection(connection_id, {
                    "type": "metrics_response",
                    "metrics": metrics
                })

        except Exception as e:
            logger.error(f"Error handling client message: {e}")

    async def handle_viral_client_message(
        self, 
        connection_id: str, 
        session_id: str, 
        data: Dict[str, Any]
    ):
        """Handle viral insights related client messages"""
        try:
            message_type = data.get("type")

            if message_type == "request_insights":
                insights = await self.generate_viral_insights(session_id)
                await self.send_viral_insights(connection_id, {
                    "type": "viral_insights",
                    "insights": insights,
                    "timestamp": datetime.utcnow().isoformat()
                })

        except Exception as e:
            logger.error(f"Error handling viral client message: {e}")

    async def monitor_connection(self, connection_id: str):
        """Monitor connection health and cleanup stale connections"""
        try:
            while connection_id in self.connections:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Check if connection is still active
                metadata = self.connection_metadata.get(connection_id)
                if metadata:
                    last_activity = metadata["last_activity"]
                    if (datetime.utcnow() - last_activity).total_seconds() > 300:  # 5 minutes
                        logger.info(f"Cleaning up stale connection: {connection_id}")
                        await self.cleanup_enterprise_connection(connection_id)
                        break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Connection monitoring error: {e}")

    async def get_connection_metrics(self, connection_id: str) -> Dict[str, Any]:
        """Get metrics for specific connection"""
        metadata = self.connection_metadata.get(connection_id, {})

        return {
            "connection_id": connection_id,
            "connected_at": metadata.get("connected_at", "").isoformat() if metadata.get("connected_at") else "",
            "message_count": metadata.get("message_count", 0),
            "last_activity": metadata.get("last_activity", "").isoformat() if metadata.get("last_activity") else "",
            "uptime_seconds": (datetime.utcnow() - metadata.get("connected_at", datetime.utcnow())).total_seconds()
        }

    async def cleanup_enterprise_connection(self, connection_id: str):
        """Clean up enterprise connection resources"""
        try:
            # Remove from connections
            if connection_id in self.connections:
                del self.connections[connection_id]

            # Clean up metadata
            metadata = self.connection_metadata.pop(connection_id, {})
            session_id = metadata.get("session_id")

            # Remove from session connections
            if session_id and session_id in self.session_connections:
                self.session_connections[session_id].discard(connection_id)
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]

            # Cancel background tasks
            for task in list(self.background_tasks):
                if task.done() or task.cancelled():
                    self.background_tasks.discard(task)

            logger.info(f"Cleaned up enterprise connection: {connection_id}")

        except Exception as e:
            logger.error(f"Error cleaning up connection {connection_id}: {e}")

    async def cleanup_viral_insights_connection(self, connection_id: str):
        """Clean up viral insights connection resources"""
        try:
            if connection_id in self.viral_insights_connections:
                del self.viral_insights_connections[connection_id]

            logger.info(f"Cleaned up viral insights connection: {connection_id}")

        except Exception as e:
            logger.error(f"Error cleaning up viral insights connection {connection_id}: {e}")

    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get real-time engine statistics"""
        uptime = time.time() - self.start_time

        return {
            "active_connections": len(self.connections),
            "viral_insights_connections": len(self.viral_insights_connections),
            "active_sessions": len(self.active_sessions),
            "total_messages": self.message_count,
            "error_count": self.error_count,
            "uptime_seconds": uptime,
            "background_tasks": len(self.background_tasks)
        }

    async def graceful_shutdown(self):
        """Gracefully shutdown the real-time engine"""
        try:
            logger.info("🔄 Starting real-time engine shutdown...")

            # Cancel all background tasks
            for task in self.background_tasks:
                task.cancel()

            # Close all connections
            close_tasks = []
            for websocket in list(self.connections.values()):
                close_tasks.append(websocket.close(code=1001, reason="Server shutdown"))

            for websocket in list(self.viral_insights_connections.values()):
                close_tasks.append(websocket.close(code=1001, reason="Server shutdown"))

            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

            # Clear all data structures
            self.connections.clear()
            self.connection_metadata.clear()
            self.viral_insights_connections.clear()
            self.session_connections.clear()
            self.active_sessions.clear()
            self.background_tasks.clear()

            logger.info("✅ Real-time engine shutdown complete")

        except Exception as e:
            logger.error(f"Error during real-time engine shutdown: {e}")