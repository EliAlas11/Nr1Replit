
"""
Enhanced Metrics Collection System
Provides comprehensive application metrics and monitoring
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class MetricEvent:
    """Represents a single metric event"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Enhanced metrics collection and aggregation system"""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.counters: defaultdict = defaultdict(int)
        self.gauges: defaultdict = defaultdict(float)
        self.histograms: defaultdict = defaultdict(list)
        self.timers: defaultdict = defaultdict(list)
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=1000)
        self.error_counts: defaultdict = defaultdict(int)
        self.upload_stats: deque = deque(maxlen=500)
        self.processing_stats: deque = deque(maxlen=500)
        
        # System health
        self.health_checks: Dict[str, Any] = {}
        self.start_time = datetime.now()
        
        logger.info("✅ Metrics collector initialized")
    
    async def record_request(
        self, 
        method: str, 
        path: str, 
        status_code: int, 
        duration: float,
        **kwargs
    ):
        """Record HTTP request metrics"""
        try:
            # Record timing
            self.request_times.append(duration)
            self.timers[f"request.{method.lower()}"].append(duration)
            
            # Update counters
            self.counters[f"requests.total"] += 1
            self.counters[f"requests.{method.lower()}"] += 1
            self.counters[f"responses.{status_code}"] += 1
            
            # Track errors
            if status_code >= 400:
                self.error_counts[f"{method}:{path}"] += 1
                self.counters["requests.errors"] += 1
            
            # Record event
            event = MetricEvent(
                name="http_request",
                value=duration,
                timestamp=datetime.now(),
                tags={
                    "method": method,
                    "path": path,
                    "status": str(status_code)
                },
                metadata=kwargs
            )
            self.events.append(event)
            
            # Update gauges
            if len(self.request_times) > 0:
                self.gauges["request.avg_duration"] = sum(self.request_times) / len(self.request_times)
                self.gauges["request.max_duration"] = max(self.request_times)
                self.gauges["request.min_duration"] = min(self.request_times)
            
        except Exception as e:
            logger.error(f"Error recording request metrics: {e}")
    
    async def record_upload(
        self, 
        file_size: int, 
        processing_time: float, 
        file_type: str,
        success: bool = True,
        **kwargs
    ):
        """Record file upload metrics"""
        try:
            upload_stat = {
                "file_size": file_size,
                "processing_time": processing_time,
                "file_type": file_type,
                "success": success,
                "timestamp": datetime.now(),
                **kwargs
            }
            self.upload_stats.append(upload_stat)
            
            # Update counters
            self.counters["uploads.total"] += 1
            if success:
                self.counters["uploads.success"] += 1
            else:
                self.counters["uploads.failed"] += 1
            
            self.counters[f"uploads.{file_type}"] += 1
            
            # Update gauges
            recent_uploads = [s for s in self.upload_stats if s["success"]]
            if recent_uploads:
                avg_size = sum(s["file_size"] for s in recent_uploads) / len(recent_uploads)
                avg_time = sum(s["processing_time"] for s in recent_uploads) / len(recent_uploads)
                
                self.gauges["upload.avg_file_size"] = avg_size
                self.gauges["upload.avg_processing_time"] = avg_time
                self.gauges["upload.success_rate"] = len(recent_uploads) / len(self.upload_stats) * 100
            
            # Record event
            event = MetricEvent(
                name="file_upload",
                value=processing_time,
                timestamp=datetime.now(),
                tags={
                    "file_type": file_type,
                    "success": str(success)
                },
                metadata={
                    "file_size": file_size,
                    **kwargs
                }
            )
            self.events.append(event)
            
        except Exception as e:
            logger.error(f"Error recording upload metrics: {e}")
    
    async def record_processing(
        self,
        video_duration: float,
        processing_time: float,
        clips_generated: int,
        quality: str,
        success: bool = True,
        **kwargs
    ):
        """Record video processing metrics"""
        try:
            processing_stat = {
                "video_duration": video_duration,
                "processing_time": processing_time,
                "clips_generated": clips_generated,
                "quality": quality,
                "success": success,
                "timestamp": datetime.now(),
                "efficiency": video_duration / processing_time if processing_time > 0 else 0,
                **kwargs
            }
            self.processing_stats.append(processing_stat)
            
            # Update counters
            self.counters["processing.total"] += 1
            self.counters[f"processing.{quality}"] += 1
            self.counters["clips.generated"] += clips_generated
            
            if success:
                self.counters["processing.success"] += 1
            else:
                self.counters["processing.failed"] += 1
            
            # Update gauges
            recent_processing = [s for s in self.processing_stats if s["success"]]
            if recent_processing:
                avg_efficiency = sum(s["efficiency"] for s in recent_processing) / len(recent_processing)
                avg_clips = sum(s["clips_generated"] for s in recent_processing) / len(recent_processing)
                
                self.gauges["processing.avg_efficiency"] = avg_efficiency
                self.gauges["processing.avg_clips"] = avg_clips
                self.gauges["processing.success_rate"] = len(recent_processing) / len(self.processing_stats) * 100
            
            # Record event
            event = MetricEvent(
                name="video_processing",
                value=processing_time,
                timestamp=datetime.now(),
                tags={
                    "quality": quality,
                    "success": str(success)
                },
                metadata={
                    "video_duration": video_duration,
                    "clips_generated": clips_generated,
                    **kwargs
                }
            )
            self.events.append(event)
            
        except Exception as e:
            logger.error(f"Error recording processing metrics: {e}")
    
    async def record_error(self, error_type: str, error_message: str, **kwargs):
        """Record error metrics"""
        try:
            self.counters[f"errors.{error_type}"] += 1
            self.counters["errors.total"] += 1
            
            event = MetricEvent(
                name="error",
                value=1,
                timestamp=datetime.now(),
                tags={
                    "error_type": error_type,
                    "error_class": error_message.__class__.__name__ if hasattr(error_message, '__class__') else "Unknown"
                },
                metadata={
                    "message": str(error_message),
                    **kwargs
                }
            )
            self.events.append(event)
            
        except Exception as e:
            logger.error(f"Error recording error metrics: {e}")
    
    async def record_custom_metric(
        self, 
        name: str, 
        value: float, 
        metric_type: str = "gauge",
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Record custom application metrics"""
        try:
            if metric_type == "counter":
                self.counters[name] += value
            elif metric_type == "gauge":
                self.gauges[name] = value
            elif metric_type == "histogram":
                self.histograms[name].append(value)
            elif metric_type == "timer":
                self.timers[name].append(value)
            
            event = MetricEvent(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metadata=kwargs
            )
            self.events.append(event)
            
        except Exception as e:
            logger.error(f"Error recording custom metric: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        try:
            now = datetime.now()
            uptime = now - self.start_time
            
            # Calculate request rates
            recent_requests = [
                e for e in self.events 
                if e.name == "http_request" and 
                now - e.timestamp < timedelta(minutes=5)
            ]
            
            request_rate = len(recent_requests) / 5 if recent_requests else 0  # per minute
            
            # Calculate error rates
            recent_errors = [
                e for e in self.events 
                if e.name == "error" and 
                now - e.timestamp < timedelta(minutes=5)
            ]
            
            error_rate = len(recent_errors) / 5 if recent_errors else 0  # per minute
            
            return {
                "uptime_seconds": uptime.total_seconds(),
                "uptime_formatted": str(uptime),
                "total_events": len(self.events),
                "request_rate_per_minute": round(request_rate, 2),
                "error_rate_per_minute": round(error_rate, 2),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "health_status": self._get_health_status(),
                "performance": self._get_performance_summary(),
                "upload_statistics": self._get_upload_summary(),
                "processing_statistics": self._get_processing_summary()
            }
            
        except Exception as e:
            logger.error(f"Error generating metrics summary: {e}")
            return {"error": str(e)}
    
    def _get_health_status(self) -> str:
        """Determine overall system health"""
        try:
            # Check error rates
            recent_errors = sum(1 for e in self.events 
                              if e.name == "error" and 
                              datetime.now() - e.timestamp < timedelta(minutes=5))
            
            if recent_errors > 10:
                return "unhealthy"
            elif recent_errors > 5:
                return "degraded"
            else:
                return "healthy"
                
        except Exception:
            return "unknown"
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        try:
            if not self.request_times:
                return {"status": "no_data"}
            
            request_times = list(self.request_times)
            
            return {
                "avg_response_time": round(sum(request_times) / len(request_times), 3),
                "max_response_time": round(max(request_times), 3),
                "min_response_time": round(min(request_times), 3),
                "p95_response_time": round(sorted(request_times)[int(len(request_times) * 0.95)], 3),
                "total_requests": len(request_times)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_upload_summary(self) -> Dict[str, Any]:
        """Get upload statistics summary"""
        try:
            if not self.upload_stats:
                return {"status": "no_data"}
            
            successful_uploads = [s for s in self.upload_stats if s["success"]]
            
            return {
                "total_uploads": len(self.upload_stats),
                "successful_uploads": len(successful_uploads),
                "success_rate": round(len(successful_uploads) / len(self.upload_stats) * 100, 2),
                "avg_file_size_mb": round(sum(s["file_size"] for s in successful_uploads) / len(successful_uploads) / (1024*1024), 2) if successful_uploads else 0,
                "avg_processing_time": round(sum(s["processing_time"] for s in successful_uploads) / len(successful_uploads), 2) if successful_uploads else 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_processing_summary(self) -> Dict[str, Any]:
        """Get processing statistics summary"""
        try:
            if not self.processing_stats:
                return {"status": "no_data"}
            
            successful_processing = [s for s in self.processing_stats if s["success"]]
            
            return {
                "total_processed": len(self.processing_stats),
                "successful_processed": len(successful_processing),
                "success_rate": round(len(successful_processing) / len(self.processing_stats) * 100, 2),
                "avg_efficiency": round(sum(s["efficiency"] for s in successful_processing) / len(successful_processing), 2) if successful_processing else 0,
                "total_clips_generated": sum(s["clips_generated"] for s in successful_processing)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def close(self):
        """Clean up metrics collector"""
        try:
            logger.info(f"Metrics collector closing - {len(self.events)} events recorded")
            self.events.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
            self.request_times.clear()
            self.upload_stats.clear()
            self.processing_stats.clear()
            
        except Exception as e:
            logger.error(f"Error closing metrics collector: {e}")
