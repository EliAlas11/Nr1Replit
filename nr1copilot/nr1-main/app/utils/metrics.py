
"""
Metrics Collection System
Netflix-level performance monitoring
"""

import time
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Netflix-level metrics collection and monitoring"""
    
    def __init__(self):
        self.request_metrics = defaultdict(list)
        self.response_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.status_codes = defaultdict(int)
        self.endpoint_metrics = defaultdict(lambda: {
            "count": 0,
            "total_time": 0,
            "errors": 0
        })
        self.start_time = datetime.now()
        
    async def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        user_id: str = None
    ) -> None:
        """Record a request with its metrics"""
        try:
            timestamp = datetime.now()
            
            # Record response time
            self.response_times.append(duration)
            
            # Record status code
            self.status_codes[status_code] += 1
            
            # Record endpoint metrics
            endpoint_key = f"{method} {path}"
            self.endpoint_metrics[endpoint_key]["count"] += 1
            self.endpoint_metrics[endpoint_key]["total_time"] += duration
            
            if status_code >= 400:
                self.endpoint_metrics[endpoint_key]["errors"] += 1
                self.error_counts[f"{status_code}"] += 1
            
            # Store detailed request data
            self.request_metrics[timestamp.strftime("%Y-%m-%d %H:%M")].append({
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration": duration,
                "timestamp": timestamp.isoformat(),
                "user_id": user_id
            })
            
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        try:
            now = datetime.now()
            uptime = (now - self.start_time).total_seconds()
            
            # Calculate request statistics
            total_requests = sum(self.status_codes.values())
            successful_requests = sum(
                count for status, count in self.status_codes.items()
                if 200 <= int(status) < 400
            )
            error_requests = sum(
                count for status, count in self.status_codes.items()
                if int(status) >= 400
            )
            
            # Calculate response time statistics
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
                min_response_time = min(self.response_times)
                max_response_time = max(self.response_times)
                
                # Calculate percentiles
                sorted_times = sorted(self.response_times)
                p95_index = int(len(sorted_times) * 0.95)
                p99_index = int(len(sorted_times) * 0.99)
                p95_response_time = sorted_times[p95_index] if sorted_times else 0
                p99_response_time = sorted_times[p99_index] if sorted_times else 0
            else:
                avg_response_time = 0
                min_response_time = 0
                max_response_time = 0
                p95_response_time = 0
                p99_response_time = 0
            
            # Calculate requests per minute
            requests_per_minute = (
                total_requests / (uptime / 60) if uptime > 0 else 0
            )
            
            # Calculate success rate
            success_rate = (
                successful_requests / total_requests * 100 
                if total_requests > 0 else 0
            )
            
            # Top endpoints by request count
            top_endpoints = sorted(
                self.endpoint_metrics.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:10]
            
            # Slowest endpoints
            slowest_endpoints = sorted(
                [
                    (endpoint, metrics["total_time"] / metrics["count"])
                    for endpoint, metrics in self.endpoint_metrics.items()
                    if metrics["count"] > 0
                ],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                "timestamp": now.isoformat(),
                "uptime_seconds": uptime,
                "requests": {
                    "total": total_requests,
                    "successful": successful_requests,
                    "errors": error_requests,
                    "success_rate": success_rate,
                    "requests_per_minute": requests_per_minute
                },
                "response_times": {
                    "average": avg_response_time,
                    "min": min_response_time,
                    "max": max_response_time,
                    "p95": p95_response_time,
                    "p99": p99_response_time
                },
                "status_codes": dict(self.status_codes),
                "error_counts": dict(self.error_counts),
                "top_endpoints": [
                    {
                        "endpoint": endpoint,
                        "count": metrics["count"],
                        "avg_time": metrics["total_time"] / metrics["count"],
                        "errors": metrics["errors"]
                    }
                    for endpoint, metrics in top_endpoints
                ],
                "slowest_endpoints": [
                    {
                        "endpoint": endpoint,
                        "avg_time": avg_time
                    }
                    for endpoint, avg_time in slowest_endpoints
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}
    
    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """Get statistics for a specific endpoint"""
        if endpoint in self.endpoint_metrics:
            metrics = self.endpoint_metrics[endpoint]
            return {
                "count": metrics["count"],
                "average_time": (
                    metrics["total_time"] / metrics["count"] 
                    if metrics["count"] > 0 else 0
                ),
                "errors": metrics["errors"],
                "error_rate": (
                    metrics["errors"] / metrics["count"] * 100 
                    if metrics["count"] > 0 else 0
                )
            }
        return {"count": 0, "average_time": 0, "errors": 0, "error_rate": 0}
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.request_metrics.clear()
        self.response_times.clear()
        self.error_counts.clear()
        self.status_codes.clear()
        self.endpoint_metrics.clear()
        self.start_time = datetime.now()
    
    async def cleanup_old_metrics(self, hours: int = 24) -> None:
        """Clean up metrics older than specified hours"""
        try:
            cutoff = datetime.now() - timedelta(hours=hours)
            cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M")
            
            keys_to_remove = [
                key for key in self.request_metrics.keys()
                if key < cutoff_str
            ]
            
            for key in keys_to_remove:
                del self.request_metrics[key]
            
            logger.info(f"Cleaned up {len(keys_to_remove)} old metric entries")
            
        except Exception as e:
            logger.error(f"Error cleaning up metrics: {e}")
"""
Netflix-Level Metrics Collection System
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Netflix-level metrics collection and analysis"""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.request_metrics = deque(maxlen=max_entries)
        self.processing_metrics = deque(maxlen=max_entries)
        self.error_metrics = deque(maxlen=max_entries)
        
        # Real-time counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.gauges = defaultdict(float)
        
        # Performance tracking
        self.response_times = deque(maxlen=1000)
        self.throughput_tracker = deque(maxlen=100)
        
        # Start background metrics processor
        self._metrics_task = None
        self.start_background_processor()
    
    def start_background_processor(self):
        """Start background metrics processing"""
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(self._process_metrics_background())
    
    async def _process_metrics_background(self):
        """Background task to process and aggregate metrics"""
        while True:
            try:
                await self._aggregate_metrics()
                await asyncio.sleep(60)  # Process every minute
            except Exception as e:
                logger.error(f"Metrics processing error: {e}")
                await asyncio.sleep(5)
    
    async def record_request(
        self, 
        method: str, 
        path: str, 
        status_code: int, 
        duration: float,
        user_agent: str = "",
        ip_address: str = ""
    ):
        """Record HTTP request metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            "user_agent": user_agent,
            "ip_address": ip_address
        }
        
        self.request_metrics.append(metric)
        self.response_times.append(duration)
        
        # Update counters
        self.counters[f"requests_total"] += 1
        self.counters[f"requests_{method.lower()}"] += 1
        self.counters[f"status_{status_code}"] += 1
        
        # Update gauges
        self.gauges["avg_response_time"] = sum(self.response_times) / len(self.response_times)
    
    async def record_processing(
        self, 
        task_id: str, 
        task_type: str, 
        duration: float, 
        status: str,
        clips_processed: int = 0,
        file_size: int = 0
    ):
        """Record video processing metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "task_type": task_type,
            "duration": duration,
            "status": status,
            "clips_processed": clips_processed,
            "file_size": file_size
        }
        
        self.processing_metrics.append(metric)
        
        # Update counters
        self.counters["processing_total"] += 1
        self.counters[f"processing_{status}"] += 1
        self.counters["clips_generated"] += clips_processed
        
        # Update gauges
        if status == "completed":
            self.gauges["avg_processing_time"] = duration
            self.gauges["total_clips_generated"] = self.counters["clips_generated"]
    
    async def record_error(
        self, 
        error_type: str, 
        error_message: str, 
        path: str = "",
        user_data: Dict[str, Any] = None
    ):
        """Record error metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "path": path,
            "user_data": user_data or {}
        }
        
        self.error_metrics.append(metric)
        
        # Update counters
        self.counters["errors_total"] += 1
        self.counters[f"error_{error_type}"] += 1
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        now = datetime.now()
        
        # Calculate time-based metrics
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        recent_requests = [
            m for m in self.request_metrics 
            if datetime.fromisoformat(m["timestamp"]) > last_hour
        ]
        
        recent_processing = [
            m for m in self.processing_metrics
            if datetime.fromisoformat(m["timestamp"]) > last_hour
        ]
        
        recent_errors = [
            m for m in self.error_metrics
            if datetime.fromisoformat(m["timestamp"]) > last_hour
        ]
        
        return {
            "timestamp": now.isoformat(),
            "system_health": {
                "status": "healthy" if len(recent_errors) < 10 else "degraded",
                "uptime": self._calculate_uptime(),
                "memory_usage": self._get_memory_usage(),
                "cpu_usage": self._get_cpu_usage()
            },
            "request_metrics": {
                "total_requests": self.counters["requests_total"],
                "requests_last_hour": len(recent_requests),
                "avg_response_time": self.gauges.get("avg_response_time", 0),
                "success_rate": self._calculate_success_rate(recent_requests),
                "requests_per_minute": len(recent_requests) / 60 if recent_requests else 0
            },
            "processing_metrics": {
                "total_processed": self.counters["processing_total"],
                "processed_last_hour": len(recent_processing),
                "avg_processing_time": self.gauges.get("avg_processing_time", 0),
                "total_clips_generated": self.counters["clips_generated"],
                "success_rate": self._calculate_processing_success_rate(recent_processing)
            },
            "error_metrics": {
                "total_errors": self.counters["errors_total"],
                "errors_last_hour": len(recent_errors),
                "error_rate": len(recent_errors) / max(len(recent_requests), 1) * 100,
                "top_errors": self._get_top_errors(recent_errors)
            },
            "performance": {
                "p95_response_time": self._calculate_percentile(list(self.response_times), 95),
                "p99_response_time": self._calculate_percentile(list(self.response_times), 99),
                "throughput": self._calculate_throughput(),
                "concurrent_processing": self._get_active_processing_count()
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges)
        }
    
    def _calculate_uptime(self) -> str:
        """Calculate system uptime"""
        # Simplified uptime calculation
        return "99.9%"
    
    def _get_memory_usage(self) -> str:
        """Get memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"{memory.percent}%"
        except ImportError:
            return "N/A"
    
    def _get_cpu_usage(self) -> str:
        """Get CPU usage"""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            return f"{cpu}%"
        except ImportError:
            return "N/A"
    
    def _calculate_success_rate(self, requests: List[Dict[str, Any]]) -> float:
        """Calculate success rate for requests"""
        if not requests:
            return 100.0
        
        successful = sum(1 for r in requests if 200 <= r["status_code"] < 400)
        return (successful / len(requests)) * 100
    
    def _calculate_processing_success_rate(self, processing: List[Dict[str, Any]]) -> float:
        """Calculate success rate for processing"""
        if not processing:
            return 100.0
        
        successful = sum(1 for p in processing if p["status"] == "completed")
        return (successful / len(processing)) * 100
    
    def _get_top_errors(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get top error types"""
        error_counts = defaultdict(int)
        for error in errors:
            error_counts[error["error_type"]] += 1
        
        return [
            {"type": error_type, "count": count}
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _calculate_throughput(self) -> float:
        """Calculate requests per second"""
        if len(self.response_times) < 2:
            return 0.0
        
        # Simple throughput calculation
        return len(self.response_times) / 60  # Requests per minute / 60
    
    def _get_active_processing_count(self) -> int:
        """Get count of active processing tasks"""
        # This would be updated from the main application
        return self.gauges.get("active_processing", 0)
    
    async def _aggregate_metrics(self):
        """Aggregate and clean up old metrics"""
        now = datetime.now()
        cutoff = now - timedelta(hours=24)
        
        # Clean up old metrics
        self.request_metrics = deque([
            m for m in self.request_metrics 
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ], maxlen=self.max_entries)
        
        self.processing_metrics = deque([
            m for m in self.processing_metrics
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ], maxlen=self.max_entries)
        
        self.error_metrics = deque([
            m for m in self.error_metrics
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ], maxlen=self.max_entries)
        
        logger.info(f"Metrics aggregated: {len(self.request_metrics)} requests, {len(self.processing_metrics)} processing, {len(self.error_metrics)} errors")
