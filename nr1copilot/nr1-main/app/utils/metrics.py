
"""
Netflix-Level Metrics Collection
"""

import time
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio

class MetricsCollector:
    """Netflix-level metrics collection and reporting"""
    
    def __init__(self):
        self.request_metrics = deque(maxlen=10000)  # Last 10k requests
        self.system_metrics = {}
        self.performance_metrics = defaultdict(list)
        self.error_metrics = defaultdict(int)
        self.start_time = time.time()
    
    async def record_request(self, method: str, path: str, status_code: int, duration: float):
        """Record request metrics"""
        now = time.time()
        
        metric = {
            "timestamp": now,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            "date": datetime.now().isoformat()
        }
        
        self.request_metrics.append(metric)
        
        # Track performance by endpoint
        endpoint_key = f"{method}:{path}"
        self.performance_metrics[endpoint_key].append(duration)
        
        # Keep only last 1000 measurements per endpoint
        if len(self.performance_metrics[endpoint_key]) > 1000:
            self.performance_metrics[endpoint_key] = self.performance_metrics[endpoint_key][-1000:]
        
        # Track errors
        if status_code >= 400:
            self.error_metrics[f"{status_code}"] += 1
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        now = time.time()
        uptime = now - self.start_time
        
        # Calculate request stats
        recent_requests = [r for r in self.request_metrics if now - r["timestamp"] < 3600]  # Last hour
        
        total_requests = len(self.request_metrics)
        recent_request_count = len(recent_requests)
        
        # Response time percentiles
        recent_durations = [r["duration"] for r in recent_requests]
        recent_durations.sort()
        
        percentiles = {}
        if recent_durations:
            percentiles = {
                "p50": self._percentile(recent_durations, 50),
                "p90": self._percentile(recent_durations, 90),
                "p95": self._percentile(recent_durations, 95),
                "p99": self._percentile(recent_durations, 99)
            }
        
        # Error rates
        recent_errors = sum(1 for r in recent_requests if r["status_code"] >= 400)
        error_rate = (recent_errors / recent_request_count * 100) if recent_request_count > 0 else 0
        
        # Status code distribution
        status_distribution = defaultdict(int)
        for request in recent_requests:
            status_range = f"{request['status_code'] // 100}xx"
            status_distribution[status_range] += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "uptime_human": self._format_duration(uptime),
            "requests": {
                "total": total_requests,
                "last_hour": recent_request_count,
                "requests_per_second": recent_request_count / 3600 if recent_request_count > 0 else 0
            },
            "performance": {
                "response_times": percentiles,
                "average_response_time": sum(recent_durations) / len(recent_durations) if recent_durations else 0
            },
            "errors": {
                "error_rate_percent": round(error_rate, 2),
                "total_errors": dict(self.error_metrics),
                "recent_errors": recent_errors
            },
            "status_codes": dict(status_distribution),
            "system": await self._get_system_metrics()
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        index = int(len(data) * percentile / 100)
        return data[min(index, len(data) - 1)]
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            import psutil
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except ImportError:
            return {"error": "psutil not available"}
