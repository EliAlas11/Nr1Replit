
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
