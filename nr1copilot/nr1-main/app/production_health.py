
"""
Netflix-tier production health monitoring
Ultra-reliable health checks optimized for Render.com
"""

import asyncio
import time
import logging
import psutil
import os
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """Health metric data structure"""
    name: str
    value: float
    status: str
    threshold: float
    timestamp: datetime

class ProductionHealthMonitor:
    """Netflix-tier health monitoring for production deployment"""
    
    def __init__(self):
        self.start_time = time.time()
        self.health_history: List[Dict[str, Any]] = []
        self.max_history = 100
        
        # Production thresholds
        self.thresholds = {
            "memory_percent": 85.0,
            "cpu_percent": 80.0,
            "disk_percent": 90.0,
            "response_time_ms": 500.0
        }
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for production"""
        check_start = time.time()
        
        try:
            # System metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')
            
            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            
            # Determine overall status
            status = "healthy"
            issues = []
            
            if memory.percent > self.thresholds["memory_percent"]:
                status = "degraded"
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if cpu_percent > self.thresholds["cpu_percent"]:
                status = "degraded"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if disk.percent > self.thresholds["disk_percent"]:
                status = "degraded"
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            response_time_ms = (time.time() - check_start) * 1000
            
            health_result = {
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": round(uptime_seconds, 2),
                "response_time_ms": round(response_time_ms, 2),
                "system": {
                    "memory": {
                        "percent": round(memory.percent, 2),
                        "available_gb": round(memory.available / (1024**3), 2),
                        "status": "healthy" if memory.percent < self.thresholds["memory_percent"] else "warning"
                    },
                    "cpu": {
                        "percent": round(cpu_percent, 2),
                        "status": "healthy" if cpu_percent < self.thresholds["cpu_percent"] else "warning"
                    },
                    "disk": {
                        "percent": round(disk.percent, 2),
                        "free_gb": round(disk.free / (1024**3), 2),
                        "status": "healthy" if disk.percent < self.thresholds["disk_percent"] else "warning"
                    }
                },
                "application": {
                    "version": "10.0.0",
                    "environment": "production",
                    "tier": "netflix-enterprise"
                },
                "issues": issues
            }
            
            # Store in history
            self._store_health_record(health_result)
            
            return health_result
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "response_time_ms": (time.time() - check_start) * 1000
            }
    
    def _store_health_record(self, record: Dict[str, Any]):
        """Store health record in history"""
        self.health_history.append({
            "timestamp": record["timestamp"],
            "status": record["status"],
            "response_time_ms": record["response_time_ms"],
            "memory_percent": record.get("system", {}).get("memory", {}).get("percent", 0),
            "cpu_percent": record.get("system", {}).get("cpu", {}).get("percent", 0)
        })
        
        # Maintain history size
        if len(self.health_history) > self.max_history:
            self.health_history.pop(0)
    
    def get_health_trends(self) -> Dict[str, Any]:
        """Analyze health trends"""
        if len(self.health_history) < 5:
            return {"status": "insufficient_data"}
        
        recent = self.health_history[-10:]
        
        # Calculate averages
        avg_response_time = sum(r["response_time_ms"] for r in recent) / len(recent)
        avg_memory = sum(r["memory_percent"] for r in recent) / len(recent)
        avg_cpu = sum(r["cpu_percent"] for r in recent) / len(recent)
        
        # Health distribution
        healthy_count = sum(1 for r in recent if r["status"] == "healthy")
        health_percentage = (healthy_count / len(recent)) * 100
        
        return {
            "trend_analysis": {
                "avg_response_time_ms": round(avg_response_time, 2),
                "avg_memory_percent": round(avg_memory, 2),
                "avg_cpu_percent": round(avg_cpu, 2),
                "health_percentage": round(health_percentage, 2)
            },
            "recent_checks": len(recent),
            "trend": self._determine_trend(recent)
        }
    
    def _determine_trend(self, checks: List[Dict[str, Any]]) -> str:
        """Determine health trend"""
        if len(checks) < 3:
            return "stable"
        
        recent_healthy = sum(1 for c in checks[-3:] if c["status"] == "healthy")
        earlier_healthy = sum(1 for c in checks[-6:-3] if c["status"] == "healthy")
        
        if recent_healthy > earlier_healthy:
            return "improving"
        elif recent_healthy < earlier_healthy:
            return "degrading"
        return "stable"

# Global health monitor instance
health_monitor = ProductionHealthMonitor()
