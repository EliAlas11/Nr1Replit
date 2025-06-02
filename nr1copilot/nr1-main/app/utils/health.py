
"""
Netflix-Grade System Health Monitor v10.0
Comprehensive health monitoring with predictive analytics and enterprise diagnostics
"""

import asyncio
import logging
import time
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration"""
    EXCELLENT = "excellent"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric with metadata"""
    name: str
    value: Any
    status: HealthStatus
    timestamp: float = field(default_factory=time.time)
    message: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp
    
    @property
    def is_stale(self) -> bool:
        return self.age_seconds > 300  # 5 minutes


@dataclass
class SystemSnapshot:
    """System state snapshot for trend analysis"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    
    @classmethod
    def capture(cls) -> 'SystemSnapshot':
        """Capture current system state"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return cls(
                timestamp=time.time(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                process_count=len(psutil.pids())
            )
        except Exception as e:
            logger.error(f"Failed to capture system snapshot: {e}")
            return cls(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                process_count=0
            )


class SystemHealthMonitor:
    """Netflix-grade system health monitoring with predictive analytics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics: Dict[str, HealthMetric] = {}
        self.snapshots: deque = deque(maxlen=100)
        self.error_counts = defaultdict(int)
        self.last_check_time = 0
        self.check_interval = 30
        self._monitoring_task: Optional[asyncio.Task] = None
        self._initialized = False
        
        # Health thresholds
        self.thresholds = {
            'cpu': {'warning': 70.0, 'critical': 90.0},
            'memory': {'warning': 80.0, 'critical': 95.0},
            'disk': {'warning': 85.0, 'critical': 95.0},
            'load_average': {'warning': 2.0, 'critical': 4.0}
        }
        
        logger.info("🏥 System Health Monitor v10.0 initialized")
    
    async def initialize(self) -> None:
        """Initialize health monitoring system"""
        if self._initialized:
            return
        
        try:
            # Capture initial snapshot
            initial_snapshot = SystemSnapshot.capture()
            self.snapshots.append(initial_snapshot)
            
            # Start background monitoring
            self._monitoring_task = asyncio.create_task(self._background_monitoring())
            
            self._initialized = True
            logger.info("✅ Health monitoring started")
            
        except Exception as e:
            logger.error(f"Health monitor initialization failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown health monitoring"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Health monitor shutdown complete")
    
    async def _background_monitoring(self) -> None:
        """Background health monitoring task"""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self._collect_system_metrics()
                
                # Capture system snapshot
                snapshot = SystemSnapshot.capture()
                self.snapshots.append(snapshot)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self) -> None:
        """Collect comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_status = self._assess_threshold_status(cpu_percent, self.thresholds['cpu'])
            
            self.metrics["cpu"] = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                status=cpu_status,
                threshold_warning=self.thresholds['cpu']['warning'],
                threshold_critical=self.thresholds['cpu']['critical'],
                unit="percent"
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_status = self._assess_threshold_status(memory.percent, self.thresholds['memory'])
            
            self.metrics["memory"] = HealthMetric(
                name="memory_usage",
                value=memory.percent,
                status=memory_status,
                threshold_warning=self.thresholds['memory']['warning'],
                threshold_critical=self.thresholds['memory']['critical'],
                unit="percent"
            )
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._assess_threshold_status(disk_percent, self.thresholds['disk'])
            
            self.metrics["disk"] = HealthMetric(
                name="disk_usage",
                value=disk_percent,
                status=disk_status,
                threshold_warning=self.thresholds['disk']['warning'],
                threshold_critical=self.thresholds['disk']['critical'],
                unit="percent"
            )
            
            # Load average (if available)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()[0]
                load_status = self._assess_threshold_status(load_avg, self.thresholds['load_average'])
                
                self.metrics["load_average"] = HealthMetric(
                    name="load_average",
                    value=load_avg,
                    status=load_status,
                    threshold_warning=self.thresholds['load_average']['warning'],
                    threshold_critical=self.thresholds['load_average']['critical'],
                    unit="ratio"
                )
            
            self.last_check_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            self.record_error("metrics_collection")
    
    def _assess_threshold_status(self, value: float, thresholds: Dict[str, float]) -> HealthStatus:
        """Assess health status based on thresholds"""
        if value >= thresholds['critical']:
            return HealthStatus.CRITICAL
        elif value >= thresholds['warning']:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall system health status"""
        if not self.metrics:
            return HealthStatus.UNKNOWN
        
        statuses = [metric.status for metric in self.metrics.values() if not metric.is_stale]
        
        if not statuses:
            return HealthStatus.UNKNOWN
        
        # Return worst status
        status_priority = {
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNHEALTHY: 1,
            HealthStatus.DEGRADED: 2,
            HealthStatus.HEALTHY: 3,
            HealthStatus.EXCELLENT: 4,
            HealthStatus.UNKNOWN: 5
        }
        
        return min(statuses, key=lambda s: status_priority.get(s, 999))
    
    def _calculate_health_score(self) -> float:
        """Calculate numerical health score (0-10)"""
        if not self.metrics:
            return 5.0
        
        status_scores = {
            HealthStatus.EXCELLENT: 10.0,
            HealthStatus.HEALTHY: 8.0,
            HealthStatus.DEGRADED: 6.0,
            HealthStatus.UNHEALTHY: 3.0,
            HealthStatus.CRITICAL: 0.0,
            HealthStatus.UNKNOWN: 5.0
        }
        
        scores = [
            status_scores.get(metric.status, 5.0) 
            for metric in self.metrics.values() 
            if not metric.is_stale
        ]
        
        return sum(scores) / len(scores) if scores else 5.0
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        overall_status = self._calculate_overall_status()
        overall_score = self._calculate_health_score()
        
        return {
            "status": overall_status.value,
            "overall_score": round(overall_score, 1),
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": {
                "seconds": time.time() - self.start_time,
                "human_readable": str(timedelta(seconds=int(time.time() - self.start_time)))
            },
            "system_metrics": {
                name: {
                    "value": metric.value,
                    "status": metric.status.value,
                    "unit": metric.unit,
                    "age_seconds": metric.age_seconds
                }
                for name, metric in self.metrics.items()
                if not metric.is_stale
            },
            "performance": {
                "last_check": self.last_check_time,
                "check_interval": self.check_interval,
                "total_errors": sum(self.error_counts.values())
            },
            "trends": await self._analyze_trends(),
            "recommendations": self._generate_recommendations()
        }
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get quick health summary"""
        overall_status = self._calculate_overall_status()
        
        return {
            "status": overall_status.value,
            "uptime": time.time() - self.start_time,
            "last_check": self.last_check_time,
            "total_errors": sum(self.error_counts.values()),
            "health_score": round(self._calculate_health_score(), 1)
        }
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze health trends"""
        if len(self.snapshots) < 3:
            return {"status": "insufficient_data"}
        
        recent_snapshots = list(self.snapshots)[-10:]
        
        cpu_trend = self._calculate_trend([s.cpu_percent for s in recent_snapshots])
        memory_trend = self._calculate_trend([s.memory_percent for s in recent_snapshots])
        
        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "analysis_period": "last_10_checks",
            "data_points": len(recent_snapshots)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction for values"""
        if len(values) < 2:
            return {"direction": "unknown", "rate": 0.0}
        
        # Simple trend calculation
        recent = values[-3:] if len(values) >= 3 else values[-2:]
        older = values[:len(recent)]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        change = recent_avg - older_avg
        direction = "increasing" if change > 1.0 else "decreasing" if change < -1.0 else "stable"
        
        return {
            "direction": direction,
            "change": round(change, 2),
            "current_average": round(recent_avg, 2)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        for metric in self.metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                recommendations.append(f"CRITICAL: {metric.name} at {metric.value}{metric.unit} - immediate attention required")
            elif metric.status == HealthStatus.DEGRADED:
                recommendations.append(f"WARNING: {metric.name} at {metric.value}{metric.unit} - monitor closely")
        
        if not recommendations:
            recommendations.append("All systems operating normally")
        
        return recommendations
    
    def record_error(self, error_type: str) -> None:
        """Record error for tracking"""
        self.error_counts[error_type] += 1
        logger.debug(f"Recorded error: {error_type}")
    
    def get_uptime(self) -> timedelta:
        """Get application uptime"""
        return timedelta(seconds=time.time() - self.start_time)


# Global health monitor instance
health_monitor = SystemHealthMonitor()
