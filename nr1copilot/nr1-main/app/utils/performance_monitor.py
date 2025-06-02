"""
Netflix-Grade Performance Monitoring System
Real-time performance tracking with predictive analytics and optimization
"""

import time
import asyncio
import logging
import psutil
from typing import Dict, Any, Optional, List, Deque
from collections import deque, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics
import json
import threading

logger = logging.getLogger(__name__)

@dataclass
class PerformanceSnapshot:
    """Performance snapshot with system and application metrics"""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    active_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    error_rate: float = 0.0
    throughput: float = 0.0

class PerformanceMonitor:
    """Netflix-tier performance monitoring and optimization"""

    def __init__(self, snapshot_interval: int = 60):
        self.snapshot_interval = snapshot_interval
        self.snapshots: Deque[PerformanceSnapshot] = deque(maxlen=1440)  # 24 hours
        self.request_times: Deque[float] = deque(maxlen=1000)
        self.active_requests = 0
        self.total_requests = 0
        self.error_count = 0
        self.monitoring_active = False
        self.startup_time = time.time()
        self.performance_alerts: List[Dict[str, Any]] = []
        self._initialized = False

        # Performance thresholds
        self.thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 85.0,
            "memory_warning": 75.0,
            "memory_critical": 90.0,
            "response_time_warning": 1.0,
            "response_time_critical": 3.0,
            "error_rate_warning": 0.05,  # 5%
            "error_rate_critical": 0.10   # 10%
        }

    async def initialize(self):
        """Initialize async components when event loop is available"""
        if self._initialized:
            return

        try:
            self._initialized = True
            logger.info("📊 PerformanceMonitor async initialization completed")
        except Exception as e:
            logger.error(f"PerformanceMonitor async initialization failed: {e}")
            raise

    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        logger.info("🔍 Starting Netflix-grade performance monitoring")

        # Start background monitoring task
        asyncio.create_task(self._continuous_monitoring())

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("⏹️ Performance monitoring stopped")

    async def _continuous_monitoring(self):
        """Continuous monitoring loop with intelligent sampling"""
        while self.monitoring_active:
            try:
                snapshot = await self._capture_performance_snapshot()
                self.snapshots.append(snapshot)

                # Analyze performance and generate alerts
                await self._analyze_performance(snapshot)

                # Dynamic sampling based on load
                sleep_time = self._calculate_dynamic_interval(snapshot)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(self.snapshot_interval)

    async def _capture_performance_snapshot(self) -> PerformanceSnapshot:
        """Capture comprehensive performance snapshot"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Network I/O (if available)
            network_io = {}
            try:
                net_io = psutil.net_io_counters()
                network_io = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            except:
                network_io = {"error": "unavailable"}

            # Application metrics
            recent_response_times = list(self.request_times)[-100:]  # Last 100 requests
            error_rate = self.error_count / max(self.total_requests, 1)

            # Calculate throughput (requests per second over last minute)
            current_time = time.time()
            recent_requests = [t for t in self.request_times if current_time - t < 60]
            throughput = len(recent_requests) / 60.0

            snapshot = PerformanceSnapshot(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                network_io=network_io,
                active_requests=self.active_requests,
                response_times=recent_response_times,
                error_rate=error_rate,
                throughput=throughput
            )

            return snapshot

        except Exception as e:
            logger.error(f"Failed to capture performance snapshot: {e}")
            return PerformanceSnapshot()

    async def _analyze_performance(self, snapshot: PerformanceSnapshot):
        """Analyze performance and generate alerts"""
        alerts = []

        # CPU analysis
        if snapshot.cpu_percent > self.thresholds["cpu_critical"]:
            alerts.append({
                "type": "cpu_critical",
                "value": snapshot.cpu_percent,
                "threshold": self.thresholds["cpu_critical"],
                "severity": "critical",
                "message": f"CPU usage critical: {snapshot.cpu_percent:.1f}%"
            })
        elif snapshot.cpu_percent > self.thresholds["cpu_warning"]:
            alerts.append({
                "type": "cpu_warning",
                "value": snapshot.cpu_percent,
                "threshold": self.thresholds["cpu_warning"],
                "severity": "warning",
                "message": f"CPU usage high: {snapshot.cpu_percent:.1f}%"
            })

        # Memory analysis
        if snapshot.memory_percent > self.thresholds["memory_critical"]:
            alerts.append({
                "type": "memory_critical",
                "value": snapshot.memory_percent,
                "threshold": self.thresholds["memory_critical"],
                "severity": "critical",
                "message": f"Memory usage critical: {snapshot.memory_percent:.1f}%"
            })
        elif snapshot.memory_percent > self.thresholds["memory_warning"]:
            alerts.append({
                "type": "memory_warning",
                "value": snapshot.memory_percent,
                "threshold": self.thresholds["memory_warning"],
                "severity": "warning",
                "message": f"Memory usage high: {snapshot.memory_percent:.1f}%"
            })

        # Response time analysis
        if snapshot.response_times:
            avg_response_time = statistics.mean(snapshot.response_times)
            if avg_response_time > self.thresholds["response_time_critical"]:
                alerts.append({
                    "type": "response_time_critical",
                    "value": avg_response_time,
                    "threshold": self.thresholds["response_time_critical"],
                    "severity": "critical",
                    "message": f"Response time critical: {avg_response_time:.2f}s"
                })
            elif avg_response_time > self.thresholds["response_time_warning"]:
                alerts.append({
                    "type": "response_time_warning",
                    "value": avg_response_time,
                    "threshold": self.thresholds["response_time_warning"],
                    "severity": "warning",
                    "message": f"Response time high: {avg_response_time:.2f}s"
                })

        # Error rate analysis
        if snapshot.error_rate > self.thresholds["error_rate_critical"]:
            alerts.append({
                "type": "error_rate_critical",
                "value": snapshot.error_rate,
                "threshold": self.thresholds["error_rate_critical"],
                "severity": "critical",
                "message": f"Error rate critical: {snapshot.error_rate:.1%}"
            })
        elif snapshot.error_rate > self.thresholds["error_rate_warning"]:
            alerts.append({
                "type": "error_rate_warning",
                "value": snapshot.error_rate,
                "threshold": self.thresholds["error_rate_warning"],
                "severity": "warning",
                "message": f"Error rate high: {snapshot.error_rate:.1%}"
            })

        # Store alerts
        for alert in alerts:
            alert["timestamp"] = snapshot.timestamp
            self.performance_alerts.append(alert)
            logger.warning(f"Performance Alert: {alert['message']}")

        # Keep only recent alerts (last 24 hours)
        cutoff_time = time.time() - 86400
        self.performance_alerts = [a for a in self.performance_alerts if a["timestamp"] > cutoff_time]

    def _calculate_dynamic_interval(self, snapshot: PerformanceSnapshot) -> float:
        """Calculate dynamic monitoring interval based on system load"""
        base_interval = self.snapshot_interval

        # Increase frequency during high load
        if snapshot.cpu_percent > 80 or snapshot.memory_percent > 80:
            return base_interval * 0.5  # Monitor more frequently
        elif snapshot.cpu_percent < 30 and snapshot.memory_percent < 50:
            return base_interval * 1.5  # Monitor less frequently during low load

        return base_interval

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"

        recent_avg = sum(values[-3:]) / len(values[-3:])
        older_avg = sum(values[:-3]) / len(values[:-3]) if len(values) > 3 else recent_avg

        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def record_request_start(self):
        """Record the start of a request"""
        self.active_requests += 1
        self.total_requests += 1

    def record_request_end(self, duration: Optional[float] = None):
        """Record the end of a request"""
        self.active_requests = max(0, self.active_requests - 1)
        if duration is not None:
            self.request_times.append(duration)

    def record_error(self):
        """Record an error occurrence"""
        self.error_count += 1

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.snapshots:
            return {"error": "no_data", "message": "No performance data available"}

        latest = self.snapshots[-1]
        uptime = time.time() - self.startup_time

        # Calculate trends
        recent_snapshots = list(self.snapshots)[-10:]  # Last 10 snapshots
        cpu_trend = self._calculate_trend([s.cpu_percent for s in recent_snapshots])
        memory_trend = self._calculate_trend([s.memory_percent for s in recent_snapshots])

        # Performance grade calculation
        performance_grade = self._calculate_performance_grade(latest)

        # Calculate percentiles for response times
        response_times = [t for s in recent_snapshots for t in s.response_times]
        percentiles = {}
        if response_times:
            percentiles = {
                "p50": statistics.median(response_times),
                "p95": statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                "p99": statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times)
            }

        return {
            "uptime_seconds": round(uptime, 2),
            "performance_grade": performance_grade,
            "current_metrics": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_percent": latest.disk_percent,
                "active_requests": latest.active_requests,
                "error_rate": latest.error_rate,
                "throughput_rps": latest.throughput
            },
            "trends": {
                "cpu": cpu_trend,
                "memory": memory_trend
            },
            "response_times": percentiles,
            "totals": {
                "total_requests": self.total_requests,
                "total_errors": self.error_count,
                "success_rate": 1 - (self.error_count / max(self.total_requests, 1))
            },
            "alerts": {
                "active_alerts": len([a for a in self.performance_alerts if time.time() - a["timestamp"] < 3600]),
                "total_alerts_24h": len(self.performance_alerts)
            },
            "netflix_tier": "Enterprise AAA+",
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_historical_data(self, hours: int = 1) -> Dict[str, Any]:
        """Get historical performance data"""
        cutoff_time = time.time() - (hours * 3600)
        historical_snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]

        if not historical_snapshots:
            return {"error": "no_historical_data"}

        # Aggregate data
        timestamps = [s.timestamp for s in historical_snapshots]
        cpu_data = [s.cpu_percent for s in historical_snapshots]
        memory_data = [s.memory_percent for s in historical_snapshots]
        throughput_data = [s.throughput for s in historical_snapshots]

        return {
            "period_hours": hours,
            "data_points": len(historical_snapshots),
            "time_series": {
                "timestamps": [datetime.fromtimestamp(ts).isoformat() for ts in timestamps],
                "cpu_percent": cpu_data,
                "memory_percent": memory_data,
                "throughput_rps": throughput_data
            },
            "statistics": {
                "cpu": {
                    "min": min(cpu_data) if cpu_data else 0,
                    "max": max(cpu_data) if cpu_data else 0,
                    "avg": sum(cpu_data) / len(cpu_data) if cpu_data else 0
                },
                "memory": {
                    "min": min(memory_data) if memory_data else 0,
                    "max": max(memory_data) if memory_data else 0,
                    "avg": sum(memory_data) / len(memory_data) if memory_data else 0
                },
                "throughput": {
                    "min": min(throughput_data) if throughput_data else 0,
                    "max": max(throughput_data) if throughput_data else 0,
                    "avg": sum(throughput_data) / len(throughput_data) if throughput_data else 0
                }
            }
        }

    def _calculate_performance_grade(self, snapshot: PerformanceSnapshot) -> str:
        """Calculate overall performance grade"""
        score = 100

        # CPU impact
        if snapshot.cpu_percent > 80:
            score -= 20
        elif snapshot.cpu_percent > 60:
            score -= 10

        # Memory impact
        if snapshot.memory_percent > 85:
            score -= 20
        elif snapshot.memory_percent > 70:
            score -= 10

        # Response time impact (if available)
        if snapshot.response_times:
            avg_response_time = statistics.mean(snapshot.response_times)
            if avg_response_time > 2.0:
                score -= 30
            elif avg_response_time > 1.0:
                score -= 15

        # Error rate impact
        if snapshot.error_rate > 0.10:
            score -= 25
        elif snapshot.error_rate > 0.05:
            score -= 10

        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "D"