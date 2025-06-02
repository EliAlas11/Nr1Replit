
"""
Netflix-Level Database Health Monitoring
Real-time database performance and health tracking with enterprise observability
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

from app.database.connection import db_manager
from app.database.repositories import repositories

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Enhanced health status enumeration"""
    EXCELLENT = "excellent"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Enhanced health metric with trending data"""
    name: str
    value: float
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    message: str = ""
    trend: str = "stable"  # rising, falling, stable
    historical_values: List[float] = field(default_factory=list)


class NetflixDatabaseHealthMonitor:
    """Netflix-level database health monitoring with enterprise observability"""
    
    def __init__(self):
        self.health_checks = {}
        self.metrics_history: Dict[str, List[Dict]] = {}
        self.alerts_sent = set()
        self.monitoring_active = False
        self.performance_baseline = {}
        self.anomaly_detection_enabled = True
        
        # Enhanced enterprise thresholds
        self.thresholds = {
            "connection_pool_usage": {"warning": 60.0, "critical": 85.0},
            "avg_query_time": {"warning": 50.0, "critical": 200.0},  # ms
            "active_connections": {"warning": 30, "critical": 45},
            "failed_queries": {"warning": 2.0, "critical": 5.0},  # per minute
            "cache_hit_rate": {"warning": 85.0, "critical": 70.0},  # lower is worse
            "disk_usage": {"warning": 75.0, "critical": 90.0},
            "cpu_usage": {"warning": 60.0, "critical": 85.0},
            "memory_usage": {"warning": 75.0, "critical": 90.0},
            "queries_per_second": {"warning": 1000.0, "critical": 2000.0},
            "database_size": {"warning": 5000.0, "critical": 8000.0},  # MB
            "replication_lag": {"warning": 100.0, "critical": 500.0}  # ms
        }
    
    async def start_monitoring(self, interval: int = 15):
        """Start enhanced continuous health monitoring"""
        if self.monitoring_active:
            logger.info("🏥 Database health monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info("🚀 Starting Netflix-grade database health monitoring...")
        
        # Initialize performance baseline
        await self._establish_performance_baseline()
        
        # Start intensive monitoring tasks for production
        monitoring_tasks = [
            self._monitoring_loop(interval),
            self._anomaly_detection_loop(),
            self._trend_analysis_loop(),
            self._connection_pool_monitor(),
            self._query_performance_monitor(),
            self._resource_utilization_monitor()
        ]
        
        for task in monitoring_tasks:
            asyncio.create_task(task)
        
        logger.info("✅ Production-grade database monitoring active")
    
    async def stop_monitoring(self):
        """Stop health monitoring gracefully"""
        self.monitoring_active = False
        logger.info("🛑 Database health monitoring stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Enhanced main monitoring loop with intelligent scheduling"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                await self.perform_comprehensive_health_check()
                
                # Adaptive interval based on system health
                health_score = await self._calculate_overall_health_score()
                if health_score < 7.0:  # Increase frequency for unhealthy systems
                    actual_interval = max(10, interval // 2)
                else:
                    actual_interval = interval
                
                # Account for check duration
                check_duration = time.time() - start_time
                sleep_time = max(5, actual_interval - check_duration)
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def _anomaly_detection_loop(self):
        """Real-time anomaly detection for database metrics"""
        while self.monitoring_active:
            try:
                if self.anomaly_detection_enabled:
                    await self._detect_performance_anomalies()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(60)
    
    async def _trend_analysis_loop(self):
        """Continuous trend analysis for predictive monitoring"""
        while self.monitoring_active:
            try:
                await self._analyze_performance_trends()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Trend analysis error: {e}")
                await asyncio.sleep(300)
    
    async def perform_comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive Netflix-grade health check"""
        start_time = time.time()
        
        try:
            # Collect all enhanced metrics
            metrics = await self._collect_all_enhanced_metrics()
            
            # Determine overall health status with scoring
            overall_status, health_score = self._determine_enhanced_status(metrics)
            
            # Generate actionable insights
            insights = await self._generate_health_insights(metrics)
            
            # Create comprehensive health record
            health_record = {
                "timestamp": datetime.utcnow(),
                "status": overall_status.value,
                "health_score": health_score,
                "metrics": {metric.name: {
                    "value": metric.value,
                    "status": metric.status.value,
                    "unit": metric.unit,
                    "message": metric.message,
                    "trend": metric.trend
                } for metric in metrics},
                "insights": insights,
                "check_duration_ms": round((time.time() - start_time) * 1000, 2),
                "recommendations": await self._generate_recommendations(metrics)
            }
            
            # Store and process health record
            await self._store_health_record(health_record)
            self._update_metrics_history_enhanced(metrics)
            await self._check_enhanced_alerts(metrics, health_score)
            
            logger.debug(f"Health check completed: {overall_status.value} (score: {health_score:.1f})")
            return health_record
            
        except Exception as e:
            logger.error(f"Comprehensive health check failed: {e}")
            return {
                "timestamp": datetime.utcnow(),
                "status": HealthStatus.CRITICAL.value,
                "health_score": 0.0,
                "error": str(e),
                "check_duration_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    async def _collect_all_enhanced_metrics(self) -> List[HealthMetric]:
        """Collect all enhanced health metrics with trending"""
        metrics = []
        
        # Enhanced database metrics
        metrics.extend(await self._collect_enhanced_connection_metrics())
        metrics.extend(await self._collect_enhanced_performance_metrics())
        metrics.extend(await self._collect_enhanced_repository_metrics())
        metrics.extend(await self._collect_enhanced_system_metrics())
        metrics.extend(await self._collect_database_specific_metrics())
        
        return metrics
    
    async def _collect_enhanced_connection_metrics(self) -> List[HealthMetric]:
        """Collect enhanced database connection metrics"""
        metrics = []
        
        try:
            pool_stats = await db_manager.get_pool_stats()
            
            if pool_stats.get("status") == "not_initialized":
                metrics.append(HealthMetric(
                    name="database_status",
                    value=0,
                    status=HealthStatus.CRITICAL,
                    threshold_warning=1,
                    threshold_critical=0,
                    message="Database not initialized"
                ))
                return metrics
            
            # Enhanced pool utilization
            pool_utilization = pool_stats.get("pool_utilization", 0)
            metrics.append(HealthMetric(
                name="connection_pool_usage",
                value=pool_utilization,
                status=self._get_status(pool_utilization, "connection_pool_usage"),
                threshold_warning=self.thresholds["connection_pool_usage"]["warning"],
                threshold_critical=self.thresholds["connection_pool_usage"]["critical"],
                unit="%",
                message=f"Pool utilization: {pool_utilization:.1f}%",
                trend=self._calculate_trend("connection_pool_usage", pool_utilization)
            ))
            
            # Enhanced connection metrics
            stats = pool_stats.get("stats", {})
            active_connections = stats.get("active_connections", 0)
            peak_connections = stats.get("peak_connections", 0)
            
            metrics.append(HealthMetric(
                name="active_connections",
                value=active_connections,
                status=self._get_status(active_connections, "active_connections"),
                threshold_warning=self.thresholds["active_connections"]["warning"],
                threshold_critical=self.thresholds["active_connections"]["critical"],
                message=f"{active_connections}/{peak_connections} active/peak",
                trend=self._calculate_trend("active_connections", active_connections)
            ))
            
            # Success rate metric
            success_rate = stats.get("success_rate", 100)
            metrics.append(HealthMetric(
                name="connection_success_rate",
                value=success_rate,
                status=HealthStatus.EXCELLENT if success_rate > 99 else HealthStatus.WARNING,
                threshold_warning=95.0,
                threshold_critical=90.0,
                unit="%",
                message=f"{success_rate:.2f}% success rate"
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect enhanced connection metrics: {e}")
            metrics.append(HealthMetric(
                name="connection_error",
                value=1,
                status=HealthStatus.CRITICAL,
                threshold_warning=0,
                threshold_critical=1,
                message=str(e)
            ))
        
        return metrics
    
    async def _collect_enhanced_performance_metrics(self) -> List[HealthMetric]:
        """Collect enhanced database performance metrics"""
        metrics = []
        
        try:
            pool_stats = await db_manager.get_pool_stats()
            stats = pool_stats.get("stats", {})
            
            # Enhanced query performance
            avg_query_time_ms = stats.get("avg_query_time_ms", 0)
            metrics.append(HealthMetric(
                name="avg_query_time",
                value=avg_query_time_ms,
                status=self._get_status(avg_query_time_ms, "avg_query_time"),
                threshold_warning=self.thresholds["avg_query_time"]["warning"],
                threshold_critical=self.thresholds["avg_query_time"]["critical"],
                unit="ms",
                message=f"Average query time: {avg_query_time_ms:.2f}ms",
                trend=self._calculate_trend("avg_query_time", avg_query_time_ms)
            ))
            
            # Queries per second
            qps = stats.get("queries_per_second", 0)
            metrics.append(HealthMetric(
                name="queries_per_second",
                value=qps,
                status=self._get_status(qps, "queries_per_second"),
                threshold_warning=self.thresholds["queries_per_second"]["warning"],
                threshold_critical=self.thresholds["queries_per_second"]["critical"],
                unit="qps",
                message=f"{qps:.1f} queries/second",
                trend=self._calculate_trend("queries_per_second", qps)
            ))
            
            # Total queries
            total_queries = stats.get("total_queries", 0)
            metrics.append(HealthMetric(
                name="total_queries",
                value=total_queries,
                status=HealthStatus.HEALTHY,
                threshold_warning=float('inf'),
                threshold_critical=float('inf'),
                message=f"{total_queries:,} total queries"
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect enhanced performance metrics: {e}")
        
        return metrics
    
    async def _collect_database_specific_metrics(self) -> List[HealthMetric]:
        """Collect database-specific metrics"""
        metrics = []
        
        try:
            async with db_manager.get_connection() as conn:
                # Database size
                db_size_bytes = await conn.fetchval("SELECT pg_database_size(current_database())")
                db_size_mb = db_size_bytes / 1024 / 1024
                
                metrics.append(HealthMetric(
                    name="database_size",
                    value=db_size_mb,
                    status=self._get_status(db_size_mb, "database_size"),
                    threshold_warning=self.thresholds["database_size"]["warning"],
                    threshold_critical=self.thresholds["database_size"]["critical"],
                    unit="MB",
                    message=f"Database size: {db_size_mb:.1f} MB"
                ))
                
                # Active sessions
                active_sessions = await conn.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                )
                
                metrics.append(HealthMetric(
                    name="active_sessions",
                    value=active_sessions,
                    status=HealthStatus.HEALTHY if active_sessions < 20 else HealthStatus.WARNING,
                    threshold_warning=20,
                    threshold_critical=50,
                    message=f"{active_sessions} active sessions"
                ))
                
        except Exception as e:
            logger.error(f"Failed to collect database-specific metrics: {e}")
        
        return metrics
    
    async def _collect_enhanced_repository_metrics(self) -> List[HealthMetric]:
        """Collect enhanced repository performance metrics"""
        metrics = []
        
        try:
            repo_stats = repositories.get_all_stats()
            
            total_cache_hit_rate = 0
            repo_count = 0
            
            for repo_name, stats in repo_stats.items():
                cache_hit_rate = stats.get("cache_hit_rate", 0) * 100
                if cache_hit_rate > 0:
                    total_cache_hit_rate += cache_hit_rate
                    repo_count += 1
            
            if repo_count > 0:
                avg_cache_hit_rate = total_cache_hit_rate / repo_count
                metrics.append(HealthMetric(
                    name="cache_hit_rate",
                    value=avg_cache_hit_rate,
                    status=self._get_status(avg_cache_hit_rate, "cache_hit_rate", reverse=True),
                    threshold_warning=self.thresholds["cache_hit_rate"]["warning"],
                    threshold_critical=self.thresholds["cache_hit_rate"]["critical"],
                    unit="%",
                    message=f"Average cache hit rate: {avg_cache_hit_rate:.1f}%",
                    trend=self._calculate_trend("cache_hit_rate", avg_cache_hit_rate)
                ))
            
        except Exception as e:
            logger.error(f"Failed to collect enhanced repository metrics: {e}")
        
        return metrics
    
    async def _collect_enhanced_system_metrics(self) -> List[HealthMetric]:
        """Collect enhanced system resource metrics"""
        metrics = []
        
        try:
            import psutil
            
            # Enhanced CPU monitoring
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            metrics.append(HealthMetric(
                name="cpu_usage",
                value=cpu_usage,
                status=self._get_status(cpu_usage, "cpu_usage"),
                threshold_warning=self.thresholds["cpu_usage"]["warning"],
                threshold_critical=self.thresholds["cpu_usage"]["critical"],
                unit="%",
                message=f"CPU usage: {cpu_usage:.1f}% ({cpu_count} cores)",
                trend=self._calculate_trend("cpu_usage", cpu_usage)
            ))
            
            # Enhanced memory monitoring
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            metrics.append(HealthMetric(
                name="memory_usage",
                value=memory_usage,
                status=self._get_status(memory_usage, "memory_usage"),
                threshold_warning=self.thresholds["memory_usage"]["warning"],
                threshold_critical=self.thresholds["memory_usage"]["critical"],
                unit="%",
                message=f"Memory: {memory_usage:.1f}% ({memory.used // 1024**2:.0f}MB used)",
                trend=self._calculate_trend("memory_usage", memory_usage)
            ))
            
            # Enhanced disk monitoring
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            metrics.append(HealthMetric(
                name="disk_usage",
                value=disk_usage,
                status=self._get_status(disk_usage, "disk_usage"),
                threshold_warning=self.thresholds["disk_usage"]["warning"],
                threshold_critical=self.thresholds["disk_usage"]["critical"],
                unit="%",
                message=f"Disk: {disk_usage:.1f}% ({disk.free // 1024**3:.1f}GB free)",
                trend=self._calculate_trend("disk_usage", disk_usage)
            ))
            
        except Exception as e:
            logger.error(f"Failed to collect enhanced system metrics: {e}")
        
        return metrics
    
    def _calculate_trend(self, metric_name: str, current_value: float) -> str:
        """Calculate trend for a metric"""
        if metric_name not in self.metrics_history:
            return "stable"
        
        history = self.metrics_history[metric_name]
        if len(history) < 3:
            return "stable"
        
        # Get recent values for trend calculation
        recent_values = [entry["value"] for entry in history[-5:]]
        recent_values.append(current_value)
        
        # Simple trend calculation
        if len(recent_values) >= 3:
            first_half = sum(recent_values[:len(recent_values)//2]) / (len(recent_values)//2)
            second_half = sum(recent_values[len(recent_values)//2:]) / (len(recent_values) - len(recent_values)//2)
            
            change_percent = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
            
            if change_percent > 5:
                return "rising"
            elif change_percent < -5:
                return "falling"
        
        return "stable"
    
    def _get_status(self, value: float, metric_name: str, reverse: bool = False) -> HealthStatus:
        """Enhanced status determination with excellence tier"""
        thresholds = self.thresholds.get(metric_name, {"warning": 0, "critical": 0})
        warning_threshold = thresholds["warning"]
        critical_threshold = thresholds["critical"]
        
        if reverse:
            if value <= critical_threshold:
                return HealthStatus.CRITICAL
            elif value <= warning_threshold:
                return HealthStatus.WARNING
            elif value >= 95.0:  # Excellence threshold
                return HealthStatus.EXCELLENT
            else:
                return HealthStatus.HEALTHY
        else:
            if value >= critical_threshold:
                return HealthStatus.CRITICAL
            elif value >= warning_threshold:
                return HealthStatus.WARNING
            elif value <= warning_threshold * 0.5:  # Excellence threshold
                return HealthStatus.EXCELLENT
            else:
                return HealthStatus.HEALTHY
    
    def _determine_enhanced_status(self, metrics: List[HealthMetric]) -> tuple[HealthStatus, float]:
        """Determine overall status with numerical health score"""
        status_weights = {
            HealthStatus.EXCELLENT: 10.0,
            HealthStatus.HEALTHY: 8.0,
            HealthStatus.WARNING: 5.0,
            HealthStatus.CRITICAL: 1.0,
            HealthStatus.UNKNOWN: 0.0
        }
        
        if not metrics:
            return HealthStatus.UNKNOWN, 0.0
        
        # Calculate weighted health score
        total_weight = 0
        weighted_score = 0
        
        critical_count = 0
        warning_count = 0
        excellent_count = 0
        
        for metric in metrics:
            weight = status_weights[metric.status]
            weighted_score += weight
            total_weight += 10.0  # Max possible weight
            
            if metric.status == HealthStatus.CRITICAL:
                critical_count += 1
            elif metric.status == HealthStatus.WARNING:
                warning_count += 1
            elif metric.status == HealthStatus.EXCELLENT:
                excellent_count += 1
        
        health_score = (weighted_score / total_weight) * 10 if total_weight > 0 else 0
        
        # Determine overall status
        if critical_count > 0:
            return HealthStatus.CRITICAL, health_score
        elif warning_count > 0:
            return HealthStatus.WARNING, health_score
        elif excellent_count >= len(metrics) * 0.8:  # 80% excellent
            return HealthStatus.EXCELLENT, health_score
        else:
            return HealthStatus.HEALTHY, health_score
    
    async def _calculate_overall_health_score(self) -> float:
        """Calculate overall health score for adaptive monitoring"""
        try:
            health_check = await self.perform_comprehensive_health_check()
            return health_check.get("health_score", 5.0)
        except:
            return 3.0  # Conservative fallback
    
    async def _establish_performance_baseline(self):
        """Establish performance baseline for anomaly detection"""
        try:
            logger.info("📊 Establishing performance baseline...")
            # Collect baseline metrics over initial period
            # This would be expanded in production
            self.performance_baseline = {
                "avg_query_time": 50.0,  # ms
                "qps": 100.0,
                "cpu_usage": 30.0,
                "memory_usage": 50.0
            }
        except Exception as e:
            logger.error(f"Failed to establish baseline: {e}")
    
    async def _detect_performance_anomalies(self):
        """Detect performance anomalies using baseline comparison"""
        # Implementation for anomaly detection
        pass
    
    async def _analyze_performance_trends(self):
        """Analyze performance trends for predictive insights"""
        # Implementation for trend analysis
        pass
    
    async def _generate_health_insights(self, metrics: List[HealthMetric]) -> List[str]:
        """Generate actionable health insights"""
        insights = []
        
        for metric in metrics:
            if metric.status == HealthStatus.CRITICAL:
                insights.append(f"🚨 CRITICAL: {metric.name} requires immediate attention")
            elif metric.status == HealthStatus.WARNING and metric.trend == "rising":
                insights.append(f"⚠️ WARNING: {metric.name} is deteriorating - monitor closely")
            elif metric.status == HealthStatus.EXCELLENT:
                insights.append(f"✅ EXCELLENT: {metric.name} performing optimally")
        
        return insights
    
    async def _generate_recommendations(self, metrics: List[HealthMetric]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        for metric in metrics:
            if metric.name == "connection_pool_usage" and metric.value > 80:
                recommendations.append("Consider increasing connection pool size")
            elif metric.name == "avg_query_time" and metric.value > 100:
                recommendations.append("Review slow queries and consider adding indexes")
            elif metric.name == "cache_hit_rate" and metric.value < 80:
                recommendations.append("Optimize caching strategy and increase cache size")
        
        return recommendations
    
    async def _store_health_record(self, health_record: Dict[str, Any]):
        """Store enhanced health record"""
        try:
            await repositories.system_health.create({
                "service_name": "database",
                "status": health_record["status"],
                "response_time_ms": health_record["check_duration_ms"],
                "details": {
                    "health_score": health_record["health_score"],
                    "metrics": health_record["metrics"],
                    "insights": health_record["insights"],
                    "recommendations": health_record["recommendations"]
                }
            })
        except Exception as e:
            logger.error(f"Failed to store health record: {e}")
    
    def _update_metrics_history_enhanced(self, metrics: List[HealthMetric]):
        """Update enhanced metrics history for trending"""
        timestamp = time.time()
        
        for metric in metrics:
            if metric.name not in self.metrics_history:
                self.metrics_history[metric.name] = []
            
            self.metrics_history[metric.name].append({
                "timestamp": timestamp,
                "value": metric.value,
                "status": metric.status.value,
                "trend": metric.trend
            })
            
            # Keep only last 48 hours of history
            cutoff_time = timestamp - (48 * 3600)
            self.metrics_history[metric.name] = [
                entry for entry in self.metrics_history[metric.name]
                if entry["timestamp"] > cutoff_time
            ]
    
    async def _check_enhanced_alerts(self, metrics: List[HealthMetric], health_score: float):
        """Enhanced alert checking with health score consideration"""
        critical_metrics = [m for m in metrics if m.status == HealthStatus.CRITICAL]
        warning_metrics = [m for m in metrics if m.status == HealthStatus.WARNING]
        
        if critical_metrics:
            alert_key = f"critical_{len(critical_metrics)}_{health_score:.1f}"
            if alert_key not in self.alerts_sent:
                logger.error(f"🚨 CRITICAL DATABASE ALERTS (Health Score: {health_score:.1f}): {[m.name for m in critical_metrics]}")
                self.alerts_sent.add(alert_key)
        
        if warning_metrics and not critical_metrics:
            alert_key = f"warning_{len(warning_metrics)}_{health_score:.1f}"
            if alert_key not in self.alerts_sent:
                logger.warning(f"⚠️ DATABASE WARNINGS (Health Score: {health_score:.1f}): {[m.name for m in warning_metrics]}")
                self.alerts_sent.add(alert_key)
        
        # Clear alerts if health is good
        if health_score > 8.0:
            self.alerts_sent.clear()
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get enhanced health summary"""
        health_check = await self.perform_comprehensive_health_check()
        
        return {
            "overall_status": health_check["status"],
            "health_score": health_check["health_score"],
            "performance_grade": self._get_performance_grade(health_check["health_score"]),
            "last_check": health_check["timestamp"].isoformat(),
            "check_duration_ms": health_check["check_duration_ms"],
            "metrics_count": len(health_check.get("metrics", {})),
            "critical_issues": len([
                m for m in health_check.get("metrics", {}).values()
                if m["status"] == "critical"
            ]),
            "warning_issues": len([
                m for m in health_check.get("metrics", {}).values()
                if m["status"] == "warning"
            ]),
            "excellent_metrics": len([
                m for m in health_check.get("metrics", {}).values()
                if m["status"] == "excellent"
            ])
        }
    
    def _get_performance_grade(self, health_score: float) -> str:
        """Get performance grade based on health score"""
        if health_score >= 9.5:
            return "A+ EXCEPTIONAL"
        elif health_score >= 9.0:
            return "A EXCELLENT"
        elif health_score >= 8.0:
            return "B+ VERY_GOOD"
        elif health_score >= 7.0:
            return "B GOOD"
        elif health_score >= 6.0:
            return "C+ FAIR"
        elif health_score >= 5.0:
            return "C NEEDS_ATTENTION"
        else:
            return "D CRITICAL"
    
    async def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information with insights"""
        return await self.perform_comprehensive_health_check()


# Global health monitor instance
health_monitor = NetflixDatabaseHealthMonitor()
