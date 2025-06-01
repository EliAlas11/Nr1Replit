
"""
Netflix-Level Health Checker
Comprehensive system health monitoring
"""

import asyncio
import time
import psutil
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthChecker:
    """Netflix-level health monitoring system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.check_history = []
        self.max_history = 100
        
    async def comprehensive_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        try:
            start_time = time.time()
            
            # Perform all health checks
            system_health = self._check_system_resources()
            disk_health = self._check_disk_space()
            memory_health = self._check_memory_usage()
            
            # Overall status determination
            all_checks = [system_health, disk_health, memory_health]
            failed_checks = [check for check in all_checks if check.get("status") != "healthy"]
            
            overall_status = "healthy" if not failed_checks else "degraded"
            if len(failed_checks) >= len(all_checks) / 2:
                overall_status = "unhealthy"
            
            # Calculate uptime
            uptime = time.time() - self.start_time
            
            result = {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": round(uptime, 2),
                "checks": {
                    "system": system_health,
                    "disk": disk_health,
                    "memory": memory_health
                },
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Store in history
            self._add_to_history(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            
            status = "healthy"
            if cpu_percent > 90:
                status = "critical"
            elif cpu_percent > 75:
                status = "warning"
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "load_average": load_avg,
                "cpu_count": psutil.cpu_count()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            disk_usage = shutil.disk_usage(".")
            total_gb = disk_usage.total / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            used_percent = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
            
            status = "healthy"
            if used_percent > 95:
                status = "critical"
            elif used_percent > 85:
                status = "warning"
            
            return {
                "status": status,
                "total_gb": round(total_gb, 2),
                "free_gb": round(free_gb, 2),
                "used_percent": round(used_percent, 2)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if memory.percent > 95:
                status = "critical"
            elif memory.percent > 85:
                status = "warning"
            
            return {
                "status": status,
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": memory.percent
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _add_to_history(self, result: Dict[str, Any]):
        """Add check result to history"""
        self.check_history.append(result)
        if len(self.check_history) > self.max_history:
            self.check_history.pop(0)
    
    def get_health_history(self) -> List[Dict[str, Any]]:
        """Get health check history"""
        return self.check_history.copy()

class HealthChecker:
    """Comprehensive health monitoring system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.health_checks = {
            "system": self._check_system_resources,
            "disk": self._check_disk_space,
            "directories": self._check_directories,
            "dependencies": self._check_dependencies,
            "memory": self._check_memory_usage,
            "performance": self._check_performance
        }
        
        self.thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "disk_percent": 90,
            "min_disk_gb": 1.0,
            "response_time_ms": 1000
        }
        
        # Performance tracking
        self.performance_history = []
        self.max_history = 100
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks"""
        start_time = time.time()
        results = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(time.time() - self.start_time),
            "checks": {},
            "overall_healthy": True,
            "warnings": [],
            "errors": []
        }
        
        # Run all health checks
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = await check_func()
                results["checks"][check_name] = check_result
                
                if not check_result.get("healthy", True):
                    results["overall_healthy"] = False
                    
                if check_result.get("warnings"):
                    results["warnings"].extend(check_result["warnings"])
                    
                if check_result.get("errors"):
                    results["errors"].extend(check_result["errors"])
                    
            except Exception as e:
                results["checks"][check_name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results["overall_healthy"] = False
                results["errors"].append(f"{check_name}: {str(e)}")
        
        # Track performance
        check_duration = (time.time() - start_time) * 1000  # ms
        results["health_check_duration_ms"] = round(check_duration, 2)
        
        self._track_performance(check_duration)
        
        return results
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system CPU and basic resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            result = {
                "healthy": True,
                "cpu_percent": cpu_percent,
                "load_average": load_avg,
                "warnings": [],
                "errors": []
            }
            
            if cpu_percent > self.thresholds["cpu_percent"]:
                result["healthy"] = False
                result["errors"].append(f"High CPU usage: {cpu_percent}%")
            elif cpu_percent > self.thresholds["cpu_percent"] * 0.8:
                result["warnings"].append(f"Elevated CPU usage: {cpu_percent}%")
            
            return result
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            result = {
                "healthy": True,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available // (1024 * 1024),
                "swap_percent": swap.percent,
                "warnings": [],
                "errors": []
            }
            
            if memory.percent > self.thresholds["memory_percent"]:
                result["healthy"] = False
                result["errors"].append(f"High memory usage: {memory.percent}%")
            elif memory.percent > self.thresholds["memory_percent"] * 0.8:
                result["warnings"].append(f"Elevated memory usage: {memory.percent}%")
            
            if swap.percent > 50:
                result["warnings"].append(f"High swap usage: {swap.percent}%")
            
            return result
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space for critical directories"""
        try:
            directories = [
                Path("uploads"),
                Path("output"), 
                Path("temp"),
                Path("cache"),
                Path(".")  # Current directory
            ]
            
            result = {
                "healthy": True,
                "disk_usage": {},
                "warnings": [],
                "errors": []
            }
            
            for directory in directories:
                if directory.exists():
                    total, used, free = shutil.disk_usage(directory)
                    
                    total_gb = total / (1024**3)
                    used_gb = used / (1024**3)
                    free_gb = free / (1024**3)
                    used_percent = (used / total) * 100
                    
                    dir_info = {
                        "total_gb": round(total_gb, 2),
                        "used_gb": round(used_gb, 2),
                        "free_gb": round(free_gb, 2),
                        "used_percent": round(used_percent, 2)
                    }
                    
                    result["disk_usage"][str(directory)] = dir_info
                    
                    # Check thresholds
                    if used_percent > self.thresholds["disk_percent"]:
                        result["healthy"] = False
                        result["errors"].append(f"Low disk space in {directory}: {used_percent:.1f}% used")
                    elif free_gb < self.thresholds["min_disk_gb"]:
                        result["healthy"] = False
                        result["errors"].append(f"Critical disk space in {directory}: {free_gb:.1f}GB free")
                    elif used_percent > self.thresholds["disk_percent"] * 0.8:
                        result["warnings"].append(f"Elevated disk usage in {directory}: {used_percent:.1f}%")
            
            return result
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_directories(self) -> Dict[str, Any]:
        """Check required directories exist and are writable"""
        try:
            required_dirs = [
                "uploads",
                "output", 
                "temp",
                "cache",
                "logs"
            ]
            
            result = {
                "healthy": True,
                "directories": {},
                "warnings": [],
                "errors": []
            }
            
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                
                dir_status = {
                    "exists": dir_path.exists(),
                    "writable": False,
                    "readable": False
                }
                
                if dir_path.exists():
                    dir_status["writable"] = os.access(dir_path, os.W_OK)
                    dir_status["readable"] = os.access(dir_path, os.R_OK)
                    
                    if not dir_status["writable"]:
                        result["healthy"] = False
                        result["errors"].append(f"Directory not writable: {dir_name}")
                        
                    if not dir_status["readable"]:
                        result["healthy"] = False
                        result["errors"].append(f"Directory not readable: {dir_name}")
                else:
                    result["warnings"].append(f"Directory missing: {dir_name}")
                    # Try to create it
                    try:
                        dir_path.mkdir(exist_ok=True, parents=True)
                        dir_status["exists"] = True
                        dir_status["writable"] = True
                        dir_status["readable"] = True
                    except Exception:
                        result["healthy"] = False
                        result["errors"].append(f"Cannot create directory: {dir_name}")
                
                result["directories"][dir_name] = dir_status
            
            return result
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies"""
        try:
            dependencies = {
                "fastapi": "FastAPI framework",
                "pydantic": "Data validation",
                "uvicorn": "ASGI server",
                "psutil": "System monitoring"
            }
            
            result = {
                "healthy": True,
                "dependencies": {},
                "warnings": [],
                "errors": []
            }
            
            for dep_name, description in dependencies.items():
                try:
                    __import__(dep_name)
                    result["dependencies"][dep_name] = {
                        "available": True,
                        "description": description
                    }
                except ImportError:
                    result["healthy"] = False
                    result["dependencies"][dep_name] = {
                        "available": False,
                        "description": description
                    }
                    result["errors"].append(f"Missing dependency: {dep_name}")
            
            return result
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_performance(self) -> Dict[str, Any]:
        """Check application performance metrics"""
        try:
            # Simple performance test
            start_time = time.time()
            
            # Simulate some work
            await asyncio.sleep(0.001)
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            result = {
                "healthy": True,
                "response_time_ms": round(response_time, 2),
                "performance_history_length": len(self.performance_history),
                "warnings": [],
                "errors": []
            }
            
            if response_time > self.thresholds["response_time_ms"]:
                result["healthy"] = False
                result["errors"].append(f"Slow response time: {response_time:.2f}ms")
            elif response_time > self.thresholds["response_time_ms"] * 0.7:
                result["warnings"].append(f"Elevated response time: {response_time:.2f}ms")
            
            # Calculate average from history
            if self.performance_history:
                avg_response = sum(self.performance_history) / len(self.performance_history)
                result["average_response_time_ms"] = round(avg_response, 2)
            
            return result
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _track_performance(self, duration_ms: float):
        """Track performance metrics"""
        self.performance_history.append(duration_ms)
        
        # Keep only recent history
        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)
    
    async def quick_health_check(self) -> Dict[str, Any]:
        """Quick health check for frequent monitoring"""
        try:
            start_time = time.time()
            
            # Basic checks
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            duration = (time.time() - start_time) * 1000
            
            healthy = (
                cpu_percent < self.thresholds["cpu_percent"] and
                memory.percent < self.thresholds["memory_percent"]
            )
            
            return {
                "healthy": healthy,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "check_duration_ms": round(duration, 2),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

import os
