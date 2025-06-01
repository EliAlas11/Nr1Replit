
"""
Netflix-Level Health Monitoring
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ViralClip Pro - SendShort.ai Killer",
        "version": "3.0.0",
        "uptime": time.time()
    }

async def detailed_health_check(redis_client=None) -> Dict[str, Any]:
    """Netflix-level detailed health check"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ViralClip Pro",
        "version": "3.0.0",
        "checks": {}
    }
    
    # System resources
    try:
        health_data["checks"]["system"] = {
            "status": "healthy",
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
        
        # Check if resources are critical
        if (health_data["checks"]["system"]["cpu_percent"] > 90 or 
            health_data["checks"]["system"]["memory_percent"] > 90 or
            health_data["checks"]["system"]["disk_percent"] > 95):
            health_data["checks"]["system"]["status"] = "critical"
            health_data["status"] = "degraded"
            
    except Exception as e:
        health_data["checks"]["system"] = {
            "status": "error",
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    # Redis connectivity
    if redis_client:
        try:
            await redis_client.ping()
            health_data["checks"]["redis"] = {
                "status": "healthy",
                "connected": True
            }
        except Exception as e:
            health_data["checks"]["redis"] = {
                "status": "error",
                "connected": False,
                "error": str(e)
            }
            health_data["status"] = "degraded"
    
    # File system checks
    try:
        directories = ["uploads", "output", "temp"]
        filesystem_status = "healthy"
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Check write permissions
            test_file = os.path.join(directory, "health_check_test.tmp")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception:
                filesystem_status = "error"
                break
        
        health_data["checks"]["filesystem"] = {
            "status": filesystem_status,
            "directories": directories
        }
        
        if filesystem_status == "error":
            health_data["status"] = "degraded"
            
    except Exception as e:
        health_data["checks"]["filesystem"] = {
            "status": "error",
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    # FFmpeg availability
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        
        health_data["checks"]["ffmpeg"] = {
            "status": "healthy" if result.returncode == 0 else "error",
            "available": result.returncode == 0
        }
        
        if result.returncode != 0:
            health_data["status"] = "degraded"
            
    except Exception as e:
        health_data["checks"]["ffmpeg"] = {
            "status": "error",
            "available": False,
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    return health_data

def get_system_metrics() -> Dict[str, Any]:
    """Get detailed system metrics"""
    try:
        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
            "network": psutil.net_io_counters()._asdict(),
            "processes": len(psutil.pids())
        }
    except Exception as e:
        return {"error": str(e)}
