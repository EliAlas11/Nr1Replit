
"""
Netflix-Grade Dependency Container v11.0
Enterprise service container with lifecycle management and health monitoring
"""

import asyncio
import logging
import time
import weakref
from typing import Dict, Any, Optional, Type, TypeVar, Generic, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceState(Enum):
    """Service lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ServiceDescriptor:
    """Service descriptor with metadata and lifecycle information"""
    name: str
    service_type: Type
    instance: Optional[Any] = None
    state: ServiceState = ServiceState.UNINITIALIZED
    dependencies: List[str] = field(default_factory=list)
    startup_order: int = 100
    is_singleton: bool = True
    is_critical: bool = True
    health_check_enabled: bool = True
    created_at: float = field(default_factory=time.time)
    last_health_check: float = 0
    error_count: int = 0
    restart_count: int = 0
    max_restart_attempts: int = 3


class ServiceHealth:
    """Service health information"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_healthy = True
        self.last_check = time.time()
        self.error_message = ""
        self.response_time_ms = 0.0
        self.uptime_seconds = 0.0


class ServiceContainer:
    """Netflix-grade dependency injection container with lifecycle management"""
    
    def __init__(self):
        self.services: Dict[str, ServiceDescriptor] = {}
        self.service_instances: Dict[str, Any] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.initialization_lock = asyncio.Lock()
        self.is_initialized = False
        self.start_time = time.time()
        
        # Service registry
        self.factory_functions: Dict[str, Callable] = {}
        self.singleton_instances: Dict[str, Any] = {}
        
        # Health monitoring
        self.health_check_interval = 60
        self.health_monitor_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = {
            "services_registered": 0,
            "services_initialized": 0,
            "services_running": 0,
            "health_checks_performed": 0,
            "errors_encountered": 0
        }
        
        logger.info("🏗️ Netflix-grade Service Container initialized")
    
    def register_service(
        self,
        name: str,
        service_type: Type[T],
        dependencies: Optional[List[str]] = None,
        startup_order: int = 100,
        is_singleton: bool = True,
        is_critical: bool = True,
        factory: Optional[Callable[[], T]] = None
    ) -> None:
        """Register a service with the container"""
        
        descriptor = ServiceDescriptor(
            name=name,
            service_type=service_type,
            dependencies=dependencies or [],
            startup_order=startup_order,
            is_singleton=is_singleton,
            is_critical=is_critical
        )
        
        self.services[name] = descriptor
        self.service_health[name] = ServiceHealth(name)
        
        if factory:
            self.factory_functions[name] = factory
        
        self.metrics["services_registered"] += 1
        
        logger.debug(f"📝 Registered service: {name} (critical: {is_critical})")
    
    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton instance"""
        self.singleton_instances[name] = instance
        self.service_health[name] = ServiceHealth(name)
        logger.debug(f"🔧 Registered singleton: {name}")
    
    async def get_service(self, name: str) -> Optional[Any]:
        """Get a service instance with dependency resolution"""
        if name in self.singleton_instances:
            return self.singleton_instances[name]
        
        if name not in self.services:
            logger.error(f"❌ Service not found: {name}")
            return None
        
        descriptor = self.services[name]
        
        # Check if already instantiated
        if descriptor.instance and descriptor.is_singleton:
            return descriptor.instance
        
        try:
            # Create new instance
            if name in self.factory_functions:
                instance = self.factory_functions[name]()
            else:
                instance = descriptor.service_type()
            
            # Store singleton instance
            if descriptor.is_singleton:
                descriptor.instance = instance
                descriptor.state = ServiceState.INITIALIZED
            
            return instance
            
        except Exception as e:
            logger.error(f"❌ Failed to create service {name}: {e}")
            descriptor.error_count += 1
            descriptor.state = ServiceState.ERROR
            return None
    
    async def initialize(self) -> None:
        """Initialize all services in dependency order"""
        async with self.initialization_lock:
            if self.is_initialized:
                return
            
            logger.info("🚀 Initializing Netflix-grade services...")
            
            try:
                # Register core services
                await self._register_core_services()
                
                # Initialize services in dependency order
                await self._initialize_services_in_order()
                
                # Start health monitoring
                await self._start_health_monitoring()
                
                self.is_initialized = True
                self.metrics["services_initialized"] = len(self.services)
                
                logger.info(f"✅ {len(self.services)} services initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Service initialization failed: {e}")
                raise
    
    async def _register_core_services(self) -> None:
        """Register core Netflix-grade services"""
        
        # Health monitor
        try:
            from app.netflix_health_monitor import NetflixHealthMonitor
            self.register_service(
                "health_monitor",
                NetflixHealthMonitor,
                startup_order=1,
                is_critical=True
            )
        except ImportError:
            logger.warning("Health monitor not available")
        
        # Metrics collector
        try:
            from app.utils.metrics import MetricsCollector
            self.register_service(
                "metrics_collector",
                MetricsCollector,
                startup_order=2,
                is_critical=True
            )
        except ImportError:
            logger.warning("Metrics collector not available")
        
        # Cache manager
        try:
            from app.utils.cache import CacheManager
            self.register_service(
                "cache_manager",
                CacheManager,
                startup_order=3,
                is_critical=False
            )
        except ImportError:
            logger.warning("Cache manager not available")
        
        # Database manager
        try:
            from app.database.connection import DatabaseManager
            self.register_service(
                "database_manager",
                DatabaseManager,
                startup_order=4,
                is_critical=True
            )
        except ImportError:
            logger.warning("Database manager not available")
        
        # Video service
        try:
            from app.services.video_service import NetflixLevelVideoService
            self.register_service(
                "video_service",
                NetflixLevelVideoService,
                dependencies=["database_manager", "cache_manager"],
                startup_order=10,
                is_critical=True
            )
        except ImportError:
            logger.warning("Video service not available")
        
        # AI engine
        try:
            from app.services.ai_intelligence_engine import AIIntelligenceEngine
            self.register_service(
                "ai_engine",
                AIIntelligenceEngine,
                dependencies=["cache_manager"],
                startup_order=11,
                is_critical=False
            )
        except ImportError:
            logger.warning("AI engine not available")
        
        # AI production engine
        try:
            from app.services.ai_production_engine import AIProductionEngine
            self.register_service(
                "ai_production_engine",
                AIProductionEngine,
                dependencies=["cache_manager"],
                startup_order=12,
                is_critical=True
            )
        except ImportError:
            logger.warning("AI production engine not available")
    
    async def _initialize_services_in_order(self) -> None:
        """Initialize services in dependency order"""
        
        # Sort services by startup order
        sorted_services = sorted(
            self.services.items(),
            key=lambda x: x[1].startup_order
        )
        
        for name, descriptor in sorted_services:
            try:
                await self._initialize_single_service(name, descriptor)
            except Exception as e:
                if descriptor.is_critical:
                    logger.error(f"❌ Critical service {name} failed to initialize: {e}")
                    raise
                else:
                    logger.warning(f"⚠️ Non-critical service {name} failed to initialize: {e}")
                    descriptor.state = ServiceState.ERROR
    
    async def _initialize_single_service(self, name: str, descriptor: ServiceDescriptor) -> None:
        """Initialize a single service"""
        
        if descriptor.state != ServiceState.UNINITIALIZED:
            return
        
        logger.debug(f"🔧 Initializing service: {name}")
        descriptor.state = ServiceState.INITIALIZING
        
        try:
            # Check dependencies
            for dep_name in descriptor.dependencies:
                if dep_name not in self.services:
                    raise ValueError(f"Missing dependency: {dep_name}")
                
                dep_descriptor = self.services[dep_name]
                if dep_descriptor.state not in [ServiceState.INITIALIZED, ServiceState.RUNNING]:
                    await self._initialize_single_service(dep_name, dep_descriptor)
            
            # Create service instance
            instance = await self.get_service(name)
            if instance is None:
                raise RuntimeError(f"Failed to create service instance: {name}")
            
            # Initialize if it has startup method
            if hasattr(instance, 'startup') and callable(getattr(instance, 'startup')):
                if inspect.iscoroutinefunction(instance.startup):
                    await instance.startup()
                else:
                    instance.startup()
            
            descriptor.state = ServiceState.RUNNING
            self.metrics["services_running"] += 1
            
            logger.debug(f"✅ Service initialized: {name}")
            
        except Exception as e:
            descriptor.state = ServiceState.ERROR
            descriptor.error_count += 1
            logger.error(f"❌ Service initialization failed: {name} - {e}")
            raise
    
    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring"""
        if self.health_monitor_task is None:
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            logger.info("🏥 Health monitoring started")
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all services"""
        
        for name, descriptor in self.services.items():
            if not descriptor.health_check_enabled:
                continue
            
            try:
                health = self.service_health[name]
                start_time = time.time()
                
                # Get service instance
                instance = descriptor.instance or await self.get_service(name)
                
                if instance is None:
                    health.is_healthy = False
                    health.error_message = "Service instance not available"
                    continue
                
                # Perform health check
                is_healthy = True
                error_message = ""
                
                if hasattr(instance, 'health_check'):
                    try:
                        if inspect.iscoroutinefunction(instance.health_check):
                            result = await instance.health_check()
                        else:
                            result = instance.health_check()
                        
                        if isinstance(result, bool):
                            is_healthy = result
                        elif isinstance(result, dict):
                            is_healthy = result.get('healthy', True)
                            error_message = result.get('error', '')
                    except Exception as e:
                        is_healthy = False
                        error_message = str(e)
                
                # Update health status
                health.is_healthy = is_healthy
                health.error_message = error_message
                health.response_time_ms = (time.time() - start_time) * 1000
                health.last_check = time.time()
                health.uptime_seconds = time.time() - descriptor.created_at
                
                if not is_healthy:
                    descriptor.error_count += 1
                    logger.warning(f"⚠️ Service health check failed: {name} - {error_message}")
                
                self.metrics["health_checks_performed"] += 1
                
            except Exception as e:
                logger.error(f"Health check error for {name}: {e}")
                self.metrics["errors_encountered"] += 1
    
    def get_health_monitor(self):
        """Get health monitor instance"""
        return self.singleton_instances.get('health_monitor') or self.services.get('health_monitor', {}).get('instance')
    
    def get_metrics_collector(self):
        """Get metrics collector instance"""
        return self.singleton_instances.get('metrics_collector') or self.services.get('metrics_collector', {}).get('instance')
    
    def get_cache_manager(self):
        """Get cache manager instance"""
        return self.singleton_instances.get('cache_manager') or self.services.get('cache_manager', {}).get('instance')
    
    def get_database_manager(self):
        """Get database manager instance"""
        return self.singleton_instances.get('database_manager') or self.services.get('database_manager', {}).get('instance')
    
    def get_video_service(self):
        """Get video service instance"""
        return self.singleton_instances.get('video_service') or self.services.get('video_service', {}).get('instance')
    
    def get_ai_engine(self):
        """Get AI engine instance"""
        return self.singleton_instances.get('ai_engine') or self.services.get('ai_engine', {}).get('instance')
    
    def get_ai_production_engine(self):
        """Get AI production engine instance"""
        return self.singleton_instances.get('ai_production_engine') or self.services.get('ai_production_engine', {}).get('instance')
    
    def is_healthy(self) -> bool:
        """Check if all critical services are healthy"""
        for name, descriptor in self.services.items():
            if descriptor.is_critical and descriptor.state != ServiceState.RUNNING:
                return False
            
            health = self.service_health.get(name)
            if health and not health.is_healthy and descriptor.is_critical:
                return False
        
        return True
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        status = {
            "total_services": len(self.services),
            "running_services": sum(1 for s in self.services.values() if s.state == ServiceState.RUNNING),
            "healthy_services": sum(1 for h in self.service_health.values() if h.is_healthy),
            "services": {}
        }
        
        for name, descriptor in self.services.items():
            health = self.service_health[name]
            status["services"][name] = {
                "state": descriptor.state.value,
                "healthy": health.is_healthy,
                "error_count": descriptor.error_count,
                "restart_count": descriptor.restart_count,
                "uptime_seconds": health.uptime_seconds,
                "last_health_check": health.last_check,
                "is_critical": descriptor.is_critical
            }
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get container metrics"""
        return {
            **self.metrics,
            "uptime_seconds": time.time() - self.start_time,
            "is_initialized": self.is_initialized,
            "healthy_percentage": (
                sum(1 for h in self.service_health.values() if h.is_healthy) / 
                len(self.service_health) * 100 if self.service_health else 0
            )
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all services"""
        logger.info("🔄 Shutting down services...")
        
        # Stop health monitoring
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown services in reverse order
        sorted_services = sorted(
            self.services.items(),
            key=lambda x: x[1].startup_order,
            reverse=True
        )
        
        for name, descriptor in sorted_services:
            try:
                if descriptor.instance and hasattr(descriptor.instance, 'shutdown'):
                    if inspect.iscoroutinefunction(descriptor.instance.shutdown):
                        await descriptor.instance.shutdown()
                    else:
                        descriptor.instance.shutdown()
                
                descriptor.state = ServiceState.STOPPED
                logger.debug(f"🔌 Shutdown service: {name}")
                
            except Exception as e:
                logger.error(f"Error shutting down service {name}: {e}")
        
        # Clear all instances
        self.service_instances.clear()
        self.singleton_instances.clear()
        
        logger.info("✅ All services shutdown complete")
