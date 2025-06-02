
"""
Netflix-Grade Dependency Container v11.0
Enterprise service container with lifecycle management and health monitoring
"""

import asyncio
import logging
import time
import weakref
from typing import Dict, Any, Optional, Type, TypeVar, Generic, List, Callable, Set
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
    RECOVERING = "recovering"


@dataclass
class ServiceDescriptor:
    """Enhanced service descriptor with metadata and lifecycle information"""
    name: str
    service_type: Type
    instance: Optional[Any] = None
    state: ServiceState = ServiceState.UNINITIALIZED
    dependencies: List[str] = field(default_factory=list)
    startup_order: int = 100
    is_singleton: bool = True
    is_critical: bool = True
    health_check_enabled: bool = True
    auto_restart: bool = False
    max_restart_attempts: int = 3
    restart_delay: float = 5.0
    
    # Metrics
    created_at: float = field(default_factory=time.time)
    last_health_check: float = 0
    error_count: int = 0
    restart_count: int = 0
    total_uptime: float = 0
    last_restart: float = 0


class ServiceHealth:
    """Enhanced service health information"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_healthy = True
        self.last_check = time.time()
        self.error_message = ""
        self.response_time_ms = 0.0
        self.uptime_seconds = 0.0
        self.check_count = 0
        self.consecutive_failures = 0
        self.last_success = time.time()


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
        
        # Dependency graph
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Metrics
        self.metrics = {
            "services_registered": 0,
            "services_initialized": 0,
            "services_running": 0,
            "health_checks_performed": 0,
            "errors_encountered": 0,
            "auto_restarts_performed": 0
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
        auto_restart: bool = False,
        factory: Optional[Callable[[], T]] = None
    ) -> None:
        """Register a service with enhanced configuration options"""
        
        dependencies = dependencies or []
        
        descriptor = ServiceDescriptor(
            name=name,
            service_type=service_type,
            dependencies=dependencies,
            startup_order=startup_order,
            is_singleton=is_singleton,
            is_critical=is_critical,
            auto_restart=auto_restart
        )
        
        self.services[name] = descriptor
        self.service_health[name] = ServiceHealth(name)
        
        # Build dependency graph
        for dep in dependencies:
            self._dependency_graph[name].add(dep)
            self._reverse_dependency_graph[dep].add(name)
        
        if factory:
            self.factory_functions[name] = factory
        
        self.metrics["services_registered"] += 1
        
        logger.debug(f"📝 Registered service: {name} (critical: {is_critical}, auto_restart: {auto_restart})")
    
    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton instance with health tracking"""
        self.singleton_instances[name] = instance
        self.service_health[name] = ServiceHealth(name)
        logger.debug(f"🔧 Registered singleton: {name}")
    
    async def get_service(self, name: str) -> Optional[Any]:
        """Get a service instance with enhanced dependency resolution"""
        if name in self.singleton_instances:
            return self.singleton_instances[name]
        
        if name not in self.services:
            logger.error(f"❌ Service not found: {name}")
            return None
        
        descriptor = self.services[name]
        
        # Check if already instantiated and healthy
        if descriptor.instance and descriptor.is_singleton:
            if descriptor.state == ServiceState.RUNNING:
                return descriptor.instance
        
        try:
            # Resolve dependencies first
            await self._resolve_dependencies(name)
            
            # Create new instance
            instance = await self._create_service_instance(name, descriptor)
            
            if descriptor.is_singleton:
                descriptor.instance = instance
                descriptor.state = ServiceState.INITIALIZED
            
            return instance
            
        except Exception as e:
            logger.error(f"❌ Failed to create service {name}: {e}")
            descriptor.error_count += 1
            descriptor.state = ServiceState.ERROR
            self.metrics["errors_encountered"] += 1
            return None
    
    async def _resolve_dependencies(self, service_name: str) -> None:
        """Resolve service dependencies recursively"""
        descriptor = self.services[service_name]
        
        for dep_name in descriptor.dependencies:
            if dep_name not in self.services:
                raise ValueError(f"Missing dependency: {dep_name} for service: {service_name}")
            
            dep_descriptor = self.services[dep_name]
            if dep_descriptor.state not in [ServiceState.INITIALIZED, ServiceState.RUNNING]:
                await self._initialize_single_service(dep_name, dep_descriptor)
    
    async def _create_service_instance(self, name: str, descriptor: ServiceDescriptor) -> Any:
        """Create service instance with factory support"""
        if name in self.factory_functions:
            factory = self.factory_functions[name]
            if inspect.iscoroutinefunction(factory):
                return await factory()
            else:
                return factory()
        else:
            return descriptor.service_type()
    
    async def initialize(self) -> None:
        """Initialize all services in optimized dependency order"""
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
        """Register core Netflix-grade services with optimized configuration"""
        
        core_services = [
            ("health_monitor", "app.netflix_health_monitor", "NetflixHealthMonitor", 1, True, False),
            ("metrics_collector", "app.utils.metrics", "MetricsCollector", 2, True, False),
            ("cache_manager", "app.utils.cache", "CacheManager", 3, False, False),
            ("database_manager", "app.database.connection", "DatabaseManager", 4, True, False),
            ("video_service", "app.services.video_service", "NetflixLevelVideoService", 10, True, True),
            ("ai_engine", "app.services.ai_intelligence_engine", "AIIntelligenceEngine", 11, False, True),
            ("ai_production_engine", "app.services.ai_production_engine", "AIProductionEngine", 12, True, True)
        ]
        
        for name, module_path, class_name, order, critical, auto_restart in core_services:
            try:
                module = __import__(module_path, fromlist=[class_name])
                service_class = getattr(module, class_name)
                
                dependencies = []
                if name == "video_service":
                    dependencies = ["database_manager", "cache_manager"]
                elif name in ["ai_engine", "ai_production_engine"]:
                    dependencies = ["cache_manager"]
                
                self.register_service(
                    name=name,
                    service_type=service_class,
                    dependencies=dependencies,
                    startup_order=order,
                    is_critical=critical,
                    auto_restart=auto_restart
                )
                
            except ImportError:
                logger.warning(f"Service {name} not available")
            except Exception as e:
                logger.error(f"Failed to register {name}: {e}")
    
    async def _initialize_services_in_order(self) -> None:
        """Initialize services in optimized dependency order"""
        
        # Topological sort for dependency resolution
        initialization_order = self._get_initialization_order()
        
        for name in initialization_order:
            descriptor = self.services[name]
            try:
                await self._initialize_single_service(name, descriptor)
            except Exception as e:
                if descriptor.is_critical:
                    logger.error(f"❌ Critical service {name} failed to initialize: {e}")
                    raise
                else:
                    logger.warning(f"⚠️ Non-critical service {name} failed to initialize: {e}")
                    descriptor.state = ServiceState.ERROR
    
    def _get_initialization_order(self) -> List[str]:
        """Get services in topological order based on dependencies"""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            
            # Visit dependencies first
            for dep in self._dependency_graph[name]:
                if dep in self.services:
                    visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)
        
        # Sort by startup order first, then resolve dependencies
        sorted_services = sorted(
            self.services.keys(),
            key=lambda x: self.services[x].startup_order
        )
        
        for service_name in sorted_services:
            visit(service_name)
        
        return order
    
    async def _initialize_single_service(self, name: str, descriptor: ServiceDescriptor) -> None:
        """Initialize a single service with enhanced error handling"""
        
        if descriptor.state in [ServiceState.INITIALIZED, ServiceState.RUNNING]:
            return
        
        logger.debug(f"🔧 Initializing service: {name}")
        descriptor.state = ServiceState.INITIALIZING
        
        try:
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
            
            # Attempt auto-restart for non-critical services
            if descriptor.auto_restart and descriptor.restart_count < descriptor.max_restart_attempts:
                logger.info(f"🔄 Scheduling auto-restart for {name}")
                asyncio.create_task(self._auto_restart_service(name, descriptor))
            
            raise
    
    async def _auto_restart_service(self, name: str, descriptor: ServiceDescriptor) -> None:
        """Auto-restart a failed service"""
        await asyncio.sleep(descriptor.restart_delay)
        
        try:
            descriptor.restart_count += 1
            descriptor.last_restart = time.time()
            descriptor.state = ServiceState.RECOVERING
            
            logger.info(f"🔄 Auto-restarting service: {name} (attempt {descriptor.restart_count})")
            
            await self._initialize_single_service(name, descriptor)
            
            self.metrics["auto_restarts_performed"] += 1
            logger.info(f"✅ Service auto-restart successful: {name}")
            
        except Exception as e:
            logger.error(f"❌ Service auto-restart failed: {name} - {e}")
            descriptor.state = ServiceState.ERROR
    
    async def _start_health_monitoring(self) -> None:
        """Start enhanced background health monitoring"""
        if self.health_monitor_task is None:
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            logger.info("🏥 Health monitoring started")
    
    async def _health_monitor_loop(self) -> None:
        """Enhanced background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def _perform_health_checks(self) -> None:
        """Perform comprehensive health checks on all services"""
        
        for name, descriptor in self.services.items():
            if not descriptor.health_check_enabled or descriptor.state != ServiceState.RUNNING:
                continue
            
            try:
                health = self.service_health[name]
                start_time = time.time()
                
                # Get service instance
                instance = descriptor.instance or await self.get_service(name)
                
                if instance is None:
                    self._update_health_failure(health, "Service instance not available")
                    continue
                
                # Perform health check
                is_healthy = await self._check_service_health(instance)
                
                # Update health status
                response_time = (time.time() - start_time) * 1000
                self._update_health_success(health, response_time)
                
                if not is_healthy:
                    self._update_health_failure(health, "Health check failed")
                    descriptor.error_count += 1
                    
                    # Consider auto-restart for failed services
                    if (descriptor.auto_restart and 
                        health.consecutive_failures >= 3 and 
                        descriptor.restart_count < descriptor.max_restart_attempts):
                        
                        logger.warning(f"⚠️ Service {name} failed health checks, scheduling restart")
                        asyncio.create_task(self._auto_restart_service(name, descriptor))
                
                self.metrics["health_checks_performed"] += 1
                
            except Exception as e:
                logger.error(f"Health check error for {name}: {e}")
                self.metrics["errors_encountered"] += 1
    
    async def _check_service_health(self, instance: Any) -> bool:
        """Check individual service health"""
        if not hasattr(instance, 'health_check'):
            return True
        
        try:
            health_check = getattr(instance, 'health_check')
            if inspect.iscoroutinefunction(health_check):
                result = await health_check()
            else:
                result = health_check()
            
            if isinstance(result, bool):
                return result
            elif isinstance(result, dict):
                return result.get('healthy', True)
            else:
                return True
                
        except Exception:
            return False
    
    def _update_health_success(self, health: ServiceHealth, response_time: float) -> None:
        """Update health metrics for successful check"""
        health.is_healthy = True
        health.error_message = ""
        health.response_time_ms = response_time
        health.last_check = time.time()
        health.last_success = health.last_check
        health.check_count += 1
        health.consecutive_failures = 0
        health.uptime_seconds = time.time() - health.last_success
    
    def _update_health_failure(self, health: ServiceHealth, error_msg: str) -> None:
        """Update health metrics for failed check"""
        health.is_healthy = False
        health.error_message = error_msg
        health.last_check = time.time()
        health.check_count += 1
        health.consecutive_failures += 1
    
    # Enhanced getter methods with fallback support
    def get_health_monitor(self):
        """Get health monitor instance with fallback"""
        return (self.singleton_instances.get('health_monitor') or 
                self.services.get('health_monitor', {}).get('instance'))
    
    def get_metrics_collector(self):
        """Get metrics collector instance with fallback"""
        return (self.singleton_instances.get('metrics_collector') or 
                self.services.get('metrics_collector', {}).get('instance'))
    
    def get_cache_manager(self):
        """Get cache manager instance with fallback"""
        return (self.singleton_instances.get('cache_manager') or 
                self.services.get('cache_manager', {}).get('instance'))
    
    def get_database_manager(self):
        """Get database manager instance with fallback"""
        return (self.singleton_instances.get('database_manager') or 
                self.services.get('database_manager', {}).get('instance'))
    
    def get_video_service(self):
        """Get video service instance with fallback"""
        return (self.singleton_instances.get('video_service') or 
                self.services.get('video_service', {}).get('instance'))
    
    def get_ai_engine(self):
        """Get AI engine instance with fallback"""
        return (self.singleton_instances.get('ai_engine') or 
                self.services.get('ai_engine', {}).get('instance'))
    
    def get_ai_production_engine(self):
        """Get AI production engine instance with fallback"""
        return (self.singleton_instances.get('ai_production_engine') or 
                self.services.get('ai_production_engine', {}).get('instance'))
    
    def is_healthy(self) -> bool:
        """Check if all critical services are healthy with enhanced logic"""
        critical_services_healthy = 0
        total_critical_services = 0
        
        for name, descriptor in self.services.items():
            if descriptor.is_critical:
                total_critical_services += 1
                
                if descriptor.state == ServiceState.RUNNING:
                    health = self.service_health.get(name)
                    if health and health.is_healthy:
                        critical_services_healthy += 1
        
        # Require at least 80% of critical services to be healthy
        if total_critical_services == 0:
            return True
        
        health_percentage = critical_services_healthy / total_critical_services
        return health_percentage >= 0.8
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status with enhanced metrics"""
        running_services = sum(1 for s in self.services.values() if s.state == ServiceState.RUNNING)
        healthy_services = sum(1 for h in self.service_health.values() if h.is_healthy)
        
        status = {
            "total_services": len(self.services),
            "running_services": running_services,
            "healthy_services": healthy_services,
            "critical_services": sum(1 for s in self.services.values() if s.is_critical),
            "health_percentage": (healthy_services / len(self.service_health) * 100) if self.service_health else 100,
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
                "response_time_ms": health.response_time_ms,
                "consecutive_failures": health.consecutive_failures,
                "is_critical": descriptor.is_critical,
                "auto_restart": descriptor.auto_restart
            }
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get enhanced container metrics"""
        uptime = time.time() - self.start_time
        
        return {
            **self.metrics,
            "uptime_seconds": uptime,
            "is_initialized": self.is_initialized,
            "healthy_percentage": (
                sum(1 for h in self.service_health.values() if h.is_healthy) / 
                len(self.service_health) * 100 if self.service_health else 100
            ),
            "average_service_uptime": (
                sum(h.uptime_seconds for h in self.service_health.values()) /
                len(self.service_health) if self.service_health else 0
            ),
            "total_health_checks": sum(h.check_count for h in self.service_health.values())
        }
    
    async def shutdown(self) -> None:
        """Enhanced graceful shutdown with dependency-aware order"""
        logger.info("🔄 Shutting down services...")
        
        # Stop health monitoring
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown services in reverse dependency order
        shutdown_order = list(reversed(self._get_initialization_order()))
        
        for name in shutdown_order:
            descriptor = self.services[name]
            
            if descriptor.state != ServiceState.RUNNING:
                continue
                
            try:
                descriptor.state = ServiceState.STOPPING
                
                if descriptor.instance and hasattr(descriptor.instance, 'shutdown'):
                    shutdown_method = getattr(descriptor.instance, 'shutdown')
                    if inspect.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    else:
                        shutdown_method()
                
                descriptor.state = ServiceState.STOPPED
                logger.debug(f"🔌 Shutdown service: {name}")
                
            except Exception as e:
                logger.error(f"Error shutting down service {name}: {e}")
                descriptor.state = ServiceState.ERROR
        
        # Clear all instances
        self.service_instances.clear()
        self.singleton_instances.clear()
        
        logger.info("✅ All services shutdown complete")
