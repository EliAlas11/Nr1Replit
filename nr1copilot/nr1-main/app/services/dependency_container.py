
"""
ViralClip Pro v6.0 - Netflix-Level Dependency Container
Enterprise dependency injection with advanced service management
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Type, TypeVar, Generic, Set
from datetime import datetime
from abc import ABC, abstractmethod
import weakref
from dataclasses import dataclass
from pathlib import Path

from app.services.realtime_engine import EnterpriseRealtimeEngine
from app.services.ai_analyzer import NetflixLevelAIAnalyzer
from app.services.video_service import NetflixLevelVideoService
from app.utils.cache import EnterpriseIntelligentCacheManager
from app.utils.metrics import EnterpriseMetricsCollector
from app.utils.health import EnterpriseHealthChecker
from app.utils.rate_limiter import EnterpriseRateLimiter
from app.utils.security import EnterpriseSecurityManager

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ServiceMetadata:
    """Service metadata for enterprise monitoring"""
    name: str
    instance_id: str
    created_at: datetime
    initialized_at: Optional[datetime] = None
    status: str = "created"
    dependencies: Set[str] = None
    health_check_url: Optional[str] = None
    performance_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()
        if self.performance_metrics is None:
            self.performance_metrics = {}


class ServiceProvider(ABC, Generic[T]):
    """Abstract service provider for enterprise dependency injection"""

    @abstractmethod
    async def create_instance(self) -> T:
        """Create service instance"""
        pass

    @abstractmethod
    async def initialize_instance(self, instance: T) -> T:
        """Initialize service instance"""
        pass

    @abstractmethod
    async def health_check(self, instance: T) -> bool:
        """Perform health check on service instance"""
        pass

    @abstractmethod
    async def cleanup_instance(self, instance: T) -> None:
        """Cleanup service instance"""
        pass


class SingletonServiceProvider(ServiceProvider[T]):
    """Singleton service provider with lazy initialization"""

    def __init__(self, service_class: Type[T], *args, **kwargs):
        self.service_class = service_class
        self.args = args
        self.kwargs = kwargs
        self._instance: Optional[T] = None
        self._lock = asyncio.Lock()

    async def get_instance(self) -> T:
        """Get singleton instance with thread-safe lazy initialization"""
        if self._instance is None:
            async with self._lock:
                if self._instance is None:
                    self._instance = await self.create_instance()
                    await self.initialize_instance(self._instance)
        return self._instance

    async def create_instance(self) -> T:
        """Create service instance"""
        return self.service_class(*self.args, **self.kwargs)

    async def initialize_instance(self, instance: T) -> T:
        """Initialize service instance if it has enterprise_warm_up method"""
        if hasattr(instance, 'enterprise_warm_up'):
            await instance.enterprise_warm_up()
        return instance

    async def health_check(self, instance: T) -> bool:
        """Perform health check"""
        if hasattr(instance, 'health_check'):
            return await instance.health_check()
        return True

    async def cleanup_instance(self, instance: T) -> None:
        """Cleanup service instance"""
        if hasattr(instance, 'graceful_shutdown'):
            await instance.graceful_shutdown()


class DependencyContainer:
    """Netflix-level dependency injection container with enterprise features"""

    def __init__(self):
        self._services: Dict[str, ServiceProvider] = {}
        self._instances: Dict[str, Any] = {}
        self._metadata: Dict[str, ServiceMetadata] = {}
        self._initialization_order: List[str] = []
        self._service_dependencies: Dict[str, Set[str]] = {}
        self._circular_dependency_detector = set()
        
        # Performance monitoring
        self._initialization_times: Dict[str, float] = {}
        self._health_check_results: Dict[str, bool] = {}
        self._last_health_check: Optional[datetime] = None
        
        # Lifecycle management
        self._shutdown_callbacks: List[callable] = []
        self._is_shutting_down = False
        
        logger.info("🏗️ Netflix-level dependency container initialized")

    async def initialize_enterprise_services(self):
        """Initialize all enterprise services with dependency resolution"""
        try:
            start_time = time.time()
            logger.info("🚀 Initializing enterprise services...")

            # Register core services
            await self._register_core_services()
            
            # Initialize services in dependency order
            await self._initialize_services_ordered()
            
            # Perform initial health checks
            await self._perform_initial_health_checks()
            
            # Setup monitoring
            await self._setup_service_monitoring()
            
            initialization_time = time.time() - start_time
            logger.info(f"✅ Enterprise services initialized in {initialization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Enterprise service initialization failed: {e}", exc_info=True)
            raise

    async def _register_core_services(self):
        """Register all core enterprise services"""
        
        # Core infrastructure services
        self.register_singleton('cache_manager', EnterpriseIntelligentCacheManager)
        self.register_singleton('metrics_collector', EnterpriseMetricsCollector)
        self.register_singleton('health_checker', EnterpriseHealthChecker)
        self.register_singleton('rate_limiter', EnterpriseRateLimiter)
        self.register_singleton('security_manager', EnterpriseSecurityManager)
        
        # Business logic services
        self.register_singleton('realtime_engine', EnterpriseRealtimeEngine)
        self.register_singleton('ai_analyzer', NetflixLevelAIAnalyzer)
        self.register_singleton('video_service', NetflixLevelVideoService)
        
        # Define service dependencies
        self._service_dependencies = {
            'realtime_engine': {'metrics_collector', 'health_checker'},
            'ai_analyzer': {'cache_manager', 'metrics_collector'},
            'video_service': {'cache_manager', 'rate_limiter', 'metrics_collector'},
            'health_checker': {'metrics_collector'},
            'rate_limiter': {'cache_manager'},
            'security_manager': {'cache_manager'}
        }

    def register_singleton(self, name: str, service_class: Type[T], *args, **kwargs):
        """Register singleton service with metadata tracking"""
        import uuid
        
        provider = SingletonServiceProvider(service_class, *args, **kwargs)
        self._services[name] = provider
        
        # Create metadata
        self._metadata[name] = ServiceMetadata(
            name=name,
            instance_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            dependencies=self._service_dependencies.get(name, set()).copy()
        )
        
        logger.debug(f"📝 Registered singleton service: {name}")

    async def get_service(self, name: str) -> Any:
        """Get service instance with dependency resolution"""
        if name not in self._services:
            raise ValueError(f"Service '{name}' not registered")
        
        # Check for circular dependencies
        if name in self._circular_dependency_detector:
            raise ValueError(f"Circular dependency detected for service '{name}'")
        
        try:
            self._circular_dependency_detector.add(name)
            
            # Initialize dependencies first
            dependencies = self._service_dependencies.get(name, set())
            for dep_name in dependencies:
                await self.get_service(dep_name)
            
            # Get or create service instance
            if name not in self._instances:
                provider = self._services[name]
                
                start_time = time.time()
                instance = await provider.get_instance()
                initialization_time = time.time() - start_time
                
                self._instances[name] = instance
                self._initialization_times[name] = initialization_time
                
                # Update metadata
                if name in self._metadata:
                    self._metadata[name].initialized_at = datetime.utcnow()
                    self._metadata[name].status = "initialized"
                
                logger.info(f"🎯 Service initialized: {name} ({initialization_time:.3f}s)")
            
            return self._instances[name]
            
        finally:
            self._circular_dependency_detector.discard(name)

    async def _initialize_services_ordered(self):
        """Initialize services in dependency order"""
        
        # Calculate initialization order using topological sort
        initialization_order = self._calculate_initialization_order()
        
        for service_name in initialization_order:
            await self.get_service(service_name)
            self._initialization_order.append(service_name)

    def _calculate_initialization_order(self) -> List[str]:
        """Calculate service initialization order using topological sort"""
        from collections import defaultdict, deque
        
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Initialize in_degree for all services
        for service in self._services.keys():
            in_degree[service] = 0
        
        # Build graph and calculate in-degrees
        for service, deps in self._service_dependencies.items():
            for dep in deps:
                graph[dep].append(service)
                in_degree[service] += 1
        
        # Topological sort
        queue = deque([service for service in self._services.keys() if in_degree[service] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self._services):
            raise ValueError("Circular dependency detected in service graph")
        
        return result

    async def _perform_initial_health_checks(self):
        """Perform initial health checks on all services"""
        logger.info("🏥 Performing initial health checks...")
        
        for service_name in self._initialization_order:
            try:
                instance = self._instances[service_name]
                provider = self._services[service_name]
                
                is_healthy = await provider.health_check(instance)
                self._health_check_results[service_name] = is_healthy
                
                if is_healthy:
                    logger.debug(f"✅ Health check passed: {service_name}")
                else:
                    logger.warning(f"⚠️ Health check failed: {service_name}")
                    
            except Exception as e:
                logger.error(f"❌ Health check error for {service_name}: {e}")
                self._health_check_results[service_name] = False
        
        self._last_health_check = datetime.utcnow()

    async def _setup_service_monitoring(self):
        """Setup continuous service monitoring"""
        # Start background health monitoring
        asyncio.create_task(self._background_health_monitoring())
        
        # Start performance monitoring
        asyncio.create_task(self._background_performance_monitoring())

    async def _background_health_monitoring(self):
        """Background health monitoring for all services"""
        while not self._is_shutting_down:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for service_name in self._instances.keys():
                    try:
                        instance = self._instances[service_name]
                        provider = self._services[service_name]
                        
                        is_healthy = await provider.health_check(instance)
                        self._health_check_results[service_name] = is_healthy
                        
                        if not is_healthy:
                            logger.warning(f"⚠️ Service unhealthy: {service_name}")
                            
                    except Exception as e:
                        logger.error(f"Health check error for {service_name}: {e}")
                        self._health_check_results[service_name] = False
                
                self._last_health_check = datetime.utcnow()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background health monitoring error: {e}")

    async def _background_performance_monitoring(self):
        """Background performance monitoring"""
        while not self._is_shutting_down:
            try:
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
                # Log performance metrics
                total_services = len(self._instances)
                healthy_services = sum(1 for h in self._health_check_results.values() if h)
                
                logger.info(
                    f"📊 Service Health: {healthy_services}/{total_services} healthy, "
                    f"Avg init time: {sum(self._initialization_times.values()) / len(self._initialization_times):.3f}s"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background performance monitoring error: {e}")

    # Property accessors for easy service access
    @property
    def realtime_engine(self) -> Optional[EnterpriseRealtimeEngine]:
        """Get realtime engine service"""
        return self._instances.get('realtime_engine')

    @property
    def ai_analyzer(self) -> Optional[NetflixLevelAIAnalyzer]:
        """Get AI analyzer service"""
        return self._instances.get('ai_analyzer')

    @property
    def video_service(self) -> Optional[NetflixLevelVideoService]:
        """Get video service"""
        return self._instances.get('video_service')

    @property
    def cache_manager(self) -> Optional[EnterpriseIntelligentCacheManager]:
        """Get cache manager service"""
        return self._instances.get('cache_manager')

    @property
    def metrics_collector(self) -> Optional[EnterpriseMetricsCollector]:
        """Get metrics collector service"""
        return self._instances.get('metrics_collector')

    @property
    def health_checker(self) -> Optional[EnterpriseHealthChecker]:
        """Get health checker service"""
        return self._instances.get('health_checker')

    @property
    def rate_limiter(self) -> Optional[EnterpriseRateLimiter]:
        """Get rate limiter service"""
        return self._instances.get('rate_limiter')

    @property
    def security_manager(self) -> Optional[EnterpriseSecurityManager]:
        """Get security manager service"""
        return self._instances.get('security_manager')

    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            "total_services": len(self._services),
            "initialized_services": len(self._instances),
            "healthy_services": sum(1 for h in self._health_check_results.values() if h),
            "unhealthy_services": sum(1 for h in self._health_check_results.values() if not h),
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "initialization_order": self._initialization_order,
            "initialization_times": self._initialization_times,
            "health_results": self._health_check_results,
            "service_metadata": {
                name: {
                    "instance_id": meta.instance_id,
                    "status": meta.status,
                    "created_at": meta.created_at.isoformat(),
                    "initialized_at": meta.initialized_at.isoformat() if meta.initialized_at else None,
                    "dependencies": list(meta.dependencies)
                }
                for name, meta in self._metadata.items()
            }
        }

    async def add_shutdown_callback(self, callback: callable):
        """Add callback to be executed during shutdown"""
        self._shutdown_callbacks.append(callback)

    async def graceful_shutdown(self):
        """Gracefully shutdown all services"""
        logger.info("🔄 Starting graceful container shutdown...")
        self._is_shutting_down = True
        
        try:
            # Execute shutdown callbacks
            for callback in self._shutdown_callbacks:
                try:
                    await callback()
                except Exception as e:
                    logger.error(f"Shutdown callback error: {e}")
            
            # Shutdown services in reverse initialization order
            shutdown_order = list(reversed(self._initialization_order))
            
            for service_name in shutdown_order:
                try:
                    if service_name in self._instances:
                        instance = self._instances[service_name]
                        provider = self._services[service_name]
                        
                        await provider.cleanup_instance(instance)
                        logger.info(f"🔄 Service shut down: {service_name}")
                        
                except Exception as e:
                    logger.error(f"Error shutting down {service_name}: {e}")
            
            # Clear all state
            self._instances.clear()
            self._health_check_results.clear()
            self._initialization_times.clear()
            
            logger.info("✅ Container graceful shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during graceful shutdown: {e}", exc_info=True)
