
"""
Netflix-Level Dependency Injection Container
Advanced IoC container with lifecycle management and health monitoring
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Type, TypeVar, Generic, Callable
from contextlib import asynccontextmanager
from datetime import datetime
import weakref
from pathlib import Path

from app.config import settings
from app.utils.cache import CacheManager
from app.utils.metrics import MetricsCollector
from app.utils.health import HealthChecker
from app.utils.security import SecurityManager
from app.utils.rate_limiter import RateLimiter

# Service imports
from app.services.video_service import VideoService
from app.services.ai_analyzer import AIAnalyzer
from app.services.realtime_engine import RealtimeEngine

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifecycle:
    """Service lifecycle management"""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class ServiceHealth:
    """Service health tracking"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.status = ServiceLifecycle.CREATED
        self.last_health_check = None
        self.error_count = 0
        self.startup_time = None
        self.shutdown_time = None
        self.dependencies = []
        self.health_checks = []

    def mark_healthy(self):
        self.status = ServiceLifecycle.READY
        self.last_health_check = datetime.utcnow()
        self.error_count = 0

    def mark_degraded(self, error: Exception = None):
        self.status = ServiceLifecycle.DEGRADED
        self.error_count += 1
        if error:
            logger.warning(f"Service {self.service_name} degraded: {error}")

    def mark_failed(self, error: Exception = None):
        self.status = ServiceLifecycle.FAILED
        self.error_count += 1
        if error:
            logger.error(f"Service {self.service_name} failed: {error}")


class ServiceRegistry:
    """Netflix-level service registry with dependency resolution"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._health_trackers: Dict[str, ServiceHealth] = {}
        self._dependency_graph: Dict[str, list] = {}
        self._initialization_order: list = []
        
    def register_factory(self, name: str, factory: Callable, dependencies: list = None):
        """Register a service factory with dependencies"""
        self._factories[name] = factory
        self._dependency_graph[name] = dependencies or []
        self._health_trackers[name] = ServiceHealth(name)
        logger.debug(f"Registered factory for {name} with dependencies: {dependencies}")
    
    def register_singleton(self, name: str, instance: Any):
        """Register a singleton instance"""
        self._singletons[name] = instance
        self._health_trackers[name] = ServiceHealth(name)
        self._health_trackers[name].mark_healthy()
        logger.debug(f"Registered singleton for {name}")
    
    async def resolve(self, name: str) -> Any:
        """Resolve a service with dependency injection"""
        if name in self._singletons:
            return self._singletons[name]
            
        if name in self._services:
            return self._services[name]
            
        if name not in self._factories:
            raise ValueError(f"Service {name} not registered")
        
        health_tracker = self._health_trackers[name]
        
        try:
            health_tracker.status = ServiceLifecycle.INITIALIZING
            
            # Resolve dependencies first
            dependencies = {}
            for dep_name in self._dependency_graph[name]:
                dependencies[dep_name] = await self.resolve(dep_name)
            
            # Create service
            factory = self._factories[name]
            if asyncio.iscoroutinefunction(factory):
                service = await factory(**dependencies)
            else:
                service = factory(**dependencies)
            
            # Initialize if needed
            if hasattr(service, 'initialize') and callable(service.initialize):
                if asyncio.iscoroutinefunction(service.initialize):
                    await service.initialize()
                else:
                    service.initialize()
            
            self._services[name] = service
            health_tracker.mark_healthy()
            health_tracker.startup_time = datetime.utcnow()
            
            logger.info(f"Successfully resolved service: {name}")
            return service
            
        except Exception as e:
            health_tracker.mark_failed(e)
            logger.error(f"Failed to resolve service {name}: {e}")
            raise
    
    async def health_check(self, name: str) -> bool:
        """Perform health check on a service"""
        if name not in self._health_trackers:
            return False
            
        health_tracker = self._health_trackers[name]
        service = self._services.get(name) or self._singletons.get(name)
        
        if not service:
            return False
        
        try:
            # Check if service has custom health check
            if hasattr(service, 'health_check') and callable(service.health_check):
                if asyncio.iscoroutinefunction(service.health_check):
                    is_healthy = await service.health_check()
                else:
                    is_healthy = service.health_check()
                    
                if is_healthy:
                    health_tracker.mark_healthy()
                else:
                    health_tracker.mark_degraded()
                    
                return is_healthy
            else:
                # Basic health check - service exists and is not in failed state
                is_healthy = health_tracker.status not in [ServiceLifecycle.FAILED, ServiceLifecycle.STOPPED]
                if is_healthy:
                    health_tracker.mark_healthy()
                return is_healthy
                
        except Exception as e:
            health_tracker.mark_failed(e)
            return False
    
    async def shutdown_service(self, name: str):
        """Gracefully shutdown a service"""
        health_tracker = self._health_trackers.get(name)
        if health_tracker:
            health_tracker.status = ServiceLifecycle.SHUTTING_DOWN
        
        service = self._services.get(name) or self._singletons.get(name)
        if service and hasattr(service, 'cleanup') and callable(service.cleanup):
            try:
                if asyncio.iscoroutinefunction(service.cleanup):
                    await service.cleanup()
                else:
                    service.cleanup()
                logger.info(f"Successfully shutdown service: {name}")
            except Exception as e:
                logger.error(f"Error shutting down service {name}: {e}")
        
        if health_tracker:
            health_tracker.status = ServiceLifecycle.STOPPED
            health_tracker.shutdown_time = datetime.utcnow()
    
    def get_service_health(self, name: str) -> Optional[ServiceHealth]:
        """Get health information for a service"""
        return self._health_trackers.get(name)
    
    def get_all_health_info(self) -> Dict[str, Dict[str, Any]]:
        """Get health information for all services"""
        result = {}
        for name, health in self._health_trackers.items():
            result[name] = {
                "status": health.status,
                "error_count": health.error_count,
                "last_health_check": health.last_health_check,
                "startup_time": health.startup_time,
                "dependencies": health.dependencies
            }
        return result


class DependencyContainer:
    """Netflix-level dependency injection container"""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        
        # Core services
        self.cache_manager: Optional[CacheManager] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.health_checker: Optional[HealthChecker] = None
        self.security_manager: Optional[SecurityManager] = None
        self.rate_limiter: Optional[RateLimiter] = None
        
        # Business services
        self.video_service: Optional[VideoService] = None
        self.ai_analyzer: Optional[AIAnalyzer] = None
        self.realtime_engine: Optional[RealtimeEngine] = None
        
    async def initialize(self):
        """Initialize all services with proper dependency order"""
        async with self._initialization_lock:
            if self._initialized:
                return
                
            logger.info("🚀 Initializing Netflix-level dependency container")
            start_time = datetime.utcnow()
            
            try:
                await self._register_services()
                await self._resolve_core_services()
                await self._resolve_business_services()
                await self._perform_health_checks()
                
                self._initialized = True
                initialization_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.info(f"✅ Dependency container initialized in {initialization_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize dependency container: {e}")
                await self.cleanup()
                raise
    
    async def _register_services(self):
        """Register all service factories"""
        
        # Register core infrastructure services
        self.registry.register_factory(
            "cache_manager",
            self._create_cache_manager,
            dependencies=[]
        )
        
        self.registry.register_factory(
            "metrics_collector", 
            self._create_metrics_collector,
            dependencies=["cache_manager"]
        )
        
        self.registry.register_factory(
            "health_checker",
            self._create_health_checker,
            dependencies=["metrics_collector"]
        )
        
        self.registry.register_factory(
            "security_manager",
            self._create_security_manager,
            dependencies=["cache_manager"]
        )
        
        self.registry.register_factory(
            "rate_limiter",
            self._create_rate_limiter,
            dependencies=["cache_manager"]
        )
        
        # Register business services
        self.registry.register_factory(
            "video_service",
            self._create_video_service,
            dependencies=["cache_manager", "metrics_collector"]
        )
        
        self.registry.register_factory(
            "ai_analyzer",
            self._create_ai_analyzer,
            dependencies=["cache_manager", "metrics_collector"]
        )
        
        self.registry.register_factory(
            "realtime_engine",
            self._create_realtime_engine,
            dependencies=["video_service", "ai_analyzer", "metrics_collector"]
        )
    
    async def _resolve_core_services(self):
        """Resolve core infrastructure services"""
        self.cache_manager = await self.registry.resolve("cache_manager")
        self.metrics_collector = await self.registry.resolve("metrics_collector")
        self.health_checker = await self.registry.resolve("health_checker")
        self.security_manager = await self.registry.resolve("security_manager")
        self.rate_limiter = await self.registry.resolve("rate_limiter")
    
    async def _resolve_business_services(self):
        """Resolve business logic services"""
        self.video_service = await self.registry.resolve("video_service")
        self.ai_analyzer = await self.registry.resolve("ai_analyzer")
        self.realtime_engine = await self.registry.resolve("realtime_engine")
    
    async def _perform_health_checks(self):
        """Perform initial health checks on all services"""
        services = [
            "cache_manager", "metrics_collector", "health_checker",
            "security_manager", "rate_limiter", "video_service", 
            "ai_analyzer", "realtime_engine"
        ]
        
        for service_name in services:
            is_healthy = await self.registry.health_check(service_name)
            if not is_healthy:
                logger.warning(f"Service {service_name} failed initial health check")
    
    # Service factory methods
    async def _create_cache_manager(self) -> CacheManager:
        """Create cache manager"""
        return CacheManager(
            default_ttl=settings.cache_ttl,
            max_size=1000,
            cleanup_interval=300
        )
    
    async def _create_metrics_collector(self, cache_manager: CacheManager) -> MetricsCollector:
        """Create metrics collector"""
        return MetricsCollector(
            cache_manager=cache_manager,
            collection_interval=30
        )
    
    async def _create_health_checker(self, metrics_collector: MetricsCollector) -> HealthChecker:
        """Create health checker"""
        return HealthChecker(
            metrics_collector=metrics_collector,
            check_interval=60
        )
    
    async def _create_security_manager(self, cache_manager: CacheManager) -> SecurityManager:
        """Create security manager"""
        return SecurityManager(
            cache_manager=cache_manager,
            secret_key=settings.secret_key
        )
    
    async def _create_rate_limiter(self, cache_manager: CacheManager) -> RateLimiter:
        """Create rate limiter"""
        return RateLimiter(
            cache_manager=cache_manager,
            default_limit=settings.rate_limit_requests
        )
    
    async def _create_video_service(self, cache_manager: CacheManager, metrics_collector: MetricsCollector) -> VideoService:
        """Create video service"""
        return VideoService(
            upload_dir=settings.upload_path,
            output_dir=settings.output_path,
            temp_dir=settings.temp_path,
            cache_manager=cache_manager,
            metrics_collector=metrics_collector
        )
    
    async def _create_ai_analyzer(self, cache_manager: CacheManager, metrics_collector: MetricsCollector) -> AIAnalyzer:
        """Create AI analyzer"""
        return AIAnalyzer(
            cache_manager=cache_manager,
            metrics_collector=metrics_collector,
            model_path=settings.ai_model_path,
            batch_size=settings.ai_batch_size
        )
    
    async def _create_realtime_engine(self, video_service: VideoService, ai_analyzer: AIAnalyzer, metrics_collector: MetricsCollector) -> RealtimeEngine:
        """Create realtime engine"""
        return RealtimeEngine(
            video_service=video_service,
            ai_analyzer=ai_analyzer,
            metrics_collector=metrics_collector
        )
    
    async def get_service(self, service_name: str) -> Any:
        """Get a service by name"""
        if not self._initialized:
            await self.initialize()
        return await self.registry.resolve(service_name)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all services"""
        if not self._initialized:
            return {}
            
        results = {}
        services = [
            "cache_manager", "metrics_collector", "health_checker",
            "security_manager", "rate_limiter", "video_service",
            "ai_analyzer", "realtime_engine"
        ]
        
        for service_name in services:
            try:
                results[service_name] = await self.registry.health_check(service_name)
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                results[service_name] = False
                
        return results
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health information"""
        health_checks = await self.health_check_all()
        service_info = self.registry.get_all_health_info()
        
        overall_health = "healthy"
        if not all(health_checks.values()):
            unhealthy_count = sum(1 for healthy in health_checks.values() if not healthy)
            if unhealthy_count > len(health_checks) // 2:
                overall_health = "unhealthy"
            else:
                overall_health = "degraded"
        
        return {
            "overall_status": overall_health,
            "services": health_checks,
            "service_details": service_info,
            "container_initialized": self._initialized,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def preload_critical_data(self):
        """Preload critical data for better performance"""
        try:
            # Warm up cache with commonly accessed data
            if self.cache_manager:
                await self.cache_manager.warm_up()
            
            # Initialize AI models
            if self.ai_analyzer:
                await self.ai_analyzer.warm_up()
            
            logger.info("✅ Critical data preloaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to preload some critical data: {e}")
    
    async def cleanup(self):
        """Cleanup all services gracefully"""
        if not self._initialized:
            return
            
        logger.info("🔄 Cleaning up dependency container")
        
        # Shutdown services in reverse dependency order
        shutdown_order = [
            "realtime_engine", "ai_analyzer", "video_service",
            "rate_limiter", "security_manager", "health_checker",
            "metrics_collector", "cache_manager"
        ]
        
        for service_name in shutdown_order:
            try:
                await self.registry.shutdown_service(service_name)
            except Exception as e:
                logger.error(f"Error shutting down {service_name}: {e}")
        
        self._initialized = False
        logger.info("✅ Dependency container cleanup complete")
    
    @asynccontextmanager
    async def service_scope(self, service_name: str):
        """Context manager for scoped service usage"""
        service = await self.get_service(service_name)
        try:
            yield service
        finally:
            # Cleanup if needed
            if hasattr(service, 'cleanup_scope') and callable(service.cleanup_scope):
                try:
                    if asyncio.iscoroutinefunction(service.cleanup_scope):
                        await service.cleanup_scope()
                    else:
                        service.cleanup_scope()
                except Exception as e:
                    logger.warning(f"Error in service scope cleanup for {service_name}: {e}")


# Global container instance
container = DependencyContainer()

__all__ = [
    "DependencyContainer",
    "ServiceRegistry", 
    "ServiceHealth",
    "ServiceLifecycle",
    "container"
]
