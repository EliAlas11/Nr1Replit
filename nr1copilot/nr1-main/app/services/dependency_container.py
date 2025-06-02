
"""
Netflix-Grade Dependency Container v14.0
Clean dependency injection with proper lifecycle management and service discovery
"""

import asyncio
import logging
import weakref
from typing import Dict, Any, Optional, Type, TypeVar, Generic, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifecycle(Enum):
    """Service lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ServiceDescriptor:
    """Service registration descriptor"""
    service_type: Type
    instance: Optional[Any] = None
    lifecycle: ServiceLifecycle = ServiceLifecycle.UNINITIALIZED
    dependencies: List[str] = field(default_factory=list)
    initialization_time: float = 0.0
    error_message: Optional[str] = None
    is_singleton: bool = True
    auto_start: bool = True


class Service(ABC):
    """Base service interface"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service"""
        pass
    
    async def startup(self) -> None:
        """Start the service (optional)"""
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the service (optional)"""
        pass
    
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return True


class ServiceContainer:
    """Netflix-grade dependency injection container with lifecycle management"""
    
    def __init__(self):
        self._services: Dict[str, ServiceDescriptor] = {}
        self._instances: Dict[str, Any] = {}
        self._weak_references: Dict[str, weakref.ref] = {}
        self._initialization_lock = asyncio.Lock()
        self._is_initialized = False
        self._startup_time = 0.0
        
        # Register core services
        self._register_core_services()
        
        logger.info("🏗️ Netflix-Grade Service Container v14.0 initialized")
    
    def _register_core_services(self) -> None:
        """Register core system services"""
        try:
            # Health monitor
            self.register_service("health_monitor", self._create_health_monitor)
            
            # Metrics collector
            self.register_service("metrics_collector", self._create_metrics_collector)
            
            # Cache manager
            self.register_service("cache_manager", self._create_cache_manager)
            
            logger.debug("✅ Core services registered")
            
        except Exception as e:
            logger.error(f"❌ Core service registration failed: {e}")
    
    def register_service(
        self, 
        name: str, 
        factory_or_type: Any, 
        dependencies: Optional[List[str]] = None,
        singleton: bool = True,
        auto_start: bool = True
    ) -> None:
        """Register a service with the container"""
        try:
            if callable(factory_or_type):
                service_type = factory_or_type
            else:
                service_type = type(factory_or_type)
            
            descriptor = ServiceDescriptor(
                service_type=service_type,
                dependencies=dependencies or [],
                is_singleton=singleton,
                auto_start=auto_start
            )
            
            self._services[name] = descriptor
            logger.debug(f"📝 Service registered: {name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register service {name}: {e}")
    
    async def get_service(self, name: str) -> Optional[Any]:
        """Get a service instance with dependency resolution"""
        try:
            if name not in self._services:
                logger.warning(f"⚠️ Service not found: {name}")
                return None
            
            descriptor = self._services[name]
            
            # Return existing singleton instance
            if descriptor.is_singleton and descriptor.instance:
                return descriptor.instance
            
            # Create new instance
            instance = await self._create_service_instance(name, descriptor)
            
            if descriptor.is_singleton:
                descriptor.instance = instance
                self._instances[name] = instance
                
                # Create weak reference for cleanup
                self._weak_references[name] = weakref.ref(
                    instance, 
                    lambda ref: self._cleanup_service(name)
                )
            
            return instance
            
        except Exception as e:
            logger.error(f"❌ Failed to get service {name}: {e}")
            return None
    
    async def _create_service_instance(self, name: str, descriptor: ServiceDescriptor) -> Any:
        """Create a service instance with dependency injection"""
        start_time = time.time()
        
        try:
            descriptor.lifecycle = ServiceLifecycle.INITIALIZING
            
            # Resolve dependencies
            dependencies = {}
            for dep_name in descriptor.dependencies:
                dep_instance = await self.get_service(dep_name)
                if dep_instance:
                    dependencies[dep_name] = dep_instance
            
            # Create instance
            if callable(descriptor.service_type):
                if dependencies:
                    instance = descriptor.service_type(**dependencies)
                else:
                    instance = descriptor.service_type()
            else:
                instance = descriptor.service_type
            
            # Initialize if it's a Service
            if isinstance(instance, Service):
                await instance.initialize()
            elif hasattr(instance, 'initialize'):
                await instance.initialize()
            
            descriptor.lifecycle = ServiceLifecycle.RUNNING
            descriptor.initialization_time = time.time() - start_time
            
            logger.debug(f"✅ Service created: {name} ({descriptor.initialization_time:.3f}s)")
            return instance
            
        except Exception as e:
            descriptor.lifecycle = ServiceLifecycle.ERROR
            descriptor.error_message = str(e)
            logger.error(f"❌ Failed to create service {name}: {e}")
            raise
    
    def _cleanup_service(self, name: str) -> None:
        """Cleanup service references"""
        try:
            if name in self._instances:
                del self._instances[name]
            if name in self._weak_references:
                del self._weak_references[name]
            
            logger.debug(f"🧹 Service cleaned up: {name}")
            
        except Exception as e:
            logger.error(f"❌ Service cleanup failed for {name}: {e}")
    
    async def initialize(self) -> None:
        """Initialize all auto-start services"""
        if self._is_initialized:
            return
        
        async with self._initialization_lock:
            if self._is_initialized:
                return
            
            start_time = time.time()
            initialized_count = 0
            
            try:
                logger.info("🚀 Initializing Netflix-grade services...")
                
                # Initialize services in dependency order
                for name, descriptor in self._services.items():
                    if descriptor.auto_start:
                        try:
                            await self.get_service(name)
                            initialized_count += 1
                        except Exception as e:
                            logger.error(f"❌ Service initialization failed: {name} - {e}")
                
                self._startup_time = time.time() - start_time
                self._is_initialized = True
                
                logger.info(f"✅ {initialized_count} services initialized in {self._startup_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Service container initialization failed: {e}")
                raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all services"""
        if not self._is_initialized:
            return
        
        logger.info("🔄 Shutting down services...")
        shutdown_count = 0
        
        try:
            # Shutdown in reverse order
            service_names = list(self._instances.keys())
            service_names.reverse()
            
            for name in service_names:
                try:
                    instance = self._instances.get(name)
                    if instance:
                        if isinstance(instance, Service):
                            await instance.shutdown()
                        elif hasattr(instance, 'shutdown'):
                            await instance.shutdown()
                        
                        shutdown_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Service shutdown failed: {name} - {e}")
            
            # Clear all references
            self._instances.clear()
            self._weak_references.clear()
            self._is_initialized = False
            
            logger.info(f"✅ {shutdown_count} services shut down gracefully")
            
        except Exception as e:
            logger.error(f"❌ Service container shutdown failed: {e}")
    
    def is_healthy(self) -> bool:
        """Check if all services are healthy"""
        try:
            for name, instance in self._instances.items():
                if isinstance(instance, Service):
                    if not instance.is_healthy():
                        return False
                elif hasattr(instance, 'is_healthy'):
                    if not instance.is_healthy():
                        return False
            
            return True
            
        except Exception:
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all registered services"""
        try:
            status = {
                "total_services": len(self._services),
                "running_services": len(self._instances),
                "initialization_time": self._startup_time,
                "is_healthy": self.is_healthy(),
                "services": {}
            }
            
            for name, descriptor in self._services.items():
                status["services"][name] = {
                    "lifecycle": descriptor.lifecycle.value,
                    "initialization_time": descriptor.initialization_time,
                    "is_singleton": descriptor.is_singleton,
                    "auto_start": descriptor.auto_start,
                    "has_instance": name in self._instances,
                    "error_message": descriptor.error_message
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get service status: {e}")
            return {"error": str(e)}
    
    # Service factory methods
    def _create_health_monitor(self):
        """Create health monitor service"""
        try:
            from app.netflix_health_monitor import NetflixHealthMonitor
            return NetflixHealthMonitor()
        except ImportError as e:
            logger.warning(f"⚠️ Health monitor not available: {e}")
            return None
    
    def _create_metrics_collector(self):
        """Create metrics collector service"""
        try:
            from app.utils.metrics import MetricsCollector
            return MetricsCollector()
        except ImportError as e:
            logger.warning(f"⚠️ Metrics collector not available: {e}")
            return None
    
    def _create_cache_manager(self):
        """Create cache manager service"""
        try:
            from app.utils.cache import CacheManager
            return CacheManager()
        except ImportError as e:
            logger.warning(f"⚠️ Cache manager not available: {e}")
            return None
    
    # Convenience methods
    def get_health_monitor(self):
        """Get health monitor instance"""
        try:
            return asyncio.run(self.get_service("health_monitor"))
        except Exception:
            return None
    
    def get_metrics_collector(self):
        """Get metrics collector instance"""
        try:
            return asyncio.run(self.get_service("metrics_collector"))
        except Exception:
            return None
    
    def get_cache_manager(self):
        """Get cache manager instance"""
        try:
            return asyncio.run(self.get_service("cache_manager"))
        except Exception:
            return None


# Global service container instance
service_container = ServiceContainer()
