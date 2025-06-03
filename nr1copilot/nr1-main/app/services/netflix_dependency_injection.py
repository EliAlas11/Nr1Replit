
"""
Netflix-Grade Dependency Injection Container v15.0
Advanced IoC container with lifecycle management and performance optimization
"""

import asyncio
import logging
import inspect
import threading
from datetime import datetime, timedelta
from typing import (
    Dict, List, Any, Optional, TypeVar, Generic, Callable, 
    Type, Union, get_type_hints, get_origin, get_args
)
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Scope(Enum):
    """Dependency scope enumeration"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
    PROTOTYPE = "prototype"


class LifecycleState(Enum):
    """Service lifecycle states"""
    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DISPOSED = "disposed"
    ERROR = "error"


@dataclass
class ServiceDescriptor:
    """Service descriptor with metadata"""
    service_type: Type[T]
    implementation_type: Type[T]
    scope: Scope
    factory: Optional[Callable] = None
    dependencies: List[Type] = field(default_factory=list)
    lifecycle_hooks: Dict[str, List[Callable]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0


@dataclass
class ServiceInstance:
    """Service instance with lifecycle management"""
    service: Any
    descriptor: ServiceDescriptor
    state: LifecycleState = LifecycleState.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    disposal_callbacks: List[Callable] = field(default_factory=list)


class ServiceScope:
    """Service scope context manager"""
    
    def __init__(self, container: 'NetflixDependencyContainer'):
        self.container = container
        self.scoped_instances: Dict[Type, Any] = {}
        self._disposed = False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.dispose()
    
    async def get_service(self, service_type: Type[T]) -> T:
        """Get service within scope"""
        if service_type in self.scoped_instances:
            return self.scoped_instances[service_type]
        
        instance = await self.container.get_service(service_type)
        self.scoped_instances[service_type] = instance
        return instance
    
    async def dispose(self) -> None:
        """Dispose scoped services"""
        if self._disposed:
            return
        
        for service in self.scoped_instances.values():
            if hasattr(service, 'dispose'):
                try:
                    await service.dispose()
                except Exception as e:
                    logger.error(f"Error disposing scoped service: {e}")
        
        self.scoped_instances.clear()
        self._disposed = True


def injectable(scope: Scope = Scope.TRANSIENT, priority: int = 0):
    """Decorator to mark classes as injectable"""
    def decorator(cls: Type[T]) -> Type[T]:
        cls._injectable_scope = scope
        cls._injectable_priority = priority
        return cls
    return decorator


def dependency(name: Optional[str] = None):
    """Decorator to mark method parameters as dependencies"""
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_dependencies'):
            func._dependencies = {}
        
        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                func._dependencies[param_name] = param.annotation
        
        return func
    return decorator


class NetflixDependencyContainer:
    """Netflix-grade dependency injection container with advanced features"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, ServiceInstance] = {}
        self._lock = threading.RLock()
        self._scopes: List[ServiceScope] = []
        
        # Performance tracking
        self._resolution_cache: Dict[Type, Any] = {}
        self._resolution_stats: Dict[Type, Dict[str, Any]] = {}
        
        # Lifecycle management
        self._startup_hooks: List[Callable] = []
        self._shutdown_hooks: List[Callable] = []
        
        logger.info("🏗️ Netflix Dependency Container v15.0 initialized")
    
    def register_singleton(self, service_type: Type[T], 
                          implementation: Optional[Union[Type[T], T]] = None,
                          factory: Optional[Callable[[], T]] = None) -> 'NetflixDependencyContainer':
        """Register singleton service"""
        return self._register_service(service_type, implementation, factory, Scope.SINGLETON)
    
    def register_transient(self, service_type: Type[T], 
                          implementation: Optional[Union[Type[T], T]] = None,
                          factory: Optional[Callable[[], T]] = None) -> 'NetflixDependencyContainer':
        """Register transient service"""
        return self._register_service(service_type, implementation, factory, Scope.TRANSIENT)
    
    def register_scoped(self, service_type: Type[T], 
                       implementation: Optional[Union[Type[T], T]] = None,
                       factory: Optional[Callable[[], T]] = None) -> 'NetflixDependencyContainer':
        """Register scoped service"""
        return self._register_service(service_type, implementation, factory, Scope.SCOPED)
    
    def _register_service(self, service_type: Type[T], 
                         implementation: Optional[Union[Type[T], T]], 
                         factory: Optional[Callable[[], T]], 
                         scope: Scope) -> 'NetflixDependencyContainer':
        """Internal service registration"""
        with self._lock:
            impl_type = implementation if inspect.isclass(implementation) else type(implementation)
            
            # Extract dependencies from constructor
            dependencies = []
            if impl_type and hasattr(impl_type, '__init__'):
                sig = inspect.signature(impl_type.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                        dependencies.append(param.annotation)
            
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=impl_type,
                scope=scope,
                factory=factory,
                dependencies=dependencies,
                priority=getattr(impl_type, '_injectable_priority', 0) if impl_type else 0
            )
            
            self._services[service_type] = descriptor
            
            # If singleton and instance provided, store it
            if scope == Scope.SINGLETON and implementation and not inspect.isclass(implementation):
                instance = ServiceInstance(
                    service=implementation,
                    descriptor=descriptor,
                    state=LifecycleState.INITIALIZED
                )
                self._instances[service_type] = instance
            
            logger.debug(f"📝 Registered {scope.value} service: {service_type.__name__}")
            return self
    
    async def get_service(self, service_type: Type[T]) -> T:
        """Get service instance with dependency resolution"""
        with self._lock:
            # Check cache first
            if service_type in self._resolution_cache:
                cached_instance = self._resolution_cache[service_type]
                if cached_instance is not None:
                    self._update_access_stats(service_type)
                    return cached_instance
            
            # Resolve service
            instance = await self._resolve_service(service_type)
            
            # Cache if appropriate
            descriptor = self._services.get(service_type)
            if descriptor and descriptor.scope == Scope.SINGLETON:
                self._resolution_cache[service_type] = instance
            
            self._update_access_stats(service_type)
            return instance
    
    async def _resolve_service(self, service_type: Type[T]) -> T:
        """Resolve service with dependency injection"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not registered")
        
        descriptor = self._services[service_type]
        
        # Check existing singleton instance
        if descriptor.scope == Scope.SINGLETON and service_type in self._instances:
            return self._instances[service_type].service
        
        # Create new instance
        instance = await self._create_instance(descriptor)
        
        # Store singleton instance
        if descriptor.scope == Scope.SINGLETON:
            service_instance = ServiceInstance(
                service=instance,
                descriptor=descriptor,
                state=LifecycleState.INITIALIZED
            )
            self._instances[service_type] = service_instance
        
        return instance
    
    async def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create service instance with dependency injection"""
        try:
            if descriptor.factory:
                # Use factory function
                if asyncio.iscoroutinefunction(descriptor.factory):
                    instance = await descriptor.factory()
                else:
                    instance = descriptor.factory()
            else:
                # Create instance using constructor
                dependencies = await self._resolve_dependencies(descriptor.dependencies)
                instance = descriptor.implementation_type(*dependencies)
            
            # Initialize if async
            if hasattr(instance, 'initialize') and asyncio.iscoroutinefunction(instance.initialize):
                await instance.initialize()
            
            # Call lifecycle hooks
            await self._call_lifecycle_hooks(descriptor, 'on_created', instance)
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create instance of {descriptor.service_type.__name__}: {e}")
            raise
    
    async def _resolve_dependencies(self, dependencies: List[Type]) -> List[Any]:
        """Resolve service dependencies"""
        resolved = []
        for dep_type in dependencies:
            dependency = await self.get_service(dep_type)
            resolved.append(dependency)
        return resolved
    
    async def _call_lifecycle_hooks(self, descriptor: ServiceDescriptor, 
                                   hook_name: str, instance: Any) -> None:
        """Call lifecycle hooks"""
        hooks = descriptor.lifecycle_hooks.get(hook_name, [])
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(instance)
                else:
                    hook(instance)
            except Exception as e:
                logger.error(f"Lifecycle hook {hook_name} failed: {e}")
    
    def _update_access_stats(self, service_type: Type) -> None:
        """Update service access statistics"""
        if service_type not in self._resolution_stats:
            self._resolution_stats[service_type] = {
                'access_count': 0,
                'last_accessed': datetime.utcnow(),
                'first_accessed': datetime.utcnow()
            }
        
        stats = self._resolution_stats[service_type]
        stats['access_count'] += 1
        stats['last_accessed'] = datetime.utcnow()
    
    def create_scope(self) -> ServiceScope:
        """Create new service scope"""
        scope = ServiceScope(self)
        self._scopes.append(scope)
        return scope
    
    async def initialize_services(self) -> None:
        """Initialize all registered services"""
        logger.info("🚀 Initializing services...")
        
        # Sort by priority
        services_by_priority = sorted(
            self._services.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        for service_type, descriptor in services_by_priority:
            try:
                if descriptor.scope == Scope.SINGLETON:
                    await self.get_service(service_type)
                    logger.debug(f"✅ Initialized singleton: {service_type.__name__}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {service_type.__name__}: {e}")
        
        # Call startup hooks
        for hook in self._startup_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                logger.error(f"Startup hook failed: {e}")
        
        logger.info("✅ Service initialization complete")
    
    async def shutdown_services(self) -> None:
        """Shutdown all services gracefully"""
        logger.info("🔄 Shutting down services...")
        
        # Call shutdown hooks
        for hook in self._shutdown_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                logger.error(f"Shutdown hook failed: {e}")
        
        # Dispose scopes
        for scope in self._scopes:
            await scope.dispose()
        
        # Dispose singleton instances
        for service_type, instance in self._instances.items():
            try:
                if hasattr(instance.service, 'dispose'):
                    if asyncio.iscoroutinefunction(instance.service.dispose):
                        await instance.service.dispose()
                    else:
                        instance.service.dispose()
                
                instance.state = LifecycleState.DISPOSED
                logger.debug(f"🗑️ Disposed: {service_type.__name__}")
                
            except Exception as e:
                logger.error(f"Error disposing {service_type.__name__}: {e}")
        
        # Clear caches
        self._resolution_cache.clear()
        self._instances.clear()
        
        logger.info("✅ Service shutdown complete")
    
    def add_startup_hook(self, hook: Callable) -> None:
        """Add startup hook"""
        self._startup_hooks.append(hook)
    
    def add_shutdown_hook(self, hook: Callable) -> None:
        """Add shutdown hook"""
        self._shutdown_hooks.append(hook)
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information"""
        with self._lock:
            services_info = {}
            
            for service_type, descriptor in self._services.items():
                instance_info = None
                if service_type in self._instances:
                    instance = self._instances[service_type]
                    instance_info = {
                        'state': instance.state.value,
                        'created_at': instance.created_at.isoformat(),
                        'last_accessed': instance.last_accessed.isoformat(),
                        'access_count': instance.access_count
                    }
                
                stats = self._resolution_stats.get(service_type, {})
                
                services_info[service_type.__name__] = {
                    'scope': descriptor.scope.value,
                    'implementation': descriptor.implementation_type.__name__,
                    'dependencies': [dep.__name__ for dep in descriptor.dependencies],
                    'priority': descriptor.priority,
                    'registered_at': descriptor.created_at.isoformat(),
                    'instance': instance_info,
                    'stats': stats
                }
            
            return {
                'services': services_info,
                'total_services': len(self._services),
                'active_instances': len(self._instances),
                'active_scopes': len(self._scopes),
                'cache_size': len(self._resolution_cache)
            }
    
    def validate_configuration(self) -> List[str]:
        """Validate container configuration"""
        errors = []
        
        # Check for circular dependencies
        for service_type, descriptor in self._services.items():
            if self._has_circular_dependency(service_type, set()):
                errors.append(f"Circular dependency detected for {service_type.__name__}")
        
        # Check for missing dependencies
        for service_type, descriptor in self._services.items():
            for dep_type in descriptor.dependencies:
                if dep_type not in self._services:
                    errors.append(f"Missing dependency {dep_type.__name__} for {service_type.__name__}")
        
        return errors
    
    def _has_circular_dependency(self, service_type: Type, visited: set) -> bool:
        """Check for circular dependencies"""
        if service_type in visited:
            return True
        
        if service_type not in self._services:
            return False
        
        visited.add(service_type)
        descriptor = self._services[service_type]
        
        for dep_type in descriptor.dependencies:
            if self._has_circular_dependency(dep_type, visited.copy()):
                return True
        
        return False


# Global container instance
netflix_container = NetflixDependencyContainer()
