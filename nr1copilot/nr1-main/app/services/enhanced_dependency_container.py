
"""
Enhanced Dependency Container v10.0
Netflix-grade dependency injection with comprehensive service management
"""

import asyncio
import logging
import time
import weakref
from typing import Dict, Any, Type, TypeVar, Optional, Callable, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifecycle(Enum):
    """Service lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ServiceMetadata:
    """Service metadata and statistics"""
    name: str
    service_type: Type
    lifecycle: ServiceLifecycle
    created_at: datetime
    initialization_time: float
    dependencies: Set[str]
    dependents: Set[str]
    health_status: str = "unknown"
    last_health_check: Optional[datetime] = None


class EnhancedDependencyContainer:
    """Netflix-grade dependency injection container with advanced service management"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._service_metadata: Dict[str, ServiceMetadata] = {}
        self._service_factories: Dict[str, Callable] = {}
        self._singletons: Set[str] = set()
        self._weak_refs: Dict[str, weakref.ref] = {}
        
        # Service management
        self._initialization_order: list = []
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._reverse_dependencies: Dict[str, Set[str]] = {}
        
        # Health monitoring
        self._health_checks: Dict[str, Callable] = {}
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("🏗️ Enhanced Dependency Container v10.0 initialized")
    
    def register_singleton(self, name: str, service_type: Type[T], factory: Optional[Callable] = None) -> None:
        """Register a singleton service"""
        self._singletons.add(name)
        self._service_factories[name] = factory or service_type
        self._dependency_graph[name] = set()
        self._reverse_dependencies[name] = set()
        
        # Create metadata
        self._service_metadata[name] = ServiceMetadata(
            name=name,
            service_type=service_type,
            lifecycle=ServiceLifecycle.UNINITIALIZED,
            created_at=datetime.utcnow(),
            initialization_time=0.0,
            dependencies=set(),
            dependents=set()
        )
        
        logger.debug(f"📝 Registered singleton service: {name}")
    
    def register_transient(self, name: str, service_type: Type[T], factory: Optional[Callable] = None) -> None:
        """Register a transient service"""
        self._service_factories[name] = factory or service_type
        self._dependency_graph[name] = set()
        self._reverse_dependencies[name] = set()
        
        # Create metadata
        self._service_metadata[name] = ServiceMetadata(
            name=name,
            service_type=service_type,
            lifecycle=ServiceLifecycle.UNINITIALIZED,
            created_at=datetime.utcnow(),
            initialization_time=0.0,
            dependencies=set(),
            dependents=set()
        )
        
        logger.debug(f"📝 Registered transient service: {name}")
    
    def add_dependency(self, service: str, dependency: str) -> None:
        """Add dependency relationship between services"""
        if service not in self._dependency_graph:
            self._dependency_graph[service] = set()
        if dependency not in self._reverse_dependencies:
            self._reverse_dependencies[dependency] = set()
        
        self._dependency_graph[service].add(dependency)
        self._reverse_dependencies[dependency].add(service)
        
        # Update metadata
        if service in self._service_metadata:
            self._service_metadata[service].dependencies.add(dependency)
        if dependency in self._service_metadata:
            self._service_metadata[dependency].dependents.add(service)
        
        logger.debug(f"🔗 Added dependency: {service} -> {dependency}")
    
    def register_health_check(self, service_name: str, health_check: Callable) -> None:
        """Register health check for a service"""
        self._health_checks[service_name] = health_check
        logger.debug(f"🏥 Registered health check for: {service_name}")
    
    async def get_service(self, name: str) -> Any:
        """Get service instance with dependency resolution"""
        # Check if singleton exists
        if name in self._singletons and name in self._services:
            return self._services[name]
        
        # Create service instance
        service = await self._create_service(name)
        
        # Store singleton
        if name in self._singletons:
            self._services[name] = service
        
        return service
    
    async def _create_service(self, name: str) -> Any:
        """Create service instance with dependency injection"""
        if name not in self._service_factories:
            raise ValueError(f"Service '{name}' not registered")
        
        metadata = self._service_metadata[name]
        metadata.lifecycle = ServiceLifecycle.INITIALIZING
        
        start_time = time.time()
        
        try:
            # Resolve dependencies first
            dependencies = {}
            for dep_name in self._dependency_graph.get(name, set()):
                dependencies[dep_name] = await self.get_service(dep_name)
            
            # Create service instance
            factory = self._service_factories[name]
            
            if inspect.iscoroutinefunction(factory):
                service = await factory(**dependencies)
            elif callable(factory):
                # Check if factory accepts dependencies
                sig = inspect.signature(factory)
                if sig.parameters:
                    service = factory(**dependencies)
                else:
                    service = factory()
            else:
                # Direct class instantiation
                service = factory()
            
            # Initialize service if it has an initialize method
            if hasattr(service, 'initialize') and callable(service.initialize):
                if inspect.iscoroutinefunction(service.initialize):
                    await service.initialize()
                else:
                    service.initialize()
            
            # Update metadata
            metadata.lifecycle = ServiceLifecycle.RUNNING
            metadata.initialization_time = time.time() - start_time
            metadata.health_status = "healthy"
            metadata.last_health_check = datetime.utcnow()
            
            logger.info(f"✅ Service created: {name} ({metadata.initialization_time:.3f}s)")
            
            return service
            
        except Exception as e:
            metadata.lifecycle = ServiceLifecycle.FAILED
            metadata.initialization_time = time.time() - start_time
            metadata.health_status = "failed"
            
            logger.error(f"❌ Service creation failed: {name} - {e}")
            raise
    
    async def initialize_all(self) -> None:
        """Initialize all registered services in dependency order"""
        logger.info("🚀 Initializing all services...")
        
        # Calculate initialization order
        initialization_order = self._calculate_initialization_order()
        
        # Initialize services in order
        for service_name in initialization_order:
            if service_name in self._singletons:
                try:
                    await self.get_service(service_name)
                except Exception as e:
                    logger.error(f"Failed to initialize service {service_name}: {e}")
        
        # Start health monitoring
        await self._start_health_monitoring()
        
        logger.info("✅ All services initialized")
    
    def _calculate_initialization_order(self) -> list:
        """Calculate service initialization order based on dependencies"""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service: str):
            if service in temp_visited:
                raise ValueError(f"Circular dependency detected involving service: {service}")
            
            if service not in visited:
                temp_visited.add(service)
                
                # Visit dependencies first
                for dependency in self._dependency_graph.get(service, set()):
                    visit(dependency)
                
                temp_visited.remove(service)
                visited.add(service)
                order.append(service)
        
        # Visit all services
        for service in self._service_factories.keys():
            if service not in visited:
                visit(service)
        
        return order
    
    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring for all services"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("🏥 Started health monitoring for all services")
    
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop"""
        while self._monitoring_active:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all services"""
        for service_name, health_check in self._health_checks.items():
            try:
                if service_name in self._services:
                    service = self._services[service_name]
                    
                    if inspect.iscoroutinefunction(health_check):
                        is_healthy = await health_check(service)
                    else:
                        is_healthy = health_check(service)
                    
                    metadata = self._service_metadata[service_name]
                    metadata.health_status = "healthy" if is_healthy else "unhealthy"
                    metadata.last_health_check = datetime.utcnow()
                    
            except Exception as e:
                metadata = self._service_metadata[service_name]
                metadata.health_status = "error"
                metadata.last_health_check = datetime.utcnow()
                logger.error(f"Health check failed for {service_name}: {e}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all services"""
        logger.info("🔄 Shutting down all services...")
        
        # Stop health monitoring
        self._monitoring_active = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown services in reverse dependency order
        shutdown_order = list(reversed(self._initialization_order))
        
        for service_name in shutdown_order:
            if service_name in self._services:
                try:
                    service = self._services[service_name]
                    metadata = self._service_metadata[service_name]
                    metadata.lifecycle = ServiceLifecycle.STOPPING
                    
                    # Call shutdown method if available
                    if hasattr(service, 'shutdown') and callable(service.shutdown):
                        if inspect.iscoroutinefunction(service.shutdown):
                            await service.shutdown()
                        else:
                            service.shutdown()
                    
                    metadata.lifecycle = ServiceLifecycle.STOPPED
                    
                    # Remove from services
                    del self._services[service_name]
                    
                    logger.debug(f"🔄 Service shutdown: {service_name}")
                    
                except Exception as e:
                    logger.error(f"Error shutting down service {service_name}: {e}")
        
        logger.info("✅ All services shutdown complete")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all registered services"""
        status = {
            "total_services": len(self._service_metadata),
            "running_services": len([s for s in self._service_metadata.values() if s.lifecycle == ServiceLifecycle.RUNNING]),
            "failed_services": len([s for s in self._service_metadata.values() if s.lifecycle == ServiceLifecycle.FAILED]),
            "healthy_services": len([s for s in self._service_metadata.values() if s.health_status == "healthy"]),
            "monitoring_active": self._monitoring_active,
            "services": {}
        }
        
        for name, metadata in self._service_metadata.items():
            status["services"][name] = {
                "lifecycle": metadata.lifecycle.value,
                "health_status": metadata.health_status,
                "initialization_time": metadata.initialization_time,
                "dependencies": list(metadata.dependencies),
                "dependents": list(metadata.dependents),
                "last_health_check": metadata.last_health_check.isoformat() if metadata.last_health_check else None
            }
        
        return status
    
    def is_healthy(self) -> bool:
        """Check if all services are healthy"""
        for metadata in self._service_metadata.values():
            if metadata.lifecycle == ServiceLifecycle.FAILED or metadata.health_status in ["unhealthy", "error"]:
                return False
        return True
    
    def get_dependency_graph(self) -> Dict[str, Any]:
        """Get the complete dependency graph"""
        return {
            "dependencies": {k: list(v) for k, v in self._dependency_graph.items()},
            "reverse_dependencies": {k: list(v) for k, v in self._reverse_dependencies.items()},
            "initialization_order": self._initialization_order
        }


# Global enhanced container instance
enhanced_container = EnhancedDependencyContainer()
