"""
Netflix-Level Dependency Injection Container
Advanced dependency management with lifecycle control and health monitoring
"""

import asyncio
import logging
from typing import Dict, Any, Type, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import weakref
from datetime import datetime
import inspect

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTDOWN = "shutdown"


@dataclass
class ServiceDescriptor:
    """Service descriptor with metadata"""
    service_type: Type
    instance: Optional[Any] = None
    state: ServiceState = ServiceState.UNINITIALIZED
    dependencies: List[str] = field(default_factory=list)
    health_check: Optional[Callable] = None
    last_health_check: Optional[datetime] = None
    initialization_time: Optional[float] = None
    error_count: int = 0
    max_retries: int = 3


class DependencyContainer:
    """Netflix-level dependency injection container"""

    def __init__(self):
        self._services: Dict[str, ServiceDescriptor] = {}
        self._instances: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._initialization_order: List[str] = []
        self._health_monitor_task: Optional[asyncio.Task] = None

    def register_service(
        self,
        name: str,
        service_type: Type,
        dependencies: List[str] = None,
        health_check: Callable = None,
        singleton: bool = True
    ):
        """Register a service with the container"""
        self._services[name] = ServiceDescriptor(
            service_type=service_type,
            dependencies=dependencies or [],
            health_check=health_check
        )

        logger.debug(f"Registered service: {name}")

    async def get_service(self, name: str) -> Any:
        """Get service instance with automatic initialization"""
        if name not in self._services:
            raise ValueError(f"Service not registered: {name}")

        descriptor = self._services[name]

        if descriptor.instance is None:
            await self._initialize_service(name)

        return descriptor.instance

    async def _initialize_service(self, name: str):
        """Initialize service with dependency resolution"""
        descriptor = self._services[name]

        if descriptor.state == ServiceState.INITIALIZING:
            # Prevent circular dependency deadlock
            return

        descriptor.state = ServiceState.INITIALIZING

        try:
            # Initialize dependencies first
            dependencies = {}
            for dep_name in descriptor.dependencies:
                dependencies[dep_name] = await self.get_service(dep_name)

            # Create service instance
            start_time = asyncio.get_event_loop().time()

            if inspect.iscoroutinefunction(descriptor.service_type.__init__):
                instance = await descriptor.service_type(**dependencies)
            else:
                instance = descriptor.service_type(**dependencies)

            # Initialize if has async init method
            if hasattr(instance, 'initialize') and inspect.iscoroutinefunction(instance.initialize):
                await instance.initialize()

            descriptor.instance = instance
            descriptor.initialization_time = asyncio.get_event_loop().time() - start_time
            descriptor.state = ServiceState.HEALTHY

            self._instances[name] = instance

            logger.info(f"Service initialized: {name} ({descriptor.initialization_time:.3f}s)")

        except Exception as e:
            descriptor.state = ServiceState.UNHEALTHY
            descriptor.error_count += 1
            logger.error(f"Service initialization failed: {name} - {e}")
            raise

    async def initialize_all_services(self):
        """Initialize all registered services in dependency order"""
        logger.info("Initializing all services...")

        # Calculate initialization order
        self._calculate_initialization_order()

        # Initialize services
        initialization_tasks = []
        for service_name in self._initialization_order:
            task = asyncio.create_task(self._initialize_service(service_name))
            initialization_tasks.append(task)

        # Wait for all services to initialize
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

        # Report results
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        total_count = len(results)

        logger.info(f"Service initialization complete: {success_count}/{total_count} successful")

        # Start health monitoring
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

        return success_count == total_count

    def _calculate_initialization_order(self):
        """Calculate service initialization order based on dependencies"""
        visited = set()
        temp_visited = set()
        self._initialization_order = []

        def visit(service_name: str):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected: {service_name}")

            if service_name in visited:
                return

            temp_visited.add(service_name)

            descriptor = self._services[service_name]
            for dep_name in descriptor.dependencies:
                if dep_name in self._services:
                    visit(dep_name)

            temp_visited.remove(service_name)
            visited.add(service_name)
            self._initialization_order.append(service_name)

        for service_name in self._services:
            if service_name not in visited:
                visit(service_name)

    async def _health_monitor_loop(self):
        """Continuous health monitoring of services"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                for name, descriptor in self._services.items():
                    if descriptor.instance and descriptor.health_check:
                        try:
                            if inspect.iscoroutinefunction(descriptor.health_check):
                                is_healthy = await descriptor.health_check(descriptor.instance)
                            else:
                                is_healthy = descriptor.health_check(descriptor.instance)

                            descriptor.last_health_check = datetime.utcnow()

                            if is_healthy:
                                if descriptor.state == ServiceState.UNHEALTHY:
                                    descriptor.state = ServiceState.HEALTHY
                                    logger.info(f"Service recovered: {name}")
                            else:
                                descriptor.state = ServiceState.DEGRADED
                                logger.warning(f"Service degraded: {name}")

                        except Exception as e:
                            descriptor.state = ServiceState.UNHEALTHY
                            descriptor.error_count += 1
                            logger.error(f"Health check failed for {name}: {e}")

                            # Attempt recovery if within retry limit
                            if descriptor.error_count <= descriptor.max_retries:
                                logger.info(f"Attempting service recovery: {name}")
                                await self._recover_service(name)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    async def _recover_service(self, name: str):
        """Attempt to recover a failed service"""
        try:
            descriptor = self._services[name]

            # Reset instance
            descriptor.instance = None
            descriptor.state = ServiceState.UNINITIALIZED

            # Reinitialize
            await self._initialize_service(name)

            logger.info(f"Service recovery successful: {name}")

        except Exception as e:
            logger.error(f"Service recovery failed: {name} - {e}")

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {}

        for name, descriptor in self._services.items():
            status[name] = {
                "state": descriptor.state.value,
                "initialized": descriptor.instance is not None,
                "initialization_time": descriptor.initialization_time,
                "last_health_check": descriptor.last_health_check.isoformat() if descriptor.last_health_check else None,
                "error_count": descriptor.error_count,
                "dependencies": descriptor.dependencies
            }

        return status

    async def shutdown_all_services(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down all services...")

        # Cancel health monitoring
        if self._health_monitor_task:
            self._health_monitor_task.cancel()

        # Shutdown services in reverse order
        shutdown_order = list(reversed(self._initialization_order))

        for service_name in shutdown_order:
            descriptor = self._services[service_name]

            if descriptor.instance:
                try:
                    if hasattr(descriptor.instance, 'shutdown'):
                        if inspect.iscoroutinefunction(descriptor.instance.shutdown):
                            await descriptor.instance.shutdown()
                        else:
                            descriptor.instance.shutdown()

                    descriptor.state = ServiceState.SHUTDOWN
                    descriptor.instance = None

                    logger.debug(f"Service shutdown: {service_name}")

                except Exception as e:
                    logger.error(f"Error shutting down service {service_name}: {e}")

        logger.info("All services shut down")


# Global container instance
container = DependencyContainer()

# Service health check helpers
async def basic_health_check(instance: Any) -> bool:
    """Basic health check for services"""
    return hasattr(instance, '_healthy') and getattr(instance, '_healthy', True)

def sync_health_check(instance: Any) -> bool:
    """Synchronous health check for services"""
    return hasattr(instance, '_healthy') and getattr(instance, '_healthy', True)

# Export container and utilities
__all__ = [
    'DependencyContainer',
    'ServiceState',
    'ServiceDescriptor',
    'container',
    'basic_health_check',
    'sync_health_check'
]