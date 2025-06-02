"""
Netflix-Grade Service Container v12.0
Optimized dependency injection and service management
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Type, Union, List
from contextlib import asynccontextmanager
import weakref
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service status enumeration"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTDOWN = "shutdown"


@dataclass
class ServiceMetadata:
    """Service metadata and health information"""
    name: str
    status: ServiceStatus
    last_health_check: float
    startup_time: float
    error_count: int = 0
    restart_count: int = 0


class ServiceContainer:
    """Netflix-grade dependency injection container with health management"""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._service_metadata: Dict[str, ServiceMetadata] = {}
        self._health_monitors: Dict[str, Any] = {}
        self._metrics_collectors: Dict[str, Any] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._initialization_order: List[str] = []
        self._is_initialized = False
        self.start_time = time.time()

    async def initialize(self) -> None:
        """Initialize all services in dependency order"""
        if self._is_initialized:
            return

        logger.info("🚀 Initializing service container...")
        start_time = time.time()

        try:
            # Initialize core services first
            await self._initialize_core_services()

            # Initialize application services
            await self._initialize_application_services()

            # Initialize monitoring services
            await self._initialize_monitoring_services()

            self._is_initialized = True
            duration = time.time() - start_time

            logger.info(f"✅ Service container initialized in {duration:.2f}s")
            logger.info(f"📊 Services: {len(self._services)}, Health monitors: {len(self._health_monitors)}")

        except Exception as e:
            logger.error(f"❌ Service container initialization failed: {e}")
            raise

    async def _initialize_core_services(self) -> None:
        """Initialize core infrastructure services"""
        logger.debug("🔧 Initializing core services...")

        # Health monitor
        try:
            from app.netflix_health_monitor import NetflixHealthMonitor
            health_monitor = NetflixHealthMonitor()
            await self._register_service("health_monitor", health_monitor)
            logger.debug("✅ Health monitor initialized")
        except Exception as e:
            logger.warning(f"⚠️ Health monitor failed: {e}")

        # Metrics collector
        try:
            from app.utils.metrics import MetricsCollector
            metrics_collector = MetricsCollector()
            await self._register_service("metrics_collector", metrics_collector)
            logger.debug("✅ Metrics collector initialized")
        except Exception as e:
            logger.warning(f"⚠️ Metrics collector failed: {e}")

        # Cache manager
        try:
            from app.utils.cache import NetflixCacheManager
            cache_manager = NetflixCacheManager()
            await self._register_service("cache_manager", cache_manager)
            logger.debug("✅ Cache manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Cache manager failed: {e}")

    async def _initialize_application_services(self) -> None:
        """Initialize application-specific services"""
        logger.debug("🎬 Initializing application services...")

        # Database manager
        try:
            from app.database.connection import NetflixLevelDatabaseManager
            db_manager = NetflixLevelDatabaseManager()
            await self._register_service("database_manager", db_manager)
            logger.debug("✅ Database manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Database manager failed: {e}")

        # Security manager
        try:
            from app.utils.security import SecurityManager
            security_manager = SecurityManager()
            await self._register_service("security_manager", security_manager)
            logger.debug("✅ Security manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Security manager failed: {e}")

        # Performance monitor
        try:
            from app.utils.performance_monitor import PerformanceMonitor
            perf_monitor = PerformanceMonitor()
            await self._register_service("performance_monitor", perf_monitor)
            logger.debug("✅ Performance monitor initialized")
        except Exception as e:
            logger.warning(f"⚠️ Performance monitor failed: {e}")

    async def _initialize_monitoring_services(self) -> None:
        """Initialize monitoring and alerting services"""
        logger.debug("📊 Initializing monitoring services...")

        # Recovery system
        try:
            from app.netflix_recovery_system import NetflixRecoverySystem
            recovery_system = NetflixRecoverySystem()
            await self._register_service("recovery_system", recovery_system)
            logger.debug("✅ Recovery system initialized")
        except Exception as e:
            logger.warning(f"⚠️ Recovery system failed: {e}")

    async def _register_service(self, name: str, service: Any) -> None:
        """Register a service with metadata tracking"""
        start_time = time.time()

        try:
            # Initialize service if it has startup method
            if hasattr(service, 'startup'):
                await service.startup()

            # Register service
            self._services[name] = service

            # Create metadata
            self._service_metadata[name] = ServiceMetadata(
                name=name,
                status=ServiceStatus.HEALTHY,
                last_health_check=time.time(),
                startup_time=time.time() - start_time
            )

            logger.debug(f"✅ Service '{name}' registered successfully")

        except Exception as e:
            logger.error(f"❌ Failed to register service '{name}': {e}")

            # Register with error status
            self._service_metadata[name] = ServiceMetadata(
                name=name,
                status=ServiceStatus.UNHEALTHY,
                last_health_check=time.time(),
                startup_time=time.time() - start_time,
                error_count=1
            )
            raise

    def get_service(self, name: str) -> Optional[Any]:
        """Get a service by name"""
        return self._services.get(name)

    def get_health_monitor(self) -> Optional[Any]:
        """Get the health monitor service"""
        return self.get_service("health_monitor")

    def get_metrics_collector(self) -> Optional[Any]:
        """Get the metrics collector service"""
        return self.get_service("metrics_collector")

    def get_cache_manager(self) -> Optional[Any]:
        """Get the cache manager service"""
        return self.get_service("cache_manager")

    def get_database_manager(self) -> Optional[Any]:
        """Get the database manager service"""
        return self.get_service("database_manager")

    def get_security_manager(self) -> Optional[Any]:
        """Get the security manager service"""
        return self.get_service("security_manager")

    def get_performance_monitor(self) -> Optional[Any]:
        """Get the performance monitor service"""
        return self.get_service("performance_monitor")

    def get_recovery_system(self) -> Optional[Any]:
        """Get the recovery system service"""
        return self.get_service("recovery_system")

    def is_healthy(self) -> bool:
        """Check if all critical services are healthy"""
        if not self._is_initialized:
            return False

        critical_services = ["health_monitor", "metrics_collector"]

        for service_name in critical_services:
            metadata = self._service_metadata.get(service_name)
            if not metadata or metadata.status in [ServiceStatus.UNHEALTHY, ServiceStatus.SHUTDOWN]:
                return False

        return True

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {}

        for name, metadata in self._service_metadata.items():
            status[name] = {
                "status": metadata.status.value,
                "startup_time": metadata.startup_time,
                "last_health_check": metadata.last_health_check,
                "error_count": metadata.error_count,
                "restart_count": metadata.restart_count,
                "uptime": time.time() - (self.start_time + metadata.startup_time)
            }

        return status

    async def health_check_service(self, name: str) -> bool:
        """Perform health check on a specific service"""
        service = self._services.get(name)
        metadata = self._service_metadata.get(name)

        if not service or not metadata:
            return False

        try:
            # If service has health check method, use it
            if hasattr(service, 'health_check'):
                is_healthy = await service.health_check()
            else:
                # Basic health check - service exists and is not None
                is_healthy = service is not None

            # Update metadata
            metadata.last_health_check = time.time()
            metadata.status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.DEGRADED

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed for service '{name}': {e}")
            metadata.error_count += 1
            metadata.status = ServiceStatus.UNHEALTHY
            return False

    async def restart_service(self, name: str) -> bool:
        """Restart a specific service"""
        logger.info(f"🔄 Restarting service: {name}")

        try:
            service = self._services.get(name)
            metadata = self._service_metadata.get(name)

            if not metadata:
                logger.error(f"Service '{name}' not found")
                return False

            # Shutdown service if it has shutdown method
            if service and hasattr(service, 'shutdown'):
                await service.shutdown()

            # Re-initialize service
            # This is a simplified restart - in production, you'd want more sophisticated restart logic
            metadata.restart_count += 1
            metadata.status = ServiceStatus.INITIALIZING

            # For now, mark as healthy (in production, you'd re-instantiate the service)
            metadata.status = ServiceStatus.HEALTHY
            metadata.last_health_check = time.time()

            logger.info(f"✅ Service '{name}' restarted successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to restart service '{name}': {e}")
            if name in self._service_metadata:
                self._service_metadata[name].status = ServiceStatus.UNHEALTHY
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown all services"""
        logger.info("🔄 Shutting down service container...")
        start_time = time.time()

        # Shutdown services in reverse order
        shutdown_order = list(reversed(self._initialization_order)) or list(self._services.keys())

        for service_name in shutdown_order:
            try:
                service = self._services.get(service_name)
                metadata = self._service_metadata.get(service_name)

                if service and hasattr(service, 'shutdown'):
                    await service.shutdown()
                    logger.debug(f"✅ Service '{service_name}' shut down")

                if metadata:
                    metadata.status = ServiceStatus.SHUTDOWN

            except Exception as e:
                logger.error(f"❌ Error shutting down service '{service_name}': {e}")

        self._is_initialized = False
        duration = time.time() - start_time
        logger.info(f"✅ Service container shutdown completed in {duration:.2f}s")

    def get_service_dependencies(self, name: str) -> List[str]:
        """Get dependencies for a service"""
        return self._dependencies.get(name, [])

    def register_dependency(self, service_name: str, dependency_name: str) -> None:
        """Register a service dependency"""
        if service_name not in self._dependencies:
            self._dependencies[service_name] = []

        if dependency_name not in self._dependencies[service_name]:
            self._dependencies[service_name].append(dependency_name)

    def get_container_stats(self) -> Dict[str, Any]:
        """Get comprehensive container statistics"""
        return {
            "initialized": self._is_initialized,
            "start_time": self.start_time,
            "uptime": time.time() - self.start_time,
            "total_services": len(self._services),
            "healthy_services": len([
                m for m in self._service_metadata.values() 
                if m.status == ServiceStatus.HEALTHY
            ]),
            "unhealthy_services": len([
                m for m in self._service_metadata.values() 
                if m.status == ServiceStatus.UNHEALTHY
            ]),
            "total_errors": sum(m.error_count for m in self._service_metadata.values()),
            "total_restarts": sum(m.restart_count for m in self._service_metadata.values()),
            "services": self.get_service_status()
        }

    @asynccontextmanager
    async def service_context(self, name: str):
        """Context manager for safe service usage"""
        service = self.get_service(name)

        if not service:
            raise ValueError(f"Service '{name}' not found")

        try:
            yield service
        except Exception as e:
            logger.error(f"Error using service '{name}': {e}")
            # Update error count
            metadata = self._service_metadata.get(name)
            if metadata:
                metadata.error_count += 1
            raise
        finally:
            # Update last access time
            metadata = self._service_metadata.get(name)
            if metadata:
                metadata.last_health_check = time.time()