"""
Netflix-Grade Dependency Injection Container
Centralized service management with lifecycle control and health monitoring
"""
import asyncio
import logging
from typing import Dict, Any, Optional, TypeVar, Type
from datetime import datetime
import weakref

from app.config import get_settings
from app.database.connection import DatabaseManager
from app.utils.cache import CacheManager
from app.utils.metrics import MetricsCollector
from app.utils.health import SystemHealthMonitor

logger = logging.getLogger(__name__)
T = TypeVar('T')


class ServiceContainer:
    """Netflix-grade service container with dependency injection and lifecycle management"""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._initialized = False
        self._startup_time: Optional[float] = None
        self._shutdown_callbacks: Dict[str, callable] = {}
        self.settings = get_settings()

        logger.info("🏗️ Service container initializing...")

    async def initialize(self) -> None:
        """Initialize all services with proper dependency order"""
        if self._initialized:
            return

        startup_start = asyncio.get_event_loop().time()

        try:
            # Phase 1: Core infrastructure
            await self._initialize_core_services()

            # Phase 2: Business services
            await self._initialize_business_services()

            # Phase 3: Monitoring services
            await self._initialize_monitoring_services()

            self._startup_time = asyncio.get_event_loop().time() - startup_start
            self._initialized = True

            logger.info(f"✅ Service container initialized in {self._startup_time:.3f}s")

        except Exception as e:
            logger.error(f"❌ Service container initialization failed: {e}")
            await self._cleanup_services()
            raise

    async def _initialize_core_services(self) -> None:
        """Initialize core infrastructure services"""
        logger.info("📊 Initializing core services...")

        # Database manager
        db_manager = DatabaseManager()
        await db_manager.initialize()
        self._services["database"] = db_manager
        self._shutdown_callbacks["database"] = db_manager.shutdown

        # Cache manager
        cache_manager = CacheManager()
        await cache_manager.initialize()
        self._services["cache"] = cache_manager
        self._shutdown_callbacks["cache"] = cache_manager.shutdown

        # Metrics collector
        metrics_collector = MetricsCollector()
        await metrics_collector.start()
        self._services["metrics"] = metrics_collector
        self._shutdown_callbacks["metrics"] = metrics_collector.shutdown

        logger.info("✅ Core services initialized")

    async def _initialize_business_services(self) -> None:
        """Initialize business logic services"""
        logger.info("🧠 Initializing business services...")

        # Lazy import to avoid circular dependencies
        from app.services.video_service import VideoService
        from app.services.ai_intelligence_engine import AIIntelligenceEngine

        # Video service
        video_service = VideoService()
        await video_service.initialize()
        self._services["video"] = video_service
        self._shutdown_callbacks["video"] = video_service.shutdown

        # AI engine
        ai_engine = AIIntelligenceEngine()
        await ai_engine.initialize()
        self._services["ai"] = ai_engine
        self._shutdown_callbacks["ai"] = ai_engine.shutdown

        logger.info("✅ Business services initialized")

    async def _initialize_monitoring_services(self) -> None:
        """Initialize monitoring and health services"""
        logger.info("🏥 Initializing monitoring services...")

        # Health monitor
        health_monitor = SystemHealthMonitor()
        await health_monitor.initialize()
        self._services["health"] = health_monitor
        self._shutdown_callbacks["health"] = health_monitor.shutdown

        logger.info("✅ Monitoring services initialized")

    def get_database_manager(self) -> DatabaseManager:
        """Get database manager instance"""
        return self._get_service("database", DatabaseManager)

    def get_cache_manager(self) -> CacheManager:
        """Get cache manager instance"""
        return self._get_service("cache", CacheManager)

    def get_metrics_collector(self) -> MetricsCollector:
        """Get metrics collector instance"""
        return self._get_service("metrics", MetricsCollector)

    def get_health_monitor(self) -> SystemHealthMonitor:
        """Get health monitor instance"""
        return self._get_service("health", SystemHealthMonitor)

    def get_video_service(self):
        """Get video service instance"""
        return self._get_service("video", None)

    def get_ai_engine(self):
        """Get AI engine instance"""
        return self._get_service("ai", None)

    def _get_service(self, name: str, expected_type: Optional[Type[T]] = None) -> T:
        """Get service with type checking"""
        if not self._initialized:
            raise RuntimeError(f"Service container not initialized")

        service = self._services.get(name)
        if service is None:
            raise RuntimeError(f"Service '{name}' not found")

        if expected_type and not isinstance(service, expected_type):
            raise RuntimeError(f"Service '{name}' has unexpected type")

        return service

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        return {
            "initialized": self._initialized,
            "startup_time": self._startup_time,
            "service_count": len(self._services),
            "services": list(self._services.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }

    def is_healthy(self) -> bool:
        """Check if all services are healthy"""
        return self._initialized and len(self._services) > 0

    async def shutdown(self) -> None:
        """Gracefully shutdown all services"""
        if not self._initialized:
            return

        logger.info("🛑 Shutting down service container...")

        # Shutdown in reverse order
        for service_name in reversed(list(self._shutdown_callbacks.keys())):
            try:
                shutdown_callback = self._shutdown_callbacks[service_name]
                if shutdown_callback:
                    await shutdown_callback()
                    logger.info(f"✅ {service_name} service shutdown complete")
            except Exception as e:
                logger.error(f"❌ {service_name} service shutdown failed: {e}")

        await self._cleanup_services()
        logger.info("✅ Service container shutdown complete")

    async def _cleanup_services(self) -> None:
        """Clean up service references"""
        self._services.clear()
        self._shutdown_callbacks.clear()
        self._initialized = False
        self._startup_time = None


# Global service container instance
_container: Optional[ServiceContainer] = None


def get_service_container() -> ServiceContainer:
    """Get global service container instance"""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


async def initialize_services() -> None:
    """Initialize global service container"""
    container = get_service_container()
    await container.initialize()


async def shutdown_services() -> None:
    """Shutdown global service container"""
    global _container
    if _container:
        await _container.shutdown()
        _container = None