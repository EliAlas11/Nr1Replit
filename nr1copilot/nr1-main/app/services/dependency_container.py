
"""
Netflix-Level Dependency Injection Container
Centralized service management with lifecycle control
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from app.config import settings
from app.services.video_service import VideoService
from app.services.ai_analyzer import AIAnalyzer
from app.services.realtime_engine import RealtimeEngine
from app.services.cloud_processor import CloudProcessor
from app.utils.metrics import MetricsCollector
from app.utils.health import HealthChecker
from app.utils.rate_limiter import RateLimiter
from app.utils.security import SecurityManager
from app.utils.cache import CacheManager

logger = logging.getLogger(__name__)


class DependencyContainer:
    """Netflix-level dependency injection with lifecycle management"""
    
    def __init__(self):
        # Core services
        self.video_service: Optional[VideoService] = None
        self.ai_analyzer: Optional[AIAnalyzer] = None
        self.realtime_engine: Optional[RealtimeEngine] = None
        self.cloud_processor: Optional[CloudProcessor] = None
        
        # Utility services
        self.metrics_collector: Optional[MetricsCollector] = None
        self.health_checker: Optional[HealthChecker] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.security_manager: Optional[SecurityManager] = None
        self.cache_manager: Optional[CacheManager] = None
        
        # State management
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        self._services_health: Dict[str, bool] = {}
        
    async def initialize(self):
        """Initialize all services with proper dependency order"""
        if self._initialized:
            return
            
        async with self._initialization_lock:
            if self._initialized:  # Double-check
                return
                
            try:
                logger.info("🔧 Initializing Netflix-Level Service Container...")
                
                # Phase 1: Core infrastructure services
                await self._init_infrastructure_services()
                
                # Phase 2: Business logic services
                await self._init_business_services()
                
                # Phase 3: Cross-cutting services
                await self._init_cross_cutting_services()
                
                # Phase 4: Start background services
                await self._start_background_services()
                
                self._initialized = True
                logger.info("✅ Service container initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Service container initialization failed: {e}", exc_info=True)
                await self.cleanup()
                raise
                
    async def _init_infrastructure_services(self):
        """Initialize infrastructure services first"""
        # Cache manager (foundational)
        self.cache_manager = CacheManager()
        await self.cache_manager.initialize()
        self._services_health["cache_manager"] = True
        
        # Security manager
        self.security_manager = SecurityManager()
        await self.security_manager.initialize()
        self._services_health["security_manager"] = True
        
        # Rate limiter
        self.rate_limiter = RateLimiter()
        await self.rate_limiter.initialize()
        self._services_health["rate_limiter"] = True
        
        logger.info("✅ Infrastructure services initialized")
        
    async def _init_business_services(self):
        """Initialize core business services"""
        # Video service
        self.video_service = VideoService(
            cache_manager=self.cache_manager,
            security_manager=self.security_manager
        )
        await self.video_service.initialize()
        self._services_health["video_service"] = True
        
        # AI analyzer
        self.ai_analyzer = AIAnalyzer(
            cache_manager=self.cache_manager
        )
        await self.ai_analyzer.initialize()
        self._services_health["ai_analyzer"] = True
        
        # Cloud processor
        self.cloud_processor = CloudProcessor(
            cache_manager=self.cache_manager,
            ai_analyzer=self.ai_analyzer
        )
        await self.cloud_processor.initialize()
        self._services_health["cloud_processor"] = True
        
        # Realtime engine
        self.realtime_engine = RealtimeEngine(
            video_service=self.video_service,
            ai_analyzer=self.ai_analyzer,
            cloud_processor=self.cloud_processor,
            cache_manager=self.cache_manager
        )
        await self.realtime_engine.initialize()
        self._services_health["realtime_engine"] = True
        
        logger.info("✅ Business services initialized")
        
    async def _init_cross_cutting_services(self):
        """Initialize cross-cutting concern services"""
        # Metrics collector
        self.metrics_collector = MetricsCollector(
            cache_manager=self.cache_manager
        )
        await self.metrics_collector.initialize()
        self._services_health["metrics_collector"] = True
        
        # Health checker
        self.health_checker = HealthChecker(
            services={
                "video_service": self.video_service,
                "ai_analyzer": self.ai_analyzer,
                "realtime_engine": self.realtime_engine,
                "cloud_processor": self.cloud_processor,
                "cache_manager": self.cache_manager,
                "security_manager": self.security_manager,
                "rate_limiter": self.rate_limiter,
                "metrics_collector": self.metrics_collector
            }
        )
        await self.health_checker.initialize()
        self._services_health["health_checker"] = True
        
        logger.info("✅ Cross-cutting services initialized")
        
    async def _start_background_services(self):
        """Start background processing services"""
        # Start realtime engine
        if self.realtime_engine:
            await self.realtime_engine.start()
            
        # Start metrics collection
        if self.metrics_collector:
            await self.metrics_collector.start()
            
        # Start health monitoring
        if self.health_checker:
            await self.health_checker.start()
            
        logger.info("✅ Background services started")
        
    async def preload_critical_data(self):
        """Preload critical data for better performance"""
        try:
            # Preload AI models
            if self.ai_analyzer:
                await self.ai_analyzer.preload_models()
                
            # Warm up cache with critical data
            if self.cache_manager:
                await self.cache_manager.preload_critical_data()
                
            # Initialize connection pools
            if self.realtime_engine:
                await self.realtime_engine.initialize_connection_pools()
                
            logger.info("✅ Critical data preloaded")
            
        except Exception as e:
            logger.warning(f"⚠️ Critical data preload partially failed: {e}")
            
    async def cleanup(self):
        """Graceful cleanup of all services"""
        if not self._initialized:
            return
            
        logger.info("🔄 Cleaning up service container...")
        
        # Stop background services first
        cleanup_tasks = []
        
        if self.realtime_engine:
            cleanup_tasks.append(self.realtime_engine.stop())
            
        if self.metrics_collector:
            cleanup_tasks.append(self.metrics_collector.stop())
            
        if self.health_checker:
            cleanup_tasks.append(self.health_checker.stop())
            
        if self.cloud_processor:
            cleanup_tasks.append(self.cloud_processor.cleanup())
            
        if self.cache_manager:
            cleanup_tasks.append(self.cache_manager.cleanup())
            
        # Execute cleanup in parallel with timeout
        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("⚠️ Cleanup timeout - some services may not have shut down cleanly")
            except Exception as e:
                logger.error(f"❌ Cleanup error: {e}")
                
        self._initialized = False
        self._services_health.clear()
        logger.info("✅ Service container cleanup complete")
        
    @property
    def is_healthy(self) -> bool:
        """Check if all services are healthy"""
        return self._initialized and all(self._services_health.values())
        
    async def get_service_status(self) -> Dict[str, Any]:
        """Get detailed service status"""
        return {
            "initialized": self._initialized,
            "services": self._services_health.copy(),
            "timestamp": datetime.utcnow().isoformat(),
            "healthy_services": sum(self._services_health.values()),
            "total_services": len(self._services_health)
        }
