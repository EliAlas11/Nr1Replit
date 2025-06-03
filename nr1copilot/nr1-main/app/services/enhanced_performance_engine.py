
"""
Netflix-Grade Performance Engine v15.0
Ultra-optimized performance management with advanced caching and resource optimization
"""

import asyncio
import logging
import time
import gc
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import wraps, lru_cache
import threading
import psutil
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics tracking"""
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    request_count: int = 0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    active_connections: int = 0
    throughput_rps: float = 0.0
    last_optimization: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage"""
        total_requests = self.request_count + self.error_count
        return (self.error_count / total_requests * 100) if total_requests > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        total_cache_requests = self.cache_hits + self.cache_misses
        return (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0.0


class NetflixGradeCache(Generic[T]):
    """Ultra-fast memory cache with TTL and intelligent eviction"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0
        }
    
    async def get(self, key: str) -> Optional[T]:
        """Get item from cache with TTL check"""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            item = self._cache[key]
            
            # Check TTL
            if datetime.utcnow() > item['expires']:
                del self._cache[key]
                del self._access_times[key]
                self._stats['misses'] += 1
                return None
            
            # Update access time for LRU
            self._access_times[key] = datetime.utcnow()
            self._stats['hits'] += 1
            return item['value']
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set item in cache with TTL"""
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                await self._evict_lru()
            
            ttl = ttl or self._default_ttl
            expires = datetime.utcnow() + timedelta(seconds=ttl)
            
            self._cache[key] = {
                'value': value,
                'expires': expires,
                'created': datetime.utcnow()
            }
            self._access_times[key] = datetime.utcnow()
            self._stats['sets'] += 1
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
        self._stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hit_rate': hit_rate,
                **self._stats
            }


class ConnectionPool:
    """High-performance connection pool for database/external services"""
    
    def __init__(self, create_connection: Callable, max_connections: int = 20, 
                 min_connections: int = 5, connection_timeout: float = 30.0):
        self._create_connection = create_connection
        self._max_connections = max_connections
        self._min_connections = min_connections
        self._connection_timeout = connection_timeout
        
        self._pool: deque = deque()
        self._active_connections: Set = set()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_connections)
        
        # Statistics
        self._stats = {
            'total_created': 0,
            'total_borrowed': 0,
            'total_returned': 0,
            'pool_exhausted_count': 0
        }
    
    async def initialize(self) -> None:
        """Initialize connection pool with minimum connections"""
        async with self._lock:
            for _ in range(self._min_connections):
                try:
                    connection = await self._create_connection()
                    self._pool.append(connection)
                    self._stats['total_created'] += 1
                except Exception as e:
                    logger.error(f"Failed to create initial connection: {e}")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool with automatic return"""
        async with self._semaphore:
            connection = await self._borrow_connection()
            try:
                yield connection
            finally:
                await self._return_connection(connection)
    
    async def _borrow_connection(self):
        """Borrow connection from pool"""
        async with self._lock:
            if self._pool:
                connection = self._pool.popleft()
            else:
                if len(self._active_connections) >= self._max_connections:
                    self._stats['pool_exhausted_count'] += 1
                    raise Exception("Connection pool exhausted")
                
                connection = await self._create_connection()
                self._stats['total_created'] += 1
            
            self._active_connections.add(connection)
            self._stats['total_borrowed'] += 1
            return connection
    
    async def _return_connection(self, connection) -> None:
        """Return connection to pool"""
        async with self._lock:
            self._active_connections.discard(connection)
            
            if len(self._pool) < self._max_connections:
                self._pool.append(connection)
                self._stats['total_returned'] += 1
            else:
                # Close excess connection
                try:
                    await connection.close()
                except Exception:
                    pass
    
    async def close_all(self) -> None:
        """Close all connections in pool"""
        async with self._lock:
            while self._pool:
                connection = self._pool.popleft()
                try:
                    await connection.close()
                except Exception:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'pool_size': len(self._pool),
            'active_connections': len(self._active_connections),
            'max_connections': self._max_connections,
            **self._stats
        }


class CircuitBreaker:
    """Netflix-style circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = 'closed'  # closed, open, half-open
        
        self._lock = threading.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == 'open':
                if self._should_attempt_reset():
                    self.state = 'half-open'
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                await self._on_success()
                return result
            except Exception as e:
                await self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (self.last_failure_time and 
                datetime.utcnow() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout))
    
    async def _on_success(self) -> None:
        """Handle successful execution"""
        if self.state == 'half-open':
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = 'closed'
                self.failure_count = 0
        elif self.state == 'closed':
            self.failure_count = 0
    
    async def _on_failure(self) -> None:
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


def performance_monitor(cache_key: Optional[str] = None, timeout: float = 30.0):
    """Decorator for performance monitoring and caching"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Try cache first if cache_key provided
            if cache_key:
                cached_result = await performance_engine.cache.get(cache_key)
                if cached_result is not None:
                    performance_engine.metrics.cache_hits += 1
                    return cached_result
                performance_engine.metrics.cache_misses += 1
            
            try:
                # Execute with timeout
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result if cache_key provided
                if cache_key:
                    await performance_engine.cache.set(cache_key, result)
                
                # Record metrics
                execution_time = time.time() - start_time
                performance_engine.metrics.response_times.append(execution_time)
                performance_engine.metrics.request_count += 1
                
                return result
                
            except Exception as e:
                performance_engine.metrics.error_count += 1
                logger.error(f"Performance monitored function failed: {e}")
                raise
        
        return wrapper
    return decorator


class NetflixGradePerformanceEngine:
    """Ultra-high performance engine with Netflix-level optimizations"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.cache = NetflixGradeCache[Any](max_size=50000, default_ttl=3600)
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Performance monitoring
        self._monitoring_active = False
        self._monitoring_tasks: List[asyncio.Task] = []
        
        # Resource optimization
        self._memory_threshold = 80.0  # Percentage
        self._cpu_threshold = 85.0     # Percentage
        
        logger.info("🚀 Netflix-Grade Performance Engine v15.0 initialized")
    
    async def initialize(self) -> None:
        """Initialize performance engine"""
        await self.cache.clear()
        await self._start_monitoring()
        logger.info("✅ Performance engine initialized")
    
    async def _start_monitoring(self) -> None:
        """Start background monitoring tasks"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        monitoring_tasks = [
            self._memory_optimizer(),
            self._performance_collector(),
            self._health_monitor(),
            self._cache_optimizer()
        ]
        
        for task_coro in monitoring_tasks:
            task = asyncio.create_task(task_coro)
            self._monitoring_tasks.append(task)
        
        logger.info("🔄 Performance monitoring started")
    
    async def _memory_optimizer(self) -> None:
        """Intelligent memory optimization"""
        while self._monitoring_active:
            try:
                memory = psutil.virtual_memory()
                self.metrics.memory_usage_mb = memory.used / (1024 * 1024)
                
                if memory.percent > self._memory_threshold:
                    # Aggressive garbage collection
                    collected = gc.collect()
                    
                    # Clear cache if memory still high
                    if memory.percent > 90:
                        cache_stats = self.cache.get_stats()
                        logger.warning(f"High memory usage, clearing cache (size: {cache_stats['size']})")
                        await self.cache.clear()
                    
                    logger.info(f"🧠 Memory optimization: collected {collected} objects")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Memory optimizer error: {e}")
                await asyncio.sleep(120)
    
    async def _performance_collector(self) -> None:
        """Collect performance metrics"""
        while self._monitoring_active:
            try:
                # CPU metrics
                self.metrics.cpu_percent = psutil.cpu_percent(interval=1.0)
                
                # Calculate throughput
                if len(self.metrics.response_times) > 0:
                    recent_requests = len([t for t in self.metrics.response_times if t < 60])
                    self.metrics.throughput_rps = recent_requests / 60.0
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance collector error: {e}")
                await asyncio.sleep(60)
    
    async def _health_monitor(self) -> None:
        """Monitor system health and trigger optimizations"""
        while self._monitoring_active:
            try:
                # Check if system is under stress
                if (self.metrics.cpu_percent > self._cpu_threshold or 
                    self.metrics.memory_usage_mb > 2048):  # 2GB threshold
                    
                    logger.warning("🔥 System under stress - initiating optimization")
                    await self._emergency_optimization()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _cache_optimizer(self) -> None:
        """Optimize cache performance"""
        while self._monitoring_active:
            try:
                cache_stats = self.cache.get_stats()
                
                # Log cache performance
                if cache_stats['hit_rate'] < 70:  # Less than 70% hit rate
                    logger.warning(f"Low cache hit rate: {cache_stats['hit_rate']:.1f}%")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Cache optimizer error: {e}")
                await asyncio.sleep(600)
    
    async def _emergency_optimization(self) -> None:
        """Emergency system optimization"""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clear old cache entries
            cache_stats = self.cache.get_stats()
            if cache_stats['size'] > 1000:
                await self.cache.clear()
            
            logger.info(f"🚨 Emergency optimization: GC collected {collected}, cache cleared")
            
        except Exception as e:
            logger.error(f"Emergency optimization failed: {e}")
    
    def create_connection_pool(self, name: str, create_connection: Callable, 
                              max_connections: int = 20) -> ConnectionPool:
        """Create named connection pool"""
        pool = ConnectionPool(create_connection, max_connections)
        self.connection_pools[name] = pool
        return pool
    
    def get_connection_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get connection pool by name"""
        return self.connection_pools.get(name)
    
    def create_circuit_breaker(self, name: str, failure_threshold: int = 5) -> CircuitBreaker:
        """Create named circuit breaker"""
        breaker = CircuitBreaker(failure_threshold)
        self.circuit_breakers[name] = breaker
        return breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        cache_stats = self.cache.get_stats()
        
        pool_stats = {}
        for name, pool in self.connection_pools.items():
            pool_stats[name] = pool.get_stats()
        
        breaker_stats = {}
        for name, breaker in self.circuit_breakers.items():
            breaker_stats[name] = breaker.get_state()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'average_response_time': self.metrics.average_response_time,
                'request_count': self.metrics.request_count,
                'error_rate': self.metrics.error_rate,
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'cpu_percent': self.metrics.cpu_percent,
                'throughput_rps': self.metrics.throughput_rps,
                'cache_hit_rate': self.metrics.cache_hit_rate
            },
            'cache_stats': cache_stats,
            'connection_pools': pool_stats,
            'circuit_breakers': breaker_stats,
            'system_health': {
                'memory_healthy': self.metrics.memory_usage_mb < 1536,  # < 1.5GB
                'cpu_healthy': self.metrics.cpu_percent < 80,
                'error_rate_healthy': self.metrics.error_rate < 1.0,
                'response_time_healthy': self.metrics.average_response_time < 0.1
            }
        }
    
    async def optimize_for_load(self, expected_rps: float) -> None:
        """Optimize system for expected load"""
        logger.info(f"🎯 Optimizing for expected load: {expected_rps} RPS")
        
        # Adjust cache size based on load
        new_cache_size = min(100000, int(expected_rps * 100))
        self.cache._max_size = new_cache_size
        
        # Pre-warm connections
        for pool in self.connection_pools.values():
            try:
                await pool.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize connection pool: {e}")
        
        # Trigger garbage collection
        gc.collect()
        
        logger.info(f"✅ System optimized for {expected_rps} RPS")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown performance engine"""
        logger.info("🔄 Shutting down performance engine...")
        
        self._monitoring_active = False
        
        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        # Close connection pools
        for pool in self.connection_pools.values():
            await pool.close_all()
        
        # Clear cache
        await self.cache.clear()
        
        logger.info("✅ Performance engine shutdown complete")


# Global performance engine instance
performance_engine = NetflixGradePerformanceEngine()
