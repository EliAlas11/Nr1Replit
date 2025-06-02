
"""
Netflix-Level Database Abstraction Layer
Repository pattern with intelligent caching and performance optimization
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic
from uuid import UUID
import json
import time
import hashlib

from app.database.connection import db_manager
from app.database.models import (
    User, Video, Project, SocialPublication, Template, Analytics, SystemHealth
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RepositoryError(Exception):
    """Base repository exception"""
    pass


class NotFoundError(RepositoryError):
    """Entity not found error"""
    pass


class CacheManager:
    """Intelligent cache manager with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 300):
        self.cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU tracking"""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        
        # Check TTL
        if time.time() - entry["timestamp"] > entry["ttl"]:
            self._evict(key)
            return None
        
        # Update access time for LRU
        self.access_times[key] = time.time()
        return entry["data"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cache value with intelligent eviction"""
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            "data": value,
            "timestamp": time.time(),
            "ttl": ttl
        }
        self.access_times[key] = time.time()
    
    def _evict(self, key: str):
        """Evict specific key"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
            
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._evict(lru_key)
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": (len(self.cache) / self.max_size) * 100
        }


class BaseRepository(Generic[T], ABC):
    """Enhanced base repository with intelligent caching and performance optimization"""
    
    def __init__(self, table_name: str, model_class: type):
        self.table_name = table_name
        self.model_class = model_class
        self.cache_manager = CacheManager(max_size=5000, default_ttl=300)
        
        # Enhanced performance statistics
        self.stats = {
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_query_time": 0.0,
            "total_query_time": 0.0,
            "errors": 0,
            "created_entities": 0,
            "updated_entities": 0,
            "deleted_entities": 0
        }
        
        # Query optimization
        self.prepared_statements = {}
        self.batch_size = 1000
    
    def _cache_key(self, operation: str, *args) -> str:
        """Generate intelligent cache key with collision resistance"""
        key_parts = [self.table_name, operation] + [str(arg) for arg in args]
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _track_query_time(self, start_time: float):
        """Enhanced query performance tracking"""
        query_time = time.time() - start_time
        self.stats["queries"] += 1
        self.stats["total_query_time"] += query_time
        
        # Calculate rolling average
        self.stats["avg_query_time"] = self.stats["total_query_time"] / self.stats["queries"]
    
    def _track_cache_hit(self):
        """Track cache hit"""
        self.stats["cache_hits"] += 1
    
    def _track_cache_miss(self):
        """Track cache miss"""
        self.stats["cache_misses"] += 1
    
    async def create(self, data: Dict[str, Any]) -> T:
        """Create new entity with optimized performance"""
        start_time = time.time()
        
        try:
            # Prepare optimized insert query
            fields = list(data.keys())
            placeholders = [f"${i+1}" for i in range(len(fields))]
            values = list(data.values())
            
            query = f"""
                INSERT INTO {self.table_name} ({', '.join(fields)})
                VALUES ({', '.join(placeholders)})
                RETURNING *
            """
            
            row = await db_manager.execute_query(query, *values, fetch="one")
            
            if not row:
                raise RepositoryError("Failed to create entity")
            
            entity = self.model_class(**dict(row))
            
            # Intelligent cache invalidation
            self._invalidate_related_cache(str(row["id"]))
            
            # Cache the new entity
            cache_key = self._cache_key("id", str(row["id"]))
            self.cache_manager.set(cache_key, entity, ttl=600)  # Longer TTL for new entities
            
            self.stats["created_entities"] += 1
            return entity
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to create {self.table_name}: {e}")
            raise RepositoryError(f"Create failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def get_by_id(self, entity_id: Union[str, UUID]) -> Optional[T]:
        """Get entity by ID with intelligent caching"""
        cache_key = self._cache_key("id", str(entity_id))
        
        # Try cache first
        cached = self.cache_manager.get(cache_key)
        if cached:
            self._track_cache_hit()
            return cached
        
        self._track_cache_miss()
        start_time = time.time()
        
        try:
            query = f"SELECT * FROM {self.table_name} WHERE id = $1"
            row = await db_manager.execute_query(query, str(entity_id), fetch="one")
            
            if not row:
                return None
            
            entity = self.model_class(**dict(row))
            
            # Cache with adaptive TTL based on entity type
            ttl = self._calculate_adaptive_ttl(entity)
            self.cache_manager.set(cache_key, entity, ttl=ttl)
            
            return entity
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to get {self.table_name} by ID: {e}")
            raise RepositoryError(f"Get by ID failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def get_multiple_by_ids(self, entity_ids: List[Union[str, UUID]]) -> List[T]:
        """Efficiently get multiple entities by IDs"""
        if not entity_ids:
            return []
        
        entities = []
        uncached_ids = []
        
        # Check cache for each ID
        for entity_id in entity_ids:
            cache_key = self._cache_key("id", str(entity_id))
            cached = self.cache_manager.get(cache_key)
            if cached:
                entities.append(cached)
                self._track_cache_hit()
            else:
                uncached_ids.append(str(entity_id))
                self._track_cache_miss()
        
        # Batch fetch uncached entities
        if uncached_ids:
            start_time = time.time()
            try:
                placeholders = [f"${i+1}" for i in range(len(uncached_ids))]
                query = f"SELECT * FROM {self.table_name} WHERE id = ANY(ARRAY[{', '.join(placeholders)}])"
                
                rows = await db_manager.execute_query(query, *uncached_ids, fetch="all")
                
                for row in rows:
                    entity = self.model_class(**dict(row))
                    entities.append(entity)
                    
                    # Cache the entity
                    cache_key = self._cache_key("id", str(row["id"]))
                    ttl = self._calculate_adaptive_ttl(entity)
                    self.cache_manager.set(cache_key, entity, ttl=ttl)
                
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Failed to get multiple {self.table_name}: {e}")
                raise RepositoryError(f"Get multiple failed: {e}")
            finally:
                self._track_query_time(start_time)
        
        return entities
    
    async def update(self, entity_id: Union[str, UUID], data: Dict[str, Any]) -> Optional[T]:
        """Update entity with optimized performance"""
        start_time = time.time()
        
        try:
            # Add updated_at timestamp
            data["updated_at"] = datetime.utcnow()
            
            # Prepare optimized update query
            set_clauses = [f"{field} = ${i+2}" for i, field in enumerate(data.keys())]
            values = [str(entity_id)] + list(data.values())
            
            query = f"""
                UPDATE {self.table_name} 
                SET {', '.join(set_clauses)}
                WHERE id = $1
                RETURNING *
            """
            
            row = await db_manager.execute_query(query, *values, fetch="one")
            
            if not row:
                return None
            
            entity = self.model_class(**dict(row))
            
            # Update cache
            cache_key = self._cache_key("id", str(entity_id))
            ttl = self._calculate_adaptive_ttl(entity)
            self.cache_manager.set(cache_key, entity, ttl=ttl)
            
            # Invalidate related cache
            self._invalidate_related_cache(str(entity_id))
            
            self.stats["updated_entities"] += 1
            return entity
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to update {self.table_name}: {e}")
            raise RepositoryError(f"Update failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def delete(self, entity_id: Union[str, UUID]) -> bool:
        """Delete entity with cache cleanup"""
        start_time = time.time()
        
        try:
            query = f"DELETE FROM {self.table_name} WHERE id = $1"
            result = await db_manager.execute_query(query, str(entity_id))
            
            success = "DELETE 1" in str(result)
            
            if success:
                # Remove from cache
                cache_key = self._cache_key("id", str(entity_id))
                self.cache_manager._evict(cache_key)
                
                # Invalidate related cache
                self._invalidate_related_cache(str(entity_id))
                
                self.stats["deleted_entities"] += 1
            
            return success
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to delete {self.table_name}: {e}")
            raise RepositoryError(f"Delete failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def list_with_pagination(
        self, 
        limit: int = 50, 
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        order_by: str = "created_at DESC"
    ) -> Dict[str, Any]:
        """Optimized paginated listing with intelligent caching"""
        # Create cache key for this query
        cache_key = self._cache_key("list", limit, offset, str(filters), order_by)
        
        # Try cache first for read-heavy workloads
        cached = self.cache_manager.get(cache_key)
        if cached:
            self._track_cache_hit()
            return cached
        
        self._track_cache_miss()
        start_time = time.time()
        
        try:
            # Build optimized WHERE clause
            where_clauses = []
            params = []
            param_count = 0
            
            if filters:
                for field, value in filters.items():
                    param_count += 1
                    if isinstance(value, list):
                        placeholders = [f"${param_count + i}" for i in range(len(value))]
                        where_clauses.append(f"{field} = ANY(ARRAY[{', '.join(placeholders)}])")
                        params.extend(value)
                        param_count += len(value) - 1
                    else:
                        where_clauses.append(f"{field} = ${param_count}")
                        params.append(value)
            
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            
            # Optimized count query
            count_query = f"SELECT COUNT(*) FROM {self.table_name} {where_sql}"
            total = await db_manager.execute_query(count_query, *params, fetch="val")
            
            # Optimized items query
            params.extend([limit, offset])
            items_query = f"""
                SELECT * FROM {self.table_name} {where_sql}
                ORDER BY {order_by}
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            
            rows = await db_manager.execute_query(items_query, *params, fetch="all")
            items = [self.model_class(**dict(row)) for row in rows]
            
            result = {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total,
                "page": (offset // limit) + 1,
                "total_pages": (total + limit - 1) // limit
            }
            
            # Cache result with shorter TTL for list queries
            self.cache_manager.set(cache_key, result, ttl=60)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to list {self.table_name}: {e}")
            raise RepositoryError(f"List failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def bulk_create(self, entities_data: List[Dict[str, Any]]) -> List[T]:
        """Bulk create entities with optimized performance"""
        if not entities_data:
            return []
        
        start_time = time.time()
        
        try:
            # Prepare bulk insert
            fields = list(entities_data[0].keys())
            entities = []
            
            # Process in batches
            for i in range(0, len(entities_data), self.batch_size):
                batch = entities_data[i:i + self.batch_size]
                
                # Build VALUES clause for batch
                values_clauses = []
                all_values = []
                
                for j, entity_data in enumerate(batch):
                    start_idx = j * len(fields)
                    placeholders = [f"${start_idx + k + 1}" for k in range(len(fields))]
                    values_clauses.append(f"({', '.join(placeholders)})")
                    all_values.extend([entity_data[field] for field in fields])
                
                query = f"""
                    INSERT INTO {self.table_name} ({', '.join(fields)})
                    VALUES {', '.join(values_clauses)}
                    RETURNING *
                """
                
                rows = await db_manager.execute_query(query, *all_values, fetch="all")
                
                for row in rows:
                    entity = self.model_class(**dict(row))
                    entities.append(entity)
                    
                    # Cache new entities
                    cache_key = self._cache_key("id", str(row["id"]))
                    ttl = self._calculate_adaptive_ttl(entity)
                    self.cache_manager.set(cache_key, entity, ttl=ttl)
            
            self.stats["created_entities"] += len(entities)
            return entities
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to bulk create {self.table_name}: {e}")
            raise RepositoryError(f"Bulk create failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    def _calculate_adaptive_ttl(self, entity: Any) -> int:
        """Calculate adaptive TTL based on entity characteristics"""
        # Base TTL
        base_ttl = 300  # 5 minutes
        
        # Adjust based on entity type and update frequency
        if hasattr(entity, 'updated_at'):
            # More recently updated entities get shorter TTL
            update_age = (datetime.utcnow() - entity.updated_at).total_seconds()
            if update_age < 3600:  # Updated in last hour
                return base_ttl // 2
            elif update_age > 86400:  # Not updated in 24 hours
                return base_ttl * 2
        
        return base_ttl
    
    def _invalidate_related_cache(self, entity_id: str):
        """Intelligently invalidate related cache entries"""
        # Clear list caches that might contain this entity
        keys_to_remove = []
        for key in self.cache_manager.cache.keys():
            if "list" in key or entity_id in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.cache_manager._evict(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive repository performance statistics"""
        cache_hit_rate = 0.0
        total_cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_cache_requests > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_cache_requests
        
        return {
            "table": self.table_name,
            "queries": self.stats["queries"],
            "avg_query_time_ms": round(self.stats["avg_query_time"] * 1000, 2),
            "cache_hit_rate": round(cache_hit_rate, 3),
            "cache_stats": self.cache_manager.get_stats(),
            "entities_created": self.stats["created_entities"],
            "entities_updated": self.stats["updated_entities"],
            "entities_deleted": self.stats["deleted_entities"],
            "errors": self.stats["errors"],
            "error_rate": round(self.stats["errors"] / max(1, self.stats["queries"]), 4)
        }
    
    def clear_cache(self):
        """Clear repository cache"""
        self.cache_manager.clear()


class UserRepository(BaseRepository[User]):
    """Enhanced user repository with specialized optimizations"""
    
    def __init__(self):
        super().__init__("users", User)
        # User-specific cache with longer TTL
        self.cache_manager = CacheManager(max_size=10000, default_ttl=600)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email with optimized caching"""
        cache_key = self._cache_key("email", email)
        
        cached = self.cache_manager.get(cache_key)
        if cached:
            self._track_cache_hit()
            return cached
        
        self._track_cache_miss()
        start_time = time.time()
        
        try:
            query = "SELECT * FROM users WHERE email = $1"
            row = await db_manager.execute_query(query, email, fetch="one")
            
            if not row:
                return None
            
            user = User(**dict(row))
            
            # Cache with multiple keys
            self.cache_manager.set(cache_key, user, ttl=600)
            id_cache_key = self._cache_key("id", str(user.id))
            self.cache_manager.set(id_cache_key, user, ttl=600)
            
            return user
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to get user by email: {e}")
            raise RepositoryError(f"Get by email failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username with optimized caching"""
        cache_key = self._cache_key("username", username)
        
        cached = self.cache_manager.get(cache_key)
        if cached:
            self._track_cache_hit()
            return cached
        
        self._track_cache_miss()
        start_time = time.time()
        
        try:
            query = "SELECT * FROM users WHERE username = $1"
            row = await db_manager.execute_query(query, username, fetch="one")
            
            if not row:
                return None
            
            user = User(**dict(row))
            
            # Cache with multiple keys
            self.cache_manager.set(cache_key, user, ttl=600)
            id_cache_key = self._cache_key("id", str(user.id))
            self.cache_manager.set(id_cache_key, user, ttl=600)
            
            return user
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to get user by username: {e}")
            raise RepositoryError(f"Get by username failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def update_last_login(self, user_id: UUID) -> bool:
        """Update user's last login with minimal cache impact"""
        try:
            query = """
                UPDATE users 
                SET last_login = NOW(), updated_at = NOW()
                WHERE id = $1
            """
            await db_manager.execute_query(query, str(user_id))
            
            # Selectively update cache instead of clearing
            cache_key = self._cache_key("id", str(user_id))
            cached_user = self.cache_manager.get(cache_key)
            if cached_user:
                cached_user.last_login = datetime.utcnow()
                self.cache_manager.set(cache_key, cached_user, ttl=600)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            return False


class VideoRepository(BaseRepository[Video]):
    """Enhanced video repository with specialized optimizations"""
    
    def __init__(self):
        super().__init__("videos", Video)
        # Video-specific optimizations
        self.cache_manager = CacheManager(max_size=20000, default_ttl=180)  # Shorter TTL for frequently changing data
    
    async def get_by_user(self, user_id: UUID, status: Optional[str] = None) -> List[Video]:
        """Get videos by user with intelligent caching"""
        cache_key = self._cache_key("user_videos", str(user_id), status or "all")
        
        cached = self.cache_manager.get(cache_key)
        if cached:
            self._track_cache_hit()
            return cached
        
        self._track_cache_miss()
        start_time = time.time()
        
        try:
            if status:
                query = """
                    SELECT * FROM videos 
                    WHERE user_id = $1 AND status = $2
                    ORDER BY created_at DESC
                """
                rows = await db_manager.execute_query(query, str(user_id), status, fetch="all")
            else:
                query = """
                    SELECT * FROM videos 
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                """
                rows = await db_manager.execute_query(query, str(user_id), fetch="all")
            
            videos = [Video(**dict(row)) for row in rows]
            
            # Cache with shorter TTL for user-specific queries
            self.cache_manager.set(cache_key, videos, ttl=120)
            
            return videos
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to get videos by user: {e}")
            raise RepositoryError(f"Get videos by user failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def update_processing_progress(self, video_id: UUID, progress: float) -> bool:
        """Update video processing progress with minimal cache impact"""
        try:
            query = """
                UPDATE videos 
                SET processing_progress = $2, updated_at = NOW()
                WHERE id = $1
            """
            await db_manager.execute_query(query, str(video_id), progress)
            
            # Update cache if exists
            cache_key = self._cache_key("id", str(video_id))
            cached_video = self.cache_manager.get(cache_key)
            if cached_video:
                cached_video.processing_progress = progress
                cached_video.updated_at = datetime.utcnow()
                self.cache_manager.set(cache_key, cached_video, ttl=60)  # Short TTL for frequently updated data
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update processing progress: {e}")
            return False
    
    async def get_viral_candidates(self, limit: int = 10) -> List[Video]:
        """Get videos with high viral potential with caching"""
        cache_key = self._cache_key("viral_candidates", limit)
        
        cached = self.cache_manager.get(cache_key)
        if cached:
            self._track_cache_hit()
            return cached
        
        self._track_cache_miss()
        start_time = time.time()
        
        try:
            query = """
                SELECT * FROM videos 
                WHERE viral_score > 0.7 AND status = 'ready'
                ORDER BY viral_score DESC, created_at DESC
                LIMIT $1
            """
            rows = await db_manager.execute_query(query, limit, fetch="all")
            videos = [Video(**dict(row)) for row in rows]
            
            # Cache with moderate TTL
            self.cache_manager.set(cache_key, videos, ttl=300)
            
            return videos
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to get viral candidates: {e}")
            raise RepositoryError(f"Get viral candidates failed: {e}")
        finally:
            self._track_query_time(start_time)


class ProjectRepository(BaseRepository[Project]):
    """Enhanced project repository"""
    
    def __init__(self):
        super().__init__("projects", Project)
    
    async def get_by_user(self, user_id: UUID, status: Optional[str] = None) -> List[Project]:
        """Get projects by user with caching"""
        cache_key = self._cache_key("user_projects", str(user_id), status or "all")
        
        cached = self.cache_manager.get(cache_key)
        if cached:
            self._track_cache_hit()
            return cached
        
        self._track_cache_miss()
        start_time = time.time()
        
        try:
            if status:
                query = """
                    SELECT * FROM projects 
                    WHERE user_id = $1 AND status = $2
                    ORDER BY updated_at DESC
                """
                rows = await db_manager.execute_query(query, str(user_id), status, fetch="all")
            else:
                query = """
                    SELECT * FROM projects 
                    WHERE user_id = $1
                    ORDER BY updated_at DESC
                """
                rows = await db_manager.execute_query(query, str(user_id), fetch="all")
            
            projects = [Project(**dict(row)) for row in rows]
            self.cache_manager.set(cache_key, projects, ttl=180)
            
            return projects
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to get projects by user: {e}")
            raise RepositoryError(f"Get projects by user failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def update_last_edited(self, project_id: UUID) -> bool:
        """Update project's last edited timestamp"""
        try:
            query = """
                UPDATE projects 
                SET last_edited = NOW(), updated_at = NOW()
                WHERE id = $1
            """
            await db_manager.execute_query(query, str(project_id))
            
            # Update cache
            cache_key = self._cache_key("id", str(project_id))
            cached_project = self.cache_manager.get(cache_key)
            if cached_project:
                cached_project.last_edited = datetime.utcnow()
                cached_project.updated_at = datetime.utcnow()
                self.cache_manager.set(cache_key, cached_project)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update last edited: {e}")
            return False


class AnalyticsRepository(BaseRepository[Analytics]):
    """Enhanced analytics repository with time-series optimizations"""
    
    def __init__(self):
        super().__init__("analytics", Analytics)
        # Analytics-specific optimizations
        self.cache_manager = CacheManager(max_size=50000, default_ttl=60)  # Large cache, short TTL
    
    async def record_event(self, event_data: Dict[str, Any]) -> Analytics:
        """Record analytics event with batch optimization"""
        return await self.create(event_data)
    
    async def get_user_activity(
        self, 
        user_id: UUID, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Analytics]:
        """Get user activity with time-range caching"""
        cache_key = self._cache_key("user_activity", str(user_id), start_date.isoformat(), end_date.isoformat())
        
        cached = self.cache_manager.get(cache_key)
        if cached:
            self._track_cache_hit()
            return cached
        
        self._track_cache_miss()
        start_time = time.time()
        
        try:
            query = """
                SELECT * FROM analytics 
                WHERE user_id = $1 AND created_at BETWEEN $2 AND $3
                ORDER BY created_at DESC
            """
            rows = await db_manager.execute_query(
                query, str(user_id), start_date, end_date, fetch="all"
            )
            analytics = [Analytics(**dict(row)) for row in rows]
            
            # Cache with very short TTL for analytics
            self.cache_manager.set(cache_key, analytics, ttl=30)
            
            return analytics
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to get user activity: {e}")
            raise RepositoryError(f"Get user activity failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def get_event_summary(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, int]:
        """Get event summary with intelligent caching"""
        cache_key = self._cache_key("event_summary", start_date.isoformat(), end_date.isoformat())
        
        cached = self.cache_manager.get(cache_key)
        if cached:
            self._track_cache_hit()
            return cached
        
        self._track_cache_miss()
        start_time = time.time()
        
        try:
            query = """
                SELECT event_type, COUNT(*) as count
                FROM analytics 
                WHERE created_at BETWEEN $1 AND $2
                GROUP BY event_type
                ORDER BY count DESC
            """
            rows = await db_manager.execute_query(query, start_date, end_date, fetch="all")
            summary = {row["event_type"]: row["count"] for row in rows}
            
            # Cache summary data longer
            self.cache_manager.set(cache_key, summary, ttl=300)
            
            return summary
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to get event summary: {e}")
            raise RepositoryError(f"Get event summary failed: {e}")
        finally:
            self._track_query_time(start_time)


class RepositoryContainer:
    """Enhanced repository container with global optimization"""
    
    def __init__(self):
        self.users = UserRepository()
        self.videos = VideoRepository()
        self.projects = ProjectRepository()
        self.social_publications = BaseRepository("social_publications", SocialPublication)
        self.templates = BaseRepository("templates", Template)
        self.analytics = AnalyticsRepository()
        self.system_health = BaseRepository("system_health", SystemHealth)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance stats for all repositories"""
        all_stats = {
            "users": self.users.get_stats(),
            "videos": self.videos.get_stats(),
            "projects": self.projects.get_stats(),
            "social_publications": self.social_publications.get_stats(),
            "templates": self.templates.get_stats(),
            "analytics": self.analytics.get_stats(),
            "system_health": self.system_health.get_stats()
        }
        
        # Calculate global metrics
        total_queries = sum(stat["queries"] for stat in all_stats.values())
        total_cache_hits = sum(stat.get("cache_stats", {}).get("size", 0) for stat in all_stats.values())
        avg_query_time = sum(stat["avg_query_time_ms"] for stat in all_stats.values()) / len(all_stats)
        
        all_stats["global_summary"] = {
            "total_queries": total_queries,
            "total_cache_entries": total_cache_hits,
            "avg_query_time_ms": round(avg_query_time, 2),
            "performance_grade": self._calculate_global_performance_grade(avg_query_time)
        }
        
        return all_stats
    
    def _calculate_global_performance_grade(self, avg_query_time: float) -> str:
        """Calculate global performance grade"""
        if avg_query_time < 10:
            return "A+ EXCEPTIONAL"
        elif avg_query_time < 25:
            return "A EXCELLENT"
        elif avg_query_time < 50:
            return "B GOOD"
        elif avg_query_time < 100:
            return "C AVERAGE"
        else:
            return "D NEEDS_OPTIMIZATION"
    
    def clear_all_caches(self):
        """Clear all repository caches"""
        for repo in [self.users, self.videos, self.projects, self.social_publications, 
                     self.templates, self.analytics, self.system_health]:
            repo.clear_cache()


# Global repository container
repositories = RepositoryContainer()
