
"""
Netflix-Level Database Abstraction Layer
Repository pattern with caching and performance optimization
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic
from uuid import UUID
import json
import time

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


class BaseRepository(Generic[T], ABC):
    """Base repository with common database operations"""
    
    def __init__(self, table_name: str, model_class: type):
        self.table_name = table_name
        self.model_class = model_class
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        self.stats = {
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_query_time": 0.0
        }
    
    def _cache_key(self, *args) -> str:
        """Generate cache key"""
        return f"{self.table_name}:{':'.join(str(arg) for arg in args)}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - cache_entry["timestamp"] < self.cache_ttl
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if valid"""
        if key in self.cache and self._is_cache_valid(self.cache[key]):
            self.stats["cache_hits"] += 1
            return self.cache[key]["data"]
        self.stats["cache_misses"] += 1
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Set cache entry"""
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
    
    def _track_query_time(self, start_time: float):
        """Track query performance"""
        query_time = time.time() - start_time
        self.stats["queries"] += 1
        
        # Calculate rolling average
        current_avg = self.stats["avg_query_time"]
        query_count = self.stats["queries"]
        self.stats["avg_query_time"] = ((current_avg * (query_count - 1)) + query_time) / query_count
    
    async def create(self, data: Dict[str, Any]) -> T:
        """Create new entity"""
        start_time = time.time()
        
        try:
            # Prepare fields and values
            fields = list(data.keys())
            placeholders = [f"${i+1}" for i in range(len(fields))]
            values = list(data.values())
            
            query = f"""
                INSERT INTO {self.table_name} ({', '.join(fields)})
                VALUES ({', '.join(placeholders)})
                RETURNING *
            """
            
            row = await db_manager.execute_query(query, *values, fetch="one")
            
            # Clear related cache
            self._clear_related_cache(str(row["id"]))
            
            return self.model_class(**dict(row))
            
        except Exception as e:
            logger.error(f"Failed to create {self.table_name}: {e}")
            raise RepositoryError(f"Create failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def get_by_id(self, entity_id: Union[str, UUID]) -> Optional[T]:
        """Get entity by ID with caching"""
        cache_key = self._cache_key("id", str(entity_id))
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        start_time = time.time()
        
        try:
            query = f"SELECT * FROM {self.table_name} WHERE id = $1"
            row = await db_manager.execute_query(query, str(entity_id), fetch="one")
            
            if not row:
                return None
            
            entity = self.model_class(**dict(row))
            self._set_cache(cache_key, entity)
            return entity
            
        except Exception as e:
            logger.error(f"Failed to get {self.table_name} by ID: {e}")
            raise RepositoryError(f"Get by ID failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def update(self, entity_id: Union[str, UUID], data: Dict[str, Any]) -> Optional[T]:
        """Update entity"""
        start_time = time.time()
        
        try:
            # Prepare SET clause
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
            
            # Clear cache
            self._clear_related_cache(str(entity_id))
            
            return self.model_class(**dict(row))
            
        except Exception as e:
            logger.error(f"Failed to update {self.table_name}: {e}")
            raise RepositoryError(f"Update failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    async def delete(self, entity_id: Union[str, UUID]) -> bool:
        """Delete entity"""
        start_time = time.time()
        
        try:
            query = f"DELETE FROM {self.table_name} WHERE id = $1"
            result = await db_manager.execute_query(query, str(entity_id))
            
            # Clear cache
            self._clear_related_cache(str(entity_id))
            
            return "DELETE 1" in str(result)
            
        except Exception as e:
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
        """List entities with pagination and filtering"""
        start_time = time.time()
        
        try:
            # Build WHERE clause
            where_clauses = []
            params = []
            param_count = 0
            
            if filters:
                for field, value in filters.items():
                    param_count += 1
                    where_clauses.append(f"{field} = ${param_count}")
                    params.append(value)
            
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            
            # Count total
            count_query = f"SELECT COUNT(*) FROM {self.table_name} {where_sql}"
            total = await db_manager.execute_query(count_query, *params, fetch="val")
            
            # Get items
            params.extend([limit, offset])
            items_query = f"""
                SELECT * FROM {self.table_name} {where_sql}
                ORDER BY {order_by}
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            
            rows = await db_manager.execute_query(items_query, *params, fetch="all")
            items = [self.model_class(**dict(row)) for row in rows]
            
            return {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
            
        except Exception as e:
            logger.error(f"Failed to list {self.table_name}: {e}")
            raise RepositoryError(f"List failed: {e}")
        finally:
            self._track_query_time(start_time)
    
    def _clear_related_cache(self, entity_id: str):
        """Clear cache entries related to an entity"""
        keys_to_remove = []
        for key in self.cache.keys():
            if entity_id in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get repository performance statistics"""
        cache_hit_rate = 0.0
        total_cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_cache_requests > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_cache_requests
        
        return {
            "table": self.table_name,
            "queries": self.stats["queries"],
            "avg_query_time_ms": round(self.stats["avg_query_time"] * 1000, 2),
            "cache_hit_rate": round(cache_hit_rate, 3),
            "cache_entries": len(self.cache)
        }


class UserRepository(BaseRepository[User]):
    """User repository with specialized methods"""
    
    def __init__(self):
        super().__init__("users", User)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        cache_key = self._cache_key("email", email)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            query = "SELECT * FROM users WHERE email = $1"
            row = await db_manager.execute_query(query, email, fetch="one")
            
            if not row:
                return None
            
            user = User(**dict(row))
            self._set_cache(cache_key, user)
            return user
            
        except Exception as e:
            logger.error(f"Failed to get user by email: {e}")
            raise RepositoryError(f"Get by email failed: {e}")
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        cache_key = self._cache_key("username", username)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            query = "SELECT * FROM users WHERE username = $1"
            row = await db_manager.execute_query(query, username, fetch="one")
            
            if not row:
                return None
            
            user = User(**dict(row))
            self._set_cache(cache_key, user)
            return user
            
        except Exception as e:
            logger.error(f"Failed to get user by username: {e}")
            raise RepositoryError(f"Get by username failed: {e}")
    
    async def update_last_login(self, user_id: UUID) -> bool:
        """Update user's last login timestamp"""
        try:
            query = """
                UPDATE users 
                SET last_login = NOW(), updated_at = NOW()
                WHERE id = $1
            """
            await db_manager.execute_query(query, str(user_id))
            
            # Clear cache
            self._clear_related_cache(str(user_id))
            return True
            
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            return False


class VideoRepository(BaseRepository[Video]):
    """Video repository with specialized methods"""
    
    def __init__(self):
        super().__init__("videos", Video)
    
    async def get_by_user(self, user_id: UUID, status: Optional[str] = None) -> List[Video]:
        """Get videos by user with optional status filter"""
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
            
            return [Video(**dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get videos by user: {e}")
            raise RepositoryError(f"Get videos by user failed: {e}")
    
    async def update_processing_progress(self, video_id: UUID, progress: float) -> bool:
        """Update video processing progress"""
        try:
            query = """
                UPDATE videos 
                SET processing_progress = $2, updated_at = NOW()
                WHERE id = $1
            """
            await db_manager.execute_query(query, str(video_id), progress)
            
            # Clear cache
            self._clear_related_cache(str(video_id))
            return True
            
        except Exception as e:
            logger.error(f"Failed to update processing progress: {e}")
            return False
    
    async def get_viral_candidates(self, limit: int = 10) -> List[Video]:
        """Get videos with high viral potential"""
        try:
            query = """
                SELECT * FROM videos 
                WHERE viral_score > 0.7 AND status = 'ready'
                ORDER BY viral_score DESC, created_at DESC
                LIMIT $1
            """
            rows = await db_manager.execute_query(query, limit, fetch="all")
            return [Video(**dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get viral candidates: {e}")
            raise RepositoryError(f"Get viral candidates failed: {e}")


class ProjectRepository(BaseRepository[Project]):
    """Project repository with specialized methods"""
    
    def __init__(self):
        super().__init__("projects", Project)
    
    async def get_by_user(self, user_id: UUID, status: Optional[str] = None) -> List[Project]:
        """Get projects by user with optional status filter"""
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
            
            return [Project(**dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get projects by user: {e}")
            raise RepositoryError(f"Get projects by user failed: {e}")
    
    async def update_last_edited(self, project_id: UUID) -> bool:
        """Update project's last edited timestamp"""
        try:
            query = """
                UPDATE projects 
                SET last_edited = NOW(), updated_at = NOW()
                WHERE id = $1
            """
            await db_manager.execute_query(query, str(project_id))
            
            # Clear cache
            self._clear_related_cache(str(project_id))
            return True
            
        except Exception as e:
            logger.error(f"Failed to update last edited: {e}")
            return False


class AnalyticsRepository(BaseRepository[Analytics]):
    """Analytics repository with specialized methods"""
    
    def __init__(self):
        super().__init__("analytics", Analytics)
    
    async def record_event(self, event_data: Dict[str, Any]) -> Analytics:
        """Record analytics event"""
        return await self.create(event_data)
    
    async def get_user_activity(
        self, 
        user_id: UUID, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Analytics]:
        """Get user activity within date range"""
        try:
            query = """
                SELECT * FROM analytics 
                WHERE user_id = $1 AND created_at BETWEEN $2 AND $3
                ORDER BY created_at DESC
            """
            rows = await db_manager.execute_query(
                query, str(user_id), start_date, end_date, fetch="all"
            )
            return [Analytics(**dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get user activity: {e}")
            raise RepositoryError(f"Get user activity failed: {e}")
    
    async def get_event_summary(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, int]:
        """Get event summary for date range"""
        try:
            query = """
                SELECT event_type, COUNT(*) as count
                FROM analytics 
                WHERE created_at BETWEEN $1 AND $2
                GROUP BY event_type
                ORDER BY count DESC
            """
            rows = await db_manager.execute_query(query, start_date, end_date, fetch="all")
            return {row["event_type"]: row["count"] for row in rows}
            
        except Exception as e:
            logger.error(f"Failed to get event summary: {e}")
            raise RepositoryError(f"Get event summary failed: {e}")


# Repository container for dependency injection
class RepositoryContainer:
    """Container for all repositories"""
    
    def __init__(self):
        self.users = UserRepository()
        self.videos = VideoRepository()
        self.projects = ProjectRepository()
        self.social_publications = BaseRepository("social_publications", SocialPublication)
        self.templates = BaseRepository("templates", Template)
        self.analytics = AnalyticsRepository()
        self.system_health = BaseRepository("system_health", SystemHealth)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get performance stats for all repositories"""
        return {
            "users": self.users.get_stats(),
            "videos": self.videos.get_stats(),
            "projects": self.projects.get_stats(),
            "social_publications": self.social_publications.get_stats(),
            "templates": self.templates.get_stats(),
            "analytics": self.analytics.get_stats(),
            "system_health": self.system_health.get_stats()
        }


# Global repository container
repositories = RepositoryContainer()
