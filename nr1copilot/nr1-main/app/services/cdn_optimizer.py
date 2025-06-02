
"""
Netflix-Grade CDN Optimization Service
Advanced caching, compression, and delivery optimization
"""

import asyncio
import gzip
import io
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from ..utils.cache import cache_manager
from .storage_service import storage_service

logger = logging.getLogger(__name__)

@dataclass
class CachePolicy:
    """CDN cache policy configuration"""
    max_age: int = 3600  # 1 hour
    stale_while_revalidate: int = 86400  # 1 day
    stale_if_error: int = 604800  # 1 week
    must_revalidate: bool = False
    public: bool = True
    vary_headers: List[str] = None

class CDNOptimizer:
    """Netflix-grade CDN optimization with intelligent caching"""

    def __init__(self):
        self.compression_enabled = True
        self.compression_threshold = 1024  # 1KB
        self.cache_policies = self._initialize_cache_policies()
        self.optimization_stats = {
            "requests_served": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "bytes_served": 0,
            "bytes_saved_compression": 0,
            "compression_ratio_avg": 0.0
        }

    def _initialize_cache_policies(self) -> Dict[str, CachePolicy]:
        """Initialize cache policies for different content types"""
        return {
            # Static assets - long cache
            "image/*": CachePolicy(
                max_age=2592000,  # 30 days
                stale_while_revalidate=86400,
                public=True
            ),
            "video/*": CachePolicy(
                max_age=604800,  # 7 days
                stale_while_revalidate=86400,
                public=True
            ),
            "audio/*": CachePolicy(
                max_age=604800,  # 7 days
                stale_while_revalidate=86400,
                public=True
            ),
            
            # Application files - medium cache
            "application/javascript": CachePolicy(
                max_age=86400,  # 1 day
                stale_while_revalidate=3600,
                public=True,
                vary_headers=["Accept-Encoding"]
            ),
            "text/css": CachePolicy(
                max_age=86400,  # 1 day
                stale_while_revalidate=3600,
                public=True,
                vary_headers=["Accept-Encoding"]
            ),
            
            # API responses - short cache
            "application/json": CachePolicy(
                max_age=300,  # 5 minutes
                stale_while_revalidate=3600,
                public=False,
                vary_headers=["Authorization", "Accept-Encoding"]
            ),
            
            # HTML - very short cache
            "text/html": CachePolicy(
                max_age=60,  # 1 minute
                stale_while_revalidate=300,
                public=False,
                vary_headers=["Accept-Encoding", "User-Agent"]
            ),
            
            # Default policy
            "default": CachePolicy(
                max_age=3600,  # 1 hour
                stale_while_revalidate=86400,
                public=True
            )
        }

    async def optimize_response(
        self,
        content: bytes,
        content_type: str,
        key: str,
        request_headers: Dict[str, str]
    ) -> Tuple[bytes, Dict[str, str]]:
        """Optimize response with compression and caching"""
        
        try:
            # Get cache policy
            policy = self._get_cache_policy(content_type)
            
            # Check if content should be compressed
            optimized_content = content
            response_headers = {}
            
            if self._should_compress(content, content_type, request_headers):
                optimized_content = await self._compress_content(content)
                response_headers["Content-Encoding"] = "gzip"
                response_headers["Vary"] = "Accept-Encoding"
                
                # Update compression stats
                compression_ratio = len(optimized_content) / len(content)
                self.optimization_stats["bytes_saved_compression"] += len(content) - len(optimized_content)
                
                # Update average compression ratio
                old_avg = self.optimization_stats["compression_ratio_avg"]
                requests = self.optimization_stats["requests_served"]
                self.optimization_stats["compression_ratio_avg"] = (
                    (old_avg * requests + compression_ratio) / (requests + 1)
                )
            
            # Set cache headers
            cache_headers = self._generate_cache_headers(policy)
            response_headers.update(cache_headers)
            
            # Set content headers
            response_headers["Content-Type"] = content_type
            response_headers["Content-Length"] = str(len(optimized_content))
            
            # Add performance headers
            response_headers.update(self._get_performance_headers())
            
            # Update stats
            self.optimization_stats["requests_served"] += 1
            self.optimization_stats["bytes_served"] += len(optimized_content)
            
            return optimized_content, response_headers
            
        except Exception as e:
            logger.error(f"Response optimization failed: {e}")
            return content, {"Content-Type": content_type}

    def _should_compress(
        self,
        content: bytes,
        content_type: str,
        request_headers: Dict[str, str]
    ) -> bool:
        """Determine if content should be compressed"""
        
        if not self.compression_enabled:
            return False
        
        # Check if client accepts gzip
        accept_encoding = request_headers.get("accept-encoding", "").lower()
        if "gzip" not in accept_encoding:
            return False
        
        # Check content size threshold
        if len(content) < self.compression_threshold:
            return False
        
        # Check if content type is compressible
        compressible_types = [
            "text/",
            "application/json",
            "application/javascript",
            "application/xml",
            "application/x-javascript"
        ]
        
        return any(content_type.startswith(t) for t in compressible_types)

    async def _compress_content(self, content: bytes) -> bytes:
        """Compress content using gzip"""
        
        try:
            # Use asyncio to avoid blocking
            loop = asyncio.get_event_loop()
            
            def compress():
                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode='wb', compresslevel=6) as gz:
                    gz.write(content)
                return buffer.getvalue()
            
            return await loop.run_in_executor(None, compress)
            
        except Exception as e:
            logger.error(f"Content compression failed: {e}")
            return content

    def _get_cache_policy(self, content_type: str) -> CachePolicy:
        """Get cache policy for content type"""
        
        # Try exact match first
        if content_type in self.cache_policies:
            return self.cache_policies[content_type]
        
        # Try wildcard match
        for pattern, policy in self.cache_policies.items():
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if content_type.startswith(prefix):
                    return policy
        
        # Return default policy
        return self.cache_policies["default"]

    def _generate_cache_headers(self, policy: CachePolicy) -> Dict[str, str]:
        """Generate cache control headers"""
        
        headers = {}
        
        # Cache-Control header
        cache_control_parts = []
        
        if policy.public:
            cache_control_parts.append("public")
        else:
            cache_control_parts.append("private")
        
        cache_control_parts.append(f"max-age={policy.max_age}")
        
        if policy.stale_while_revalidate:
            cache_control_parts.append(f"stale-while-revalidate={policy.stale_while_revalidate}")
        
        if policy.stale_if_error:
            cache_control_parts.append(f"stale-if-error={policy.stale_if_error}")
        
        if policy.must_revalidate:
            cache_control_parts.append("must-revalidate")
        
        headers["Cache-Control"] = ", ".join(cache_control_parts)
        
        # Expires header (fallback for older clients)
        expires_time = datetime.utcnow() + timedelta(seconds=policy.max_age)
        headers["Expires"] = expires_time.strftime("%a, %d %b %Y %H:%M:%S GMT")
        
        # ETag for validation
        headers["ETag"] = f'"{hash(str(policy))}"'
        
        # Vary header
        if policy.vary_headers:
            headers["Vary"] = ", ".join(policy.vary_headers)
        
        return headers

    def _get_performance_headers(self) -> Dict[str, str]:
        """Get performance-related headers"""
        
        return {
            "X-CDN-Optimized": "true",
            "X-Cache-Status": "optimized",
            "X-Compression-Enabled": "true" if self.compression_enabled else "false",
            "Server-Timing": f"cdn;dur=0.1, cache;dur=0.05"
        }

    async def get_cached_response(
        self,
        cache_key: str,
        etag: Optional[str] = None
    ) -> Optional[Tuple[bytes, Dict[str, str]]]:
        """Get cached response if available and valid"""
        
        try:
            # Check cache
            cached_data = cache_manager.get(cache_key, namespace="cdn")
            
            if cached_data:
                content, headers, cached_etag = cached_data
                
                # Check ETag for validation
                if etag and etag == cached_etag:
                    # Return 304 Not Modified
                    return b"", {"Status": "304"}
                
                self.optimization_stats["cache_hits"] += 1
                return content, headers
            
            self.optimization_stats["cache_misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None

    async def cache_response(
        self,
        cache_key: str,
        content: bytes,
        headers: Dict[str, str],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache optimized response"""
        
        try:
            # Extract ETag from headers
            etag = headers.get("ETag", "")
            
            # Cache the response
            cache_data = (content, headers, etag)
            
            return cache_manager.set(
                cache_key,
                cache_data,
                ttl=ttl,
                namespace="cdn",
                priority=2  # Medium priority
            )
            
        except Exception as e:
            logger.error(f"Response caching failed: {e}")
            return False

    def generate_cache_key(
        self,
        url: str,
        request_headers: Dict[str, str]
    ) -> str:
        """Generate cache key based on URL and relevant headers"""
        
        import hashlib
        
        # Base key from URL
        key_parts = [url]
        
        # Add relevant headers that affect response
        relevant_headers = ["accept-encoding", "user-agent", "authorization"]
        
        for header in relevant_headers:
            value = request_headers.get(header.lower())
            if value:
                key_parts.append(f"{header}:{value}")
        
        # Create hash
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def prefetch_content(self, keys: List[str]) -> Dict[str, bool]:
        """Prefetch content for better cache hit rates"""
        
        results = {}
        
        for key in keys:
            try:
                # Get content from storage
                storage_obj = await storage_service.retrieve_file(key)
                
                if storage_obj:
                    # Generate cache key
                    cache_key = f"prefetch:{key}"
                    
                    # Cache with long TTL
                    success = cache_manager.set(
                        cache_key,
                        storage_obj,
                        ttl=86400,  # 1 day
                        namespace="cdn",
                        priority=3  # High priority
                    )
                    
                    results[key] = success
                else:
                    results[key] = False
                    
            except Exception as e:
                logger.error(f"Prefetch failed for {key}: {e}")
                results[key] = False
        
        return results

    async def invalidate_cache(
        self,
        pattern: Optional[str] = None,
        keys: Optional[List[str]] = None
    ) -> int:
        """Invalidate cached content"""
        
        try:
            if keys:
                # Invalidate specific keys
                deleted = 0
                for key in keys:
                    if cache_manager.delete(key, namespace="cdn"):
                        deleted += 1
                return deleted
            
            elif pattern:
                # Clear by pattern (simplified - would need pattern matching)
                return cache_manager.clear(namespace="cdn")
            
            else:
                # Clear all CDN cache
                return cache_manager.clear(namespace="cdn")
            
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return 0

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get CDN optimization statistics"""
        
        cache_hit_rate = 0.0
        total_requests = self.optimization_stats["cache_hits"] + self.optimization_stats["cache_misses"]
        
        if total_requests > 0:
            cache_hit_rate = (self.optimization_stats["cache_hits"] / total_requests) * 100
        
        return {
            **self.optimization_stats,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "total_cache_requests": total_requests,
            "compression_enabled": self.compression_enabled,
            "compression_threshold_bytes": self.compression_threshold,
            "cache_policies_count": len(self.cache_policies)
        }


# Global CDN optimizer instance
cdn_optimizer = CDNOptimizer()

logger.info("✅ Netflix-grade CDN optimizer initialized")
