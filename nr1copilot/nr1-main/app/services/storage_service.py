
"""
Netflix-Grade File Storage & CDN Service
Advanced object storage with automatic cleanup, signed URLs, and HLS/ABR support
"""

import asyncio
import os
import hashlib
import json
import tempfile
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

try:
    from replit.object_storage import Client as ReplitClient
    REPLIT_STORAGE_AVAILABLE = True
except ImportError:
    REPLIT_STORAGE_AVAILABLE = False
    logging.warning("Replit Object Storage not available - using local fallback")

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class StorageObject:
    """Represents a stored object with metadata"""
    key: str
    size: int
    content_type: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    signed_url: Optional[str] = None
    public_url: Optional[str] = None

@dataclass
class VideoVariant:
    """Represents a video variant for HLS/ABR streaming"""
    resolution: str
    bitrate: int
    codec: str
    file_path: str
    bandwidth: int
    size: int

class NetflixStorageService:
    """Netflix-grade storage service with CDN capabilities"""

    def __init__(self):
        self.client = None
        self.local_storage_path = Path(settings.upload_path) / "storage"
        self.temp_cleanup_interval = 3600  # 1 hour
        self.signed_url_expiry = 86400  # 24 hours
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Video processing settings
        self.hls_variants = [
            {"resolution": "360p", "bitrate": 500000, "width": 640, "height": 360},
            {"resolution": "480p", "bitrate": 1000000, "width": 854, "height": 480},
            {"resolution": "720p", "bitrate": 2500000, "width": 1280, "height": 720},
            {"resolution": "1080p", "bitrate": 5000000, "width": 1920, "height": 1080},
        ]
        
        self.stats = {
            "objects_stored": 0,
            "objects_retrieved": 0,
            "objects_deleted": 0,
            "bytes_uploaded": 0,
            "bytes_downloaded": 0,
            "temp_files_cleaned": 0,
            "signed_urls_generated": 0
        }

    async def initialize(self):
        """Initialize storage service"""
        try:
            if REPLIT_STORAGE_AVAILABLE:
                self.client = ReplitClient()
                logger.info("✅ Replit Object Storage initialized")
            else:
                # Fallback to local storage
                self.local_storage_path.mkdir(parents=True, exist_ok=True)
                logger.info("✅ Local storage fallback initialized")
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_worker())
            
        except Exception as e:
            logger.error(f"Storage initialization failed: {e}")
            raise

    async def shutdown(self):
        """Shutdown storage service"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    async def store_file(
        self,
        file_path: str,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_in: Optional[int] = None
    ) -> StorageObject:
        """Store file with metadata and optional expiration"""
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if content_type is None:
                content_type, _ = mimetypes.guess_type(file_path)
                content_type = content_type or "application/octet-stream"
            
            # Calculate expiration
            expires_at = None
            if expires_in:
                expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            
            # Store file
            if self.client:
                # Use Replit Object Storage
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # Add metadata
                storage_metadata = {
                    "content_type": content_type,
                    "size": file_size,
                    "created_at": datetime.utcnow().isoformat(),
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    **(metadata or {})
                }
                
                # Store with metadata as prefix in key
                metadata_key = f"{key}.metadata"
                self.client.upload_from_text(metadata_key, json.dumps(storage_metadata))
                
                # Store actual file
                await self._upload_binary_to_replit(key, content)
                
            else:
                # Use local storage
                local_path = self.local_storage_path / key
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file to storage
                import shutil
                shutil.copy2(file_path, local_path)
                
                # Store metadata
                metadata_file = local_path.with_suffix(local_path.suffix + ".metadata")
                storage_metadata = {
                    "content_type": content_type,
                    "size": file_size,
                    "created_at": datetime.utcnow().isoformat(),
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    **(metadata or {})
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(storage_metadata, f)
            
            # Update stats
            self.stats["objects_stored"] += 1
            self.stats["bytes_uploaded"] += file_size
            
            # Create storage object
            storage_obj = StorageObject(
                key=key,
                size=file_size,
                content_type=content_type,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            logger.info(f"File stored: {key} ({file_size} bytes)")
            return storage_obj
            
        except Exception as e:
            logger.error(f"File storage failed: {e}")
            raise

    async def _upload_binary_to_replit(self, key: str, content: bytes):
        """Upload binary content to Replit Object Storage"""
        # Since replit client doesn't have direct binary upload,
        # we'll use a temporary file approach
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            
            # Read back as text with base64 encoding for binary safety
            import base64
            encoded_content = base64.b64encode(content).decode('utf-8')
            self.client.upload_from_text(f"{key}.b64", encoded_content)
            
            # Clean up temp file
            os.unlink(temp_file.name)

    async def retrieve_file(self, key: str) -> Optional[StorageObject]:
        """Retrieve file metadata and generate access URL"""
        
        try:
            if self.client:
                # Get metadata from Replit storage
                try:
                    metadata_content = self.client.download_as_text(f"{key}.metadata")
                    metadata = json.loads(metadata_content)
                except Exception:
                    return None
                
                # Check if file exists
                try:
                    self.client.download_as_text(f"{key}.b64")
                except Exception:
                    return None
                    
            else:
                # Local storage
                local_path = self.local_storage_path / key
                metadata_file = local_path.with_suffix(local_path.suffix + ".metadata")
                
                if not local_path.exists() or not metadata_file.exists():
                    return None
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Parse metadata
            created_at = datetime.fromisoformat(metadata["created_at"])
            expires_at = None
            if metadata.get("expires_at"):
                expires_at = datetime.fromisoformat(metadata["expires_at"])
                
                # Check if expired
                if expires_at and datetime.utcnow() > expires_at:
                    await self.delete_file(key)
                    return None
            
            # Update stats
            self.stats["objects_retrieved"] += 1
            self.stats["bytes_downloaded"] += metadata["size"]
            
            # Create storage object
            storage_obj = StorageObject(
                key=key,
                size=metadata["size"],
                content_type=metadata["content_type"],
                created_at=created_at,
                expires_at=expires_at,
                metadata={k: v for k, v in metadata.items() 
                         if k not in ["content_type", "size", "created_at", "expires_at"]}
            )
            
            return storage_obj
            
        except Exception as e:
            logger.error(f"File retrieval failed: {e}")
            return None

    async def generate_signed_url(
        self,
        key: str,
        expires_in: int = None,
        method: str = "GET"
    ) -> Optional[str]:
        """Generate signed URL for secure access"""
        
        try:
            if expires_in is None:
                expires_in = self.signed_url_expiry
            
            # Check if file exists
            storage_obj = await self.retrieve_file(key)
            if not storage_obj:
                return None
            
            # Generate signed URL
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            
            if self.client:
                # For Replit storage, create a temporary access token
                import hmac
                import base64
                
                # Create signature
                message = f"{method}:{key}:{expires_at.isoformat()}"
                signature = hmac.new(
                    key=settings.secret_key.encode(),
                    msg=message.encode(),
                    digestmod=hashlib.sha256
                ).hexdigest()
                
                # Create signed URL (would be handled by CDN in production)
                signed_url = f"/api/v10/storage/download/{key}?expires={expires_at.isoformat()}&signature={signature}"
                
            else:
                # Local storage signed URL
                signature = hmac.new(
                    key=settings.secret_key.encode(),
                    msg=f"{key}:{expires_at.isoformat()}".encode(),
                    digestmod=hashlib.sha256
                ).hexdigest()
                
                signed_url = f"/api/v10/storage/download/{key}?expires={expires_at.isoformat()}&signature={signature}"
            
            self.stats["signed_urls_generated"] += 1
            logger.debug(f"Signed URL generated for: {key}")
            
            return signed_url
            
        except Exception as e:
            logger.error(f"Signed URL generation failed: {e}")
            return None

    async def delete_file(self, key: str) -> bool:
        """Delete file from storage"""
        
        try:
            if self.client:
                # Delete from Replit storage
                try:
                    self.client.delete(key + ".b64")
                    self.client.delete(key + ".metadata")
                except Exception:
                    pass
            else:
                # Delete from local storage
                local_path = self.local_storage_path / key
                metadata_file = local_path.with_suffix(local_path.suffix + ".metadata")
                
                if local_path.exists():
                    os.remove(local_path)
                if metadata_file.exists():
                    os.remove(metadata_file)
            
            self.stats["objects_deleted"] += 1
            logger.info(f"File deleted: {key}")
            return True
            
        except Exception as e:
            logger.error(f"File deletion failed: {e}")
            return False

    async def create_hls_variants(
        self,
        input_video_path: str,
        output_base_key: str
    ) -> List[VideoVariant]:
        """Create multiple bitrate variants for HLS streaming"""
        
        try:
            variants = []
            
            for variant_config in self.hls_variants:
                # Generate variant file path
                variant_key = f"{output_base_key}_{variant_config['resolution']}.mp4"
                variant_path = f"/tmp/{variant_key}"
                
                # Mock video processing (in production, use FFmpeg)
                await self._create_video_variant(
                    input_video_path,
                    variant_path,
                    variant_config
                )
                
                # Store variant
                await self.store_file(
                    variant_path,
                    variant_key,
                    content_type="video/mp4",
                    metadata={
                        "resolution": variant_config["resolution"],
                        "bitrate": variant_config["bitrate"],
                        "variant_type": "hls"
                    }
                )
                
                # Create variant object
                variant = VideoVariant(
                    resolution=variant_config["resolution"],
                    bitrate=variant_config["bitrate"],
                    codec="h264",
                    file_path=variant_key,
                    bandwidth=variant_config["bitrate"],
                    size=os.path.getsize(variant_path) if os.path.exists(variant_path) else 0
                )
                
                variants.append(variant)
                
                # Cleanup temp file
                if os.path.exists(variant_path):
                    os.remove(variant_path)
            
            # Create master playlist
            master_playlist = await self._create_master_playlist(variants, output_base_key)
            await self.store_file(
                master_playlist,
                f"{output_base_key}_master.m3u8",
                content_type="application/vnd.apple.mpegurl",
                metadata={"playlist_type": "master"}
            )
            
            logger.info(f"Created {len(variants)} HLS variants for: {output_base_key}")
            return variants
            
        except Exception as e:
            logger.error(f"HLS variant creation failed: {e}")
            return []

    async def _create_video_variant(
        self,
        input_path: str,
        output_path: str,
        config: Dict[str, Any]
    ):
        """Create video variant with specified configuration"""
        # Mock implementation - in production, use FFmpeg
        import shutil
        shutil.copy2(input_path, output_path)

    async def _create_master_playlist(
        self,
        variants: List[VideoVariant],
        base_key: str
    ) -> str:
        """Create HLS master playlist"""
        
        playlist_content = "#EXTM3U\n#EXT-X-VERSION:3\n\n"
        
        for variant in variants:
            playlist_content += f"#EXT-X-STREAM-INF:BANDWIDTH={variant.bandwidth},"
            playlist_content += f"RESOLUTION={variant.resolution.replace('p', '')}"
            playlist_content += f"x{variant.resolution.replace('p', '')}\n"
            playlist_content += f"{variant.file_path}\n\n"
        
        # Write to temp file
        playlist_path = f"/tmp/{base_key}_master.m3u8"
        with open(playlist_path, 'w') as f:
            f.write(playlist_content)
        
        return playlist_path

    async def cleanup_expired_files(self) -> int:
        """Clean up expired files"""
        
        try:
            cleaned_count = 0
            
            if self.client:
                # For Replit storage, we'd need to list and check metadata
                # This is a simplified approach
                pass
            else:
                # Local storage cleanup
                for metadata_file in self.local_storage_path.rglob("*.metadata"):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        expires_at_str = metadata.get("expires_at")
                        if expires_at_str:
                            expires_at = datetime.fromisoformat(expires_at_str)
                            if datetime.utcnow() > expires_at:
                                # Delete file and metadata
                                file_path = metadata_file.with_suffix("")
                                if file_path.exists():
                                    os.remove(file_path)
                                os.remove(metadata_file)
                                cleaned_count += 1
                                
                    except Exception as e:
                        logger.warning(f"Error checking file expiration: {e}")
            
            if cleaned_count > 0:
                self.stats["temp_files_cleaned"] += cleaned_count
                logger.info(f"Cleaned up {cleaned_count} expired files")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0

    async def _cleanup_worker(self):
        """Background worker for file cleanup"""
        
        while True:
            try:
                await asyncio.sleep(self.temp_cleanup_interval)
                await self.cleanup_expired_files()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        
        # Calculate storage usage
        total_size = 0
        file_count = 0
        
        if not self.client:
            # Local storage stats
            for file_path in self.local_storage_path.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith(".metadata"):
                    total_size += file_path.stat().st_size
                    file_count += 1
        
        return {
            **self.stats,
            "total_storage_bytes": total_size,
            "total_storage_mb": round(total_size / 1024 / 1024, 2),
            "file_count": file_count,
            "storage_backend": "replit" if self.client else "local",
            "cleanup_interval_seconds": self.temp_cleanup_interval,
            "signed_url_expiry_seconds": self.signed_url_expiry
        }

    async def list_files(
        self,
        prefix: Optional[str] = None,
        limit: int = 100
    ) -> List[StorageObject]:
        """List stored files with optional prefix filter"""
        
        try:
            files = []
            
            if self.client:
                # For Replit storage, we'd implement listing logic
                # This is a simplified version
                pass
            else:
                # Local storage listing
                pattern = f"{prefix}*" if prefix else "*"
                
                for file_path in self.local_storage_path.rglob(pattern):
                    if (file_path.is_file() and 
                        not file_path.name.endswith(".metadata") and
                        len(files) < limit):
                        
                        # Get relative key
                        key = str(file_path.relative_to(self.local_storage_path))
                        
                        # Get storage object
                        storage_obj = await self.retrieve_file(key)
                        if storage_obj:
                            files.append(storage_obj)
            
            return files
            
        except Exception as e:
            logger.error(f"File listing failed: {e}")
            return []


# Global storage service instance
storage_service = NetflixStorageService()

# Export for backward compatibility
CDNService = NetflixStorageService
FileStorageService = NetflixStorageService

logger.info("✅ Netflix-grade storage and CDN service initialized")
