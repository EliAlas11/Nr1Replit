
"""
Netflix-Grade Storage API Routes
Comprehensive file storage, CDN, and streaming endpoints
"""

import os
import tempfile
import mimetypes
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..services.storage_service import storage_service
from ..utils.api_responses import APIResponseBuilder
from ..middleware.security import SecurityMiddleware
from ..middleware.validation import ValidationMiddleware

router = APIRouter(prefix="/api/v10/storage", tags=["Storage & CDN"])

class StorageUploadRequest(BaseModel):
    key: str
    content_type: Optional[str] = None
    expires_in: Optional[int] = None
    metadata: Optional[dict] = None

class StorageResponse(BaseModel):
    key: str
    size: int
    content_type: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    signed_url: Optional[str] = None
    public_url: Optional[str] = None

class HLSVariantResponse(BaseModel):
    resolution: str
    bitrate: int
    codec: str
    file_path: str
    bandwidth: int
    size: int

@router.post("/upload", response_model=dict)
async def upload_file(
    file: UploadFile = File(...),
    key: Optional[str] = None,
    expires_in: Optional[int] = None,
    generate_signed_url: bool = True
):
    """Upload file to storage with optional expiration"""
    
    try:
        # Generate key if not provided
        if not key:
            import uuid
            file_ext = Path(file.filename).suffix if file.filename else ""
            key = f"uploads/{uuid.uuid4().hex}{file_ext}"
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Read and write file content
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            try:
                # Store file
                storage_obj = await storage_service.store_file(
                    file_path=temp_file.name,
                    key=key,
                    content_type=file.content_type,
                    expires_in=expires_in,
                    metadata={
                        "original_filename": file.filename,
                        "upload_timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # Generate signed URL if requested
                signed_url = None
                if generate_signed_url:
                    signed_url = await storage_service.generate_signed_url(key)
                
                return APIResponseBuilder.success(
                    data={
                        "key": storage_obj.key,
                        "size": storage_obj.size,
                        "content_type": storage_obj.content_type,
                        "created_at": storage_obj.created_at.isoformat(),
                        "expires_at": storage_obj.expires_at.isoformat() if storage_obj.expires_at else None,
                        "signed_url": signed_url,
                        "download_url": f"/api/v10/storage/download/{key}"
                    },
                    message="File uploaded successfully"
                )
                
            finally:
                # Clean up temp file
                os.unlink(temp_file.name)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/download/{key:path}")
async def download_file(
    key: str,
    expires: Optional[str] = Query(None),
    signature: Optional[str] = Query(None),
    inline: bool = Query(False, description="Display inline instead of download")
):
    """Download file with optional signature verification"""
    
    try:
        # Verify signed URL if signature provided
        if signature and expires:
            import hmac
            import hashlib
            from ..config import get_settings
            
            settings = get_settings()
            
            # Verify signature
            expected_signature = hmac.new(
                key=settings.secret_key.encode(),
                msg=f"{key}:{expires}".encode(),
                digestmod=hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                raise HTTPException(status_code=403, detail="Invalid signature")
            
            # Check expiration
            try:
                expires_dt = datetime.fromisoformat(expires)
                if datetime.utcnow() > expires_dt:
                    raise HTTPException(status_code=403, detail="URL expired")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid expiration format")
        
        # Get file from storage
        storage_obj = await storage_service.retrieve_file(key)
        if not storage_obj:
            raise HTTPException(status_code=404, detail="File not found")
        
        # For local storage, return file directly
        if not storage_service.client:
            local_path = storage_service.local_storage_path / key
            if not local_path.exists():
                raise HTTPException(status_code=404, detail="File not found")
            
            # Determine disposition
            disposition = "inline" if inline else "attachment"
            filename = storage_obj.metadata.get("original_filename", key.split("/")[-1])
            
            return FileResponse(
                path=str(local_path),
                media_type=storage_obj.content_type,
                filename=filename,
                headers={"Content-Disposition": f"{disposition}; filename={filename}"}
            )
        
        # For Replit storage, stream content
        try:
            import base64
            encoded_content = storage_service.client.download_as_text(f"{key}.b64")
            content = base64.b64decode(encoded_content)
            
            def generate():
                yield content
            
            disposition = "inline" if inline else "attachment"
            filename = storage_obj.metadata.get("original_filename", key.split("/")[-1])
            
            return StreamingResponse(
                generate(),
                media_type=storage_obj.content_type,
                headers={
                    "Content-Disposition": f"{disposition}; filename={filename}",
                    "Content-Length": str(storage_obj.size)
                }
            )
            
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to retrieve file content")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.get("/info/{key:path}")
async def get_file_info(key: str):
    """Get file information and metadata"""
    
    try:
        storage_obj = await storage_service.retrieve_file(key)
        if not storage_obj:
            raise HTTPException(status_code=404, detail="File not found")
        
        return APIResponseBuilder.success(
            data={
                "key": storage_obj.key,
                "size": storage_obj.size,
                "content_type": storage_obj.content_type,
                "created_at": storage_obj.created_at.isoformat(),
                "expires_at": storage_obj.expires_at.isoformat() if storage_obj.expires_at else None,
                "metadata": storage_obj.metadata
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get file info: {str(e)}")

@router.post("/signed-url/{key:path}")
async def generate_signed_url(
    key: str,
    expires_in: Optional[int] = Query(3600, description="URL expiration in seconds"),
    method: str = Query("GET", description="HTTP method")
):
    """Generate signed URL for secure file access"""
    
    try:
        signed_url = await storage_service.generate_signed_url(
            key=key,
            expires_in=expires_in,
            method=method
        )
        
        if not signed_url:
            raise HTTPException(status_code=404, detail="File not found")
        
        return APIResponseBuilder.success(
            data={
                "signed_url": signed_url,
                "expires_in": expires_in,
                "method": method
            },
            message="Signed URL generated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {str(e)}")

@router.delete("/{key:path}")
async def delete_file(key: str):
    """Delete file from storage"""
    
    try:
        success = await storage_service.delete_file(key)
        
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        
        return APIResponseBuilder.success(
            message="File deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@router.post("/video/hls/{key:path}")
async def create_hls_variants(
    key: str,
    output_prefix: Optional[str] = None
):
    """Create HLS variants for adaptive bitrate streaming"""
    
    try:
        # Get original video file
        storage_obj = await storage_service.retrieve_file(key)
        if not storage_obj:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        if not storage_obj.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File is not a video")
        
        # Generate output prefix
        if not output_prefix:
            output_prefix = f"hls/{key.replace('/', '_')}"
        
        # Download original video to temp file
        if storage_service.client:
            import base64
            encoded_content = storage_service.client.download_as_text(f"{key}.b64")
            content = base64.b64decode(encoded_content)
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                temp_video_path = temp_file.name
        else:
            temp_video_path = str(storage_service.local_storage_path / key)
        
        try:
            # Create HLS variants
            variants = await storage_service.create_hls_variants(
                input_video_path=temp_video_path,
                output_base_key=output_prefix
            )
            
            return APIResponseBuilder.success(
                data={
                    "master_playlist": f"{output_prefix}_master.m3u8",
                    "variants": [
                        {
                            "resolution": v.resolution,
                            "bitrate": v.bitrate,
                            "codec": v.codec,
                            "file_path": v.file_path,
                            "bandwidth": v.bandwidth,
                            "size": v.size
                        }
                        for v in variants
                    ],
                    "variant_count": len(variants)
                },
                message="HLS variants created successfully"
            )
            
        finally:
            # Clean up temp file if we created it
            if storage_service.client and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create HLS variants: {str(e)}")

@router.get("/list")
async def list_files(
    prefix: Optional[str] = Query(None, description="Filter by key prefix"),
    limit: int = Query(100, description="Maximum number of files to return")
):
    """List stored files with optional filtering"""
    
    try:
        files = await storage_service.list_files(prefix=prefix, limit=limit)
        
        return APIResponseBuilder.success(
            data={
                "files": [
                    {
                        "key": f.key,
                        "size": f.size,
                        "content_type": f.content_type,
                        "created_at": f.created_at.isoformat(),
                        "expires_at": f.expires_at.isoformat() if f.expires_at else None,
                        "metadata": f.metadata
                    }
                    for f in files
                ],
                "count": len(files),
                "prefix": prefix
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@router.post("/cleanup")
async def cleanup_expired_files():
    """Manually trigger cleanup of expired files"""
    
    try:
        cleaned_count = await storage_service.cleanup_expired_files()
        
        return APIResponseBuilder.success(
            data={"cleaned_files": cleaned_count},
            message=f"Cleaned up {cleaned_count} expired files"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/stats")
async def get_storage_stats():
    """Get comprehensive storage statistics"""
    
    try:
        stats = await storage_service.get_storage_stats()
        
        return APIResponseBuilder.success(
            data=stats,
            message="Storage statistics retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# Health check endpoint
@router.get("/health")
async def storage_health():
    """Storage service health check"""
    
    try:
        # Test basic storage operations
        test_key = "health_check_test"
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("health check test")
            temp_path = temp_file.name
        
        try:
            # Test store and retrieve
            await storage_service.store_file(
                temp_path,
                test_key,
                expires_in=60  # 1 minute expiration
            )
            
            storage_obj = await storage_service.retrieve_file(test_key)
            
            if storage_obj:
                await storage_service.delete_file(test_key)
                status = "healthy"
            else:
                status = "unhealthy"
            
        finally:
            os.unlink(temp_path)
        
        return APIResponseBuilder.success(
            data={
                "status": status,
                "backend": "replit" if storage_service.client else "local",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        return APIResponseBuilder.error(
            error_code="STORAGE_HEALTH_FAILED",
            message=f"Storage health check failed: {str(e)}"
        )
