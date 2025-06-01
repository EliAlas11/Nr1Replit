
"""
Video processing routes with comprehensive functionality
Handles video uploads, YouTube downloads, analysis, and clip generation
"""

import os
import logging
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..services.video_service import video_service, VideoServiceError
from ..utils.extract_video_id import extract_video_id
from ..config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Request/Response models
class YouTubeDownloadRequest(BaseModel):
    url: str
    
class VideoAnalysisRequest(BaseModel):
    video_id: str
    
class ClipCreateRequest(BaseModel):
    video_id: str
    start_time: int
    duration: int = 60
    title: str = "Generated Clip"

class VideoResponse(BaseModel):
    success: bool
    message: str
    data: dict = {}

@router.post("/download", response_model=VideoResponse)
async def download_youtube_video(
    request: YouTubeDownloadRequest,
    background_tasks: BackgroundTasks
):
    """Download video from YouTube URL"""
    try:
        # Validate YouTube URL
        video_id = extract_video_id(request.url)
        if not video_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid YouTube URL"
            )
        
        logger.info(f"Starting download for video ID: {video_id}")
        
        # Start download
        result = await video_service.download_youtube_video(request.url, video_id)
        
        return VideoResponse(
            success=True,
            message="Video downloaded successfully",
            data={
                "video_id": video_id,
                "filepath": result["filepath"],
                "title": result["title"],
                "duration": result["duration"],
                "metadata": {
                    "uploader": result["uploader"],
                    "view_count": result["view_count"],
                    "upload_date": result["upload_date"]
                }
            }
        )
        
    except VideoServiceError as e:
        logger.error(f"Video service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/analyze", response_model=VideoResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """Analyze video for potential highlights"""
    try:
        # Find video file
        video_path = None
        search_dir = os.path.join(settings.video_storage_path, request.video_id)
        
        if os.path.exists(search_dir):
            for file in os.listdir(search_dir):
                if file.endswith(('.mp4', '.mkv', '.avi', '.mov')):
                    video_path = os.path.join(search_dir, file)
                    break
        
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found"
            )
        
        # Analyze video
        highlights = await video_service.analyze_video_highlights(video_path)
        video_info = await video_service.get_video_info(video_path)
        
        return VideoResponse(
            success=True,
            message="Video analyzed successfully",
            data={
                "video_id": request.video_id,
                "highlights": highlights,
                "video_info": video_info,
                "total_highlights": len(highlights)
            }
        )
        
    except VideoServiceError as e:
        logger.error(f"Video service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/create-clip", response_model=VideoResponse)
async def create_video_clip(request: ClipCreateRequest):
    """Create a clip from the original video"""
    try:
        # Find source video
        video_path = None
        search_dir = os.path.join(settings.video_storage_path, request.video_id)
        
        if os.path.exists(search_dir):
            for file in os.listdir(search_dir):
                if file.endswith(('.mp4', '.mkv', '.avi', '.mov')):
                    video_path = os.path.join(search_dir, file)
                    break
        
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source video not found"
            )
        
        # Generate output filename
        output_filename = f"{request.video_id}_clip_{request.start_time}s_{request.duration}s.mp4"
        
        # Create clip
        result = await video_service.create_clip(
            input_path=video_path,
            output_filename=output_filename,
            start_time=request.start_time,
            duration=request.duration
        )
        
        return VideoResponse(
            success=True,
            message="Clip created successfully",
            data={
                "clip_id": f"{request.video_id}_{request.start_time}s",
                "output_path": result["output_path"],
                "filename": result["filename"],
                "start_time": result["start_time"],
                "duration": result["duration"],
                "file_size": result["file_size"],
                "title": request.title
            }
        )
        
    except VideoServiceError as e:
        logger.error(f"Video service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/download-clip/{video_id}/{clip_filename}")
async def download_clip(video_id: str, clip_filename: str):
    """Download a generated clip"""
    try:
        clip_path = os.path.join(settings.video_storage_path, "clips", clip_filename)
        
        if not os.path.exists(clip_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clip not found"
            )
        
        return FileResponse(
            path=clip_path,
            filename=clip_filename,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename={clip_filename}",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Download failed"
        )

@router.get("/preview/{video_id}/{clip_filename}")
async def preview_clip(video_id: str, clip_filename: str):
    """Preview a generated clip"""
    try:
        clip_path = os.path.join(settings.video_storage_path, "clips", clip_filename)
        
        if not os.path.exists(clip_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Clip not found"
            )
        
        return FileResponse(
            path=clip_path,
            media_type="video/mp4",
            headers={
                "Cache-Control": "max-age=3600",
                "Accept-Ranges": "bytes"
            }
        )
        
    except Exception as e:
        logger.error(f"Preview error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Preview failed"
        )

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing"""
    try:
        # Validate file type
        allowed_types = ["video/mp4", "video/avi", "video/mov", "video/mkv"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only video files are allowed."
            )
        
        # Validate file size
        if hasattr(file.file, 'seek'):
            file.file.seek(0, 2)  # Seek to end
            size = file.file.tell()
            file.file.seek(0)  # Reset to beginning
            
            if size > settings.max_file_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Maximum size: {settings.max_file_size // (1024*1024)}MB"
                )
        
        # Generate unique filename
        import uuid
        video_id = str(uuid.uuid4())
        filename = f"{video_id}_{file.filename}"
        upload_dir = os.path.join(settings.upload_storage_path, video_id)
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded successfully: {file_path}")
        
        return VideoResponse(
            success=True,
            message="File uploaded successfully",
            data={
                "video_id": video_id,
                "filename": filename,
                "file_path": file_path,
                "file_size": len(content),
                "content_type": file.content_type
            }
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed"
        )

@router.get("/list")
async def list_videos():
    """List all available videos and clips"""
    try:
        videos = []
        clips = []
        
        # List original videos
        if os.path.exists(settings.video_storage_path):
            for item in os.listdir(settings.video_storage_path):
                item_path = os.path.join(settings.video_storage_path, item)
                if os.path.isdir(item_path):
                    for file in os.listdir(item_path):
                        if file.endswith(('.mp4', '.mkv', '.avi', '.mov')):
                            file_path = os.path.join(item_path, file)
                            file_stat = os.stat(file_path)
                            videos.append({
                                "video_id": item,
                                "filename": file,
                                "size": file_stat.st_size,
                                "created": file_stat.st_ctime
                            })
        
        # List clips
        clips_dir = os.path.join(settings.video_storage_path, "clips")
        if os.path.exists(clips_dir):
            for file in os.listdir(clips_dir):
                if file.endswith('.mp4'):
                    file_path = os.path.join(clips_dir, file)
                    file_stat = os.stat(file_path)
                    clips.append({
                        "filename": file,
                        "size": file_stat.st_size,
                        "created": file_stat.st_ctime
                    })
        
        return VideoResponse(
            success=True,
            message="Videos listed successfully",
            data={
                "videos": videos,
                "clips": clips,
                "total_videos": len(videos),
                "total_clips": len(clips)
            }
        )
        
    except Exception as e:
        logger.error(f"List error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list videos"
        )
