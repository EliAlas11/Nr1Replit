"""
Video Processing Routes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging
import os
from ..schemas import VideoProcessRequest, VideoOut, SuccessResponse
from ..services.video_service import download_video, get_video_info, validate_youtube_url, process_video_clip
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

@router.post("/video/validate", response_model=SuccessResponse)
async def validate_video_url(data: Dict[str, str]):
    """Validate YouTube URL"""
    url = data.get("url", "")

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    is_valid = await validate_youtube_url(url)

    return SuccessResponse(
        message="URL validation completed",
        data={"valid": is_valid, "url": url}
    )

@router.get("/video/info/{video_id}")
async def get_video_metadata(video_id: str):
    """Get video information"""
    try:
        # Mock response for now
        return {
            "id": video_id,
            "title": "Sample Video",
            "duration": 180,
            "thumbnail": "https://img.youtube.com/vi/sample/maxresdefault.jpg"
        }
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        raise HTTPException(status_code=404, detail="Video not found")

@router.post("/video/download")
async def download_youtube_video(data: Dict[str, str], background_tasks: BackgroundTasks):
    """Download video from YouTube"""
    url = data.get("url", "")

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    # Validate URL first
    if not await validate_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    try:
        # Get video info first
        info = await get_video_info(url)

        return SuccessResponse(
            message="Video download initiated",
            data={
                "video_info": info,
                "status": "processing"
            }
        )

    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.post("/video/process", response_model=SuccessResponse)
async def process_video(request: VideoProcessRequest):
    """Process video to create a clip"""
    try:
        # Validate the request
        if not await validate_youtube_url(request.youtube_url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        # Mock processing for now
        clip_info = {
            "id": "clip_123",
            "status": "processing",
            "start_time": request.start_time,
            "end_time": request.end_time,
            "duration": request.end_time - request.start_time,
            "quality": request.quality
        }

        return SuccessResponse(
            message="Video processing initiated",
            data=clip_info
        )

    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.get("/videos", response_model=SuccessResponse)
async def list_videos():
    """List all processed videos"""
    try:
        videos = []

        # Mock video list
        videos.append({
            "id": "video_1",
            "title": "Sample Video 1",
            "duration": 180,
            "status": "completed",
            "created_at": "2024-01-01T00:00:00Z"
        })

        return SuccessResponse(
            message="Videos retrieved successfully",
            data={"videos": videos, "total": len(videos)}
        )

    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail="Failed to list videos")