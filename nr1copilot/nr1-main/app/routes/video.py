from fastapi import APIRouter
from app.schemas import (
    VideoValidateIn, VideoValidateOut, VideoInfoOut, VideoProcessIn, VideoProcessOut, VideoJobStatusOut, VideoServeOut
)
from ..controllers.video_controller import (
    validate_youtube_url,
    get_video_info,
    process_video_job,
    check_job_status,
    serve_processed_video,
    serve_sample_video,
)

router = APIRouter(prefix="/api/v1/videos", tags=["Videos"])

@router.post("/validate", response_model=VideoValidateOut)
def validate(data: VideoValidateIn):
    return validate_youtube_url(data)

@router.get("/info/{video_id}", response_model=VideoInfoOut)
def info(video_id: str):
    return get_video_info(video_id)

@router.post("/process", response_model=VideoProcessOut)
def process(data: VideoProcessIn):
    return process_video_job(data)

@router.get("/job/{job_id}", response_model=VideoJobStatusOut)
def job_status(job_id: str):
    return check_job_status(job_id)

@router.get("/serve/{video_id}", response_model=VideoServeOut)
def serve(video_id: str):
    return serve_processed_video(video_id)

@router.get("/sample", response_model=VideoServeOut)
def sample():
    return serve_sample_video()
