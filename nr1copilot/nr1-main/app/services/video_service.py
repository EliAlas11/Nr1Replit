"""
Video Service Layer (Refactored)

- Handles video validation, info retrieval, processing, job status, and serving logic.
- All business logic, validation, and error handling for video operations is centralized here.
- Designed for auditability, security, and testability (Stripe/Netflix standards).
- All functions are stateless and side-effect free except for job queueing and file serving.
- Implements audit logging and custom exceptions for compliance.
"""

import logging
from app.schemas import (
    VideoValidateOut, VideoInfoOut, VideoProcessIn, VideoProcessOut, VideoJobStatusOut, VideoServeOut
)
from app.utils.extract_video_id import extract_video_id

logger = logging.getLogger("video_service")

class VideoServiceError(Exception):
    """Custom exception for VideoService errors."""
    pass

def validate_youtube_url_service(url: str) -> VideoValidateOut:
    """
    Validate a YouTube URL and extract the video ID.

    Args:
        url (str): YouTube video URL.

    Returns:
        VideoValidateOut: Validation result with video ID or error message.

    Raises:
        VideoServiceError: If the URL is invalid.
    """
    video_id = extract_video_id(url)
    if not video_id:
        logger.warning(f"Invalid YouTube URL: {url}")
        raise VideoServiceError("Invalid YouTube URL format")
    logger.info(f"Validated YouTube URL: {url}, video_id: {video_id}")
    return VideoValidateOut(valid=True, video_id=video_id, message=None)

def get_video_info_service(video_id: str) -> VideoInfoOut:
    """
    Retrieve video information by video ID.

    Args:
        video_id (str): YouTube video ID.

    Returns:
        VideoInfoOut: Video information (stubbed for now).

    Raises:
        VideoServiceError: If video info cannot be retrieved.
    """
    # TODO: Implement real info fetch (duration, title, etc.)
    logger.info(f"Fetching info for video_id: {video_id}")
    return VideoInfoOut(video_id=video_id, duration=60, title="Sample Video")

def process_video_job_service(data: VideoProcessIn) -> VideoProcessOut:
    """
    Queue a video processing job.

    Args:
        data (VideoProcessIn): Video processing parameters.

    Returns:
        VideoProcessOut: Job ID and status (stubbed for now).

    Raises:
        VideoServiceError: If job cannot be queued.
    """
    # TODO: Enqueue job in Celery/RQ
    logger.info(f"Queued video processing job for video_id: {data.video_id}")
    return VideoProcessOut(job_id="sample-job-id", status="queued")

def check_job_status_service(job_id: str) -> VideoJobStatusOut:
    """
    Check the status of a video processing job.

    Args:
        job_id (str): Job ID.

    Returns:
        VideoJobStatusOut: Job status (stubbed for now).

    Raises:
        VideoServiceError: If job status cannot be checked.
    """
    # TODO: Check job status in queue
    logger.info(f"Checked status for job_id: {job_id}")
    return VideoJobStatusOut(job_id=job_id, status="processing")

def serve_processed_video_service(video_id: str) -> VideoServeOut:
    """
    Serve a processed video file by video ID.

    Args:
        video_id (str): Video ID.

    Returns:
        VideoServeOut: Video URL (stubbed for now).

    Raises:
        VideoServiceError: If video cannot be served.
    """
    # TODO: Serve video file from processed directory
    logger.info(f"Serving processed video for video_id: {video_id}")
    return VideoServeOut(video_url=f"/videos/processed/{video_id}.mp4")

def serve_sample_video_service() -> VideoServeOut:
    """
    Serve a sample video file.

    Returns:
        VideoServeOut: Sample video URL (stubbed for now).
    """
    logger.info("Serving sample video")
    return VideoServeOut(video_url="/videos/sample.mp4")

# Test block for service sanity (not for production)
if __name__ == "__main__":
    import sys
    try:
        print(validate_youtube_url_service("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        print(get_video_info_service("dQw4w9WgXcQ"))
        print(process_video_job_service(VideoProcessIn(video_id="dQw4w9WgXcQ", start_time=0, end_time=10)))
        print(check_job_status_service("sample-job-id"))
        print(serve_processed_video_service("dQw4w9WgXcQ"))
        print(serve_sample_video_service())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
