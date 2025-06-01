"""
RQ worker for processing video jobs in the background.
Listens to the 'video-processing' queue and runs jobs using process_video_job_service.
"""

from rq import Worker, Queue, Connection
from ..services.video_service import process_video_job_service
from ..queue.video_queue import redis_conn

listen = ['video-processing']

def start_worker() -> None:
    """Start the RQ worker for video-processing jobs."""
    with Connection(redis_conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()

if __name__ == '__main__':
    start_worker()
