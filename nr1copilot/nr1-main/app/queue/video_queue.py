import redis
from rq import Queue
from ..config import settings

redis_conn = redis.Redis.from_url(settings.REDIS_URL)
video_queue = Queue('video-processing', connection=redis_conn)

# Example enqueue usage:
# job = video_queue.enqueue(process_video_job, args=(...))
