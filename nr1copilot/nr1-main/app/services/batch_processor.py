"""
ViralClip Pro v6.0 - Netflix-Level Batch Processing Service
Advanced queue management with prioritization and real-time error handling
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import heapq
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)


class JobPriority(IntEnum):
    """Job priority levels (lower number = higher priority)"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class JobStatus(Enum):
    """Job processing status"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobType(Enum):
    """Types of processing jobs"""
    VIDEO_ANALYSIS = "video_analysis"
    CAPTION_GENERATION = "caption_generation"
    TEMPLATE_RENDERING = "template_rendering"
    BATCH_EXPORT = "batch_export"
    AI_ENHANCEMENT = "ai_enhancement"
    VIRAL_OPTIMIZATION = "viral_optimization"
    BULK_PROCESSING = "bulk_processing"


@dataclass
class BatchJob:
    """Individual batch processing job"""
    job_id: str
    job_type: JobType
    priority: JobPriority
    input_data: Dict[str, Any]
    output_path: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    user_id: str = ""
    session_id: str = ""
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Priority comparison for heap queue"""
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)


@dataclass
class BatchQueue:
    """Batch processing queue with advanced management"""
    queue_id: str
    name: str
    max_concurrent_jobs: int
    job_types: List[JobType]
    active_jobs: Dict[str, BatchJob] = field(default_factory=dict)
    pending_jobs: List[BatchJob] = field(default_factory=list)
    completed_jobs: List[BatchJob] = field(default_factory=list)
    failed_jobs: List[BatchJob] = field(default_factory=list)
    is_paused: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DistributedNode:
    """Distributed processing node with advanced capabilities"""
    node_id: str
    node_type: str
    location: str
    max_concurrent_jobs: int
    current_jobs: int = 0
    cpu_cores: int = 8
    memory_gb: int = 16
    gpu_available: bool = False
    gpu_memory_gb: int = 0
    network_bandwidth_mbps: int = 1000
    storage_type: str = "ssd"
    capabilities: List[JobType] = field(default_factory=list)
    health_metrics: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    load_balancer_weight: float = 1.0
    geographic_region: str = "us-east-1"


@dataclass
class WorkerNode:
    """Processing worker node"""
    worker_id: str
    node_type: str
    max_jobs: int
    current_jobs: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    is_healthy: bool = True
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    capabilities: List[JobType] = field(default_factory=list)


class NetflixLevelBatchProcessor:
    """Netflix-level batch processing with enterprise features"""

    def __init__(self):
        # Core processing infrastructure
        self.queues: Dict[str, BatchQueue] = {}
        self.workers: Dict[str, WorkerNode] = {}
        self.job_registry: Dict[str, BatchJob] = {}

        # Performance monitoring
        self.processing_stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "average_processing_time": 0.0,
            "queue_throughput": 0.0,
            "error_rate": 0.0
        }

        # Job handlers registry
        self.job_handlers: Dict[JobType, Callable] = {}

        # Background tasks
        self.background_tasks: set = set()

        # Initialize default queues
        asyncio.create_task(self._initialize_default_queues())

        # Start background monitoring
        asyncio.create_task(self._start_background_monitoring())

        logger.info("🚀 Netflix-level batch processor initialized")

    async def _initialize_default_queues(self):
        """Initialize default processing queues"""

        default_queues = [
            {
                "name": "High Priority",
                "max_concurrent": 10,
                "job_types": [JobType.VIDEO_ANALYSIS, JobType.AI_ENHANCEMENT]
            },
            {
                "name": "Standard Processing",
                "max_concurrent": 20,
                "job_types": [JobType.CAPTION_GENERATION, JobType.TEMPLATE_RENDERING]
            },
            {
                "name": "Bulk Operations",
                "max_concurrent": 5,
                "job_types": [JobType.BATCH_EXPORT, JobType.BULK_PROCESSING]
            },
            {
                "name": "Background Tasks",
                "max_concurrent": 15,
                "job_types": [JobType.VIRAL_OPTIMIZATION]
            }
        ]

        for queue_config in default_queues:
            queue_id = str(uuid.uuid4())
            queue = BatchQueue(
                queue_id=queue_id,
                name=queue_config["name"],
                max_concurrent_jobs=queue_config["max_concurrent"],
                job_types=queue_config["job_types"]
            )
            self.queues[queue_id] = queue

        logger.info(f"✅ Initialized {len(self.queues)} processing queues")

    async def _start_background_monitoring(self):
        """Start background monitoring tasks"""

        # Queue processor
        monitor_task = asyncio.create_task(self._queue_processor())
        self.background_tasks.add(monitor_task)

        # Health checker
        health_task = asyncio.create_task(self._health_monitor())
        self.background_tasks.add(health_task)

        # Performance tracker
        perf_task = asyncio.create_task(self._performance_tracker())
        self.background_tasks.add(perf_task)

        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_completed_jobs())
        self.background_tasks.add(cleanup_task)

    async def submit_job(
        self, 
        job_type: JobType,
        input_data: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        user_id: str = "",
        session_id: str = "",
        dependencies: List[str] = None,
        estimated_duration: float = 0.0
    ) -> str:
        """Submit a new job to the batch processor"""

        try:
            job_id = str(uuid.uuid4())

            job = BatchJob(
                job_id=job_id,
                job_type=job_type,
                priority=priority,
                input_data=input_data,
                user_id=user_id,
                session_id=session_id,
                dependencies=dependencies or [],
                estimated_duration=estimated_duration
            )

            # Register job
            self.job_registry[job_id] = job

            # Find appropriate queue
            target_queue = await self._find_optimal_queue(job_type)
            if not target_queue:
                raise ValueError(f"No suitable queue found for job type: {job_type}")

            # Add to queue
            heapq.heappush(target_queue.pending_jobs, job)
            job.status = JobStatus.QUEUED

            self.processing_stats["total_jobs"] += 1

            logger.info(f"📝 Job submitted: {job_id} ({job_type.value}) - Priority: {priority.name}")

            return job_id

        except Exception as e:
            logger.error(f"Job submission failed: {e}", exc_info=True)
            raise

    async def _find_optimal_queue(self, job_type: JobType) -> Optional[BatchQueue]:
        """Find the optimal queue for a job type"""

        suitable_queues = [
            queue for queue in self.queues.values()
            if job_type in queue.job_types and not queue.is_paused
        ]

        if not suitable_queues:
            return None

        # Sort by current load (ascending)
        suitable_queues.sort(
            key=lambda q: len(q.active_jobs) / q.max_concurrent_jobs
        )

        return suitable_queues[0]

    async def _queue_processor(self):
        """Main queue processing loop"""

        while True:
            try:
                for queue in self.queues.values():
                    if queue.is_paused:
                        continue

                    # Process pending jobs if capacity available
                    available_slots = queue.max_concurrent_jobs - len(queue.active_jobs)

                    while available_slots > 0 and queue.pending_jobs:
                        # Get highest priority job
                        job = heapq.heappop(queue.pending_jobs)

                        # Check dependencies
                        if not await self._check_dependencies(job):
                            # Re-queue if dependencies not met
                            heapq.heappush(queue.pending_jobs, job)
                            break

                        # Start processing
                        await self._start_job_processing(job, queue)
                        available_slots -= 1

                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Queue processor error: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _check_dependencies(self, job: BatchJob) -> bool:
        """Check if job dependencies are satisfied"""

        if not job.dependencies:
            return True

        for dep_job_id in job.dependencies:
            dep_job = self.job_registry.get(dep_job_id)
            if not dep_job or dep_job.status != JobStatus.COMPLETED:
                return False

        return True

    async def _start_job_processing(self, job: BatchJob, queue: BatchQueue):
        """Start processing a job"""

        try:
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()
            queue.active_jobs[job.job_id] = job

            logger.info(f"🚀 Starting job: {job.job_id} ({job.job_type.value})")

            # Process job in background
            task = asyncio.create_task(self._process_job(job))
            task.add_done_callback(lambda t: self._job_completed(job, queue, t))

        except Exception as e:
            logger.error(f"Failed to start job {job.job_id}: {e}", exc_info=True)
            await self._handle_job_failure(job, queue, str(e))

    def _job_completed(self, job: BatchJob, queue: BatchQueue, task: asyncio.Task):
        """Handle job completion"""

        try:
            # Remove from active jobs
            queue.active_jobs.pop(job.job_id, None)

            if task.exception():
                # Job failed
                asyncio.create_task(
                    self._handle_job_failure(job, queue, str(task.exception()))
                )
            else:
                # Job succeeded
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.actual_duration = (job.completed_at - job.started_at).total_seconds()

                queue.completed_jobs.append(job)
                self.processing_stats["completed_jobs"] += 1

                logger.info(f"✅ Job completed: {job.job_id} in {job.actual_duration:.2f}s")

        except Exception as e:
            logger.error(f"Job completion handling failed: {e}", exc_info=True)

    async def _handle_job_failure(self, job: BatchJob, queue: BatchQueue, error: str):
        """Handle job failure with retry logic"""

        job.error_message = error
        job.retry_count += 1

        if job.retry_count <= job.max_retries:
            # Retry with exponential backoff
            delay = min(300, 2 ** job.retry_count)  # Max 5 minutes
            job.status = JobStatus.RETRYING

            logger.warning(f"⚠️ Job {job.job_id} failed, retrying in {delay}s (attempt {job.retry_count})")

            # Schedule retry
            asyncio.create_task(self._schedule_retry(job, queue, delay))
        else:
            # Max retries exceeded
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()

            queue.failed_jobs.append(job)
            self.processing_stats["failed_jobs"] += 1

            logger.error(f"❌ Job {job.job_id} failed permanently after {job.retry_count} retries")

    async def _schedule_retry(self, job: BatchJob, queue: BatchQueue, delay: float):
        """Schedule job retry after delay"""

        await asyncio.sleep(delay)

        # Reset job state for retry
        job.status = JobStatus.QUEUED
        job.progress = 0.0
        job.started_at = None

        # Re-add to queue
        heapq.heappush(queue.pending_jobs, job)

        logger.info(f"🔄 Job {job.job_id} queued for retry")

    async def _process_job(self, job: BatchJob):
        """Process an individual job"""

        handler = self.job_handlers.get(job.job_type)
        if not handler:
            raise ValueError(f"No handler registered for job type: {job.job_type}")

        # Update progress periodically
        progress_task = asyncio.create_task(self._update_job_progress(job))

        try:
            # Execute job handler
            result = await handler(job)
            job.metadata["result"] = result

        finally:
            progress_task.cancel()

    async def _update_job_progress(self, job: BatchJob):
        """Update job progress periodically"""

        try:
            while job.status == JobStatus.PROCESSING:
                # Simulate progress update (in real implementation, this would come from handlers)
                if job.estimated_duration > 0:
                    elapsed = (datetime.utcnow() - job.started_at).total_seconds()
                    job.progress = min(0.9, elapsed / job.estimated_duration)

                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def register_job_handler(self, job_type: JobType, handler: Callable):
        """Register a job handler for specific job type"""

        self.job_handlers[job_type] = handler
        logger.info(f"📋 Registered handler for job type: {job_type.value}")

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific job"""

        job = self.job_registry.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "job_type": job.job_type.value,
            "status": job.status.value,
            "progress": job.progress,
            "priority": job.priority.name,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "estimated_duration": job.estimated_duration,
            "actual_duration": job.actual_duration,
            "retry_count": job.retry_count,
            "error_message": job.error_message,
            "user_id": job.user_id,
            "session_id": job.session_id,
            "metadata": job.metadata
        }

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all queues"""

        queue_status = {}

        for queue_id, queue in self.queues.items():
            queue_status[queue_id] = {
                "name": queue.name,
                "is_paused": queue.is_paused,
                "max_concurrent_jobs": queue.max_concurrent_jobs,
                "active_jobs": len(queue.active_jobs),
                "pending_jobs": len(queue.pending_jobs),
                "completed_jobs": len(queue.completed_jobs),
                "failed_jobs": len(queue.failed_jobs),
                "job_types": [jt.value for jt in queue.job_types],
                "utilization": len(queue.active_jobs) / queue.max_concurrent_jobs * 100
            }

        return {
            "queues": queue_status,
            "overall_stats": self.processing_stats,
            "total_jobs_in_system": len(self.job_registry),
            "system_health": await self._get_system_health()
        }

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available / 1024 / 1024 / 1024,  # GB
                "disk_usage": disk.percent,
                "disk_free": disk.free / 1024 / 1024 / 1024,  # GB
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                "healthy": cpu_percent < 90 and memory.percent < 90 and disk.percent < 90
            }
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {"healthy": False, "error": str(e)}

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job"""

        job = self.job_registry.get(job_id)
        if not job:
            return False

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()

        # Remove from active jobs if present
        for queue in self.queues.values():
            queue.active_jobs.pop(job_id, None)

            # Remove from pending jobs
            queue.pending_jobs = [j for j in queue.pending_jobs if j.job_id != job_id]
            heapq.heapify(queue.pending_jobs)

        logger.info(f"🚫 Job cancelled: {job_id}")
        return True

    async def pause_queue(self, queue_id: str) -> bool:
        """Pause a specific queue"""

        queue = self.queues.get(queue_id)
        if not queue:
            return False

        queue.is_paused = True
        logger.info(f"⏸️ Queue paused: {queue.name}")
        return True

    async def resume_queue(self, queue_id: str) -> bool:
        """Resume a specific queue"""

        queue = self.queues.get(queue_id)
        if not queue:
            return False

        queue.is_paused = False
        logger.info(f"▶️ Queue resumed: {queue.name}")
        return True

    async def get_user_jobs(
        self, 
        user_id: str, 
        status_filter: Optional[JobStatus] = None
    ) -> List[Dict[str, Any]]:
        """Get all jobs for a specific user"""

        user_jobs = [
            job for job in self.job_registry.values()
            if job.user_id == user_id
        ]

        if status_filter:
            user_jobs = [job for job in user_jobs if job.status == status_filter]

        # Sort by creation time (newest first)
        user_jobs.sort(key=lambda j: j.created_at, reverse=True)

        return [
            {
                "job_id": job.job_id,
                "job_type": job.job_type.value,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "estimated_duration": job.estimated_duration,
                "actual_duration": job.actual_duration
            }
            for job in user_jobs
        ]

    async def _health_monitor(self):
        """Background health monitoring"""

        while True:
            try:
                # Check system resources
                health = await self._get_system_health()

                if not health.get("healthy", False):
                    logger.warning("⚠️ System health degraded")

                    # Implement adaptive throttling
                    for queue in self.queues.values():
                        if health.get("cpu_usage", 0) > 95:
                            # Reduce concurrent jobs temporarily
                            queue.max_concurrent_jobs = max(1, queue.max_concurrent_jobs // 2)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Health monitor error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _performance_tracker(self):
        """Background performance tracking"""

        while True:
            try:
                # Update processing statistics
                total_jobs = self.processing_stats["total_jobs"]
                completed_jobs = self.processing_stats["completed_jobs"]
                failed_jobs = self.processing_stats["failed_jobs"]

                if total_jobs > 0:
                    self.processing_stats["error_rate"] = failed_jobs / total_jobs * 100

                # Calculate average processing time
                recent_jobs = [
                    job for job in self.job_registry.values()
                    if job.status == JobStatus.COMPLETED and job.actual_duration > 0
                ]

                if recent_jobs:
                    avg_time = sum(job.actual_duration for job in recent_jobs) / len(recent_jobs)
                    self.processing_stats["average_processing_time"] = avg_time

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Performance tracker error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _cleanup_completed_jobs(self):
        """Background cleanup of old completed jobs"""

        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)

                # Clean up old completed jobs
                jobs_to_remove = []
                for job_id, job in self.job_registry.items():
                    if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                        job.completed_at and job.completed_at < cutoff_time):
                        jobs_to_remove.append(job_id)

                for job_id in jobs_to_remove:
                    del self.job_registry[job_id]

                # Clean up queue histories
                for queue in self.queues.values():
                    queue.completed_jobs = [
                        job for job in queue.completed_jobs
                        if not job.completed_at or job.completed_at >= cutoff_time
                    ]
                    queue.failed_jobs = [
                        job for job in queue.failed_jobs
                        if not job.completed_at or job.completed_at >= cutoff_time
                    ]

                if jobs_to_remove:
                    logger.info(f"🧹 Cleaned up {len(jobs_to_remove)} old jobs")

                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                logger.error(f"Cleanup task error: {e}", exc_info=True)
                await asyncio.sleep(3600)

    async def initialize_distributed_cluster(self):
        """Initialize distributed processing cluster"""
        
        # Initialize distributed nodes
        distributed_nodes = [
            DistributedNode(
                node_id="node_primary_001",
                node_type="high_performance",
                location="primary_datacenter",
                max_concurrent_jobs=50,
                cpu_cores=32,
                memory_gb=128,
                gpu_available=True,
                gpu_memory_gb=24,
                capabilities=[JobType.VIDEO_ANALYSIS, JobType.AI_ENHANCEMENT, JobType.CAPTION_GENERATION],
                geographic_region="us-east-1"
            ),
            DistributedNode(
                node_id="node_worker_002",
                node_type="standard",
                location="secondary_datacenter",
                max_concurrent_jobs=30,
                cpu_cores=16,
                memory_gb=64,
                capabilities=[JobType.TEMPLATE_RENDERING, JobType.BATCH_EXPORT],
                geographic_region="us-west-2"
            ),
            DistributedNode(
                node_id="node_gpu_003",
                node_type="gpu_optimized",
                location="gpu_cluster",
                max_concurrent_jobs=20,
                cpu_cores=24,
                memory_gb=96,
                gpu_available=True,
                gpu_memory_gb=48,
                capabilities=[JobType.AI_ENHANCEMENT, JobType.VIRAL_OPTIMIZATION],
                geographic_region="us-central-1"
            )
        ]
        
        # Add nodes to cluster
        self.distributed_nodes = {node.node_id: node for node in distributed_nodes}
        
        # Start cluster monitoring
        await self._start_cluster_monitoring()
        
        logger.info(f"🌐 Distributed cluster initialized with {len(distributed_nodes)} nodes")

    async def _start_cluster_monitoring(self):
        """Start monitoring distributed cluster health"""
        
        cluster_monitor_task = asyncio.create_task(self._monitor_cluster_health())
        self.background_tasks.add(cluster_monitor_task)
        
        load_balancer_task = asyncio.create_task(self._dynamic_load_balancing())
        self.background_tasks.add(load_balancer_task)

    async def _monitor_cluster_health(self):
        """Monitor health of distributed nodes"""
        
        while True:
            try:
                for node_id, node in self.distributed_nodes.items():
                    # Simulate health check
                    health_status = await self._check_node_health(node)
                    
                    node.health_metrics.update(health_status)
                    node.last_heartbeat = datetime.utcnow()
                    
                    # Update node status based on health
                    if health_status["overall_health"] < 0.5:
                        node.is_active = False
                        logger.warning(f"⚠️ Node {node_id} marked as unhealthy")
                    else:
                        node.is_active = True
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cluster health monitoring error: {e}")
                await asyncio.sleep(60)

    async def _check_node_health(self, node: DistributedNode) -> Dict[str, float]:
        """Check health metrics for a distributed node"""
        
        import random
        
        # Simulate health metrics
        cpu_usage = random.uniform(0.2, 0.8)
        memory_usage = random.uniform(0.3, 0.7)
        network_latency = random.uniform(10, 100)  # ms
        
        # Calculate overall health score
        health_factors = [
            1.0 - cpu_usage,  # Lower CPU usage is better
            1.0 - memory_usage,  # Lower memory usage is better
            max(0, 1.0 - (network_latency / 100))  # Lower latency is better
        ]
        
        overall_health = sum(health_factors) / len(health_factors)
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "network_latency": network_latency,
            "disk_io": random.uniform(0.1, 0.6),
            "network_throughput": random.uniform(500, 1000),  # Mbps
            "overall_health": overall_health
        }

    async def _dynamic_load_balancing(self):
        """Dynamic load balancing across distributed nodes"""
        
        while True:
            try:
                # Calculate load balancer weights based on node performance
                for node in self.distributed_nodes.values():
                    if node.is_active:
                        # Calculate weight based on current load and health
                        current_load = node.current_jobs / node.max_concurrent_jobs
                        health_score = node.health_metrics.get("overall_health", 0.5)
                        
                        # Higher weight for nodes with lower load and better health
                        node.load_balancer_weight = (1.0 - current_load) * health_score
                    else:
                        node.load_balancer_weight = 0.0
                
                await asyncio.sleep(10)  # Update weights every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Load balancing error: {e}")
                await asyncio.sleep(30)

    async def distribute_job_to_optimal_node(self, job: BatchJob) -> Optional[str]:
        """Distribute job to optimal node based on capabilities and load"""
        
        # Filter nodes by job type capability
        capable_nodes = [
            node for node in self.distributed_nodes.values()
            if (job.job_type in node.capabilities and 
                node.is_active and 
                node.current_jobs < node.max_concurrent_jobs)
        ]
        
        if not capable_nodes:
            return None
        
        # Sort by load balancer weight (highest first)
        capable_nodes.sort(key=lambda n: n.load_balancer_weight, reverse=True)
        
        # Select optimal node
        optimal_node = capable_nodes[0]
        
        # Update node job count
        optimal_node.current_jobs += 1
        
        logger.info(f"📡 Job {job.job_id} distributed to node {optimal_node.node_id}")
        
        return optimal_node.node_id

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive distributed cluster status"""
        
        cluster_stats = {
            "total_nodes": len(self.distributed_nodes),
            "active_nodes": len([n for n in self.distributed_nodes.values() if n.is_active]),
            "total_capacity": sum(n.max_concurrent_jobs for n in self.distributed_nodes.values()),
            "current_utilization": sum(n.current_jobs for n in self.distributed_nodes.values()),
            "nodes": {}
        }
        
        for node_id, node in self.distributed_nodes.items():
            cluster_stats["nodes"][node_id] = {
                "node_type": node.node_type,
                "location": node.location,
                "is_active": node.is_active,
                "current_jobs": node.current_jobs,
                "max_jobs": node.max_concurrent_jobs,
                "utilization": (node.current_jobs / node.max_concurrent_jobs) * 100,
                "health_score": node.health_metrics.get("overall_health", 0.0) * 100,
                "load_balancer_weight": node.load_balancer_weight,
                "capabilities": [cap.value for cap in node.capabilities],
                "hardware": {
                    "cpu_cores": node.cpu_cores,
                    "memory_gb": node.memory_gb,
                    "gpu_available": node.gpu_available,
                    "gpu_memory_gb": node.gpu_memory_gb
                }
            }
        
        return cluster_stats

    async def graceful_shutdown(self):
        """Gracefully shutdown the batch processor"""

        logger.info("🔄 Starting batch processor shutdown...")

        # Pause all queues
        for queue in self.queues.values():
            queue.is_paused = True

        # Notify distributed nodes of shutdown
        if hasattr(self, 'distributed_nodes'):
            for node in self.distributed_nodes.values():
                node.is_active = False

        # Wait for active jobs to complete (with timeout)
        timeout = 300  # 5 minutes
        start_time = datetime.utcnow()

        while datetime.utcnow() - start_time < timedelta(seconds=timeout):
            active_count = sum(len(queue.active_jobs) for queue in self.queues.values())
            if active_count == 0:
                break

            logger.info(f"Waiting for {active_count} active jobs to complete...")
            await asyncio.sleep(5)

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        logger.info("✅ Batch processor shutdown complete")

    async def submit_job(
        self, 
        job_request: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        queue_id: Optional[str] = None,
        dependencies: List[str] = None,
        retry_config: Dict[str, Any] = None
    ) -> str:
        """Submit a new job to the batch processor"""

        try:
            job_id = str(uuid.uuid4())

            # Smart priority adjustment based on job characteristics
            adjusted_priority = await self._calculate_smart_priority(
                priority, job_request, queue_id
            )

            # Enhanced retry configuration
            retry_settings = self._get_retry_settings(job_request["job_type"], retry_config)

            # Create job with enhanced error handling and dependencies
            job = BatchJob(
                job_id=job_id,
                job_type=JobType(job_request["job_type"]),
                priority=adjusted_priority,
                payload=job_request.get("payload", {}),
                created_at=datetime.utcnow(),
                user_id=job_request.get("user_id"),
                session_id=job_request.get("session_id"),
                estimated_duration=job_request.get("estimated_duration", 300),
                retry_count=0,
                max_retries=retry_settings["max_retries"],
                dependencies=dependencies or [],
                timeout_seconds=retry_settings["timeout"],
                failure_threshold=retry_settings["failure_threshold"]
            )

            # Find target queue
            target_queue = self.queues.get(queue_id)
            if not target_queue:
                raise ValueError(f"Queue not found: {queue_id}")

            # Add to queue
            heapq.heappush(target_queue.pending_jobs, job)
            job.status = JobStatus.QUEUED

            self.processing_stats["total_jobs"] += 1

            logger.info(f"📝 Job submitted: {job_id} ({job.job_type.value}) - Priority: {priority.name}")

            return job_id

        except Exception as e:
            logger.error(f"Job submission failed: {e}", exc_info=True)
            raise