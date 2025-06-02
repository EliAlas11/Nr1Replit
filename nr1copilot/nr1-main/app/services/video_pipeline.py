
"""
Netflix-Level Video Processing Pipeline
Complete end-to-end pipeline: Upload ➜ Processing ➜ Ready-to-Serve
"""

import asyncio
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

from .video_service import NetflixLevelVideoService, UploadSession, UploadStatus
from .ffmpeg_processor import (
    NetflixLevelFFmpegProcessor, 
    EncodingSettings, 
    VideoQuality, 
    DeviceProfile,
    ProcessingJob
)
from .cloud_processor import CloudVideoProcessor

logger = logging.getLogger(__name__)


class PipelineStage:
    """Represents a stage in the video processing pipeline"""
    
    def __init__(self, name: str, processor_func, required: bool = True):
        self.name = name
        self.processor_func = processor_func
        self.required = required
        self.status = "pending"
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.output_data: Dict[str, Any] = {}
    
    @property
    def duration(self) -> float:
        """Get stage duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "status": self.status,
            "required": self.required,
            "duration": self.duration,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message,
            "output_data": self.output_data
        }


class VideoPipeline:
    """Complete video processing pipeline with stages"""
    
    def __init__(self, pipeline_id: str, upload_session_id: str, user_id: str = "anonymous"):
        self.pipeline_id = pipeline_id
        self.upload_session_id = upload_session_id
        self.user_id = user_id
        self.status = "initializing"
        self.current_stage_index = 0
        self.stages: List[PipelineStage] = []
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.input_file_path: Optional[str] = None
        self.output_file_path: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.error_count = 0
        self.retry_count = 0
        self.max_retries = 2
    
    def add_stage(self, stage: PipelineStage):
        """Add a processing stage"""
        self.stages.append(stage)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage"""
        if not self.stages:
            return 0.0
        
        completed_stages = sum(1 for stage in self.stages if stage.status == "completed")
        
        # Add partial progress for current stage
        if (self.current_stage_index < len(self.stages) and 
            self.stages[self.current_stage_index].status == "processing"):
            # Assume current stage is 50% complete
            completed_stages += 0.5
        
        return (completed_stages / len(self.stages)) * 100
    
    @property
    def total_duration(self) -> float:
        """Get total pipeline duration"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.utcnow() - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "pipeline_id": self.pipeline_id,
            "upload_session_id": self.upload_session_id,
            "user_id": self.user_id,
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "current_stage_index": self.current_stage_index,
            "total_stages": len(self.stages),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": self.total_duration,
            "input_file_path": self.input_file_path,
            "output_file_path": self.output_file_path,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": self.metadata
        }


class NetflixLevelVideoPipeline:
    """Netflix-level video processing pipeline orchestrator"""
    
    def __init__(self):
        self.video_service = NetflixLevelVideoService()
        self.ffmpeg_processor = NetflixLevelFFmpegProcessor(max_workers=6)
        self.cloud_processor = CloudVideoProcessor()
        
        self.active_pipelines: Dict[str, VideoPipeline] = {}
        self.completed_pipelines: Dict[str, VideoPipeline] = {}
        self.failed_pipelines: Dict[str, VideoPipeline] = {}
        
        self.pipeline_lock = asyncio.Lock()
        self.processing_task: Optional[asyncio.Task] = None
        
        # Pipeline statistics
        self.stats = {
            "total_pipelines": 0,
            "successful_pipelines": 0,
            "failed_pipelines": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
            "active_pipelines": 0
        }
    
    async def startup(self):
        """Initialize the video pipeline system"""
        logger.info("Starting Netflix-level video pipeline...")
        
        try:
            # Initialize all components
            await self.video_service.startup()
            await self.ffmpeg_processor.startup()
            
            # Start pipeline processing task
            self.processing_task = asyncio.create_task(self._pipeline_processor())
            
            logger.info("Video pipeline system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start video pipeline: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the pipeline system"""
        logger.info("Shutting down video pipeline...")
        
        try:
            # Cancel processing task
            if self.processing_task:
                self.processing_task.cancel()
            
            # Shutdown components
            await self.video_service.shutdown()
            await self.ffmpeg_processor.shutdown()
            
            logger.info("Video pipeline shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during pipeline shutdown: {e}")
    
    async def create_pipeline_from_upload(
        self,
        upload_session_id: str,
        processing_options: Dict[str, Any] = None
    ) -> str:
        """Create processing pipeline from completed upload"""
        
        try:
            # Get upload session
            upload_status = await self.video_service.get_upload_status(upload_session_id)
            
            if not upload_status or upload_status["status"] != "completed":
                raise ValueError(f"Upload session not ready: {upload_session_id}")
            
            # Create pipeline
            pipeline_id = f"pipeline_{uuid.uuid4().hex[:12]}"
            pipeline = VideoPipeline(
                pipeline_id=pipeline_id,
                upload_session_id=upload_session_id,
                user_id=upload_status.get("user_id", "anonymous")
            )
            
            # Find the uploaded file
            session = None
            for sess in self.video_service.active_sessions.values():
                if sess.id == upload_session_id:
                    session = sess
                    break
            
            if not session or not session.file_path:
                raise ValueError("Upload file not found")
            
            pipeline.input_file_path = str(session.file_path)
            pipeline.metadata.update({
                "filename": session.filename,
                "file_size": session.file_size,
                "mime_type": session.mime_type,
                "upload_metadata": session.metadata
            })
            
            # Configure processing stages
            await self._configure_pipeline_stages(pipeline, processing_options or {})
            
            # Store pipeline
            async with self.pipeline_lock:
                self.active_pipelines[pipeline_id] = pipeline
                self.stats["total_pipelines"] += 1
                self.stats["active_pipelines"] = len(self.active_pipelines)
            
            logger.info(f"Processing pipeline created: {pipeline_id}")
            return pipeline_id
            
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise
    
    async def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline status"""
        # Check active pipelines
        if pipeline_id in self.active_pipelines:
            return self.active_pipelines[pipeline_id].to_dict()
        
        # Check completed pipelines
        if pipeline_id in self.completed_pipelines:
            return self.completed_pipelines[pipeline_id].to_dict()
        
        # Check failed pipelines
        if pipeline_id in self.failed_pipelines:
            return self.failed_pipelines[pipeline_id].to_dict()
        
        return None
    
    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel an active pipeline"""
        try:
            async with self.pipeline_lock:
                if pipeline_id in self.active_pipelines:
                    pipeline = self.active_pipelines[pipeline_id]
                    pipeline.status = "cancelled"
                    
                    # Cancel any active FFmpeg jobs
                    # Note: In a real implementation, track FFmpeg job IDs
                    
                    logger.info(f"Pipeline cancelled: {pipeline_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel pipeline {pipeline_id}: {e}")
            return False
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            **self.stats,
            "active_pipelines": len(self.active_pipelines),
            "completed_pipelines": len(self.completed_pipelines),
            "failed_pipelines": len(self.failed_pipelines),
            "ffmpeg_stats": await self.ffmpeg_processor.get_queue_stats()
        }
    
    # Private methods
    
    async def _configure_pipeline_stages(
        self, 
        pipeline: VideoPipeline, 
        options: Dict[str, Any]
    ):
        """Configure processing stages for the pipeline"""
        
        # Stage 1: File Validation
        pipeline.add_stage(PipelineStage(
            name="validation",
            processor_func=self._stage_validate_file,
            required=True
        ))
        
        # Stage 2: Content Analysis
        if options.get("enable_content_analysis", True):
            pipeline.add_stage(PipelineStage(
                name="content_analysis",
                processor_func=self._stage_content_analysis,
                required=False
            ))
        
        # Stage 3: Video Processing (FFmpeg)
        pipeline.add_stage(PipelineStage(
            name="video_processing",
            processor_func=self._stage_video_processing,
            required=True
        ))
        
        # Stage 4: Quality Assurance
        pipeline.add_stage(PipelineStage(
            name="quality_assurance",
            processor_func=self._stage_quality_assurance,
            required=True
        ))
        
        # Stage 5: Thumbnail Generation
        if options.get("generate_thumbnails", True):
            pipeline.add_stage(PipelineStage(
                name="thumbnail_generation",
                processor_func=self._stage_thumbnail_generation,
                required=False
            ))
        
        # Stage 6: Metadata Enrichment
        pipeline.add_stage(PipelineStage(
            name="metadata_enrichment",
            processor_func=self._stage_metadata_enrichment,
            required=True
        ))
        
        # Stage 7: Finalization
        pipeline.add_stage(PipelineStage(
            name="finalization",
            processor_func=self._stage_finalization,
            required=True
        ))
    
    async def _pipeline_processor(self):
        """Background task to process pipelines"""
        logger.info("Pipeline processor started")
        
        while True:
            try:
                # Get pipelines ready for processing
                ready_pipelines = []
                
                async with self.pipeline_lock:
                    for pipeline in self.active_pipelines.values():
                        if pipeline.status in ["initializing", "processing"]:
                            ready_pipelines.append(pipeline)
                
                # Process each pipeline
                for pipeline in ready_pipelines:
                    if pipeline.status == "cancelled":
                        continue
                    
                    try:
                        await self._process_pipeline_stage(pipeline)
                    except Exception as e:
                        logger.error(f"Pipeline stage error {pipeline.pipeline_id}: {e}")
                        await self._handle_pipeline_error(pipeline, str(e))
                
                # Clean up completed pipelines
                await self._cleanup_completed_pipelines()
                
                # Brief pause
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline processor error: {e}")
                await asyncio.sleep(10)
    
    async def _process_pipeline_stage(self, pipeline: VideoPipeline):
        """Process pipeline stage with production-grade error handling"""
        
        if pipeline.status == "cancelled":
            return
            
        if pipeline.current_stage_index >= len(pipeline.stages):
            # Pipeline completed
            await self._complete_pipeline(pipeline)
            return
        
        current_stage = pipeline.stages[pipeline.current_stage_index]
        
        if current_stage.status == "pending":
            # Start stage with timeout protection
            current_stage.status = "processing"
            current_stage.start_time = datetime.utcnow()
            pipeline.status = "processing"
            
            logger.info(f"🚀 Starting stage {current_stage.name} for pipeline {pipeline.pipeline_id}")
            
            try:
                # Execute stage processor with timeout
                stage_timeout = 1800  # 30 minutes per stage max
                result = await asyncio.wait_for(
                    current_stage.processor_func(pipeline, current_stage),
                    timeout=stage_timeout
                )
                
                if result.get("success", False):
                    current_stage.status = "completed"
                    current_stage.output_data = result.get("data", {})
                    pipeline.current_stage_index += 1
                    
                    # Log performance metrics
                    stage_duration = current_stage.duration
                    logger.info(
                        f"✅ Stage {current_stage.name} completed for pipeline {pipeline.pipeline_id} "
                        f"in {stage_duration:.2f}s"
                    )
                else:
                    error_msg = result.get("error", "Stage processing failed")
                    raise Exception(error_msg)
                
            except asyncio.TimeoutError:
                error_msg = f"Stage {current_stage.name} timed out after {stage_timeout}s"
                logger.error(error_msg)
                current_stage.status = "failed"
                current_stage.error_message = error_msg
                pipeline.error_count += 1
                
                if current_stage.required:
                    await self._handle_pipeline_error(pipeline, error_msg)
                else:
                    logger.warning(f"Optional stage {current_stage.name} timed out, continuing")
                    pipeline.current_stage_index += 1
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Stage {current_stage.name} failed for pipeline {pipeline.pipeline_id}: {error_msg}")
                
                current_stage.status = "failed"
                current_stage.error_message = error_msg
                pipeline.error_count += 1
                
                if current_stage.required:
                    await self._handle_pipeline_error(pipeline, error_msg)
                else:
                    # Skip optional stage with warning
                    logger.warning(f"⚠️ Optional stage {current_stage.name} failed, continuing: {error_msg}")
                    pipeline.current_stage_index += 1
            
            finally:
                current_stage.end_time = datetime.utcnow()
                
                # Force garbage collection after heavy processing stages
                if current_stage.name in ["video_processing", "content_analysis"]:
                    import gc
                    gc.collect()
    
    async def _complete_pipeline(self, pipeline: VideoPipeline):
        """Complete a pipeline"""
        pipeline.status = "completed"
        pipeline.end_time = datetime.utcnow()
        
        async with self.pipeline_lock:
            # Move to completed pipelines
            if pipeline.pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline.pipeline_id]
                self.completed_pipelines[pipeline.pipeline_id] = pipeline
                
                # Update stats
                self.stats["successful_pipelines"] += 1
                self.stats["total_processing_time"] += pipeline.total_duration
                self.stats["active_pipelines"] = len(self.active_pipelines)
                
                if self.stats["successful_pipelines"] > 0:
                    self.stats["average_processing_time"] = (
                        self.stats["total_processing_time"] / self.stats["successful_pipelines"]
                    )
        
        logger.info(f"Pipeline completed: {pipeline.pipeline_id} in {pipeline.total_duration:.2f}s")
    
    async def _handle_pipeline_error(self, pipeline: VideoPipeline, error: str):
        """Handle pipeline error with retry logic"""
        pipeline.status = "failed"
        pipeline.end_time = datetime.utcnow()
        
        if pipeline.retry_count < pipeline.max_retries:
            # Retry pipeline
            pipeline.retry_count += 1
            pipeline.status = "retrying"
            pipeline.current_stage_index = 0  # Restart from beginning
            
            # Reset all stages
            for stage in pipeline.stages:
                stage.status = "pending"
                stage.start_time = None
                stage.end_time = None
                stage.error_message = None
                stage.output_data = {}
            
            logger.info(f"Retrying pipeline {pipeline.pipeline_id} (attempt {pipeline.retry_count})")
        else:
            # Permanently failed
            async with self.pipeline_lock:
                if pipeline.pipeline_id in self.active_pipelines:
                    del self.active_pipelines[pipeline.pipeline_id]
                    self.failed_pipelines[pipeline.pipeline_id] = pipeline
                    
                    self.stats["failed_pipelines"] += 1
                    self.stats["active_pipelines"] = len(self.active_pipelines)
            
            logger.error(f"Pipeline permanently failed: {pipeline.pipeline_id}: {error}")
    
    async def _cleanup_completed_pipelines(self):
        """Clean up old completed pipelines"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)  # Keep for 24 hours
            
            expired_pipelines = []
            for pipeline_id, pipeline in self.completed_pipelines.items():
                if pipeline.end_time and pipeline.end_time < cutoff_time:
                    expired_pipelines.append(pipeline_id)
            
            for pipeline_id in expired_pipelines:
                del self.completed_pipelines[pipeline_id]
                logger.debug(f"Cleaned up expired pipeline: {pipeline_id}")
                
        except Exception as e:
            logger.error(f"Pipeline cleanup error: {e}")
    
    # Stage processor methods
    
    async def _stage_validate_file(
        self, 
        pipeline: VideoPipeline, 
        stage: PipelineStage
    ) -> Dict[str, Any]:
        """Validate input file"""
        try:
            if not pipeline.input_file_path or not os.path.exists(pipeline.input_file_path):
                return {"success": False, "error": "Input file not found"}
            
            # Basic validation
            file_size = os.path.getsize(pipeline.input_file_path)
            if file_size == 0:
                return {"success": False, "error": "Input file is empty"}
            
            return {
                "success": True,
                "data": {
                    "file_size": file_size,
                    "file_exists": True
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _stage_content_analysis(
        self, 
        pipeline: VideoPipeline, 
        stage: PipelineStage
    ) -> Dict[str, Any]:
        """Analyze video content"""
        try:
            # Use cloud processor for content analysis
            analysis = await self.cloud_processor.video_processor.analyze_video_content(
                pipeline.input_file_path
            )
            
            return {
                "success": True,
                "data": {"content_analysis": analysis}
            }
            
        except Exception as e:
            logger.error(f"Content analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _stage_video_processing(
        self, 
        pipeline: VideoPipeline, 
        stage: PipelineStage
    ) -> Dict[str, Any]:
        """Process video with FFmpeg"""
        try:
            # Generate output path
            input_path = Path(pipeline.input_file_path)
            output_dir = Path("output") / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{pipeline.pipeline_id}_{input_path.stem}_processed.mp4"
            pipeline.output_file_path = str(output_path)
            
            # Auto-detect device profile (simplified)
            device_profile = DeviceProfile.get_mobile_profile()
            
            # Submit to FFmpeg processor
            job_id = await self.ffmpeg_processor.process_video(
                input_path=pipeline.input_file_path,
                output_path=str(output_path),
                device_profile=device_profile,
                user_id=pipeline.user_id,
                priority=7  # High priority for pipeline jobs
            )
            
            # Wait for completion (simplified - in production, use async monitoring)
            max_wait_time = 300  # 5 minutes
            wait_interval = 5
            waited_time = 0
            
            while waited_time < max_wait_time:
                job_status = await self.ffmpeg_processor.get_job_status(job_id)
                
                if not job_status:
                    break
                
                if job_status["status"] == "completed":
                    return {
                        "success": True,
                        "data": {
                            "ffmpeg_job_id": job_id,
                            "output_path": str(output_path),
                            "processing_time": job_status.get("actual_duration", 0)
                        }
                    }
                elif job_status["status"] == "failed":
                    return {
                        "success": False,
                        "error": f"FFmpeg processing failed: {job_status.get('error_message', 'Unknown error')}"
                    }
                
                # Update stage progress
                stage.output_data["ffmpeg_progress"] = job_status.get("progress", 0)
                
                await asyncio.sleep(wait_interval)
                waited_time += wait_interval
            
            return {"success": False, "error": "FFmpeg processing timeout"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _stage_quality_assurance(
        self, 
        pipeline: VideoPipeline, 
        stage: PipelineStage
    ) -> Dict[str, Any]:
        """Perform quality assurance checks"""
        try:
            if not pipeline.output_file_path or not os.path.exists(pipeline.output_file_path):
                return {"success": False, "error": "Processed file not found"}
            
            # Basic QA checks
            output_size = os.path.getsize(pipeline.output_file_path)
            if output_size == 0:
                return {"success": False, "error": "Processed file is empty"}
            
            # Use cloud processor for QA
            qa_result = await self.cloud_processor._perform_quality_assurance(
                pipeline.output_file_path
            )
            
            return {
                "success": qa_result.get("passed", False),
                "data": {"qa_result": qa_result},
                "error": None if qa_result.get("passed") else "QA checks failed"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _stage_thumbnail_generation(
        self, 
        pipeline: VideoPipeline, 
        stage: PipelineStage
    ) -> Dict[str, Any]:
        """Generate video thumbnails"""
        try:
            if not pipeline.output_file_path:
                return {"success": False, "error": "No output file for thumbnail generation"}
            
            # Use video service thumbnail generation
            thumbnail_path = self.video_service.video_processor._generate_thumbnail(
                pipeline.output_file_path
            )
            
            return {
                "success": True,
                "data": {"thumbnail_path": thumbnail_path}
            }
            
        except Exception as e:
            logger.error(f"Thumbnail generation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _stage_metadata_enrichment(
        self, 
        pipeline: VideoPipeline, 
        stage: PipelineStage
    ) -> Dict[str, Any]:
        """Enrich video metadata"""
        try:
            # Collect metadata from all stages
            enriched_metadata = {
                "pipeline_id": pipeline.pipeline_id,
                "processing_time": pipeline.total_duration,
                "stages_completed": pipeline.current_stage_index,
                "error_count": pipeline.error_count,
                "retry_count": pipeline.retry_count
            }
            
            # Add stage outputs
            for i, stage in enumerate(pipeline.stages[:pipeline.current_stage_index]):
                enriched_metadata[f"stage_{i}_{stage.name}"] = stage.output_data
            
            pipeline.metadata.update(enriched_metadata)
            
            return {
                "success": True,
                "data": {"enriched_metadata": enriched_metadata}
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _stage_finalization(
        self, 
        pipeline: VideoPipeline, 
        stage: PipelineStage
    ) -> Dict[str, Any]:
        """Finalize processing and prepare for serving"""
        try:
            # Final validation
            if not pipeline.output_file_path or not os.path.exists(pipeline.output_file_path):
                return {"success": False, "error": "Final output file missing"}
            
            # Create final metadata file
            metadata_path = f"{pipeline.output_file_path}.metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(pipeline.metadata, f, indent=2, default=str)
            
            return {
                "success": True,
                "data": {
                    "final_output_path": pipeline.output_file_path,
                    "metadata_path": metadata_path,
                    "ready_to_serve": True
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Export main class
__all__ = ["NetflixLevelVideoPipeline", "VideoPipeline", "PipelineStage"]
