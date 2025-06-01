
"""
Netflix-Level Session Manager
Enterprise-grade session management with persistence
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SessionManager:
    """Netflix-level session management system"""
    
    def __init__(self, storage_path: str = "nr1copilot/nr1-main/cache/sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = 3600  # 1 hour
        
        logger.info("🗃️ Netflix-level SessionManager initialized")
    
    async def create_session(self, video_info: Dict[str, Any]) -> str:
        """Create a new processing session"""
        try:
            session_id = f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            
            session_data = {
                "session_id": session_id,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "status": "active",
                "video_info": video_info,
                "processing_stages": [],
                "results": {},
                "metadata": {}
            }
            
            # Store in memory
            self.active_sessions[session_id] = session_data
            
            # Persist to disk
            await self._persist_session(session_id, session_data)
            
            logger.info(f"📝 Created session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        try:
            # Check memory first
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session["last_accessed"] = time.time()
                return session
            
            # Try to load from disk
            session = await self._load_session(session_id)
            if session:
                session["last_accessed"] = time.time()
                self.active_sessions[session_id] = session
                return session
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {str(e)}")
            return None
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data"""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.warning(f"Session not found for update: {session_id}")
                return False
            
            # Apply updates
            for key, value in updates.items():
                if key in ["session_id", "created_at"]:
                    continue  # Don't allow updating these fields
                session[key] = value
            
            session["last_accessed"] = time.time()
            
            # Persist changes
            await self._persist_session(session_id, session)
            
            logger.debug(f"Updated session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {str(e)}")
            return False
    
    async def add_processing_stage(self, session_id: str, stage: str, data: Dict[str, Any]) -> bool:
        """Add a processing stage to the session"""
        try:
            stage_entry = {
                "stage": stage,
                "timestamp": time.time(),
                "data": data
            }
            
            return await self.update_session(session_id, {
                "processing_stages": (await self.get_session(session_id))["processing_stages"] + [stage_entry]
            })
            
        except Exception as e:
            logger.error(f"Failed to add processing stage for session {session_id}: {str(e)}")
            return False
    
    async def set_session_results(self, session_id: str, results: Dict[str, Any]) -> bool:
        """Set session results"""
        try:
            return await self.update_session(session_id, {
                "results": results,
                "status": "completed",
                "completed_at": time.time()
            })
            
        except Exception as e:
            logger.error(f"Failed to set results for session {session_id}: {str(e)}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        try:
            # Remove from memory
            self.active_sessions.pop(session_id, None)
            
            # Remove from disk
            session_file = self.storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            logger.info(f"🗑️ Deleted session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {str(e)}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        try:
            current_time = time.time()
            expired_sessions = []
            
            # Check active sessions
            for session_id, session in self.active_sessions.items():
                last_accessed = session.get("last_accessed", 0)
                if current_time - last_accessed > self.session_timeout:
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                await self.delete_session(session_id)
            
            logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            return 0
    
    async def list_sessions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List sessions with optional status filter"""
        try:
            sessions = []
            
            # Get from memory
            for session in self.active_sessions.values():
                if status is None or session.get("status") == status:
                    sessions.append(session.copy())
            
            # Load from disk if not in memory
            for session_file in self.storage_path.glob("*.json"):
                session_id = session_file.stem
                if session_id not in self.active_sessions:
                    session = await self._load_session(session_id)
                    if session and (status is None or session.get("status") == status):
                        sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {str(e)}")
            return []
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        try:
            all_sessions = await self.list_sessions()
            
            stats = {
                "total_sessions": len(all_sessions),
                "active_sessions": len([s for s in all_sessions if s.get("status") == "active"]),
                "completed_sessions": len([s for s in all_sessions if s.get("status") == "completed"]),
                "failed_sessions": len([s for s in all_sessions if s.get("status") == "failed"]),
                "memory_sessions": len(self.active_sessions),
                "average_session_duration": 0
            }
            
            # Calculate average session duration for completed sessions
            completed_sessions = [s for s in all_sessions if s.get("completed_at")]
            if completed_sessions:
                durations = [
                    s["completed_at"] - s["created_at"] 
                    for s in completed_sessions 
                    if "completed_at" in s and "created_at" in s
                ]
                if durations:
                    stats["average_session_duration"] = sum(durations) / len(durations)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {str(e)}")
            return {}
    
    async def _persist_session(self, session_id: str, session_data: Dict[str, Any]):
        """Persist session to disk"""
        try:
            session_file = self.storage_path / f"{session_id}.json"
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to persist session {session_id}: {str(e)}")
    
    async def _load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session from disk"""
        try:
            session_file = self.storage_path / f"{session_id}.json"
            
            if not session_file.exists():
                return None
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {str(e)}")
            return None
    
    async def export_session(self, session_id: str, export_path: str) -> bool:
        """Export session data"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w') as f:
                json.dump(session, f, indent=2, default=str)
            
            logger.info(f"📤 Exported session {session_id} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export session {session_id}: {str(e)}")
            return False
    
    async def import_session(self, import_path: str) -> Optional[str]:
        """Import session data"""
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                return None
            
            with open(import_file, 'r') as f:
                session_data = json.load(f)
            
            # Generate new session ID
            new_session_id = f"imported_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            session_data["session_id"] = new_session_id
            session_data["imported_at"] = time.time()
            
            # Store session
            self.active_sessions[new_session_id] = session_data
            await self._persist_session(new_session_id, session_data)
            
            logger.info(f"📥 Imported session as {new_session_id}")
            return new_session_id
            
        except Exception as e:
            logger.error(f"Failed to import session from {import_path}: {str(e)}")
            return None
