
"""
ViralClip Pro v8.0 - Netflix-Level Team Collaboration Engine
Enterprise collaboration with real-time editing, permissions, and workflows
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path

import psutil
from fastapi import WebSocket, HTTPException

from ..config import settings
from ..utils.cache import cache_manager

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles with hierarchical permissions"""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"
    GUEST = "guest"


class WorkspacePermission(str, Enum):
    """Workspace permissions"""
    CREATE_PROJECT = "create_project"
    EDIT_PROJECT = "edit_project"
    DELETE_PROJECT = "delete_project"
    MANAGE_USERS = "manage_users"
    EXPORT_PROJECT = "export_project"
    APPROVE_CONTENT = "approve_content"
    VIEW_ANALYTICS = "view_analytics"
    COMMENT = "comment"
    VIEW_ONLY = "view_only"


class ApprovalStatus(str, Enum):
    """Content approval status"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"


@dataclass
class TeamMember:
    """Team member with role and permissions"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: Set[WorkspacePermission] = field(default_factory=set)
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    is_online: bool = False
    current_project: Optional[str] = None
    avatar_url: Optional[str] = None
    
    def __post_init__(self):
        self.permissions = self._get_default_permissions()
    
    def _get_default_permissions(self) -> Set[WorkspacePermission]:
        """Get default permissions based on role"""
        role_permissions = {
            UserRole.OWNER: {
                WorkspacePermission.CREATE_PROJECT,
                WorkspacePermission.EDIT_PROJECT,
                WorkspacePermission.DELETE_PROJECT,
                WorkspacePermission.MANAGE_USERS,
                WorkspacePermission.EXPORT_PROJECT,
                WorkspacePermission.APPROVE_CONTENT,
                WorkspacePermission.VIEW_ANALYTICS,
                WorkspacePermission.COMMENT
            },
            UserRole.ADMIN: {
                WorkspacePermission.CREATE_PROJECT,
                WorkspacePermission.EDIT_PROJECT,
                WorkspacePermission.MANAGE_USERS,
                WorkspacePermission.EXPORT_PROJECT,
                WorkspacePermission.APPROVE_CONTENT,
                WorkspacePermission.VIEW_ANALYTICS,
                WorkspacePermission.COMMENT
            },
            UserRole.EDITOR: {
                WorkspacePermission.EDIT_PROJECT,
                WorkspacePermission.EXPORT_PROJECT,
                WorkspacePermission.COMMENT
            },
            UserRole.REVIEWER: {
                WorkspacePermission.APPROVE_CONTENT,
                WorkspacePermission.COMMENT,
                WorkspacePermission.VIEW_ANALYTICS
            },
            UserRole.VIEWER: {
                WorkspacePermission.COMMENT,
                WorkspacePermission.VIEW_ONLY
            },
            UserRole.GUEST: {
                WorkspacePermission.VIEW_ONLY
            }
        }
        return role_permissions.get(self.role, set())


@dataclass
class Comment:
    """Timestamped comment with mentions and replies"""
    id: str
    author_id: str
    author_name: str
    content: str
    timestamp: float
    project_id: str
    mentions: List[str] = field(default_factory=list)
    replies: List['Comment'] = field(default_factory=list)
    resolved: bool = False
    priority: str = "normal"  # low, normal, high, urgent
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "content": self.content,
            "timestamp": self.timestamp,
            "project_id": self.project_id,
            "mentions": self.mentions,
            "replies": [reply.to_dict() for reply in self.replies],
            "resolved": self.resolved,
            "priority": self.priority,
            "tags": self.tags,
            "attachments": self.attachments
        }


@dataclass
class ProjectVersion:
    """Project version with complete state"""
    version_id: str
    project_id: str
    author_id: str
    author_name: str
    timestamp: datetime
    changes: Dict[str, Any]
    message: str
    tags: List[str] = field(default_factory=list)
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "project_id": self.project_id,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "timestamp": self.timestamp.isoformat(),
            "changes": self.changes,
            "message": self.message,
            "tags": self.tags,
            "parent_version": self.parent_version
        }


@dataclass
class SharedLink:
    """Secure shared link with expiration"""
    link_id: str
    project_id: str
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    password: Optional[str] = None
    view_count: int = 0
    max_views: Optional[int] = None
    permissions: Set[str] = field(default_factory=set)
    custom_domain: Optional[str] = None
    branding_enabled: bool = True
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        if self.max_views and self.view_count >= self.max_views:
            return True
        return False


@dataclass
class ApprovalWorkflow:
    """Content approval workflow"""
    workflow_id: str
    project_id: str
    status: ApprovalStatus
    created_by: str
    created_at: datetime
    reviewers: List[str] = field(default_factory=list)
    approvals: Dict[str, bool] = field(default_factory=dict)
    comments: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    priority: str = "normal"
    
    @property
    def approval_progress(self) -> float:
        if not self.reviewers:
            return 0.0
        approved = sum(1 for approved in self.approvals.values() if approved)
        return approved / len(self.reviewers)
    
    @property
    def is_approved(self) -> bool:
        return all(self.approvals.get(reviewer, False) for reviewer in self.reviewers)


class NetflixLevelCollaborationEngine:
    """Netflix-level team collaboration engine"""
    
    def __init__(self):
        self.workspaces: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.user_presence: Dict[str, Dict[str, Any]] = {}
        self.project_locks: Dict[str, Dict[str, Any]] = {}
        self.real_time_operations: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance tracking
        self.metrics = {
            "active_collaborators": 0,
            "real_time_operations": 0,
            "comments_created": 0,
            "versions_created": 0,
            "approvals_completed": 0,
            "shared_links_created": 0
        }
        
    async def create_workspace(
        self,
        workspace_id: str,
        name: str,
        owner_id: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """Create new team workspace"""
        try:
            workspace = {
                "id": workspace_id,
                "name": name,
                "description": description,
                "owner_id": owner_id,
                "created_at": datetime.utcnow(),
                "members": {},
                "projects": {},
                "shared_assets": {},
                "settings": {
                    "real_time_collaboration": True,
                    "comment_notifications": True,
                    "approval_required": False,
                    "version_retention_days": 30,
                    "max_members": 100
                }
            }
            
            # Add owner as first member
            owner = TeamMember(
                user_id=owner_id,
                username=f"user_{owner_id}",
                email=f"{owner_id}@example.com",
                role=UserRole.OWNER
            )
            workspace["members"][owner_id] = owner
            
            self.workspaces[workspace_id] = workspace
            
            # Cache workspace
            await cache_manager.set(
                f"workspace:{workspace_id}",
                workspace,
                ttl=86400,
                tags=["workspace", "collaboration"]
            )
            
            logger.info(f"✅ Workspace created: {workspace_id}")
            
            return {
                "success": True,
                "workspace_id": workspace_id,
                "workspace": workspace
            }
            
        except Exception as e:
            logger.error(f"❌ Workspace creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def add_team_member(
        self,
        workspace_id: str,
        user_id: str,
        inviter_id: str,
        role: UserRole,
        email: str,
        username: str = None
    ) -> Dict[str, Any]:
        """Add team member with role and permissions"""
        try:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")
            
            # Check permissions
            inviter = workspace["members"].get(inviter_id)
            if not inviter or WorkspacePermission.MANAGE_USERS not in inviter.permissions:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            # Create member
            member = TeamMember(
                user_id=user_id,
                username=username or f"user_{user_id}",
                email=email,
                role=role
            )
            
            workspace["members"][user_id] = member
            
            # Update cache
            await cache_manager.set(
                f"workspace:{workspace_id}",
                workspace,
                ttl=86400,
                tags=["workspace", "collaboration"]
            )
            
            # Notify other members
            await self._broadcast_to_workspace(workspace_id, {
                "type": "member_added",
                "member": member.__dict__,
                "added_by": inviter_id
            })
            
            self.metrics["active_collaborators"] += 1
            
            return {
                "success": True,
                "member": member.__dict__
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to add team member: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def start_collaboration_session(
        self,
        workspace_id: str,
        project_id: str,
        user_id: str,
        websocket: WebSocket
    ) -> Dict[str, Any]:
        """Start real-time collaboration session"""
        try:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")
            
            member = workspace["members"].get(user_id)
            if not member:
                raise HTTPException(status_code=403, detail="User not in workspace")
            
            session_id = f"session_{uuid.uuid4().hex[:12]}"
            
            # Create session
            session = {
                "session_id": session_id,
                "workspace_id": workspace_id,
                "project_id": project_id,
                "user_id": user_id,
                "username": member.username,
                "started_at": datetime.utcnow(),
                "websocket": websocket,
                "cursor_position": None,
                "active_edits": []
            }
            
            self.active_sessions[session_id] = session
            self.websocket_connections[user_id] = websocket
            
            # Update user presence
            self.user_presence[user_id] = {
                "workspace_id": workspace_id,
                "project_id": project_id,
                "status": "active",
                "last_seen": datetime.utcnow(),
                "session_id": session_id
            }
            
            # Notify other collaborators
            await self._broadcast_to_project(workspace_id, project_id, {
                "type": "user_joined",
                "user_id": user_id,
                "username": member.username,
                "session_id": session_id
            }, exclude_user=user_id)
            
            # Send current project state
            project_state = await self._get_project_state(workspace_id, project_id)
            await websocket.send_text(json.dumps({
                "type": "project_state",
                "data": project_state
            }))
            
            return {
                "success": True,
                "session_id": session_id,
                "active_collaborators": await self._get_active_collaborators(workspace_id, project_id)
            }
            
        except Exception as e:
            logger.error(f"❌ Collaboration session failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def handle_real_time_operation(
        self,
        session_id: str,
        operation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle real-time collaborative operation"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            workspace_id = session["workspace_id"]
            project_id = session["project_id"]
            user_id = session["user_id"]
            
            # Add metadata to operation
            operation.update({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "username": session["username"],
                "timestamp": time.time(),
                "session_id": session_id
            })
            
            # Store operation
            if project_id not in self.real_time_operations:
                self.real_time_operations[project_id] = []
            
            self.real_time_operations[project_id].append(operation)
            
            # Broadcast to other collaborators
            await self._broadcast_to_project(workspace_id, project_id, {
                "type": "real_time_operation",
                "operation": operation
            }, exclude_user=user_id)
            
            self.metrics["real_time_operations"] += 1
            
            return {
                "success": True,
                "operation_id": operation["id"]
            }
            
        except Exception as e:
            logger.error(f"❌ Real-time operation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def add_comment(
        self,
        workspace_id: str,
        project_id: str,
        user_id: str,
        content: str,
        timestamp: float,
        mentions: List[str] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Add timestamped comment with mentions"""
        try:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")
            
            member = workspace["members"].get(user_id)
            if not member or WorkspacePermission.COMMENT not in member.permissions:
                raise HTTPException(status_code=403, detail="Comment permission required")
            
            comment_id = f"comment_{uuid.uuid4().hex[:12]}"
            
            comment = Comment(
                id=comment_id,
                author_id=user_id,
                author_name=member.username,
                content=content,
                timestamp=timestamp,
                project_id=project_id,
                mentions=mentions or [],
                priority=priority
            )
            
            # Store comment
            project_comments = workspace["projects"].get(project_id, {}).get("comments", [])
            project_comments.append(comment)
            
            if project_id not in workspace["projects"]:
                workspace["projects"][project_id] = {}
            workspace["projects"][project_id]["comments"] = project_comments
            
            # Update cache
            await cache_manager.set(
                f"workspace:{workspace_id}",
                workspace,
                ttl=86400,
                tags=["workspace", "collaboration"]
            )
            
            # Broadcast comment
            await self._broadcast_to_project(workspace_id, project_id, {
                "type": "comment_added",
                "comment": comment.to_dict()
            })
            
            # Send notifications for mentions
            for mentioned_user in mentions or []:
                await self._send_mention_notification(workspace_id, mentioned_user, comment)
            
            self.metrics["comments_created"] += 1
            
            return {
                "success": True,
                "comment_id": comment_id,
                "comment": comment.to_dict()
            }
            
        except Exception as e:
            logger.error(f"❌ Comment creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def create_project_version(
        self,
        workspace_id: str,
        project_id: str,
        user_id: str,
        changes: Dict[str, Any],
        message: str,
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Create project version for rollback support"""
        try:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")
            
            member = workspace["members"].get(user_id)
            if not member:
                raise HTTPException(status_code=403, detail="User not in workspace")
            
            version_id = f"version_{uuid.uuid4().hex[:12]}"
            
            # Get current version as parent
            project_versions = workspace["projects"].get(project_id, {}).get("versions", [])
            parent_version = project_versions[-1]["version_id"] if project_versions else None
            
            version = ProjectVersion(
                version_id=version_id,
                project_id=project_id,
                author_id=user_id,
                author_name=member.username,
                timestamp=datetime.utcnow(),
                changes=changes,
                message=message,
                tags=tags or [],
                parent_version=parent_version
            )
            
            # Store version
            project_versions.append(version)
            
            if project_id not in workspace["projects"]:
                workspace["projects"][project_id] = {}
            workspace["projects"][project_id]["versions"] = project_versions
            
            # Update cache
            await cache_manager.set(
                f"workspace:{workspace_id}",
                workspace,
                ttl=86400,
                tags=["workspace", "collaboration"]
            )
            
            # Broadcast version creation
            await self._broadcast_to_project(workspace_id, project_id, {
                "type": "version_created",
                "version": version.to_dict()
            })
            
            self.metrics["versions_created"] += 1
            
            return {
                "success": True,
                "version_id": version_id,
                "version": version.to_dict()
            }
            
        except Exception as e:
            logger.error(f"❌ Version creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def create_shared_link(
        self,
        workspace_id: str,
        project_id: str,
        user_id: str,
        expires_hours: Optional[int] = None,
        password: Optional[str] = None,
        max_views: Optional[int] = None,
        permissions: List[str] = None,
        branded: bool = True
    ) -> Dict[str, Any]:
        """Create secure shared link with expiration"""
        try:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")
            
            member = workspace["members"].get(user_id)
            if not member:
                raise HTTPException(status_code=403, detail="User not in workspace")
            
            link_id = f"link_{uuid.uuid4().hex[:16]}"
            
            expires_at = None
            if expires_hours:
                expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
            
            shared_link = SharedLink(
                link_id=link_id,
                project_id=project_id,
                created_by=user_id,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                password=password,
                max_views=max_views,
                permissions=set(permissions or ["view"]),
                branding_enabled=branded
            )
            
            # Store shared link
            project_links = workspace["projects"].get(project_id, {}).get("shared_links", [])
            project_links.append(shared_link)
            
            if project_id not in workspace["projects"]:
                workspace["projects"][project_id] = {}
            workspace["projects"][project_id]["shared_links"] = project_links
            
            # Update cache
            await cache_manager.set(
                f"workspace:{workspace_id}",
                workspace,
                ttl=86400,
                tags=["workspace", "collaboration"]
            )
            
            self.metrics["shared_links_created"] += 1
            
            # Generate public URL
            public_url = f"https://viralclip.pro/shared/{link_id}"
            
            return {
                "success": True,
                "link_id": link_id,
                "public_url": public_url,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "protected": bool(password)
            }
            
        except Exception as e:
            logger.error(f"❌ Shared link creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def create_approval_workflow(
        self,
        workspace_id: str,
        project_id: str,
        created_by: str,
        reviewers: List[str],
        deadline_hours: Optional[int] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Create content approval workflow"""
        try:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")
            
            # Validate reviewers
            for reviewer_id in reviewers:
                reviewer = workspace["members"].get(reviewer_id)
                if not reviewer or WorkspacePermission.APPROVE_CONTENT not in reviewer.permissions:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"User {reviewer_id} cannot approve content"
                    )
            
            workflow_id = f"workflow_{uuid.uuid4().hex[:12]}"
            
            deadline = None
            if deadline_hours:
                deadline = datetime.utcnow() + timedelta(hours=deadline_hours)
            
            workflow = ApprovalWorkflow(
                workflow_id=workflow_id,
                project_id=project_id,
                status=ApprovalStatus.PENDING_REVIEW,
                created_by=created_by,
                created_at=datetime.utcnow(),
                reviewers=reviewers,
                deadline=deadline,
                priority=priority
            )
            
            # Store workflow
            project_workflows = workspace["projects"].get(project_id, {}).get("workflows", [])
            project_workflows.append(workflow)
            
            if project_id not in workspace["projects"]:
                workspace["projects"][project_id] = {}
            workspace["projects"][project_id]["workflows"] = project_workflows
            
            # Update cache
            await cache_manager.set(
                f"workspace:{workspace_id}",
                workspace,
                ttl=86400,
                tags=["workspace", "collaboration"]
            )
            
            # Notify reviewers
            for reviewer_id in reviewers:
                await self._send_review_notification(workspace_id, reviewer_id, workflow)
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": workflow.status.value,
                "reviewers": reviewers,
                "deadline": deadline.isoformat() if deadline else None
            }
            
        except Exception as e:
            logger.error(f"❌ Approval workflow creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_workspace_analytics(self, workspace_id: str) -> Dict[str, Any]:
        """Get comprehensive workspace analytics"""
        try:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                raise HTTPException(status_code=404, detail="Workspace not found")
            
            # Calculate analytics
            total_members = len(workspace["members"])
            active_members = sum(1 for member in workspace["members"].values() if member.is_online)
            total_projects = len(workspace["projects"])
            
            # Recent activity
            recent_comments = 0
            recent_versions = 0
            recent_approvals = 0
            
            for project in workspace["projects"].values():
                comments = project.get("comments", [])
                recent_comments += len([c for c in comments if c.timestamp > time.time() - 86400])
                
                versions = project.get("versions", [])
                recent_versions += len([v for v in versions if v.timestamp > datetime.utcnow() - timedelta(days=1)])
                
                workflows = project.get("workflows", [])
                recent_approvals += len([w for w in workflows if w.status == ApprovalStatus.APPROVED])
            
            analytics = {
                "workspace_id": workspace_id,
                "generated_at": datetime.utcnow().isoformat(),
                "team_metrics": {
                    "total_members": total_members,
                    "active_members": active_members,
                    "member_roles": {
                        role.value: sum(1 for m in workspace["members"].values() if m.role == role)
                        for role in UserRole
                    }
                },
                "project_metrics": {
                    "total_projects": total_projects,
                    "projects_with_activity": len([p for p in workspace["projects"].values() if p.get("comments") or p.get("versions")]),
                    "average_collaborators_per_project": active_members / max(total_projects, 1)
                },
                "collaboration_metrics": {
                    "comments_24h": recent_comments,
                    "versions_24h": recent_versions,
                    "approvals_completed": recent_approvals,
                    "active_sessions": len([s for s in self.active_sessions.values() if s["workspace_id"] == workspace_id])
                },
                "system_metrics": self.metrics
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Analytics generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Private helper methods
    
    async def _broadcast_to_workspace(self, workspace_id: str, message: Dict[str, Any]):
        """Broadcast message to all workspace members"""
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            return
        
        for member_id in workspace["members"]:
            websocket = self.websocket_connections.get(member_id)
            if websocket:
                try:
                    await websocket.send_text(json.dumps(message))
                except:
                    # Connection closed
                    del self.websocket_connections[member_id]
    
    async def _broadcast_to_project(
        self, 
        workspace_id: str, 
        project_id: str, 
        message: Dict[str, Any], 
        exclude_user: str = None
    ):
        """Broadcast message to project collaborators"""
        active_sessions = [
            s for s in self.active_sessions.values() 
            if s["workspace_id"] == workspace_id and s["project_id"] == project_id
            and s["user_id"] != exclude_user
        ]
        
        for session in active_sessions:
            websocket = session["websocket"]
            try:
                await websocket.send_text(json.dumps(message))
            except:
                # Remove dead session
                if session["session_id"] in self.active_sessions:
                    del self.active_sessions[session["session_id"]]
    
    async def _get_project_state(self, workspace_id: str, project_id: str) -> Dict[str, Any]:
        """Get current project state for new collaborators"""
        workspace = self.workspaces.get(workspace_id, {})
        project = workspace.get("projects", {}).get(project_id, {})
        
        # Get active collaborators
        collaborators = await self._get_active_collaborators(workspace_id, project_id)
        
        return {
            "project_id": project_id,
            "comments": [c.to_dict() for c in project.get("comments", [])],
            "latest_version": project.get("versions", [])[-1].to_dict() if project.get("versions") else None,
            "active_collaborators": collaborators,
            "recent_operations": self.real_time_operations.get(project_id, [])[-50:]  # Last 50 operations
        }
    
    async def _get_active_collaborators(self, workspace_id: str, project_id: str) -> List[Dict[str, Any]]:
        """Get list of active collaborators for project"""
        collaborators = []
        
        for session in self.active_sessions.values():
            if session["workspace_id"] == workspace_id and session["project_id"] == project_id:
                collaborators.append({
                    "user_id": session["user_id"],
                    "username": session["username"],
                    "session_id": session["session_id"],
                    "joined_at": session["started_at"].isoformat(),
                    "cursor_position": session.get("cursor_position")
                })
        
        return collaborators
    
    async def _send_mention_notification(
        self, 
        workspace_id: str, 
        mentioned_user: str, 
        comment: Comment
    ):
        """Send notification for user mention"""
        websocket = self.websocket_connections.get(mentioned_user)
        if websocket:
            try:
                await websocket.send_text(json.dumps({
                    "type": "mention_notification",
                    "comment": comment.to_dict(),
                    "workspace_id": workspace_id
                }))
            except:
                pass
    
    async def _send_review_notification(
        self, 
        workspace_id: str, 
        reviewer_id: str, 
        workflow: ApprovalWorkflow
    ):
        """Send review request notification"""
        websocket = self.websocket_connections.get(reviewer_id)
        if websocket:
            try:
                await websocket.send_text(json.dumps({
                    "type": "review_request",
                    "workflow_id": workflow.workflow_id,
                    "project_id": workflow.project_id,
                    "deadline": workflow.deadline.isoformat() if workflow.deadline else None,
                    "priority": workflow.priority
                }))
            except:
                pass
    
    async def enterprise_warm_up(self):
        """Enterprise service warm-up"""
        logger.info("🔥 Warming up collaboration engine...")
        
        # Pre-allocate structures
        self.workspaces = {}
        self.active_sessions = {}
        self.websocket_connections = {}
        
        logger.info("✅ Collaboration engine ready")

