
"""
ViralClip Pro v10.0 - WebSocket Routes
Enterprise-grade WebSocket endpoints with optimized routing
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from ..services.websocket_engine import websocket_engine, MessagePriority, WebSocketMessage
from ..services.auth_service import get_current_user_optional
from ..utils.security import verify_websocket_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str = Query(..., description="Session ID for connection"),
    token: Optional[str] = Query(None, description="Authentication token"),
    user_id: Optional[str] = Query(None, description="User ID"),
    metadata: Optional[str] = Query(None, description="JSON metadata")
):
    """
    Main WebSocket endpoint with enterprise features
    
    Features:
    - Automatic retry/reconnect handling
    - Message batching and debouncing
    - Priority-based message routing
    - Session and user targeting
    - Authentication support
    - Real-time metrics
    """
    try:
        # Parse metadata if provided
        parsed_metadata = {}
        if metadata:
            import json
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                await websocket.close(code=4400, reason="Invalid metadata JSON")
                return
        
        # Verify authentication if token provided
        authenticated_user = None
        if token:
            try:
                authenticated_user = await verify_websocket_token(token)
                user_id = authenticated_user.get("user_id") or user_id
                parsed_metadata["authenticated"] = True
                parsed_metadata["user_info"] = authenticated_user
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
                parsed_metadata["authenticated"] = False
        
        # Connect to WebSocket engine
        connection_id = await websocket_engine.connect_websocket(
            websocket=websocket,
            session_id=session_id,
            user_id=user_id,
            metadata=parsed_metadata
        )
        
        if not connection_id:
            await websocket.close(code=4003, reason="Connection rejected")
            return
        
        # Keep connection alive
        try:
            await websocket.receive_text()  # This will block until disconnect
        except WebSocketDisconnect:
            pass
        
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}")
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close(code=1011, reason="Internal server error")


@router.websocket("/realtime/{session_id}")
async def realtime_websocket(
    websocket: WebSocket,
    session_id: str,
    user_id: Optional[str] = Query(None),
    channels: Optional[str] = Query(None, description="Comma-separated channels to subscribe")
):
    """
    Real-time updates WebSocket for video processing
    
    Optimized for:
    - Upload progress
    - Processing status
    - Preview generation
    - Timeline updates
    - Viral insights
    """
    try:
        # Connect with realtime metadata
        metadata = {
            "endpoint_type": "realtime",
            "auto_subscribe": ["processing", "upload", "preview", "insights"]
        }
        
        connection_id = await websocket_engine.connect_websocket(
            websocket=websocket,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        
        if not connection_id:
            await websocket.close(code=4003, reason="Connection rejected")
            return
        
        # Subscribe to channels
        if channels:
            for channel in channels.split(","):
                channel = channel.strip()
                if channel:
                    await websocket_engine.subscribe_to_channel(connection_id, channel)
        
        # Auto-subscribe to default channels
        for channel in metadata["auto_subscribe"]:
            await websocket_engine.subscribe_to_channel(connection_id, channel)
        
        # Keep connection alive
        try:
            await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        
    except Exception as e:
        logger.error(f"Realtime WebSocket error: {e}")
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close(code=1011, reason="Internal server error")


@router.websocket("/analytics/{session_id}")
async def analytics_websocket(
    websocket: WebSocket,
    session_id: str,
    user_id: Optional[str] = Query(None),
    dashboard_id: Optional[str] = Query(None)
):
    """
    Analytics dashboard WebSocket
    
    Features:
    - Real-time metrics streaming
    - Performance data
    - User engagement tracking
    - Custom dashboard support
    """
    try:
        metadata = {
            "endpoint_type": "analytics",
            "dashboard_id": dashboard_id,
            "auto_subscribe": ["metrics", "performance", "engagement"]
        }
        
        connection_id = await websocket_engine.connect_websocket(
            websocket=websocket,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        
        if not connection_id:
            await websocket.close(code=4003, reason="Connection rejected")
            return
        
        # Subscribe to analytics channels
        for channel in metadata["auto_subscribe"]:
            await websocket_engine.subscribe_to_channel(connection_id, channel)
        
        # Keep connection alive
        try:
            await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        
    except Exception as e:
        logger.error(f"Analytics WebSocket error: {e}")


@router.websocket("/collaboration/{project_id}")
async def collaboration_websocket(
    websocket: WebSocket,
    project_id: str,
    user_id: str = Query(...),
    role: Optional[str] = Query("viewer")
):
    """
    Collaboration WebSocket for team editing
    
    Features:
    - Real-time cursor tracking
    - Live editing synchronization
    - User presence indicators
    - Permission-based access
    """
    try:
        metadata = {
            "endpoint_type": "collaboration",
            "project_id": project_id,
            "role": role,
            "auto_subscribe": ["cursors", "edits", "presence", "comments"]
        }
        
        connection_id = await websocket_engine.connect_websocket(
            websocket=websocket,
            session_id=f"project_{project_id}",
            user_id=user_id,
            metadata=metadata
        )
        
        if not connection_id:
            await websocket.close(code=4003, reason="Connection rejected")
            return
        
        # Subscribe to collaboration channels
        for channel in metadata["auto_subscribe"]:
            await websocket_engine.subscribe_to_channel(
                connection_id, 
                f"{project_id}_{channel}"
            )
        
        # Keep connection alive
        try:
            await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        
    except Exception as e:
        logger.error(f"Collaboration WebSocket error: {e}")


# REST API endpoints for WebSocket management

@router.post("/broadcast/session/{session_id}")
async def broadcast_to_session(
    session_id: str,
    message: Dict[str, Any],
    priority: str = "normal",
    current_user: Dict[str, Any] = Depends(get_current_user_optional)
):
    """Broadcast message to all connections in a session"""
    try:
        priority_enum = MessagePriority(priority.lower())
        
        count = await websocket_engine.broadcast_to_session(
            session_id=session_id,
            message_data=message,
            priority=priority_enum
        )
        
        return {
            "success": True,
            "connections_reached": count,
            "session_id": session_id,
            "message_type": message.get("type", "unknown")
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid priority level")
    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        raise HTTPException(status_code=500, detail="Broadcast failed")


@router.post("/broadcast/user/{user_id}")
async def broadcast_to_user(
    user_id: str,
    message: Dict[str, Any],
    priority: str = "normal",
    current_user: Dict[str, Any] = Depends(get_current_user_optional)
):
    """Broadcast message to all user's connections"""
    try:
        priority_enum = MessagePriority(priority.lower())
        
        count = await websocket_engine.broadcast_to_user(
            user_id=user_id,
            message_data=message,
            priority=priority_enum
        )
        
        return {
            "success": True,
            "connections_reached": count,
            "user_id": user_id,
            "message_type": message.get("type", "unknown")
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid priority level")
    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        raise HTTPException(status_code=500, detail="Broadcast failed")


@router.post("/broadcast/all")
async def broadcast_to_all(
    message: Dict[str, Any],
    priority: str = "normal",
    current_user: Dict[str, Any] = Depends(get_current_user_optional)
):
    """Broadcast message to all connections"""
    try:
        priority_enum = MessagePriority(priority.lower())
        
        count = await websocket_engine.broadcast_to_all(
            message_data=message,
            priority=priority_enum
        )
        
        return {
            "success": True,
            "connections_reached": count,
            "message_type": message.get("type", "unknown")
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid priority level")
    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        raise HTTPException(status_code=500, detail="Broadcast failed")


@router.get("/stats")
async def get_websocket_stats(
    current_user: Dict[str, Any] = Depends(get_current_user_optional)
):
    """Get WebSocket engine statistics"""
    try:
        stats = await websocket_engine.get_engine_stats()
        return {
            "success": True,
            "stats": stats,
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")


@router.get("/health")
async def websocket_health_check():
    """WebSocket engine health check"""
    try:
        health = await websocket_engine.health_check()
        
        status_code = 200 if health["healthy"] else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "healthy": health["healthy"],
                "issues": health.get("issues", []),
                "metrics": health,
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "healthy": False,
                "issues": [f"Health check failed: {e}"],
                "timestamp": asyncio.get_event_loop().time()
            }
        )


@router.post("/subscribe/{connection_id}")
async def subscribe_to_channel(
    connection_id: str,
    channel: str,
    current_user: Dict[str, Any] = Depends(get_current_user_optional)
):
    """Subscribe connection to a channel"""
    try:
        success = await websocket_engine.subscribe_to_channel(connection_id, channel)
        
        if success:
            return {
                "success": True,
                "connection_id": connection_id,
                "channel": channel,
                "action": "subscribed"
            }
        else:
            raise HTTPException(status_code=404, detail="Connection not found")
            
    except Exception as e:
        logger.error(f"Subscribe error: {e}")
        raise HTTPException(status_code=500, detail="Subscription failed")


@router.post("/unsubscribe/{connection_id}")
async def unsubscribe_from_channel(
    connection_id: str,
    channel: str,
    current_user: Dict[str, Any] = Depends(get_current_user_optional)
):
    """Unsubscribe connection from a channel"""
    try:
        success = await websocket_engine.unsubscribe_from_channel(connection_id, channel)
        
        if success:
            return {
                "success": True,
                "connection_id": connection_id,
                "channel": channel,
                "action": "unsubscribed"
            }
        else:
            raise HTTPException(status_code=404, detail="Connection not found")
            
    except Exception as e:
        logger.error(f"Unsubscribe error: {e}")
        raise HTTPException(status_code=500, detail="Unsubscription failed")


# Message handlers for custom message types

async def handle_upload_progress(connection, message):
    """Handle upload progress updates"""
    progress_data = message.data
    
    # Broadcast to session
    await websocket_engine.broadcast_to_session(
        session_id=connection.session_id,
        message_data={
            "type": "upload_progress_update",
            "progress": progress_data.get("progress", 0),
            "speed": progress_data.get("speed", 0),
            "eta": progress_data.get("eta", 0),
            "status": progress_data.get("status", "uploading")
        },
        priority=MessagePriority.HIGH
    )


async def handle_processing_status(connection, message):
    """Handle video processing status updates"""
    status_data = message.data
    
    # Broadcast to session with high priority
    await websocket_engine.broadcast_to_session(
        session_id=connection.session_id,
        message_data={
            "type": "processing_status_update",
            "stage": status_data.get("stage", "unknown"),
            "progress": status_data.get("progress", 0),
            "estimated_time": status_data.get("estimated_time", 0),
            "current_operation": status_data.get("current_operation", "Processing...")
        },
        priority=MessagePriority.HIGH
    )


async def handle_viral_insights(connection, message):
    """Handle viral insights updates"""
    insights_data = message.data
    
    # Broadcast insights with normal priority
    await websocket_engine.broadcast_to_session(
        session_id=connection.session_id,
        message_data={
            "type": "viral_insights_update",
            "insights": insights_data.get("insights", []),
            "viral_score": insights_data.get("viral_score", 0),
            "confidence": insights_data.get("confidence", 0),
            "recommendations": insights_data.get("recommendations", [])
        },
        priority=MessagePriority.NORMAL
    )


# Register message handlers
websocket_engine.add_message_handler("upload_progress", handle_upload_progress)
websocket_engine.add_message_handler("processing_status", handle_processing_status)
websocket_engine.add_message_handler("viral_insights", handle_viral_insights)
