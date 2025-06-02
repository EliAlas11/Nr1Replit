
"""
ViralClip Pro v10.0 - Netflix-Level WebSocket Engine
Enterprise-grade real-time communication with horizontal scaling
"""

import asyncio
import json
import logging
import time
import uuid
import weakref
from typing import Dict, List, Any, Optional, Set, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import psutil
import hashlib

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels for intelligent routing"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ConnectionState(Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """Enhanced WebSocket message with routing and priority"""
    id: str
    type: str
    data: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    target_session: Optional[str] = None
    target_user: Optional[str] = None
    broadcast: bool = False
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    middleware_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionMetrics:
    """Connection performance metrics"""
    connected_at: datetime
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0
    reconnection_count: int = 0
    error_count: int = 0


@dataclass
class SocketConnection:
    """Enhanced WebSocket connection wrapper"""
    websocket: WebSocket
    connection_id: str
    session_id: str
    user_id: Optional[str]
    state: ConnectionState
    metrics: ConnectionMetrics
    subscriptions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1000))
    heartbeat_task: Optional[asyncio.Task] = None
    send_task: Optional[asyncio.Task] = None


class WebSocketMiddleware:
    """Base class for WebSocket middleware"""
    
    async def process_incoming(self, connection: SocketConnection, message: WebSocketMessage) -> bool:
        """Process incoming message. Return False to block message."""
        return True
    
    async def process_outgoing(self, connection: SocketConnection, message: WebSocketMessage) -> bool:
        """Process outgoing message. Return False to block message."""
        return True
    
    async def on_connect(self, connection: SocketConnection) -> bool:
        """Called when connection is established. Return False to reject."""
        return True
    
    async def on_disconnect(self, connection: SocketConnection) -> None:
        """Called when connection is closed."""
        pass


class RateLimitMiddleware(WebSocketMiddleware):
    """Rate limiting middleware for WebSocket connections"""
    
    def __init__(self, messages_per_minute: int = 100):
        self.messages_per_minute = messages_per_minute
        self.message_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=messages_per_minute))
    
    async def process_incoming(self, connection: SocketConnection, message: WebSocketMessage) -> bool:
        now = time.time()
        timestamps = self.message_timestamps[connection.connection_id]
        
        # Remove old timestamps
        while timestamps and timestamps[0] < now - 60:
            timestamps.popleft()
        
        if len(timestamps) >= self.messages_per_minute:
            logger.warning(f"Rate limit exceeded for connection {connection.connection_id}")
            return False
        
        timestamps.append(now)
        return True


class AuthenticationMiddleware(WebSocketMiddleware):
    """Authentication middleware for WebSocket connections"""
    
    async def on_connect(self, connection: SocketConnection) -> bool:
        # Check if user is authenticated
        if not connection.user_id and connection.session_id != "public":
            logger.warning(f"Unauthenticated connection rejected: {connection.connection_id}")
            return False
        return True


class LoggingMiddleware(WebSocketMiddleware):
    """Logging middleware for WebSocket connections"""
    
    async def process_incoming(self, connection: SocketConnection, message: WebSocketMessage) -> bool:
        logger.debug(f"Incoming message: {connection.connection_id} -> {message.type}")
        return True
    
    async def process_outgoing(self, connection: SocketConnection, message: WebSocketMessage) -> bool:
        logger.debug(f"Outgoing message: {connection.connection_id} <- {message.type}")
        return True


class NetflixWebSocketEngine:
    """Netflix-grade WebSocket engine with enterprise features"""
    
    def __init__(self, redis_client=None):
        # Connection management
        self.connections: Dict[str, SocketConnection] = {}
        self.session_connections: Dict[str, Set[str]] = defaultdict(set)
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Message handling
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.middleware_stack: List[WebSocketMiddleware] = []
        self.message_buffer: Dict[MessagePriority, deque] = {
            priority: deque(maxlen=10000) for priority in MessagePriority
        }
        
        # Performance optimization
        self.batch_size = 50
        self.batch_timeout = 0.1  # 100ms
        self.debounce_window = 0.05  # 50ms
        self.debounced_messages: Dict[str, WebSocketMessage] = {}
        
        # Redis for horizontal scaling
        self.redis_client = redis_client
        self.redis_channel = "viralclip_websocket"
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.message_processor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.redis_subscriber_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.engine_metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_processed": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "reconnections": 0
        }
        
        # Default middleware
        self._setup_default_middleware()
        
        logger.info("🚀 Netflix WebSocket Engine initialized")
    
    def _setup_default_middleware(self):
        """Setup default middleware stack"""
        self.add_middleware(RateLimitMiddleware(messages_per_minute=120))
        self.add_middleware(AuthenticationMiddleware())
        self.add_middleware(LoggingMiddleware())
    
    def add_middleware(self, middleware: WebSocketMiddleware):
        """Add middleware to the stack"""
        self.middleware_stack.append(middleware)
        logger.info(f"Added middleware: {middleware.__class__.__name__}")
    
    def add_message_handler(self, message_type: str, handler: Callable):
        """Add message handler for specific message type"""
        self.message_handlers[message_type].append(handler)
        logger.info(f"Added handler for message type: {message_type}")
    
    async def start_engine(self):
        """Start the WebSocket engine background tasks"""
        try:
            # Start message processor
            self.message_processor_task = asyncio.create_task(self._process_message_queue())
            self.background_tasks.add(self.message_processor_task)
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_connections())
            self.background_tasks.add(self.cleanup_task)
            
            # Start Redis subscriber if Redis is available
            if self.redis_client:
                self.redis_subscriber_task = asyncio.create_task(self._redis_subscriber())
                self.background_tasks.add(self.redis_subscriber_task)
            
            logger.info("✅ WebSocket engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket engine: {e}")
            raise
    
    async def connect_websocket(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Connect a new WebSocket with enterprise features"""
        try:
            # Accept the connection
            await websocket.accept()
            
            # Generate connection ID
            connection_id = f"ws_{uuid.uuid4().hex[:16]}"
            
            # Create connection object
            connection = SocketConnection(
                websocket=websocket,
                connection_id=connection_id,
                session_id=session_id,
                user_id=user_id,
                state=ConnectionState.CONNECTING,
                metrics=ConnectionMetrics(connected_at=datetime.utcnow()),
                metadata=metadata or {}
            )
            
            # Run middleware for connection
            for middleware in self.middleware_stack:
                if not await middleware.on_connect(connection):
                    await websocket.close(code=4003, reason="Connection rejected")
                    return ""
            
            # Store connection
            self.connections[connection_id] = connection
            self.session_connections[session_id].add(connection_id)
            if user_id:
                self.user_connections[user_id].add(connection_id)
            
            # Update connection state
            connection.state = ConnectionState.CONNECTED
            
            # Start connection tasks
            connection.heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(connection)
            )
            connection.send_task = asyncio.create_task(
                self._send_loop(connection)
            )
            
            # Update metrics
            self.engine_metrics["total_connections"] += 1
            self.engine_metrics["active_connections"] = len(self.connections)
            
            # Send welcome message
            await self._send_welcome_message(connection)
            
            logger.info(f"✅ WebSocket connected: {connection_id} (session: {session_id})")
            
            # Start listening for messages
            asyncio.create_task(self._listen_for_messages(connection))
            
            return connection_id
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1011, reason="Internal error")
            return ""
    
    async def disconnect_websocket(self, connection_id: str, reason: str = "Normal closure"):
        """Disconnect WebSocket with cleanup"""
        try:
            connection = self.connections.get(connection_id)
            if not connection:
                return
            
            # Update state
            connection.state = ConnectionState.DISCONNECTED
            
            # Run middleware
            for middleware in self.middleware_stack:
                await middleware.on_disconnect(connection)
            
            # Cancel tasks
            if connection.heartbeat_task:
                connection.heartbeat_task.cancel()
            if connection.send_task:
                connection.send_task.cancel()
            
            # Close WebSocket
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                await connection.websocket.close(code=1000, reason=reason)
            
            # Remove from collections
            del self.connections[connection_id]
            self.session_connections[connection.session_id].discard(connection_id)
            if connection.user_id:
                self.user_connections[connection.user_id].discard(connection_id)
            
            # Clean up empty sets
            if not self.session_connections[connection.session_id]:
                del self.session_connections[connection.session_id]
            if connection.user_id and not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
            
            # Update metrics
            self.engine_metrics["active_connections"] = len(self.connections)
            
            logger.info(f"🔌 WebSocket disconnected: {connection_id} ({reason})")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket {connection_id}: {e}")
    
    async def send_message(
        self,
        message: WebSocketMessage,
        target_connection: Optional[str] = None,
        target_session: Optional[str] = None,
        target_user: Optional[str] = None,
        broadcast: bool = False
    ) -> bool:
        """Send message with intelligent routing and retry logic"""
        try:
            # Update message routing info
            if target_connection:
                message.target_session = None
                message.target_user = None
                message.broadcast = False
            elif target_session:
                message.target_session = target_session
                message.target_user = None
                message.broadcast = False
            elif target_user:
                message.target_user = target_user
                message.target_session = None
                message.broadcast = False
            elif broadcast:
                message.broadcast = True
            
            # Check for debouncing
            if await self._should_debounce_message(message):
                return True
            
            # Add to appropriate queue based on priority
            self.message_buffer[message.priority].append(message)
            
            # Publish to Redis for horizontal scaling
            if self.redis_client and message.broadcast:
                await self._publish_to_redis(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def broadcast_to_session(self, session_id: str, message_data: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL) -> int:
        """Broadcast message to all connections in a session"""
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=message_data.get("type", "broadcast"),
            data=message_data,
            priority=priority,
            target_session=session_id
        )
        
        await self.send_message(message, target_session=session_id)
        return len(self.session_connections.get(session_id, set()))
    
    async def broadcast_to_user(self, user_id: str, message_data: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL) -> int:
        """Broadcast message to all user's connections"""
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=message_data.get("type", "broadcast"),
            data=message_data,
            priority=priority,
            target_user=user_id
        )
        
        await self.send_message(message, target_user=user_id)
        return len(self.user_connections.get(user_id, set()))
    
    async def broadcast_to_all(self, message_data: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL) -> int:
        """Broadcast message to all connections"""
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=message_data.get("type", "broadcast"),
            data=message_data,
            priority=priority,
            broadcast=True
        )
        
        await self.send_message(message, broadcast=True)
        return len(self.connections)
    
    async def subscribe_to_channel(self, connection_id: str, channel: str) -> bool:
        """Subscribe connection to a channel"""
        connection = self.connections.get(connection_id)
        if not connection:
            return False
        
        connection.subscriptions.add(channel)
        logger.info(f"Connection {connection_id} subscribed to channel: {channel}")
        return True
    
    async def unsubscribe_from_channel(self, connection_id: str, channel: str) -> bool:
        """Unsubscribe connection from a channel"""
        connection = self.connections.get(connection_id)
        if not connection:
            return False
        
        connection.subscriptions.discard(channel)
        logger.info(f"Connection {connection_id} unsubscribed from channel: {channel}")
        return True
    
    async def _process_message_queue(self):
        """Process message queue with batching and priority handling"""
        while True:
            try:
                batch = []
                batch_start = time.time()
                
                # Collect messages by priority
                for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, MessagePriority.NORMAL, MessagePriority.LOW]:
                    queue = self.message_buffer[priority]
                    
                    while queue and len(batch) < self.batch_size and (time.time() - batch_start) < self.batch_timeout:
                        message = queue.popleft()
                        
                        # Check if message expired
                        if message.expires_at and message.expires_at < datetime.utcnow():
                            continue
                        
                        batch.append(message)
                
                if not batch:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process batch
                await self._process_message_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message queue processing error: {e}")
                self.engine_metrics["errors"] += 1
                await asyncio.sleep(0.1)
    
    async def _process_message_batch(self, messages: List[WebSocketMessage]):
        """Process a batch of messages efficiently"""
        try:
            # Group messages by target
            target_groups = defaultdict(list)
            
            for message in messages:
                if message.broadcast:
                    target_groups["broadcast"].append(message)
                elif message.target_session:
                    target_groups[f"session_{message.target_session}"].append(message)
                elif message.target_user:
                    target_groups[f"user_{message.target_user}"].append(message)
            
            # Process each group
            for target_key, group_messages in target_groups.items():
                if target_key == "broadcast":
                    await self._process_broadcast_messages(group_messages)
                elif target_key.startswith("session_"):
                    session_id = target_key[8:]  # Remove "session_" prefix
                    await self._process_session_messages(session_id, group_messages)
                elif target_key.startswith("user_"):
                    user_id = target_key[5:]  # Remove "user_" prefix
                    await self._process_user_messages(user_id, group_messages)
            
            self.engine_metrics["messages_processed"] += len(messages)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    async def _process_broadcast_messages(self, messages: List[WebSocketMessage]):
        """Process broadcast messages"""
        for connection in list(self.connections.values()):
            for message in messages:
                await self._queue_message_to_connection(connection, message)
    
    async def _process_session_messages(self, session_id: str, messages: List[WebSocketMessage]):
        """Process session-targeted messages"""
        connection_ids = self.session_connections.get(session_id, set())
        
        for connection_id in list(connection_ids):
            connection = self.connections.get(connection_id)
            if connection:
                for message in messages:
                    await self._queue_message_to_connection(connection, message)
    
    async def _process_user_messages(self, user_id: str, messages: List[WebSocketMessage]):
        """Process user-targeted messages"""
        connection_ids = self.user_connections.get(user_id, set())
        
        for connection_id in list(connection_ids):
            connection = self.connections.get(connection_id)
            if connection:
                for message in messages:
                    await self._queue_message_to_connection(connection, message)
    
    async def _queue_message_to_connection(self, connection: SocketConnection, message: WebSocketMessage):
        """Queue message to specific connection"""
        try:
            # Run outgoing middleware
            for middleware in self.middleware_stack:
                if not await middleware.process_outgoing(connection, message):
                    return
            
            # Add to connection queue
            if connection.message_queue.full():
                # Remove oldest message to make room
                try:
                    connection.message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            
            await connection.message_queue.put(message)
            
        except Exception as e:
            logger.error(f"Failed to queue message to connection {connection.connection_id}: {e}")
    
    async def _send_loop(self, connection: SocketConnection):
        """Connection-specific send loop with retry logic"""
        retry_count = 0
        max_retries = 3
        
        while connection.state == ConnectionState.CONNECTED:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    connection.message_queue.get(),
                    timeout=30.0
                )
                
                # Serialize message
                message_json = json.dumps({
                    "id": message.id,
                    "type": message.type,
                    "data": message.data,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Send message
                await connection.websocket.send_text(message_json)
                
                # Update metrics
                connection.metrics.messages_sent += 1
                connection.metrics.bytes_sent += len(message_json)
                connection.metrics.last_activity = datetime.utcnow()
                
                # Reset retry count on successful send
                retry_count = 0
                
            except asyncio.TimeoutError:
                # No messages to send, continue
                continue
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected during send: {connection.connection_id}")
                break
                
            except Exception as e:
                logger.error(f"Send error for connection {connection.connection_id}: {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error(f"Max retries exceeded for connection {connection.connection_id}")
                    await self.disconnect_websocket(connection.connection_id, "Send error")
                    break
                
                # Exponential backoff
                await asyncio.sleep(min(2 ** retry_count, 10))
    
    async def _listen_for_messages(self, connection: SocketConnection):
        """Listen for incoming messages from WebSocket"""
        try:
            async for message_text in connection.websocket.iter_text():
                try:
                    # Parse message
                    message_data = json.loads(message_text)
                    
                    # Create message object
                    message = WebSocketMessage(
                        id=message_data.get("id", str(uuid.uuid4())),
                        type=message_data.get("type", "unknown"),
                        data=message_data.get("data", {}),
                        priority=MessagePriority(message_data.get("priority", "normal"))
                    )
                    
                    # Run incoming middleware
                    should_process = True
                    for middleware in self.middleware_stack:
                        if not await middleware.process_incoming(connection, message):
                            should_process = False
                            break
                    
                    if not should_process:
                        continue
                    
                    # Update metrics
                    connection.metrics.messages_received += 1
                    connection.metrics.bytes_received += len(message_text)
                    connection.metrics.last_activity = datetime.utcnow()
                    
                    # Handle message
                    await self._handle_incoming_message(connection, message)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from connection {connection.connection_id}")
                    connection.metrics.error_count += 1
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection.connection_id}")
        except Exception as e:
            logger.error(f"Listen error for connection {connection.connection_id}: {e}")
            connection.metrics.error_count += 1
        finally:
            await self.disconnect_websocket(connection.connection_id, "Connection closed")
    
    async def _handle_incoming_message(self, connection: SocketConnection, message: WebSocketMessage):
        """Handle incoming message with routing"""
        try:
            # Built-in message handlers
            if message.type == "ping":
                await self._handle_ping(connection, message)
            elif message.type == "subscribe":
                await self._handle_subscribe(connection, message)
            elif message.type == "unsubscribe":
                await self._handle_unsubscribe(connection, message)
            else:
                # Custom message handlers
                handlers = self.message_handlers.get(message.type, [])
                for handler in handlers:
                    try:
                        await handler(connection, message)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")
            
        except Exception as e:
            logger.error(f"Message handling error: {e}")
    
    async def _handle_ping(self, connection: SocketConnection, message: WebSocketMessage):
        """Handle ping message"""
        pong_message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type="pong",
            data={"timestamp": datetime.utcnow().isoformat()}
        )
        await self._queue_message_to_connection(connection, pong_message)
    
    async def _handle_subscribe(self, connection: SocketConnection, message: WebSocketMessage):
        """Handle channel subscription"""
        channel = message.data.get("channel")
        if channel:
            await self.subscribe_to_channel(connection.connection_id, channel)
    
    async def _handle_unsubscribe(self, connection: SocketConnection, message: WebSocketMessage):
        """Handle channel unsubscription"""
        channel = message.data.get("channel")
        if channel:
            await self.unsubscribe_from_channel(connection.connection_id, channel)
    
    async def _heartbeat_loop(self, connection: SocketConnection):
        """Connection heartbeat loop"""
        while connection.state == ConnectionState.CONNECTED:
            try:
                # Send heartbeat
                heartbeat_message = WebSocketMessage(
                    id=str(uuid.uuid4()),
                    type="heartbeat",
                    data={"timestamp": datetime.utcnow().isoformat()},
                    priority=MessagePriority.LOW
                )
                
                await self._queue_message_to_connection(connection, heartbeat_message)
                
                # Wait for next heartbeat
                await asyncio.sleep(30)  # 30 second intervals
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error for connection {connection.connection_id}: {e}")
                break
    
    async def _send_welcome_message(self, connection: SocketConnection):
        """Send welcome message to new connection"""
        welcome_message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type="welcome",
            data={
                "connection_id": connection.connection_id,
                "session_id": connection.session_id,
                "server_time": datetime.utcnow().isoformat(),
                "features": {
                    "retry_logic": True,
                    "message_batching": True,
                    "priority_routing": True,
                    "horizontal_scaling": bool(self.redis_client)
                }
            },
            priority=MessagePriority.HIGH
        )
        
        await self._queue_message_to_connection(connection, welcome_message)
    
    async def _should_debounce_message(self, message: WebSocketMessage) -> bool:
        """Check if message should be debounced"""
        # Create debounce key based on message type and target
        debounce_key = f"{message.type}_{message.target_session or message.target_user or 'broadcast'}"
        
        # Check if we have a recent message of the same type
        if debounce_key in self.debounced_messages:
            existing_message = self.debounced_messages[debounce_key]
            time_diff = (message.created_at - existing_message.created_at).total_seconds()
            
            if time_diff < self.debounce_window:
                # Update existing message with new data
                existing_message.data.update(message.data)
                existing_message.created_at = message.created_at
                return True
        
        # Store message for debouncing
        self.debounced_messages[debounce_key] = message
        
        # Schedule cleanup
        asyncio.create_task(self._cleanup_debounced_message(debounce_key))
        
        return False
    
    async def _cleanup_debounced_message(self, debounce_key: str):
        """Cleanup debounced message after window"""
        await asyncio.sleep(self.debounce_window * 2)
        self.debounced_messages.pop(debounce_key, None)
    
    async def _cleanup_expired_connections(self):
        """Cleanup expired and stale connections"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection in self.connections.items():
                    # Check for stale connections (no activity for 5 minutes)
                    if (now - connection.metrics.last_activity).total_seconds() > 300:
                        stale_connections.append(connection_id)
                
                # Disconnect stale connections
                for connection_id in stale_connections:
                    await self.disconnect_websocket(connection_id, "Stale connection")
                
                if stale_connections:
                    logger.info(f"Cleaned up {len(stale_connections)} stale connections")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection cleanup error: {e}")
    
    async def _publish_to_redis(self, message: WebSocketMessage):
        """Publish message to Redis for horizontal scaling"""
        if not self.redis_client:
            return
        
        try:
            message_data = {
                "id": message.id,
                "type": message.type,
                "data": message.data,
                "priority": message.priority.value,
                "target_session": message.target_session,
                "target_user": message.target_user,
                "broadcast": message.broadcast,
                "timestamp": message.created_at.isoformat()
            }
            
            await self.redis_client.publish(
                self.redis_channel,
                json.dumps(message_data)
            )
            
        except Exception as e:
            logger.error(f"Redis publish error: {e}")
    
    async def _redis_subscriber(self):
        """Subscribe to Redis for horizontal scaling"""
        if not self.redis_client:
            return
        
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(self.redis_channel)
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        message_data = json.loads(message["data"])
                        
                        # Recreate WebSocket message
                        ws_message = WebSocketMessage(
                            id=message_data["id"],
                            type=message_data["type"],
                            data=message_data["data"],
                            priority=MessagePriority(message_data["priority"]),
                            target_session=message_data.get("target_session"),
                            target_user=message_data.get("target_user"),
                            broadcast=message_data.get("broadcast", False)
                        )
                        
                        # Process message locally
                        self.message_buffer[ws_message.priority].append(ws_message)
                        
                    except Exception as e:
                        logger.error(f"Redis message processing error: {e}")
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Redis subscriber error: {e}")
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        return {
            **self.engine_metrics,
            "active_connections": len(self.connections),
            "active_sessions": len(self.session_connections),
            "active_users": len(self.user_connections),
            "message_buffer_sizes": {
                priority.value: len(queue) for priority, queue in self.message_buffer.items()
            },
            "background_tasks": len(self.background_tasks),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "redis_enabled": bool(self.redis_client)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform engine health check"""
        try:
            healthy = True
            issues = []
            
            # Check message processor
            if self.message_processor_task and self.message_processor_task.done():
                healthy = False
                issues.append("Message processor not running")
            
            # Check message buffer sizes
            total_buffered = sum(len(queue) for queue in self.message_buffer.values())
            if total_buffered > 5000:
                healthy = False
                issues.append(f"Message buffer overflow: {total_buffered}")
            
            # Check connection health
            unhealthy_connections = 0
            for connection in self.connections.values():
                if connection.metrics.error_count > 10:
                    unhealthy_connections += 1
            
            if unhealthy_connections > len(self.connections) * 0.1:  # More than 10% unhealthy
                healthy = False
                issues.append(f"Too many unhealthy connections: {unhealthy_connections}")
            
            return {
                "healthy": healthy,
                "issues": issues,
                "active_connections": len(self.connections),
                "total_buffered_messages": total_buffered,
                "unhealthy_connections": unhealthy_connections
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "issues": [f"Health check error: {e}"],
                "error": str(e)
            }
    
    async def graceful_shutdown(self):
        """Gracefully shutdown the WebSocket engine"""
        try:
            logger.info("🔄 Starting WebSocket engine shutdown...")
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Disconnect all connections
            disconnect_tasks = []
            for connection_id in list(self.connections.keys()):
                disconnect_tasks.append(
                    self.disconnect_websocket(connection_id, "Server shutdown")
                )
            
            if disconnect_tasks:
                await asyncio.gather(*disconnect_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Clear all data
            self.connections.clear()
            self.session_connections.clear()
            self.user_connections.clear()
            self.message_handlers.clear()
            for queue in self.message_buffer.values():
                queue.clear()
            self.debounced_messages.clear()
            
            logger.info("✅ WebSocket engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during WebSocket engine shutdown: {e}")


# Global engine instance
websocket_engine = NetflixWebSocketEngine()
