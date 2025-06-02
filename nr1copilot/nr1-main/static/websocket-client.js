
class NetflixWebSocketClient {
    constructor(options = {}) {
        this.options = {
            reconnectInterval: 1000,
            maxReconnectInterval: 30000,
            reconnectDecay: 1.5,
            maxReconnectAttempts: 10,
            heartbeatInterval: 30000,
            messageTimeout: 5000,
            enableDebouncing: true,
            debounceDelay: 100,
            batchSize: 10,
            ...options
        };

        this.websocket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.currentReconnectInterval = this.options.reconnectInterval;
        this.heartbeatTimer = null;
        this.messageQueue = [];
        this.pendingMessages = new Map();
        this.debouncedMessages = new Map();
        this.eventHandlers = new Map();
        this.connectionId = null;
        this.sessionId = null;
        this.userId = null;

        // Metrics
        this.metrics = {
            messagesReceived: 0,
            messagesSent: 0,
            reconnections: 0,
            errors: 0,
            bytesReceived: 0,
            bytesSent: 0,
            latency: 0
        };

        // Connection state callbacks
        this.onConnectionStateChange = null;
        this.onError = null;
        this.onMessage = null;
    }

    /**
     * Connect to WebSocket with enterprise features
     */
    async connect(endpoint, sessionId, userId = null, metadata = {}) {
        try {
            this.sessionId = sessionId;
            this.userId = userId;

            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsHost = window.location.host;
            
            // Build WebSocket URL with parameters
            const params = new URLSearchParams({
                session_id: sessionId,
                ...(userId && { user_id: userId }),
                ...(Object.keys(metadata).length && { metadata: JSON.stringify(metadata) })
            });

            const wsUrl = `${wsProtocol}//${wsHost}/ws/${endpoint}?${params}`;

            console.log(`🔗 Connecting to WebSocket: ${wsUrl}`);

            this.websocket = new WebSocket(wsUrl);
            this.setupEventHandlers();

            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Connection timeout'));
                }, 10000);

                this.websocket.onopen = () => {
                    clearTimeout(timeout);
                    this.onConnected();
                    resolve(this.connectionId);
                };

                this.websocket.onerror = (error) => {
                    clearTimeout(timeout);
                    reject(error);
                };
            });

        } catch (error) {
            console.error('❌ WebSocket connection failed:', error);
            this.handleError(error);
            throw error;
        }
    }

    /**
     * Setup WebSocket event handlers
     */
    setupEventHandlers() {
        this.websocket.onopen = () => this.onConnected();
        this.websocket.onclose = (event) => this.onDisconnected(event);
        this.websocket.onerror = (error) => this.handleError(error);
        this.websocket.onmessage = (event) => this.handleMessage(event);
    }

    /**
     * Handle successful connection
     */
    onConnected() {
        console.log('✅ WebSocket connected successfully');
        
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.currentReconnectInterval = this.options.reconnectInterval;
        
        this.startHeartbeat();
        this.flushMessageQueue();
        
        if (this.onConnectionStateChange) {
            this.onConnectionStateChange('connected');
        }
    }

    /**
     * Handle disconnection
     */
    onDisconnected(event) {
        console.log(`🔌 WebSocket disconnected: ${event.code} - ${event.reason}`);
        
        this.isConnected = false;
        this.connectionId = null;
        this.stopHeartbeat();
        
        if (this.onConnectionStateChange) {
            this.onConnectionStateChange('disconnected');
        }

        // Attempt reconnection if not a normal closure
        if (event.code !== 1000 && this.reconnectAttempts < this.options.maxReconnectAttempts) {
            this.attemptReconnection();
        }
    }

    /**
     * Attempt reconnection with exponential backoff
     */
    attemptReconnection() {
        if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            console.error('❌ Max reconnection attempts reached');
            if (this.onConnectionStateChange) {
                this.onConnectionStateChange('failed');
            }
            return;
        }

        this.reconnectAttempts++;
        console.log(`🔄 Reconnection attempt ${this.reconnectAttempts}/${this.options.maxReconnectAttempts} in ${this.currentReconnectInterval}ms`);

        if (this.onConnectionStateChange) {
            this.onConnectionStateChange('reconnecting');
        }

        setTimeout(() => {
            if (!this.isConnected) {
                this.connect('connect', this.sessionId, this.userId)
                    .catch(() => {
                        // Increase reconnection interval with exponential backoff
                        this.currentReconnectInterval = Math.min(
                            this.currentReconnectInterval * this.options.reconnectDecay,
                            this.options.maxReconnectInterval
                        );
                    });
            }
        }, this.currentReconnectInterval);

        this.metrics.reconnections++;
    }

    /**
     * Handle incoming messages
     */
    handleMessage(event) {
        try {
            const message = JSON.parse(event.data);
            this.metrics.messagesReceived++;
            this.metrics.bytesReceived += event.data.length;

            // Handle built-in message types
            switch (message.type) {
                case 'welcome':
                    this.handleWelcomeMessage(message);
                    break;
                case 'pong':
                    this.handlePongMessage(message);
                    break;
                case 'heartbeat':
                    this.handleHeartbeatMessage(message);
                    break;
                default:
                    this.dispatchMessage(message);
                    break;
            }

            if (this.onMessage) {
                this.onMessage(message);
            }

        } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error);
            this.handleError(error);
        }
    }

    /**
     * Handle welcome message
     */
    handleWelcomeMessage(message) {
        this.connectionId = message.data.connection_id;
        console.log(`🎉 Welcome message received. Connection ID: ${this.connectionId}`);
        
        // Log available features
        if (message.data.features) {
            console.log('🚀 Available features:', message.data.features);
        }
    }

    /**
     * Handle pong message for latency calculation
     */
    handlePongMessage(message) {
        const pingTime = this.pendingMessages.get('ping');
        if (pingTime) {
            this.metrics.latency = Date.now() - pingTime;
            this.pendingMessages.delete('ping');
        }
    }

    /**
     * Handle heartbeat message
     */
    handleHeartbeatMessage(message) {
        // Respond with pong
        this.send({
            type: 'pong',
            data: { timestamp: new Date().toISOString() }
        });
    }

    /**
     * Dispatch message to registered handlers
     */
    dispatchMessage(message) {
        const handlers = this.eventHandlers.get(message.type);
        if (handlers) {
            handlers.forEach(handler => {
                try {
                    handler(message);
                } catch (error) {
                    console.error(`❌ Error in message handler for type '${message.type}':`, error);
                }
            });
        }
    }

    /**
     * Send message with retry logic and batching
     */
    send(message, options = {}) {
        if (!message.id) {
            message.id = this.generateMessageId();
        }

        const messageOptions = {
            priority: 'normal',
            retry: true,
            timeout: this.options.messageTimeout,
            debounce: this.options.enableDebouncing,
            ...options
        };

        // Handle debouncing
        if (messageOptions.debounce && this.shouldDebounceMessage(message)) {
            this.debounceMessage(message, messageOptions);
            return message.id;
        }

        // Queue message if not connected
        if (!this.isConnected) {
            this.messageQueue.push({ message, options: messageOptions });
            return message.id;
        }

        // Send immediately
        this.sendMessage(message, messageOptions);
        return message.id;
    }

    /**
     * Send message immediately
     */
    sendMessage(message, options) {
        try {
            const messageString = JSON.stringify(message);
            this.websocket.send(messageString);

            this.metrics.messagesSent++;
            this.metrics.bytesSent += messageString.length;

            // Track pending message for timeout handling
            if (options.timeout > 0) {
                setTimeout(() => {
                    if (this.pendingMessages.has(message.id)) {
                        this.pendingMessages.delete(message.id);
                        console.warn(`⏰ Message timeout: ${message.id}`);
                    }
                }, options.timeout);
            }

        } catch (error) {
            console.error('❌ Error sending message:', error);
            this.handleError(error);

            // Retry if enabled
            if (options.retry) {
                this.messageQueue.push({ message, options: { ...options, retry: false } });
            }
        }
    }

    /**
     * Check if message should be debounced
     */
    shouldDebounceMessage(message) {
        if (!this.options.enableDebouncing) {
            return false;
        }

        // Debounce based on message type and target
        const debounceKey = `${message.type}_${message.target || 'default'}`;
        return this.debouncedMessages.has(debounceKey);
    }

    /**
     * Debounce message
     */
    debounceMessage(message, options) {
        const debounceKey = `${message.type}_${message.target || 'default'}`;
        
        // Clear existing timeout
        if (this.debouncedMessages.has(debounceKey)) {
            clearTimeout(this.debouncedMessages.get(debounceKey).timeout);
        }

        // Set new timeout
        const timeout = setTimeout(() => {
            this.debouncedMessages.delete(debounceKey);
            this.sendMessage(message, options);
        }, this.options.debounceDelay);

        this.debouncedMessages.set(debounceKey, { message, timeout });
    }

    /**
     * Flush queued messages
     */
    flushMessageQueue() {
        while (this.messageQueue.length > 0 && this.isConnected) {
            const batch = this.messageQueue.splice(0, this.options.batchSize);
            
            batch.forEach(({ message, options }) => {
                this.sendMessage(message, options);
            });

            // Small delay between batches to prevent overwhelming
            if (this.messageQueue.length > 0) {
                setTimeout(() => this.flushMessageQueue(), 10);
                break;
            }
        }
    }

    /**
     * Start heartbeat mechanism
     */
    startHeartbeat() {
        this.stopHeartbeat(); // Clear any existing heartbeat
        
        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected) {
                const pingMessage = {
                    type: 'ping',
                    data: { timestamp: new Date().toISOString() }
                };

                this.pendingMessages.set('ping', Date.now());
                this.send(pingMessage, { retry: false, debounce: false });
            }
        }, this.options.heartbeatInterval);
    }

    /**
     * Stop heartbeat mechanism
     */
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }

    /**
     * Subscribe to message type
     */
    on(messageType, handler) {
        if (!this.eventHandlers.has(messageType)) {
            this.eventHandlers.set(messageType, []);
        }
        this.eventHandlers.get(messageType).push(handler);
    }

    /**
     * Unsubscribe from message type
     */
    off(messageType, handler = null) {
        if (!this.eventHandlers.has(messageType)) {
            return;
        }

        if (handler) {
            const handlers = this.eventHandlers.get(messageType);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        } else {
            this.eventHandlers.delete(messageType);
        }
    }

    /**
     * Subscribe to server-side channel
     */
    subscribe(channel) {
        this.send({
            type: 'subscribe',
            data: { channel }
        });
    }

    /**
     * Unsubscribe from server-side channel
     */
    unsubscribe(channel) {
        this.send({
            type: 'unsubscribe',
            data: { channel }
        });
    }

    /**
     * Send upload progress update
     */
    sendUploadProgress(progressData) {
        this.send({
            type: 'upload_progress',
            data: progressData
        }, { priority: 'high', debounce: true });
    }

    /**
     * Send processing status update
     */
    sendProcessingStatus(statusData) {
        this.send({
            type: 'processing_status',
            data: statusData
        }, { priority: 'high', debounce: true });
    }

    /**
     * Send viral insights update
     */
    sendViralInsights(insightsData) {
        this.send({
            type: 'viral_insights',
            data: insightsData
        }, { priority: 'normal', debounce: true });
    }

    /**
     * Handle errors
     */
    handleError(error) {
        console.error('❌ WebSocket error:', error);
        this.metrics.errors++;
        
        if (this.onError) {
            this.onError(error);
        }
    }

    /**
     * Generate unique message ID
     */
    generateMessageId() {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Get connection metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            isConnected: this.isConnected,
            connectionId: this.connectionId,
            reconnectAttempts: this.reconnectAttempts,
            queuedMessages: this.messageQueue.length,
            debouncedMessages: this.debouncedMessages.size
        };
    }

    /**
     * Disconnect WebSocket
     */
    disconnect() {
        if (this.websocket) {
            this.stopHeartbeat();
            this.websocket.close(1000, 'Client disconnect');
            this.websocket = null;
            this.isConnected = false;
            this.connectionId = null;
        }
    }
}

// Export for use in other modules
window.NetflixWebSocketClient = NetflixWebSocketClient;
