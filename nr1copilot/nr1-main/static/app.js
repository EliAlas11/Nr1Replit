/**
 * ViralClip Pro - Netflix-Level Frontend Application v4.0
 * Production-ready with comprehensive error handling and real-time features
 */

class ViralClipApp {
    constructor() {
        this.config = {
            apiBase: window.location.origin,
            wsBase: window.location.origin.replace('http', 'ws'),
            version: '4.0.0',
            features: {
                realtime: true,
                websockets: true,
                preview: true,
                analytics: true,
                livePreview: true,
                interactiveTimeline: true
            }
        };

        this.state = {
            currentSession: null,
            uploadProgress: 0,
            processingStatus: 'idle',
            isConnected: false,
            errors: [],
            metrics: {},
            timeline: null,
            currentPreview: null,
            selectedClips: new Set(),
            playbackTime: 0,
            isPlaying: false
        };

        this.websockets = new Map();
        this.eventListeners = new Map();
        this.uploadQueue = [];
        this.retryAttempts = 0;
        this.maxRetries = 3;
        this.previewCache = new Map();
        this.timelinePlayer = null;

        this.initializeApp();
    }

    async initializeApp() {
        try {
            console.log('🚀 Initializing ViralClip Pro v4.0 with Netflix-level features...');

            // Initialize core components
            await this.setupEventListeners();
            await this.initializeUI();
            await this.setupWebSockets();
            await this.loadUserPreferences();
            await this.initializeRealTimePreview();
            await this.initializeInteractiveTimeline();

            // Performance monitoring
            this.startPerformanceMonitoring();

            // Health check
            await this.performHealthCheck();

            this.logEvent('app_initialized_v4', {
                version: this.config.version,
                features: this.config.features,
                timestamp: Date.now()
            });

            console.log('✅ ViralClip Pro v4.0 initialized successfully');

        } catch (error) {
            this.handleError('App initialization failed', error);
        }
    }

    async setupEventListeners() {
        // File upload events
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        if (uploadArea) {
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadArea.addEventListener('drop', this.handleFileDrop.bind(this));
            uploadArea.addEventListener('click', () => fileInput?.click());
        }

        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Processing controls
        document.addEventListener('click', this.handleButtonClick.bind(this));
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));

        // Timeline events
        const timelineContainer = document.getElementById('timelineContainer');
        if (timelineContainer) {
            timelineContainer.addEventListener('click', this.handleTimelineClick.bind(this));
            timelineContainer.addEventListener('mousemove', this.handleTimelineHover.bind(this));
        }

        // Preview events
        const previewContainer = document.getElementById('previewContainer');
        if (previewContainer) {
            previewContainer.addEventListener('timeupdate', this.handlePreviewTimeUpdate.bind(this));
        }

        // Window events
        window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
        window.addEventListener('online', this.handleOnline.bind(this));
        window.addEventListener('offline', this.handleOffline.bind(this));

        // Error handling
        window.addEventListener('error', this.handleGlobalError.bind(this));
        window.addEventListener('unhandledrejection', this.handleUnhandledRejection.bind(this));
    }

    async initializeUI() {
        // Update UI with current state
        this.updateUploadUI();
        this.updateProcessingUI();
        this.updateConnectionStatus();
        this.initializeProgressBars();
        this.setupTooltips();

        // Initialize timeline UI
        this.initializeTimelineUI();

        // Initialize preview UI
        this.initializePreviewUI();
    }

    async initializeRealTimePreview() {
        try {
            console.log('🎬 Initializing Netflix-level real-time preview system...');

            this.previewSystem = {
                activePreview: null,
                previewQueue: [],
                isGenerating: false,
                quality: 'preview',
                autoGenerate: true,
                streamingEnabled: true
            };

            // Setup preview container
            await this.setupPreviewContainer();

            // Initialize streaming capabilities
            await this.initializePreviewStreaming();

            console.log('✅ Real-time preview system initialized');
        } catch (error) {
            console.error('❌ Real-time preview initialization failed:', error);
        }
    }

    async initializeInteractiveTimeline() {
        try {
            console.log('🎯 Initializing Netflix-level interactive timeline...');

            this.timelineSystem = {
                canvas: null,
                ctx: null,
                width: 0,
                height: 100,
                viralHeatmap: [],
                keyMoments: [],
                playheadPosition: 0,
                isInteracting: false,
                zoomLevel: 1,
                selectedRange: null,
                animations: new Map()
            };

            // Setup timeline canvas
            await this.setupTimelineCanvas();

            // Initialize viral score visualization
            await this.initializeViralScoreVisualization();

            // Setup interactive controls
            await this.setupTimelineControls();

            console.log('✅ Interactive timeline initialized');
        } catch (error) {
            console.error('❌ Interactive timeline initialization failed:', error);
        }
    }

    async setupWebSockets() {
        try {
            // Main application WebSocket
            await this.connectWebSocket('main', '/api/v3/ws/app');

            this.state.isConnected = true;
            this.updateConnectionStatus();

        } catch (error) {
            console.warn('WebSocket setup failed:', error);
            this.state.isConnected = false;
            this.updateConnectionStatus();
        }
    }

    async connectWebSocket(type, endpoint) {
        try {
            const wsUrl = `${this.config.wsBase}${endpoint}`;
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log(`✅ WebSocket connected: ${type}`);
                this.websockets.set(type, ws);
                this.retryAttempts = 0; // Reset retry counter on successful connection
            };

            ws.onmessage = (event) => {
                this.handleWebSocketMessage(type, event);
            };

            ws.onclose = (event) => {
                console.log(`🔌 WebSocket disconnected: ${type}`, event.code);
                this.websockets.delete(type);
                this.scheduleReconnect(type, endpoint);
            };

            ws.onerror = (error) => {
                console.error(`❌ WebSocket error (${type}):`, error);
                this.handleWebSocketError(type, error);
            };

        } catch (error) {
            console.error(`Failed to connect WebSocket (${type}):`, error);
            throw error;
        }
    }

    handleWebSocketMessage(type, event) {
        try {
            const data = JSON.parse(event.data);

            switch (data.type) {
                case 'upload_progress':
                    this.updateUploadProgress(data.progress);
                    break;
                case 'processing_status':
                    this.updateProcessingStatus(data);
                    break;
                case 'viral_score_update':
                    this.updateViralScore(data);
                    break;
                case 'timeline_update':
                    this.updateTimeline(data);
                    break;
                case 'preview_ready':
                    this.handlePreviewReady(data);
                    break;
                case 'live_preview_data':
                    this.handleLivePreviewData(data);
                    break;
                case 'interactive_timeline_data':
                    this.handleInteractiveTimelineData(data);
                    break;
                case 'entertainment_update':
                    this.handleEntertainmentUpdate(data);
                    break;
                case 'error':
                    this.handleError('WebSocket error', data.error);
                    break;
                default:
                    console.log(`Unhandled WebSocket message: ${data.type}`);
            }

        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }

    // Netflix-Level Real-Time Preview System
    async generateLivePreview(startTime, endTime, quality = 'preview') {
        try {
            if (!this.state.currentSession) {
                throw new Error('No active session');
            }

            console.log(`🎬 Generating live preview: ${startTime}-${endTime}s`);

            // Show preview loading state
            this.showPreviewLoading();

            // Check cache first
            const cacheKey = `${this.state.currentSession}_${startTime}_${endTime}_${quality}`;
            if (this.previewCache.has(cacheKey)) {
                const cachedPreview = this.previewCache.get(cacheKey);
                this.displayPreview(cachedPreview);
                return cachedPreview;
            }

            // Request live preview generation
            const response = await fetch('/api/v3/generate-live-preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.state.currentSession,
                    start_time: startTime,
                    end_time: endTime,
                    quality: quality
                })
            });

            if (!response.ok) {
                throw new Error(`Preview generation failed: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.success) {
                // Cache the preview
                this.previewCache.set(cacheKey, result);

                // Display the preview
                this.displayPreview(result);

                // Update viral analysis display
                this.updatePreviewAnalysis(result.viral_analysis);

                return result;
            } else {
                throw new Error(result.error || 'Preview generation failed');
            }

        } catch (error) {
            this.handleError('Live preview generation failed', error);
            this.hidePreviewLoading();
        }
    }

    displayPreview(previewData) {
        const previewContainer = document.getElementById('livePreviewContainer');
        if (!previewContainer) return;

        // Create video element
        const videoElement = document.createElement('video');
        videoElement.src = previewData.preview_url;
        videoElement.controls = true;
        videoElement.autoplay = false;
        videoElement.className = 'live-preview-video';

        // Clear container and add video
        previewContainer.innerHTML = '';
        previewContainer.appendChild(videoElement);

        // Add preview metadata
        this.addPreviewMetadata(previewContainer, previewData);

        // Update state
        this.state.currentPreview = previewData;

        // Hide loading state
        this.hidePreviewLoading();

        console.log('✅ Preview displayed successfully');
    }

    addPreviewMetadata(container, previewData) {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'preview-metadata';

        const viralScore = previewData.viral_analysis?.viral_score || 0;
        const duration = previewData.duration || 0;
        const suggestions = previewData.suggestions || [];

        metadataDiv.innerHTML = `
            <div class="preview-stats">
                <div class="stat">
                    <span class="stat-label">Viral Score:</span>
                    <span class="stat-value viral-score-${this.getScoreClass(viralScore)}">${viralScore}/100</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Duration:</span>
                    <span class="stat-value">${duration.toFixed(1)}s</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Quality:</span>
                    <span class="stat-value">${previewData.quality}</span>
                </div>
            </div>
            <div class="preview-suggestions">
                <h4>AI Suggestions:</h4>
                <ul>
                    ${suggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
                </ul>
            </div>
        `;

        container.appendChild(metadataDiv);
    }

    showPreviewLoading() {
        const previewContainer = document.getElementById('livePreviewContainer');
        if (previewContainer) {
            previewContainer.innerHTML = `
                <div class="preview-loading">
                    <div class="loading-spinner"></div>
                    <p>Generating live preview...</p>
                    <div class="loading-progress">
                        <div class="progress-bar" id="previewGenerationProgress"></div>
                    </div>
                </div>
            `;
        }
    }

    hidePreviewLoading() {
        const loadingElement = document.querySelector('.preview-loading');
        if (loadingElement) {
            loadingElement.remove();
        }
    }

    // Netflix-Level Interactive Timeline with Viral Score Visualization
    async setupTimelineCanvas() {
        const timelineContainer = document.getElementById('timelineVisualization');
        if (!timelineContainer) return;

        // Create canvas element
        const canvas = document.createElement('canvas');
        canvas.id = 'timelineCanvas';
        canvas.className = 'timeline-canvas';

        // Set canvas size
        const containerRect = timelineContainer.getBoundingClientRect();
        canvas.width = containerRect.width || 800;
        canvas.height = this.timelineSystem.height;

        // Get context
        const ctx = canvas.getContext('2d');

        // Store references
        this.timelineSystem.canvas = canvas;
        this.timelineSystem.ctx = ctx;
        this.timelineSystem.width = canvas.width;

        // Add to container
        timelineContainer.innerHTML = '';
        timelineContainer.appendChild(canvas);

        // Setup canvas events
        this.setupCanvasEvents(canvas);
    }

    setupCanvasEvents(canvas) {
        let isDragging = false;
        let lastX = 0;

        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastX = e.clientX;
            this.timelineSystem.isInteracting = true;
            this.handleTimelineInteraction(e);
        });

        canvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const deltaX = e.clientX - lastX;
                this.handleTimelineDrag(deltaX);
                lastX = e.clientX;
            }
            this.handleTimelineHover(e);
        });

        canvas.addEventListener('mouseup', () => {
            isDragging = false;
            this.timelineSystem.isInteracting = false;
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.handleTimelineZoom(e.deltaY, e.clientX);
        });

        canvas.addEventListener('click', (e) => {
            this.handleTimelineClick(e);
        });
    }

    async initializeViralScoreVisualization() {
        // Initialize viral score gradient
        this.viralScoreGradient = {
            low: '#ef4444',      // Red
            medium: '#f59e0b',   // Orange
            high: '#10b981',     // Green
            excellent: '#8b5cf6' // Purple
        };

        // Setup real-time viral score updates
        this.setupViralScoreUpdates();
    }

    setupViralScoreUpdates() {
        // Connect to viral score WebSocket if session exists
        if (this.state.currentSession) {
            this.connectViralScoreWebSocket(this.state.currentSession);
        }
    }

    async connectViralScoreWebSocket(sessionId) {
        await this.connectWebSocket('viral_scores', `/api/v3/ws/viral-scores/${sessionId}`);
    }

    drawTimelineVisualization() {
        if (!this.timelineSystem.ctx) return;

        const ctx = this.timelineSystem.ctx;
        const canvas = this.timelineSystem.canvas;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw viral score heatmap
        this.drawViralHeatmap(ctx);

        // Draw key moments
        this.drawKeyMoments(ctx);

        // Draw playhead
        this.drawPlayhead(ctx);

        // Draw selection range
        this.drawSelectionRange(ctx);

        // Draw hover indicator
        this.drawHoverIndicator(ctx);
    }

    drawViralHeatmap(ctx) {
        const heatmap = this.timelineSystem.viralHeatmap;
        if (!heatmap || heatmap.length === 0) return;

        const width = this.timelineSystem.width;
        const height = this.timelineSystem.height;
        const segmentWidth = width / heatmap.length;

        heatmap.forEach((score, index) => {
            const x = index * segmentWidth;
            const normalizedScore = score / 100;
            const barHeight = normalizedScore * (height * 0.8);

            // Get color based on score
            const color = this.getViralScoreColor(score);

            // Draw bar
            ctx.fillStyle = color;
            ctx.fillRect(x, height - barHeight, segmentWidth - 1, barHeight);

            // Add glow effect for high scores
            if (score > 80) {
                ctx.shadowColor = color;
                ctx.shadowBlur = 5;
                ctx.fillRect(x, height - barHeight, segmentWidth - 1, barHeight);
                ctx.shadowBlur = 0;
                ctx.shadowOffsetY = 0;
            }
        });
    }

    drawKeyMoments(ctx) {
        const moments = this.timelineSystem.keyMoments;
        if (!moments || moments.length === 0) return;

        const width = this.timelineSystem.width;
        const height = this.timelineSystem.height;

        moments.forEach(moment => {
            const x = (moment.timestamp / this.state.timeline?.duration || 1) * width;

            // Draw vertical line
            ctx.strokeStyle = '#ff6b6b';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw moment indicator
            ctx.fillStyle = '#ff6b6b';
            ctx.beginPath();
            ctx.arc(x, 10, 4, 0, 2 * Math.PI);
            ctx.fill();

            // Add tooltip on hover
            this.addMomentTooltip(x, moment);
        });
    }

    drawPlayhead(ctx) {
        const width = this.timelineSystem.width;
        const height = this.timelineSystem.height;
        const duration = this.state.timeline?.duration || 1;
        const x = (this.timelineSystem.playheadPosition / duration) * width;

        // Draw playhead line
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        // Draw playhead handle
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(x, height / 2, 6, 0, 2 * Math.PI);
        ctx.fill();

        // Add shadow
        ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        ctx.shadowBlur = 4;
        ctx.shadowOffsetY = 2;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.shadowOffsetY = 0;
    }

    drawSelectionRange(ctx) {
        if (!this.timelineSystem.selectedRange) return;

        const width = this.timelineSystem.width;
        const height = this.timelineSystem.height;
        const duration = this.state.timeline?.duration || 1;
        const { start, end } = this.timelineSystem.selectedRange;

        const startX = (start / duration) * width;
        const endX = (end / duration) * width;

        // Draw selection overlay
        ctx.fillStyle = 'rgba(138, 92, 246, 0.3)';
        ctx.fillRect(startX, 0, endX - startX, height);

        // Draw selection borders
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, 0, endX - startX, height);
    }

    drawHoverIndicator(ctx) {
        // This will be updated in handleTimelineHover
    }

    getViralScoreColor(score) {
        if (score >= 90) return '#8b5cf6'; // Excellent - Purple
        if (score >= 75) return '#10b981'; // High - Green
        if (score >= 50) return '#f59e0b'; // Medium - Orange
        return '#ef4444'; // Low - Red
    }

    // Netflix-Level Live Processing Status with Entertaining Content
    updateProcessingStatus(data) {
        const statusElement = document.getElementById('processingStatus');
        const messageElement = document.getElementById('processingMessage');
        const progressElement = document.getElementById('processingProgress');
        const entertainmentElement = document.getElementById('entertainingContent');

        if (statusElement) {
            statusElement.textContent = data.stage;
            statusElement.className = `processing-stage stage-${data.stage.replace(/\s+/g, '-').toLowerCase()}`;
        }

        if (messageElement) {
            messageElement.textContent = data.message;
        }

        if (progressElement) {
            progressElement.style.width = `${data.progress}%`;
            progressElement.setAttribute('data-progress', data.progress);
        }

        // Display entertaining content
        if (data.entertaining_fact && entertainmentElement) {
            this.displayEntertainingContent(data.entertaining_fact, entertainmentElement);
        }

        // Add processing animations
        this.addProcessingAnimations(data);

        this.state.processingStatus = data.stage;
    }

    displayEntertainingContent(content, container) {
        // Create animated content display
        const contentDiv = document.createElement('div');
        contentDiv.className = 'entertaining-content-item fade-in';
        contentDiv.innerHTML = `
            <div class="content-icon">🎬</div>
            <div class="content-text">${content}</div>
        `;

        // Clear previous content
        container.innerHTML = '';
        container.appendChild(contentDiv);

        // Auto-hide after 8 seconds with fade out
        setTimeout(() => {
            contentDiv.classList.add('fade-out');
            setTimeout(() => {
                if (contentDiv.parentNode) {
                    contentDiv.remove();
                }
            }, 500);
        }, 8000);

        // Add typing animation effect
        this.addTypingAnimation(contentDiv.querySelector('.content-text'));
    }

    addTypingAnimation(element) {
        const text = element.textContent;
        element.textContent = '';

        let index = 0;
        const typingInterval = setInterval(() => {
            element.textContent += text[index];
            index++;

            if (index >= text.length) {
                clearInterval(typingInterval);
            }
        }, 30); // Typing speed
    }

    addProcessingAnimations(data) {
        const processingContainer = document.getElementById('processingContainer');
        if (!processingContainer) return;

        // Add stage-specific animations
        switch (data.stage) {
            case 'analyzing':
                this.addAnalyzingAnimation(processingContainer);
                break;
            case 'generating_clips':
                this.addGeneratingAnimation(processingContainer);
                break;
            case 'optimizing':
                this.addOptimizingAnimation(processingContainer);
                break;
            case 'complete':
                this.addCompleteAnimation(processingContainer);
                break;
        }
    }

    addAnalyzingAnimation(container) {
        container.classList.add('analyzing');
        // Add scanning line effect
        const scanLine = document.createElement('div');
        scanLine.className = 'scan-line';
        container.appendChild(scanLine);
    }

    addGeneratingAnimation(container) {
        container.classList.add('generating');
        // Add sparkle effects
        for (let i = 0; i < 5; i++) {
            setTimeout(() => {
                this.createSparkle(container);
            }, i * 200);
        }
    }

    addOptimizingAnimation(container) {
        container.classList.add('optimizing');
        // Add gear rotation effect
        const gear = document.createElement('div');
        gear.className = 'rotating-gear';
        gear.innerHTML = '⚙️';
        container.appendChild(gear);
    }

    addCompleteAnimation(container) {
        container.classList.add('complete');
        // Add celebration effect
        this.createConfetti(container);
    }

    createSparkle(container) {
        const sparkle = document.createElement('div');
        sparkle.className = 'sparkle';
        sparkle.innerHTML = '✨';
        sparkle.style.left = Math.random() * 100 + '%';
        sparkle.style.top = Math.random() * 100 + '%';
        container.appendChild(sparkle);

        setTimeout(() => sparkle.remove(), 2000);
    }

    createConfetti(container) {
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b'];

        for (let i = 0; i < 20; i++) {
            setTimeout(() => {
                const confetti = document.createElement('div');
                confetti.className = 'confetti';
                confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
                confetti.style.left = Math.random() * 100 + '%';
                confetti.style.animationDelay = Math.random() * 2 + 's';
                container.appendChild(confetti);

                setTimeout(() => confetti.remove(), 3000);
            }, i * 50);
        }
    }

    // Event Handlers
    handleTimelineClick(event) {
        if (!this.timelineSystem.canvas) return;

        const rect = this.timelineSystem.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const percentage = x / this.timelineSystem.width;

        if (this.state.timeline && this.state.timeline.duration) {
            const timestamp = percentage * this.state.timeline.duration;
            this.seekToTimestamp(timestamp);

            // Generate live preview for clicked segment
            const segmentStart = Math.max(0, timestamp - 5);
            const segmentEnd = Math.min(this.state.timeline.duration, timestamp + 5);
            this.generateLivePreview(segmentStart, segmentEnd);
        }
    }

    handleTimelineHover(event) {
        if (!this.timelineSystem.canvas) return;

        const rect = this.timelineSystem.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const percentage = x / this.timelineSystem.width;

        // Update hover indicator
        this.updateHoverIndicator(x, percentage);

        // Show tooltip with viral score
        this.showTimelineTooltip(event, percentage);
    }

    handleTimelineDrag(deltaX) {
        // Handle timeline dragging for playhead movement
        const newX = this.timelineSystem.playheadPosition + deltaX;
        const clampedX = Math.max(0, Math.min(this.timelineSystem.width, newX));

        if (this.state.timeline) {
            const timestamp = (clampedX / this.timelineSystem.width) * this.state.timeline.duration;
            this.seekToTimestamp(timestamp);
        }
    }

    handleTimelineZoom(deltaY, centerX) {
        const zoomFactor = deltaY > 0 ? 0.9 : 1.1;
        this.timelineSystem.zoomLevel *= zoomFactor;
        this.timelineSystem.zoomLevel = Math.max(0.5, Math.min(5, this.timelineSystem.zoomLevel));

        // Redraw timeline with new zoom level
        this.drawTimelineVisualization();
    }

    handleKeyboardShortcuts(event) {
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return; // Don't handle shortcuts when typing in inputs
        }

        switch (event.key) {
            case ' ':
                event.preventDefault();
                this.togglePlayback();
                break;
            case 'ArrowLeft':
                event.preventDefault();
                this.seekRelative(-5);
                break;
            case 'ArrowRight':
                event.preventDefault();
                this.seekRelative(5);
                break;
            case 'Enter':
                if (this.timelineSystem.selectedRange) {
                    const { start, end } = this.timelineSystem.selectedRange;
                    this.generateLivePreview(start, end);
                }
                break;
        }
    }

    // Utility Methods
    seekToTimestamp(timestamp) {
        this.timelineSystem.playheadPosition = timestamp;
        this.state.playbackTime = timestamp;

        // Update preview if available
        if (this.state.currentPreview) {
            const videoElement = document.querySelector('.live-preview-video');
            if (videoElement) {
                videoElement.currentTime = timestamp;
            }
        }

        // Redraw timeline
        this.drawTimelineVisualization();

        // Broadcast playhead update
        this.broadcastPlayheadUpdate(timestamp);
    }

    seekRelative(seconds) {
        const currentTime = this.timelineSystem.playheadPosition;
        const newTime = Math.max(0, currentTime + seconds);

        if (this.state.timeline) {
            const maxTime = this.state.timeline.duration;
            this.seekToTimestamp(Math.min(newTime, maxTime));
        }
    }

    togglePlayback() {
        this.state.isPlaying = !this.state.isPlaying;

        const videoElement = document.querySelector('.live-preview-video');
        if (videoElement) {
            if (this.state.isPlaying) {
                videoElement.play();
            } else {
                videoElement.pause();
            }
        }

        // Update UI
        this.updatePlaybackControls();
    }

    updatePlaybackControls() {
        const playButton = document.querySelector('[data-action="play-pause"]');
        if (playButton) {
            const icon = playButton.querySelector('span');
            if (icon) {
                icon.textContent = this.state.isPlaying ? '⏸️' : '⏯️';
            }
        }
    }

    broadcastPlayheadUpdate(timestamp) {
        // Broadcast to other components or WebSocket
        if (this.websockets.has('main')) {
            const ws = this.websockets.get('main');
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'playhead_update',
                    timestamp: timestamp,
                    session_id: this.state.currentSession
                }));
            }
        }
    }

    updateHoverIndicator(x, percentage) {
        // Draw hover line on timeline
        if (this.timelineSystem.ctx) {
            this.drawTimelineVisualization();

            const ctx = this.timelineSystem.ctx;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.timelineSystem.height);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    showTimelineTooltip(event, percentage) {
        if (!this.state.timeline) return;

        const timestamp = percentage * this.state.timeline.duration;
        const viralScore = this.getViralScoreAtTime(timestamp);

        // Create or update tooltip
        let tooltip = document.getElementById('timelineTooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'timelineTooltip';
            tooltip.className = 'timeline-tooltip';
            document.body.appendChild(tooltip);
        }

        tooltip.innerHTML = `
            <div class="tooltip-time">${this.formatTime(timestamp)}</div>
            <div class="tooltip-score">Viral Score: ${viralScore}/100</div>
        `;

        // Position tooltip
        tooltip.style.left = event.clientX + 10 + 'px';
        tooltip.style.top = event.clientY - 50 + 'px';
        tooltip.style.display = 'block';

        // Hide tooltip after delay
        clearTimeout(this.tooltipTimeout);
        this.tooltipTimeout = setTimeout(() => {
            tooltip.style.display = 'none';
        }, 3000);
    }

    getViralScoreAtTime(timestamp) {
        if (!this.state.timeline || !this.timelineSystem.viralHeatmap.length) {
            return 50; // Default score
        }

        const duration = this.state.timeline.duration;
        const index = Math.floor((timestamp / duration) * this.timelineSystem.viralHeatmap.length);
        const clampedIndex = Math.max(0, Math.min(this.timelineSystem.viralHeatmap.length - 1, index));

        return this.timelineSystem.viralHeatmap[clampedIndex] || 50;
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    getScoreClass(score) {
        if (score >= 80) return 'excellent';
        if (score >= 60) return 'good';
        if (score >= 40) return 'average';
        return 'poor';
    }

    // WebSocket message handlers
    handleLivePreviewData(data) {
        console.log('📺 Received live preview data:', data);

        if (data.preview_url) {
            this.displayPreview(data);
        }

        if (data.progress !== undefined) {
            this.updatePreviewProgress(data.progress);
        }
    }

    handleInteractiveTimelineData(data) {
        console.log('📊 Received interactive timeline data:', data);

        if (data.viral_heatmap) {
            this.timelineSystem.viralHeatmap = data.viral_heatmap;
        }

        if (data.key_moments) {
            this.timelineSystem.keyMoments = data.key_moments;
        }

        // Redraw timeline with new data
        this.drawTimelineVisualization();
    }

    handleEntertainmentUpdate(data) {
        console.log('🎪 Received entertainment update:', data);

        const entertainmentElement = document.getElementById('entertainingContent');
        if (entertainmentElement && data.content) {
            this.displayEntertainingContent(data.content, entertainmentElement);
        }
    }

    updatePreviewProgress(progress) {
        const progressBar = document.getElementById('previewGenerationProgress');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
    }

    // Existing methods from original class...
    async handleFileDrop(event) {
        event.preventDefault();
        event.stopPropagation();

        const uploadArea = document.getElementById('uploadArea');
        uploadArea?.classList.remove('drag-over');

        const files = Array.from(event.dataTransfer.files);
        await this.processFileUpload(files);
    }

    async handleFileSelect(event) {
        const files = Array.from(event.target.files);
        await this.processFileUpload(files);
    }

    async processFileUpload(files) {
        try {
            if (files.length === 0) return;

            const file = files[0];

            // Validate file
            const validation = this.validateFile(file);
            if (!validation.valid) {
                this.showError(validation.error);
                return;
            }

            // Show upload UI
            this.showUploadUI();

            // Start upload with real-time progress
            const uploadId = this.generateUploadId();
            await this.uploadFileWithProgress(file, uploadId);

        } catch (error) {
            this.handleError('File upload failed', error);
        }
    }

    validateFile(file) {
        const maxSize = 2 * 1024 * 1024 * 1024; // 2GB
        const allowedTypes = ['video/mp4', 'video/mov', 'video/avi', 'video/webm'];

        if (!file) {
            return { valid: false, error: 'No file selected' };
        }

        if (file.size > maxSize) {
            return { valid: false, error: 'File too large (max 2GB)' };
        }

        if (!allowedTypes.includes(file.type)) {
            return { valid: false, error: 'Unsupported file type' };
        }

        return { valid: true };
    }

    async uploadFileWithProgress(file, uploadId) {
        try {
            // Connect upload WebSocket
            await this.connectWebSocket('upload', `/api/v3/ws/upload/${uploadId}`);

            const formData = new FormData();
            formData.append('file', file);
            formData.append('upload_id', uploadId);

            const response = await fetch('/api/v3/upload-video', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.success) {
                this.state.currentSession = result.session_id;
                this.handleUploadSuccess(result);
            } else {
                throw new Error(result.error || 'Upload failed');
            }

        } catch (error) {
            this.handleError('Upload failed', error);
        }
    }

    handleUploadSuccess(result) {
        console.log('Upload successful:', result);

        // Update UI
        this.hideUploadUI();
        this.showTimelineUI(result);

        // Connect additional WebSockets
        this.connectViralScoreWebSocket(result.session_id);
        this.connectTimelineWebSocket(result.session_id);

        // Show success message
        this.showSuccess('Video uploaded successfully! Analyzing for viral potential...');
    }

    async connectTimelineWebSocket(sessionId) {
        await this.connectWebSocket('timeline', `/api/v3/ws/timeline/${sessionId}`);
    }

    // Continue with all existing methods but with improved error handling...
    handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
        event.currentTarget.classList.add('drag-over');
    }

    handleDragLeave(event) {
        event.preventDefault();
        event.stopPropagation();
        event.currentTarget.classList.remove('drag-over');
    }

    handleButtonClick(event) {
        const target = event.target;
        const action = target.dataset.action;

        if (!action) return;

        switch (action) {
            case 'generate-clips':
                this.generateClips();
                break;
            case 'download-clip':
                this.downloadClip(target.dataset.clipId);
                break;
            case 'preview-clip':
                this.previewClip(target.dataset.clipId);
                break;
            case 'play-pause':
                this.togglePlayback();
                break;
            case 'add-clip':
                this.addClipFromSelection();
                break;
            case 'export':
                this.exportSelectedClips();
                break;
            case 'retry':
                this.retryLastOperation();
                break;
            default:
                console.log(`Unhandled action: ${action}`);
        }
    }

    addClipFromSelection() {
        if (this.timelineSystem.selectedRange) {
            const { start, end } = this.timelineSystem.selectedRange;
            this.generateLivePreview(start, end, 'high');
        } else {
            this.showError('Please select a range on the timeline first');
        }
    }

    exportSelectedClips() {
        const selectedClips = this.getSelectedClips();
        if (selectedClips.length === 0) {
            this.showError('No clips selected for export');
            return;
        }

        // Implement export functionality
        console.log('Exporting clips:', selectedClips);
    }

    // Error handling and utility methods
    handleError(message, error) {
        console.error(message, error);

        this.state.errors.push({
            message,
            error: error.toString(),
            timestamp: Date.now()
        });

        this.showError(`${message}: ${error.message || error}`);
    }

    handleGlobalError(event) {
        console.log('Global error:', event.error);
        this.handleError('Unexpected error occurred', event.error);
    }

    handleUnhandledRejection(event) {
        console.log('Unhandled promise rejection:', event.reason);
        this.handleError('Promise rejection', event.reason);
    }

    showError(message) {
        const errorContainer = document.getElementById('errorContainer');
        if (!errorContainer) {
            console.error(message);
            return;
        }

        const errorElement = document.createElement('div');
        errorElement.className = 'error-message fade-in';
        errorElement.textContent = message;

        errorContainer.appendChild(errorElement);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            errorElement.classList.add('fade-out');
            setTimeout(() => errorElement.remove(), 300);
        }, 5000);
    }

    showSuccess(message) {
        const successContainer = document.getElementById('successContainer');
        if (!successContainer) {
            console.log(message);
            return;
        }

        const successElement = document.createElement('div');
        successElement.className = 'success-message fade-in';
        successElement.textContent = message;

        successContainer.appendChild(successElement);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            successElement.classList.add('fade-out');
            setTimeout(() => successElement.remove(), 300);
        }, 3000);
    }

    // Initialization helpers and utility methods...
    generateUploadId() {
        return `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    logEvent(eventName, data) {
        console.log('📊 Event:', eventName, data);

        // Send to analytics if enabled
        if (this.config.features.analytics) {
            this.sendAnalytics(eventName, data);
        }
    }

    sendAnalytics(eventName, data) {
        // Placeholder for analytics integration
        console.log('Analytics:', eventName, data);
    }

    scheduleReconnect(type, endpoint) {
        if (this.retryAttempts >= this.maxRetries) {
            console.error(`Max WebSocket reconnection attempts reached for ${type}`);
            return;
        }

        this.retryAttempts++;
        const delay = Math.pow(2, this.retryAttempts) * 1000; // Exponential backoff

        setTimeout(() => {
            console.log(`Attempting to reconnect WebSocket: ${type}`);
            this.connectWebSocket(type, endpoint);
        }, delay);
    }

    handleWebSocketError(type, error) {
        console.error(`WebSocket error (${type}):`, error);
        this.handleError(`WebSocket connection failed (${type})`, error);
    }

    startPerformanceMonitoring() {
        const domContentLoaded = performance.now();
        console.log('📊 Page Load Performance:');
        console.log('DOM Content Loaded:', domContentLoaded, 'ms');

        window.addEventListener('load', () => {
            const fullLoadTime = performance.now() - domContentLoaded;
            console.log('Full Load Time:', fullLoadTime, 'ms');

            const totalLoadTime = performance.now();
            console.log(`📊 Total Load Time: ${totalLoadTime}ms`);
        });
    }

    async performHealthCheck() {
        try {
            const response = await fetch('/api/v3/health');
            const health = await response.json();

            if (health.status !== 'healthy') {
                console.warn('Application health check failed:', health);
            }

        } catch (error) {
            console.warn('Health check failed:', error);
        }
    }

    async loadUserPreferences() {
        try {
            const preferences = localStorage.getItem('viralclip_preferences');
            if (preferences) {
                const parsed = JSON.parse(preferences);
                this.applyUserPreferences(parsed);
            }
        } catch (error) {
            console.warn('Failed to load user preferences:', error);
        }
    }

    applyUserPreferences(preferences) {
        if (preferences.theme) {
            document.body.classList.add(`theme-${preferences.theme}`);
        }

        if (preferences.quality) {
            const qualitySelect = document.getElementById('qualitySelect');
            if (qualitySelect) {
                qualitySelect.value = preferences.quality;
            }
        }
    }

    initializeProgressBars() {
        const progressBars = document.querySelectorAll('.progress-bar');
        progressBars.forEach(bar => {
            bar.style.transition = 'width 0.3s ease';
        });
    }

    setupTooltips() {
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', this.showTooltip.bind(this));
            element.addEventListener('mouseleave', this.hideTooltip.bind(this));
        });
    }

    showTooltip(event) {
        const text = event.target.dataset.tooltip;
        if (!text) return;

        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = text;
        document.body.appendChild(tooltip);

        const rect = event.target.getBoundingClientRect();
        tooltip.style.left = rect.left + rect.width / 2 + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + 'px';

        event.target._tooltip = tooltip;
    }

    hideTooltip(event) {
        if (event.target._tooltip) {
            event.target._tooltip.remove();
            event.target._tooltip = null;
        }
    }

    initializeTimelineUI() {
        // Initialize timeline-specific UI elements
        const timelineSection = document.getElementById('timelineSection');
        if (timelineSection) {
            timelineSection.style.display = 'none'; // Hidden until video is uploaded
        }
    }

    initializePreviewUI() {
        // Initialize preview-specific UI elements
        const previewSection = document.getElementById('previewSection');
        if (previewSection) {
            previewSection.style.display = 'none'; // Hidden until preview is generated
        }
    }

    updateUploadUI() {
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea && this.state.uploadProgress > 0) {
            uploadArea.classList.add('uploading');
        }
    }

    updateProcessingUI() {
        const processingSection = document.getElementById('processingSection');
        if (processingSection && this.state.processingStatus !== 'idle') {
            processingSection.style.display = 'block';
        }
    }

    updateConnectionStatus() {
        const statusIndicator = document.getElementById('connectionStatus');
        if (statusIndicator) {
            statusIndicator.className = this.state.isConnected ? 'connected' : 'disconnected';
            statusIndicator.textContent = this.state.isConnected ? 'Connected' : 'Disconnected';
        }
    }

    showUploadUI() {
        const uploadSection = document.getElementById('uploadSection');
        const processingSection = document.getElementById('processingSection');

        if (uploadSection) uploadSection.style.display = 'block';
        if (processingSection) processingSection.style.display = 'none';
    }

    hideUploadUI() {
        const uploadSection = document.getElementById('uploadSection');
        if (uploadSection) uploadSection.style.display = 'none';
    }

    showTimelineUI(data) {
        const timelineSection = document.getElementById('timelineSection');
        const processingSection = document.getElementById('processingSection');

        if (timelineSection) timelineSection.style.display = 'block';
        if (processingSection) processingSection.style.display = 'block';
    }

    updateUploadProgress(progress) {
        const progressBar = document.getElementById('uploadProgress');
        const progressText = document.getElementById('uploadProgressText');

        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }

        if (progressText) {
            progressText.textContent = `${Math.round(progress)}%`;
        }

        this.state.uploadProgress = progress;
    }

    updateViralScore(data) {
        const scoreElement = document.getElementById('viralScore');
        const confidenceElement = document.getElementById('confidence');

        if (scoreElement) {
            scoreElement.textContent = data.viral_score;
            scoreElement.className = this.getScoreClass(data.viral_score);
        }

        if (confidenceElement) {
            confidenceElement.textContent = `${Math.round(data.confidence * 100)}%`;
        }

        // Update factors list
        this.updateFactorsList(data.factors);
    }

    updateTimeline(data) {
        const timelineContainer = document.getElementById('timelineContainer');
        if (!timelineContainer) return;

        // Update timeline system data
        if (data.viral_heatmap) {
            this.timelineSystem.viralHeatmap = data.viral_heatmap;
        }

        if (data.key_moments) {
            this.timelineSystem.keyMoments = data.key_moments;
        }

        // Render updated timeline
        this.drawTimelineVisualization();
        this.state.timeline = data;
    }

    updateFactorsList(factors) {
        const factorsContainer = document.getElementById('viralFactors');
        if (!factorsContainer) return;

        factorsContainer.innerHTML = '';

        factors.forEach(factor => {
            const factorElement = document.createElement('div');
            factorElement.className = 'viral-factor';
            factorElement.textContent = factor;
            factorsContainer.appendChild(factorElement);
        });
    }

    handleBeforeUnload() {
        // Close WebSocket connections
        this.websockets.forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        });

        // Save user preferences
        this.saveUserPreferences();
    }

    handleOnline() {
        console.log('Connection restored');
        this.state.isConnected = true;
        this.updateConnectionStatus();
        this.setupWebSockets();
    }

    handleOffline() {
        console.log('Connection lost');
        this.state.isConnected = false;
        this.updateConnectionStatus();
    }

    saveUserPreferences() {
        try {
            const preferences = {
                theme: document.body.classList.contains('theme-dark') ? 'dark' : 'light',
                quality: document.getElementById('qualitySelect')?.value || 'high'
            };

            localStorage.setItem('viralclip_preferences', JSON.stringify(preferences));
        } catch (error) {
            console.warn('Failed to save user preferences:', error);
        }
    }

    // Add missing methods referenced in existing code
    async generateClips() {
        try {
            if (!this.state.currentSession) {
                throw new Error('No active session');
            }

            const clips = this.getSelectedClips();
            if (clips.length === 0) {
                this.showError('Please select at least one clip to generate');
                return;
            }

            const response = await fetch('/api/v3/process-clips', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.state.currentSession,
                    clips: clips,
                    options: this.getProcessingOptions()
                })
            });

            const result = await response.json();

            if (result.success) {
                this.handleClipGenerationStart(result);
            } else {
                throw new Error(result.error);
            }

        } catch (error) {
            this.handleError('Clip generation failed', error);
        }
    }

    getSelectedClips() {
        const clipElements = document.querySelectorAll('.clip-item.selected');
        return Array.from(clipElements).map(element => ({
            start_time: parseFloat(element.dataset.startTime),
            end_time: parseFloat(element.dataset.endTime),
            title: element.dataset.title
        }));
    }

    getProcessingOptions() {
        return {
            quality: document.getElementById('qualitySelect')?.value || 'high',
            platforms: this.getSelectedPlatforms()
        };
    }

    getSelectedPlatforms() {
        const platformElements = document.querySelectorAll('.platform-option:checked');
        return Array.from(platformElements).map(el => el.value);
    }

    handleClipGenerationStart(result) {
        console.log('Clip generation started:', result);
        this.showSuccess('Clip generation started! Check processing status for updates.');
    }

    downloadClip(clipId) {
        if (!clipId) return;

        const downloadUrl = `/api/v3/download/${clipId}`;
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `clip_${clipId}.mp4`;
        link.click();
    }

    previewClip(clipId) {
        if (!clipId) return;

        const previewUrl = `/api/v3/preview/${clipId}`;
        // Implement preview functionality
        console.log('Previewing clip:', clipId, previewUrl);
    }

    retryLastOperation() {
        // Implement retry functionality
        console.log('Retrying last operation...');
    }

    setupPreviewContainer() {
        const container = document.getElementById('livePreviewContainer');
        if (container) {
            container.innerHTML = `
                <div class="preview-placeholder">
                    <div class="placeholder-icon">🎬</div>
                    <p>Select a timeline segment to generate live preview</p>
                </div>
            `;
        }
    }

    async initializePreviewStreaming() {
        // Initialize streaming capabilities for live previews
        this.streamingConfig = {
            enabled: true,
            quality: 'preview',
            bufferSize: 1024 * 1024, // 1MB buffer
            chunkSize: 8192 // 8KB chunks
        };
    }

    setupTimelineControls() {
        // Setup timeline control buttons
        const controls = document.querySelectorAll('.timeline-btn');
        controls.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = btn.dataset.action;
                this.handleTimelineControlAction(action);
            });
        });
    }

    handleTimelineControlAction(action) {
        switch (action) {
            case 'play-pause':
                this.togglePlayback();
                break;
            case 'add-clip':
                this.addClipFromSelection();
                break;
            case 'export':
                this.exportSelectedClips();
                break;
        }
    }

    addMomentTooltip(x, moment) {
        // This method would add hover tooltips for key moments
        // Implementation would involve creating tooltip elements
    }

    handlePreviewReady(data) {
        console.log('Preview ready:', data);
        if (data.preview_url) {
            this.displayPreview(data);
        }
    }

    updatePreviewAnalysis(analysis) {
        const analysisContainer = document.getElementById('previewAnalysis');
        if (analysisContainer && analysis) {
            analysisContainer.innerHTML = `
                <div class="analysis-item">
                    <span class="label">Viral Score:</span>
                    <span class="value">${analysis.viral_score}/100</span>
                </div>
                <div class="analysis-item">
                    <span class="label">Confidence:</span>
                    <span class="value">${Math.round(analysis.confidence * 100)}%</span>
                </div>
            `;
        }
    }

    handlePreviewTimeUpdate(event) {
        const video = event.target;
        if (video && this.state.timeline) {
            const currentTime = video.currentTime;
            this.seekToTimestamp(currentTime);
        }
    }
}

// Initialize the application
let app;

document.addEventListener('DOMContentLoaded', () => {
    try {
        app = new ViralClipApp();
    } catch (error) {
        console.error('Failed to initialize ViralClip Pro:', error);
    }
});

// Export for global access
if (typeof window !== 'undefined') {
    window.ViralClipApp = ViralClipApp;
}