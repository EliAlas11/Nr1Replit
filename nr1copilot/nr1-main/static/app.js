
/**
 * ViralClip Pro v4.0 - Netflix-Level Client Application
 * Real-time instant feedback system with advanced WebSocket integration
 */

class NetflixLevelApp {
    constructor() {
        this.ws = null;
        this.uploadWs = null;
        this.currentSession = null;
        this.viralTimeline = null;
        this.previewPlayer = null;
        this.processingStatus = null;
        this.entertainmentFacts = [];
        this.animationFrameId = null;
        
        // Performance monitoring
        this.performanceMetrics = {
            previewGenerationTime: [],
            websocketLatency: [],
            timelineRenderTime: []
        };

        // Cache for instant feedback
        this.previewCache = new Map();
        this.timelineCache = new Map();
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing ViralClip Pro v4.0 - Netflix-Level Experience');
        
        this.setupUI();
        this.setupWebSocket();
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.setupKeyboardShortcuts();
        
        // Initialize performance monitoring
        this.initPerformanceMonitoring();
        
        console.log('✅ Netflix-Level App initialized successfully');
    }

    setupUI() {
        // Create main container if it doesn't exist
        if (!document.getElementById('app-container')) {
            document.body.innerHTML = `
                <div id="app-container" class="netflix-app">
                    <header class="app-header">
                        <div class="header-content">
                            <h1 class="app-title">
                                <span class="logo">🎬</span>
                                ViralClip Pro v4.0
                                <span class="version-badge">Netflix-Level</span>
                            </h1>
                            <div class="header-stats">
                                <div class="stat-item">
                                    <span class="stat-label">Processing</span>
                                    <span class="stat-value" id="processing-count">0</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Queue</span>
                                    <span class="stat-value" id="queue-count">0</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">WebSocket</span>
                                    <span class="stat-value websocket-status" id="ws-status">Connecting...</span>
                                </div>
                            </div>
                        </div>
                    </header>

                    <main class="app-main">
                        <div class="upload-section">
                            <div class="upload-zone" id="uploadZone">
                                <div class="upload-content">
                                    <div class="upload-icon">📤</div>
                                    <h2>Drop Your Video Here</h2>
                                    <p>Netflix-level AI will analyze for viral potential instantly</p>
                                    <input type="file" id="fileInput" accept="video/*" style="display: none;">
                                    <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                                        Choose Video File
                                    </button>
                                </div>
                                <div class="upload-progress" id="uploadProgress" style="display: none;">
                                    <div class="progress-bar">
                                        <div class="progress-fill" id="progressFill"></div>
                                    </div>
                                    <div class="progress-text" id="progressText">Uploading...</div>
                                    <div class="upload-stats" id="uploadStats"></div>
                                </div>
                            </div>
                        </div>

                        <div class="analysis-section" id="analysisSection" style="display: none;">
                            <div class="analysis-header">
                                <div class="viral-score-display">
                                    <div class="score-circle" id="viralScoreCircle">
                                        <div class="score-number" id="viralScoreNumber">--</div>
                                        <div class="score-label">Viral Score</div>
                                    </div>
                                    <div class="score-details">
                                        <div class="confidence-bar">
                                            <span class="confidence-label">Confidence</span>
                                            <div class="confidence-meter">
                                                <div class="confidence-fill" id="confidenceFill"></div>
                                            </div>
                                            <span class="confidence-value" id="confidenceValue">--</span>
                                        </div>
                                        <div class="viral-factors" id="viralFactors"></div>
                                    </div>
                                </div>
                                <div class="analysis-actions">
                                    <button class="action-btn primary" id="generateClipsBtn">
                                        🎯 Generate Viral Clips
                                    </button>
                                    <button class="action-btn secondary" id="exportTimelineBtn">
                                        📊 Export Analysis
                                    </button>
                                </div>
                            </div>

                            <div class="timeline-container">
                                <div class="timeline-header">
                                    <h3>Interactive Viral Timeline</h3>
                                    <div class="timeline-controls">
                                        <button class="timeline-btn" id="playBtn">▶️</button>
                                        <button class="timeline-btn" id="pauseBtn">⏸️</button>
                                        <span class="timeline-time" id="currentTime">0:00</span>
                                        <span class="timeline-separator">/</span>
                                        <span class="timeline-time" id="totalTime">0:00</span>
                                    </div>
                                </div>
                                <div class="timeline-wrapper">
                                    <canvas id="viralTimeline" class="viral-timeline-canvas"></canvas>
                                    <div class="timeline-markers" id="timelineMarkers"></div>
                                    <div class="playhead" id="playhead"></div>
                                </div>
                                <div class="timeline-legend">
                                    <div class="legend-item">
                                        <div class="legend-color" style="background: #ff4444;"></div>
                                        <span>Low Viral (0-40)</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color" style="background: #ffaa00;"></div>
                                        <span>Medium Viral (40-70)</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color" style="background: #00ff44;"></div>
                                        <span>High Viral (70-100)</span>
                                    </div>
                                </div>
                            </div>

                            <div class="preview-section">
                                <div class="preview-header">
                                    <h3>Live Preview Generator</h3>
                                    <div class="preview-controls">
                                        <label>Quality:</label>
                                        <select id="previewQuality">
                                            <option value="draft">Draft (Fast)</option>
                                            <option value="standard" selected>Standard</option>
                                            <option value="high">High Quality</option>
                                            <option value="premium">Premium (4K)</option>
                                        </select>
                                        <label>Platform:</label>
                                        <select id="platformOptimization">
                                            <option value="">Auto-detect</option>
                                            <option value="tiktok">TikTok</option>
                                            <option value="instagram">Instagram</option>
                                            <option value="youtube_shorts">YouTube Shorts</option>
                                            <option value="twitter">Twitter</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="preview-area">
                                    <div class="preview-player" id="previewPlayer">
                                        <div class="preview-placeholder">
                                            <div class="placeholder-icon">🎬</div>
                                            <p>Select a segment on the timeline to generate instant preview</p>
                                        </div>
                                    </div>
                                    <div class="preview-info" id="previewInfo"></div>
                                </div>
                            </div>
                        </div>

                        <div class="processing-status" id="processingStatus" style="display: none;">
                            <div class="processing-header">
                                <h3>Live Processing Status</h3>
                                <div class="processing-controls">
                                    <button class="processing-btn" id="cancelProcessing">Cancel</button>
                                </div>
                            </div>
                            <div class="processing-content">
                                <div class="current-task">
                                    <div class="task-icon" id="taskIcon">⚡</div>
                                    <div class="task-details">
                                        <div class="task-name" id="taskName">Initializing...</div>
                                        <div class="task-description" id="taskDescription">Setting up Netflix-level processing pipeline</div>
                                    </div>
                                    <div class="task-progress">
                                        <div class="progress-ring" id="progressRing">
                                            <svg class="progress-svg" width="60" height="60">
                                                <circle class="progress-circle-bg" cx="30" cy="30" r="25"></circle>
                                                <circle class="progress-circle-fill" cx="30" cy="30" r="25" id="progressCircle"></circle>
                                            </svg>
                                            <div class="progress-text" id="progressPercentage">0%</div>
                                        </div>
                                    </div>
                                </div>
                                <div class="entertainment-section">
                                    <div class="entertainment-content" id="entertainmentContent">
                                        <div class="entertainment-icon">🎭</div>
                                        <div class="entertainment-text" id="entertainmentText">
                                            Did you know? Netflix processes over 1 billion hours of content daily! 📺
                                        </div>
                                    </div>
                                </div>
                                <div class="processing-queue" id="processingQueue"></div>
                            </div>
                        </div>
                    </main>

                    <footer class="app-footer">
                        <div class="footer-content">
                            <div class="performance-stats" id="performanceStats"></div>
                            <div class="version-info">
                                ViralClip Pro v4.0 | Netflix-Level AI Processing
                            </div>
                        </div>
                    </footer>
                </div>
            `;
        }

        // Initialize UI components
        this.initTimelineCanvas();
        this.initPreviewPlayer();
        this.updatePerformanceStats();
    }

    initTimelineCanvas() {
        const canvas = document.getElementById('viralTimeline');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = canvas.offsetWidth * window.devicePixelRatio;
        canvas.height = 120 * window.devicePixelRatio;
        canvas.style.width = canvas.offsetWidth + 'px';
        canvas.style.height = '120px';
        
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        this.timelineCanvas = canvas;
        this.timelineContext = ctx;
        
        // Add timeline interaction
        canvas.addEventListener('click', (e) => this.handleTimelineClick(e));
        canvas.addEventListener('mousemove', (e) => this.handleTimelineHover(e));
        canvas.addEventListener('mousedown', (e) => this.handleTimelineSelectionStart(e));
        canvas.addEventListener('mousemove', (e) => this.handleTimelineSelectionMove(e));
        canvas.addEventListener('mouseup', (e) => this.handleTimelineSelectionEnd(e));
    }

    initPreviewPlayer() {
        const previewPlayer = document.getElementById('previewPlayer');
        this.previewPlayer = {
            element: previewPlayer,
            isPlaying: false,
            currentTime: 0,
            duration: 0
        };
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v4/ws/app`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('✅ WebSocket connected');
            document.getElementById('ws-status').textContent = 'Connected';
            document.getElementById('ws-status').className = 'stat-value websocket-status connected';
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('❌ WebSocket disconnected');
            document.getElementById('ws-status').textContent = 'Disconnected';
            document.getElementById('ws-status').className = 'stat-value websocket-status disconnected';
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.setupWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    handleWebSocketMessage(data) {
        const startTime = performance.now();
        
        switch (data.type) {
            case 'connection_established':
                console.log('WebSocket connection established:', data.connection_id);
                break;
                
            case 'upload_progress':
                this.updateUploadProgress(data);
                break;
                
            case 'processing_status':
                this.updateProcessingStatus(data);
                break;
                
            case 'viral_score_update':
                this.updateViralScore(data);
                break;
                
            case 'interactive_timeline_data':
                this.updateTimeline(data);
                break;
                
            case 'live_preview_data':
                this.updateLivePreview(data);
                break;
                
            case 'timeline_update':
                this.handleTimelineUpdate(data);
                break;
                
            case 'heartbeat':
                // Respond to heartbeat
                if (this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
                }
                break;
        }
        
        // Track WebSocket latency
        const latency = performance.now() - startTime;
        this.performanceMetrics.websocketLatency.push(latency);
        if (this.performanceMetrics.websocketLatency.length > 100) {
            this.performanceMetrics.websocketLatency.shift();
        }
    }

    setupEventListeners() {
        // File input
        const fileInput = document.getElementById('fileInput');
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Timeline controls
        document.getElementById('playBtn').addEventListener('click', () => this.playTimeline());
        document.getElementById('pauseBtn').addEventListener('click', () => this.pauseTimeline());
        
        // Preview controls
        document.getElementById('previewQuality').addEventListener('change', () => this.regeneratePreview());
        document.getElementById('platformOptimization').addEventListener('change', () => this.regeneratePreview());
        
        // Generate clips button
        document.getElementById('generateClipsBtn').addEventListener('click', () => this.generateClips());
        
        // Export timeline button
        document.getElementById('exportTimelineBtn').addEventListener('click', () => this.exportTimeline());
        
        // Window resize
        window.addEventListener('resize', () => this.handleResize());
    }

    setupDragAndDrop() {
        const uploadZone = document.getElementById('uploadZone');
        
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });
        
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('drag-over');
        });
        
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('video/')) {
                this.handleFileSelect({ target: { files } });
            }
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 's':
                        e.preventDefault();
                        this.exportTimeline();
                        break;
                    case 'g':
                        e.preventDefault();
                        this.generateClips();
                        break;
                }
            }
            
            switch (e.key) {
                case ' ':
                    e.preventDefault();
                    this.toggleTimelinePlayback();
                    break;
                case 'ArrowLeft':
                    this.seekTimeline(-5);
                    break;
                case 'ArrowRight':
                    this.seekTimeline(5);
                    break;
            }
        });
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        console.log('📁 File selected:', file.name);
        
        // Show upload progress
        document.getElementById('uploadProgress').style.display = 'block';
        document.querySelector('.upload-content').style.display = 'none';
        
        // Generate upload ID
        const uploadId = 'upload_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        this.currentUploadId = uploadId;
        
        // Setup upload WebSocket
        this.setupUploadWebSocket(uploadId);
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('upload_id', uploadId);
        
        try {
            const response = await fetch('/api/v4/upload-video', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentSession = result.session_id;
                this.showAnalysisSection(result);
            } else {
                throw new Error(result.error || 'Upload failed');
            }
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showError('Upload failed: ' + error.message);
        }
    }

    setupUploadWebSocket(uploadId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v4/ws/upload/${uploadId}`;
        
        this.uploadWs = new WebSocket(wsUrl);
        
        this.uploadWs.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleUploadWebSocketMessage(data);
        };
    }

    handleUploadWebSocketMessage(data) {
        switch (data.type) {
            case 'processing_status':
                this.updateProcessingStatus(data);
                break;
                
            case 'viral_score_update':
                this.updateViralScore(data);
                break;
                
            case 'timeline_update':
                this.handleTimelineUpdate(data);
                break;
        }
    }

    showAnalysisSection(uploadResult) {
        // Hide upload section
        document.querySelector('.upload-section').style.display = 'none';
        
        // Show analysis section
        const analysisSection = document.getElementById('analysisSection');
        analysisSection.style.display = 'block';
        
        // Initialize with upload result data
        if (uploadResult.preview && uploadResult.preview.viral_analysis) {
            this.updateViralScore({
                viral_score: uploadResult.preview.viral_analysis.viral_score,
                confidence: uploadResult.preview.viral_analysis.confidence,
                factors: uploadResult.preview.viral_analysis.factors
            });
        }
        
        // Show processing status
        this.showProcessingStatus();
    }

    showProcessingStatus() {
        const processingStatus = document.getElementById('processingStatus');
        processingStatus.style.display = 'block';
        
        // Start entertainment rotation
        this.startEntertainmentRotation();
    }

    updateUploadProgress(data) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const uploadStats = document.getElementById('uploadStats');
        
        if (progressFill) {
            progressFill.style.width = data.progress + '%';
        }
        
        if (progressText) {
            progressText.textContent = `Uploading... ${Math.round(data.progress)}%`;
        }
        
        if (uploadStats && data.uploaded_size && data.total_size) {
            const uploadedMB = (data.uploaded_size / 1024 / 1024).toFixed(1);
            const totalMB = (data.total_size / 1024 / 1024).toFixed(1);
            uploadStats.textContent = `${uploadedMB} MB / ${totalMB} MB`;
        }
    }

    updateProcessingStatus(data) {
        const taskIcon = document.getElementById('taskIcon');
        const taskName = document.getElementById('taskName');
        const taskDescription = document.getElementById('taskDescription');
        const progressPercentage = document.getElementById('progressPercentage');
        const progressCircle = document.getElementById('progressCircle');
        
        // Update task info
        if (taskName) {
            taskName.textContent = this.getStageDisplayName(data.stage);
        }
        
        if (taskDescription) {
            taskDescription.textContent = data.message || 'Processing with Netflix-level quality...';
        }
        
        // Update progress
        if (progressPercentage) {
            progressPercentage.textContent = Math.round(data.progress) + '%';
        }
        
        if (progressCircle) {
            const circumference = 2 * Math.PI * 25;
            const offset = circumference - (data.progress / 100) * circumference;
            progressCircle.style.strokeDasharray = circumference;
            progressCircle.style.strokeDashoffset = offset;
        }
        
        // Update icon based on stage
        if (taskIcon) {
            taskIcon.textContent = this.getStageIcon(data.stage);
        }
        
        // Show entertaining fact if provided
        if (data.entertaining_fact) {
            this.showEntertainingFact(data.entertaining_fact);
        }
        
        // Hide processing status when complete
        if (data.stage === 'complete') {
            setTimeout(() => {
                document.getElementById('processingStatus').style.display = 'none';
            }, 2000);
        }
    }

    updateViralScore(data) {
        const scoreNumber = document.getElementById('viralScoreNumber');
        const scoreCircle = document.getElementById('viralScoreCircle');
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceValue = document.getElementById('confidenceValue');
        const viralFactors = document.getElementById('viralFactors');
        
        // Animate score update
        if (scoreNumber) {
            this.animateNumberTo(scoreNumber, data.viral_score);
        }
        
        // Update score circle color
        if (scoreCircle) {
            scoreCircle.className = 'score-circle ' + this.getScoreClass(data.viral_score);
        }
        
        // Update confidence
        if (confidenceFill && data.confidence) {
            confidenceFill.style.width = (data.confidence * 100) + '%';
        }
        
        if (confidenceValue && data.confidence) {
            confidenceValue.textContent = Math.round(data.confidence * 100) + '%';
        }
        
        // Update viral factors
        if (viralFactors && data.factors) {
            viralFactors.innerHTML = data.factors.map(factor => 
                `<div class="factor-item">✓ ${factor}</div>`
            ).join('');
        }
    }

    handleTimelineUpdate(data) {
        const startTime = performance.now();
        
        this.timelineData = data;
        this.renderTimeline();
        
        // Track timeline render performance
        const renderTime = performance.now() - startTime;
        this.performanceMetrics.timelineRenderTime.push(renderTime);
        if (this.performanceMetrics.timelineRenderTime.length > 100) {
            this.performanceMetrics.timelineRenderTime.shift();
        }
        
        // Add key moment markers
        if (data.key_moments) {
            this.addTimelineMarkers(data.key_moments);
        }
    }

    renderTimeline() {
        if (!this.timelineData || !this.timelineContext) return;
        
        const canvas = this.timelineCanvas;
        const ctx = this.timelineContext;
        const width = canvas.offsetWidth;
        const height = 120;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw viral score heatmap
        const scores = this.timelineData.viral_heatmap || [];
        const segmentWidth = width / scores.length;
        
        scores.forEach((score, index) => {
            const x = index * segmentWidth;
            const barHeight = (score / 100) * height;
            const y = height - barHeight;
            
            // Color based on viral score
            const color = this.getViralScoreColor(score);
            ctx.fillStyle = color;
            ctx.fillRect(x, y, segmentWidth, barHeight);
            
            // Add subtle gradient
            const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
            gradient.addColorStop(0, color);
            gradient.addColorStop(1, color + '80');
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, segmentWidth, barHeight);
        });
        
        // Draw timeline grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        
        // Vertical grid lines (time markers)
        for (let i = 0; i <= 10; i++) {
            const x = (i / 10) * width;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        // Horizontal grid lines (score markers)
        for (let i = 0; i <= 4; i++) {
            const y = (i / 4) * height;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // Draw current selection if any
        if (this.timelineSelection) {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
            ctx.fillRect(
                this.timelineSelection.start * width,
                0,
                (this.timelineSelection.end - this.timelineSelection.start) * width,
                height
            );
        }
    }

    addTimelineMarkers(keyMoments) {
        const markersContainer = document.getElementById('timelineMarkers');
        markersContainer.innerHTML = '';
        
        keyMoments.forEach(moment => {
            const marker = document.createElement('div');
            marker.className = 'timeline-marker ' + moment.type;
            marker.style.left = (moment.timestamp / this.timelineData.duration * 100) + '%';
            marker.title = moment.description;
            marker.innerHTML = this.getMarkerIcon(moment.type);
            
            marker.addEventListener('click', () => {
                this.seekToTime(moment.timestamp);
                this.generateInstantPreview(moment.timestamp, moment.timestamp + 10);
            });
            
            markersContainer.appendChild(marker);
        });
    }

    handleTimelineClick(event) {
        const rect = this.timelineCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const clickRatio = x / rect.width;
        
        if (this.timelineData && this.timelineData.duration) {
            const clickTime = clickRatio * this.timelineData.duration;
            this.seekToTime(clickTime);
            
            // Generate instant preview around click point
            const previewStart = Math.max(0, clickTime - 5);
            const previewEnd = Math.min(this.timelineData.duration, clickTime + 5);
            this.generateInstantPreview(previewStart, previewEnd);
        }
    }

    handleTimelineSelectionStart(event) {
        const rect = this.timelineCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const ratio = x / rect.width;
        
        this.isSelecting = true;
        this.timelineSelection = {
            start: ratio,
            end: ratio
        };
    }

    handleTimelineSelectionMove(event) {
        if (!this.isSelecting) return;
        
        const rect = this.timelineCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const ratio = Math.max(0, Math.min(1, x / rect.width));
        
        this.timelineSelection.end = ratio;
        this.renderTimeline();
    }

    handleTimelineSelectionEnd(event) {
        if (!this.isSelecting) return;
        
        this.isSelecting = false;
        
        if (this.timelineSelection && this.timelineData) {
            const startTime = this.timelineSelection.start * this.timelineData.duration;
            const endTime = this.timelineSelection.end * this.timelineData.duration;
            
            if (Math.abs(endTime - startTime) > 1) { // Minimum 1 second selection
                this.generateInstantPreview(Math.min(startTime, endTime), Math.max(startTime, endTime));
            }
        }
    }

    async generateInstantPreview(startTime, endTime) {
        if (!this.currentSession) return;
        
        const startGenTime = performance.now();
        
        // Check cache first
        const cacheKey = `${this.currentSession}_${startTime}_${endTime}`;
        if (this.previewCache.has(cacheKey)) {
            this.displayPreview(this.previewCache.get(cacheKey));
            return;
        }
        
        // Show loading state
        this.showPreviewLoading();
        
        try {
            const response = await fetch('/api/v4/generate-preview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.currentSession,
                    start_time: startTime,
                    end_time: endTime,
                    quality: document.getElementById('previewQuality').value,
                    platform_optimizations: document.getElementById('platformOptimization').value ? 
                        [document.getElementById('platformOptimization').value] : null
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Cache the result
                this.previewCache.set(cacheKey, result);
                
                // Track generation time
                const genTime = performance.now() - startGenTime;
                this.performanceMetrics.previewGenerationTime.push(genTime);
                if (this.performanceMetrics.previewGenerationTime.length > 100) {
                    this.performanceMetrics.previewGenerationTime.shift();
                }
                
                this.displayPreview(result);
            } else {
                throw new Error(result.error || 'Preview generation failed');
            }
            
        } catch (error) {
            console.error('Preview generation error:', error);
            this.showPreviewError(error.message);
        }
    }

    showPreviewLoading() {
        const previewPlayer = document.getElementById('previewPlayer');
        previewPlayer.innerHTML = `
            <div class="preview-loading">
                <div class="loading-spinner"></div>
                <p>Generating Netflix-level preview...</p>
            </div>
        `;
    }

    displayPreview(previewData) {
        const previewPlayer = document.getElementById('previewPlayer');
        const previewInfo = document.getElementById('previewInfo');
        
        // Display video preview
        previewPlayer.innerHTML = `
            <video controls class="preview-video">
                <source src="${previewData.preview_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        `;
        
        // Display viral analysis
        if (previewData.viral_analysis) {
            previewInfo.innerHTML = `
                <div class="preview-analysis">
                    <div class="analysis-score">
                        <span class="score-label">Viral Score:</span>
                        <span class="score-value ${this.getScoreClass(previewData.viral_analysis.viral_score)}">
                            ${previewData.viral_analysis.viral_score}
                        </span>
                    </div>
                    <div class="analysis-factors">
                        ${previewData.viral_analysis.factors.map(factor => 
                            `<div class="factor-tag">✓ ${factor}</div>`
                        ).join('')}
                    </div>
                </div>
            `;
        }
        
        // Display suggestions
        if (previewData.suggestions) {
            previewInfo.innerHTML += `
                <div class="preview-suggestions">
                    <h4>💡 Optimization Suggestions</h4>
                    ${previewData.suggestions.map(suggestion => 
                        `<div class="suggestion-item">${suggestion}</div>`
                    ).join('')}
                </div>
            `;
        }
    }

    showPreviewError(error) {
        const previewPlayer = document.getElementById('previewPlayer');
        previewPlayer.innerHTML = `
            <div class="preview-error">
                <div class="error-icon">❌</div>
                <p>Preview generation failed</p>
                <p class="error-message">${error}</p>
            </div>
        `;
    }

    startEntertainmentRotation() {
        const entertainmentFacts = [
            "Did you know? The first viral video was a dancing baby in 1996! 👶",
            "Netflix processes over 1 billion hours of content daily! 📺",
            "TikTok videos under 15 seconds have 85% higher engagement rates! ⚡",
            "The human attention span is now shorter than a goldfish (8 seconds)! 🐠",
            "Vertical videos get 9x more engagement than horizontal ones! 📱",
            "Sound-on videos have 30% better retention rates! 🔊",
            "The golden ratio applies to viral content timing: 1.618 seconds! ✨",
            "Videos with captions get 40% more views! 📝",
            "The best posting time is 6-10 PM in your audience's timezone! ⏰",
            "Using trending sounds can boost views by 300%! 🎵"
        ];
        
        let factIndex = 0;
        
        const rotateFacts = () => {
            if (document.getElementById('processingStatus').style.display === 'none') {
                return; // Stop rotation if processing is hidden
            }
            
            const entertainmentText = document.getElementById('entertainmentText');
            if (entertainmentText) {
                entertainmentText.style.opacity = '0';
                
                setTimeout(() => {
                    entertainmentText.textContent = entertainmentFacts[factIndex];
                    entertainmentText.style.opacity = '1';
                    factIndex = (factIndex + 1) % entertainmentFacts.length;
                }, 300);
            }
            
            setTimeout(rotateFacts, 3000); // Rotate every 3 seconds
        };
        
        rotateFacts();
    }

    showEntertainingFact(fact) {
        const entertainmentText = document.getElementById('entertainmentText');
        if (entertainmentText) {
            entertainmentText.style.opacity = '0';
            setTimeout(() => {
                entertainmentText.textContent = fact;
                entertainmentText.style.opacity = '1';
            }, 300);
        }
    }

    initPerformanceMonitoring() {
        setInterval(() => {
            this.updatePerformanceStats();
        }, 5000); // Update every 5 seconds
    }

    updatePerformanceStats() {
        const performanceStats = document.getElementById('performanceStats');
        if (!performanceStats) return;
        
        const avgPreviewTime = this.getAverageTime(this.performanceMetrics.previewGenerationTime);
        const avgWebSocketLatency = this.getAverageTime(this.performanceMetrics.websocketLatency);
        const avgTimelineRender = this.getAverageTime(this.performanceMetrics.timelineRenderTime);
        
        performanceStats.innerHTML = `
            <div class="perf-stat">
                <span class="perf-label">Preview Gen:</span>
                <span class="perf-value">${avgPreviewTime.toFixed(1)}ms</span>
            </div>
            <div class="perf-stat">
                <span class="perf-label">WebSocket:</span>
                <span class="perf-value">${avgWebSocketLatency.toFixed(1)}ms</span>
            </div>
            <div class="perf-stat">
                <span class="perf-label">Timeline:</span>
                <span class="perf-value">${avgTimelineRender.toFixed(1)}ms</span>
            </div>
        `;
    }

    getAverageTime(times) {
        if (times.length === 0) return 0;
        return times.reduce((a, b) => a + b, 0) / times.length;
    }

    // Utility methods
    getStageDisplayName(stage) {
        const stageNames = {
            'analyzing': 'AI Analysis',
            'extracting_features': 'Feature Extraction',
            'scoring_segments': 'Viral Scoring',
            'generating_timeline': 'Timeline Generation',
            'complete': 'Complete'
        };
        return stageNames[stage] || stage;
    }

    getStageIcon(stage) {
        const stageIcons = {
            'analyzing': '🔍',
            'extracting_features': '⚡',
            'scoring_segments': '📊',
            'generating_timeline': '🎬',
            'complete': '✅'
        };
        return stageIcons[stage] || '⚙️';
    }

    getScoreClass(score) {
        if (score >= 80) return 'excellent';
        if (score >= 60) return 'good';
        if (score >= 40) return 'average';
        return 'poor';
    }

    getViralScoreColor(score) {
        if (score >= 80) return '#00ff44';
        if (score >= 60) return '#ffff00';
        if (score >= 40) return '#ffaa00';
        return '#ff4444';
    }

    getMarkerIcon(type) {
        const icons = {
            'hook': '🎣',
            'peak': '🚀',
            'climax': '💥',
            'cta': '📢',
            'emotional_peak': '❤️'
        };
        return icons[type] || '📍';
    }

    animateNumberTo(element, targetValue) {
        const startValue = parseInt(element.textContent) || 0;
        const duration = 1000; // 1 second animation
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function
            const easeOutCubic = 1 - Math.pow(1 - progress, 3);
            const currentValue = Math.round(startValue + (targetValue - startValue) * easeOutCubic);
            
            element.textContent = currentValue;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }

    // Timeline control methods
    playTimeline() {
        // Implementation for timeline playback
        console.log('Playing timeline');
    }

    pauseTimeline() {
        // Implementation for timeline pause
        console.log('Pausing timeline');
    }

    toggleTimelinePlayback() {
        if (this.previewPlayer.isPlaying) {
            this.pauseTimeline();
        } else {
            this.playTimeline();
        }
    }

    seekTimeline(seconds) {
        // Implementation for timeline seeking
        console.log('Seeking timeline by', seconds, 'seconds');
    }

    seekToTime(time) {
        // Implementation for seeking to specific time
        console.log('Seeking to time', time);
    }

    regeneratePreview() {
        if (this.timelineSelection && this.timelineData) {
            const startTime = this.timelineSelection.start * this.timelineData.duration;
            const endTime = this.timelineSelection.end * this.timelineData.duration;
            this.generateInstantPreview(startTime, endTime);
        }
    }

    generateClips() {
        // Implementation for clip generation
        console.log('Generating clips');
    }

    exportTimeline() {
        // Implementation for timeline export
        console.log('Exporting timeline');
    }

    handleResize() {
        // Reinitialize timeline canvas on resize
        this.initTimelineCanvas();
        if (this.timelineData) {
            this.renderTimeline();
        }
    }

    showError(message) {
        // Show error notification
        console.error(message);
        
        // You could implement a toast notification system here
        alert('Error: ' + message);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.viralClipApp = new NetflixLevelApp();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.viralClipApp && window.viralClipApp.ws) {
        window.viralClipApp.ws.close();
    }
    if (window.viralClipApp && window.viralClipApp.uploadWs) {
        window.viralClipApp.uploadWs.close();
    }
});
