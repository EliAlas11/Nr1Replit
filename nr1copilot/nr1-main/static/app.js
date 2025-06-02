/**
 * ViralClip Pro v5.0 - Netflix-Level Frontend
 * Ultra-responsive UI with real-time feedback
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
            timelineRenderTime: [],
            interactionResponseTime: []
        };

        // Ultra-fast cache
        this.cache = {
            previews: new Map(),
            viralScores: new Map(),
            interactions: new Map(),
            timeline: new Map()
        };

        // Netflix-level configuration
        this.config = {
            maxRetries: 3,
            retryDelay: 1000,
            chunkSize: 1024 * 1024, // 1MB chunks
            previewQuality: 'high',
            timelineResolution: 100,
            websocketHeartbeat: 30000
        };

        this.init();
    }

    async init() {
        console.log('🎬 Initializing ViralClip Pro v5.0 - Netflix-Level Experience');

        try {
            await this.setupUI();
            await this.setupWebSocket();
            await this.setupEventListeners();
            await this.setupDragAndDrop();
            await this.setupKeyboardShortcuts();
            await this.initPerformanceMonitoring();

            // Show app after initialization
            this.showApp();

            console.log('✅ Netflix-Level App initialized successfully');
        } catch (error) {
            console.error('❌ App initialization failed:', error);
            this.showError('Initialization failed. Please refresh the page.');
        }
    }

    async setupUI() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve));
        }

        // Initialize UI components
        this.initializeProgress();
        this.initializeTimeline();
        this.initializePreview();
        this.initializeMetrics();

        // Create main container if it doesn't exist
        if (!document.getElementById('app-container')) {
            document.body.innerHTML = `
                <div id="app-container" class="netflix-app" style="display: none;">
                    <header class="app-header">
                        <div class="header-content">
                            <h1 class="app-title">
                                <span class="logo">🎬</span>
                                ViralClip Pro v5.0
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
                                    <span class="stat-value websocket-status" id="connectionStatus">Connecting...</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">FPS</span>
                                    <span class="stat-value" id="fpsCounter">--</span>
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
                                    <input type="file" id="videoFile" accept="video/*" style="display: none;">
                                    <button class="upload-btn" id="uploadButton">
                                        Choose Video File
                                    </button>
                                </div>
                            </div>
                        </div>

                        <div class="progress-section" id="progressSection" style="display: none;">
                            <div class="progress-bar-container">
                                <div class="progress-bar" id="progressBar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                                    <span class="progress-text" id="progressText">0%</span>
                                </div>
                            </div>
                            <p class="progress-message" id="progressMessage">Uploading...</p>
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
                        <div class="error-message" id="errorMessage" style="display: none;"></div>
                    </main>

                    <footer class="app-footer">
                        <div class="footer-content">
                            <div class="performance-stats" id="performanceStats"></div>
                            <div class="version-info">
                                ViralClip Pro v5.0 | Netflix-Level AI Processing
                            </div>
                        </div>
                    </footer>
                </div>
            `;
        }
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v5/ws/app`;

        console.log('🔗 Connecting to Netflix-Level WebSocket:', wsUrl);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('✅ WebSocket connected');
            this.updateConnectionStatus(true);
            this.startHeartbeat();
        };

        this.ws.onmessage = (event) => {
            this.handleWebSocketMessage(JSON.parse(event.data));
        };

        this.ws.onclose = (event) => {
            console.log('🔌 WebSocket disconnected:', event.code, event.reason);
            this.updateConnectionStatus(false);
            this.reconnectWebSocket();
        };

        this.ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
    }

    async setupDragAndDrop() {
        const uploadZone = document.getElementById('uploadZone');
        if (!uploadZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => {
                uploadZone.classList.add('drag-over');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => {
                uploadZone.classList.remove('drag-over');
            }, false);
        });

        uploadZone.addEventListener('drop', this.handleDrop.bind(this), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    async handleDrop(e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            await this.uploadVideo(files[0]);
        }
    }

    async uploadVideo(file) {
        if (!file || !file.type.startsWith('video/')) {
            this.showError('Please select a valid video file');
            return;
        }

        const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.currentSession = sessionId;

        try {
            this.showProgress(0, 'Preparing upload...');

            const formData = new FormData();
            formData.append('file', file);
            formData.append('upload_id', sessionId);
            formData.append('title', file.name);

            const response = await fetch('/api/v5/upload-video', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const result = await response.json();
            console.log('📤 Upload successful:', result);

            this.showProgress(100, 'Upload complete!');
            this.startAnalysis(result);

        } catch (error) {
            console.error('❌ Upload failed:', error);
            this.showError(`Upload failed: ${error.message}`);
        }
    }

    showProgress(percentage, message) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const progressMessage = document.getElementById('progressMessage');

        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
        }

        if (progressText) {
            progressText.textContent = `${Math.round(percentage)}%`;
        }

        if (progressMessage) {
            progressMessage.textContent = message;
        }

        // Show progress section
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'block';
        }
    }

    showError(message) {
        console.error('🚨 Error:', message);

        // Show error in UI
        const errorElement = document.getElementById('errorMessage');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';

            // Auto-hide after 5 seconds
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 5000);
        }
    }

    showApp() {
        const loadingScreen = document.getElementById('loadingScreen');
        const appContainer = document.getElementById('app-container');

        if (loadingScreen) {
            loadingScreen.style.display = 'none';
        }

        if (appContainer) {
            appContainer.style.display = 'flex';
        }
    }

    initializeProgress() {
        // Initialize progress components
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'none';
        }
    }

    initializeTimeline() {
        // Initialize timeline component
        this.viralTimeline = {
            data: [],
            markers: [],
            currentTime: 0
        };
    }

    initializePreview() {
        // Initialize preview player
        this.previewPlayer = {
            currentVideo: null,
            isPlaying: false,
            currentTime: 0
        };
    }

    initializeMetrics() {
        // Initialize performance metrics display
        this.updateMetrics();
    }

    updateMetrics() {
        const metricsElements = {
            processing: document.getElementById('processing-count'),
            queue: document.getElementById('queue-count')
        };

        Object.entries(metricsElements).forEach(([key, element]) => {
            if (element) {
                element.textContent = this.getMetricValue(key);
            }
        });
    }

    getMetricValue(metric) {
        switch (metric) {
            case 'processing':
                return this.currentSession ? 1 : 0;
            case 'queue':
                return 0; // Implement queue logic
            default:
                return 0;
        }
    }

    handleWebSocketMessage(data) {
        console.log('📨 WebSocket message:', data);

        switch (data.type) {
            case 'progress':
                this.handleProgressUpdate(data);
                break;
            case 'analysis':
                this.handleAnalysisUpdate(data);
                break;
            case 'preview':
                this.handlePreviewUpdate(data);
                break;
            case 'error':
                this.showError(data.message);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    handleProgressUpdate(data) {
        if (data.session_id === this.currentSession) {
            this.showProgress(data.progress, data.message);
        }
    }

    handleAnalysisUpdate(data) {
        console.log('📊 Analysis update:', data);
        // Update analysis display
    }

    handlePreviewUpdate(data) {
        console.log('🎥 Preview update:', data);
        // Update preview display
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.textContent = connected ? 'Connected' : 'Disconnected';
            statusElement.className = connected ? 'status-connected' : 'status-disconnected';
        }
    }

    startHeartbeat() {
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'heartbeat' }));
            }
        }, this.config.websocketHeartbeat);
    }

    reconnectWebSocket() {
        setTimeout(() => {
            console.log('🔄 Attempting WebSocket reconnection...');
            this.setupWebSocket();
        }, this.config.retryDelay);
    }

    setupEventListeners() {
        // File input change
        const fileInput = document.getElementById('videoFile');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.uploadVideo(e.target.files[0]);
                }
            });
        }

        // Upload button click
        const uploadButton = document.getElementById('uploadButton');
        if (uploadButton) {
            uploadButton.addEventListener('click', () => {
                fileInput?.click();
            });
        }

        // Timeline controls
        document.getElementById('playBtn')?.addEventListener('click', () => this.playTimeline());
        document.getElementById('pauseBtn')?.addEventListener('click', () => this.pauseTimeline());

        // Preview controls
        document.getElementById('previewQuality')?.addEventListener('change', () => this.regeneratePreview());
        document.getElementById('platformOptimization')?.addEventListener('change', () => this.regeneratePreview());

        // Generate clips button
        document.getElementById('generateClipsBtn')?.addEventListener('click', () => this.generateClips());

        // Export timeline button
        document.getElementById('exportTimelineBtn')?.addEventListener('click', () => this.exportTimeline());

        // Window resize
        window.addEventListener('resize', () => this.handleResize());
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + U for upload
            if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
                e.preventDefault();
                document.getElementById('videoFile')?.click();
            }

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

    async initPerformanceMonitoring() {
        // Start performance monitoring
        this.startPerformanceTimer = performance.now();

        // Monitor FPS
        this.monitorFPS();

        console.log('📊 Performance monitoring started');
    }

    monitorFPS() {
        let lastTime = performance.now();
        let frames = 0;

        const tick = (currentTime) => {
            frames++;

            if (currentTime - lastTime >= 1000) {
                const fps = Math.round((frames * 1000) / (currentTime - lastTime));
                this.updateFPSDisplay(fps);

                frames = 0;
                lastTime = currentTime;
            }

            this.animationFrameId = requestAnimationFrame(tick);
        };

        requestAnimationFrame(tick);
    }

    updateFPSDisplay(fps) {
        const fpsElement = document.getElementById('fpsCounter');
        if (fpsElement) {
            fpsElement.textContent = `${fps} FPS`;
        }
    }

    async startAnalysis(uploadResult) {
        console.log('🧠 Starting Netflix-level analysis...');

        try {
            this.showProgress(10, 'Analyzing video content...');

            // This would trigger server-side analysis
            // The WebSocket will handle progress updates

        } catch (error) {
            console.error('❌ Analysis failed:', error);
            this.showError('Analysis failed. Please try again.');
        }
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

        const renderStart = performance.now();
        const canvas = this.timelineCanvas;
        const ctx = this.timelineContext;
        const width = canvas.offsetWidth;
        const height = 120;

        // Netflix-level high-DPI rendering
        const devicePixelRatio = window.devicePixelRatio || 1;
        ctx.save();
        ctx.scale(devicePixelRatio, devicePixelRatio);

        // Clear with Netflix black background
        ctx.fillStyle = '#141414';
        ctx.fillRect(0, 0, width, height);

        // Enhanced viral score heatmap with Netflix-level quality
        const scores = this.timelineData.viral_heatmap || [];
        const segmentWidth = width / scores.length;

        // Draw Netflix-level viral heatmap with smooth gradients
        scores.forEach((score, index) => {
            const x = index * segmentWidth;
            const barHeight = (score / 100) * height;
            const y = height - barHeight;

            // Netflix-level color mapping with premium gradients
            const colorData = this.getNetflixViralScoreColor(score);
            
            // Create vertical gradient for each segment
            const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
            gradient.addColorStop(0, colorData.primary);
            gradient.addColorStop(0.5, colorData.mid);
            gradient.addColorStop(1, colorData.base);
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, segmentWidth, barHeight);

            // Add Netflix-style glow effect for high scores
            if (score > 80) {
                ctx.shadowColor = colorData.glow;
                ctx.shadowBlur = 8;
                ctx.fillStyle = colorData.highlight;
                ctx.fillRect(x, y, segmentWidth, Math.min(barHeight, 10));
                ctx.shadowBlur = 0;
            }
        });

        // Enhanced timeline grid with Netflix aesthetics
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 0.5;

        // Vertical time markers with labels
        for (let i = 0; i <= 10; i++) {
            const x = (i / 10) * width;
            
            // Main grid line
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();

            // Time labels
            if (this.timelineData.duration) {
                const timeSeconds = (i / 10) * this.timelineData.duration;
                const timeLabel = this.formatTime(timeSeconds);
                
                ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
                ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
                ctx.textAlign = 'center';
                ctx.fillText(timeLabel, x, height - 5);
            }
        }

        // Horizontal score markers with viral level indicators
        const viralLevels = [
            { threshold: 80, label: 'Viral', color: '#00ff44' },
            { threshold: 60, label: 'High', color: '#ffff00' },
            { threshold: 40, label: 'Good', color: '#ffaa00' },
            { threshold: 20, label: 'Low', color: '#ff4444' }
        ];

        viralLevels.forEach(level => {
            const y = height - (level.threshold / 100) * height;
            
            ctx.strokeStyle = level.color + '40';
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
            ctx.setLineDash([]);

            // Level labels
            ctx.fillStyle = level.color;
            ctx.font = 'bold 9px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
            ctx.textAlign = 'left';
            ctx.fillText(level.label, 5, y - 2);
        });

        // Enhanced selection visualization
        if (this.timelineSelection) {
            const selectionGradient = ctx.createLinearGradient(
                this.timelineSelection.start * width, 0,
                this.timelineSelection.end * width, 0
            );
            selectionGradient.addColorStop(0, 'rgba(229, 9, 20, 0.3)'); // Netflix red
            selectionGradient.addColorStop(0.5, 'rgba(229, 9, 20, 0.5)');
            selectionGradient.addColorStop(1, 'rgba(229, 9, 20, 0.3)');
            
            ctx.fillStyle = selectionGradient;
            ctx.fillRect(
                this.timelineSelection.start * width,
                0,
                (this.timelineSelection.end - this.timelineSelection.start) * width,
                height
            );

            // Selection border
            ctx.strokeStyle = '#e50914';
            ctx.lineWidth = 2;
            ctx.strokeRect(
                this.timelineSelection.start * width,
                0,
                (this.timelineSelection.end - this.timelineSelection.start) * width,
                height
            );
        }

        // Draw engagement peaks with Netflix-style markers
        if (this.timelineData.engagement_peaks) {
            this.timelineData.engagement_peaks.forEach(peak => {
                const x = (peak.timestamp / this.timelineData.duration) * width;
                
                // Peak marker
                ctx.fillStyle = '#ff6b35';
                ctx.beginPath();
                ctx.arc(x, height - (peak.score / 100) * height, 4, 0, 2 * Math.PI);
                ctx.fill();

                // Peak glow
                ctx.shadowColor = '#ff6b35';
                ctx.shadowBlur = 12;
                ctx.beginPath();
                ctx.arc(x, height - (peak.score / 100) * height, 2, 0, 2 * Math.PI);
                ctx.fill();
                ctx.shadowBlur = 0;
            });
        }

        // Current playhead with Netflix red
        if (this.currentPlayheadPosition !== undefined) {
            const playheadX = this.currentPlayheadPosition * width;
            
            // Playhead line
            ctx.strokeStyle = '#e50914';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(playheadX, 0);
            ctx.lineTo(playheadX, height);
            ctx.stroke();

            // Playhead handle
            ctx.fillStyle = '#e50914';
            ctx.beginPath();
            ctx.arc(playheadX, 8, 6, 0, 2 * Math.PI);
            ctx.fill();

            // Playhead shadow
            ctx.shadowColor = '#e50914';
            ctx.shadowBlur = 8;
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        ctx.restore();

        // Track rendering performance
        const renderTime = performance.now() - renderStart;
        this.trackNetflixLevelMetric('timeline_render', renderTime);

        // Performance optimization alert
        if (renderTime > 16.67) { // 60fps threshold
            console.warn(`Timeline render took ${renderTime.toFixed(2)}ms (>16.67ms for 60fps)`);
        }
    }

    getNetflixViralScoreColor(score) {
        // Netflix-level premium color mapping
        if (score >= 90) {
            return {
                primary: '#00ff44',
                mid: '#00dd33',
                base: '#00bb22',
                highlight: '#44ff77',
                glow: '#00ff44'
            };
        } else if (score >= 80) {
            return {
                primary: '#88ff00',
                mid: '#77dd00',
                base: '#66bb00',
                highlight: '#99ff33',
                glow: '#88ff00'
            };
        } else if (score >= 60) {
            return {
                primary: '#ffff00',
                mid: '#dddd00',
                base: '#bbbb00',
                highlight: '#ffff33',
                glow: '#ffff00'
            };
        } else if (score >= 40) {
            return {
                primary: '#ffaa00',
                mid: '#dd9900',
                base: '#bb7700',
                highlight: '#ffbb33',
                glow: '#ffaa00'
            };
        } else {
            return {
                primary: '#ff4444',
                mid: '#dd3333',
                base: '#bb2222',
                highlight: '#ff6666',
                glow: '#ff4444'
            };
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

    async handleTimelineClick(event) {
        const startTime = performance.now();
        const rect = this.timelineCanvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const clickRatio = x / rect.width;

        if (this.timelineData && this.timelineData.duration) {
            const clickTime = clickRatio * this.timelineData.duration;

            // Ultra-fast timeline interaction
            await this.handleUltraFastTimelineInteraction({
                type: 'click',
                timestamp: clickTime,
                position: { x, y: event.clientY - rect.top },
                clickRatio
            });

            // Instant preview generation with predictive caching
            const previewStart = Math.max(0, clickTime - 5);
            const previewEnd = Math.min(this.timelineData.duration, clickTime + 5);
            await this.generateUltraFastPreview(previewStart, previewEnd, 'instant');

            // Track interaction performance
            const responseTime = performance.now() - startTime;
            this.performanceMetrics.interactionResponseTime.push(responseTime);

            // Update performance display
            this.updatePerformanceDisplay('timeline_click', responseTime);
        }
    }

    async handleUltraFastTimelineInteraction(interaction) {
        const interactionStart = performance.now();

        try {
            // Netflix-level ultra-fast cache with predictive loading
            const cacheKey = `interaction_${interaction.type}_${Math.floor(interaction.timestamp)}`;
            if (this.cache.interactions.has(cacheKey)) {
                const cachedResponse = this.cache.interactions.get(cacheKey);
                await this.processNetflixLevelInteractionResponse(cachedResponse);
                this.showCacheHitIndicator('interaction', 'ultra_fast');
                return;
            }

            // Parallel processing: Server request + Local prediction
            const serverPromise = this.sendNetflixLevelInteraction(interaction);
            const localPredictionPromise = this.generateLocalPrediction(interaction);
            const visualFeedbackPromise = this.provideInstantVisualFeedback(interaction);

            // Execute all in parallel for maximum responsiveness
            const [serverResponse, localPrediction] = await Promise.allSettled([
                serverPromise,
                localPredictionPromise,
                visualFeedbackPromise
            ]);

            // Use server response if available, fallback to prediction
            if (serverResponse.status === 'fulfilled') {
                await this.processNetflixLevelInteractionResponse(serverResponse.value);
            } else if (localPrediction.status === 'fulfilled') {
                await this.processLocalPredictionResponse(localPrediction.value);
                this.showPredictionModeIndicator();
            }

            // Enhanced performance tracking with Netflix metrics
            const responseTime = performance.now() - interactionStart;
            this.trackNetflixLevelMetric('timeline_interaction', responseTime, {
                interaction_type: interaction.type,
                cache_status: 'miss',
                prediction_used: localPrediction.status === 'fulfilled'
            });

            // Auto-optimization for future interactions
            await this.optimizeForFutureInteractions(interaction, responseTime);

        } catch (error) {
            console.error('Netflix-level timeline interaction failed:', error);
            await this.handleInteractionError(error, interaction);
        }
    }

    async sendNetflixLevelInteraction(interaction) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            return new Promise((resolve, reject) => {
                const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                const timeout = setTimeout(() => reject(new Error('Interaction timeout')), 5000);

                const handleResponse = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.message_id === messageId || data.type === 'timeline_interaction_response') {
                        clearTimeout(timeout);
                        this.ws.removeEventListener('message', handleResponse);
                        resolve(data);
                    }
                };

                this.ws.addEventListener('message', handleResponse);
                this.ws.send(JSON.stringify({
                    type: 'timeline_interaction',
                    message_id: messageId,
                    session_id: this.currentSession,
                    interaction_data: {
                        ...interaction,
                        client_timestamp: performance.now(),
                        netflix_quality: true,
                        ultra_fast_mode: true
                    }
                }));
            });
        }
        throw new Error('WebSocket not available');
    }

    async generateLocalPrediction(interaction) {
        // Netflix-level local AI prediction for instant feedback
        const predictionStart = performance.now();

        try {
            let prediction = {};

            switch (interaction.type) {
                case 'hover':
                    prediction = await this.predictHoverResponse(interaction);
                    break;
                case 'selection':
                    prediction = await this.predictSelectionAnalysis(interaction);
                    break;
                case 'viral_score_request':
                    prediction = await this.predictViralScore(interaction);
                    break;
                default:
                    prediction = await this.generateGenericPrediction(interaction);
            }

            prediction.prediction_time = performance.now() - predictionStart;
            prediction.confidence = 0.85; // Netflix confidence level
            prediction.source = 'local_ai_prediction';

            return prediction;

        } catch (error) {
            console.error('Local prediction failed:', error);
            return this.getFallbackPrediction(interaction);
        }
    }

    async provideInstantVisualFeedback(interaction) {
        // Netflix-level instant visual feedback
        const feedbackStart = performance.now();

        try {
            switch (interaction.type) {
                case 'hover':
                    await this.showInstantHoverEffect(interaction);
                    break;
                case 'selection':
                    await this.highlightSelectionArea(interaction);
                    break;
                case 'viral_score_request':
                    await this.pulseViralScoreIndicator(interaction);
                    break;
            }

            // Add Netflix-level visual enhancement
            await this.addNetflixQualityEffects(interaction);

            this.trackNetflixLevelMetric('visual_feedback', performance.now() - feedbackStart);

        } catch (error) {
            console.error('Visual feedback failed:', error);
        }
    }

    async generateUltraFastPreview(startTime, endTime, quality = 'ultra_fast') {
        const generationStart = performance.now();

        try {
            // Ultra-fast cache check
            const cacheKey = `preview_${this.currentSession}_${startTime}_${endTime}_${quality}`;
            if (this.cache.previews.has(cacheKey)) {
                const cachedPreview = this.cache.previews.get(cacheKey);
                await this.displayUltraFastPreview(cachedPreview);
                this.trackCacheHit('preview');
                return;
            }

            // Show instant loading with smooth animation
            this.showUltraFastPreviewLoading(startTime, endTime);

            // Parallel processing: request + local preparation
            const [serverResponse] = await Promise.all([
                this.requestServerPreview(startTime, endTime, quality),
                this.preparePreviewContainer(),
                this.preloadViralAnalysis(startTime, endTime)
            ]);

            if (serverResponse.success) {
                // Cache the result for instant future access
                this.cache.previews.set(cacheKey, serverResponse);

                // Display with smooth transition
                await this.displayUltraFastPreview(serverResponse);

                // Track generation performance
                const generationTime = performance.now() - generationStart;
                this.performanceMetrics.previewGenerationTime.push(generationTime);

                // Update real-time performance display
                this.updatePerformanceDisplay('preview_generation', generationTime);
            }

        } catch (error) {
            console.error('Ultra-fast preview generation failed:', error);
            this.showPreviewError(error.message);
        }
    }

    async displayUltraFastPreview(previewData) {
        const previewPlayer = document.getElementById('previewPlayer');
        const previewInfo = document.getElementById('previewInfo');

        // Ultra-smooth transition
        previewPlayer.style.transition = 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)';

        // Enhanced preview display with Netflix-level quality indicators
        previewPlayer.innerHTML = `
            <div class="ultra-fast-preview-container">
                <video controls class="preview-video ultra-fast" autoplay muted>
                    <source src="${previewData.preview_url}" type="video/mp4">
                    Your browser does not support ultra-fast video previews.
                </video>
                <div class="preview-quality-badge">⚡ Ultra-Fast</div>
                <div class="preview-generation-time">${previewData.generation_time?.toFixed(1)}ms</div>
            </div>
        `;

        // Enhanced viral analysis display
        if (previewData.viral_analysis) {
            previewInfo.innerHTML = `
                <div class="ultra-fast-preview-analysis">
                    <div class="analysis-header">
                        <div class="viral-score-indicator ${this.getScoreClass(previewData.viral_analysis.viral_score)}">
                            <span class="score-number">${previewData.viral_analysis.viral_score}</span>
                            <span class="score-label">Viral Score</span>
                        </div>
                        <div class="confidence-meter">
                            <div class="confidence-fill" style="width: ${(previewData.viral_analysis.confidence * 100)}%"></div>
                            <span class="confidence-text">${Math.round(previewData.viral_analysis.confidence * 100)}% Confidence</span>
                        </div>
                    </div>
                    <div class="viral-factors-grid">
                        ${previewData.viral_analysis.factors?.map(factor =>
                            `<div class="factor-chip">✓ ${factor}</div>`
                        ).join('') || ''}
                    </div>
                </div>
            `;
        }

        // Display optimization suggestions with interactive elements
        if (previewData.suggestions && previewData.suggestions.length > 0) {
            previewInfo.innerHTML += `
                <div class="ultra-fast-suggestions">
                    <h4>⚡ Instant Optimization Suggestions</h4>
                    <div class="suggestions-list">
                        ${previewData.suggestions.map((suggestion, index) =>
                            `<div class="suggestion-item interactive" data-suggestion="${index}">
                                <span class="suggestion-icon">💡</span>
                                <span class="suggestion-text">${suggestion}</span>
                                <button class="apply-suggestion-btn" onclick="window.viralClipApp.applySuggestion('${index}')">Apply</button>
                            </div>`
                        ).join('')}
                    </div>
                </div>
            `;
        }

        // Add performance indicators
        if (previewData.cache_hit) {
            this.showCacheHitIndicator('preview');
        }
    }

    showUltraFastPreviewLoading(startTime, endTime) {
        const previewPlayer = document.getElementById('previewPlayer');
        previewPlayer.innerHTML = `
            <div class="ultra-fast-loading">
                <div class="loading-animation">
                    <div class="netflix-spinner"></div>
                    <div class="loading-particles"></div>
                </div>
                <div class="loading-info">
                    <h3>⚡ Generating Ultra-Fast Preview</h3>
                    <p>Segment: ${startTime.toFixed(1)}s - ${endTime.toFixed(1)}s</p>
                    <div class="loading-progress">
                        <div class="progress-bar ultra-fast" id="ultraFastProgress"></div>
                    </div>
                    <div class="loading-stats">
                        <span>Target: < 75ms</span>
                        <span>Quality: Netflix-Level</span>
                    </div>
                </div>
            </div>
        `;

        // Animate progress bar for visual feedback
        this.animateUltraFastProgress();
    }

    animateUltraFastProgress() {
        const progressBar = document.getElementById('ultraFastProgress');
        if (!progressBar) return;

        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15 + 5; // Random but fast progress
            progressBar.style.width = Math.min(progress, 95) + '%';

            if (progress >= 95) {
                clearInterval(interval);
            }
        }, 10);
    }

    async requestServerPreview(startTime, endTime, quality) {
        try {
            const response = await fetch('/api/v5/generate-preview', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.currentSession,
                    start_time: startTime,
                    end_time: endTime,
                    quality: quality,
                    ultra_fast_mode: true,
                    client_timestamp: performance.now()
                })
            });

            return await response.json();
        } catch (error) {
            console.error('Server preview request failed:', error);
            throw error;
        }
    }

    async preparePreviewContainer() {
        // Pre-prepare container elements for smoother transitions
        const previewPlayer = document.getElementById('previewPlayer');
        previewPlayer.classList.add('preparing');

        // Pre-allocate DOM elements to avoid layout thrashing
        await new Promise(resolve => setTimeout(resolve, 5));

        previewPlayer.classList.remove('preparing');
    }

    async preloadViralAnalysis(startTime, endTime) {
        // Pre-calculate or cache viral analysis for instant display
        const cacheKey = `viral_${this.currentSession}_${Math.floor(startTime)}_${Math.floor(endTime)}`;

        if (!this.cache.viralScores.has(cacheKey)) {
            // Mock pre-calculation - in production, this would be real analysis
            const mockAnalysis = {
                viral_score: Math.floor(Math.random() * 40) + 60,
                confidence: Math.random() * 0.3 + 0.7,
                factors: ['High engagement potential', 'Good timing', 'Strong visuals']
            };

            this.cache.viralScores.set(cacheKey, mockAnalysis);
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
        if (this.cache.previews.has(cacheKey)) {
            this.displayPreview(this.cache.previews.get(cacheKey));
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
                this.cache.previews.set(cacheKey, result);

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

    // New methods for ultra-fast interactions
    async processInteractionResponse(response) {
        console.log('Processing interaction response:', response);
        // Implement response handling logic
    }

    async provideInstantLocalFeedback(interaction) {
        console.log('Providing instant local feedback for interaction:', interaction);
        // Implement visual or auditory feedback
    }

    trackPerformanceMetric(metricName, duration) {
        console.log(`Performance Metric ${metricName}:`, duration, 'ms');
        // Store and display performance metrics
    }

    trackCacheHit(cacheType) {
        console.log(`Cache hit for ${cacheType}`);
        // Implement cache hit tracking
    }

    updatePerformanceDisplay(metricName, duration) {
        console.log(`Updating display for ${metricName}:`, duration, 'ms');
        // Update UI elements with performance data
    }

    showCacheHitIndicator(type) {
        console.log(`Showing cache hit indicator for ${type}`);
        // Implement cache hit indicator
    }

    applySuggestion(suggestionIndex) {
        console.log(`Applying suggestion at index ${suggestionIndex}`);
        // Implement suggestion application logic
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

// Export for global access
window.NetflixLevelApp = NetflixLevelApp;