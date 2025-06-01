/**
 * ViralClip Pro - Netflix-Level Frontend Application v4.0
 * One-Click Upload System with Instant Preview and Mobile-First Design
 */

class ViralClipApp {
    constructor() {
        this.config = {
            apiBase: window.location.origin,
            wsBase: window.location.origin.replace('http', 'ws'),
            version: '4.0.0',
            maxFileSize: 2 * 1024 * 1024 * 1024, // 2GB
            allowedTypes: ['video/mp4', 'video/mov', 'video/avi', 'video/webm', 'video/quicktime'],
            chunkSize: 8192 // 8KB chunks for upload
        };

        this.state = {
            currentSession: null,
            uploadProgress: 0,
            processingStatus: 'idle',
            isConnected: false,
            isDragging: false,
            currentUpload: null,
            previewData: null,
            isMobile: this.detectMobile(),
            errors: [],
            metrics: {}
        };

        this.websockets = new Map();
        this.uploadQueue = [];
        this.retryAttempts = 0;
        this.maxRetries = 3;

        this.initializeApp();
    }

    detectMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
               window.innerWidth <= 768;
    }

    async initializeApp() {
        try {
            console.log('🚀 Initializing ViralClip Pro v4.0 with Netflix-level features...');

            // Initialize core components
            await this.setupEventListeners();
            await this.initializeUI();
            await this.setupWebSockets();
            await this.setupMobileOptimizations();

            // Optimize for mobile
            if (this.state.isMobile) {
                this.enableMobileFeatures();
            }

            // Performance monitoring
            this.startPerformanceMonitoring();

            console.log('✅ ViralClip Pro v4.0 initialized successfully');

        } catch (error) {
            this.handleError('App initialization failed', error);
        }
    }

    async setupEventListeners() {
        // Enhanced drag-drop for desktop and mobile
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        if (uploadArea) {
            // Desktop drag-drop events
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadArea.addEventListener('drop', this.handleFileDrop.bind(this));
            uploadArea.addEventListener('click', () => fileInput?.click());

            // Mobile touch events for enhanced UX
            uploadArea.addEventListener('touchstart', this.handleTouchStart.bind(this));
            uploadArea.addEventListener('touchend', this.handleTouchEnd.bind(this));
        }

        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Global mobile optimizations
        document.addEventListener('touchmove', this.preventOverscroll.bind(this), { passive: false });
        window.addEventListener('resize', this.handleResize.bind(this));
        window.addEventListener('orientationchange', this.handleOrientationChange.bind(this));

        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));

        // Error handling
        window.addEventListener('error', this.handleGlobalError.bind(this));
        window.addEventListener('unhandledrejection', this.handleUnhandledRejection.bind(this));
    }

    async initializeUI() {
        this.updateConnectionStatus();
        this.setupProgressBars();
        this.initializeMobileUI();

        // Netflix-style loading animations
        this.setupLoadingAnimations();
    }

    async setupMobileOptimizations() {
        if (this.state.isMobile) {
            // Add mobile-specific classes
            document.body.classList.add('mobile-optimized');

            // Optimize viewport
            const viewport = document.querySelector('meta[name="viewport"]');
            if (viewport) {
                viewport.content = 'width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no';
            }

            // Enable touch-friendly interactions
            this.enableTouchFriendlyUI();
        }
    }

    enableMobileFeatures() {
        // Enhanced mobile upload area
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.classList.add('mobile-upload');

            // Add mobile-specific styling
            uploadArea.style.minHeight = '200px';
            uploadArea.style.padding = '2rem 1rem';
            uploadArea.style.fontSize = '1.2rem';
        }

        // Mobile-optimized progress display
        this.setupMobileProgress();
    }

    enableTouchFriendlyUI() {
        // Increase touch targets
        const buttons = document.querySelectorAll('button, .clickable');
        buttons.forEach(button => {
            button.style.minHeight = '44px';
            button.style.minWidth = '44px';
            button.style.padding = '12px 16px';
        });

        // Add haptic feedback simulation
        this.setupHapticFeedback();
    }

    setupHapticFeedback() {
        // Simulate haptic feedback with visual cues
        const touchElements = document.querySelectorAll('button, .upload-area, .clickable');
        touchElements.forEach(element => {
            element.addEventListener('touchstart', () => {
                element.style.transform = 'scale(0.95)';
                element.style.transition = 'transform 0.1s ease';
            });

            element.addEventListener('touchend', () => {
                element.style.transform = 'scale(1)';
            });
        });
    }

    setupMobileProgress() {
        // Create mobile-optimized progress overlay
        const progressOverlay = document.createElement('div');
        progressOverlay.id = 'mobileProgressOverlay';
        progressOverlay.className = 'mobile-progress-overlay';
        progressOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.9);
            display: none;
            z-index: 10000;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        `;

        document.body.appendChild(progressOverlay);
    }

    async setupWebSockets() {
        try {
            await this.connectWebSocket('main', '/api/v3/ws/app');
            this.state.isConnected = true;
            this.updateConnectionStatus();
        } catch (error) {
            console.warn('WebSocket setup failed:', error);
            this.state.isConnected = false;
            this.updateConnectionStatus();
            this.scheduleReconnect();
        }
    }

    async connectWebSocket(type, endpoint) {
        try {
            const wsUrl = `${this.config.wsBase}${endpoint}`;
            const ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log(`✅ WebSocket connected: ${type}`);
                this.websockets.set(type, ws);
                this.retryAttempts = 0;
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
                    this.updateUploadProgress(data);
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
                case 'connection_established':
                    this.handleConnectionEstablished(data);
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

    // Netflix-Level Drag and Drop with Instant Preview
    handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();

        if (!this.state.isDragging) {
            this.state.isDragging = true;
            this.showDragFeedback(true);
        }

        // Enhanced visual feedback
        const uploadArea = event.currentTarget;
        uploadArea.classList.add('drag-over');

        // Show file type validation in real-time
        const items = event.dataTransfer.items;
        if (items && items.length > 0) {
            const file = items[0];
            this.showDragPreview(file);
        }
    }

    handleDragLeave(event) {
        event.preventDefault();
        event.stopPropagation();

        // Only remove drag state if leaving the upload area entirely
        const uploadArea = event.currentTarget;
        const rect = uploadArea.getBoundingClientRect();
        const x = event.clientX;
        const y = event.clientY;

        if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
            this.state.isDragging = false;
            uploadArea.classList.remove('drag-over');
            this.showDragFeedback(false);
            this.hideDragPreview();
        }
    }

    async handleFileDrop(event) {
        event.preventDefault();
        event.stopPropagation();

        const uploadArea = event.currentTarget;
        uploadArea.classList.remove('drag-over');
        this.state.isDragging = false;
        this.showDragFeedback(false);

        const files = Array.from(event.dataTransfer.files);
        await this.processFileUpload(files);
    }

    async handleFileSelect(event) {
        const files = Array.from(event.target.files);
        await this.processFileUpload(files);
    }

    handleTouchStart(event) {
        // Visual feedback for touch
        const uploadArea = event.currentTarget;
        uploadArea.classList.add('touch-active');
    }

    handleTouchEnd(event) {
        const uploadArea = event.currentTarget;
        uploadArea.classList.remove('touch-active');

        // Trigger file picker on touch
        setTimeout(() => {
            const fileInput = document.getElementById('fileInput');
            if (fileInput) {
                fileInput.click();
            }
        }, 100);
    }

    showDragFeedback(show) {
        const uploadArea = document.getElementById('uploadArea');
        if (!uploadArea) return;

        if (show) {
            uploadArea.innerHTML = `
                <div class="upload-icon animate-bounce">📹</div>
                <div class="upload-text">
                    Drop your video here for instant preview!
                </div>
                <div class="upload-hint">
                    Netflix-level processing • Instant analysis • Mobile optimized
                </div>
            `;
        } else {
            this.resetUploadArea();
        }
    }

    showDragPreview(file) {
        // Real-time file validation feedback
        const isValid = this.validateFileType(file.type);
        const uploadArea = document.getElementById('uploadArea');

        if (isValid) {
            uploadArea.classList.add('valid-file');
            uploadArea.classList.remove('invalid-file');
        } else {
            uploadArea.classList.add('invalid-file');
            uploadArea.classList.remove('valid-file');
        }
    }

    hideDragPreview() {
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.classList.remove('valid-file', 'invalid-file');
        }
    }

    resetUploadArea() {
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.innerHTML = `
                <div class="upload-icon">📹</div>
                <div class="upload-text">
                    ${this.state.isMobile ? 'Tap to select video' : 'Drop your video here or click to browse'}
                </div>
                <div class="upload-hint">
                    Supports MP4, MOV, AVI, WebM • Max 2GB • Netflix-level quality processing
                </div>
            `;
        }
    }

    // Netflix-Level File Processing with Instant Preview
    async processFileUpload(files) {
        try {
            if (files.length === 0) return;

            const file = files[0];

            // Enhanced validation
            const validation = this.validateFile(file);
            if (!validation.valid) {
                this.showError(validation.error);
                return;
            }

            // Generate upload ID
            const uploadId = this.generateUploadId();

            // Show instant loading state
            this.showInstantFeedback(file);

            // Start upload with real-time progress
            await this.uploadFileWithProgress(file, uploadId);

        } catch (error) {
            this.handleError('File upload failed', error);
        }
    }

    showInstantFeedback(file) {
        // Instant visual feedback
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.innerHTML = `
                <div class="upload-processing">
                    <div class="upload-icon processing">🎬</div>
                    <div class="upload-text">
                        Processing ${file.name}...
                    </div>
                    <div class="instant-analysis">
                        <div class="analysis-item">📊 File size: ${this.formatFileSize(file.size)}</div>
                        <div class="analysis-item">🎯 Type: ${file.type}</div>
                        <div class="analysis-item">⚡ Netflix-level processing starting...</div>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar-wrapper">
                            <div id="uploadProgress" class="progress-bar"></div>
                        </div>
                        <div id="uploadProgressText" class="progress-text">0%</div>
                    </div>
                </div>
            `;
        }

        // Show mobile overlay if needed
        if (this.state.isMobile) {
            this.showMobileUploadOverlay(file);
        }
    }

    showMobileUploadOverlay(file) {
        const overlay = document.getElementById('mobileProgressOverlay');
        if (overlay) {
            overlay.innerHTML = `
                <div class="mobile-upload-content">
                    <div class="upload-icon mobile-icon">🎬</div>
                    <h2>Processing Video</h2>
                    <p>${file.name}</p>
                    <div class="mobile-progress">
                        <div class="mobile-progress-bar">
                            <div id="mobileUploadProgress" class="mobile-progress-fill"></div>
                        </div>
                        <div id="mobileProgressText" class="mobile-progress-text">0%</div>
                    </div>
                    <div class="mobile-analysis">
                        <div class="analysis-stat">
                            <span class="stat-label">Size:</span>
                            <span class="stat-value">${this.formatFileSize(file.size)}</span>
                        </div>
                        <div class="analysis-stat">
                            <span class="stat-label">Type:</span>
                            <span class="stat-value">${file.type}</span>
                        </div>
                    </div>
                </div>
            `;
            overlay.style.display = 'flex';
        }
    }

    async uploadFileWithProgress(file, uploadId) {
        try {
            // Connect upload WebSocket for real-time progress
            await this.connectWebSocket('upload', `/api/v3/ws/upload/${uploadId}`);

            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('upload_id', uploadId);

            // Track upload progress
            const xhr = new XMLHttpRequest();

            // Upload progress handler
            xhr.upload.addEventListener('progress', (event) => {
                if (event.lengthComputable) {
                    const progress = (event.loaded / event.total) * 100;
                    this.updateUploadProgressUI(progress);
                }
            });

            // Response handler
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    const result = JSON.parse(xhr.responseText);
                    this.handleUploadSuccess(result);
                } else {
                    throw new Error(`Upload failed: ${xhr.statusText}`);
                }
            });

            // Error handler
            xhr.addEventListener('error', () => {
                throw new Error('Upload failed: Network error');
            });

            // Start upload
            xhr.open('POST', '/api/v3/upload-video');
            xhr.send(formData);

        } catch (error) {
            this.handleError('Upload failed', error);
        }
    }

    updateUploadProgressUI(progress) {
        // Update desktop progress
        const progressBar = document.getElementById('uploadProgress');
        const progressText = document.getElementById('uploadProgressText');

        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }

        if (progressText) {
            progressText.textContent = `${Math.round(progress)}%`;
        }

        // Update mobile progress
        if (this.state.isMobile) {
            const mobileProgressBar = document.getElementById('mobileUploadProgress');
            const mobileProgressText = document.getElementById('mobileProgressText');

            if (mobileProgressBar) {
                mobileProgressBar.style.width = `${progress}%`;
            }

            if (mobileProgressText) {
                mobileProgressText.textContent = `${Math.round(progress)}%`;
            }
        }

        this.state.uploadProgress = progress;
    }

    handleUploadSuccess(result) {
        console.log('Upload successful:', result);

        this.state.currentSession = result.session_id;

        // Show instant preview if available
        if (result.preview) {
            this.displayInstantPreview(result.preview);
        }

        // Hide mobile overlay
        if (this.state.isMobile) {
            const overlay = document.getElementById('mobileProgressOverlay');
            if (overlay) {
                overlay.style.display = 'none';
            }
        }

        // Show success message
        this.showSuccess('Video uploaded successfully! Analyzing for viral potential...');

        // Show processing UI
        this.showProcessingUI();
    }

    displayInstantPreview(previewData) {
        const previewContainer = document.getElementById('livePreviewContainer');
        if (!previewContainer) return;

        // Show instant preview
        previewContainer.innerHTML = `
            <div class="instant-preview">
                <div class="preview-header">
                    <h3>📺 Instant Preview Ready</h3>
                    <div class="preview-score">
                        Viral Score: <span class="score-value">${previewData.viral_analysis?.viral_score || 'Analyzing...'}</span>
                    </div>
                </div>
                <div class="preview-content">
                    <img src="${previewData.thumbnail_url}" alt="Video thumbnail" class="preview-thumbnail">
                    <div class="preview-analysis">
                        <h4>🚀 Viral Factors Detected:</h4>
                        <ul class="factors-list">
                            ${previewData.viral_analysis?.factors?.map(factor => `<li>${factor}</li>`).join('') || '<li>Analyzing...</li>'}
                        </ul>
                    </div>
                </div>
                <div class="preview-actions">
                    <button class="preview-btn primary" onclick="app.startProcessing()">
                        🎬 Generate Clips
                    </button>
                    <button class="preview-btn secondary" onclick="app.analyzeMore()">
                        📊 Deep Analysis
                    </button>
                </div>
            </div>
        `;

        // Show preview section
        const previewSection = document.getElementById('previewSection');
        if (previewSection) {
            previewSection.style.display = 'block';
        }
    }

    // Real-time Progress Updates via WebSocket
    updateUploadProgress(data) {
        if (data.progress !== undefined) {
            this.updateUploadProgressUI(data.progress);
        }

        // Real-time status updates
        if (data.status) {
            this.updateUploadStatus(data.status);
        }
    }

    updateProcessingStatus(data) {
        const statusElement = document.getElementById('processingStatus');
        const messageElement = document.getElementById('processingMessage');
        const progressElement = document.getElementById('processingProgress');

        if (statusElement) {
            statusElement.textContent = data.stage;
            statusElement.className = `processing-stage stage-${data.stage.replace(/\s+/g, '-').toLowerCase()}`;
        }

        if (messageElement) {
            messageElement.textContent = data.message;
        }

        if (progressElement) {
            progressElement.style.width = `${data.progress}%`;
        }

        // Show processing section
        const processingSection = document.getElementById('processingSection');
        if (processingSection) {
            processingSection.style.display = 'block';
        }

        this.state.processingStatus = data.stage;
    }

    // Utility Methods
    validateFile(file) {
        if (!file) {
            return { valid: false, error: 'No file selected' };
        }

        if (file.size > this.config.maxFileSize) {
            return { valid: false, error: 'File too large (max 2GB)' };
        }

        if (!this.validateFileType(file.type)) {
            return { valid: false, error: `Unsupported file type: ${file.type}` };
        }

        return { valid: true };
    }

    validateFileType(type) {
        return this.config.allowedTypes.includes(type);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    generateUploadId() {
        return `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    // Mobile Responsive Handlers
    handleResize() {
        const wasMobile = this.state.isMobile;
        this.state.isMobile = this.detectMobile();

        if (wasMobile !== this.state.isMobile) {
            if (this.state.isMobile) {
                this.enableMobileFeatures();
            } else {
                this.disableMobileFeatures();
            }
        }
    }

    handleOrientationChange() {
        // Handle orientation changes on mobile
        setTimeout(() => {
            this.handleResize();
            this.adjustMobileLayout();
        }, 100);
    }

    adjustMobileLayout() {
        if (this.state.isMobile) {
            const orientation = window.orientation;
            document.body.classList.toggle('landscape', Math.abs(orientation) === 90);
        }
    }

    preventOverscroll(event) {
        // Prevent iOS bounce scrolling when not needed
        if (this.state.isMobile && event.touches.length === 1) {
            const touch = event.touches[0];
            const target = touch.target;

            // Allow scrolling on specific elements
            if (!target.closest('.scrollable')) {
                event.preventDefault();
            }
        }
    }

    disableMobileFeatures() {
        document.body.classList.remove('mobile-optimized');
        this.resetToDesktopUI();
    }

    resetToDesktopUI() {
        this.resetUploadArea();

        // Hide mobile overlay
        const overlay = document.getElementById('mobileProgressOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    // Error Handling
    handleError(message, error) {
        console.error(message, error);
        this.showError(`${message}: ${error.message || error}`);
    }

    showError(message) {
        const errorContainer = document.getElementById('errorContainer') || document.body;

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
        const successContainer = document.getElementById('successContainer') || document.body;

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

    // Connection Management
    updateConnectionStatus() {
        const statusIndicator = document.getElementById('connectionStatus');
        if (statusIndicator) {
            statusIndicator.className = this.state.isConnected ? 'connection-status connected' : 'connection-status disconnected';
            statusIndicator.textContent = this.state.isConnected ? 'Connected' : 'Connecting...';
        }
    }

    scheduleReconnect(type = 'main', endpoint = '/api/v3/ws/app') {
        if (this.retryAttempts >= this.maxRetries) {
            console.error(`Max WebSocket reconnection attempts reached for ${type}`);
            return;
        }

        this.retryAttempts++;
        const delay = Math.pow(2, this.retryAttempts) * 1000;

        setTimeout(() => {
            console.log(`Attempting to reconnect WebSocket: ${type}`);
            this.connectWebSocket(type, endpoint);
        }, delay);
    }

    // Performance Monitoring
    startPerformanceMonitoring() {
        const domContentLoaded = performance.now();
        console.log('📊 Page Load Performance:', domContentLoaded, 'ms');

        window.addEventListener('load', () => {
            const fullLoadTime = performance.now();
            console.log(`📊 Total Load Time: ${fullLoadTime}ms`);
        });
    }

    // Additional handlers and utility methods...
    handleConnectionEstablished(data) {
        console.log('Connection established:', data);
        this.state.isConnected = true;
        this.updateConnectionStatus();
    }

    handlePreviewReady(data) {
        console.log('Preview ready:', data);
        if (data.preview_url) {
            this.displayInstantPreview(data);
        }
    }

    handleKeyboardShortcuts(event) {
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return;
        }

        switch (event.key) {
            case ' ':
                event.preventDefault();
                // Space bar functionality
                break;
            case 'u':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    document.getElementById('fileInput')?.click();
                }
                break;
        }
    }

    handleGlobalError(event) {
        this.handleError('Unexpected error occurred', event.error);
    }

    handleUnhandledRejection(event) {
        this.handleError('Promise rejection', event.reason);
    }

    handleWebSocketError(type, error) {
        console.error(`WebSocket error (${type}):`, error);
        this.handleError(`WebSocket connection failed (${type})`, error);
    }

    setupProgressBars() {
        const progressBars = document.querySelectorAll('.progress-bar');
        progressBars.forEach(bar => {
            bar.style.transition = 'width 0.3s ease';
        });
    }

    initializeMobileUI() {
        if (this.state.isMobile) {
            document.body.classList.add('mobile');

            // Add mobile-specific meta tags
            if (!document.querySelector('meta[name="mobile-web-app-capable"]')) {
                const meta = document.createElement('meta');
                meta.name = 'mobile-web-app-capable';
                meta.content = 'yes';
                document.head.appendChild(meta);
            }
        }
    }

    setupLoadingAnimations() {
        // Netflix-style loading animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes netflixPulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.8; transform: scale(1.05); }
            }

            .processing { animation: netflixPulse 2s infinite; }
            .animate-bounce { animation: bounce 1s infinite; }

            @keyframes bounce {
                0%, 20%, 53%, 80%, 100% { transform: translateY(0); }
                40%, 43% { transform: translateY(-10px); }
                70% { transform: translateY(-5px); }
                90% { transform: translateY(-2px); }
            }
        `;
        document.head.appendChild(style);
    }

    showProcessingUI() {
        const processingSection = document.getElementById('processingSection');
        if (processingSection) {
            processingSection.style.display = 'block';
        }
    }

    updateViralScore(data) {
        const scoreElement = document.getElementById('viralScore');
        if (scoreElement) {
            scoreElement.textContent = data.viral_score;
            scoreElement.className = this.getScoreClass(data.viral_score);
        }
    }

    updateTimeline(data) {
        console.log('Timeline update:', data);
        // Timeline visualization would be implemented here
    }

    getScoreClass(score) {
        if (score >= 80) return 'score-excellent';
        if (score >= 60) return 'score-good';
        if (score >= 40) return 'score-average';
        return 'score-poor';
    }

    startProcessing() {
        console.log('Starting video processing...');
        // Implement processing logic
    }

    analyzeMore() {
        console.log('Starting deep analysis...');
        // Implement deep analysis logic
    }
}

// Initialize the application
let app;

document.addEventListener('DOMContentLoaded', () => {
    try {
        app = new ViralClipApp();
        window.app = app; // Make available globally for button clicks
    } catch (error) {
        console.error('Failed to initialize ViralClip Pro:', error);
    }
});

// Export for global access
if (typeof window !== 'undefined') {
    window.ViralClipApp = ViralClipApp;
}