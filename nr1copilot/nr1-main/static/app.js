
/**
 * ViralClip Pro - Netflix-Level Frontend Application
 * Production-ready with comprehensive error handling and real-time features
 */

class ViralClipApp {
    constructor() {
        this.config = {
            apiBase: window.location.origin,
            wsBase: window.location.origin.replace('http', 'ws'),
            version: '3.0.0',
            features: {
                realtime: true,
                websockets: true,
                preview: true,
                analytics: true
            }
        };
        
        this.state = {
            currentSession: null,
            uploadProgress: 0,
            processingStatus: 'idle',
            isConnected: false,
            errors: [],
            metrics: {},
            timeline: null
        };
        
        this.websockets = new Map();
        this.eventListeners = new Map();
        this.uploadQueue = [];
        this.retryAttempts = 0;
        this.maxRetries = 3;
        
        this.initializeApp();
    }

    async initializeApp() {
        try {
            console.log('🚀 Initializing ViralClip Pro v3.0 with real-time features...');
            
            // Initialize core components
            await this.setupEventListeners();
            await this.initializeUI();
            await this.setupWebSockets();
            await this.loadUserPreferences();
            
            // Performance monitoring
            this.startPerformanceMonitoring();
            
            // Health check
            await this.performHealthCheck();
            
            this.logEvent('app_initialized_v3', {
                version: this.config.version,
                features: this.config.features,
                timestamp: Date.now()
            });
            
            console.log('✅ ViralClip Pro v3.0 initialized successfully');
            
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
        
        // Initialize tooltips and interactive elements
        this.initializeTooltips();
        this.initializeProgressBars();
        this.initializeTimeline();
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
            };
            
            ws.onmessage = (event) => {
                this.handleWebSocketMessage(type, event);
            };
            
            ws.onclose = () => {
                console.log(`🔌 WebSocket disconnected: ${type}`);
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

    async connectViralScoreWebSocket(sessionId) {
        await this.connectWebSocket('viral_scores', `/api/v3/ws/viral-scores/${sessionId}`);
    }

    async connectTimelineWebSocket(sessionId) {
        await this.connectWebSocket('timeline', `/api/v3/ws/timeline/${sessionId}`);
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

    updateProcessingStatus(data) {
        const statusElement = document.getElementById('processingStatus');
        const messageElement = document.getElementById('processingMessage');
        
        if (statusElement) {
            statusElement.textContent = data.stage;
        }
        
        if (messageElement) {
            messageElement.textContent = data.message;
        }
        
        if (data.entertaining_fact) {
            this.showEntertainingFact(data.entertaining_fact);
        }
        
        this.state.processingStatus = data.stage;
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
        
        // Render interactive timeline
        this.renderTimeline(data);
        this.state.timeline = data;
    }

    renderTimeline(data) {
        const container = document.getElementById('timelineVisualization');
        if (!container) return;
        
        container.innerHTML = '';
        
        // Create timeline SVG
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100');
        svg.setAttribute('viewBox', '0 0 1000 100');
        
        // Draw viral score heatmap
        if (data.viral_heatmap) {
            this.drawViralHeatmap(svg, data.viral_heatmap);
        }
        
        // Draw key moments
        if (data.key_moments) {
            this.drawKeyMoments(svg, data.key_moments);
        }
        
        container.appendChild(svg);
    }

    drawViralHeatmap(svg, heatmap) {
        heatmap.forEach((score, index) => {
            const x = (index / heatmap.length) * 1000;
            const height = (score / 100) * 80;
            const color = this.getHeatmapColor(score);
            
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x);
            rect.setAttribute('y', 100 - height);
            rect.setAttribute('width', 1000 / heatmap.length);
            rect.setAttribute('height', height);
            rect.setAttribute('fill', color);
            rect.setAttribute('opacity', '0.7');
            
            svg.appendChild(rect);
        });
    }

    drawKeyMoments(svg, moments) {
        moments.forEach(moment => {
            const x = (moment.timestamp / moment.duration) * 1000;
            
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', x);
            line.setAttribute('y1', 0);
            line.setAttribute('x2', x);
            line.setAttribute('y2', 100);
            line.setAttribute('stroke', '#ff6b6b');
            line.setAttribute('stroke-width', '2');
            line.setAttribute('stroke-dasharray', '5,5');
            
            svg.appendChild(line);
        });
    }

    getHeatmapColor(score) {
        if (score >= 80) return '#4ecdc4';
        if (score >= 60) return '#45b7aa';
        if (score >= 40) return '#f9ca24';
        if (score >= 20) return '#f0932b';
        return '#eb4d4b';
    }

    getScoreClass(score) {
        if (score >= 80) return 'score-excellent';
        if (score >= 60) return 'score-good';
        if (score >= 40) return 'score-average';
        return 'score-poor';
    }

    showEntertainingFact(fact) {
        const factElement = document.getElementById('entertainingFact');
        if (factElement) {
            factElement.textContent = fact;
            factElement.style.display = 'block';
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                factElement.style.display = 'none';
            }, 5000);
        }
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

    // UI Helper Methods
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

    // Event Handlers
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
            case 'retry':
                this.retryLastOperation();
                break;
            default:
                console.log(`Unhandled action: ${action}`);
        }
    }

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

    // Error Handling
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
        errorElement.className = 'error-message';
        errorElement.textContent = message;
        
        errorContainer.appendChild(errorElement);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            errorElement.remove();
        }, 5000);
    }

    showSuccess(message) {
        const successContainer = document.getElementById('successContainer');
        if (!successContainer) {
            console.log(message);
            return;
        }
        
        const successElement = document.createElement('div');
        successElement.className = 'success-message';
        successElement.textContent = message;
        
        successContainer.appendChild(successElement);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            successElement.remove();
        }, 3000);
    }

    // Utility Methods
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

    startPerformanceMonitoring() {
        // DOM Content Loaded
        const domContentLoaded = performance.now();
        console.log('📊 Page Load Performance:');
        console.log('DOM Content Loaded:', domContentLoaded, 'ms');
        
        // Full page load
        window.addEventListener('load', () => {
            const fullLoadTime = performance.now() - domContentLoaded;
            console.log('Full Load Time:', fullLoadTime, 'ms');
            
            // Log total load time
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

    // WebSocket Management
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
        // Apply user preferences to the UI
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

    // Initialization helpers
    initializeTooltips() {
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', this.showTooltip.bind(this));
            element.addEventListener('mouseleave', this.hideTooltip.bind(this));
        });
    }

    initializeProgressBars() {
        const progressBars = document.querySelectorAll('.progress-bar');
        progressBars.forEach(bar => {
            bar.style.transition = 'width 0.3s ease';
        });
    }

    initializeTimeline() {
        const timelineContainer = document.getElementById('timelineContainer');
        if (timelineContainer) {
            timelineContainer.addEventListener('click', this.handleTimelineClick.bind(this));
        }
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

    handleTimelineClick(event) {
        const rect = event.currentTarget.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const percentage = x / rect.width;
        
        if (this.state.timeline && this.state.timeline.duration) {
            const timestamp = percentage * this.state.timeline.duration;
            this.seekToTimestamp(timestamp);
        }
    }

    seekToTimestamp(timestamp) {
        console.log(`Seeking to timestamp: ${timestamp}s`);
        // Implement timeline seeking functionality
    }

    // Cleanup
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
}

// Initialize the application
let app;

document.addEventListener('DOMContentLoaded', () => {
    app = new ViralClipApp();
});

// Export for global access (if needed)
if (typeof window !== 'undefined') {
    window.ViralClipApp = ViralClipApp;
}
