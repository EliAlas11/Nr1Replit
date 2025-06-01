/**
 * ViralClip Pro v4.0 - Netflix-Level Frontend Architecture
 * Enterprise-grade real-time viral video processor
 */

class ViralClipApp {
    constructor() {
        this.config = {
            api: {
                base: '/api/v1',
                websocket: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
                timeout: 30000,
                retries: 3
            },
            upload: {
                maxSize: 500 * 1024 * 1024, // 500MB
                allowedTypes: ['video/mp4', 'video/mov', 'video/avi', 'video/mkv'],
                chunkSize: 1024 * 1024 // 1MB chunks
            },
            ui: {
                animations: {
                    fast: 150,
                    normal: 300,
                    slow: 500
                },
                breakpoints: {
                    mobile: 768,
                    tablet: 1024,
                    desktop: 1200
                }
            }
        };

        this.state = {
            isProcessing: false,
            currentFile: null,
            uploadProgress: 0,
            processingProgress: 0,
            websocket: null,
            retryCount: 0,
            currentSession: null,
            realtimeData: {},
            viralScores: [],
            timeline: {
                duration: 0,
                currentTime: 0,
                segments: []
            }
        };

        this.eventEmitter = new EventTarget();
        this.init();
    }

    async init() {
        try {
            console.log('🚀 Initializing ViralClip Pro v4.0 with Netflix-level architecture...');

            await this.setupUI();
            await this.setupWebSocket();
            await this.registerServiceWorker();
            await this.loadUserPreferences();

            this.trackEvent('app_initialized_v4');
            console.log('✅ ViralClip Pro v4.0 initialized successfully');
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            this.showError('Failed to initialize application. Please refresh the page.');
        }
    }

    async setupUI() {
        this.bindEvents();
        this.setupDragDrop();
        this.setupResponsiveDesign();
        this.initializeComponents();
        this.measurePerformance();
    }

    bindEvents() {
        // Upload triggers
        document.getElementById('fileInput')?.addEventListener('change', this.handleFileSelect.bind(this));
        document.getElementById('uploadButton')?.addEventListener('click', this.triggerFileSelect.bind(this));

        // Processing controls
        document.getElementById('processButton')?.addEventListener('click', this.startProcessing.bind(this));
        document.getElementById('cancelButton')?.addEventListener('click', this.cancelProcessing.bind(this));

        // Timeline interactions
        document.getElementById('timeline')?.addEventListener('click', this.handleTimelineClick.bind(this));
        document.getElementById('playButton')?.addEventListener('click', this.togglePlayback.bind(this));

        // Quality controls
        document.querySelectorAll('input[name="quality"]').forEach(radio => {
            radio.addEventListener('change', this.handleQualityChange.bind(this));
        });

        // Mobile-specific events
        if (this.isMobile()) {
            this.setupMobileEvents();
        }
    }

    setupDragDrop() {
        const dropZone = document.getElementById('dropZone');
        if (!dropZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults.bind(this), false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => this.addDropHighlight(dropZone), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => this.removeDropHighlight(dropZone), false);
        });

        dropZone.addEventListener('drop', this.handleDrop.bind(this), false);
    }

    setupResponsiveDesign() {
        const mediaQuery = window.matchMedia(`(max-width: ${this.config.ui.breakpoints.mobile}px)`);
        mediaQuery.addEventListener('change', this.handleScreenSizeChange.bind(this));
        this.handleScreenSizeChange(mediaQuery);
    }

    setupMobileEvents() {
        // Touch-specific optimizations
        document.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: true });
        document.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        document.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: true });
    }

    async setupWebSocket() {
        try {
            this.state.websocket = new WebSocket(this.config.api.websocket);

            this.state.websocket.onopen = () => {
                console.log('🔌 WebSocket connected');
                this.state.retryCount = 0;
            };

            this.state.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };

            this.state.websocket.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                this.scheduleWebSocketReconnect();
            };

            this.state.websocket.onerror = (error) => {
                console.error('🔌 WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'progress':
                this.updateProgress(data.progress);
                break;
            case 'viral_score':
                this.updateViralScore(data.score, data.timestamp);
                break;
            case 'segment_analysis':
                this.updateSegmentAnalysis(data.segments);
                break;
            case 'processing_complete':
                this.handleProcessingComplete(data.result);
                break;
            case 'error':
                this.handleProcessingError(data.error);
                break;
            case 'realtime_preview':
                this.updateRealtimePreview(data.preview);
                break;
        }
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            await this.validateFile(file);
            this.state.currentFile = file;
            await this.previewFile(file);
            this.updateUI('file-selected');
        } catch (error) {
            this.showError(error.message);
        }
    }

    async validateFile(file) {
        if (!this.config.upload.allowedTypes.includes(file.type)) {
            throw new Error(`Unsupported file type. Please use: ${this.config.upload.allowedTypes.join(', ')}`);
        }

        if (file.size > this.config.upload.maxSize) {
            throw new Error(`File too large. Maximum size: ${this.formatFileSize(this.config.upload.maxSize)}`);
        }

        // Additional validation for video files
        return new Promise((resolve, reject) => {
            const video = document.createElement('video');
            video.preload = 'metadata';

            video.onloadedmetadata = () => {
                if (video.duration > 600) { // 10 minutes max
                    reject(new Error('Video too long. Maximum duration: 10 minutes'));
                } else {
                    resolve();
                }
            };

            video.onerror = () => reject(new Error('Invalid video file'));
            video.src = URL.createObjectURL(file);
        });
    }

    async previewFile(file) {
        const preview = document.getElementById('videoPreview');
        if (!preview) return;

        const url = URL.createObjectURL(file);
        preview.src = url;
        preview.style.display = 'block';

        // Initialize timeline
        preview.onloadedmetadata = () => {
            this.state.timeline.duration = preview.duration;
            this.initializeTimeline(preview.duration);
        };
    }

    initializeTimeline(duration) {
        const timeline = document.getElementById('timeline');
        if (!timeline) return;

        timeline.innerHTML = '';
        timeline.className = 'timeline-container';

        // Create timeline segments for viral score visualization
        const segmentCount = Math.min(Math.ceil(duration / 5), 20); // Max 20 segments

        for (let i = 0; i < segmentCount; i++) {
            const segment = document.createElement('div');
            segment.className = 'timeline-segment';
            segment.dataset.segment = i;
            segment.style.width = `${100 / segmentCount}%`;

            const scoreBar = document.createElement('div');
            scoreBar.className = 'viral-score-bar';
            segment.appendChild(scoreBar);

            timeline.appendChild(segment);
        }
    }

    async startProcessing() {
        if (!this.state.currentFile) {
            this.showError('Please select a video file first');
            return;
        }

        try {
            this.state.isProcessing = true;
            this.updateUI('processing-start');

            // Generate session ID
            this.state.currentSession = this.generateSessionId();

            // Start file upload with progress
            await this.uploadFile(this.state.currentFile);

            // Start real-time analysis
            await this.startRealtimeAnalysis();

        } catch (error) {
            console.error('Processing failed:', error);
            this.showError(error.message);
            this.resetProcessing();
        }
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('video', file);
        formData.append('session_id', this.state.currentSession);

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            xhr.upload.onprogress = (event) => {
                if (event.lengthComputable) {
                    this.state.uploadProgress = (event.loaded / event.total) * 100;
                    this.updateProgressBar('upload', this.state.uploadProgress);
                }
            };

            xhr.onload = () => {
                if (xhr.status === 200) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new Error(`Upload failed: ${xhr.statusText}`));
                }
            };

            xhr.onerror = () => reject(new Error('Upload failed'));

            xhr.open('POST', `${this.config.api.base}/upload`);
            xhr.send(formData);
        });
    }

    async startRealtimeAnalysis() {
        if (!this.state.websocket || this.state.websocket.readyState !== WebSocket.OPEN) {
            throw new Error('WebSocket not connected');
        }

        this.state.websocket.send(JSON.stringify({
            type: 'start_analysis',
            session_id: this.state.currentSession,
            options: {
                realtime_preview: true,
                viral_scoring: true,
                segment_analysis: true
            }
        }));
    }

    updateProgress(progress) {
        this.state.processingProgress = progress.percentage;
        this.updateProgressBar('processing', progress.percentage);

        if (progress.stage) {
            this.updateProcessingStage(progress.stage);
        }

        if (progress.eta) {
            this.updateETA(progress.eta);
        }
    }

    updateViralScore(score, timestamp) {
        this.state.viralScores.push({ score, timestamp });
        this.visualizeViralScore(score, timestamp);
        this.updateOverallViralScore();
    }

    visualizeViralScore(score, timestamp) {
        const timeline = document.getElementById('timeline');
        if (!timeline) return;

        const segmentIndex = Math.floor((timestamp / this.state.timeline.duration) * timeline.children.length);
        const segment = timeline.children[segmentIndex];

        if (segment) {
            const scoreBar = segment.querySelector('.viral-score-bar');
            scoreBar.style.height = `${score}%`;
            scoreBar.className = `viral-score-bar ${this.getScoreClass(score)}`;
        }
    }

    getScoreClass(score) {
        if (score >= 80) return 'viral-high';
        if (score >= 60) return 'viral-medium';
        return 'viral-low';
    }

    updateOverallViralScore() {
        if (this.state.viralScores.length === 0) return;

        const avgScore = this.state.viralScores.reduce((sum, item) => sum + item.score, 0) / this.state.viralScores.length;
        const scoreElement = document.getElementById('overallViralScore');

        if (scoreElement) {
            scoreElement.textContent = Math.round(avgScore);
            scoreElement.className = `viral-score ${this.getScoreClass(avgScore)}`;
        }
    }

    updateRealtimePreview(previewData) {
        const previewContainer = document.getElementById('realtimePreview');
        if (!previewContainer) return;

        if (previewData.thumbnail) {
            const img = previewContainer.querySelector('img') || document.createElement('img');
            img.src = `data:image/jpeg;base64,${previewData.thumbnail}`;
            img.className = 'preview-thumbnail';

            if (!previewContainer.contains(img)) {
                previewContainer.appendChild(img);
            }
        }
    }

    handleProcessingComplete(result) {
        this.state.isProcessing = false;
        this.updateUI('processing-complete');

        // Display results
        this.displayResults(result);

        // Track completion
        this.trackEvent('processing_complete', {
            duration: result.processing_time,
            viral_score: result.viral_score
        });
    }

    displayResults(result) {
        const resultsContainer = document.getElementById('results');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = `
            <div class="results-header">
                <h2>🎉 Your Viral Clips Are Ready!</h2>
                <div class="viral-score-display">
                    <span class="score-value ${this.getScoreClass(result.viral_score)}">${result.viral_score}</span>
                    <span class="score-label">Viral Score</span>
                </div>
            </div>

            <div class="clips-grid">
                ${result.clips.map(clip => this.renderClipCard(clip)).join('')}
            </div>

            <div class="actions">
                <button class="btn-primary" onclick="app.downloadAll()">Download All</button>
                <button class="btn-secondary" onclick="app.shareClips()">Share Clips</button>
                <button class="btn-tertiary" onclick="app.processNew()">Process New Video</button>
            </div>
        `;

        resultsContainer.style.display = 'block';
        this.animateResults();
    }

    renderClipCard(clip) {
        return `
            <div class="clip-card" data-clip-id="${clip.id}">
                <div class="clip-preview">
                    <video poster="${clip.thumbnail}" preload="none">
                        <source src="${clip.url}" type="video/mp4">
                    </video>
                    <div class="play-button" onclick="app.playClip('${clip.id}')">▶</div>
                </div>

                <div class="clip-info">
                    <h3>${clip.title}</h3>
                    <div class="clip-stats">
                        <span class="duration">${this.formatDuration(clip.duration)}</span>
                        <span class="viral-score ${this.getScoreClass(clip.viral_score)}">${clip.viral_score}% viral</span>
                    </div>
                    <p class="clip-description">${clip.description}</p>
                </div>

                <div class="clip-actions">
                    <button onclick="app.downloadClip('${clip.id}')" class="btn-download">Download</button>
                    <button onclick="app.shareClip('${clip.id}')" class="btn-share">Share</button>
                </div>
            </div>
        `;
    }

    // Utility methods
    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    isMobile() {
        return window.innerWidth <= this.config.ui.breakpoints.mobile;
    }

    trackEvent(eventName, data = {}) {
        console.log('📊 Event:', eventName, data);

        // Send to analytics if available
        if (typeof gtag !== 'undefined') {
            gtag('event', eventName, data);
        }
    }

    measurePerformance() {
        if (performance.mark) {
            performance.mark('app-init-complete');

            const navigation = performance.getEntriesByType('navigation')[0];
            if (navigation) {
                console.log('📊 Page Load Performance:');
                console.log('DOM Content Loaded:', navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart, 'ms');
                console.log('Full Load Time:', navigation.loadEventEnd - navigation.loadEventStart, 'ms');
            }
        }
    }

    async registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                await navigator.serviceWorker.register('/sw.js');
                console.log('📱 Service Worker registered successfully');
            } catch (error) {
                console.log('📱 SW registration failed:', error);
            }
        }
    }

    // Event handlers (stub implementations)
    preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
    addDropHighlight(element) { element.classList.add('drop-highlight'); }
    removeDropHighlight(element) { element.classList.remove('drop-highlight'); }
    handleDrop(e) { this.handleFileSelect({ target: { files: e.dataTransfer.files } }); }
    handleScreenSizeChange(e) { document.body.classList.toggle('mobile', e.matches); }
    handleTouchStart(e) { /* Touch handling */ }
    handleTouchMove(e) { /* Touch handling */ }
    handleTouchEnd(e) { /* Touch handling */ }
    handleTimelineClick(e) { /* Timeline interaction */ }
    togglePlayback() { /* Playback control */ }
    handleQualityChange(e) { /* Quality selection */ }
    triggerFileSelect() { document.getElementById('fileInput')?.click(); }
    cancelProcessing() { this.resetProcessing(); }
    updateUI(state) { document.body.dataset.state = state; }
    updateProgressBar(type, progress) { /* Progress bar updates */ }
    updateProcessingStage(stage) { /* Stage updates */ }
    updateETA(eta) { /* ETA display */ }
    animateResults() { /* Results animation */ }
    showError(message) { console.error(message); alert(message); }
    resetProcessing() { this.state.isProcessing = false; this.updateUI('idle'); }
    scheduleWebSocketReconnect() { /* Reconnection logic */ }
    loadUserPreferences() { /* Load preferences */ }
    downloadAll() { /* Download implementation */ }
    shareClips() { /* Share implementation */ }
    processNew() { location.reload(); }
    playClip(id) { /* Play clip */ }
    downloadClip(id) { /* Download single clip */ }
    shareClip(id) { /* Share single clip */ }
}

// Global app instance
let app;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        app = new ViralClipApp();
    });
} else {
    app = new ViralClipApp();
}

// Global error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (app) {
        app.trackEvent('javascript_error', {
            message: event.error?.message,
            filename: event.filename,
            lineno: event.lineno
        });
    }
});