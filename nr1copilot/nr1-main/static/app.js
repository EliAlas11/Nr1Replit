
/**
 * ViralClip Pro v6.0 - Netflix-Level Frontend Architecture
 * Enterprise-grade client with advanced performance and user experience
 */

class NetflixLevelApp {
    constructor() {
        // Core state management
        this.state = {
            isUploading: false,
            isProcessing: false,
            currentSession: null,
            uploadProgress: 0,
            files: new Map(),
            templates: new Map(),
            captions: new Map(),
            batchJobs: new Map(),
            brandKits: new Map()
        };

        // Performance optimization
        this.performanceObserver = new PerformanceObserver((list) => {
            this.handlePerformanceEntries(list.getEntries());
        });

        // WebSocket connections
        this.websockets = new Map();
        this.reconnectAttempts = new Map();
        this.maxReconnectAttempts = 5;

        // UI state
        this.activeTab = 'upload';
        this.notificationQueue = [];
        this.modalStack = [];

        // Initialize
        this.init();
    }

    async init() {
        try {
            console.log('🎬 Initializing ViralClip Pro v6.0 - Netflix Architecture');

            // Initialize performance monitoring
            this.initPerformanceMonitoring();

            // Setup event listeners
            this.setupEventListeners();

            // Initialize UI components
            this.initializeUI();

            // Load user preferences
            await this.loadUserPreferences();

            // Initialize real-time features
            this.initializeRealTimeFeatures();

            // Start health monitoring
            this.startHealthMonitoring();

            console.log('✅ ViralClip Pro initialized successfully');

        } catch (error) {
            console.error('❌ Initialization failed:', error);
            this.showErrorNotification('Application initialization failed', error.message);
        }
    }

    initPerformanceMonitoring() {
        // Monitor Core Web Vitals
        this.performanceObserver.observe({ entryTypes: ['navigation', 'paint', 'largest-contentful-paint'] });

        // Monitor resource loading
        this.performanceObserver.observe({ entryTypes: ['resource'] });

        // Custom performance metrics
        this.metrics = {
            startTime: performance.now(),
            interactionCount: 0,
            errorCount: 0,
            uploadCount: 0,
            averageUploadTime: 0
        };
    }

    setupEventListeners() {
        // File upload drag and drop
        this.setupDragAndDrop();

        // Form submissions
        this.setupFormHandlers();

        // Navigation
        this.setupNavigation();

        // Real-time updates
        this.setupWebSocketHandlers();

        // Keyboard shortcuts
        this.setupKeyboardShortcuts();

        // Window events
        window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
        window.addEventListener('online', this.handleOnline.bind(this));
        window.addEventListener('offline', this.handleOffline.bind(this));
    }

    setupDragAndDrop() {
        const dropZones = document.querySelectorAll('.drop-zone');
        
        dropZones.forEach(zone => {
            zone.addEventListener('dragover', this.handleDragOver.bind(this));
            zone.addEventListener('dragleave', this.handleDragLeave.bind(this));
            zone.addEventListener('drop', this.handleFileDrop.bind(this));
        });
    }

    handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
        
        const dropZone = event.currentTarget;
        dropZone.classList.add('drag-over');
        
        // Add visual feedback
        if (!dropZone.querySelector('.drag-indicator')) {
            const indicator = document.createElement('div');
            indicator.className = 'drag-indicator';
            indicator.innerHTML = `
                <div class="drag-icon">📁</div>
                <div class="drag-text">Drop your video here</div>
            `;
            dropZone.appendChild(indicator);
        }
    }

    handleDragLeave(event) {
        event.preventDefault();
        
        const dropZone = event.currentTarget;
        dropZone.classList.remove('drag-over');
        
        const indicator = dropZone.querySelector('.drag-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    async handleFileDrop(event) {
        event.preventDefault();
        
        const dropZone = event.currentTarget;
        dropZone.classList.remove('drag-over');
        
        const indicator = dropZone.querySelector('.drag-indicator');
        if (indicator) {
            indicator.remove();
        }

        const files = Array.from(event.dataTransfer.files);
        
        if (files.length === 0) {
            this.showErrorNotification('No files detected', 'Please drop valid video files');
            return;
        }

        // Validate files
        const validFiles = this.validateFiles(files);
        
        if (validFiles.length === 0) {
            this.showErrorNotification('Invalid files', 'Please drop valid video files (MP4, MOV, AVI)');
            return;
        }

        // Process files
        for (const file of validFiles) {
            await this.processFile(file);
        }
    }

    validateFiles(files) {
        const allowedTypes = ['video/mp4', 'video/mov', 'video/avi', 'video/quicktime'];
        const maxSize = 500 * 1024 * 1024; // 500MB

        return files.filter(file => {
            if (!allowedTypes.includes(file.type)) {
                this.showWarningNotification(`Unsupported file type: ${file.name}`, 'Please use MP4, MOV, or AVI files');
                return false;
            }

            if (file.size > maxSize) {
                this.showWarningNotification(`File too large: ${file.name}`, 'Maximum file size is 500MB');
                return false;
            }

            return true;
        });
    }

    async processFile(file) {
        try {
            const sessionId = this.generateSessionId();
            
            // Update UI immediately
            this.showFilePreview(file, sessionId);
            
            // Start upload
            await this.uploadFile(file, sessionId);
            
        } catch (error) {
            console.error('File processing failed:', error);
            this.showErrorNotification('Upload failed', error.message);
        }
    }

    showFilePreview(file, sessionId) {
        const previewContainer = document.getElementById('file-previews');
        
        if (!previewContainer) {
            console.warn('Preview container not found');
            return;
        }

        const preview = document.createElement('div');
        preview.className = 'file-preview';
        preview.id = `preview-${sessionId}`;
        
        // Create thumbnail
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.addEventListener('loadedmetadata', () => {
            // Capture thumbnail at 1 second
            video.currentTime = 1;
        });
        
        video.addEventListener('seeked', () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            const thumbnail = canvas.toDataURL('image/jpeg', 0.8);
            
            preview.innerHTML = `
                <div class="preview-thumbnail">
                    <img src="${thumbnail}" alt="Video thumbnail" />
                    <div class="preview-overlay">
                        <div class="play-button">▶</div>
                    </div>
                </div>
                <div class="preview-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${this.formatFileSize(file.size)}</div>
                    <div class="upload-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 0%"></div>
                        </div>
                        <div class="progress-text">Preparing upload...</div>
                    </div>
                </div>
                <div class="preview-actions">
                    <button class="btn-cancel" onclick="app.cancelUpload('${sessionId}')">Cancel</button>
                </div>
            `;
            
            URL.revokeObjectURL(video.src);
        });

        previewContainer.appendChild(preview);
    }

    async uploadFile(file, sessionId) {
        try {
            this.state.isUploading = true;
            
            // Initialize upload session
            const initResponse = await this.apiCall('/api/v6/upload/init', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: file.name,
                    file_size: file.size,
                    upload_id: sessionId,
                    total_chunks: Math.ceil(file.size / (1024 * 1024)) // 1MB chunks
                })
            });

            if (!initResponse.success) {
                throw new Error(initResponse.message || 'Upload initialization failed');
            }

            // Start chunked upload
            await this.uploadFileInChunks(file, sessionId);
            
            // Start real-time analysis
            this.startRealTimeAnalysis(sessionId);
            
        } catch (error) {
            console.error('Upload failed:', error);
            this.updateUploadProgress(sessionId, 0, 'Upload failed', true);
            throw error;
        }
    }

    async uploadFileInChunks(file, sessionId) {
        const chunkSize = 1024 * 1024; // 1MB chunks
        const totalChunks = Math.ceil(file.size / chunkSize);
        
        for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
            const start = chunkIndex * chunkSize;
            const end = Math.min(start + chunkSize, file.size);
            const chunk = file.slice(start, end);
            
            const formData = new FormData();
            formData.append('file', chunk);
            formData.append('upload_id', sessionId);
            formData.append('chunk_index', chunkIndex.toString());
            formData.append('total_chunks', totalChunks.toString());
            
            const response = await this.apiCall('/api/v6/upload/chunk', {
                method: 'POST',
                body: formData
            });

            if (!response.success) {
                throw new Error(`Chunk upload failed: ${response.message}`);
            }

            // Update progress
            const progress = ((chunkIndex + 1) / totalChunks) * 100;
            this.updateUploadProgress(sessionId, progress, 'Uploading...');
        }

        this.updateUploadProgress(sessionId, 100, 'Upload complete');
    }

    updateUploadProgress(sessionId, progress, message, isError = false) {
        const preview = document.getElementById(`preview-${sessionId}`);
        
        if (!preview) return;

        const progressFill = preview.querySelector('.progress-fill');
        const progressText = preview.querySelector('.progress-text');

        if (progressFill) {
            progressFill.style.width = `${progress}%`;
            
            if (isError) {
                progressFill.style.background = 'var(--error-gradient)';
            } else if (progress === 100) {
                progressFill.style.background = 'var(--success-gradient)';
            }
        }

        if (progressText) {
            progressText.textContent = message;
            
            if (isError) {
                progressText.style.color = 'var(--error-color)';
            }
        }
    }

    startRealTimeAnalysis(sessionId) {
        // Connect to viral insights WebSocket
        this.connectWebSocket(`viral-insights/${sessionId}`, (data) => {
            this.handleViralInsights(sessionId, data);
        });

        // Update UI for analysis phase
        this.updateUploadProgress(sessionId, 100, 'Analyzing viral potential...');
    }

    handleViralInsights(sessionId, data) {
        console.log('🎯 Viral insights received:', data);

        const preview = document.getElementById(`preview-${sessionId}`);
        if (!preview) return;

        switch (data.type) {
            case 'early_analysis':
                this.showEarlyInsights(preview, data.insights);
                break;
            
            case 'viral_score_update':
                this.updateViralScore(preview, data.viral_score);
                break;
            
            case 'sentiment_analysis':
                this.showSentimentAnalysis(preview, data.sentiment);
                break;
            
            case 'analysis_complete':
                this.showCompleteAnalysis(preview, data.results);
                break;
        }
    }

    showEarlyInsights(preview, insights) {
        const insightsContainer = preview.querySelector('.preview-info');
        
        if (!insightsContainer.querySelector('.viral-insights')) {
            const viralInsights = document.createElement('div');
            viralInsights.className = 'viral-insights';
            viralInsights.innerHTML = `
                <div class="viral-score">
                    <span class="score-label">Viral Score:</span>
                    <span class="score-value">${insights.viral_score}/100</span>
                </div>
                <div class="key-factors">
                    ${insights.key_factors.map(factor => `<span class="factor-tag">${factor}</span>`).join('')}
                </div>
            `;
            insightsContainer.appendChild(viralInsights);
        }
    }

    // Template Management
    async loadTemplates() {
        try {
            const response = await this.apiCall('/api/v6/templates');
            
            if (response.success) {
                this.state.templates.clear();
                
                response.templates.forEach(template => {
                    this.state.templates.set(template.template_id, template);
                });
                
                this.renderTemplateGallery();
            }
            
        } catch (error) {
            console.error('Failed to load templates:', error);
            this.showErrorNotification('Template loading failed', error.message);
        }
    }

    renderTemplateGallery() {
        const gallery = document.getElementById('template-gallery');
        if (!gallery) return;

        gallery.innerHTML = '';

        this.state.templates.forEach(template => {
            const templateCard = document.createElement('div');
            templateCard.className = 'template-card';
            templateCard.innerHTML = `
                <div class="template-preview">
                    <img src="${template.thumbnail_url}" alt="${template.name}" loading="lazy" />
                    <div class="template-overlay">
                        <div class="viral-score">${template.viral_score}/100</div>
                        <div class="template-actions">
                            <button class="btn-preview" onclick="app.previewTemplate('${template.template_id}')">
                                Preview
                            </button>
                            <button class="btn-use" onclick="app.useTemplate('${template.template_id}')">
                                Use Template
                            </button>
                        </div>
                    </div>
                </div>
                <div class="template-info">
                    <h3 class="template-name">${template.name}</h3>
                    <p class="template-description">${template.description}</p>
                    <div class="template-meta">
                        <span class="category">${template.category}</span>
                        <span class="platform">${template.platform}</span>
                        <span class="usage-count">${template.usage_count} uses</span>
                    </div>
                </div>
            `;

            gallery.appendChild(templateCard);
        });
    }

    async generateCaptions(sessionId, options = {}) {
        try {
            this.showProcessingIndicator('Generating smart captions...');

            const formData = new FormData();
            formData.append('session_id', sessionId);
            formData.append('language', options.language || 'en');
            formData.append('platform', options.platform || 'auto');
            formData.append('viral_enhancement', options.viral_enhancement !== false);

            // Get the uploaded file for this session
            const fileInput = document.querySelector(`input[data-session="${sessionId}"]`);
            if (fileInput && fileInput.files[0]) {
                formData.append('file', fileInput.files[0]);
            }

            const response = await this.apiCall('/api/v6/captions/generate', {
                method: 'POST',
                body: formData
            });

            if (response.success) {
                this.state.captions.set(sessionId, response.captions);
                this.showCaptionResults(sessionId, response.captions);
            } else {
                throw new Error(response.message || 'Caption generation failed');
            }

        } catch (error) {
            console.error('Caption generation failed:', error);
            this.showErrorNotification('Caption generation failed', error.message);
        } finally {
            this.hideProcessingIndicator();
        }
    }

    showCaptionResults(sessionId, captions) {
        const resultsContainer = document.getElementById('caption-results');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = `
            <div class="caption-header">
                <h3>Generated Captions</h3>
                <div class="caption-stats">
                    <span class="viral-score">Viral Score: ${captions.overall_viral_score}/100</span>
                    <span class="speaker-count">Speakers: ${captions.speaker_count || 1}</span>
                </div>
            </div>

            <div class="caption-segments">
                ${captions.segments.map(segment => `
                    <div class="caption-segment" data-start="${segment.start_time}" data-end="${segment.end_time}">
                        <div class="segment-time">
                            ${this.formatTime(segment.start_time)} - ${this.formatTime(segment.end_time)}
                        </div>
                        <div class="segment-text">${segment.text}</div>
                        <div class="segment-meta">
                            <span class="confidence">Confidence: ${Math.round(segment.confidence * 100)}%</span>
                            <span class="viral-score">Viral: ${segment.viral_score}/100</span>
                            <span class="emotion">${segment.emotion}</span>
                        </div>
                    </div>
                `).join('')}
            </div>

            <div class="caption-actions">
                <button class="btn btn-primary" onclick="app.exportCaptions('${sessionId}', 'srt')">
                    Export SRT
                </button>
                <button class="btn btn-secondary" onclick="app.exportCaptions('${sessionId}', 'vtt')">
                    Export VTT
                </button>
                <button class="btn btn-secondary" onclick="app.editCaptions('${sessionId}')">
                    Edit Captions
                </button>
            </div>
        `;

        resultsContainer.style.display = 'block';
    }

    // Batch Processing
    async submitBatchJob(jobType, inputData, priority = 'normal') {
        try {
            const response = await this.apiCall('/api/v6/batch/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_type: jobType,
                    input_data: inputData,
                    priority: priority,
                    session_id: this.state.currentSession
                })
            });

            if (response.success) {
                this.state.batchJobs.set(response.job_id, {
                    id: response.job_id,
                    type: jobType,
                    priority: priority,
                    status: 'queued',
                    submitted_at: new Date().toISOString()
                });

                this.showSuccessNotification('Job submitted', `Job ${response.job_id} queued successfully`);
                this.updateBatchJobsUI();
                
                return response.job_id;
            } else {
                throw new Error(response.message || 'Job submission failed');
            }

        } catch (error) {
            console.error('Batch job submission failed:', error);
            this.showErrorNotification('Job submission failed', error.message);
            throw error;
        }
    }

    async loadBatchJobs() {
        try {
            const response = await this.apiCall('/api/v6/user/jobs');
            
            if (response.success) {
                this.state.batchJobs.clear();
                
                response.jobs.forEach(job => {
                    this.state.batchJobs.set(job.job_id, job);
                });
                
                this.updateBatchJobsUI();
            }
            
        } catch (error) {
            console.error('Failed to load batch jobs:', error);
        }
    }

    updateBatchJobsUI() {
        const container = document.getElementById('batch-jobs');
        if (!container) return;

        container.innerHTML = `
            <div class="batch-header">
                <h3>Batch Jobs</h3>
                <button class="btn btn-sm" onclick="app.loadBatchJobs()">Refresh</button>
            </div>
            <div class="jobs-list">
                ${Array.from(this.state.batchJobs.values()).map(job => `
                    <div class="job-item" data-status="${job.status}">
                        <div class="job-info">
                            <div class="job-id">${job.job_id}</div>
                            <div class="job-type">${job.job_type}</div>
                            <div class="job-status ${job.status}">${job.status}</div>
                        </div>
                        <div class="job-actions">
                            ${job.status === 'queued' || job.status === 'running' ? 
                                `<button class="btn-cancel" onclick="app.cancelBatchJob('${job.job_id}')">Cancel</button>` : 
                                ''
                            }
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    // Utility Methods
    connectWebSocket(endpoint, onMessage) {
        const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/v6/ws/${endpoint}`;
        
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log(`🔌 WebSocket connected: ${endpoint}`);
            this.reconnectAttempts.set(endpoint, 0);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(data);
            } catch (error) {
                console.error('WebSocket message parsing failed:', error);
            }
        };

        ws.onclose = (event) => {
            console.log(`🔌 WebSocket disconnected: ${endpoint}`);
            
            // Attempt reconnection
            const attempts = this.reconnectAttempts.get(endpoint) || 0;
            if (attempts < this.maxReconnectAttempts) {
                setTimeout(() => {
                    this.reconnectAttempts.set(endpoint, attempts + 1);
                    this.connectWebSocket(endpoint, onMessage);
                }, Math.pow(2, attempts) * 1000); // Exponential backoff
            }
        };

        ws.onerror = (error) => {
            console.error(`WebSocket error: ${endpoint}`, error);
        };

        this.websockets.set(endpoint, ws);
        return ws;
    }

    async apiCall(endpoint, options = {}) {
        try {
            const response = await fetch(endpoint, {
                ...options,
                headers: {
                    ...options.headers
                }
            });

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status} ${response.statusText}`);
            }

            return await response.json();

        } catch (error) {
            console.error(`API call failed: ${endpoint}`, error);
            throw error;
        }
    }

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

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    showNotification(type, title, message) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-icon">${type === 'error' ? '❌' : type === 'success' ? '✅' : '⚠️'}</div>
            <div class="notification-content">
                <div class="notification-title">${title}</div>
                <div class="notification-message">${message}</div>
            </div>
            <button class="notification-close" onclick="this.parentElement.remove()">×</button>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    showSuccessNotification(title, message) {
        this.showNotification('success', title, message);
    }

    showErrorNotification(title, message) {
        this.showNotification('error', title, message);
    }

    showWarningNotification(title, message) {
        this.showNotification('warning', title, message);
    }

    showProcessingIndicator(message) {
        let indicator = document.getElementById('processing-indicator');
        
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'processing-indicator';
            indicator.className = 'processing-indicator';
            document.body.appendChild(indicator);
        }

        indicator.innerHTML = `
            <div class="processing-content">
                <div class="spinner"></div>
                <div class="processing-message">${message}</div>
            </div>
        `;
        
        indicator.style.display = 'flex';
    }

    hideProcessingIndicator() {
        const indicator = document.getElementById('processing-indicator');
        if (indicator) {
            indicator.style.display = 'none';
        }
    }

    // Initialize the app when the page loads
    initializeUI() {
        // Setup tab navigation
        this.setupTabNavigation();
        
        // Load initial data
        this.loadTemplates();
        this.loadBatchJobs();
        
        // Setup periodic updates
        setInterval(() => {
            this.loadBatchJobs();
        }, 30000); // Update every 30 seconds
    }

    setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;
                
                // Update active tab
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                button.classList.add('active');
                document.getElementById(`${targetTab}-tab`).classList.add('active');
                
                this.activeTab = targetTab;
            });
        });
    }

    startHealthMonitoring() {
        setInterval(async () => {
            try {
                const response = await this.apiCall('/api/v6/health');
                
                if (response.status !== 'healthy') {
                    this.showWarningNotification('System Health', 'Some services may be experiencing issues');
                }
                
            } catch (error) {
                console.warn('Health check failed:', error);
            }
        }, 60000); // Check every minute
    }
}

// Initialize the application
const app = new NetflixLevelApp();

// Export for global access
window.app = app;
