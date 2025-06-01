
/**
 * ViralClip Pro - Frontend Application
 * Netflix-level user experience with real-time updates
 */

class ViralClipApp {
    constructor() {
        this.currentStep = 'upload';
        this.uploadConnection = null;
        this.processingConnection = null;
        this.uploadId = null;
        this.sessionId = null;
        this.taskId = null;
        this.selectedClips = [];
        this.isDragging = false;
        this.uploadProgress = 0;
        this.processingProgress = 0;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.setupMobileOptimizations();
        this.showStep('upload');
        
        // Initialize performance monitoring
        this.startPerformanceMonitoring();
        
        console.log('🚀 ViralClip Pro initialized successfully');
    }

    setupEventListeners() {
        // Upload form
        const uploadForm = document.getElementById('upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', this.handleFileUpload.bind(this));
        }

        // File input
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // URL form
        const urlForm = document.getElementById('url-form');
        if (urlForm) {
            urlForm.addEventListener('submit', this.handleUrlAnalysis.bind(this));
        }

        // Processing form
        const processForm = document.getElementById('process-form');
        if (processForm) {
            processForm.addEventListener('submit', this.handleProcessing.bind(this));
        }

        // Navigation buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-action]')) {
                this.handleAction(e.target.dataset.action, e.target);
            }
        });

        // Escape key to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideAllModals();
            }
        });

        // Handle visibility change for WebSocket reconnection
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                this.reconnectWebSockets();
            }
        });
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('upload-area');
        const dragOverlay = document.getElementById('drag-overlay');

        if (!uploadArea || !dragOverlay) return;

        // Click to upload
        uploadArea.addEventListener('click', () => {
            document.getElementById('file-input').click();
        });

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, this.preventDefaults, false);
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            document.addEventListener(eventName, this.handleDragEnter.bind(this), false);
            uploadArea.addEventListener(eventName, this.handleDragOver.bind(this), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, this.handleDragLeave.bind(this), false);
            uploadArea.addEventListener(eventName, this.handleDragLeave.bind(this), false);
        });

        // Handle dropped files
        uploadArea.addEventListener('drop', this.handleDrop.bind(this), false);
        document.addEventListener('drop', this.handleDrop.bind(this), false);

        // Handle paste events for files
        document.addEventListener('paste', this.handlePaste.bind(this));
    }

    setupMobileOptimizations() {
        // Touch event optimizations
        document.addEventListener('touchstart', () => {}, { passive: true });
        document.addEventListener('touchmove', () => {}, { passive: true });

        // Viewport height fix for mobile browsers
        const setVH = () => {
            const vh = window.innerHeight * 0.01;
            document.documentElement.style.setProperty('--vh', `${vh}px`);
        };
        
        setVH();
        window.addEventListener('resize', setVH);
        window.addEventListener('orientationchange', setVH);

        // Prevent zoom on double tap for iOS
        let lastTouchEnd = 0;
        document.addEventListener('touchend', (event) => {
            const now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                event.preventDefault();
            }
            lastTouchEnd = now;
        }, false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDragEnter(e) {
        if (!this.isDragging) {
            this.isDragging = true;
            this.showDragOverlay();
        }
    }

    handleDragOver(e) {
        const uploadArea = document.getElementById('upload-area');
        if (uploadArea && e.target.closest('#upload-area')) {
            uploadArea.classList.add('drag-over');
        }
    }

    handleDragLeave(e) {
        // Only hide overlay if leaving the entire window
        if (e.clientX === 0 && e.clientY === 0) {
            this.isDragging = false;
            this.hideDragOverlay();
            const uploadArea = document.getElementById('upload-area');
            if (uploadArea) {
                uploadArea.classList.remove('drag-over');
            }
        }
    }

    handleDrop(e) {
        this.isDragging = false;
        this.hideDragOverlay();
        
        const uploadArea = document.getElementById('upload-area');
        if (uploadArea) {
            uploadArea.classList.remove('drag-over');
        }

        const files = e.dataTransfer.files;
        if (files.length > 0 && e.target.closest('#upload-area')) {
            this.processFiles(files);
        }
    }

    showDragOverlay() {
        const overlay = document.getElementById('drag-overlay');
        if (overlay) {
            overlay.classList.add('show');
        }
    }

    hideDragOverlay() {
        const overlay = document.getElementById('drag-overlay');
        if (overlay) {
            overlay.classList.remove('show');
        }
    }

    async handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            await this.processFiles(files);
        }
    }

    async processFiles(files) {
        const file = files[0];
        
        if (!this.validateFile(file)) {
            return;
        }

        // Show instant preview
        await this.showInstantPreview(file);
        
        // Start upload after preview
        setTimeout(() => {
            this.uploadFile(file);
        }, 1500);
    }

    async showInstantPreview(file) {
        const uploadArea = document.getElementById('upload-area');
        const preview = document.createElement('div');
        preview.className = 'instant-preview';
        preview.innerHTML = `
            <video class="preview-video" autoplay muted loop>
                <source src="${URL.createObjectURL(file)}" type="${file.type}">
            </video>
            <div class="preview-overlay">
                <div class="preview-info">
                    <div class="file-details">
                        <h4>${file.name}</h4>
                        <div class="file-stats">
                            <span>${this.formatBytes(file.size)}</span>
                            <span>${file.type}</span>
                            <span>Ready to process</span>
                        </div>
                    </div>
                </div>
                <div class="upload-progress-mini">
                    <div class="progress-bar-mini">
                        <div class="progress-fill-mini" style="width: 0%"></div>
                    </div>
                </div>
            </div>
        `;
        
        uploadArea.appendChild(preview);
        
        // Animate in
        setTimeout(() => {
            preview.classList.add('show');
        }, 100);

        // Update file input to trigger upload button enable
        const fileInput = document.getElementById('file-input');
        const uploadBtn = document.querySelector('#upload-form button[type="submit"]');
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload & Create Clips';
        }
    }

    handlePaste(e) {
        const items = e.clipboardData?.items;
        if (!items) return;

        for (let item of items) {
            if (item.type.startsWith('video/')) {
                const file = item.getAsFile();
                if (file) {
                    this.processFiles([file]);
                    break;
                }
            }
        }
    }

    validateFile(file) {
        const maxSize = 2 * 1024 * 1024 * 1024; // 2GB
        const allowedTypes = ['video/mp4', 'video/mov', 'video/avi', 'video/mkv', 'video/webm'];

        if (file.size > maxSize) {
            this.showError('File too large. Maximum size is 2GB.');
            return false;
        }

        if (!allowedTypes.includes(file.type)) {
            this.showError('Invalid file type. Please upload a video file.');
            return false;
        }

        return true;
    }

    async handleFileUpload(e) {
        e.preventDefault();
        const fileInput = document.getElementById('file-input');
        const files = fileInput.files;
        
        if (files.length === 0) {
            this.showError('Please select a file to upload.');
            return;
        }

        await this.processFiles(files);
    }

    async uploadFile(file) {
        try {
            this.uploadId = this.generateId();
            
            // Connect to upload WebSocket first
            await this.connectUploadWebSocket();
            
            // Update instant preview to show upload starting
            this.updateInstantPreviewProgress(0, 'Starting upload...');
            
            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('upload_id', this.uploadId);

            // Upload with progress tracking
            const response = await this.uploadWithProgress(formData);
            
            if (response.success) {
                this.sessionId = response.session_id;
                
                // Update preview to show analysis
                this.updateInstantPreviewProgress(100, 'Analysis complete!');
                
                // Show success animation
                setTimeout(() => {
                    this.hideInstantPreview();
                    this.showStep('analysis');
                    this.displayVideoAnalysis(response);
                }, 2000);
            } else {
                throw new Error(response.error || 'Upload failed');
            }
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showError(`Upload failed: ${error.message}`);
            this.hideInstantPreview();
        }
    }

    updateInstantPreviewProgress(progress, message) {
        const preview = document.querySelector('.instant-preview');
        if (!preview) return;

        const progressFill = preview.querySelector('.progress-fill-mini');
        const fileStats = preview.querySelector('.file-stats');
        
        if (progressFill) {
            progressFill.style.width = `${progress}%`;
        }
        
        if (fileStats && message) {
            const lastSpan = fileStats.querySelector('span:last-child');
            if (lastSpan) {
                lastSpan.textContent = message;
            }
        }

        // Add completion glow effect
        if (progress >= 100) {
            preview.classList.add('upload-complete');
        }
    }

    hideInstantPreview() {
        const preview = document.querySelector('.instant-preview');
        if (preview) {
            preview.classList.remove('show');
            setTimeout(() => {
                preview.remove();
            }, 300);
        }
    }

    async uploadWithProgress(formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const progress = Math.round((e.loaded / e.total) * 100);
                    this.updateUploadProgress(progress);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response);
                    } catch (error) {
                        reject(new Error('Invalid response format'));
                    }
                } else {
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Network error during upload'));
            });

            xhr.open('POST', '/api/v2/upload-video');
            xhr.send(formData);
        });
    }

    async handleUrlAnalysis(e) {
        e.preventDefault();
        const urlInput = document.getElementById('url-input');
        const url = urlInput.value.trim();
        
        if (!url) {
            this.showError('Please enter a video URL.');
            return;
        }

        if (!this.validateUrl(url)) {
            this.showError('Please enter a valid YouTube, TikTok, or Instagram URL.');
            return;
        }

        try {
            this.showAnalysisProgress();
            
            const response = await fetch('/api/v2/analyze-video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url })
            });

            const data = await response.json();
            
            if (data.success) {
                this.sessionId = data.session_id;
                this.hideAnalysisProgress();
                this.showStep('analysis');
                this.displayVideoAnalysis(data);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(`Analysis failed: ${error.message}`);
            this.hideAnalysisProgress();
        }
    }

    validateUrl(url) {
        const patterns = [
            /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)/,
            /^https?:\/\/(www\.)?tiktok\.com/,
            /^https?:\/\/(www\.)?instagram\.com/
        ];
        
        return patterns.some(pattern => pattern.test(url));
    }

    displayVideoAnalysis(data) {
        const container = document.getElementById('analysis-results');
        if (!container) return;

        const videoInfo = data.video_info || {};
        const insights = data.ai_insights || {};
        const clips = data.suggested_clips || [];

        container.innerHTML = `
            <div class="analysis-header">
                <div class="video-thumbnail">
                    <img src="${videoInfo.thumbnail || '/public/placeholder-thumb.jpg'}" 
                         alt="Video thumbnail" 
                         onerror="this.src='/public/placeholder-thumb.jpg'">
                </div>
                <div class="video-details">
                    <h3>${videoInfo.title || 'Uploaded Video'}</h3>
                    <div class="video-stats">
                        <span class="stat">
                            <i class="icon">⏱️</i>
                            ${this.formatDuration(videoInfo.duration || 0)}
                        </span>
                        <span class="stat">
                            <i class="icon">👀</i>
                            ${this.formatNumber(videoInfo.view_count || 0)} views
                        </span>
                        <span class="stat">
                            <i class="icon">🎯</i>
                            ${insights.viral_potential || 0}% viral potential
                        </span>
                    </div>
                </div>
            </div>

            <div class="ai-insights">
                <h4>🤖 AI Insights</h4>
                <div class="insights-grid">
                    <div class="insight-card">
                        <div class="insight-score">${insights.viral_potential || 0}%</div>
                        <div class="insight-label">Viral Potential</div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-score">${insights.confidence_score || 0}%</div>
                        <div class="insight-label">AI Confidence</div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-score">${insights.engagement_prediction || 0}%</div>
                        <div class="insight-label">Engagement Rate</div>
                    </div>
                </div>
            </div>

            <div class="suggested-clips">
                <h4>🎬 Suggested Viral Clips</h4>
                <div class="clips-grid">
                    ${clips.map((clip, index) => this.renderClipCard(clip, index)).join('')}
                </div>
            </div>
        `;

        // Setup clip selection
        this.setupClipSelection();
    }

    renderClipCard(clip, index) {
        return `
            <div class="clip-card" data-clip-index="${index}">
                <div class="clip-header">
                    <h5>${clip.title}</h5>
                    <div class="viral-score">
                        <span class="score">${clip.viral_score}%</span>
                        <span class="label">Viral Score</span>
                    </div>
                </div>
                
                <div class="clip-timeline">
                    <div class="timeline-bar">
                        <div class="timeline-segment" 
                             style="left: ${(clip.start_time / 180) * 100}%; 
                                    width: ${((clip.end_time - clip.start_time) / 180) * 100}%;">
                        </div>
                    </div>
                    <div class="timeline-labels">
                        <span>${this.formatTime(clip.start_time)}</span>
                        <span>${this.formatTime(clip.end_time)}</span>
                    </div>
                </div>

                <div class="clip-details">
                    <p>${clip.description}</p>
                    <div class="clip-stats">
                        <span>📱 ${clip.recommended_platforms?.join(', ') || 'All platforms'}</span>
                        <span>⏱️ ${clip.end_time - clip.start_time}s</span>
                        <span>📊 ${clip.estimated_views}</span>
                    </div>
                </div>

                <div class="clip-actions">
                    <label class="clip-checkbox">
                        <input type="checkbox" value="${index}">
                        <span class="checkmark"></span>
                        Select for processing
                    </label>
                </div>
            </div>
        `;
    }

    setupClipSelection() {
        const checkboxes = document.querySelectorAll('.clip-checkbox input[type="checkbox"]');
        const processButton = document.getElementById('process-clips-btn');
        
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateSelectedClips();
                if (processButton) {
                    processButton.disabled = this.selectedClips.length === 0;
                }
            });
        });
    }

    updateSelectedClips() {
        const checkboxes = document.querySelectorAll('.clip-checkbox input[type="checkbox"]:checked');
        this.selectedClips = Array.from(checkboxes).map(cb => parseInt(cb.value));
    }

    async handleProcessing(e) {
        e.preventDefault();
        
        if (this.selectedClips.length === 0) {
            this.showError('Please select at least one clip to process.');
            return;
        }

        try {
            this.taskId = this.generateId();
            
            // Connect to processing WebSocket
            await this.connectProcessingWebSocket();
            
            // Show processing step
            this.showStep('processing');
            
            // Start processing
            const formData = new FormData();
            formData.append('session_id', this.sessionId);
            formData.append('clips', JSON.stringify(this.selectedClips.map(index => ({
                index,
                title: `Clip ${index + 1}`,
                start_time: index * 60,
                end_time: (index + 1) * 60
            }))));
            formData.append('priority', 'high');

            const response = await fetch('/api/v2/process-video', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (data.success) {
                this.taskId = data.task_id;
                this.showProcessingProgress();
            } else {
                throw new Error(data.error || 'Processing failed to start');
            }
            
        } catch (error) {
            console.error('Processing error:', error);
            this.showError(`Processing failed: ${error.message}`);
        }
    }

    async connectUploadWebSocket() {
        if (this.uploadConnection) {
            this.uploadConnection.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v2/upload-progress/${this.uploadId}`;
        
        this.uploadConnection = new WebSocket(wsUrl);
        
        this.uploadConnection.onopen = () => {
            console.log('📡 Upload WebSocket connected');
        };
        
        this.uploadConnection.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleUploadMessage(message);
            } catch (error) {
                console.error('Upload WebSocket message error:', error);
            }
        };
        
        this.uploadConnection.onerror = (error) => {
            console.error('Upload WebSocket error:', error);
        };
        
        this.uploadConnection.onclose = () => {
            console.log('📡 Upload WebSocket disconnected');
        };

        // Send ping every 30 seconds to keep connection alive
        setInterval(() => {
            if (this.uploadConnection?.readyState === WebSocket.OPEN) {
                this.uploadConnection.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    async connectProcessingWebSocket() {
        if (this.processingConnection) {
            this.processingConnection.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v2/ws/${this.taskId}`;
        
        this.processingConnection = new WebSocket(wsUrl);
        
        this.processingConnection.onopen = () => {
            console.log('📡 Processing WebSocket connected');
        };
        
        this.processingConnection.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleProcessingMessage(message);
            } catch (error) {
                console.error('Processing WebSocket message error:', error);
            }
        };
        
        this.processingConnection.onerror = (error) => {
            console.error('Processing WebSocket error:', error);
        };
        
        this.processingConnection.onclose = () => {
            console.log('📡 Processing WebSocket disconnected');
        };

        // Send ping every 30 seconds to keep connection alive
        setInterval(() => {
            if (this.processingConnection?.readyState === WebSocket.OPEN) {
                this.processingConnection.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    handleUploadMessage(message) {
        switch (message.type) {
            case 'connected':
                console.log('🔗 Upload WebSocket connected:', message.upload_id);
                break;
            case 'upload_progress':
                this.updateUploadProgress(message.progress);
                this.updateInstantPreviewProgress(message.progress, `${Math.round(message.progress)}% uploaded`);
                break;
            case 'upload_complete':
                this.updateInstantPreviewProgress(100, 'Analyzing video...');
                setTimeout(() => {
                    this.updateInstantPreviewProgress(100, message.message);
                }, 1000);
                break;
            case 'upload_error':
                this.showError(message.error);
                this.hideInstantPreview();
                break;
            case 'keep_alive':
            case 'pong':
                // Connection is alive
                break;
        }
    }

    handleProcessingMessage(message) {
        switch (message.type) {
            case 'connected':
                console.log('🔗 Processing WebSocket connected:', message.task_id);
                break;
            case 'processing_started':
                this.updateProcessingStatus('🚀 AI processing started...');
                this.showLiveProcessingFeed(message);
                break;
            case 'progress_update':
                this.updateProcessingProgress(message.data);
                this.updateLiveProcessingFeed(message.data);
                break;
            case 'processing_complete':
                this.handleProcessingComplete(message.data);
                break;
            case 'processing_error':
                this.showError(message.data.error);
                break;
            case 'keep_alive':
            case 'pong':
                // Connection is alive
                break;
        }
    }

    showLiveProcessingFeed(message) {
        const container = document.getElementById('processing-status');
        if (!container) return;

        const existingFeed = container.querySelector('.live-updates');
        if (existingFeed) return;

        const feedHTML = `
            <div class="live-updates">
                <h4>🔴 Live Processing Feed</h4>
                <div class="updates-feed" id="updates-feed">
                    <div class="update-item">
                        <span class="update-time">${new Date().toLocaleTimeString()}</span>
                        <span class="update-message">AI processing started for ${message.total_clips} clips</span>
                    </div>
                </div>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', feedHTML);
    }

    updateLiveProcessingFeed(data) {
        const feed = document.getElementById('updates-feed');
        if (!feed) return;

        const updateHTML = `
            <div class="update-item">
                <span class="update-time">${new Date().toLocaleTimeString()}</span>
                <span class="update-message">${data.message}</span>
            </div>
        `;
        
        feed.insertAdjacentHTML('beforeend', updateHTML);
        
        // Keep only last 5 updates
        const updates = feed.querySelectorAll('.update-item');
        if (updates.length > 5) {
            updates[0].remove();
        }
        
        // Scroll to bottom
        feed.scrollTop = feed.scrollHeight;
    }

    showStep(step) {
        // Hide all steps
        document.querySelectorAll('.step').forEach(el => {
            el.style.display = 'none';
        });
        
        // Show target step
        const targetStep = document.getElementById(`step-${step}`);
        if (targetStep) {
            targetStep.style.display = 'block';
        }
        
        this.currentStep = step;
        
        // Update progress indicators
        this.updateStepProgress(step);
    }

    updateStepProgress(currentStep) {
        const steps = ['upload', 'analysis', 'processing', 'results'];
        const currentIndex = steps.indexOf(currentStep);
        
        steps.forEach((step, index) => {
            const indicator = document.querySelector(`[data-step="${step}"]`);
            if (indicator) {
                indicator.classList.toggle('active', index === currentIndex);
                indicator.classList.toggle('completed', index < currentIndex);
            }
        });
    }

    showUploadProgress() {
        const modal = document.getElementById('upload-progress-modal');
        if (modal) {
            modal.style.display = 'flex';
            modal.classList.add('show');
        }
    }

    hideUploadProgress() {
        const modal = document.getElementById('upload-progress-modal');
        if (modal) {
            modal.classList.remove('show');
            setTimeout(() => {
                modal.style.display = 'none';
            }, 300);
        }
    }

    updateUploadProgress(progress) {
        const progressBar = document.querySelector('#upload-progress-modal .progress-fill');
        const progressText = document.querySelector('#upload-progress-modal .progress-percentage');
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${progress}%`;
        }
    }

    showUploadComplete(data) {
        const modal = document.getElementById('upload-progress-modal');
        if (modal) {
            const content = modal.querySelector('.progress-content');
            if (content) {
                content.innerHTML = `
                    <div class="success-animation">
                        <div class="checkmark">✓</div>
                    </div>
                    <h3>Upload Complete!</h3>
                    <p>Your video has been uploaded successfully and is ready for analysis.</p>
                    <div class="upload-stats">
                        <span>Size: ${this.formatBytes(data.file_size)}</span>
                        <span>Duration: ${this.formatDuration(data.metadata?.duration || 0)}</span>
                    </div>
                `;
            }
        }
        
        setTimeout(() => {
            this.hideUploadProgress();
        }, 2000);
    }

    showAnalysisProgress() {
        // Implementation for URL analysis progress
        const button = document.querySelector('#url-form button');
        if (button) {
            button.disabled = true;
            button.classList.add('loading');
            button.innerHTML = `
                <div class="btn-loading">
                    <div class="spinner"></div>
                    <span>Analyzing...</span>
                </div>
            `;
        }
    }

    hideAnalysisProgress() {
        const button = document.querySelector('#url-form button');
        if (button) {
            button.disabled = false;
            button.classList.remove('loading');
            button.innerHTML = 'Analyze Video';
        }
    }

    showProcessingProgress() {
        const container = document.getElementById('processing-status');
        if (container) {
            container.innerHTML = `
                <div class="processing-header">
                    <h3>🎬 Creating Your Viral Clips</h3>
                    <p>Our AI is working hard to create amazing clips from your video...</p>
                </div>
                
                <div class="processing-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 0%"></div>
                    </div>
                    <div class="progress-text">
                        <span class="progress-percentage">0%</span>
                        <span class="progress-status">Initializing...</span>
                    </div>
                </div>
                
                <div class="processing-stats">
                    <div class="stat">
                        <span class="stat-label">Clips to Process</span>
                        <span class="stat-value">${this.selectedClips.length}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Estimated Time</span>
                        <span class="stat-value">${this.selectedClips.length * 30}s</span>
                    </div>
                </div>
            `;
        }
    }

    updateProcessingProgress(data) {
        const progressBar = document.querySelector('#processing-status .progress-fill');
        const progressPercentage = document.querySelector('#processing-status .progress-percentage');
        const progressStatus = document.querySelector('#processing-status .progress-status');
        
        if (progressBar) {
            progressBar.style.width = `${data.percentage}%`;
        }
        
        if (progressPercentage) {
            progressPercentage.textContent = `${Math.round(data.percentage)}%`;
        }
        
        if (progressStatus) {
            progressStatus.textContent = data.message || data.stage;
        }
    }

    handleProcessingComplete(data) {
        this.showStep('results');
        this.displayResults(data.results);
    }

    displayResults(results) {
        const container = document.getElementById('results-container');
        if (!container) return;

        container.innerHTML = `
            <div class="results-header">
                <h3>🎉 Your Viral Clips Are Ready!</h3>
                <p>Successfully processed ${results.length} viral clips</p>
            </div>
            
            <div class="results-grid">
                ${results.map((result, index) => this.renderResultCard(result, index)).join('')}
            </div>
        `;
    }

    renderResultCard(result, index) {
        return `
            <div class="result-card">
                <div class="result-header">
                    <h4>${result.title}</h4>
                    <div class="viral-score">
                        <span class="score">${result.viral_score}%</span>
                        <span class="label">Viral Potential</span>
                    </div>
                </div>
                
                <div class="result-thumbnail">
                    <img src="${result.thumbnail_path}" alt="${result.title}" 
                         onerror="this.src='/public/placeholder-thumb.jpg'">
                    <div class="play-overlay">▶️</div>
                </div>
                
                <div class="result-stats">
                    <span>⏱️ ${this.formatDuration(result.duration)}</span>
                    <span>📦 ${this.formatBytes(result.file_size)}</span>
                    <span>📊 ${result.performance_prediction?.estimated_views}</span>
                </div>
                
                <div class="result-actions">
                    <button class="btn btn-primary" onclick="app.downloadClip('${this.taskId}', ${index})">
                        📥 Download
                    </button>
                    <button class="btn btn-secondary" onclick="app.previewClip('${this.taskId}', ${index})">
                        👁️ Preview
                    </button>
                </div>
            </div>
        `;
    }

    async downloadClip(taskId, clipIndex) {
        try {
            const link = document.createElement('a');
            link.href = `/api/v2/download/${taskId}/${clipIndex}`;
            link.download = `viral_clip_${clipIndex + 1}.mp4`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            this.showError('Download failed. Please try again.');
        }
    }

    async previewClip(taskId, clipIndex) {
        // Implementation for video preview
        this.showError('Preview feature coming soon!');
    }

    handleAction(action, element) {
        switch (action) {
            case 'restart':
                this.restart();
                break;
            case 'back':
                this.goBack();
                break;
            case 'next':
                this.goNext();
                break;
        }
    }

    restart() {
        // Reset state
        this.currentStep = 'upload';
        this.sessionId = null;
        this.taskId = null;
        this.selectedClips = [];
        
        // Close WebSocket connections
        if (this.uploadConnection) {
            this.uploadConnection.close();
        }
        if (this.processingConnection) {
            this.processingConnection.close();
        }
        
        // Show upload step
        this.showStep('upload');
        
        // Reset forms
        document.querySelectorAll('form').forEach(form => form.reset());
    }

    goBack() {
        const steps = ['upload', 'analysis', 'processing', 'results'];
        const currentIndex = steps.indexOf(this.currentStep);
        if (currentIndex > 0) {
            this.showStep(steps[currentIndex - 1]);
        }
    }

    goNext() {
        const steps = ['upload', 'analysis', 'processing', 'results'];
        const currentIndex = steps.indexOf(this.currentStep);
        if (currentIndex < steps.length - 1) {
            this.showStep(steps[currentIndex + 1]);
        }
    }

    showError(message) {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = 'toast toast-error';
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">❌</span>
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(toast);
        
        // Show toast
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    showSuccess(message) {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = 'toast toast-success';
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">✅</span>
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        document.body.appendChild(toast);
        setTimeout(() => toast.classList.add('show'), 100);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    hideAllModals() {
        document.querySelectorAll('.modal, .overlay').forEach(modal => {
            modal.classList.remove('show');
            setTimeout(() => {
                modal.style.display = 'none';
            }, 300);
        });
    }

    reconnectWebSockets() {
        if (this.uploadConnection?.readyState === WebSocket.CLOSED) {
            this.connectUploadWebSocket();
        }
        if (this.processingConnection?.readyState === WebSocket.CLOSED) {
            this.connectProcessingWebSocket();
        }
    }

    startPerformanceMonitoring() {
        // Monitor page load performance
        window.addEventListener('load', () => {
            const navigationTiming = performance.getEntriesByType('navigation')[0];
            console.log('📊 Page Load Time:', navigationTiming.loadEventEnd - navigationTiming.loadEventStart, 'ms');
        });
    }

    // Utility functions
    generateId() {
        return 'id_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
    }

    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ViralClipApp();
});

// Service Worker registration for PWA capabilities
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('📱 SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('📱 SW registration failed: ', registrationError);
            });
    });
}
