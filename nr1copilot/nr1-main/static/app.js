/**
 * ViralClip Pro - Netflix-Level Frontend Application
 * Mobile-first responsive design with real-time WebSocket updates
 * 
 * Features:
 * - Advanced error handling and recovery
 * - Performance monitoring and optimization
 * - Comprehensive WebSocket management
 * - Mobile-first responsive design
 * - Real-time progress tracking
 * - One-click upload with instant preview
 */

class ViralClipPro {
    constructor() {
        // Core application state
        this.state = {
            currentStep: 'upload',
            sessionId: null,
            taskId: null,
            uploadId: null,
            selectedClips: [],
            currentFile: null,
            isProcessing: false
        };

        // WebSocket connections
        this.connections = {
            upload: null,
            processing: null
        };

        // UI state
        this.ui = {
            isDragging: false,
            isUploading: false,
            uploadProgress: 0,
            processingProgress: 0
        };

        // Performance metrics
        this.metrics = {
            startTime: Date.now(),
            uploadStartTime: null,
            processingStartTime: null,
            lastActivity: Date.now()
        };

        // Configuration
        this.config = {
            maxFileSize: 2 * 1024 * 1024 * 1024, // 2GB
            allowedTypes: ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'],
            websocketReconnectDelay: 5000,
            heartbeatInterval: 30000,
            maxRetries: 3
        };

        // Error tracking
        this.errors = {
            count: 0,
            lastError: null,
            retryAttempts: {}
        };

        this.init();
    }

    async init() {
        try {
            console.log('🚀 Initializing ViralClip Pro v2.0...');

            this.checkBrowserCompatibility();
            this.setupEventListeners();
            this.setupDragAndDrop();
            this.setupMobileOptimizations();
            this.setupErrorHandling();
            this.setupPerformanceMonitoring();

            this.startActivityTracking();
            this.showStep('upload');
            this.hideLoadingSplash();

            console.log('✅ ViralClip Pro initialized successfully');
            this.trackEvent('app_initialized');

        } catch (error) {
            console.error('❌ Failed to initialize ViralClip Pro:', error);
            this.showCriticalError('Failed to initialize application', error);
        }
    }

    checkBrowserCompatibility() {
        const requiredFeatures = ['WebSocket', 'FileReader', 'FormData', 'fetch', 'Promise'];
        const missingFeatures = requiredFeatures.filter(feature => !(feature in window));

        if (missingFeatures.length > 0) {
            throw new Error(`Browser missing required features: ${missingFeatures.join(', ')}`);
        }

        // Check for recommended features
        const recommendedFeatures = ['serviceWorker', 'localStorage', 'sessionStorage'];
        const missingRecommended = recommendedFeatures.filter(feature => !(feature in window));

        if (missingRecommended.length > 0) {
            console.warn('⚠️ Browser missing recommended features:', missingRecommended);
        }
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

        // Visibility change for WebSocket reconnection
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                this.reconnectWebSockets();
            }
        });

        // Tab switching
        this.setupTabSwitching();
    }

    setupTabSwitching() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;

                // Remove active class from all tabs and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked tab and corresponding content
                button.classList.add('active');
                const targetContent = document.getElementById(targetTab);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
            });
        });
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('upload-area');
        if (!uploadArea) return;

        // Enhanced click to upload
        uploadArea.addEventListener('click', (e) => {
            if (!e.target.closest('.instant-preview')) {
                document.getElementById('file-input')?.click();
            }
        });

        // Mobile touch support with haptic feedback
        uploadArea.addEventListener('touchstart', (e) => {
            uploadArea.classList.add('touch-active');
            if (navigator.vibrate) {
                navigator.vibrate(50);
            }
        }, { passive: true });

        uploadArea.addEventListener('touchend', (e) => {
            uploadArea.classList.remove('touch-active');
            if (e.touches.length === 0) {
                uploadArea.style.transform = 'scale(0.98)';
                setTimeout(() => {
                    uploadArea.style.transform = '';
                    document.getElementById('file-input')?.click();
                }, 100);
            }
        }, { passive: true });

        // Prevent defaults
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, this.preventDefaults, false);
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        // Drag counter for better state management
        this.dragCounter = 0;

        // Global drag handlers
        document.addEventListener('dragenter', (e) => {
            this.dragCounter++;
            if (this.dragCounter === 1) {
                this.showDragOverlay();
            }
        });

        document.addEventListener('dragleave', (e) => {
            this.dragCounter--;
            if (this.dragCounter === 0) {
                this.hideDragOverlay();
            }
        });

        // Upload area specific handlers
        uploadArea.addEventListener('dragover', (e) => {
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            this.dragCounter = 0;
            this.hideDragOverlay();
            uploadArea.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.processFiles(files);
            }
        });

        // Paste support
        document.addEventListener('paste', this.handlePaste.bind(this));
    }

    setupMobileOptimizations() {
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
            const now = Date.now();
            if (now - lastTouchEnd <= 300) {
                event.preventDefault();
            }
            lastTouchEnd = now;
        }, false);

        // Optimize touch events
        document.addEventListener('touchstart', () => {}, { passive: true });
        document.addEventListener('touchmove', () => {}, { passive: true });
    }

    setupErrorHandling() {
        // Global error handler
        window.addEventListener('error', (event) => {
            this.handleGlobalError('JavaScript Error', event.error);
        });

        // Unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            this.handleGlobalError('Unhandled Promise Rejection', event.reason);
            event.preventDefault();
        });

        // Network error detection
        window.addEventListener('online', () => {
            this.handleNetworkChange(true);
        });

        window.addEventListener('offline', () => {
            this.handleNetworkChange(false);
        });
    }

    setupPerformanceMonitoring() {
        // Monitor page load performance
        window.addEventListener('load', () => {
            if ('performance' in window) {
                const navigation = performance.getEntriesByType('navigation')[0];
                console.log('📊 Page Load Time:', navigation.loadEventEnd - navigation.loadEventStart, 'ms');
            }
        });
    }

    startActivityTracking() {
        ['click', 'touchstart', 'keydown', 'scroll'].forEach(event => {
            document.addEventListener(event, () => {
                this.metrics.lastActivity = Date.now();
            }, { passive: true });
        });
    }

    hideLoadingSplash() {
        const splash = document.querySelector('.loading-splash');
        if (splash) {
            splash.style.opacity = '0';
            setTimeout(() => {
                splash.style.display = 'none';
            }, 500);
        }
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
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

    async handleFileUpload(e) {
        e.preventDefault();
        if (this.state.currentFile) {
            await this.uploadFile(this.state.currentFile);
        } else {
            this.showError('Please select a file to upload.');
        }
    }

    async processFiles(files) {
        const file = files[0];

        if (!this.validateFile(file)) {
            return;
        }

        // Show instant preview
        await this.showInstantPreview(file);
        this.state.currentFile = file;

        // Enable upload button
        const uploadBtn = document.querySelector('#upload-form button[type="submit"]');
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.textContent = '🚀 Upload & Create Clips';
            uploadBtn.classList.add('has-file');
        }
    }

    validateFile(file) {
        if (file.size > this.config.maxFileSize) {
            this.showError('File too large. Maximum size is 2GB.');
            return false;
        }

        const extension = '.' + file.name.split('.').pop().toLowerCase();
        if (!this.config.allowedTypes.includes(extension)) {
            this.showError('Invalid file type. Please upload a video file.');
            return false;
        }

        return true;
    }

    async showInstantPreview(file) {
        const uploadArea = document.getElementById('upload-area');

        // Remove existing preview
        const existingPreview = uploadArea.querySelector('.instant-preview');
        if (existingPreview) {
            existingPreview.remove();
        }

        const preview = document.createElement('div');
        preview.className = 'instant-preview';

        // Create video element
        const video = document.createElement('video');
        video.className = 'preview-video';
        video.autoplay = true;
        video.muted = true;
        video.loop = true;
        video.controls = false;
        video.playsInline = true;
        video.preload = 'metadata';

        const fileURL = URL.createObjectURL(file);
        video.src = fileURL;

        preview.innerHTML = `
            <div class="preview-video-container">
                <div class="video-overlay">
                    <div class="play-indicator">▶</div>
                </div>
            </div>
            <div class="preview-overlay">
                <div class="preview-info">
                    <div class="file-details">
                        <div class="file-name">${this.truncateFileName(file.name, 25)}</div>
                        <div class="file-stats">
                            <span class="file-status" data-status="ready">Ready to upload</span>
                            <span>${this.formatBytes(file.size)}</span>
                            <span>${this.getFileTypeDisplay(file.type)}</span>
                        </div>
                    </div>
                    <div class="preview-actions">
                        <button class="btn-mini btn-remove" onclick="app.removePreview()" title="Remove">×</button>
                    </div>
                </div>
                <div class="upload-progress-mini">
                    <div class="progress-bar-mini">
                        <div class="progress-fill-mini" style="width: 0%"></div>
                    </div>
                    <div class="progress-stats">
                        <span class="progress-current">0%</span>
                        <span class="progress-eta">Ready</span>
                    </div>
                </div>
            </div>
        `;

        // Insert video
        const videoContainer = preview.querySelector('.preview-video-container');
        videoContainer.insertBefore(video, videoContainer.firstChild);

        uploadArea.appendChild(preview);

        // Video event handlers
        video.addEventListener('loadedmetadata', () => {
            video.classList.add('loaded');
            const overlay = preview.querySelector('.video-overlay');
            if (overlay) overlay.style.opacity = '0';

            // Update duration info
            if (video.duration > 0) {
                const durationSpan = document.createElement('span');
                durationSpan.textContent = this.formatDuration(video.duration);
                preview.querySelector('.file-stats').appendChild(durationSpan);
            }
        });

        video.addEventListener('error', () => {
            console.warn('Video preview failed, showing placeholder');
            video.style.display = 'none';
            const placeholder = document.createElement('div');
            placeholder.className = 'video-placeholder';
            placeholder.innerHTML = `
                <div class="placeholder-icon">🎬</div>
                <div class="placeholder-text">Video Preview</div>
            `;
            videoContainer.appendChild(placeholder);
        });

        // Cleanup URL when preview is removed
        preview.addEventListener('remove', () => {
            URL.revokeObjectURL(fileURL);
        });

        // Show preview with animation
        setTimeout(() => {
            preview.classList.add('show');
        }, 100);
    }

    async uploadFile(file) {
        try {
            this.state.uploadId = this.generateId();
            this.ui.isUploading = true;

            // Connect WebSocket for real-time progress
            await this.connectUploadWebSocket();

            // Update preview to show upload starting
            this.updateInstantPreviewProgress(0, 'Starting upload...');

            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('upload_id', this.state.uploadId);

            // Upload with progress tracking
            const response = await this.uploadWithProgress(formData);

            if (response.success) {
                this.state.sessionId = response.session_id;

                // Update preview to show analysis complete
                this.updateInstantPreviewProgress(100, 'Analysis complete!');

                // Show success and transition
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
        } finally {
            this.ui.isUploading = false;
        }
    }

    async uploadWithProgress(formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            let startTime = Date.now();
            let lastLoaded = 0;
            let lastTime = startTime;

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const progress = Math.round((e.loaded / e.total) * 100);
                    const currentTime = Date.now();
                    const deltaTime = (currentTime - lastTime) / 1000;
                    const deltaLoaded = e.loaded - lastLoaded;

                    let speed = 0;
                    let eta = 0;

                    if (deltaTime > 0) {
                        speed = deltaLoaded / deltaTime;
                        const remaining = e.total - e.loaded;
                        eta = speed > 0 ? remaining / speed : 0;
                    }

                    this.updateInstantPreviewProgress(
                        progress,
                        `Uploading... ${progress}%`,
                        speed,
                        eta
                    );

                    lastLoaded = e.loaded;
                    lastTime = currentTime;
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        this.updateInstantPreviewProgress(100, 'Upload complete! Analyzing...');
                        resolve(response);
                    } catch (error) {
                        this.updateInstantPreviewProgress(0, 'Upload failed');
                        reject(new Error('Invalid response format'));
                    }
                } else {
                    this.updateInstantPreviewProgress(0, 'Upload failed');
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                }
            });

            xhr.addEventListener('error', () => {
                this.updateInstantPreviewProgress(0, 'Upload failed');
                reject(new Error('Network error during upload'));
            });

            xhr.addEventListener('timeout', () => {
                this.updateInstantPreviewProgress(0, 'Upload timeout');
                reject(new Error('Upload timeout'));
            });

            xhr.timeout = 5 * 60 * 1000; // 5 minutes
            xhr.open('POST', '/api/v2/upload-video');
            xhr.send(formData);
        });
    }

    updateInstantPreviewProgress(progress, message, speed = null, eta = null) {
        const preview = document.querySelector('.instant-preview');
        if (!preview) return;

        const progressFill = preview.querySelector('.progress-fill-mini');
        const progressCurrent = preview.querySelector('.progress-current');
        const progressEta = preview.querySelector('.progress-eta');
        const fileStatus = preview.querySelector('.file-status');

        // Update progress bar
        if (progressFill) {
            progressFill.style.width = `${Math.min(progress, 100)}%`;

            // Dynamic color coding
            if (progress < 30) {
                progressFill.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
            } else if (progress < 70) {
                progressFill.style.background = 'linear-gradient(90deg, #3b82f6, #60a5fa)';
            } else {
                progressFill.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
            }

            if (progress > 0 && progress < 100) {
                preview.classList.add('uploading');
            }
        }

        // Update progress text
        if (progressCurrent) {
            progressCurrent.textContent = `${Math.round(progress)}%`;
        }

        // Update status message
        if (fileStatus && message) {
            fileStatus.textContent = message;
            fileStatus.setAttribute('data-status', 
                progress === 0 ? 'ready' : 
                progress === 100 ? 'complete' : 'uploading'
            );
        }

        // Update ETA or speed
        if (progressEta) {
            if (eta && eta > 0) {
                const etaSeconds = Math.round(eta);
                progressEta.textContent = etaSeconds > 60 ? 
                    `${Math.round(etaSeconds / 60)}m remaining` : 
                    `${etaSeconds}s remaining`;
            } else if (speed && speed > 0) {
                progressEta.textContent = this.formatSpeed(speed);
            } else if (progress === 100) {
                progressEta.textContent = 'Complete!';
            } else {
                progressEta.textContent = 'Uploading...';
            }
        }

        // Completion effects
        if (progress >= 100) {
            preview.classList.remove('uploading');
            preview.classList.add('upload-complete');

            const celebration = document.createElement('div');
            celebration.className = 'celebration-effect';
            celebration.textContent = '🎉';
            preview.appendChild(celebration);

            setTimeout(() => {
                celebration.remove();
            }, 2000);
        }
    }

    hideInstantPreview() {
        const preview = document.querySelector('.instant-preview');
        if (preview) {
            preview.classList.remove('show');
            setTimeout(() => {
                const event = new Event('remove');
                preview.dispatchEvent(event);
                preview.remove();
            }, 300);
        }

        // Reset upload button
        const uploadBtn = document.querySelector('#upload-form button[type="submit"]');
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Select a video to upload';
            uploadBtn.classList.remove('has-file');
        }

        this.state.currentFile = null;
    }

    removePreview() {
        this.hideInstantPreview();

        // Clear file input
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.value = '';
        }

        this.showSuccess('Video removed. You can select another file.');
    }

    async connectUploadWebSocket() {
        if (this.connections.upload) {
            this.connections.upload.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v2/upload-progress/${this.state.uploadId}`;

        this.connections.upload = new WebSocket(wsUrl);

        this.connections.upload.onopen = () => {
            console.log('📡 Upload WebSocket connected');
        };

        this.connections.upload.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleUploadMessage(message);
            } catch (error) {
                console.error('Upload WebSocket message error:', error);
            }
        };

        this.connections.upload.onerror = (error) => {
            console.error('Upload WebSocket error:', error);
        };

        this.connections.upload.onclose = () => {
            console.log('📡 Upload WebSocket disconnected');
        };
    }

    handleUploadMessage(message) {
        switch (message.type) {
            case 'connected':
                console.log('🔗 Upload WebSocket connected:', message.upload_id);
                break;
            case 'upload_progress':
                this.updateInstantPreviewProgress(message.progress, `${Math.round(message.progress)}% uploaded`);
                break;
            case 'upload_complete':
                this.updateInstantPreviewProgress(100, message.message);
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
                    processButton.disabled = this.state.selectedClips.length === 0;
                }
            });
        });
    }

    updateSelectedClips() {
        const checkboxes = document.querySelectorAll('.clip-checkbox input[type="checkbox"]:checked');
        this.state.selectedClips = Array.from(checkboxes).map(cb => parseInt(cb.value));
    }

    async handleProcessing(e) {
        e.preventDefault();

        if (this.state.selectedClips.length === 0) {
            this.showError('Please select at least one clip to process.');
            return;
        }

        try {
            this.state.taskId = this.generateId();

            // Connect to processing WebSocket
            await this.connectProcessingWebSocket();

            // Show processing step
            this.showStep('processing');

            // Start processing
            const response = await fetch('/api/v2/process-video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.state.sessionId,
                    clips: this.state.selectedClips.map(index => ({
                        index,
                        title: `Clip ${index + 1}`,
                        start_time: index * 60,
                        end_time: (index + 1) * 60
                    })),
                    quality: 'high',
                    priority: 'high'
                })
            });

            const data = await response.json();

            if (data.success) {
                this.state.taskId = data.task_id;
                this.showProcessingProgress();
            } else {
                throw new Error(data.error || 'Processing failed to start');
            }

        } catch (error) {
            console.error('Processing error:', error);
            this.showError(`Processing failed: ${error.message}`);
        }
    }

    async connectProcessingWebSocket() {
        if (this.connections.processing) {
            this.connections.processing.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v2/ws/${this.state.taskId}`;

        this.connections.processing = new WebSocket(wsUrl);

        this.connections.processing.onopen = () => {
            console.log('📡 Processing WebSocket connected');
        };

        this.connections.processing.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleProcessingMessage(message);
            } catch (error) {
                console.error('Processing WebSocket message error:', error);
            }
        };

        this.connections.processing.onerror = (error) => {
            console.error('Processing WebSocket error:', error);
        };

        this.connections.processing.onclose = () => {
            console.log('📡 Processing WebSocket disconnected');
        };
    }

    handleProcessingMessage(message) {
        switch (message.type) {
            case 'connected':
                console.log('🔗 Processing WebSocket connected:', message.task_id);
                break;
            case 'processing_started':
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
                        <span class="stat-value">${this.state.selectedClips.length}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Estimated Time</span>
                        <span class="stat-value">${this.state.selectedClips.length * 30}s</span>
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

        feed.scrollTop = feed.scrollHeight;
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
                    <h4>${result.clip_title || `Clip ${index + 1}`}</h4>
                    <div class="viral-score">
                        <span class="score">${result.viral_score || 85}%</span>
                        <span class="label">Viral Potential</span>
                    </div>
                </div>

                <div class="result-thumbnail">
                    <img src="${result.thumbnail_path || '/public/placeholder-thumb.jpg'}" 
                         alt="${result.clip_title}" 
                         onerror="this.src='/public/placeholder-thumb.jpg'">
                    <div class="play-overlay">▶️</div>
                </div>

                <div class="result-stats">
                    <span>⏱️ ${this.formatDuration(result.duration || 30)}</span>
                    <span>📦 ${this.formatBytes(result.file_size || 5000000)}</span>
                    <span>📊 ${result.estimated_views || '500K+'}</span>
                </div>

                <div class="result-actions">
                    <button class="btn btn-primary" onclick="app.downloadClip('${this.state.taskId}', ${index})">
                        📥 Download
                    </button>
                    <button class="btn btn-secondary" onclick="app.previewClip('${this.state.taskId}', ${index})">
                        👁️ Preview
                    </button>
                </div>
            </div>
        `;
    }

    showStep(step) {
        // Hide all steps
        document.querySelectorAll('.step').forEach(el => {
            el.classList.remove('active');
        });

        // Show target step
        const targetStep = document.getElementById(`step-${step}`);
        if (targetStep) {
            targetStep.classList.add('active');
        }

        this.state.currentStep = step;
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
        this.state = {
            currentStep: 'upload',
            sessionId: null,
            taskId: null,
            uploadId: null,
            selectedClips: [],
            currentFile: null,
            isProcessing: false
        };

        // Close WebSocket connections
        Object.values(this.connections).forEach(connection => {
            if (connection) connection.close();
        });

        this.connections = { upload: null, processing: null };

        // Show upload step
        this.showStep('upload');

        // Reset forms
        document.querySelectorAll('form').forEach(form => form.reset());

        // Clear file input
        const fileInput = document.getElementById('file-input');
        if (fileInput) fileInput.value = '';

        // Reset upload button
        const uploadBtn = document.querySelector('#upload-form button[type="submit"]');
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Select a video to upload';
            uploadBtn.classList.remove('has-file');
        }
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

    showError(message) {
        this.showToast(message, 'error');
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${type === 'error' ? '❌' : type === 'success' ? '✅' : 'ℹ️'}</span>
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        document.body.appendChild(toast);
        setTimeout(() => toast.classList.add('show'), 100);

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, type === 'error' ? 5000 : 3000);
    }

    reconnectWebSockets() {
        if (this.connections.upload?.readyState === WebSocket.CLOSED) {
            this.connectUploadWebSocket();
        }
        if (this.connections.processing?.readyState === WebSocket.CLOSED) {
            this.connectProcessingWebSocket();
        }
    }

    handleGlobalError(type, error) {
        this.errors.count++;
        this.errors.lastError = { type, error, timestamp: Date.now() };

        console.error(`🚨 Global Error [${type}]:`, error);

        if (this.errors.count <= 3) {
            this.showError('Something went wrong. Please try again.');
        }
    }

    handleNetworkChange(isOnline) {
        if (isOnline) {
            this.hideOfflineIndicator();
            this.reconnectWebSockets();
            this.showSuccess('Connection restored');
        } else {
            this.showOfflineIndicator();
            this.showError('Connection lost. Working offline.');
        }
    }

    showOfflineIndicator() {
        let indicator = document.getElementById('offline-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'offline-indicator';
            indicator.className = 'offline-indicator';
            indicator.textContent = '📶 Offline - Limited functionality';
            document.body.appendChild(indicator);
        }
        indicator.classList.add('show');
    }

    hideOfflineIndicator() {
        const indicator = document.getElementById('offline-indicator');
        if (indicator) {
            indicator.classList.remove('show');
            setTimeout(() => indicator.remove(), 300);
        }
    }

    trackEvent(eventName, data = {}) {
        if (this.config.enableAnalytics) {
            console.log(`📊 Event: ${eventName}`, data);
        }
    }

    hideAllModals() {
        document.querySelectorAll('.modal, .overlay').forEach(modal => {
            modal.classList.remove('show');
            setTimeout(() => {
                modal.style.display = 'none';
            }, 300);
        });
    }

    // Utility functions
    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatTime(seconds) {
        return this.formatDuration(seconds);
    }

    formatSpeed(bytesPerSecond) {
        if (bytesPerSecond < 1024) return `${Math.round(bytesPerSecond)} B/s`;
        if (bytesPerSecond < 1024 * 1024) return `${(bytesPerSecond / 1024).toFixed(1)} KB/s`;
        return `${(bytesPerSecond / (1024 * 1024)).toFixed(1)} MB/s`;
    }

    truncateFileName(filename, maxLength) {
        if (filename.length <= maxLength) return filename;

        const extension = filename.split('.').pop();
        const nameWithoutExt = filename.substring(0, filename.lastIndexOf('.'));
        const truncatedName = nameWithoutExt.substring(0, maxLength - extension.length - 4) + '...';

        return `${truncatedName}.${extension}`;
    }

    getFileTypeDisplay(mimeType) {
        const typeMap = {
            'video/mp4': 'MP4',
            'video/mov': 'MOV',
            'video/avi': 'AVI',
            'video/mkv': 'MKV',
            'video/webm': 'WebM',
            'video/m4v': 'M4V'
        };
        return typeMap[mimeType] || 'Video';
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
                this.state.sessionId = data.session_id;
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

    showAnalysisProgress() {
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

    showCriticalError(message, error) {
        const errorModal = document.createElement('div');
        errorModal.className = 'critical-error-modal';
        errorModal.innerHTML = `
            <div class="error-content">
                <h3>🚨 Critical Error</h3>
                <p>${message}</p>
                <details>
                    <summary>Technical Details</summary>
                    <pre>${error.stack || error.message || error}</pre>
                </details>
                <div class="error-actions">
                    <button onclick="location.reload()" class="btn btn-primary">
                        Reload Page
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(errorModal);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ViralClipPro();
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