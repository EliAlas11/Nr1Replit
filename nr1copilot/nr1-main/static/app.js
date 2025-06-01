/**
 * ViralClip Pro - Advanced Frontend Application
 * Netflix-level UI/UX with real-time features
 */

class ViralClipApp {
    constructor() {
        this.currentSession = null;
        this.currentTask = null;
        this.websocket = null;
        this.uploadWebSocket = null;
        this.analysisData = null;
        this.processingData = [];
        this.isProcessing = false;
        this.retryAttempts = 0;
        this.maxRetries = 3;

        // Enhanced state management
        this.state = {
            currentStep: 'upload', // upload, analyze, process, results
            isLoading: false,
            error: null,
            progress: 0,
            connectionStatus: 'connected'
        };

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupServiceWorker();
        this.setupConnectionMonitoring();
        this.setupKeyboardShortcuts();
        this.updateUIState();
        this.preloadAssets();

        console.log('🚀 ViralClip Pro initialized');
    }

    setupEventListeners() {
        // URL Analysis
        const analyzeBtn = document.getElementById('analyze-btn');
        const urlInput = document.getElementById('video-url');

        if (analyzeBtn && urlInput) {
            analyzeBtn.addEventListener('click', () => this.handleAnalyze());
            urlInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleAnalyze();
            });
            urlInput.addEventListener('input', this.debounce((e) => {
                this.validateURL(e.target.value);
            }, 300));
        }

        // File Upload
        this.setupFileUpload();

        // Processing controls
        document.addEventListener('click', (e) => {
            if (e.target.matches('.process-btn')) {
                this.handleProcess();
            }
            if (e.target.matches('.download-btn')) {
                this.handleDownload(e.target.dataset.clipIndex);
            }
            if (e.target.matches('.regenerate-btn')) {
                this.handleRegenerate();
            }
            if (e.target.matches('.share-btn')) {
                this.handleShare(e.target.dataset.clipIndex);
            }
        });

        // Step navigation
        document.addEventListener('click', (e) => {
            if (e.target.matches('.step-btn')) {
                this.navigateToStep(e.target.dataset.step);
            }
        });

        // Global error handling
        window.addEventListener('error', (e) => {
            this.handleGlobalError(e);
        });

        window.addEventListener('unhandledrejection', (e) => {
            this.handleGlobalError(e);
        });
    }

    setupFileUpload() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const uploadBtn = document.getElementById('upload-btn');

        if (!uploadArea || !fileInput) return;

        // Enhanced drag and drop with global drop zone
        this.setupGlobalDragDrop();
        this.setupLocalDragDrop(uploadArea);

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // Upload button click
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => fileInput.click());
        }

        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());

        // Paste support
        document.addEventListener('paste', (e) => {
            const items = Array.from(e.clipboardData.items);
            const videoItem = items.find(item => item.type.startsWith('video/'));
            
            if (videoItem) {
                const file = videoItem.getAsFile();
                if (file) {
                    this.handleFileUpload(file);
                }
            }
        });
    }

    setupGlobalDragDrop() {
        let dragCounter = 0;
        const dragOverlay = this.createDragOverlay();

        document.addEventListener('dragenter', (e) => {
            e.preventDefault();
            dragCounter++;
            
            if (this.isDraggedFile(e)) {
                document.body.appendChild(dragOverlay);
                dragOverlay.classList.add('show');
            }
        });

        document.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dragCounter--;
            
            if (dragCounter === 0) {
                dragOverlay.classList.remove('show');
                setTimeout(() => {
                    if (dragOverlay.parentNode) {
                        dragOverlay.parentNode.removeChild(dragOverlay);
                    }
                }, 200);
            }
        });

        document.addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        document.addEventListener('drop', (e) => {
            e.preventDefault();
            dragCounter = 0;
            
            dragOverlay.classList.remove('show');
            setTimeout(() => {
                if (dragOverlay.parentNode) {
                    dragOverlay.parentNode.removeChild(dragOverlay);
                }
            }, 200);

            if (this.isDraggedFile(e)) {
                const files = Array.from(e.dataTransfer.files);
                const videoFile = files.find(file => file.type.startsWith('video/'));

                if (videoFile) {
                    this.handleFileUpload(videoFile);
                } else {
                    this.showNotification('Please upload a video file', 'error');
                }
            }
        });
    }

    setupLocalDragDrop(uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.remove('drag-over');

            const files = Array.from(e.dataTransfer.files);
            const videoFile = files.find(file => file.type.startsWith('video/'));

            if (videoFile) {
                this.handleFileUpload(videoFile);
            } else {
                this.showNotification('Please upload a video file', 'error');
            }
        });
    }

    createDragOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'drag-overlay';
        overlay.innerHTML = `
            <div class="drag-content">
                <div class="drag-icon">📁</div>
                <h3>Drop your video here</h3>
                <p>Release to upload and start creating viral clips</p>
            </div>
        `;
        return overlay;
    }

    isDraggedFile(e) {
        return e.dataTransfer && 
               e.dataTransfer.types && 
               Array.from(e.dataTransfer.types).includes('Files');
    }

    setupServiceWorker() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js')
                .then(registration => {
                    console.log('✅ Service Worker registered');
                })
                .catch(error => {
                    console.log('❌ SW registration failed');
                });
        }
    }

    setupConnectionMonitoring() {
        this.connectionMonitor = new ConnectionMonitor(this);
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + U for upload
            if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
                e.preventDefault();
                document.getElementById('file-input')?.click();
            }

            // Ctrl/Cmd + Enter for process
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this.handleProcess();
            }

            // Escape to cancel/close
            if (e.key === 'Escape') {
                this.handleCancel();
            }
        });
    }

    preloadAssets() {
        // Preload critical images and icons
        const preloadImages = [
            '/public/icons/upload.svg',
            '/public/icons/ai.svg',
            '/public/icons/success.svg'
        ];

        preloadImages.forEach(src => {
            const img = new Image();
            img.src = src;
        });
    }

    // URL Analysis
    async handleAnalyze() {
        const urlInput = document.getElementById('video-url');
        const url = urlInput?.value?.trim();

        if (!url) {
            this.showNotification('Please enter a video URL', 'error');
            return;
        }

        if (!this.validateURL(url)) {
            this.showNotification('Please enter a valid video URL', 'error');
            return;
        }

        this.setState({ isLoading: true, currentStep: 'analyze' });
        this.updateProgress(0, 'Analyzing video...');

        try {
            const response = await this.apiRequest('/api/v2/analyze-video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    url: url,
                    clip_duration: parseInt(document.getElementById('clip-duration')?.value || 60),
                    viral_optimization: true,
                    suggested_formats: ['tiktok', 'instagram', 'youtube_shorts']
                })
            });

            if (response.success) {
                this.analysisData = response;
                this.currentSession = response.session_id;
                this.displayAnalysisResults(response);
                this.setState({ currentStep: 'results' });
                this.showNotification('Analysis complete!', 'success');
            } else {
                throw new Error(response.error || 'Analysis failed');
            }
        } catch (error) {
            this.handleError(error, 'Analysis failed');
        } finally {
            this.setState({ isLoading: false });
        }
    }

    // Enhanced File Upload with instant preview
    async handleFileUpload(file) {
        if (!this.validateFile(file)) return;

        // Show instant preview first
        this.showInstantPreview(file);
        
        this.setState({ isLoading: true, currentStep: 'upload' });

        const uploadId = this.generateId();
        this.setupUploadWebSocket(uploadId);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('upload_id', uploadId);

        try {
            this.updateProgress(0, 'Starting upload...');

            const response = await this.uploadWithProgress(formData, (progress) => {
                this.updateProgress(progress, `Uploading... ${Math.round(progress)}%`);
                this.updateInstantPreview(progress);
            });

            if (response.success) {
                this.currentSession = response.session_id;
                this.updateProgress(100, 'Upload complete!');
                this.displayUploadResults(response);
                this.setState({ currentStep: 'process' });
                this.showNotification('Upload successful!', 'success');
                this.hideInstantPreview();
            } else {
                throw new Error(response.error || 'Upload failed');
            }
        } catch (error) {
            this.handleError(error, 'Upload failed');
            this.hideInstantPreview();
        } finally {
            this.setState({ isLoading: false });
        }
    }

    showInstantPreview(file) {
        const uploadArea = document.getElementById('upload-area');
        if (!uploadArea) return;

        // Create preview element
        const preview = document.createElement('div');
        preview.className = 'instant-preview';
        preview.id = 'instant-preview';

        // Create video preview if possible
        const videoURL = URL.createObjectURL(file);
        
        preview.innerHTML = `
            <video class="preview-video" src="${videoURL}" muted controls></video>
            <div class="preview-overlay">
                <div class="preview-info">
                    <div class="file-details">
                        <h4>${file.name}</h4>
                        <div class="file-stats">
                            <span>📁 ${this.formatFileSize(file.size)}</span>
                            <span>🎬 ${file.type.split('/')[1].toUpperCase()}</span>
                            <span>⏱️ Analyzing...</span>
                        </div>
                    </div>
                    <div class="preview-actions">
                        <button class="btn btn-primary btn-small">
                            ⚡ Processing...
                        </button>
                        <button class="btn btn-ghost btn-small" onclick="viralClipApp.cancelUpload()">
                            ❌ Cancel
                        </button>
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
        }, 50);

        // Get video duration if possible
        const video = preview.querySelector('video');
        video.addEventListener('loadedmetadata', () => {
            const duration = this.formatDuration(video.duration);
            const durationSpan = preview.querySelector('.file-stats span:last-child');
            if (durationSpan) {
                durationSpan.textContent = `⏱️ ${duration}`;
            }
        });
    }

    updateInstantPreview(progress) {
        const preview = document.getElementById('instant-preview');
        if (!preview) return;

        const progressFill = preview.querySelector('.progress-fill-mini');
        const processingBtn = preview.querySelector('.btn-primary');

        if (progressFill) {
            progressFill.style.width = `${progress}%`;
        }

        if (processingBtn) {
            if (progress >= 100) {
                processingBtn.innerHTML = '✅ Complete!';
                processingBtn.classList.add('success');
            } else {
                processingBtn.innerHTML = `⚡ ${Math.round(progress)}%`;
            }
        }
    }

    hideInstantPreview() {
        const preview = document.getElementById('instant-preview');
        if (preview) {
            preview.classList.remove('show');
            setTimeout(() => {
                if (preview.parentNode) {
                    preview.parentNode.removeChild(preview);
                }
            }, 300);
        }
    }

    cancelUpload() {
        this.hideInstantPreview();
        
        if (this.uploadWebSocket) {
            this.uploadWebSocket.close();
        }
        
        this.setState({ isLoading: false });
        this.updateProgress(0, 'Upload cancelled');
        this.showNotification('Upload cancelled', 'info');
    }

    // Processing
    async handleProcess() {
        if (!this.currentSession) {
            this.showNotification('No video session found', 'error');
            return;
        }

        this.setState({ isLoading: true });
        this.isProcessing = true;

        const clips = this.getSelectedClips();
        if (clips.length === 0) {
            this.showNotification('Please select at least one clip to process', 'error');
            this.setState({ isLoading: false });
            return;
        }

        try {
            const formData = new FormData();
            formData.append('session_id', this.currentSession);
            formData.append('clips', JSON.stringify(clips));
            formData.append('priority', 'high');

            const response = await this.apiRequest('/api/v2/process-video', {
                method: 'POST',
                body: formData
            });

            if (response.success) {
                this.currentTask = response.task_id;
                this.setupProcessingWebSocket(response.task_id);
                this.updateProgress(0, 'Processing started...');
                this.showNotification('Processing started!', 'info');
            } else {
                throw new Error(response.error || 'Processing failed to start');
            }
        } catch (error) {
            this.handleError(error, 'Failed to start processing');
            this.isProcessing = false;
            this.setState({ isLoading: false });
        }
    }

    // WebSocket Management
    setupUploadWebSocket(uploadId) {
        const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/v2/upload-progress/${uploadId}`;

        this.uploadWebSocket = new WebSocket(wsUrl);

        this.uploadWebSocket.onopen = () => {
            console.log('📡 Upload WebSocket connected');
        };

        this.uploadWebSocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleUploadWebSocketMessage(data);
        };

        this.uploadWebSocket.onerror = (error) => {
            console.error('Upload WebSocket error:', error);
        };

        this.uploadWebSocket.onclose = () => {
            console.log('📡 Upload WebSocket disconnected');
        };
    }

    setupProcessingWebSocket(taskId) {
        const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/v2/ws/${taskId}`;

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            console.log('📡 Processing WebSocket connected');
        };

        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleProcessingWebSocketMessage(data);
        };

        this.websocket.onerror = (error) => {
            console.error('Processing WebSocket error:', error);
        };

        this.websocket.onclose = () => {
            console.log('📡 Processing WebSocket disconnected');
            if (this.isProcessing) {
                this.retryConnection();
            }
        };
    }

    handleUploadWebSocketMessage(data) {
        switch (data.type) {
            case 'upload_complete':
                this.updateProgress(100, 'Upload complete!');
                break;
            case 'upload_progress':
                this.updateProgress(data.progress, `Uploading... ${Math.round(data.progress)}%`);
                break;
        }
    }

    handleProcessingWebSocketMessage(data) {
        switch (data.type) {
            case 'connected':
                console.log('WebSocket ready for processing updates');
                break;

            case 'progress_update':
                this.handleProgressUpdate(data.data);
                break;

            case 'processing_complete':
                this.handleProcessingComplete(data.data);
                break;

            case 'processing_error':
                this.handleProcessingError(data.data);
                break;

            case 'keep_alive':
                // Send pong back
                if (this.websocket?.readyState === WebSocket.OPEN) {
                    this.websocket.send(JSON.stringify({ type: 'pong' }));
                }
                break;
        }
    }

    handleProgressUpdate(progressData) {
        const { current, total, percentage, stage, message } = progressData;
        this.updateProgress(percentage, message || `Processing ${current}/${total}...`);

        // Update detailed progress
        this.updateDetailedProgress(progressData);
    }

    handleProcessingComplete(data) {
        this.isProcessing = false;
        this.processingData = data.results;
        this.updateProgress(100, 'Processing complete!');
        this.displayResults(data.results);
        this.setState({ isLoading: false, currentStep: 'results' });
        this.showNotification('All clips processed successfully!', 'success');

        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }

    handleProcessingError(data) {
        this.isProcessing = false;
        this.setState({ isLoading: false });
        this.handleError(new Error(data.error), 'Processing failed');

        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }

    // UI Updates
    updateProgress(percentage, message) {
        const progressBar = document.querySelector('.progress-fill');
        const progressText = document.querySelector('.progress-text');
        const progressDetail = document.querySelector('.progress-detail');

        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
        }

        if (progressText) {
            progressText.textContent = message;
        }

        if (progressDetail) {
            progressDetail.textContent = `${Math.round(percentage)}% complete`;
        }

        this.state.progress = percentage;
    }

    updateDetailedProgress(progressData) {
        const container = document.getElementById('detailed-progress');
        if (!container) return;

        container.innerHTML = `
            <div class="progress-stage">
                <span class="stage-name">${progressData.stage}</span>
                <span class="stage-progress">${progressData.current}/${progressData.total}</span>
            </div>
            ${progressData.estimated_time_remaining ? 
                `<div class="time-remaining">ETA: ${this.formatTime(progressData.estimated_time_remaining)}</div>` : 
                ''
            }
        `;
    }

    displayAnalysisResults(data) {
        const container = document.getElementById('analysis-results');
        if (!container) return;

        const { video_info, ai_insights, suggested_clips } = data;

        container.innerHTML = `
            <div class="analysis-header">
                <h3>AI Analysis Complete</h3>
                <div class="viral-score">
                    <span class="score-label">Viral Potential</span>
                    <div class="score-circle">
                        <span class="score-value">${ai_insights.viral_potential}%</span>
                    </div>
                </div>
            </div>

            <div class="video-info">
                <div class="video-preview">
                    <img src="${video_info.thumbnail}" alt="Video thumbnail" />
                    <div class="video-overlay">
                        <div class="viral-badge">
                            ${ai_insights.viral_potential}% Viral Score
                        </div>
                    </div>
                </div>

                <div class="video-details">
                    <h4>${video_info.title}</h4>
                    <div class="video-stats">
                        <span>Duration: ${this.formatDuration(video_info.duration)}</span>
                        <span>Views: ${this.formatNumber(video_info.view_count)}</span>
                        <span>Platform: ${video_info.platform}</span>
                    </div>
                </div>
            </div>

            <div class="ai-insights">
                <h4>AI Insights</h4>
                <div class="insights-grid">
                    <div class="insight">
                        <span class="insight-label">Content Type</span>
                        <span class="insight-value">${ai_insights.content_type}</span>
                    </div>
                    <div class="insight">
                        <span class="insight-label">Optimal Length</span>
                        <span class="insight-value">${ai_insights.optimal_length}s</span>
                    </div>
                    <div class="insight">
                        <span class="insight-label">Hook Quality</span>
                        <span class="insight-value">${ai_insights.hook_quality}%</span>
                    </div>
                    <div class="insight">
                        <span class="insight-label">Retention</span>
                        <span class="insight-value">${ai_insights.audience_retention}%</span>
                    </div>
                </div>
            </div>

            <div class="suggested-clips">
                <h4>Suggested Clips (${suggested_clips.length})</h4>
                <div class="clips-list">
                    ${suggested_clips.map((clip, index) => `
                        <div class="clip-card" data-clip-index="${index}">
                            <div class="clip-header">
                                <input type="checkbox" class="clip-select" checked>
                                <h5>${clip.title}</h5>
                                <span class="viral-score">${clip.viral_score}%</span>
                            </div>
                            <div class="clip-details">
                                <p>${clip.description}</p>
                                <div class="clip-info">
                                    <span>⏱️ ${this.formatDuration(clip.end_time - clip.start_time)}</span>
                                    <span>📱 ${clip.recommended_platforms.join(', ')}</span>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="action-buttons">
                <button class="btn btn-primary process-btn">
                    🚀 Process Selected Clips
                </button>
                <button class="btn btn-secondary regenerate-btn">
                    🔄 Regenerate Analysis
                </button>
            </div>
        `;
    }

    displayUploadResults(data) {
        const container = document.getElementById('upload-results');
        if (!container) return;

        container.innerHTML = `
            <div class="upload-success">
                <div class="success-icon">✅</div>
                <h3>Upload Successful</h3>
                <div class="file-info">
                    <div class="file-name">${data.filename}</div>
                    <div class="file-size">${this.formatFileSize(data.file_size)}</div>
                    ${data.thumbnail ? `<img src="${data.thumbnail}" alt="Video thumbnail" class="file-thumbnail" />` : ''}
                </div>

                <div class="metadata">
                    <h4>Video Information</h4>
                    <div class="metadata-grid">
                        ${Object.entries(data.metadata || {}).map(([key, value]) => `
                            <div class="metadata-item">
                                <span class="metadata-key">${key}</span>
                                <span class="metadata-value">${value}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <div class="next-steps">
                    <h4>Next Steps</h4>
                    <p>Your video is ready for processing. You can now create viral clips!</p>
                    <button class="btn btn-primary process-btn">
                        🎬 Start Processing
                    </button>
                </div>
            </div>
        `;
    }

    displayResults(results) {
        const container = document.getElementById('results-container');
        if (!container) return;

        const successfulClips = results.filter(r => r.success);
        const failedClips = results.filter(r => !r.success);

        container.innerHTML = `
            <div class="results-header">
                <h3>Processing Complete</h3>
                <div class="results-stats">
                    <span class="success-count">${successfulClips.length} successful</span>
                    ${failedClips.length > 0 ? `<span class="error-count">${failedClips.length} failed</span>` : ''}
                </div>
            </div>

            <div class="results-grid">
                ${successfulClips.map((result, index) => `
                    <div class="result-card">
                        <div class="result-preview">
                            ${result.thumbnail_path ? 
                                `<img src="${result.thumbnail_path}" alt="Clip thumbnail" />` :
                                '<div class="no-thumbnail">🎬</div>'
                            }
                            <div class="result-overlay">
                                <button class="btn btn-primary download-btn" data-clip-index="${result.clip_index}">
                                    📥 Download
                                </button>
                            </div>
                        </div>

                        <div class="result-info">
                            <h4>Clip ${result.clip_index + 1}</h4>
                            <div class="result-stats">
                                <span>Duration: ${this.formatDuration(result.duration)}</span>
                                <span>Size: ${this.formatFileSize(result.file_size)}</span>
                                <span>Viral Score: ${result.viral_score}%</span>
                            </div>

                            <div class="result-enhancements">
                                <h5>AI Enhancements</h5>
                                <ul>
                                    ${result.ai_enhancements.map(enhancement => 
                                        `<li>${enhancement}</li>`
                                    ).join('')}
                                </ul>
                            </div>

                            <div class="result-actions">
                                <button class="btn btn-secondary share-btn" data-clip-index="${result.clip_index}">
                                    📤 Share
                                </button>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>

            ${failedClips.length > 0 ? `
                <div class="failed-clips">
                    <h4>Failed Clips</h4>
                    ${failedClips.map(result => `
                        <div class="error-item">
                            <span>Clip ${result.clip_index + 1}</span>
                            <span class="error-message">${result.error}</span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        `;
    }

    // Utility Functions
    validateURL(url) {
        try {
            new URL(url);
            return url.includes('youtube.com') || url.includes('youtu.be') || 
                   url.includes('tiktok.com') || url.includes('instagram.com');
        } catch {
            return false;
        }
    }

    validateFile(file) {
        const maxSize = 2 * 1024 * 1024 * 1024; // 2GB
        const allowedTypes = ['video/mp4', 'video/mov', 'video/avi', 'video/mkv', 'video/webm'];

        if (!allowedTypes.includes(file.type)) {
            this.showNotification('Please upload a supported video format (MP4, MOV, AVI, MKV, WebM)', 'error');
            return false;
        }

        if (file.size > maxSize) {
            this.showNotification('File size must be less than 2GB', 'error');
            return false;
        }

        return true;
    }

    getSelectedClips() {
        const checkboxes = document.querySelectorAll('.clip-select:checked');
        const clips = [];

        checkboxes.forEach(checkbox => {
            const clipCard = checkbox.closest('.clip-card');
            const clipIndex = parseInt(clipCard.dataset.clipIndex);

            if (this.analysisData?.suggested_clips[clipIndex]) {
                clips.push(this.analysisData.suggested_clips[clipIndex]);
            }
        });

        return clips;
    }

    async handleDownload(clipIndex) {
        if (!this.currentTask) return;

        try {
            window.open(`/api/v2/download/${this.currentTask}/${clipIndex}`, '_blank');
            this.showNotification('Download started!', 'success');
        } catch (error) {
            this.showNotification('Download failed', 'error');
        }
    }

    async handleShare(clipIndex) {
        if (navigator.share) {
            try {
                await navigator.share({
                    title: 'Check out my viral clip!',
                    text: 'Created with ViralClip Pro',
                    url: window.location.href
                });
            } catch (error) {
                this.copyToClipboard(window.location.href);
            }
        } else {
            this.copyToClipboard(window.location.href);
        }
    }

    async handleRegenerate() {
        if (this.analysisData) {
            this.handleAnalyze();
        }
    }

    handleCancel() {
        if (this.isProcessing && this.websocket) {
            this.websocket.close();
            this.isProcessing = false;
        }

        this.setState({ isLoading: false });
        this.updateProgress(0, 'Cancelled');
    }

    navigateToStep(step) {
        this.setState({ currentStep: step });
        this.updateUIState();
    }

    setState(newState) {
        this.state = { ...this.state, ...newState };
        this.updateUIState();
    }

    updateUIState() {
        // Update step indicators
        document.querySelectorAll('.step').forEach(step => {
            step.classList.toggle('active', step.dataset.step === this.state.currentStep);
            step.classList.toggle('completed', this.isStepCompleted(step.dataset.step));
        });

        // Update loading states
        document.body.classList.toggle('loading', this.state.isLoading);

        // Update connection status
        document.body.classList.toggle('offline', this.state.connectionStatus === 'offline');
    }

    isStepCompleted(step) {
        switch (step) {
            case 'upload': return this.currentSession !== null;
            case 'analyze': return this.analysisData !== null;
            case 'process': return this.processingData.length > 0;
            default: return false;
        }
    }

    // API and Network
    async apiRequest(url, options = {}) {
        const response = await fetch(url, {
            ...options,
            headers: {
                ...options.headers,
                'X-Requested-With': 'XMLHttpRequest'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    async uploadWithProgress(formData, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const progress = (e.loaded / e.total) * 100;
                    onProgress(progress);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    try {
                        resolve(JSON.parse(xhr.responseText));
                    } catch (error) {
                        reject(new Error('Invalid response format'));
                    }
                } else {
                    reject(new Error(`Upload failed: ${xhr.statusText}`));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Upload failed'));
            });

            xhr.open('POST', '/api/v2/upload-video');
            xhr.send(formData);
        });
    }

    retryConnection() {
        if (this.retryAttempts < this.maxRetries) {
            this.retryAttempts++;
            setTimeout(() => {
                if (this.currentTask) {
                    this.setupProcessingWebSocket(this.currentTask);
                }
            }, 1000 * this.retryAttempts);
        }
    }

    // Error Handling
    handleError(error, context = 'Operation') {
        console.error(`${context}:`, error);

        const message = error.message || 'An unexpected error occurred';
        this.showNotification(`${context}: ${message}`, 'error');

        this.setState({ error: message });
    }

    handleGlobalError(event) {
        console.error('Global error:', event);
        this.showNotification('Something went wrong. Please try again.', 'error');
    }

    // UI Helpers
    showNotification(message, type = 'info', duration = 5000) {
        // Remove existing notifications
        document.querySelectorAll('.notification').forEach(n => n.remove());

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-message">${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto remove
        if (duration > 0) {
            setTimeout(() => notification.remove(), duration);
        }

        // Manual close
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });

        // Animate in
        requestAnimationFrame(() => {
            notification.classList.add('show');
        });
    }

    getNotificationIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }

    copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showNotification('Copied to clipboard!', 'success', 2000);
        });
    }

    // Formatting Helpers
    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatFileSize(bytes) {
        const units = ['B', 'KB', 'MB', 'GB'];
        let size = bytes;
        let unitIndex = 0;

        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }

        return `${size.toFixed(1)} ${units[unitIndex]}`;
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    formatTime(seconds) {
        if (seconds < 60) {
            return `${seconds}s`;
        } else if (seconds < 3600) {
            return `${Math.floor(seconds / 60)}m`;
        } else {
            return `${Math.floor(seconds / 3600)}h`;
        }
    }

    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Enhanced connection monitoring
class ConnectionMonitor {
    constructor(app) {
        this.app = app;
        this.isOnline = navigator.onLine;
        this.retryQueue = [];
        this.heartbeatInterval = null;

        this.setupEventListeners();
        this.startHeartbeat();
    }

    setupEventListeners() {
        window.addEventListener('online', () => this.handleOnline());
        window.addEventListener('offline', () => this.handleOffline());

        // Monitor connection quality
        if ('connection' in navigator) {
            navigator.connection.addEventListener('change', () => {
                this.handleConnectionChange();
            });
        }
    }

    handleOnline() {
        console.log('📶 Back online');
        this.isOnline = true;
        this.app.showNotification('Connection restored', 'success', 3000);

        // Process retry queue
        this.processRetryQueue();

        // Update UI
        document.body.classList.remove('offline');
        this.startHeartbeat();
    }

    handleOffline() {
        console.log('📵 Gone offline');
        this.isOnline = false;
        this.app.showNotification(
            'Connection lost - some features may be limited',
            'warning',
            0
        );

        // Update UI
        document.body.classList.add('offline');
        this.stopHeartbeat();
    }

    handleConnectionChange() {
        if ('connection' in navigator) {
            const connection = navigator.connection;
            console.log(`Connection: ${connection.effectiveType}, ${connection.downlink}Mbps`);

            // Update connection indicator
            this.updateConnectionIndicator(connection);
        }
    }

    updateConnectionIndicator(connection) {
        const indicator = document.querySelector('.connection-indicator');
        if (!indicator) return;

        const signal = indicator.querySelector('.connection-signal');
        const text = indicator.querySelector('.connection-text');

        if (connection.effectiveType === '4g' && connection.downlink > 1) {
            signal.className = 'connection-signal strong';
            text.textContent = 'Strong connection';
        } else if (connection.effectiveType === '3g' || connection.downlink > 0.5) {
            signal.className = 'connection-signal weak';
            text.textContent = 'Weak connection';
        } else {
            signal.className = 'connection-signal poor';
            text.textContent = 'Poor connection';
        }

        indicator.classList.add('show');
        setTimeout(() => indicator.classList.remove('show'), 3000);
    }

    startHeartbeat() {
        this.stopHeartbeat();
        this.heartbeatInterval = setInterval(() => {
            this.checkConnection();
        }, 30000); // Every 30 seconds
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    async checkConnection() {
        try {
            const response = await fetch('/health', { 
                method: 'HEAD',
                cache: 'no-cache'
            });

            if (!response.ok) {
                throw new Error('Health check failed');
            }
        } catch (error) {
            console.warn('Connection check failed:', error);
            this.handleOffline();
        }
    }

    processRetryQueue() {
        while (this.retryQueue.length > 0) {
            const request = this.retryQueue.shift();
            request();
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.viralClipApp = new ViralClipApp();
});

// Export for global access
window.ViralClipApp = ViralClipApp;