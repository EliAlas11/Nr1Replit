
/*
ViralClip Pro v6.0 - Netflix-Level Upload Experience
Enterprise-grade upload system with real-time feedback and mobile optimization
*/

class NetflixLevelUploadManager {
    constructor() {
        this.uploadQueue = new Map();
        this.activeUploads = new Map();
        this.socket = null;
        this.maxConcurrentUploads = 3;
        this.chunkSize = 5 * 1024 * 1024; // 5MB chunks
        this.supportedFormats = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v', '3gp', 'mp3', 'wav'];
        this.maxFileSize = 500 * 1024 * 1024; // 500MB
        
        // Performance monitoring
        this.performanceMetrics = {
            totalUploads: 0,
            successfulUploads: 0,
            averageUploadSpeed: 0,
            currentConnections: 0
        };
        
        this.initializeUploadSystem();
        this.setupWebSocketConnection();
        this.initializeTouchOptimizations();
    }

    initializeUploadSystem() {
        // Create enhanced upload interface
        const uploadSection = document.createElement('div');
        uploadSection.id = 'enhanced-upload-section';
        uploadSection.className = 'netflix-upload-section';
        uploadSection.innerHTML = `
            <div class="upload-header">
                <h1 class="upload-title">🎬 Netflix-Grade Upload Experience</h1>
                <div class="upload-stats">
                    <span class="stat-item">
                        <span class="stat-icon">⚡</span>
                        <span class="stat-label">Speed: <span id="upload-speed">0 MB/s</span></span>
                    </span>
                    <span class="stat-item">
                        <span class="stat-icon">📊</span>
                        <span class="stat-label">Queue: <span id="queue-count">0</span></span>
                    </span>
                    <span class="stat-item">
                        <span class="stat-icon">✅</span>
                        <span class="stat-label">Success: <span id="success-rate">100%</span></span>
                    </span>
                </div>
            </div>

            <!-- Enhanced Drag & Drop Zone -->
            <div class="netflix-drop-zone" id="mainDropZone">
                <div class="drop-zone-content">
                    <div class="drop-zone-icon" id="dropZoneIcon">
                        <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <polyline points="14,2 14,8 20,8"/>
                            <line x1="12" y1="18" x2="12" y2="12"/>
                            <line x1="9" y1="15" x2="12" y2="12"/>
                            <line x1="15" y1="15" x2="12" y2="12"/>
                        </svg>
                    </div>
                    <div class="drop-zone-text">
                        <h2 id="dropZoneTitle">Drop your video files here</h2>
                        <p id="dropZoneSubtitle">or click to browse files</p>
                        <div class="supported-formats">
                            <span class="format-label">Supported formats:</span>
                            <span class="format-list">${this.supportedFormats.join(', ').toUpperCase()}</span>
                        </div>
                        <div class="file-limits">
                            <span class="limit-item">📐 Max size: 500MB</span>
                            <span class="limit-item">🎯 Best quality: 1080p+</span>
                            <span class="limit-item">⚡ Lightning fast processing</span>
                        </div>
                    </div>
                    <div class="drop-zone-actions">
                        <button class="netflix-btn primary-btn" id="browseFilesBtn">
                            <span class="btn-icon">📁</span>
                            <span class="btn-text">Browse Files</span>
                        </button>
                        <button class="netflix-btn secondary-btn" id="pasteUrlBtn">
                            <span class="btn-icon">🔗</span>
                            <span class="btn-text">Paste URL</span>
                        </button>
                    </div>
                </div>
                
                <!-- Upload Progress Overlay -->
                <div class="upload-overlay" id="uploadOverlay" style="display: none;">
                    <div class="upload-animation">
                        <div class="upload-pulse"></div>
                        <div class="upload-icon">📤</div>
                    </div>
                    <div class="upload-text">Processing your upload...</div>
                </div>
            </div>

            <!-- File Preview Section -->
            <div class="file-preview-section" id="filePreviewSection" style="display: none;">
                <h3 class="preview-title">📋 Upload Queue</h3>
                <div class="file-previews" id="filePreviews"></div>
            </div>

            <!-- Upload Queue Manager -->
            <div class="queue-manager" id="queueManager" style="display: none;">
                <div class="queue-header">
                    <h3>🎬 Upload Manager</h3>
                    <div class="queue-controls">
                        <button class="queue-btn" id="pauseAllBtn">⏸️ Pause All</button>
                        <button class="queue-btn" id="resumeAllBtn">▶️ Resume All</button>
                        <button class="queue-btn danger" id="clearQueueBtn">🗑️ Clear Queue</button>
                    </div>
                </div>
                <div class="upload-items" id="uploadItems"></div>
            </div>

            <!-- Real-time Analytics -->
            <div class="upload-analytics" id="uploadAnalytics" style="display: none;">
                <div class="analytics-grid">
                    <div class="metric-card">
                        <div class="metric-icon">⚡</div>
                        <div class="metric-content">
                            <div class="metric-value" id="currentSpeed">0 MB/s</div>
                            <div class="metric-label">Current Speed</div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">⏱️</div>
                        <div class="metric-content">
                            <div class="metric-value" id="remainingTime">--:--</div>
                            <div class="metric-label">Time Remaining</div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">📊</div>
                        <div class="metric-content">
                            <div class="metric-value" id="throughput">0%</div>
                            <div class="metric-label">Efficiency</div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">🌐</div>
                        <div class="metric-content">
                            <div class="metric-value" id="connectionQuality">Excellent</div>
                            <div class="metric-label">Connection</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Hidden file input -->
            <input type="file" id="fileInput" multiple accept="${this.getAcceptString()}" style="display: none;">
        `;

        // Insert into page
        const mainContent = document.querySelector('main') || document.body;
        const existingUpload = document.querySelector('#upload-section');
        if (existingUpload) {
            existingUpload.replaceWith(uploadSection);
        } else {
            mainContent.insertBefore(uploadSection, mainContent.firstChild);
        }

        this.setupUploadEventListeners();
        this.initializeDragAndDrop();
        this.setupMobileOptimizations();
    }

    setupUploadEventListeners() {
        // File browser
        document.getElementById('browseFilesBtn').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });

        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.handleFileSelection(Array.from(e.target.files));
        });

        // URL paste functionality
        document.getElementById('pasteUrlBtn').addEventListener('click', () => {
            this.showUrlPasteModal();
        });

        // Queue controls
        document.getElementById('pauseAllBtn')?.addEventListener('click', () => {
            this.pauseAllUploads();
        });

        document.getElementById('resumeAllBtn')?.addEventListener('click', () => {
            this.resumeAllUploads();
        });

        document.getElementById('clearQueueBtn')?.addEventListener('click', () => {
            this.clearUploadQueue();
        });
    }

    initializeDragAndDrop() {
        const dropZone = document.getElementById('mainDropZone');
        let dragCounter = 0;

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dragCounter++;
                this.highlightDropZone(true);
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dragCounter--;
                if (dragCounter === 0) {
                    this.highlightDropZone(false);
                }
            }, false);
        });

        // Handle dropped files
        dropZone.addEventListener('drop', (e) => {
            const files = Array.from(e.dataTransfer.files);
            this.handleFileSelection(files);
        }, false);

        // Click to upload
        dropZone.addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlightDropZone(highlight) {
        const dropZone = document.getElementById('mainDropZone');
        const icon = document.getElementById('dropZoneIcon');
        const title = document.getElementById('dropZoneTitle');
        const subtitle = document.getElementById('dropZoneSubtitle');

        if (highlight) {
            dropZone.classList.add('drag-active');
            title.textContent = '🎯 Drop files to upload';
            subtitle.textContent = 'Release to start uploading';
            icon.style.transform = 'scale(1.1) rotate(5deg)';
        } else {
            dropZone.classList.remove('drag-active');
            title.textContent = 'Drop your video files here';
            subtitle.textContent = 'or click to browse files';
            icon.style.transform = 'scale(1) rotate(0deg)';
        }
    }

    async handleFileSelection(files) {
        if (files.length === 0) return;

        // Show file preview section
        document.getElementById('filePreviewSection').style.display = 'block';
        document.getElementById('queueManager').style.display = 'block';
        document.getElementById('uploadAnalytics').style.display = 'block';

        // Validate and process each file
        for (const file of files) {
            const validation = await this.validateFile(file);
            
            if (validation.valid) {
                await this.addFileToQueue(file, validation);
            } else {
                this.showFileError(file, validation.error);
            }
        }

        // Start processing queue
        this.processUploadQueue();
    }

    async validateFile(file) {
        const fileExtension = file.name.split('.').pop().toLowerCase();
        
        // Format validation
        if (!this.supportedFormats.includes(fileExtension)) {
            return {
                valid: false,
                error: `Unsupported format: ${fileExtension}. Supported: ${this.supportedFormats.join(', ')}`
            };
        }

        // Size validation
        if (file.size > this.maxFileSize) {
            return {
                valid: false,
                error: `File too large: ${this.formatFileSize(file.size)}. Maximum: ${this.formatFileSize(this.maxFileSize)}`
            };
        }

        if (file.size === 0) {
            return {
                valid: false,
                error: 'File is empty'
            };
        }

        // Generate thumbnail for video files
        let thumbnail = null;
        if (file.type.startsWith('video/')) {
            thumbnail = await this.generateVideoThumbnail(file);
        }

        return {
            valid: true,
            thumbnail,
            estimatedTime: this.estimateUploadTime(file.size),
            chunks: Math.ceil(file.size / this.chunkSize)
        };
    }

    async generateVideoThumbnail(file) {
        return new Promise((resolve) => {
            const video = document.createElement('video');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            video.addEventListener('loadedmetadata', () => {
                canvas.width = 160;
                canvas.height = 90;
                
                video.currentTime = Math.min(2, video.duration / 4); // Seek to 25% or 2s
            });

            video.addEventListener('seeked', () => {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const thumbnailUrl = canvas.toDataURL('image/jpeg', 0.8);
                resolve(thumbnailUrl);
                
                // Cleanup
                URL.revokeObjectURL(video.src);
            });

            video.addEventListener('error', () => {
                resolve(null);
            });

            video.src = URL.createObjectURL(file);
            video.load();
        });
    }

    async addFileToQueue(file, validation) {
        const uploadId = this.generateUploadId();
        const uploadItem = {
            id: uploadId,
            file,
            validation,
            status: 'queued',
            progress: 0,
            speed: 0,
            remainingTime: validation.estimatedTime,
            uploadedChunks: 0,
            totalChunks: validation.chunks,
            retryCount: 0,
            maxRetries: 3,
            startTime: null,
            pauseOffset: 0
        };

        this.uploadQueue.set(uploadId, uploadItem);
        this.renderFilePreview(uploadItem);
        this.updateQueueStats();
    }

    renderFilePreview(uploadItem) {
        const previewsContainer = document.getElementById('filePreviews');
        const uploadItemsContainer = document.getElementById('uploadItems');

        // File preview card
        const previewCard = document.createElement('div');
        previewCard.className = 'file-preview-card';
        previewCard.innerHTML = `
            <div class="preview-thumbnail">
                ${uploadItem.validation.thumbnail ? 
                    `<img src="${uploadItem.validation.thumbnail}" alt="Video thumbnail">` :
                    `<div class="thumbnail-placeholder">
                        <span class="file-icon">${this.getFileIcon(uploadItem.file.type)}</span>
                    </div>`
                }
                <div class="file-format">.${uploadItem.file.name.split('.').pop().toUpperCase()}</div>
            </div>
            <div class="preview-info">
                <div class="file-name" title="${uploadItem.file.name}">${uploadItem.file.name}</div>
                <div class="file-details">
                    <span class="file-size">${this.formatFileSize(uploadItem.file.size)}</span>
                    <span class="file-duration">~${uploadItem.validation.estimatedTime}s</span>
                    <span class="file-chunks">${uploadItem.totalChunks} chunks</span>
                </div>
            </div>
        `;

        previewsContainer.appendChild(previewCard);

        // Upload manager item
        const uploadManagerItem = document.createElement('div');
        uploadManagerItem.className = 'upload-item';
        uploadManagerItem.id = `upload-${uploadItem.id}`;
        uploadManagerItem.innerHTML = `
            <div class="upload-item-header">
                <div class="item-thumbnail">
                    ${uploadItem.validation.thumbnail ? 
                        `<img src="${uploadItem.validation.thumbnail}" alt="Thumbnail">` :
                        `<div class="placeholder-thumb">${this.getFileIcon(uploadItem.file.type)}</div>`
                    }
                </div>
                <div class="item-info">
                    <div class="item-name">${uploadItem.file.name}</div>
                    <div class="item-meta">
                        <span class="item-size">${this.formatFileSize(uploadItem.file.size)}</span>
                        <span class="item-status" id="status-${uploadItem.id}">Queued</span>
                    </div>
                </div>
                <div class="item-controls">
                    <button class="control-btn pause-btn" id="pause-${uploadItem.id}" title="Pause">⏸️</button>
                    <button class="control-btn retry-btn" id="retry-${uploadItem.id}" title="Retry" style="display: none;">🔄</button>
                    <button class="control-btn remove-btn" id="remove-${uploadItem.id}" title="Remove">🗑️</button>
                </div>
            </div>
            
            <div class="upload-progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-${uploadItem.id}" style="width: 0%"></div>
                    <div class="progress-text" id="progress-text-${uploadItem.id}">0%</div>
                </div>
                <div class="upload-stats">
                    <span class="upload-speed" id="speed-${uploadItem.id}">0 MB/s</span>
                    <span class="upload-eta" id="eta-${uploadItem.id}">--:--</span>
                    <span class="upload-chunks" id="chunks-${uploadItem.id}">0/${uploadItem.totalChunks}</span>
                </div>
            </div>

            <div class="chunk-progress" id="chunk-progress-${uploadItem.id}">
                ${Array(Math.min(uploadItem.totalChunks, 20)).fill(0).map((_, i) => 
                    `<div class="chunk-indicator" id="chunk-${uploadItem.id}-${i}"></div>`
                ).join('')}
                ${uploadItem.totalChunks > 20 ? '<span class="chunk-more">...</span>' : ''}
            </div>
        `;

        uploadItemsContainer.appendChild(uploadManagerItem);

        // Add event listeners
        this.setupUploadItemControls(uploadItem);
    }

    setupUploadItemControls(uploadItem) {
        const pauseBtn = document.getElementById(`pause-${uploadItem.id}`);
        const retryBtn = document.getElementById(`retry-${uploadItem.id}`);
        const removeBtn = document.getElementById(`remove-${uploadItem.id}`);

        pauseBtn?.addEventListener('click', () => {
            this.toggleUploadPause(uploadItem.id);
        });

        retryBtn?.addEventListener('click', () => {
            this.retryUpload(uploadItem.id);
        });

        removeBtn?.addEventListener('click', () => {
            this.removeUpload(uploadItem.id);
        });
    }

    async processUploadQueue() {
        const queuedUploads = Array.from(this.uploadQueue.values())
            .filter(item => item.status === 'queued')
            .slice(0, this.maxConcurrentUploads - this.activeUploads.size);

        for (const uploadItem of queuedUploads) {
            if (this.activeUploads.size >= this.maxConcurrentUploads) break;
            
            this.startUpload(uploadItem);
        }
    }

    async startUpload(uploadItem) {
        try {
            uploadItem.status = 'initializing';
            uploadItem.startTime = Date.now();
            this.activeUploads.set(uploadItem.id, uploadItem);
            
            this.updateUploadStatus(uploadItem.id, 'Initializing upload...');

            // Initialize upload session
            const sessionResponse = await fetch('/api/v6/upload/init', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this.getAuthHeaders()
                },
                body: JSON.stringify({
                    filename: uploadItem.file.name,
                    file_size: uploadItem.file.size,
                    total_chunks: uploadItem.totalChunks,
                    upload_id: uploadItem.id
                })
            });

            const sessionData = await sessionResponse.json();
            
            if (!sessionData.success) {
                throw new Error(sessionData.error || 'Failed to initialize upload');
            }

            uploadItem.status = 'uploading';
            this.updateUploadStatus(uploadItem.id, 'Uploading...');

            // Start chunked upload
            await this.uploadFileInChunks(uploadItem);

        } catch (error) {
            console.error('Upload failed:', error);
            this.handleUploadError(uploadItem, error.message);
        }
    }

    async uploadFileInChunks(uploadItem) {
        const file = uploadItem.file;
        const totalChunks = uploadItem.totalChunks;

        for (let chunkIndex = uploadItem.uploadedChunks; chunkIndex < totalChunks; chunkIndex++) {
            // Check if upload is paused
            if (uploadItem.status === 'paused') {
                return;
            }

            const start = chunkIndex * this.chunkSize;
            const end = Math.min(start + this.chunkSize, file.size);
            const chunk = file.slice(start, end);

            await this.uploadChunk(uploadItem, chunk, chunkIndex);
            
            uploadItem.uploadedChunks = chunkIndex + 1;
            this.updateUploadProgress(uploadItem);
        }

        // Upload complete
        uploadItem.status = 'completed';
        this.updateUploadStatus(uploadItem.id, 'Upload complete!');
        this.activeUploads.delete(uploadItem.id);
        
        // Continue processing queue
        this.processUploadQueue();
    }

    async uploadChunk(uploadItem, chunk, chunkIndex) {
        const maxRetries = 3;
        let attempt = 0;

        while (attempt < maxRetries) {
            try {
                const formData = new FormData();
                formData.append('file', chunk, `chunk_${chunkIndex}`);
                formData.append('upload_id', uploadItem.id);
                formData.append('chunk_index', chunkIndex);
                formData.append('total_chunks', uploadItem.totalChunks);

                const startTime = Date.now();
                
                const response = await fetch('/api/v6/upload/chunk', {
                    method: 'POST',
                    headers: this.getAuthHeaders(),
                    body: formData
                });

                const result = await response.json();
                
                if (!result.success) {
                    throw new Error(result.error || 'Chunk upload failed');
                }

                // Update speed calculation
                const uploadTime = (Date.now() - startTime) / 1000;
                const speed = chunk.size / uploadTime; // bytes per second
                uploadItem.speed = speed;

                // Update chunk indicator
                this.updateChunkIndicator(uploadItem.id, chunkIndex, 'completed');

                return result;

            } catch (error) {
                attempt++;
                this.updateChunkIndicator(uploadItem.id, chunkIndex, 'error');
                
                if (attempt >= maxRetries) {
                    throw error;
                }
                
                // Exponential backoff
                await this.delay(Math.pow(2, attempt) * 1000);
                this.updateChunkIndicator(uploadItem.id, chunkIndex, 'retrying');
            }
        }
    }

    updateUploadProgress(uploadItem) {
        const progress = (uploadItem.uploadedChunks / uploadItem.totalChunks) * 100;
        uploadItem.progress = progress;

        // Update progress bar
        const progressFill = document.getElementById(`progress-${uploadItem.id}`);
        const progressText = document.getElementById(`progress-text-${uploadItem.id}`);
        const speedElement = document.getElementById(`speed-${uploadItem.id}`);
        const etaElement = document.getElementById(`eta-${uploadItem.id}`);
        const chunksElement = document.getElementById(`chunks-${uploadItem.id}`);

        if (progressFill) progressFill.style.width = `${progress}%`;
        if (progressText) progressText.textContent = `${Math.round(progress)}%`;
        if (speedElement) speedElement.textContent = this.formatSpeed(uploadItem.speed);
        if (chunksElement) chunksElement.textContent = `${uploadItem.uploadedChunks}/${uploadItem.totalChunks}`;

        // Calculate ETA
        if (uploadItem.speed > 0) {
            const remainingBytes = (uploadItem.totalChunks - uploadItem.uploadedChunks) * this.chunkSize;
            const remainingTime = remainingBytes / uploadItem.speed;
            uploadItem.remainingTime = remainingTime;
            
            if (etaElement) etaElement.textContent = this.formatTime(remainingTime);
        }

        // Update global stats
        this.updateGlobalStats();
    }

    updateChunkIndicator(uploadId, chunkIndex, status) {
        const indicator = document.getElementById(`chunk-${uploadId}-${chunkIndex}`);
        if (indicator) {
            indicator.className = `chunk-indicator ${status}`;
        }
    }

    updateUploadStatus(uploadId, status) {
        const statusElement = document.getElementById(`status-${uploadId}`);
        if (statusElement) {
            statusElement.textContent = status;
        }
    }

    handleUploadError(uploadItem, errorMessage) {
        uploadItem.status = 'error';
        uploadItem.retryCount++;
        
        this.updateUploadStatus(uploadItem.id, `Error: ${errorMessage}`);
        this.activeUploads.delete(uploadItem.id);

        // Show retry button if under max retries
        if (uploadItem.retryCount < uploadItem.maxRetries) {
            const retryBtn = document.getElementById(`retry-${uploadItem.id}`);
            if (retryBtn) {
                retryBtn.style.display = 'block';
            }
        }

        // Continue processing queue
        this.processUploadQueue();
    }

    // Queue management methods
    toggleUploadPause(uploadId) {
        const uploadItem = this.uploadQueue.get(uploadId);
        if (!uploadItem) return;

        const pauseBtn = document.getElementById(`pause-${uploadId}`);
        
        if (uploadItem.status === 'uploading') {
            uploadItem.status = 'paused';
            this.updateUploadStatus(uploadId, 'Paused');
            if (pauseBtn) {
                pauseBtn.innerHTML = '▶️';
                pauseBtn.title = 'Resume';
            }
            this.activeUploads.delete(uploadId);
        } else if (uploadItem.status === 'paused') {
            uploadItem.status = 'queued';
            this.updateUploadStatus(uploadId, 'Resuming...');
            if (pauseBtn) {
                pauseBtn.innerHTML = '⏸️';
                pauseBtn.title = 'Pause';
            }
            this.processUploadQueue();
        }
    }

    retryUpload(uploadId) {
        const uploadItem = this.uploadQueue.get(uploadId);
        if (!uploadItem) return;

        uploadItem.status = 'queued';
        uploadItem.uploadedChunks = 0;
        uploadItem.progress = 0;
        
        this.updateUploadStatus(uploadId, 'Retrying...');
        this.updateUploadProgress(uploadItem);
        
        // Hide retry button
        const retryBtn = document.getElementById(`retry-${uploadId}`);
        if (retryBtn) {
            retryBtn.style.display = 'none';
        }

        this.processUploadQueue();
    }

    removeUpload(uploadId) {
        const uploadItem = this.uploadQueue.get(uploadId);
        if (!uploadItem) return;

        // Remove from active uploads if necessary
        this.activeUploads.delete(uploadId);
        
        // Remove from queue
        this.uploadQueue.delete(uploadId);

        // Remove from UI
        const uploadElement = document.getElementById(`upload-${uploadId}`);
        if (uploadElement) {
            uploadElement.remove();
        }

        this.updateQueueStats();
        this.processUploadQueue();
    }

    pauseAllUploads() {
        this.activeUploads.forEach((uploadItem) => {
            if (uploadItem.status === 'uploading') {
                this.toggleUploadPause(uploadItem.id);
            }
        });
    }

    resumeAllUploads() {
        this.uploadQueue.forEach((uploadItem) => {
            if (uploadItem.status === 'paused') {
                this.toggleUploadPause(uploadItem.id);
            }
        });
    }

    clearUploadQueue() {
        if (confirm('Are you sure you want to clear the upload queue? This will cancel all uploads.')) {
            // Cancel all active uploads
            this.activeUploads.clear();
            
            // Clear queue
            this.uploadQueue.clear();

            // Clear UI
            document.getElementById('filePreviews').innerHTML = '';
            document.getElementById('uploadItems').innerHTML = '';
            
            // Hide sections
            document.getElementById('filePreviewSection').style.display = 'none';
            document.getElementById('queueManager').style.display = 'none';
            document.getElementById('uploadAnalytics').style.display = 'none';

            this.updateQueueStats();
        }
    }

    // Mobile optimizations
    setupMobileOptimizations() {
        // Touch-friendly drag and drop
        const dropZone = document.getElementById('mainDropZone');
        
        // Add touch events for mobile
        dropZone.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.highlightDropZone(true);
        });

        dropZone.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.highlightDropZone(false);
            // On mobile, just trigger file browser
            document.getElementById('fileInput').click();
        });

        // Optimize for small screens
        this.setupResponsiveLayout();
        
        // Add haptic feedback if available
        this.setupHapticFeedback();
    }

    setupResponsiveLayout() {
        const mediaQuery = window.matchMedia('(max-width: 768px)');
        
        const handleMobileLayout = (e) => {
            const dropZone = document.getElementById('mainDropZone');
            const analytics = document.getElementById('uploadAnalytics');
            
            if (e.matches) {
                // Mobile layout adjustments
                dropZone.classList.add('mobile-layout');
                analytics?.classList.add('mobile-analytics');
                this.maxConcurrentUploads = 1; // Limit concurrent uploads on mobile
            } else {
                // Desktop layout
                dropZone.classList.remove('mobile-layout');
                analytics?.classList.remove('mobile-analytics');
                this.maxConcurrentUploads = 3;
            }
        };

        mediaQuery.addListener(handleMobileLayout);
        handleMobileLayout(mediaQuery);
    }

    setupHapticFeedback() {
        if ('vibrate' in navigator) {
            this.hapticFeedback = {
                success: () => navigator.vibrate([100]),
                error: () => navigator.vibrate([200, 100, 200]),
                progress: () => navigator.vibrate([50])
            };
        } else {
            this.hapticFeedback = {
                success: () => {},
                error: () => {},
                progress: () => {}
            };
        }
    }

    // WebSocket connection for real-time updates
    setupWebSocketConnection() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/v6/ws/enterprise/upload_manager`;
            
            this.socket = new WebSocket(wsUrl);
            
            this.socket.onopen = () => {
                console.log('📡 Upload WebSocket connected');
                this.updateConnectionStatus('connected');
            };
            
            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.socket.onclose = () => {
                console.log('📡 Upload WebSocket disconnected');
                this.updateConnectionStatus('disconnected');
                // Attempt to reconnect
                setTimeout(() => this.setupWebSocketConnection(), 5000);
            };
            
            this.socket.onerror = (error) => {
                console.error('📡 Upload WebSocket error:', error);
                this.updateConnectionStatus('error');
            };
            
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'upload_progress':
                this.handleProgressUpdate(data.progress);
                break;
            case 'upload_complete':
                this.handleUploadComplete(data);
                break;
            case 'upload_error':
                this.handleUploadError(data);
                break;
            case 'system_stats':
                this.updateSystemStats(data.stats);
                break;
        }
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            const statusMap = {
                connected: '🟢 Connected',
                disconnected: '🔴 Disconnected',
                error: '🟡 Connection Error'
            };
            statusElement.textContent = statusMap[status] || status;
        }
    }

    // Utility methods
    generateUploadId() {
        return `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    getAcceptString() {
        return this.supportedFormats.map(format => `.${format}`).join(',');
    }

    getFileIcon(mimeType) {
        if (mimeType.startsWith('video/')) return '🎬';
        if (mimeType.startsWith('audio/')) return '🎵';
        return '📄';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatSpeed(bytesPerSecond) {
        if (bytesPerSecond === 0) return '0 MB/s';
        const mbps = bytesPerSecond / (1024 * 1024);
        return `${mbps.toFixed(1)} MB/s`;
    }

    formatTime(seconds) {
        if (!seconds || seconds === Infinity) return '--:--';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    estimateUploadTime(fileSize) {
        // Estimate based on average upload speed (assuming 5 MB/s)
        const averageSpeed = 5 * 1024 * 1024; // 5 MB/s
        return Math.ceil(fileSize / averageSpeed);
    }

    updateQueueStats() {
        const queueCount = this.uploadQueue.size;
        const queueElement = document.getElementById('queue-count');
        if (queueElement) {
            queueElement.textContent = queueCount;
        }
    }

    updateGlobalStats() {
        // Update global performance metrics
        const totalSpeed = Array.from(this.activeUploads.values())
            .reduce((sum, item) => sum + (item.speed || 0), 0);
        
        const speedElement = document.getElementById('upload-speed');
        const currentSpeedElement = document.getElementById('currentSpeed');
        
        if (speedElement) speedElement.textContent = this.formatSpeed(totalSpeed);
        if (currentSpeedElement) currentSpeedElement.textContent = this.formatSpeed(totalSpeed);

        // Update success rate
        const totalUploads = this.performanceMetrics.totalUploads;
        const successfulUploads = this.performanceMetrics.successfulUploads;
        const successRate = totalUploads > 0 ? (successfulUploads / totalUploads) * 100 : 100;
        
        const successElement = document.getElementById('success-rate');
        if (successElement) successElement.textContent = `${successRate.toFixed(0)}%`;
    }

    getAuthHeaders() {
        // Return authentication headers if needed
        return {
            'Authorization': 'Bearer ' + (localStorage.getItem('token') || '')
        };
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    showFileError(file, error) {
        const notification = document.createElement('div');
        notification.className = 'upload-notification error';
        notification.innerHTML = `
            <div class="notification-icon">❌</div>
            <div class="notification-content">
                <div class="notification-title">Upload Error</div>
                <div class="notification-message">${file.name}: ${error}</div>
            </div>
            <button class="notification-close">×</button>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);

        // Close button
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });
    }

    showUrlPasteModal() {
        const modal = document.createElement('div');
        modal.className = 'modal upload-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>🔗 Upload from URL</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="url-input-group">
                        <input type="url" id="videoUrl" placeholder="https://example.com/video.mp4" class="url-input">
                        <button class="url-fetch-btn" id="fetchUrlBtn">Fetch</button>
                    </div>
                    <div class="url-info" id="urlInfo" style="display: none;">
                        <div class="url-preview" id="urlPreview"></div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn secondary-btn" id="cancelUrlBtn">Cancel</button>
                    <button class="btn primary-btn" id="downloadUrlBtn" disabled>Download & Upload</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Event listeners
        modal.querySelector('.modal-close').addEventListener('click', () => modal.remove());
        modal.querySelector('#cancelUrlBtn').addEventListener('click', () => modal.remove());
        
        // URL fetch functionality would go here
        modal.querySelector('#fetchUrlBtn').addEventListener('click', () => {
            // Implement URL fetching logic
            console.log('URL fetch not implemented in demo');
        });
    }

    initializeTouchOptimizations() {
        // Add touch-specific CSS classes
        if ('ontouchstart' in window) {
            document.body.classList.add('touch-device');
        }

        // Optimize for touch interactions
        const style = document.createElement('style');
        style.textContent = `
            @media (max-width: 768px) {
                .netflix-btn {
                    min-height: 44px;
                    font-size: 16px;
                }
                
                .upload-item-header {
                    padding: 16px;
                }
                
                .control-btn {
                    min-width: 44px;
                    min-height: 44px;
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// Initialize upload manager when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (typeof window.uploadManager === 'undefined') {
        window.uploadManager = new NetflixLevelUploadManager();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NetflixLevelUploadManager;
}
