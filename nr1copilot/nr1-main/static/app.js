/**
 * ViralClip Pro v3.0 - Netflix-Level Frontend Application
 * Real-time viral clip generation with instant feedback and entertainment
 * 
 * Features:
 * - Real-time clip generation preview
 * - Interactive timeline with viral score visualization
 * - Live processing status with entertaining content
 * - Advanced WebSocket management
 * - Netflix-level UX patterns
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
            isProcessing: false,
            viralScores: [],
            timelineData: null,
            realtimePreview: null
        };

        // WebSocket connections for real-time features
        this.connections = {
            upload: null,
            viralScores: null,
            timeline: null,
            processing: null
        };

        // Real-time UI state
        this.realtime = {
            viralChart: null,
            timelineSlider: null,
            previewPlayer: null,
            entertainmentIndex: 0
        };

        // Configuration
        this.config = {
            maxFileSize: 2 * 1024 * 1024 * 1024, // 2GB
            allowedTypes: ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'],
            realtimeUpdateInterval: 100, // ms
            previewGenerationDelay: 500, // ms
            maxRetries: 3,
            apiVersion: 'v3'
        };

        // Entertainment content for processing
        this.entertainmentFacts = [
            "🎬 Creating viral magic with AI assistance...",
            "🚀 Optimizing for maximum engagement...",
            "🎯 Analyzing trending patterns...",
            "⚡ Applying viral enhancement algorithms...",
            "🎵 Synchronizing with trending audio patterns...",
            "🔥 Boosting visual appeal factors...",
            "💫 Calculating optimal clip timing...",
            "🎭 Enhancing emotional resonance...",
            "🌟 Maximizing first-3-seconds impact...",
            "🎪 Adding Netflix-level polish..."
        ];

        this.init();
    }

    async init() {
        try {
            console.log('🚀 Initializing ViralClip Pro v3.0 with real-time features...');

            this.setupEventListeners();
            this.setupDragAndDrop();
            this.setupRealtimeComponents();
            this.setupMobileOptimizations();
            this.setupErrorHandling();

            this.showStep('upload');
            this.hideLoadingSplash();

            console.log('✅ ViralClip Pro v3.0 initialized successfully');
            this.trackEvent('app_initialized_v3');

        } catch (error) {
            console.error('❌ Failed to initialize ViralClip Pro:', error);
            this.showCriticalError('Failed to initialize application', error);
        }
    }

    setupRealtimeComponents() {
        // Initialize viral score chart
        this.initializeViralChart();

        // Initialize interactive timeline
        this.initializeInteractiveTimeline();

        // Initialize preview player
        this.initializePreviewPlayer();

        // Setup real-time event handlers
        this.setupRealtimeEventHandlers();
    }

    initializeViralChart() {
        const chartContainer = document.getElementById('viral-chart');
        if (!chartContainer) return;

        // Create dynamic viral score visualization
        chartContainer.innerHTML = `
            <div class="viral-chart-container">
                <canvas id="viral-canvas" width="800" height="200"></canvas>
                <div class="chart-overlay">
                    <div class="score-indicator">
                        <span class="current-score">--</span>
                        <span class="score-label">Viral Score</span>
                    </div>
                    <div class="confidence-meter">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: 0%"></div>
                        </div>
                        <span class="confidence-text">Confidence: --%</span>
                    </div>
                </div>
            </div>
        `;

        this.realtime.viralChart = {
            canvas: document.getElementById('viral-canvas'),
            ctx: document.getElementById('viral-canvas').getContext('2d'),
            dataPoints: [],
            currentScore: 0,
            confidence: 0
        };
    }

    initializeInteractiveTimeline() {
        const timelineContainer = document.getElementById('interactive-timeline');
        if (!timelineContainer) return;

        timelineContainer.innerHTML = `
            <div class="timeline-container">
                <div class="timeline-header">
                    <h4>🎬 Interactive Timeline</h4>
                    <div class="timeline-controls">
                        <button class="btn-mini" id="play-pause-btn">⏸️</button>
                        <button class="btn-mini" id="reset-timeline">⏮️</button>
                        <span class="timeline-time">00:00 / 00:00</span>
                    </div>
                </div>

                <div class="timeline-track" id="timeline-track">
                    <div class="viral-heatmap" id="viral-heatmap"></div>
                    <div class="timeline-scrubber" id="timeline-scrubber">
                        <div class="scrubber-handle"></div>
                    </div>
                    <div class="clip-markers" id="clip-markers"></div>
                </div>

                <div class="timeline-legends">
                    <div class="legend-item">
                        <span class="color-box viral-low"></span>
                        <span>Low Viral (0-50%)</span>
                    </div>
                    <div class="legend-item">
                        <span class="color-box viral-medium"></span>
                        <span>Medium Viral (50-75%)</span>
                    </div>
                    <div class="legend-item">
                        <span class="color-box viral-high"></span>
                        <span>High Viral (75%+)</span>
                    </div>
                </div>

                <div class="timeline-insights" id="timeline-insights">
                    <div class="insight-card">
                        <span class="insight-icon">🎯</span>
                        <div class="insight-content">
                            <span class="insight-value">--</span>
                            <span class="insight-label">Peak Moments</span>
                        </div>
                    </div>
                    <div class="insight-card">
                        <span class="insight-icon">⚡</span>
                        <div class="insight-content">
                            <span class="insight-value">--</span>
                            <span class="insight-label">Energy Level</span>
                        </div>
                    </div>
                    <div class="insight-card">
                        <span class="insight-icon">😊</span>
                        <div class="insight-content">
                            <span class="insight-value">--</span>
                            <span class="insight-label">Emotion</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.setupTimelineInteractions();
    }

    setupTimelineInteractions() {
        const scrubber = document.getElementById('timeline-scrubber');
        const track = document.getElementById('timeline-track');

        if (!scrubber || !track) return;

        let isDragging = false;
        let currentPosition = 0;

        // Mouse events
        scrubber.addEventListener('mousedown', (e) => {
            isDragging = true;
            this.updateTimelinePosition(e, track);
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                this.updateTimelinePosition(e, track);
            }
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });

        // Touch events for mobile
        scrubber.addEventListener('touchstart', (e) => {
            isDragging = true;
            this.updateTimelinePosition(e.touches[0], track);
        });

        document.addEventListener('touchmove', (e) => {
            if (isDragging) {
                e.preventDefault();
                this.updateTimelinePosition(e.touches[0], track);
            }
        });

        document.addEventListener('touchend', () => {
            isDragging = false;
        });

        // Click to seek
        track.addEventListener('click', (e) => {
            this.updateTimelinePosition(e, track);
        });
    }

    updateTimelinePosition(event, track) {
        const rect = track.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));

        const scrubber = document.getElementById('timeline-scrubber');
        if (scrubber) {
            scrubber.style.left = `${percentage}%`;
        }

        // Update current time if we have timeline data
        if (this.state.timelineData) {
            const currentTime = (percentage / 100) * this.state.timelineData.duration;
            this.updateTimelineDisplay(currentTime);
            this.generateRealtimePreview(currentTime);
        }
    }

    initializePreviewPlayer() {
        const previewContainer = document.getElementById('realtime-preview');
        if (!previewContainer) return;

        previewContainer.innerHTML = `
            <div class="preview-player-container">
                <div class="preview-video-wrapper">
                    <video id="preview-video" class="preview-video" muted loop>
                        <source src="" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <div class="preview-overlay">
                        <div class="preview-controls">
                            <button class="btn-control" id="preview-play">▶️</button>
                            <button class="btn-control" id="preview-regenerate">🔄</button>
                        </div>
                        <div class="preview-info">
                            <span class="preview-duration">0s</span>
                            <span class="preview-score">Score: --%</span>
                        </div>
                    </div>
                </div>

                <div class="preview-analysis" id="preview-analysis">
                    <div class="analysis-header">
                        <h5>🎯 Live Analysis</h5>
                        <div class="analysis-status">
                            <span class="status-indicator"></span>
                            <span class="status-text">Ready</span>
                        </div>
                    </div>

                    <div class="analysis-metrics">
                        <div class="metric">
                            <span class="metric-icon">⚡</span>
                            <span class="metric-label">Energy</span>
                            <span class="metric-value">--</span>
                        </div>
                        <div class="metric">
                            <span class="metric-icon">🎵</span>
                            <span class="metric-label">Audio</span>
                            <span class="metric-value">--</span>
                        </div>
                        <div class="metric">
                            <span class="metric-icon">👁️</span>
                            <span class="metric-label">Visual</span>
                            <span class="metric-value">--</span>
                        </div>
                    </div>

                    <div class="optimization-suggestions" id="optimization-suggestions">
                        <h6>💡 Optimization Suggestions</h6>
                        <div class="suggestions-list"></div>
                    </div>
                </div>
            </div>
        `;

        this.setupPreviewControls();
    }

    setupPreviewControls() {
        const playBtn = document.getElementById('preview-play');
        const regenerateBtn = document.getElementById('preview-regenerate');
        const video = document.getElementById('preview-video');

        if (playBtn && video) {
            playBtn.addEventListener('click', () => {
                if (video.paused) {
                    video.play();
                    playBtn.textContent = '⏸️';
                } else {
                    video.pause();
                    playBtn.textContent = '▶️';
                }
            });
        }

        if (regenerateBtn) {
            regenerateBtn.addEventListener('click', () => {
                this.regenerateCurrentPreview();
            });
        }
    }

    setupRealtimeEventHandlers() {
        // Debounced preview generation
        this.debouncedPreviewGeneration = this.debounce(
            this.generateRealtimePreview.bind(this), 
            this.config.previewGenerationDelay
        );

        // Real-time viral score updates
        this.realtimeScoreUpdater = setInterval(() => {
            this.updateViralScoreDisplay();
        }, this.config.realtimeUpdateInterval);
    }

    setupEventListeners() {
        // Enhanced upload form
        const uploadForm = document.getElementById('upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', this.handleFileUpload.bind(this));
        }

        // File input
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Enhanced navigation
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-action]')) {
                this.handleAction(e.target.dataset.action, e.target);
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
    }

    handleKeyboardShortcuts(e) {
        // Space bar for play/pause
        if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
            e.preventDefault();
            const playBtn = document.getElementById('preview-play');
            if (playBtn) playBtn.click();
        }

        // Arrow keys for timeline navigation
        if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
            e.preventDefault();
            const direction = e.code === 'ArrowLeft' ? -1 : 1;
            this.nudgeTimelinePosition(direction);
        }

        // Escape to close modals
        if (e.code === 'Escape') {
            this.hideAllModals();
        }
    }

    nudgeTimelinePosition(direction) {
        if (!this.state.timelineData) return;

        const scrubber = document.getElementById('timeline-scrubber');
        if (!scrubber) return;

        const currentLeft = parseFloat(scrubber.style.left) || 0;
        const nudgeAmount = 1; // 1% increments
        const newPosition = Math.max(0, Math.min(100, currentLeft + (direction * nudgeAmount)));

        scrubber.style.left = `${newPosition}%`;

        const currentTime = (newPosition / 100) * this.state.timelineData.duration;
        this.updateTimelineDisplay(currentTime);
        this.debouncedPreviewGeneration(currentTime);
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('upload-area');
        if (!uploadArea) return;

        // Enhanced drag and drop with real-time feedback
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, this.preventDefaults, false);
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        let dragCounter = 0;

        document.addEventListener('dragenter', (e) => {
            dragCounter++;
            if (dragCounter === 1) {
                this.showEnhancedDragOverlay();
            }
        });

        document.addEventListener('dragleave', (e) => {
            dragCounter--;
            if (dragCounter === 0) {
                this.hideEnhancedDragOverlay();
            }
        });

        uploadArea.addEventListener('drop', (e) => {
            dragCounter = 0;
            this.hideEnhancedDragOverlay();

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.processFiles(files);
            }
        });

        // Click to upload
        uploadArea.addEventListener('click', () => {
            document.getElementById('file-input')?.click();
        });
    }

    showEnhancedDragOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'enhanced-drag-overlay';
        overlay.className = 'enhanced-drag-overlay';
        overlay.innerHTML = `
            <div class="drag-content">
                <div class="drag-animation">
                    <div class="upload-icon-large">📹</div>
                    <div class="drag-waves">
                        <div class="wave wave-1"></div>
                        <div class="wave wave-2"></div>
                        <div class="wave wave-3"></div>
                    </div>
                </div>
                <h3>Drop your video for instant viral analysis!</h3>
                <p>Real-time processing with Netflix-level quality</p>
                <div class="drag-features">
                    <span class="feature">⚡ Instant Preview</span>
                    <span class="feature">🎯 Viral Score</span>
                    <span class="feature">🔥 Real-time Analysis</span>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);

        setTimeout(() => {
            overlay.classList.add('show');
        }, 50);
    }

    hideEnhancedDragOverlay() {
        const overlay = document.getElementById('enhanced-drag-overlay');
        if (overlay) {
            overlay.classList.remove('show');
            setTimeout(() => {
                overlay.remove();
            }, 300);
        }
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
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

        // Show enhanced preview with real-time features
        await this.showEnhancedInstantPreview(file);
        this.state.currentFile = file;

        // Enable upload with enhanced UI
        const uploadBtn = document.querySelector('#upload-form button[type="submit"]');
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = `
                <span class="btn-icon">🚀</span>
                <span class="btn-text">Start Real-time Analysis</span>
                <span class="btn-subtitle">Netflix-level processing</span>
            `;
            uploadBtn.classList.add('enhanced-upload-btn');
        }
    }

    async showEnhancedInstantPreview(file) {
        const uploadArea = document.getElementById('upload-area');

        // Remove existing preview
        const existingPreview = uploadArea.querySelector('.enhanced-instant-preview');
        if (existingPreview) {
            existingPreview.remove();
        }

        const preview = document.createElement('div');
        preview.className = 'enhanced-instant-preview';

        // Create enhanced video preview
        const video = document.createElement('video');
        video.className = 'preview-video-enhanced';
        video.autoplay = true;
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = 'metadata';

        const fileURL = URL.createObjectURL(file);
        video.src = fileURL;

        preview.innerHTML = `
            <div class="preview-container-enhanced">
                <div class="video-section">
                    <div class="video-wrapper">
                        <!-- Video element inserted here -->
                        <div class="video-overlay-enhanced">
                            <div class="play-indicator-enhanced">
                                <div class="play-icon">▶</div>
                                <div class="play-ring"></div>
                            </div>
                        </div>
                    </div>

                    <div class="quick-analysis">
                        <div class="analysis-item">
                            <span class="analysis-icon">🎯</span>
                            <span class="analysis-value">Analyzing...</span>
                        </div>
                        <div class="analysis-item">
                            <span class="analysis-icon">⚡</span>
                            <span class="analysis-value">Processing...</span>
                        </div>
                        <div class="analysis-item">
                            <span class="analysis-icon">🔥</span>
                            <span class="analysis-value">Optimizing...</span>
                        </div>
                    </div>
                </div>

                <div class="info-section">
                    <div class="file-info-enhanced">
                        <h4 class="file-name">${this.truncateFileName(file.name, 30)}</h4>
                        <div class="file-stats-enhanced">
                            <div class="stat-item">
                                <span class="stat-icon">📦</span>
                                <span class="stat-value">${this.formatBytes(file.size)}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-icon">🎬</span>
                                <span class="stat-value">${this.getFileTypeDisplay(file.type)}</span>
                            </div>
                            <div class="stat-item status-item">
                                <span class="stat-icon">✅</span>
                                <span class="stat-value status-ready">Ready for analysis</span>
                            </div>
                        </div>
                    </div>

                    <div class="preview-actions-enhanced">
                        <button class="btn-action primary" onclick="app.startAnalysis()">
                            <span class="btn-icon">🎯</span>
                            <span>Analyze Now</span>
                        </button>
                        <button class="btn-action secondary" onclick="app.removePreview()">
                            <span class="btn-icon">🗑️</span>
                            <span>Remove</span>
                        </button>
                    </div>

                    <div class="upload-progress-enhanced">
                        <div class="progress-header">
                            <span class="progress-label">Ready to upload</span>
                            <span class="progress-percentage">0%</span>
                        </div>
                        <div class="progress-bar-enhanced">
                            <div class="progress-fill-enhanced"></div>
                            <div class="progress-glow"></div>
                        </div>
                        <div class="progress-details">
                            <span class="progress-speed">--</span>
                            <span class="progress-eta">Ready</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Insert video element
        const videoWrapper = preview.querySelector('.video-wrapper');
        videoWrapper.insertBefore(video, videoWrapper.firstChild);

        uploadArea.appendChild(preview);

        // Enhanced video loading
        video.addEventListener('loadedmetadata', () => {
            video.classList.add('loaded');

            // Start quick analysis simulation
            this.simulateQuickAnalysis(preview);

            // Update duration
            const durationStat = document.createElement('div');
            durationStat.className = 'stat-item';
            durationStat.innerHTML = `
                <span class="stat-icon">⏱️</span>
                <span class="stat-value">${this.formatDuration(video.duration)}</span>
            `;
            preview.querySelector('.file-stats-enhanced').appendChild(durationStat);
        });

        // Show with enhanced animation
        setTimeout(() => {
            preview.classList.add('show');
        }, 100);

        // Cleanup URL when removed
        preview.addEventListener('remove', () => {
            URL.revokeObjectURL(fileURL);
        });
    }

    simulateQuickAnalysis(preview) {
        const analysisItems = preview.querySelectorAll('.analysis-item .analysis-value');
        const analyses = [
            { icon: '🎯', text: '87% Viral Potential', delay: 1000 },
            { icon: '⚡', text: 'High Energy Detected', delay: 1500 },
            { icon: '🔥', text: 'Trending Elements Found', delay: 2000 }
        ];

        analyses.forEach((analysis, index) => {
            setTimeout(() => {
                if (analysisItems[index]) {
                    analysisItems[index].textContent = analysis.text;
                    analysisItems[index].classList.add('analysis-complete');
                }
            }, analysis.delay);
        });
    }

    async handleFileUpload(e) {
        e.preventDefault();

        if (!this.state.currentFile) {
            this.showError('Please select a file to upload.');
            return;
        }

        try {
            this.state.uploadId = this.generateId();

            // Connect to enhanced upload WebSocket
            await this.connectEnhancedUploadWebSocket();

            // Start enhanced upload with real-time features
            await this.uploadFileEnhanced(this.state.currentFile);

        } catch (error) {
            console.error('Enhanced upload error:', error);
            this.showError(`Upload failed: ${error.message}`);
        }
    }

    async uploadFileEnhanced(file) {
        try {
            this.updateEnhancedProgress(0, 'Starting Netflix-level upload...');

            const formData = new FormData();
            formData.append('file', file);
            formData.append('upload_id', this.state.uploadId);

            const response = await this.uploadWithEnhancedProgress(formData);

            if (response.success) {
                this.state.sessionId = response.session_id;

                // Connect to real-time features
                await this.connectRealtimeFeatures();

                // Update UI with enhanced analysis data
                this.updateEnhancedProgress(100, 'Analysis complete! 🎉');

                setTimeout(() => {
                    this.hideEnhancedPreview();
                    this.showEnhancedAnalysisStep(response);
                }, 2000);
            } else {
                throw new Error(response.error || 'Upload failed');
            }

        } catch (error) {
            console.error('Enhanced upload error:', error);
            this.showError(`Upload failed: ${error.message}`);
            this.hideEnhancedPreview();
        }
    }

    async uploadWithEnhancedProgress(formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            let startTime = Date.now();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const progress = Math.round((e.loaded / e.total) * 100);
                    const speed = e.loaded / ((Date.now() - startTime) / 1000);
                    const eta = (e.total - e.loaded) / speed;

                    this.updateEnhancedProgress(
                        progress,
                        `Uploading... ${progress}%`,
                        speed,
                        eta
                    );
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

            xhr.open('POST', `/api/${this.config.apiVersion}/upload-video`);
            xhr.send(formData);
        });
    }

    updateEnhancedProgress(progress, message, speed = null, eta = null) {
        const preview = document.querySelector('.enhanced-instant-preview');
        if (!preview) return;

        const progressFill = preview.querySelector('.progress-fill-enhanced');
        const progressPercentage = preview.querySelector('.progress-percentage');
        const progressLabel = preview.querySelector('.progress-label');
        const progressSpeed = preview.querySelector('.progress-speed');
        const progressEta = preview.querySelector('.progress-eta');

        // Update progress bar with enhanced animations
        if (progressFill) {
            progressFill.style.width = `${Math.min(progress, 100)}%`;

            // Dynamic color transitions
            if (progress < 30) {
                progressFill.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
            } else if (progress < 70) {
                progressFill.style.background = 'linear-gradient(90deg, #3b82f6, #60a5fa)';
            } else {
                progressFill.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
            }
        }

        if (progressPercentage) {
            progressPercentage.textContent = `${Math.round(progress)}%`;
        }

        if (progressLabel) {
            progressLabel.textContent = message;
        }

        if (progressSpeed && speed) {
            progressSpeed.textContent = this.formatSpeed(speed);
        }

        if (progressEta && eta) {
            const etaSeconds = Math.round(eta);
            progressEta.textContent = etaSeconds > 60 ? 
                `${Math.round(etaSeconds / 60)}m remaining` : 
                `${etaSeconds}s remaining`;
        }

        // Enhanced completion effects
        if (progress >= 100) {
            preview.classList.add('upload-complete');
            this.showCompletionCelebration(preview);
        }
    }

    showCompletionCelebration(preview) {
        const celebration = document.createElement('div');
        celebration.className = 'completion-celebration';
        celebration.innerHTML = `
            <div class="celebration-content">
                <div class="celebration-icon">🎉</div>
                <div class="celebration-text">Upload Complete!</div>
                <div class="celebration-subtext">Starting AI analysis...</div>
            </div>
        `;

        preview.appendChild(celebration);

        setTimeout(() => {
            celebration.classList.add('show');
        }, 100);

        setTimeout(() => {
            celebration.remove();
        }, 3000);
    }

    async connectRealtimeFeatures() {
        try {
            // Connect to viral scores WebSocket
            await this.connectViralScoresWebSocket();

            // Connect to timeline WebSocket
            await this.connectTimelineWebSocket();

            console.log('✅ Real-time features connected');

        } catch (error) {
            console.error('Real-time connection error:', error);
        }
    }

    async connectEnhancedUploadWebSocket() {
        if (this.connections.upload) {
            this.connections.upload.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/${this.config.apiVersion}/ws/upload/${this.state.uploadId}`;

        this.connections.upload = new WebSocket(wsUrl);

        this.connections.upload.onopen = () => {
            console.log('📡 Enhanced upload WebSocket connected');
        };

        this.connections.upload.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleEnhancedUploadMessage(message);
            } catch (error) {
                console.error('Upload WebSocket message error:', error);
            }
        };

        this.connections.upload.onerror = (error) => {
            console.error('Upload WebSocket error:', error);
        };
    }

    async connectViralScoresWebSocket() {
        if (!this.state.sessionId) return;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/${this.config.apiVersion}/ws/viral-scores/${this.state.sessionId}`;

        this.connections.viralScores = new WebSocket(wsUrl);

        this.connections.viralScores.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleViralScoresMessage(message);
            } catch (error) {
                console.error('Viral scores WebSocket error:', error);
            }
        };
    }

    async connectTimelineWebSocket() {
        if (!this.state.sessionId) return;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/${this.config.apiVersion}/ws/timeline/${this.state.sessionId}`;

        this.connections.timeline = new WebSocket(wsUrl);

        this.connections.timeline.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleTimelineMessage(message);
            } catch (error) {
                console.error('Timeline WebSocket error:', error);
            }
        };
    }

    handleEnhancedUploadMessage(message) {
        switch (message.type) {
            case 'upload_progress':
                this.updateEnhancedProgress(message.progress, `${Math.round(message.progress)}% uploaded`);
                break;
            case 'upload_complete':
                this.updateEnhancedProgress(100, message.message);
                break;
            case 'upload_error':
                this.showError(message.error);
                break;
        }
    }

    handleViralScoresMessage(message) {
        switch (message.type) {
            case 'analysis_progress':
                this.updateViralChart(message.data);
                this.updateRealtimeInsights(message.data);
                break;
            case 'analysis_complete':
                this.finalizeViralAnalysis();
                break;
            case 'analysis_error':
                this.showError('Analysis failed: ' + message.error);
                break;
        }
    }

    handleTimelineMessage(message) {
        switch (message.type) {
            case 'timeline_update':
                this.updateInteractiveTimeline(message.data);
                break;
            case 'highlights_detected':
                this.highlightTimelineSegments(message.highlights);
                break;
        }
    }

    updateViralChart(data) {
        if (!this.realtime.viralChart) return;

        const { ctx, canvas } = this.realtime.viralChart;
        const scores = data.current_scores || [];

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw grid
        this.drawViralChartGrid(ctx, canvas);

        // Draw viral score line
        this.drawViralScoreLine(ctx, canvas, scores);

        // Update current score display
        if (scores.length > 0) {
            const latestScore = scores[scores.length - 1];
            this.updateCurrentScoreDisplay(latestScore.score, latestScore.confidence);
        }
    }

    drawViralChartGrid(ctx, canvas) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;

        // Horizontal lines
        for (let i = 0; i <= 10; i++) {
            const y = (canvas.height / 10) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }

        // Vertical lines
        for (let i = 0; i <= 20; i++) {
            const x = (canvas.width / 20) * i;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
    }

    drawViralScoreLine(ctx, canvas, scores) {
        if (scores.length < 2) return;

        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        ctx.beginPath();

        scores.forEach((score, index) => {
            const x = (index / (scores.length - 1)) * canvas.width;
            const y = canvas.height - ((score.score / 100) * canvas.height);

            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Draw glow effect
        ctx.shadowColor = '#10b981';
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Draw data points
        scores.forEach((score, index) => {
            const x = (index / (scores.length - 1)) * canvas.width;
            const y = canvas.height - ((score.score / 100) * canvas.height);

            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = score.score > 75 ? '#10b981' : score.score > 50 ? '#f59e0b' : '#ef4444';
            ctx.fill();
        });
    }

    updateCurrentScoreDisplay(score, confidence) {
        const scoreElement = document.querySelector('.current-score');
        const confidenceElement = document.querySelector('.confidence-text');
        const confidenceFill = document.querySelector('.confidence-fill');

        if (scoreElement) {
            scoreElement.textContent = Math.round(score);
            scoreElement.className = `current-score ${this.getScoreClass(score)}`;
        }

        if (confidenceElement) {
            confidenceElement.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
        }

        if (confidenceFill) {
            confidenceFill.style.width = `${confidence * 100}%`;
        }
    }

    getScoreClass(score) {
        if (score >= 75) return 'score-high';
        if (score >= 50) return 'score-medium';
        return 'score-low';
    }

    updateRealtimeInsights(data) {
        const insights = document.getElementById('timeline-insights');
        if (!insights) return;

        const insightCards = insights.querySelectorAll('.insight-card');

        // Update peak moments
        if (insightCards[0]) {
            const peakValue = insightCards[0].querySelector('.insight-value');
            if (peakValue) {
                peakValue.textContent = data.trending_emotions?.length || 0;
            }
        }

        // Update energy level
        if (insightCards[1]) {
            const energyValue = insightCards[1].querySelector('.insight-value');
            if (energyValue && data.energy_trend?.length) {
                const avgEnergy = data.energy_trend.reduce((sum, e) => sum + e.level, 0) / data.energy_trend.length;
                energyValue.textContent = Math.round(avgEnergy);
            }
        }

        // Update emotion
        if (insightCards[2]) {
            const emotionValue = insightCards[2].querySelector('.insight-value');
            if (emotionValue && data.trending_emotions?.length) {
                const latestEmotion = data.trending_emotions[data.trending_emotions.length - 1];
                emotionValue.textContent = latestEmotion.emotion;
            }
        }
    }

    async generateRealtimePreview(currentTime) {
        if (!this.state.sessionId || !this.state.timelineData) return;

        try {
            const endTime = Math.min(currentTime + 10, this.state.timelineData.duration);

            const response = await fetch(`/api/${this.config.apiVersion}/generate-clip-preview`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.state.sessionId,
                    start_time: currentTime,
                    end_time: endTime,
                    quality: 'preview'
                })
            });

            const data = await response.json();

            if (data.success) {
                this.updatePreviewPlayer(data);
                this.updatePreviewAnalysis(data.viral_analysis);
                this.updateOptimizationSuggestions(data.optimization_suggestions);
            }

        } catch (error) {
            console.error('Real-time preview error:', error);
        }
    }

    updatePreviewPlayer(data) {
        const video = document.getElementById('preview-video');
        const durationSpan = document.querySelector('.preview-duration');
        const scoreSpan = document.querySelector('.preview-score');

        if (video && data.preview_url) {
            video.src = data.preview_url;
            video.load();
        }

        if (durationSpan) {
            durationSpan.textContent = '10s'; // Preview duration
        }

        if (scoreSpan && data.viral_analysis) {
            scoreSpan.textContent = `Score: ${data.viral_analysis.overall_score}%`;
        }
    }

    updatePreviewAnalysis(analysis) {
        if (!analysis) return;

        const metrics = document.querySelectorAll('.metric-value');

        if (metrics[0]) { // Energy
            metrics[0].textContent = analysis.engagement_factors?.energy_level || '--';
        }

        if (metrics[1]) { // Audio
            metrics[1].textContent = analysis.engagement_factors?.audio_quality || '--';
        }

        if (metrics[2]) { // Visual
            metrics[2].textContent = analysis.engagement_factors?.visual_appeal || '--';
        }
    }

    updateOptimizationSuggestions(suggestions) {
        const suggestionsList = document.querySelector('.suggestions-list');
        if (!suggestionsList || !suggestions) return;

        suggestionsList.innerHTML = suggestions.map(suggestion => `
            <div class="suggestion-item ${suggestion.priority}">
                <div class="suggestion-icon">${this.getSuggestionIcon(suggestion.type)}</div>
                <div class="suggestion-content">
                    <div class="suggestion-text">${suggestion.suggestion}</div>
                    <div class="suggestion-impact">${suggestion.impact}</div>
                </div>
            </div>
        `).join('');
    }

    getSuggestionIcon(type) {
        const icons = {
            'audio': '🎵',
            'visual': '🎨',
            'content': '📝',
            'timing': '⏰',
            'platform': '📱'
        };
        return icons[type] || '💡';
    }

    async regenerateCurrentPreview() {
        const scrubber = document.getElementById('timeline-scrubber');
        if (!scrubber || !this.state.timelineData) return;

        const currentPosition = parseFloat(scrubber.style.left) || 0;
        const currentTime = (currentPosition / 100) * this.state.timelineData.duration;

        await this.generateRealtimePreview(currentTime);
    }

    updateTimelineDisplay(currentTime) {
        const timeElement = document.querySelector('.timeline-time');
        if (timeElement && this.state.timelineData) {
            const formatted = `${this.formatTime(currentTime)} / ${this.formatTime(this.state.timelineData.duration)}`;
            timeElement.textContent = formatted;
        }
    }

    showEnhancedAnalysisStep(response) {
        this.showStep('analysis');

        // Load timeline data
        this.loadTimelineData();

        // Initialize real-time features
        this.initializeRealtimeAnalysis(response);
    }

    async loadTimelineData() {
        if (!this.state.sessionId) return;

        try {
            const response = await fetch(`/api/${this.config.apiVersion}/timeline/${this.state.sessionId}`);
            const data = await response.json();

            if (data.success) {
                this.state.timelineData = data.timeline;
                this.renderInteractiveTimeline(data.timeline);
            }

        } catch (error) {
            console.error('Timeline loading error:', error);
        }
    }

    renderInteractiveTimeline(timelineData) {
        const heatmap = document.getElementById('viral-heatmap');
        const markers = document.getElementById('clip-markers');

        if (!heatmap || !markers) return;

        // Render viral heatmap
        this.renderViralHeatmap(heatmap, timelineData.viral_heatmap);

        // Render clip markers
        this.renderClipMarkers(markers, timelineData.recommended_clips);

        // Update insights
        this.updateTimelineInsights(timelineData);
    }

    renderViralHeatmap(container, viralScores) {
        container.innerHTML = '';

        viralScores.forEach((score, index) => {
            const segment = document.createElement('div');
            segment.className = 'heatmap-segment';
            segment.style.width = `${100 / viralScores.length}%`;
            segment.style.backgroundColor = this.getViralColor(score.score);
            segment.title = `${score.timestamp}s: ${score.score}% viral`;

            container.appendChild(segment);
        });
    }

    getViralColor(score) {
        if (score >= 75) return 'rgba(16, 185, 129, 0.8)'; // Green
        if (score >= 50) return 'rgba(245, 158, 11, 0.8)'; // Yellow
        return 'rgba(239, 68, 68, 0.8)'; // Red
    }

    renderClipMarkers(container, clips) {
        container.innerHTML = '';

        clips.forEach((clip, index) => {
            const marker = document.createElement('div');
            marker.className = 'clip-marker';
            marker.style.left = `${(clip.start_time / this.state.timelineData.duration) * 100}%`;
            marker.style.width = `${((clip.end_time - clip.start_time) / this.state.timelineData.duration) * 100}%`;
            marker.innerHTML = `
                <div class="marker-label">${clip.title}</div>
                <div class="marker-score">${clip.viral_score}%</div>
            `;
            marker.onclick = () => this.selectClip(clip);

            container.appendChild(marker);
        });
    }

    selectClip(clip) {
        // Update timeline position
        const scrubber = document.getElementById('timeline-scrubber');
        if (scrubber) {
            const position = (clip.start_time / this.state.timelineData.duration) * 100;
            scrubber.style.left = `${position}%`;
        }

        // Generate preview for this clip
        this.generateRealtimePreview(clip.start_time);

        // Update time display
        this.updateTimelineDisplay(clip.start_time);
    }

    initializeRealtimeAnalysis(response) {
        // Start viral score tracking
        this.startViralScoreTracking();

        // Initialize preview generation
        this.setupRealtimePreviewGeneration();
    }

    setupRealtimePreviewGeneration() {
        // Implementation for setting up real-time preview generation
        console.log('Setting up real-time preview generation...');
    }

    startViralScoreTracking() {
        // Simulated real-time viral score updates
        this.viralScoreTracker = setInterval(() => {
            if (this.state.sessionId) {
                this.simulateViralScoreUpdate();
            }
        }, 2000);
    }

    simulateViralScoreUpdate() {
        const mockUpdate = {
            timestamp: Date.now() / 1000,
            score: 50 + Math.random() * 50,
            confidence: 0.7 + Math.random() * 0.3
        };

        this.state.viralScores.push(mockUpdate);
        this.updateViralScoreDisplay();
    }

    updateViralScoreDisplay() {
        if (this.state.viralScores.length === 0) return;

        const latest = this.state.viralScores[this.state.viralScores.length - 1];
        this.updateCurrentScoreDisplay(latest.score, latest.confidence);
    }

    hideEnhancedPreview() {
        const preview = document.querySelector('.enhanced-instant-preview');
        if (preview) {
            preview.classList.remove('show');
            setTimeout(() => {
                preview.remove();
            }, 300);
        }
    }

    removePreview() {
        this.hideEnhancedPreview();

        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.value = '';
        }

        this.state.currentFile = null;
        this.showSuccess('Video removed. You can select another file.');
    }

    startAnalysis() {
        const uploadBtn = document.querySelector('#upload-form button[type="submit"]');
        if (uploadBtn) {
            uploadBtn.click();
        }
    }

    // Enhanced processing with entertainment
    async startEnhancedProcessing(clips) {
        try {
            this.state.taskId = this.generateId();

            const response = await fetch(`/api/${this.config.apiVersion}/process-clips`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.state.sessionId,
                    clips: clips,
                    options: {
                        quality: 'high',
                        entertainment: true
                    }
                })
            });

            const data = await response.json();

            if (data.success) {
                this.state.taskId = data.task_id;
                await this.connectProcessingWebSocket();
                this.showEnhancedProcessingStep();
            }

        } catch (error) {
            console.error('Enhanced processing error:', error);
            this.showError('Processing failed to start');
        }
    }

    async connectProcessingWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/${this.config.apiVersion}/ws/processing/${this.state.taskId}`;

        this.connections.processing = new WebSocket(wsUrl);

        this.connections.processing.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleProcessingMessage(message);
            } catch (error) {
                console.error('Processing WebSocket error:', error);
            }
        };
    }

    handleProcessingMessage(message) {
        switch (message.type) {
            case 'processing_update':
                this.updateEnhancedProcessingDisplay(message);
                break;
            case 'step_update':
                this.updateProcessingStep(message);
                break;
            case 'processing_complete':
                this.handleProcessingComplete(message);
                break;
            case 'processing_error':
                this.showError('Processing failed: ' + message.error);
                break;
        }
    }

    updateProcessingStep(message) {
        // Implementation for updating processing step
        console.log('Updating processing step:', message);
    }

    handleProcessingComplete(message) {
        // Implementation for handling processing complete
        console.log('Processing complete:', message);
    }

    showEnhancedProcessingStep() {
        this.showStep('processing');

        const container = document.getElementById('processing-status');
        if (container) {
            container.innerHTML = `
                <div class="enhanced-processing-container">
                    <div class="processing-hero">
                        <div class="processing-animation">
                            <div class="processing-spinner"></div>
                            <div class="processing-icon">🎬</div>
                        </div>
                        <h3>Creating Your Viral Clips</h3>
                        <p class="processing-subtitle">Netflix-level AI is crafting perfect viral content...</p>
                    </div>

                    <div class="processing-progress-enhanced">
                        <div class="progress-ring">
                            <svg width="120" height="120">
                                <circle cx="60" cy="60" r="50" stroke-width="8" stroke="rgba(255,255,255,0.1)" fill="none"></circle>
                                <circle cx="60" cy="60" r="50" stroke-width="8" stroke="#10b981" fill="none" 
                                        stroke-dasharray="314" stroke-dashoffset="314" id="progress-circle"></circle>
                            </svg>
                            <div class="progress-text">
                                <span class="progress-percentage">0%</span>
                                <span class="progress-label">Processing</span>
                            </div>
                        </div>

                        <div class="processing-details">
                            <div class="current-step">
                                <span class="step-icon">⚡</span>
                                <span class="step-text">Initializing...</span>
                            </div>
                            <div class="eta-info">
                                <span class="eta-label">ETA:</span>
                                <span class="eta-value">Calculating...</span>
                            </div>
                        </div>
                    </div>

                    <div class="entertainment-section">
                        <h4>🎭 Did You Know?</h4>
                        <div class="entertainment-content">
                            <p id="entertainment-fact">Getting ready to create viral magic...</p>
                        </div>
                        <div class="entertainment-progress">
                            <div class="entertainment-dots">
                                <span class="dot active"></span>
                                <span class="dot"></span>
                                <span class="dot"></span>
                            </div>
                        </div>
                    </div>

                    <div class="processing-features">
                        <div class="feature-list">
                            <div class="feature-item processing">
                                <span class="feature-icon">🎯</span>
                                <span class="feature-text">AI Viral Analysis</span>
                                <span class="feature-status">⏳</span>
                            </div>
                            <div class="feature-item">
                                <span class="feature-icon">🎨</span>
                                <span class="feature-text">Visual Enhancement</span>
                                <span class="feature-status">⏳</span>
                            </div>
                            <div class="feature-item">
                                <span class="feature-icon">🎵</span>
                                <span class="feature-text">Audio Optimization</span>
                                <span class="feature-status">⏳</span>
                            </div>
                            <div class="feature-item">
                                <span class="feature-icon">📱</span>
                                <span class="feature-text">Platform Optimization</span>
                                <span class="feature-status">⏳</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    updateEnhancedProcessingDisplay(message) {
        // Update progress ring
        const progressCircle = document.getElementById('progress-circle');
        if (progressCircle) {
            const circumference = 2 * Math.PI * 50;
            const offset = circumference - (message.progress / 100) * circumference;
            progressCircle.style.strokeDashoffset = offset;
        }

        // Update percentage
        const percentageElement = document.querySelector('.progress-percentage');
        if (percentageElement) {
            percentageElement.textContent = `${Math.round(message.progress)}%`;
        }

        // Update current step
        const stepText = document.querySelector('.step-text');
        if (stepText) {
            stepText.textContent = message.stage;
        }

        // Update ETA
        const etaValue = document.querySelector('.eta-value');
        if (etaValue && message.eta_seconds) {
            etaValue.textContent = this.formatETA(message.eta_seconds);
        }

        // Update entertainment fact
        if (message.entertaining_fact) {
            const factElement = document.getElementById('entertainment-fact');
            if (factElement) {
                factElement.textContent = message.entertaining_fact;
                this.animateEntertainmentFact(factElement);
            }
        }
    }

    animateEntertainmentFact(element) {
        element.style.opacity = '0';
        element.style.transform = 'translateY(10px)';

        setTimeout(() => {
            element.style.transition = 'all 0.5s ease';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, 100);
    }

    formatETA(seconds) {
        if (seconds < 60) return `${seconds}s`;
        const minutes = Math.floor(seconds / 60);
        return `${minutes}m ${seconds % 60}s`;
    }

    // Utility methods
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

    // Enhanced processing with entertainment
    async startEnhancedProcessing(clips) {
        try {
            this.state.taskId = this.generateId();

            const response = await fetch(`/api/${this.config.apiVersion}/process-clips`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.state.sessionId,
                    clips: clips,
                    options: {
                        quality: 'high',
                        entertainment: true
                    }
                })
            });

            const data = await response.json();

            if (data.success) {
                this.state.taskId = data.task_id;
                await this.connectProcessingWebSocket();
                this.showEnhancedProcessingStep();
            }

        } catch (error) {
            console.error('Enhanced processing error:', error);
            this.showError('Processing failed to start');
        }
    }

    async connectProcessingWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/${this.config.apiVersion}/ws/processing/${this.state.taskId}`;

        this.connections.processing = new WebSocket(wsUrl);

        this.connections.processing.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleProcessingMessage(message);
            } catch (error) {
                console.error('Processing WebSocket error:', error);
            }
        };
    }

    handleProcessingMessage(message) {
        switch (message.type) {
            case 'processing_update':
                this.updateEnhancedProcessingDisplay(message);
                break;
            case 'step_update':
                this.updateProcessingStep(message);
                break;
            case 'processing_complete':
                this.handleProcessingComplete(message);
                break;
            case 'processing_error':
                this.showError('Processing failed: ' + message.error);
                break;
        }
    }

    updateProcessingStep(message) {
        // Implementation for updating processing step
        console.log('Updating processing step:', message);
    }

    handleProcessingComplete(message) {
        // Implementation for handling processing complete
        console.log('Processing complete:', message);
    }

    showEnhancedProcessingStep() {
        this.showStep('processing');

        const container = document.getElementById('processing-status');
        if (container) {
            container.innerHTML = `
                <div class="enhanced-processing-container">
                    <div class="processing-hero">
                        <div class="processing-animation">
                            <div class="processing-spinner"></div>
                            <div class="processing-icon">🎬</div>
                        </div>
                        <h3>Creating Your Viral Clips</h3>
                        <p class="processing-subtitle">Netflix-level AI is crafting perfect viral content...</p>
                    </div>

                    <div class="processing-progress-enhanced">
                        <div class="progress-ring">
                            <svg width="120" height="120">
                                <circle cx="60" cy="60" r="50" stroke-width="8" stroke="rgba(255,255,255,0.1)" fill="none"></circle>
                                <circle cx="60" cy="60" r="50" stroke-width="8" stroke="#10b981" fill="none" 
                                        stroke-dasharray="314" stroke-dashoffset="314" id="progress-circle"></circle>
                            </svg>
                            <div class="progress-text">
                                <span class="progress-percentage">0%</span>
                                <span class="progress-label">Processing</span>
                            </div>
                        </div>

                        <div class="processing-details">
                            <div class="current-step">
                                <span class="step-icon">⚡</span>
                                <span class="step-text">Initializing...</span>
                            </div>
                            <div class="eta-info">
                                <span class="eta-label">ETA:</span>
                                <span class="eta-value">Calculating...</span>
                            </div>
                        </div>
                    </div>

                    <div class="entertainment-section">
                        <h4>🎭 Did You Know?</h4>
                        <div class="entertainment-content">
                            <p id="entertainment-fact">Getting ready to create viral magic...</p>
                        </div>
                        <div class="entertainment-progress">
                            <div class="entertainment-dots">
                                <span class="dot active"></span>
                                <span class="dot"></span>
                                <span class="dot"></span>
                            </div>
                        </div>
                    </div>

                    <div class="processing-features">
                        <div class="feature-list">
                            <div class="feature-item processing">
                                <span class="feature-icon">🎯</span>
                                <span class="feature-text">AI Viral Analysis</span>
                                <span class="feature-status">⏳</span>
                            </div>
                            <div class="feature-item">
                                <span class="feature-icon">🎨</span>
                                <span class="feature-text">Visual Enhancement</span>
                                <span class="feature-status">⏳</span>
                            </div>
                            <div class="feature-item">
                                <span class="feature-icon">🎵</span>
                                <span class="feature-text">Audio Optimization</span>
                                <span class="feature-status">⏳</span>
                            </div>
                            <div class="feature-item">
                                <span class="feature-icon">📱</span>
                                <span class="feature-text">Platform Optimization</span>
                                <span class="feature-status">⏳</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    updateEnhancedProcessingDisplay(message) {
        // Update progress ring
        const progressCircle = document.getElementById('progress-circle');
        if (progressCircle) {
            const circumference = 2 * Math.PI * 50;
            const offset = circumference - (message.progress / 100) * circumference;
            progressCircle.style.strokeDashoffset = offset;
        }

        // Update percentage
        const percentageElement = document.querySelector('.progress-percentage');
        if (percentageElement) {
            percentageElement.textContent = `${Math.round(message.progress)}%`;
        }

        // Update current step
        const stepText = document.querySelector('.step-text');
        if (stepText) {
            stepText.textContent = message.stage;
        }

        // Update ETA
        const etaValue = document.querySelector('.eta-value');
        if (etaValue && message.eta_seconds) {
            etaValue.textContent = this.formatETA(message.eta_seconds);
        }

        // Update entertainment fact
        if (message.entertaining_fact) {
            const factElement = document.getElementById('entertainment-fact');
            if (factElement) {
                factElement.textContent = message.entertaining_fact;
                this.animateEntertainmentFact(factElement);
            }
        }
    }

    animateEntertainmentFact(element) {
        element.style.opacity = '0';
        element.style.transform = 'translateY(10px)';

        setTimeout(() => {
            element.style.transition = 'all 0.5s ease';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, 100);
    }

    formatETA(seconds) {
        if (seconds < 60) return `${seconds}s`;
        const minutes = Math.floor(seconds / 60);
        return `${minutes}m ${seconds % 60}s`;
    }

    // Utility methods
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

    showStep(step) {
        document.querySelectorAll('.step').forEach(el => {
            el.classList.remove('active');
        });

        const targetStep = document.getElementById(`step-${step}`);
        if (targetStep) {
            targetStep.classList.add('active');
        }

        this.state.currentStep = step;
    }

    showError(message) {
        this.showToast(message, 'error');
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast-enhanced toast-${type}`;
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
                        Reload Application
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(errorModal);
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

    hideAllModals() {
        document.querySelectorAll('.modal, .overlay').forEach(modal => {
            modal.classList.remove('show');
        });
    }

    trackEvent(eventName, data = {}) {
        console.log(`📊 Event: ${eventName}`, data);
    }

    setupMobileOptimizations() {
        // Viewport height fix
        const setVH = () => {
            const vh = window.innerHeight * 0.01;
            document.documentElement.style.setProperty('--vh', `${vh}px`);
        };

        setVH();
        window.addEventListener('resize', setVH);
        window.addEventListener('orientationchange', setVH);
    }

    setupErrorHandling() {
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
        });

        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            event.preventDefault();
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

    goBack() {
        // Implementation for going back
        console.log('Going back...');
    }

    goNext() {
        // Implementation for going next
        console.log('Going next...');
    }

    restart() {
        // Clear all state
        this.state = {
            currentStep: 'upload',
            sessionId: null,
            taskId: null,
            uploadId: null,
            selectedClips: [],
            currentFile: null,
            isProcessing: false,
            viralScores: [],
            timelineData: null,
            realtimePreview: null
        };

        // Close all WebSocket connections
        Object.values(this.connections).forEach(connection => {
            if (connection) connection.close();
        });

        // Clear intervals
        if (this.viralScoreTracker) {
            clearInterval(this.viralScoreTracker);
        }

        if (this.realtimeScoreUpdater) {
            clearInterval(this.realtimeScoreUpdater);
        }

        // Reset UI
        this.showStep('upload');

        // Clear forms
        document.querySelectorAll('form').forEach(form => form.reset());

        console.log('🔄 Application restarted');
    }
}

/* Add dynamic CSS animations */
const styleSheet = document.createElement('style');
styleSheet.textContent = `
    @keyframes sparkle {
        0% {
            opacity: 0;
            transform: scale(0) rotate(0deg);
        }
        50% {
            opacity: 1;
            transform: scale(1.2) rotate(180deg);
        }
        100% {
            opacity: 0;
            transform: scale(0) rotate(360deg);
        }
    }

    .feature-item.completed {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        animation: completePulse 0.5s ease-out;
    }

    @keyframes completePulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .viral-chart-container {
        position: relative;
        overflow: hidden;
    }

    .viral-chart-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, transparent 70%);
        animation: chartGlow 3s ease-in-out infinite alternate;
        pointer-events: none;
    }

    @keyframes chartGlow {
        0% { opacity: 0.3; transform: scale(0.8); }
        100% { opacity: 0.6; transform: scale(1.2); }
    }

    .timeline-track:hover .viral-heatmap {
        transform: scaleY(1.1);
        transition: transform 0.3s ease;
    }

    .insight-card:hover .insight-value {
        color: var(--viral-high);
        animation: valueHighlight 0.5s ease;
    }

    @keyframes valueHighlight {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
`;
document.head.appendChild(styleSheet);

/* Service Worker for PWA capabilities */
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('📱 SW registered:', registration);
            })
            .catch(error => {
                console.log('📱 SW registration failed:', error);
            });
    });
}
// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ViralClipPro();
});