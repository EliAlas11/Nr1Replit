
/**
 * ViralClip Pro - Netflix-Level Frontend Application
 * Advanced video processing interface with real-time features
 * 
 * Features:
 * - Real-time processing updates via WebSocket
 * - Instant drag-drop upload with preview
 * - AI-powered clip selection
 * - Mobile-first responsive design
 * - Progressive Web App capabilities
 * - Advanced error handling and user feedback
 */

class ViralClipApp {
    constructor() {
        this.currentStep = 1;
        this.analysisData = null;
        this.selectedClips = [];
        this.processingTaskId = null;
        this.websocket = null;
        this.uploadProgress = 0;
        this.apiBase = window.location.origin;
        
        this.init();
    }

    async init() {
        console.log('🎬 Initializing ViralClip Pro - SendShort.ai Killer');
        
        this.setupEventListeners();
        this.setupDragDrop();
        this.setupTimeline();
        this.setupKeyboardShortcuts();
        this.setupServiceWorker();
        this.updateNavigationState();
        
        // Show welcome message
        this.showToast('🚀 ViralClip Pro Ready - Let\'s create viral content!', 'success');
    }

    // ====================
    // Event Listeners Setup
    // ====================
    
    setupEventListeners() {
        // File input
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (e.target.files[0]) {
                    this.handleFileUpload(e.target.files[0]);
                }
            });
        }

        // URL input with real-time validation
        const videoUrl = document.getElementById('videoUrl');
        if (videoUrl) {
            videoUrl.addEventListener('input', this.debounce(this.validateUrl.bind(this), 500));
            videoUrl.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.analyzeUrl();
                }
            });
        }

        // Window events
        window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
        window.addEventListener('online', () => this.showToast('✅ Connection restored', 'success'));
        window.addEventListener('offline', () => this.showToast('⚠️ Connection lost', 'warning'));
    }

    setupDragDrop() {
        const dropZone = document.getElementById('dropZone');
        if (!dropZone) return;

        // Prevent default behavior for all drag events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Add visual feedback
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('dragover');
            }, false);
        });

        // Handle file drop
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (this.validateFile(file)) {
                    this.handleFileUpload(file);
                } else {
                    this.showToast('❌ Invalid file type. Please upload a video file.', 'error');
                }
            }
        }, false);
    }

    setupTimeline() {
        const timeline = document.getElementById('timeline');
        const clipSelector = document.getElementById('clipSelector');

        if (!timeline || !clipSelector) return;

        let isDragging = false;
        let isResizing = false;
        let startX = 0;
        let startLeft = 0;
        let startWidth = 0;

        // Mouse down on clip selector
        clipSelector.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const rect = clipSelector.getBoundingClientRect();

            if (e.clientX > rect.right - 10) {
                // Resize mode
                isResizing = true;
                startWidth = rect.width;
            } else {
                // Drag mode
                isDragging = true;
                startLeft = parseFloat(clipSelector.style.left) || 0;
            }

            startX = e.clientX;
            document.body.style.userSelect = 'none';
        });

        // Mouse move
        document.addEventListener('mousemove', (e) => {
            if (!isDragging && !isResizing) return;

            const timelineRect = timeline.getBoundingClientRect();
            const deltaX = e.clientX - startX;
            const deltaPercent = (deltaX / timelineRect.width) * 100;

            if (isDragging) {
                const newLeft = Math.max(0, Math.min(85, startLeft + deltaPercent));
                clipSelector.style.left = `${newLeft}%`;
            } else if (isResizing) {
                const currentLeft = parseFloat(clipSelector.style.left) || 0;
                const maxWidth = 100 - currentLeft;
                const newWidth = Math.max(5, Math.min(maxWidth, (startWidth / timelineRect.width * 100) + deltaPercent));
                clipSelector.style.width = `${newWidth}%`;
            }

            this.updateClipTime();
        });

        // Mouse up
        document.addEventListener('mouseup', () => {
            isDragging = false;
            isResizing = false;
            document.body.style.userSelect = '';
        });

        // Timeline click
        timeline.addEventListener('click', (e) => {
            if (isDragging || isResizing) return;

            const rect = timeline.getBoundingClientRect();
            const percent = ((e.clientX - rect.left) / rect.width) * 100;
            const currentWidth = parseFloat(clipSelector.style.width) || 30;
            
            clipSelector.style.left = `${Math.max(0, Math.min(100 - currentWidth, percent - currentWidth/2))}%`;
            this.updateClipTime();
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Only handle shortcuts when not typing in inputs
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            switch (e.key) {
                case ' ':
                    e.preventDefault();
                    this.toggleVideoPlayback();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    this.previousStep();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.nextStep();
                    break;
                case 'Escape':
                    e.preventDefault();
                    this.hideLoading();
                    break;
            }
        });
    }

    async setupServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                await navigator.serviceWorker.register('/sw.js');
                console.log('✅ Service Worker registered');
            } catch (error) {
                console.warn('Service Worker registration failed:', error);
            }
        }
    }

    // ====================
    // File Handling
    // ====================

    validateFile(file) {
        const validTypes = ['video/mp4', 'video/mov', 'video/avi', 'video/mkv', 'video/webm'];
        const maxSize = 500 * 1024 * 1024; // 500MB

        if (!validTypes.includes(file.type)) {
            return false;
        }

        if (file.size > maxSize) {
            this.showToast('❌ File too large. Maximum size is 500MB.', 'error');
            return false;
        }

        return true;
    }

    async handleFileUpload(file) {
        try {
            this.showLoading('Uploading and analyzing video... 🎬', 0);

            // Create video preview
            const video = document.getElementById('previewVideo');
            const url = URL.createObjectURL(file);
            video.src = url;

            // Wait for video metadata
            await new Promise((resolve, reject) => {
                video.addEventListener('loadedmetadata', resolve);
                video.addEventListener('error', reject);
            });

            // Update progress
            this.updateLoadingProgress(30, 'Analyzing video content...');

            // Simulate AI analysis (replace with actual API call)
            await this.performAIAnalysis(video, file);

            this.hideLoading();
            this.goToStep(2);

        } catch (error) {
            console.error('File upload error:', error);
            this.hideLoading();
            this.showToast('❌ Upload failed. Please try again.', 'error');
        }
    }

    async performAIAnalysis(video, file) {
        // Simulate advanced AI analysis
        const duration = video.duration;
        
        // Mock data - replace with actual API call
        this.analysisData = {
            session_id: 'session_' + Date.now(),
            video_info: {
                title: file.name.replace(/\.[^/.]+$/, ""),
                duration: duration,
                thumbnail: video.src,
                file_size: file.size,
                resolution: `${video.videoWidth}x${video.videoHeight}`,
                fps: 30 // Mock value
            },
            ai_insights: {
                viral_potential: 85 + Math.floor(Math.random() * 15),
                engagement_prediction: 78 + Math.floor(Math.random() * 20),
                optimal_length: 45 + Math.floor(Math.random() * 30),
                best_clips: this.generateMockClips(duration),
                trending_topics: ["viral", "trending", "amazing", "content"],
                sentiment: "positive",
                hook_moments: [5, 15, 32, 67],
                emotional_peaks: [12, 28, 45, 89],
                action_scenes: [20, 55, 78]
            }
        };

        // Update progress incrementally
        for (let i = 30; i <= 80; i += 10) {
            await this.delay(200);
            this.updateLoadingProgress(i, `AI analyzing frame ${i-20}/50...`);
        }

        await this.delay(500);
        this.updateLoadingProgress(100, 'Analysis complete! 🎉');

        // Update UI with analysis results
        this.updateAnalysisDisplay();
    }

    generateMockClips(duration) {
        const clips = [];
        const numClips = Math.min(5, Math.floor(duration / 30));
        
        for (let i = 0; i < numClips; i++) {
            const start = Math.floor((duration / numClips) * i);
            const end = Math.min(start + 30 + Math.floor(Math.random() * 30), duration);
            
            clips.push({
                start: start,
                end: end,
                score: 85 + Math.floor(Math.random() * 15),
                reason: this.getRandomClipReason(),
                title: `Viral Moment ${i + 1}`
            });
        }
        
        return clips.sort((a, b) => b.score - a.score);
    }

    getRandomClipReason() {
        const reasons = [
            "High engagement hook",
            "Emotional peak moment", 
            "Action sequence",
            "Trending topic mention",
            "Visual appeal spike",
            "Audio engagement boost",
            "Humor detection",
            "Suspense buildup"
        ];
        return reasons[Math.floor(Math.random() * reasons.length)];
    }

    // ====================
    // URL Analysis
    // ====================

    async validateUrl() {
        const urlInput = document.getElementById('videoUrl');
        const url = urlInput.value.trim();
        
        if (!url) return;

        const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/shorts\/)/;
        
        if (youtubeRegex.test(url)) {
            urlInput.style.borderColor = '#10b981';
            urlInput.style.boxShadow = '0 0 0 3px rgba(16, 185, 129, 0.1)';
        } else {
            urlInput.style.borderColor = '#ef4444';
            urlInput.style.boxShadow = '0 0 0 3px rgba(239, 68, 68, 0.1)';
        }
    }

    async analyzeUrl() {
        const urlInput = document.getElementById('videoUrl');
        const url = urlInput.value.trim();

        if (!url) {
            this.showToast('❌ Please enter a YouTube URL', 'error');
            return;
        }

        try {
            this.showLoading('Downloading and analyzing video... 🎬', 0);

            // Call the analysis API
            const response = await this.apiCall('/api/v2/analyze-video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: url,
                    clip_duration: 60,
                    output_format: "mp4",
                    resolution: "1080p",
                    aspect_ratio: "9:16",
                    enable_captions: true,
                    viral_optimization: true,
                    language: "en"
                })
            });

            if (response.success) {
                this.analysisData = response;
                this.updateAnalysisDisplay();
                this.hideLoading();
                this.goToStep(2);
                this.showToast('✅ Video analyzed successfully!', 'success');
            } else {
                throw new Error(response.error || 'Analysis failed');
            }

        } catch (error) {
            console.error('URL analysis error:', error);
            this.hideLoading();
            this.showToast(`❌ Analysis failed: ${error.message}`, 'error');
        }
    }

    // ====================
    // UI Updates
    // ====================

    updateAnalysisDisplay() {
        if (!this.analysisData) return;

        const { video_info, ai_insights } = this.analysisData;

        // Update video preview
        const video = document.getElementById('previewVideo');
        if (video && video_info.thumbnail) {
            video.poster = video_info.thumbnail;
        }

        // Update scores and insights
        this.updateElement('viralScore', ai_insights.viral_potential || '--');
        this.updateElement('engagementScore', `${ai_insights.engagement_prediction || '--'}%`);
        this.updateElement('optimalLength', `${ai_insights.optimal_length || '--'} sec`);
        this.updateElement('trendingTopics', (ai_insights.trending_topics || []).slice(0, 3).join(', ') || '--');
        this.updateElement('sentiment', ai_insights.sentiment || '--');

        // Update best clips list
        this.updateBestClipsList(ai_insights.best_clips || []);

        // Set initial clip selector position
        if (ai_insights.best_clips && ai_insights.best_clips.length > 0) {
            this.setClipFromBestMoment(0);
        }
    }

    updateBestClipsList(clips) {
        const clipsList = document.getElementById('bestClipsList');
        if (!clipsList) return;

        clipsList.innerHTML = '';

        clips.forEach((clip, index) => {
            const clipItem = document.createElement('div');
            clipItem.className = 'clip-item';
            clipItem.innerHTML = `
                <div class="clip-icon">${this.getClipIcon(clip.reason)}</div>
                <div class="clip-info">
                    <div class="clip-title">${clip.title || `Clip ${index + 1}`}</div>
                    <div class="clip-reason">${clip.reason}</div>
                </div>
                <div class="clip-score">${clip.score}%</div>
            `;

            clipItem.addEventListener('click', () => {
                this.setClipFromBestMoment(index);
                this.toggleClipSelection(index);
            });

            clipsList.appendChild(clipItem);
        });
    }

    getClipIcon(reason) {
        const iconMap = {
            'High engagement hook': '🎯',
            'Emotional peak moment': '💫',
            'Action sequence': '🎬',
            'Trending topic mention': '🔥',
            'Visual appeal spike': '✨',
            'Audio engagement boost': '🎵',
            'Humor detection': '😂',
            'Suspense buildup': '⚡'
        };
        return iconMap[reason] || '🎭';
    }

    toggleClipSelection(index) {
        const clipItems = document.querySelectorAll('.clip-item');
        const clipItem = clipItems[index];
        
        if (!clipItem) return;

        if (this.selectedClips.includes(index)) {
            this.selectedClips = this.selectedClips.filter(i => i !== index);
            clipItem.classList.remove('selected');
        } else {
            this.selectedClips.push(index);
            clipItem.classList.add('selected');
        }

        this.updateNavigationState();
    }

    setClipFromBestMoment(index) {
        if (!this.analysisData?.ai_insights?.best_clips?.[index]) return;

        const clip = this.analysisData.ai_insights.best_clips[index];
        const duration = this.analysisData.video_info.duration || 180;
        
        const startPercent = (clip.start / duration) * 100;
        const lengthPercent = ((clip.end - clip.start) / duration) * 100;

        const clipSelector = document.getElementById('clipSelector');
        if (clipSelector) {
            clipSelector.style.left = `${Math.max(0, startPercent)}%`;
            clipSelector.style.width = `${Math.min(lengthPercent, 100 - startPercent)}%`;
        }

        this.updateClipTime();
    }

    updateClipTime() {
        const clipSelector = document.getElementById('clipSelector');
        if (!clipSelector || !this.analysisData) return;

        const duration = this.analysisData.video_info.duration || 180;
        const leftPercent = parseFloat(clipSelector.style.left) || 0;
        const widthPercent = parseFloat(clipSelector.style.width) || 30;

        const startTime = (leftPercent / 100) * duration;
        const endTime = ((leftPercent + widthPercent) / 100) * duration;

        // Update video current time if playing
        const video = document.getElementById('previewVideo');
        if (video && !video.paused) {
            video.currentTime = startTime;
        }
    }

    // ====================
    // Video Processing
    // ====================

    async goToProcessing() {
        if (this.selectedClips.length === 0) {
            // Auto-select top 3 clips if none selected
            this.selectedClips = [0, 1, 2].filter(i => 
                this.analysisData?.ai_insights?.best_clips?.[i]
            );
            
            if (this.selectedClips.length === 0) {
                this.showToast('❌ No clips available for processing', 'error');
                return;
            }
        }

        this.goToStep(3);
        await this.startProcessing();
    }

    async startProcessing() {
        try {
            // Prepare clip data
            const clipsData = this.selectedClips.map(clipIndex => {
                const clipData = this.analysisData.ai_insights.best_clips[clipIndex] || {};
                return {
                    start_time: clipData.start || clipIndex * 30,
                    end_time: clipData.end || (clipIndex + 1) * 30,
                    title: clipData.title || `ViralClip Pro - Clip ${clipIndex + 1}`,
                    description: `AI-optimized viral clip`,
                    tags: this.analysisData.ai_insights.trending_topics || []
                };
            });

            // Create form data
            const formData = new FormData();
            formData.append('session_id', this.analysisData.session_id);
            formData.append('clips', JSON.stringify(clipsData));
            formData.append('priority', 'normal');

            // Start processing
            const response = await this.apiCall('/api/v2/process-video', {
                method: 'POST',
                body: formData
            });

            if (response.success) {
                this.processingTaskId = response.task_id;
                this.initializeProcessingDashboard(response);
                this.startWebSocketConnection();
                this.startProgressPolling();
            } else {
                throw new Error(response.error || 'Processing failed to start');
            }

        } catch (error) {
            console.error('Processing error:', error);
            this.showToast(`❌ Processing failed: ${error.message}`, 'error');
        }
    }

    initializeProcessingDashboard(response) {
        this.updateElement('processingSpeed', '-- clips/min');
        this.updateElement('queuePosition', `#${response.position_in_queue}`);
        this.updateElement('estimatedTime', `${Math.ceil(response.estimated_time / 60)} min`);
        this.updateElement('currentStep', 'Initializing processing...');
        this.updateElement('overallProgress', '0%');

        // Initialize clip progress
        this.initializeClipProgress();
        
        // Add initial update
        this.addLiveUpdate('Processing started', 'Processing queued successfully');
    }

    initializeClipProgress() {
        const clipsProgress = document.getElementById('clipsProgress');
        if (!clipsProgress) return;

        clipsProgress.innerHTML = '';

        this.selectedClips.forEach((clipIndex, i) => {
            const clipData = this.analysisData.ai_insights.best_clips[clipIndex];
            const clipProgressItem = document.createElement('div');
            clipProgressItem.className = 'clip-progress';
            clipProgressItem.innerHTML = `
                <div class="clip-progress-info">
                    <div class="clip-progress-title">${clipData?.title || `Clip ${i + 1}`}</div>
                    <div class="clip-progress-status">Queued</div>
                </div>
                <div class="clip-progress-bar">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 0%"></div>
                    </div>
                </div>
            `;
            clipsProgress.appendChild(clipProgressItem);
        });
    }

    async startProgressPolling() {
        if (!this.processingTaskId) return;

        const pollInterval = setInterval(async () => {
            try {
                const response = await this.apiCall(`/api/v2/processing-status/${this.processingTaskId}`);
                
                if (response.success) {
                    this.updateProcessingStatus(response.data);
                    
                    if (response.data.status === 'completed') {
                        clearInterval(pollInterval);
                        this.handleProcessingComplete(response.data);
                    } else if (response.data.status === 'failed') {
                        clearInterval(pollInterval);
                        this.handleProcessingError(response.data.error);
                    }
                }
            } catch (error) {
                console.error('Status polling error:', error);
            }
        }, 2000);

        // Store interval for cleanup
        this.pollingInterval = pollInterval;
    }

    updateProcessingStatus(data) {
        // Update main progress
        this.updateElement('overallProgress', `${data.progress || 0}%`);
        const progressFill = document.getElementById('mainProgressFill');
        if (progressFill) {
            progressFill.style.width = `${data.progress || 0}%`;
        }

        // Update current step
        this.updateElement('currentStep', data.current_step || 'Processing...');

        // Update queue position if available
        if (data.queue_position) {
            this.updateElement('queuePosition', `#${data.queue_position}`);
        }

        // Add live update
        if (data.current_step) {
            this.addLiveUpdate('System', data.current_step);
        }
    }

    handleProcessingComplete(data) {
        this.showToast('🎉 Processing completed successfully!', 'success');
        this.addLiveUpdate('Completed', 'All clips processed successfully');
        
        // Update results
        this.updateResultsDisplay(data.results);
        this.goToStep(4);
    }

    handleProcessingError(error) {
        this.showToast(`❌ Processing failed: ${error}`, 'error');
        this.addLiveUpdate('Error', error);
    }

    updateResultsDisplay(results) {
        const resultsGrid = document.getElementById('resultsGrid');
        if (!resultsGrid || !results) return;

        resultsGrid.innerHTML = '';

        results.forEach((result, index) => {
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';
            resultCard.innerHTML = `
                <div class="result-video">
                    <video controls poster="${result.thumbnail || ''}">
                        <source src="${result.file_path}" type="video/mp4">
                    </video>
                    <div class="result-overlay">
                        <div class="viral-badge">
                            Viral Score: ${result.viral_score}%
                        </div>
                    </div>
                </div>
                <div class="result-info">
                    <div class="result-title">${result.title}</div>
                    <div class="result-meta">
                        <span>⏱️ ${result.duration.toFixed(1)}s</span>
                        <span>📁 ${this.formatFileSize(result.file_size)}</span>
                        <span>🎯 ${result.viral_score}%</span>
                    </div>
                    <div class="result-actions">
                        <button class="btn btn-secondary" onclick="app.previewClip(${index})">
                            👁️ Preview
                        </button>
                        <button class="btn btn-primary" onclick="app.downloadClip(${index})">
                            📥 Download
                        </button>
                    </div>
                </div>
            `;
            resultsGrid.appendChild(resultCard);
        });

        // Store results for later use
        this.processedResults = results;
    }

    // ====================
    // WebSocket Connection
    // ====================

    startWebSocketConnection() {
        if (!this.processingTaskId) return;

        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/api/v2/ws/${this.processingTaskId}`;

        try {
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected');
                this.addLiveUpdate('Connection', 'Real-time updates enabled');
            };

            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('WebSocket message error:', error);
                }
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.addLiveUpdate('Connection', 'Real-time updates disconnected');
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

        } catch (error) {
            console.error('WebSocket connection failed:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'progress_update':
                this.updateProcessingStatus(data.data);
                break;
            case 'clip_completed':
                this.updateClipProgress(data.clip_index, 100, 'Completed');
                this.addLiveUpdate('Clip Ready', `Clip ${data.clip_index + 1} completed`);
                break;
            case 'error':
                this.addLiveUpdate('Error', data.message);
                break;
            case 'system_message':
                this.addLiveUpdate('System', data.message);
                break;
        }
    }

    updateClipProgress(clipIndex, progress, status) {
        const clipProgressItems = document.querySelectorAll('.clip-progress');
        const clipItem = clipProgressItems[clipIndex];
        
        if (!clipItem) return;

        const progressFill = clipItem.querySelector('.progress-fill');
        const statusElement = clipItem.querySelector('.clip-progress-status');

        if (progressFill) {
            progressFill.style.width = `${progress}%`;
        }
        
        if (statusElement) {
            statusElement.textContent = status;
        }
    }

    addLiveUpdate(source, message) {
        const updatesContainer = document.getElementById('liveUpdates');
        if (!updatesContainer) return;

        const updateItem = document.createElement('div');
        updateItem.className = 'update-item';
        
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        updateItem.innerHTML = `
            <span class="update-time">${timeStr}</span>
            <span class="update-message"><strong>${source}:</strong> ${message}</span>
        `;

        updatesContainer.insertBefore(updateItem, updatesContainer.firstChild);

        // Keep only last 10 updates
        while (updatesContainer.children.length > 10) {
            updatesContainer.removeChild(updatesContainer.lastChild);
        }
    }

    // ====================
    // Navigation & Step Management
    // ====================

    goToStep(stepNumber) {
        if (stepNumber < 1 || stepNumber > 4) return;

        // Update UI
        document.querySelectorAll('.step-section').forEach(section => {
            section.classList.remove('active');
        });

        const stepElement = document.getElementById(this.getStepElementId(stepNumber));
        if (stepElement) {
            stepElement.classList.add('active');
        }

        this.currentStep = stepNumber;
        this.updateNavigationState();

        // Smooth scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    getStepElementId(stepNumber) {
        const stepIds = ['', 'step-upload', 'step-analyze', 'step-processing', 'step-complete'];
        return stepIds[stepNumber] || 'step-upload';
    }

    updateNavigationState() {
        // Update bottom navigation
        document.querySelectorAll('.nav-step').forEach((btn, index) => {
            btn.classList.toggle('active', index + 1 === this.currentStep);
        });

        // Enable/disable navigation based on progress
        const canGoToStep2 = this.analysisData !== null;
        const canGoToStep3 = canGoToStep2 && (this.selectedClips.length > 0 || this.analysisData?.ai_insights?.best_clips?.length > 0);
        const canGoToStep4 = this.processedResults !== null;

        // Update navigation buttons state (if they exist)
        this.updateNavigationButtons(canGoToStep2, canGoToStep3, canGoToStep4);
    }

    updateNavigationButtons(canGoToStep2, canGoToStep3, canGoToStep4) {
        // This would update any navigation buttons in the UI
        // Implementation depends on specific UI design
    }

    nextStep() {
        if (this.currentStep < 4) {
            this.goToStep(this.currentStep + 1);
        }
    }

    previousStep() {
        if (this.currentStep > 1) {
            this.goToStep(this.currentStep - 1);
        }
    }

    // ====================
    // User Actions
    // ====================

    selectAllClips() {
        if (!this.analysisData?.ai_insights?.best_clips) return;

        this.selectedClips = [...Array(this.analysisData.ai_insights.best_clips.length).keys()];
        
        document.querySelectorAll('.clip-item').forEach((item, index) => {
            item.classList.toggle('selected', this.selectedClips.includes(index));
        });

        this.updateNavigationState();
        this.showToast(`✅ Selected ${this.selectedClips.length} clips`, 'success');
    }

    async downloadClip(clipIndex) {
        if (!this.processingTaskId || !this.processedResults?.[clipIndex]) return;

        try {
            const response = await fetch(`/api/v2/download/${this.processingTaskId}/${clipIndex}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `viralclip_pro_${clipIndex + 1}_${Date.now()}.mp4`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showToast('✅ Download started', 'success');
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showToast('❌ Download failed', 'error');
        }
    }

    async downloadAll() {
        if (!this.processedResults) return;

        for (let i = 0; i < this.processedResults.length; i++) {
            await this.downloadClip(i);
            await this.delay(1000); // Prevent overwhelming the server
        }
    }

    previewClip(clipIndex) {
        if (!this.processedResults?.[clipIndex]) return;

        const result = this.processedResults[clipIndex];
        
        // Create modal or fullscreen preview
        const modal = document.createElement('div');
        modal.className = 'preview-modal';
        modal.innerHTML = `
            <div class="preview-content">
                <div class="preview-header">
                    <h3>${result.title}</h3>
                    <button class="close-btn" onclick="this.closest('.preview-modal').remove()">✕</button>
                </div>
                <video controls autoplay style="width: 100%; max-height: 70vh;">
                    <source src="${result.file_path}" type="video/mp4">
                </video>
                <div class="preview-actions">
                    <button class="btn btn-primary" onclick="app.downloadClip(${clipIndex})">
                        📥 Download
                    </button>
                </div>
            </div>
        `;

        // Add modal styles
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            padding: 2rem;
        `;

        document.body.appendChild(modal);

        // Close on background click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    shareToSocial() {
        this.showToast('🚀 Social sharing integration coming soon!', 'info');
    }

    createNew() {
        // Clean up current session
        this.cleanup();

        // Reset state
        this.currentStep = 1;
        this.analysisData = null;
        this.selectedClips = [];
        this.processingTaskId = null;
        this.processedResults = null;

        // Reset UI
        document.getElementById('videoUrl').value = '';
        document.getElementById('fileInput').value = '';
        
        // Reset video
        const video = document.getElementById('previewVideo');
        if (video) {
            video.src = '';
            video.poster = '';
        }

        this.goToStep(1);
        this.showToast('✨ Ready for new video!', 'success');
    }

    // ====================
    // Utility Methods
    // ====================

    async apiCall(endpoint, options = {}) {
        const url = `${this.apiBase}${endpoint}`;
        
        const defaultOptions = {
            headers: {
                'Accept': 'application/json',
            }
        };

        const mergedOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, mergedOptions);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API call failed: ${endpoint}`, error);
            throw error;
        }
    }

    showLoading(message = 'Processing...', progress = null) {
        const overlay = document.getElementById('loadingOverlay');
        const messageEl = document.getElementById('loadingMessage');
        const detailEl = document.getElementById('loadingDetail');

        if (overlay) overlay.classList.remove('hidden');
        if (messageEl) messageEl.textContent = message;
        if (detailEl) detailEl.textContent = '';

        if (progress !== null) {
            this.updateLoadingProgress(progress);
        }
    }

    updateLoadingProgress(progress, detail = '') {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const detailEl = document.getElementById('loadingDetail');

        if (progressFill) progressFill.style.width = `${progress}%`;
        if (progressText) progressText.textContent = `${Math.round(progress)}%`;
        if (detailEl && detail) detailEl.textContent = detail;
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.classList.add('hidden');
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }

    updateElement(id, content) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = content;
        }
    }

    toggleVideoPlayback() {
        const video = document.getElementById('previewVideo');
        if (!video) return;

        if (video.paused) {
            video.play();
        } else {
            video.pause();
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
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

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    handleBeforeUnload(e) {
        if (this.processingTaskId) {
            e.preventDefault();
            e.returnValue = 'Video processing is in progress. Are you sure you want to leave?';
        }
    }

    cleanup() {
        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }

        // Clear polling interval
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }

        // Clean up any object URLs
        const video = document.getElementById('previewVideo');
        if (video && video.src && video.src.startsWith('blob:')) {
            URL.revokeObjectURL(video.src);
        }
    }

    // ====================
    // Template & Analytics Methods (Placeholders)
    // ====================

    showTemplates() {
        this.showToast('📋 Template library coming soon!', 'info');
    }

    showAnalytics() {
        this.showToast('📊 Analytics dashboard coming soon!', 'info');
    }

    showSettings() {
        this.showToast('⚙️ Settings panel coming soon!', 'info');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.app = new ViralClipApp();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ViralClipApp;
}
