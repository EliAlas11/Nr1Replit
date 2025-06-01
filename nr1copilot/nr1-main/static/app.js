
/**
 * ViralClip Pro - Frontend Application
 * Netflix-level JavaScript with SendShort.ai killer features
 */

class ViralClipApp {
    constructor() {
        this.currentSession = null;
        this.websocket = null;
        this.dragDropHandler = null;
        this.progressInterval = null;
        this.retryCount = 0;
        this.maxRetries = 3;
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing ViralClip Pro - SendShort.ai Killer');
        
        // Hide loading screen
        this.hideLoadingScreen();
        
        // Initialize components
        this.initializeEventListeners();
        this.initializeDragDrop();
        this.initializeExamples();
        this.initializeTheme();
        this.initializeAnalytics();
        
        // Check for saved session
        this.restoreSession();
        
        console.log('✅ ViralClip Pro ready - Netflix-level performance activated');
    }

    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loadingScreen');
        if (loadingScreen) {
            setTimeout(() => {
                loadingScreen.style.opacity = '0';
                setTimeout(() => {
                    loadingScreen.style.display = 'none';
                }, 300);
            }, 1000);
        }
    }

    initializeEventListeners() {
        // URL Analysis
        const analyzeBtn = document.getElementById('analyzeBtn');
        const videoUrl = document.getElementById('videoUrl');
        
        if (analyzeBtn && videoUrl) {
            analyzeBtn.addEventListener('click', () => this.handleAnalyzeVideo());
            videoUrl.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleAnalyzeVideo();
            });
            
            // Real-time URL validation
            videoUrl.addEventListener('input', (e) => this.validateUrl(e.target.value));
        }

        // Upload method toggle
        const toggleBtns = document.querySelectorAll('.toggle-btn');
        toggleBtns.forEach(btn => {
            btn.addEventListener('click', () => this.toggleUploadMethod(btn.dataset.method));
        });

        // Custom clip addition
        const addCustomClip = document.getElementById('addCustomClip');
        if (addCustomClip) {
            addCustomClip.addEventListener('click', () => this.addCustomClip());
        }

        // Example URLs
        const exampleUrls = document.querySelectorAll('.example-url');
        exampleUrls.forEach(btn => {
            btn.addEventListener('click', () => {
                document.getElementById('videoUrl').value = btn.dataset.url;
                this.handleAnalyzeVideo();
            });
        });

        // Mobile menu
        const mobileMenuBtn = document.getElementById('mobileMenuBtn');
        if (mobileMenuBtn) {
            mobileMenuBtn.addEventListener('click', () => this.toggleMobileMenu());
        }

        // File browse
        const browseBtn = document.getElementById('browseBtn');
        const fileInput = document.getElementById('fileInput');
        
        if (browseBtn && fileInput) {
            browseBtn.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files[0]));
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    }

    initializeDragDrop() {
        const dropZone = document.getElementById('fileDropZone');
        if (!dropZone) return;

        this.dragDropHandler = {
            dragCounter: 0,
            
            handleDragEnter: (e) => {
                e.preventDefault();
                this.dragDropHandler.dragCounter++;
                dropZone.classList.add('drag-over');
            },
            
            handleDragLeave: (e) => {
                e.preventDefault();
                this.dragDropHandler.dragCounter--;
                if (this.dragDropHandler.dragCounter === 0) {
                    dropZone.classList.remove('drag-over');
                }
            },
            
            handleDragOver: (e) => {
                e.preventDefault();
            },
            
            handleDrop: (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                this.dragDropHandler.dragCounter = 0;
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileUpload(files[0]);
                }
            }
        };

        // Add event listeners
        dropZone.addEventListener('dragenter', this.dragDropHandler.handleDragEnter);
        dropZone.addEventListener('dragleave', this.dragDropHandler.handleDragLeave);
        dropZone.addEventListener('dragover', this.dragDropHandler.handleDragOver);
        dropZone.addEventListener('drop', this.dragDropHandler.handleDrop);
    }

    initializeExamples() {
        // Add some viral video examples for testing
        const examples = [
            {
                title: "Mr. Beast Challenge",
                url: "https://www.youtube.com/watch?v=example1",
                viral_score: 95
            },
            {
                title: "Dancing Baby Trend",
                url: "https://www.youtube.com/watch?v=example2", 
                viral_score: 89
            }
        ];

        // Could add these to the UI dynamically
        console.log('📺 Example videos loaded:', examples.length);
    }

    initializeTheme() {
        // Check for saved theme preference
        const savedTheme = localStorage.getItem('viralclip-theme');
        if (savedTheme) {
            document.body.classList.add(`theme-${savedTheme}`);
        }

        // Add theme toggle functionality
        this.addThemeToggle();
    }

    initializeAnalytics() {
        // Track page load
        this.trackEvent('page_load', {
            timestamp: new Date().toISOString(),
            user_agent: navigator.userAgent,
            screen_resolution: `${screen.width}x${screen.height}`
        });
    }

    validateUrl(url) {
        const analyzeBtn = document.getElementById('analyzeBtn');
        const urlInput = document.getElementById('videoUrl');
        
        const isValidYouTube = /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)/i.test(url);
        
        if (url.length > 10 && isValidYouTube) {
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('disabled');
            urlInput.classList.remove('error');
            urlInput.classList.add('valid');
        } else if (url.length > 0) {
            analyzeBtn.disabled = true;
            analyzeBtn.classList.add('disabled');
            urlInput.classList.add('error');
            urlInput.classList.remove('valid');
        } else {
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('disabled');
            urlInput.classList.remove('error', 'valid');
        }
    }

    toggleUploadMethod(method) {
        const toggleBtns = document.querySelectorAll('.toggle-btn');
        const uploadMethods = document.querySelectorAll('.upload-method');
        
        // Update toggle buttons
        toggleBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.method === method);
        });
        
        // Update upload methods
        uploadMethods.forEach(methodEl => {
            methodEl.classList.toggle('active', methodEl.id === `${method}Method`);
        });

        this.trackEvent('upload_method_changed', { method });
    }

    async handleAnalyzeVideo() {
        const videoUrl = document.getElementById('videoUrl').value.trim();
        
        if (!videoUrl) {
            this.showNotification('Please enter a YouTube URL', 'error');
            return;
        }

        this.trackEvent('analyze_video_started', { url: videoUrl });

        try {
            this.setAnalyzeButtonLoading(true);
            
            const requestData = {
                url: videoUrl,
                language: 'en',
                viral_optimization: true,
                ai_editing: true
            };

            const response = await fetch('/api/v2/analyze-video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.currentSession = result.session_id;
                this.saveSession(result);
                this.displayAnalysisResults(result);
                this.trackEvent('analyze_video_success', { 
                    session_id: result.session_id,
                    viral_score: result.ai_insights.viral_potential 
                });
            } else {
                throw new Error(result.error || 'Analysis failed');
            }

        } catch (error) {
            console.error('Analysis error:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
            this.trackEvent('analyze_video_error', { error: error.message });
        } finally {
            this.setAnalyzeButtonLoading(false);
        }
    }

    async handleFileUpload(file) {
        if (!file) return;

        // Validate file
        if (!file.type.startsWith('video/')) {
            this.showNotification('Please select a video file', 'error');
            return;
        }

        if (file.size > 2 * 1024 * 1024 * 1024) { // 2GB limit
            this.showNotification('File size must be less than 2GB', 'error');
            return;
        }

        this.trackEvent('file_upload_started', { 
            filename: file.name, 
            size: file.size,
            type: file.type 
        });

        try {
            this.showUploadProgress(0);
            
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/v2/upload-video', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Upload failed: HTTP ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.showUploadProgress(100);
                this.showNotification('File uploaded successfully!', 'success');
                
                // TODO: Integrate with analysis pipeline
                this.trackEvent('file_upload_success', { 
                    file_id: result.file_id,
                    filename: result.filename 
                });
            } else {
                throw new Error(result.error || 'Upload failed');
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
            this.trackEvent('file_upload_error', { error: error.message });
        } finally {
            this.hideUploadProgress();
        }
    }

    displayAnalysisResults(result) {
        // Show analysis section
        document.getElementById('analysisSection').style.display = 'block';
        
        // Smooth scroll to results
        document.getElementById('analysisSection').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });

        // Update video info
        this.updateVideoInfo(result.video_info);
        
        // Update AI insights
        this.updateAIInsights(result.ai_insights);
        
        // Generate suggested clips
        this.generateSuggestedClips(result.ai_insights);
        
        // Update analysis stats
        this.updateAnalysisStats(result);
    }

    updateVideoInfo(videoInfo) {
        // Thumbnail
        const thumbnail = document.getElementById('videoThumbnail');
        if (thumbnail && videoInfo.thumbnail) {
            thumbnail.src = videoInfo.thumbnail;
            thumbnail.alt = videoInfo.title;
        }

        // Title
        const title = document.getElementById('videoTitle');
        if (title) {
            title.textContent = videoInfo.title || 'Unknown Title';
        }

        // Duration
        const duration = document.getElementById('videoDuration');
        if (duration) {
            duration.textContent = this.formatDuration(videoInfo.duration);
        }

        // Views
        const views = document.getElementById('videoViews');
        if (views && videoInfo.view_count) {
            views.textContent = this.formatNumber(videoInfo.view_count) + ' views';
        }

        // Uploader
        const uploader = document.getElementById('videoUploader');
        if (uploader) {
            uploader.textContent = videoInfo.uploader || 'Unknown';
        }
    }

    updateAIInsights(aiInsights) {
        // Viral score
        const viralScore = aiInsights.viral_potential || 0;
        const scoreFill = document.getElementById('viralScoreFill');
        const scoreText = document.getElementById('viralScoreText');
        
        if (scoreFill && scoreText) {
            scoreFill.style.width = `${viralScore}%`;
            scoreFill.className = `score-fill ${this.getScoreClass(viralScore)}`;
            scoreText.textContent = `${viralScore}%`;
        }

        // Engagement score
        const engagementEl = document.getElementById('engagementScore');
        if (engagementEl) {
            engagementEl.textContent = `${aiInsights.engagement_prediction || 0}%`;
        }

        // Best platforms
        const platformsEl = document.getElementById('bestPlatforms');
        if (platformsEl) {
            const platforms = aiInsights.suggested_formats || ['TikTok', 'Instagram'];
            platformsEl.innerHTML = platforms.slice(0, 2).map(p => 
                `<span class="platform-tag">${p}</span>`
            ).join('');
        }

        // Optimal length
        const lengthEl = document.getElementById('optimalLength');
        if (lengthEl) {
            lengthEl.textContent = `${aiInsights.optimal_length || 60}s`;
        }

        // Sentiment
        const sentimentEl = document.getElementById('sentiment');
        if (sentimentEl) {
            const sentiment = aiInsights.sentiment_analysis || 'positive';
            sentimentEl.innerHTML = `<span class="sentiment-${sentiment}">${sentiment}</span>`;
        }
    }

    generateSuggestedClips(aiInsights) {
        const container = document.getElementById('suggestedClips');
        if (!container) return;

        // Generate clips based on AI insights
        const clips = this.generateClipsFromInsights(aiInsights);
        
        container.innerHTML = clips.map((clip, index) => `
            <div class="clip-card" data-clip-index="${index}">
                <div class="clip-timeline">
                    <div class="timeline-bar">
                        <div class="timeline-segment" style="left: ${(clip.start_time / 300) * 100}%; width: ${((clip.end_time - clip.start_time) / 300) * 100}%"></div>
                    </div>
                    <div class="timeline-labels">
                        <span>${this.formatTime(clip.start_time)}</span>
                        <span>${this.formatTime(clip.end_time)}</span>
                    </div>
                </div>
                <div class="clip-info">
                    <h5>${clip.title}</h5>
                    <p>${clip.description}</p>
                    <div class="clip-stats">
                        <span class="duration">${this.formatTime(clip.end_time - clip.start_time)}</span>
                        <span class="viral-score score-${this.getScoreClass(clip.viral_score)}">${clip.viral_score}% viral</span>
                    </div>
                </div>
                <div class="clip-actions">
                    <button class="btn btn-sm btn-outline" onclick="app.editClip(${index})">Edit</button>
                    <button class="btn btn-sm btn-primary" onclick="app.processClip(${index})">Process</button>
                </div>
            </div>
        `).join('');

        // Add process all button
        container.innerHTML += `
            <div class="clip-actions-footer">
                <button class="btn btn-success btn-large" onclick="app.processAllClips()">
                    🚀 Process All Clips (${clips.length})
                </button>
            </div>
        `;

        this.suggestedClips = clips;
    }

    generateClipsFromInsights(aiInsights) {
        const clips = [];
        
        // Generate clips from AI insights
        const hookMoments = aiInsights.hook_moments || [];
        const emotionalPeaks = aiInsights.emotional_peaks || [];
        const actionScenes = aiInsights.action_scenes || [];
        
        // Combine all interesting moments
        const allMoments = [
            ...hookMoments.map(m => ({ ...m, type: 'hook' })),
            ...emotionalPeaks.map(m => ({ ...m, type: 'emotional' })),
            ...actionScenes.map(m => ({ ...m, type: 'action' }))
        ].sort((a, b) => a.timestamp - b.timestamp);

        // Generate clips from moments
        allMoments.forEach((moment, index) => {
            const duration = moment.duration || 30;
            clips.push({
                start_time: Math.max(0, moment.timestamp - 5),
                end_time: moment.timestamp + duration,
                title: this.generateClipTitle(moment.type, index + 1),
                description: moment.description || this.generateClipDescription(moment.type),
                viral_score: moment.score || (75 + Math.random() * 20),
                tags: [moment.type, 'viral', 'ai-generated']
            });
        });

        // If no AI moments, generate default clips
        if (clips.length === 0) {
            clips.push(
                {
                    start_time: 0,
                    end_time: 30,
                    title: "Opening Hook",
                    description: "Engaging opening to capture attention",
                    viral_score: 82,
                    tags: ['hook', 'viral']
                },
                {
                    start_time: 60,
                    end_time: 90,
                    title: "Key Moment",
                    description: "Main content with high engagement",
                    viral_score: 87,
                    tags: ['highlight', 'viral']
                }
            );
        }

        return clips.slice(0, 5); // Limit to 5 clips
    }

    async processAllClips() {
        if (!this.currentSession || !this.suggestedClips) {
            this.showNotification('No clips to process', 'error');
            return;
        }

        this.trackEvent('process_all_clips_started', {
            session_id: this.currentSession,
            clip_count: this.suggestedClips.length
        });

        try {
            const formData = new FormData();
            formData.append('session_id', this.currentSession);
            formData.append('clips', JSON.stringify(this.suggestedClips));
            formData.append('priority', 'normal');

            const response = await fetch('/api/v2/process-video', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.currentTaskId = result.task_id;
                this.showProcessingSection(result);
                this.initializeWebSocket(result.task_id);
                this.trackEvent('process_all_clips_success', { task_id: result.task_id });
            } else {
                throw new Error(result.error || 'Processing failed');
            }

        } catch (error) {
            console.error('Processing error:', error);
            this.showNotification(`Processing failed: ${error.message}`, 'error');
            this.trackEvent('process_all_clips_error', { error: error.message });
        }
    }

    showProcessingSection(processingData) {
        const section = document.getElementById('processingSectionContainer');
        if (!section) return;

        section.style.display = 'block';
        section.scrollIntoView({ behavior: 'smooth' });

        // Update queue info
        document.getElementById('queuePosition').textContent = processingData.position_in_queue;
        document.getElementById('estimatedTime').textContent = this.formatDuration(processingData.estimated_time);

        // Initialize progress
        this.updateProgress(0, 'Queued for processing...');
        
        // Initialize timeline
        this.initializeTimeline();
    }

    initializeWebSocket(taskId) {
        // Close existing connection
        if (this.websocket) {
            this.websocket.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v2/ws/${taskId}`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('🔗 WebSocket connected');
            this.addLiveUpdate('Connected to processing server', 'success');
        };
        
        this.websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.addLiveUpdate('Connection error - retrying...', 'error');
            this.retryWebSocket(taskId);
        };
        
        this.websocket.onclose = () => {
            console.log('🔌 WebSocket disconnected');
            this.addLiveUpdate('Connection closed', 'warning');
        };

        // Send periodic pings
        this.pingInterval = setInterval(() => {
            if (this.websocket?.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    handleWebSocketMessage(message) {
        console.log('📡 WebSocket message:', message);
        
        switch (message.type) {
            case 'status_update':
                this.updateProgress(message.data.progress, message.data.message);
                break;
                
            case 'progress_update':
                this.updateProgress(message.data.progress, message.data.message);
                this.updateTimeline(message.data.current_step);
                break;
                
            case 'processing_complete':
                this.handleProcessingComplete(message.data);
                break;
                
            case 'processing_error':
                this.handleProcessingError(message.data);
                break;
                
            case 'pong':
                // Ping response
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
        
        this.addLiveUpdate(message.data?.message || 'Processing update received', 'info');
    }

    updateProgress(progress, message) {
        const progressFill = document.getElementById('progressFill');
        const progressPercent = document.getElementById('progressPercent');
        const progressStep = document.getElementById('progressStep');
        
        if (progressFill) {
            progressFill.style.width = `${progress}%`;
            progressFill.className = `progress-fill ${this.getProgressClass(progress)}`;
        }
        
        if (progressPercent) {
            progressPercent.textContent = `${Math.round(progress)}%`;
        }
        
        if (progressStep && message) {
            progressStep.textContent = message;
        }
    }

    updateTimeline(currentStep) {
        const timeline = document.getElementById('statusTimeline');
        if (!timeline) return;

        const steps = [
            { id: 'queued', name: 'Queued', icon: '⏳' },
            { id: 'downloading', name: 'Downloading', icon: '📥' },
            { id: 'ai_analysis', name: 'AI Analysis', icon: '🤖' },
            { id: 'processing_clip_1', name: 'Processing Clips', icon: '🎬' },
            { id: 'completed', name: 'Completed', icon: '✅' }
        ];

        timeline.innerHTML = steps.map(step => {
            const isActive = currentStep === step.id;
            const isCompleted = steps.findIndex(s => s.id === currentStep) > steps.findIndex(s => s.id === step.id);
            
            return `
                <div class="timeline-step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}">
                    <div class="step-icon">${step.icon}</div>
                    <div class="step-name">${step.name}</div>
                </div>
            `;
        }).join('');
    }

    addLiveUpdate(message, type = 'info') {
        const feed = document.getElementById('updatesFeed');
        if (!feed) return;

        const timestamp = new Date().toLocaleTimeString();
        const updateEl = document.createElement('div');
        updateEl.className = `update-item ${type}`;
        updateEl.innerHTML = `
            <span class="update-time">${timestamp}</span>
            <span class="update-message">${message}</span>
        `;

        feed.insertBefore(updateEl, feed.firstChild);

        // Limit to 10 updates
        while (feed.children.length > 10) {
            feed.removeChild(feed.lastChild);
        }

        // Auto-scroll
        updateEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    handleProcessingComplete(data) {
        console.log('🎉 Processing completed:', data);
        
        this.trackEvent('processing_completed', {
            task_id: this.currentTaskId,
            total_time: data.total_time,
            results_count: data.results?.length || 0
        });

        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }

        // Clear ping interval
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
        }

        // Show results
        this.showResults(data);
        
        // Show success notification
        this.showNotification('All clips processed successfully! 🎉', 'success');
    }

    handleProcessingError(data) {
        console.error('❌ Processing error:', data);
        
        this.trackEvent('processing_error', {
            task_id: this.currentTaskId,
            error: data.error
        });

        this.showNotification(`Processing failed: ${data.error}`, 'error');
        this.addLiveUpdate(`Error: ${data.error}`, 'error');
    }

    showResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        if (!resultsSection) return;

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Update summary
        document.getElementById('totalClips').textContent = data.results?.length || 0;
        document.getElementById('totalProcessingTime').textContent = `${Math.round(data.total_time)}s`;
        
        const avgViralScore = data.results?.reduce((sum, clip) => sum + (clip.viral_score || 0), 0) / (data.results?.length || 1);
        document.getElementById('avgViralScore').textContent = `${Math.round(avgViralScore)}%`;

        // Display clips
        this.displayClipResults(data.results || []);
    }

    displayClipResults(results) {
        const grid = document.getElementById('clipsGrid');
        if (!grid) return;

        grid.innerHTML = results.map((clip, index) => `
            <div class="clip-result-card">
                <div class="clip-preview">
                    <video 
                        src="/api/v2/download/${this.currentTaskId}/${index}" 
                        poster="${clip.thumbnail || ''}"
                        controls
                        preload="metadata"
                    ></video>
                </div>
                <div class="clip-details">
                    <h4>${clip.title}</h4>
                    <div class="clip-meta">
                        <span>Duration: ${this.formatTime(clip.duration)}</span>
                        <span>Size: ${this.formatFileSize(clip.file_size)}</span>
                        <span class="viral-score score-${this.getScoreClass(clip.viral_score)}">
                            ${clip.viral_score}% viral
                        </span>
                    </div>
                    <div class="clip-enhancements">
                        ${(clip.ai_enhancements || []).map(enhancement => 
                            `<span class="enhancement-tag">${enhancement}</span>`
                        ).join('')}
                    </div>
                </div>
                <div class="clip-actions">
                    <button class="btn btn-primary" onclick="app.downloadClip(${index})">
                        Download
                    </button>
                    <button class="btn btn-outline" onclick="app.shareClip(${index})">
                        Share
                    </button>
                </div>
            </div>
        `).join('');
    }

    async downloadClip(clipIndex) {
        if (!this.currentTaskId) return;

        try {
            const url = `/api/v2/download/${this.currentTaskId}/${clipIndex}`;
            const link = document.createElement('a');
            link.href = url;
            link.download = `viralclip_pro_${clipIndex + 1}.mp4`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            this.trackEvent('clip_downloaded', { 
                task_id: this.currentTaskId, 
                clip_index: clipIndex 
            });

        } catch (error) {
            console.error('Download error:', error);
            this.showNotification('Download failed', 'error');
        }
    }

    // Utility functions
    formatDuration(seconds) {
        if (!seconds) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatNumber(num) {
        if (!num) return '0';
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    }

    formatFileSize(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    getScoreClass(score) {
        if (score >= 80) return 'high';
        if (score >= 60) return 'medium';
        return 'low';
    }

    getProgressClass(progress) {
        if (progress >= 80) return 'high';
        if (progress >= 40) return 'medium';
        return 'low';
    }

    generateClipTitle(type, index) {
        const titles = {
            hook: ['Opening Hook', 'Attention Grabber', 'Viral Opener'],
            emotional: ['Emotional Peak', 'Heart Moment', 'Feel-Good Clip'],
            action: ['Action Scene', 'Dynamic Moment', 'High Energy']
        };
        return titles[type]?.[index % titles[type].length] || `Clip ${index}`;
    }

    generateClipDescription(type) {
        const descriptions = {
            hook: 'Engaging opening designed to capture viewer attention',
            emotional: 'Emotional peak moment with high engagement potential',
            action: 'High-energy scene perfect for viral content'
        };
        return descriptions[type] || 'AI-generated viral clip';
    }

    setAnalyzeButtonLoading(loading) {
        const btn = document.getElementById('analyzeBtn');
        const text = btn?.querySelector('.btn-text');
        const loader = btn?.querySelector('.btn-loader');
        
        if (btn) {
            btn.disabled = loading;
            btn.classList.toggle('loading', loading);
        }
        if (text) text.style.display = loading ? 'none' : 'inline';
        if (loader) loader.style.display = loading ? 'inline' : 'none';
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-message">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    getNotificationIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || 'ℹ️';
    }

    trackEvent(eventName, data = {}) {
        try {
            // Send to analytics
            if (typeof gtag !== 'undefined') {
                gtag('event', eventName, data);
            }
            
            // Log for debugging
            console.log(`📊 Event: ${eventName}`, data);
            
            // Could also send to your own analytics endpoint
            // fetch('/api/analytics/event', { ... });
            
        } catch (error) {
            console.error('Analytics error:', error);
        }
    }

    saveSession(data) {
        try {
            localStorage.setItem('viralclip-session', JSON.stringify({
                session_id: data.session_id,
                timestamp: new Date().toISOString(),
                video_info: data.video_info
            }));
        } catch (error) {
            console.error('Failed to save session:', error);
        }
    }

    restoreSession() {
        try {
            const saved = localStorage.getItem('viralclip-session');
            if (saved) {
                const session = JSON.parse(saved);
                // Check if session is less than 24 hours old
                const sessionTime = new Date(session.timestamp);
                const now = new Date();
                if (now - sessionTime < 24 * 60 * 60 * 1000) {
                    console.log('🔄 Restored session:', session.session_id);
                    // Could restore the session here
                }
            }
        } catch (error) {
            console.error('Failed to restore session:', error);
        }
    }

    retryWebSocket(taskId) {
        if (this.retryCount < this.maxRetries) {
            this.retryCount++;
            setTimeout(() => {
                console.log(`🔄 Retrying WebSocket connection (${this.retryCount}/${this.maxRetries})`);
                this.initializeWebSocket(taskId);
            }, 2000 * this.retryCount);
        }
    }

    handleKeyboardShortcuts(e) {
        // Ctrl+Enter to analyze
        if (e.ctrlKey && e.key === 'Enter') {
            this.handleAnalyzeVideo();
        }
        
        // Esc to close modals
        if (e.key === 'Escape') {
            // Close any open modals
        }
    }

    addThemeToggle() {
        // Add theme toggle button
        const themeToggle = document.createElement('button');
        themeToggle.className = 'theme-toggle';
        themeToggle.innerHTML = '🌙';
        themeToggle.onclick = () => this.toggleTheme();
        
        // Add to navbar
        const navbar = document.querySelector('.navbar .nav-menu');
        if (navbar) {
            navbar.appendChild(themeToggle);
        }
    }

    toggleTheme() {
        document.body.classList.toggle('dark-theme');
        const isDark = document.body.classList.contains('dark-theme');
        localStorage.setItem('viralclip-theme', isDark ? 'dark' : 'light');
        
        // Update toggle icon
        const toggle = document.querySelector('.theme-toggle');
        if (toggle) {
            toggle.innerHTML = isDark ? '☀️' : '🌙';
        }
    }

    toggleMobileMenu() {
        const navbar = document.querySelector('.navbar');
        navbar?.classList.toggle('mobile-menu-open');
    }

    showUploadProgress(progress) {
        // Implementation for file upload progress
        console.log(`Upload progress: ${progress}%`);
    }

    hideUploadProgress() {
        // Implementation to hide upload progress
        console.log('Upload progress hidden');
    }

    updateAnalysisStats(result) {
        const statsEl = document.getElementById('analysisStats');
        if (statsEl) {
            statsEl.innerHTML = `
                <div class="stat">
                    <span class="stat-label">Processing Time</span>
                    <span class="stat-value">${result.processing_time.toFixed(1)}s</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Cache Hit</span>
                    <span class="stat-value">${result.cache_hit ? 'Yes' : 'No'}</span>
                </div>
            `;
        }
    }

    addCustomClip() {
        // Implementation for adding custom clips
        console.log('Adding custom clip...');
        
        // Could open a modal or inline form
        this.showNotification('Custom clip editor coming soon!', 'info');
    }

    editClip(index) {
        console.log(`Editing clip ${index}`);
        this.showNotification('Clip editor coming soon!', 'info');
    }

    processClip(index) {
        console.log(`Processing single clip ${index}`);
        // Process just one clip
        const clip = this.suggestedClips[index];
        if (clip) {
            this.suggestedClips = [clip];
            this.processAllClips();
        }
    }

    shareClip(index) {
        console.log(`Sharing clip ${index}`);
        
        if (navigator.share) {
            navigator.share({
                title: 'Check out my viral clip!',
                text: 'Created with ViralClip Pro',
                url: window.location.href
            });
        } else {
            // Fallback to copy link
            navigator.clipboard.writeText(window.location.href);
            this.showNotification('Link copied to clipboard!', 'success');
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ViralClipApp();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && window.app) {
        // Page became visible, check for updates
        if (window.app.currentTaskId) {
            // Reconnect WebSocket if needed
        }
    }
});

// Handle online/offline status
window.addEventListener('online', () => {
    console.log('📶 Back online');
    if (window.app) {
        window.app.showNotification('Connection restored', 'success');
    }
});

window.addEventListener('offline', () => {
    console.log('📵 Gone offline');
    if (window.app) {
        window.app.showNotification('Connection lost - working offline', 'warning');
    }
});
