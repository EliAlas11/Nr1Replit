
// SendShort AI Clone - Advanced JavaScript Application
// Modern, high-performance implementation with real-time features

class ViralClipGenerator {
    constructor() {
        this.currentStep = 1;
        this.sessionId = null;
        this.taskId = null;
        this.processingInterval = null;
        this.videoData = null;
        this.selectedClips = [];
        this.templates = [];
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.setupTheme();
        this.loadTemplates();
        await this.hideLoadingScreen();
        this.setupProgressiveEnhancement();
    }

    // Event Listeners
    setupEventListeners() {
        // URL Analysis
        const analyzeBtn = document.getElementById('analyze-btn');
        const urlInput = document.getElementById('video-url');
        
        if (analyzeBtn && urlInput) {
            analyzeBtn.addEventListener('click', () => this.analyzeVideo());
            urlInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.analyzeVideo();
            });
            urlInput.addEventListener('input', this.validateUrlInput.bind(this));
        }

        // File Upload
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('video-file');
        
        if (uploadZone && fileInput) {
            uploadZone.addEventListener('click', () => fileInput.click());
            uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadZone.addEventListener('drop', this.handleDrop.bind(this));
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Input method switching
        document.querySelectorAll('.input-method').forEach(method => {
            method.addEventListener('click', () => this.switchInputMethod(method));
        });

        // Navigation buttons
        this.setupNavigationButtons();

        // Processing controls
        const startProcessingBtn = document.getElementById('start-processing');
        if (startProcessingBtn) {
            startProcessingBtn.addEventListener('click', () => this.startProcessing());
        }

        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', this.toggleTheme.bind(this));
        }

        // Timeline interaction
        this.setupTimelineInteraction();
    }

    setupNavigationButtons() {
        const buttons = {
            'back-to-input': () => this.goToStep(1),
            'proceed-to-editing': () => this.goToStep(3),
            'back-to-analysis': () => this.goToStep(2),
            'create-more': () => this.goToStep(1),
        };

        Object.entries(buttons).forEach(([id, handler]) => {
            const btn = document.getElementById(id);
            if (btn) btn.addEventListener('click', handler);
        });
    }

    setupTimelineInteraction() {
        const timeline = document.getElementById('video-timeline');
        if (timeline) {
            timeline.addEventListener('click', this.handleTimelineClick.bind(this));
        }

        const addClipBtn = document.getElementById('add-clip');
        const autoDetectBtn = document.getElementById('auto-detect');
        
        if (addClipBtn) addClipBtn.addEventListener('click', this.addCustomClip.bind(this));
        if (autoDetectBtn) autoDetectBtn.addEventListener('click', this.autoDetectClips.bind(this));
    }

    // Core Functions
    async analyzeVideo() {
        const urlInput = document.getElementById('video-url');
        const analyzeBtn = document.getElementById('analyze-btn');
        
        if (!urlInput || !analyzeBtn) return;
        
        const url = urlInput.value.trim();
        if (!this.isValidYouTubeUrl(url)) {
            this.showToast('Please enter a valid YouTube URL', 'error');
            return;
        }

        this.setButtonLoading(analyzeBtn, true);

        try {
            const response = await fetch('/api/analyze-video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    url: url,
                    clip_duration: 60,
                    output_format: 'mp4',
                    resolution: '1080p',
                    aspect_ratio: '9:16',
                    enable_captions: true,
                    viral_optimization: true,
                    language: 'en'
                })
            });

            const data = await response.json();

            if (data.success) {
                this.sessionId = data.data.session_id;
                this.videoData = data.data;
                this.displayAnalysisResults(data.data);
                this.goToStep(2);
                this.showToast('Video analyzed successfully!', 'success');
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showToast(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.setButtonLoading(analyzeBtn, false);
        }
    }

    displayAnalysisResults(data) {
        // Update video info
        const thumbnail = document.getElementById('video-thumbnail');
        const title = document.getElementById('video-title');
        const duration = document.getElementById('video-duration');
        const uploader = document.getElementById('video-uploader');
        const views = document.getElementById('video-views');
        const scoreText = document.getElementById('viral-score-text');
        const scoreFill = document.getElementById('viral-score-fill');

        if (thumbnail) thumbnail.src = data.video_info.thumbnail;
        if (title) title.textContent = data.video_info.title;
        if (duration) duration.textContent = this.formatDuration(data.video_info.duration);
        if (uploader) uploader.textContent = data.video_info.uploader;
        if (views) views.textContent = `${data.video_info.view_count?.toLocaleString() || 0} views`;
        
        if (scoreText && scoreFill) {
            const viralScore = data.ai_insights.viral_potential;
            scoreText.textContent = `${viralScore}%`;
            scoreFill.style.width = `${viralScore}%`;
        }

        // Display suggested clips
        this.displaySuggestedClips(data.ai_insights.best_clips);
    }

    displaySuggestedClips(clips) {
        const container = document.getElementById('suggested-clips');
        if (!container) return;

        container.innerHTML = clips.map((clip, index) => `
            <div class="clip-card" data-clip-index="${index}">
                <div class="clip-header">
                    <span class="clip-duration">${this.formatDuration(clip.end - clip.start)}</span>
                    <div class="clip-score">
                        <span>🔥</span>
                        <span>${clip.score}%</span>
                    </div>
                </div>
                <p class="clip-reason">${clip.reason}</p>
                <div class="clip-preview">
                    <span>▶️</span>
                </div>
                <div class="clip-actions">
                    <button class="btn-small select-clip">Select</button>
                    <button class="btn-small edit-clip">Edit</button>
                    <button class="btn-small preview-clip">Preview</button>
                </div>
            </div>
        `).join('');

        // Add click handlers
        container.querySelectorAll('.clip-card').forEach(card => {
            card.addEventListener('click', () => this.selectClip(card));
        });

        container.querySelectorAll('.select-clip').forEach((btn, index) => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleClipSelection(index);
            });
        });
    }

    selectClip(card) {
        const index = parseInt(card.dataset.clipIndex);
        this.toggleClipSelection(index);
    }

    toggleClipSelection(index) {
        const card = document.querySelector(`[data-clip-index="${index}"]`);
        if (!card) return;

        if (card.classList.contains('selected')) {
            card.classList.remove('selected');
            this.selectedClips = this.selectedClips.filter(i => i !== index);
        } else {
            card.classList.add('selected');
            this.selectedClips.push(index);
        }

        this.updateTimelineVisualization();
    }

    updateTimelineVisualization() {
        const timelineClips = document.getElementById('timeline-clips');
        if (!timelineClips || !this.videoData) return;

        const duration = this.videoData.video_info.duration;
        const clips = this.videoData.ai_insights.best_clips;

        timelineClips.innerHTML = this.selectedClips.map(index => {
            const clip = clips[index];
            const left = (clip.start / duration) * 100;
            const width = ((clip.end - clip.start) / duration) * 100;
            
            return `
                <div class="timeline-clip" style="left: ${left}%; width: ${width}%">
                    <span class="clip-label">${index + 1}</span>
                </div>
            `;
        }).join('');
    }

    async startProcessing() {
        if (!this.sessionId || this.selectedClips.length === 0) {
            this.showToast('Please select at least one clip', 'warning');
            return;
        }

        this.goToStep(4);

        const clips = this.selectedClips.map(index => {
            const clip = this.videoData.ai_insights.best_clips[index];
            return {
                start_time: clip.start,
                end_time: clip.end,
                title: `Viral Clip ${index + 1}`,
                description: clip.reason
            };
        });

        try {
            const formData = new FormData();
            formData.append('session_id', this.sessionId);
            formData.append('clips', JSON.stringify(clips));

            const response = await fetch('/api/process-video', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.taskId = data.task_id;
                this.startProgressTracking();
                this.showToast('Processing started!', 'success');
            } else {
                throw new Error(data.error || 'Processing failed to start');
            }
        } catch (error) {
            console.error('Processing error:', error);
            this.showToast(`Processing failed: ${error.message}`, 'error');
            this.goToStep(3);
        }
    }

    startProgressTracking() {
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
        }

        this.processingInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/processing-status/${this.taskId}`);
                const data = await response.json();

                if (data.success) {
                    this.updateProcessingProgress(data.data);

                    if (data.data.status === 'completed') {
                        clearInterval(this.processingInterval);
                        this.displayResults(data.data.results);
                        this.goToStep(5);
                        this.showToast('Processing completed!', 'success');
                    } else if (data.data.status === 'failed') {
                        clearInterval(this.processingInterval);
                        this.showToast(`Processing failed: ${data.data.error}`, 'error');
                        this.goToStep(3);
                    }
                }
            } catch (error) {
                console.error('Progress tracking error:', error);
            }
        }, 2000);
    }

    updateProcessingProgress(taskData) {
        const progressFill = document.getElementById('processing-progress');
        const progressText = document.getElementById('progress-percentage');
        const statusText = document.getElementById('progress-status');

        if (progressFill) progressFill.style.width = `${taskData.progress}%`;
        if (progressText) progressText.textContent = `${taskData.progress}%`;
        if (statusText) statusText.textContent = this.getStatusText(taskData.progress);

        // Update step indicators
        this.updateProcessingSteps(taskData.progress);
    }

    updateProcessingSteps(progress) {
        const steps = [
            { id: 'step-download', threshold: 20 },
            { id: 'step-analyze', threshold: 40 },
            { id: 'step-extract', threshold: 60 },
            { id: 'step-enhance', threshold: 80 },
            { id: 'step-render', threshold: 100 }
        ];

        steps.forEach(step => {
            const element = document.getElementById(step.id);
            if (element) {
                if (progress >= step.threshold) {
                    element.classList.add('completed');
                    element.classList.remove('active');
                } else if (progress >= step.threshold - 20) {
                    element.classList.add('active');
                    element.classList.remove('completed');
                } else {
                    element.classList.remove('active', 'completed');
                }
            }
        });
    }

    getStatusText(progress) {
        if (progress < 20) return 'Downloading video...';
        if (progress < 40) return 'Analyzing content...';
        if (progress < 60) return 'Extracting clips...';
        if (progress < 80) return 'Applying AI enhancements...';
        if (progress < 100) return 'Final rendering...';
        return 'Processing complete!';
    }

    displayResults(results) {
        const container = document.getElementById('results-grid');
        if (!container) return;

        container.innerHTML = results.map((result, index) => `
            <div class="result-card">
                <div class="result-video" data-result-index="${index}">
                    <span>▶️</span>
                </div>
                <div class="result-info">
                    <h4 class="result-title">${result.title}</h4>
                    <div class="result-meta">
                        <span>${this.formatDuration(result.duration)}</span>
                        <span>${this.formatFileSize(result.file_size)}</span>
                        <div class="viral-score-small">
                            <span>🔥</span>
                            <span>${result.viral_score}%</span>
                        </div>
                    </div>
                </div>
                <div class="result-actions">
                    <button class="btn-download" onclick="app.downloadClip(${this.taskId}, ${index})">
                        📥 Download
                    </button>
                    <button class="btn-preview" onclick="app.previewClip(${index})">
                        👁️
                    </button>
                </div>
            </div>
        `).join('');
    }

    async downloadClip(taskId, clipIndex) {
        try {
            const response = await fetch(`/api/download/${taskId}/${clipIndex}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `viral_clip_${clipIndex + 1}.mp4`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                this.showToast('Download started!', 'success');
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showToast('Download failed', 'error');
        }
    }

    // UI Helper Functions
    goToStep(step) {
        // Hide all steps
        document.querySelectorAll('.step-section').forEach(section => {
            section.classList.remove('active');
        });

        // Show target step
        const targetStep = document.getElementById(`step-${this.getStepName(step)}`);
        if (targetStep) {
            targetStep.classList.add('active');
            targetStep.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        this.currentStep = step;
    }

    getStepName(step) {
        const names = {
            1: 'input',
            2: 'analysis',
            3: 'editing',
            4: 'processing',
            5: 'results'
        };
        return names[step] || 'input';
    }

    setButtonLoading(button, loading) {
        const textElement = button.querySelector('.btn-text');
        const loaderElement = button.querySelector('.btn-loader');

        if (loading) {
            button.disabled = true;
            if (textElement) textElement.classList.add('hidden');
            if (loaderElement) loaderElement.classList.remove('hidden');
        } else {
            button.disabled = false;
            if (textElement) textElement.classList.remove('hidden');
            if (loaderElement) loaderElement.classList.add('hidden');
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-header">
                <h5 class="toast-title">${this.getToastTitle(type)}</h5>
                <button class="toast-close">&times;</button>
            </div>
            <p class="toast-message">${message}</p>
        `;

        container.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);

        // Close button
        toast.querySelector('.toast-close').addEventListener('click', () => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
    }

    getToastTitle(type) {
        const titles = {
            success: 'Success!',
            error: 'Error',
            warning: 'Warning',
            info: 'Info'
        };
        return titles[type] || 'Notification';
    }

    // File handling
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processUploadedFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processUploadedFile(file);
        }
    }

    async processUploadedFile(file) {
        if (!this.isValidVideoFile(file)) {
            this.showToast('Please select a valid video file (MP4, MOV, AVI, MKV)', 'error');
            return;
        }

        if (file.size > 2 * 1024 * 1024 * 1024) { // 2GB limit
            this.showToast('File size must be less than 2GB', 'error');
            return;
        }

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload-video', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.showToast('File uploaded successfully!', 'success');
                // Process uploaded file similar to URL analysis
            } else {
                throw new Error(data.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showToast(`Upload failed: ${error.message}`, 'error');
        }
    }

    // Validation functions
    isValidYouTubeUrl(url) {
        const patterns = [
            /^https?:\/\/(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[\w-]+/,
            /^https?:\/\/(www\.)?youtube\.com\/embed\/[\w-]+/
        ];
        return patterns.some(pattern => pattern.test(url));
    }

    isValidVideoFile(file) {
        const validTypes = ['video/mp4', 'video/mov', 'video/avi', 'video/x-msvideo', 'video/x-matroska'];
        return validTypes.includes(file.type) || 
               /\.(mp4|mov|avi|mkv)$/i.test(file.name);
    }

    validateUrlInput() {
        const urlInput = document.getElementById('video-url');
        const analyzeBtn = document.getElementById('analyze-btn');
        
        if (urlInput && analyzeBtn) {
            const isValid = this.isValidYouTubeUrl(urlInput.value.trim());
            analyzeBtn.disabled = !isValid;
        }
    }

    // Utility functions
    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatFileSize(bytes) {
        const sizes = ['B', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 B';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    // Theme management
    setupTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
    }

    toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const newTheme = current === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        this.updateThemeIcon(newTheme);
    }

    updateThemeIcon(theme) {
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
        }
    }

    // Advanced features
    async loadTemplates() {
        try {
            const response = await fetch('/api/templates');
            const data = await response.json();
            
            if (data.success) {
                this.templates = data.templates;
                this.displayTemplates();
            }
        } catch (error) {
            console.error('Failed to load templates:', error);
        }
    }

    displayTemplates() {
        const container = document.getElementById('template-grid');
        if (!container) return;

        container.innerHTML = this.templates.map(template => `
            <div class="template-card" data-template-id="${template.id}">
                <div class="template-icon">🎬</div>
                <div class="template-name">${template.name}</div>
            </div>
        `).join('');

        container.querySelectorAll('.template-card').forEach(card => {
            card.addEventListener('click', () => this.applyTemplate(card.dataset.templateId));
        });
    }

    async applyTemplate(templateId) {
        try {
            const response = await fetch('/api/apply-template', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    template_id: templateId,
                    video_url: this.videoData?.video_info?.webpage_url || ''
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.showToast('Template applied successfully!', 'success');
            }
        } catch (error) {
            console.error('Template application error:', error);
            this.showToast('Failed to apply template', 'error');
        }
    }

    switchInputMethod(methodElement) {
        document.querySelectorAll('.input-method').forEach(method => {
            method.classList.remove('active');
        });
        methodElement.classList.add('active');
    }

    handleTimelineClick(e) {
        const timeline = e.currentTarget;
        const rect = timeline.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = x / rect.width;
        
        if (this.videoData) {
            const time = percentage * this.videoData.video_info.duration;
            this.updateTimelineCursor(percentage);
            console.log(`Timeline clicked at ${this.formatDuration(time)}`);
        }
    }

    updateTimelineCursor(percentage) {
        const cursor = document.getElementById('timeline-cursor');
        if (cursor) {
            cursor.style.left = `${percentage * 100}%`;
        }
    }

    addCustomClip() {
        this.showToast('Custom clip creation coming soon!', 'info');
    }

    autoDetectClips() {
        this.showToast('Auto-detecting optimal clips...', 'info');
        // Simulate auto-detection
        setTimeout(() => {
            if (this.videoData && this.videoData.ai_insights.best_clips) {
                this.selectedClips = [0, 1]; // Select first two clips
                this.updateClipSelectionUI();
                this.showToast('Auto-detection complete!', 'success');
            }
        }, 2000);
    }

    updateClipSelectionUI() {
        document.querySelectorAll('.clip-card').forEach((card, index) => {
            if (this.selectedClips.includes(index)) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        });
        this.updateTimelineVisualization();
    }

    previewClip(index) {
        this.showToast('Clip preview coming soon!', 'info');
    }

    async hideLoadingScreen() {
        return new Promise(resolve => {
            setTimeout(() => {
                const loadingScreen = document.getElementById('loading-screen');
                if (loadingScreen) {
                    loadingScreen.classList.add('hidden');
                }
                resolve();
            }, 1500);
        });
    }

    setupProgressiveEnhancement() {
        // Add CSS animations based on user preferences
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            document.documentElement.style.setProperty('--transition-fast', '0ms');
            document.documentElement.style.setProperty('--transition-base', '0ms');
            document.documentElement.style.setProperty('--transition-slow', '0ms');
        }

        // Setup intersection observer for animations
        if ('IntersectionObserver' in window) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('animate-in');
                    }
                });
            });

            document.querySelectorAll('.feature-card, .step-section').forEach(el => {
                observer.observe(el);
            });
        }

        // Setup service worker for PWA features
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js').catch(error => {
                console.log('Service Worker registration failed:', error);
            });
        }
    }
}

// Initialize the application
const app = new ViralClipGenerator();

// Global error handling
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    if (app) {
        app.showToast('An unexpected error occurred', 'error');
    }
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
    if (app) {
        app.showToast('An unexpected error occurred', 'error');
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Escape key to close modals
    if (e.key === 'Escape') {
        const modal = document.querySelector('.modal-overlay:not(.hidden)');
        if (modal) {
            modal.classList.add('hidden');
        }
    }
    
    // Ctrl/Cmd + Enter to proceed to next step
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (app.currentStep < 5) {
            app.goToStep(app.currentStep + 1);
        }
    }
});

// Performance monitoring
if ('PerformanceObserver' in window) {
    const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
            if (entry.entryType === 'navigation') {
                console.log(`Page load time: ${entry.loadEventEnd - entry.loadEventStart}ms`);
            }
        }
    });
    observer.observe({ entryTypes: ['navigation'] });
}

// Export for global access
window.app = app;
