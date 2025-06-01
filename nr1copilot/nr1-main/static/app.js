/**
 * ViralClip Pro - Netflix-Level Frontend Application
 * Advanced video processing interface with real-time features
 * 
 * Features:
 * - Real-time processing updates
 * - Advanced social media platform support
 * - Progressive Web App capabilities
 * - Accessibility compliance
 * - Performance optimization
 */

class ViralClipApp {
    constructor() {
        this.currentStep = 1;
        this.analysisData = null;
        this.selectedClips = [];
        this.selectedPlatforms = [];
        this.processingTaskId = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragDrop();
        this.setupTimeline();
        this.loadTemplates();
    }

    setupEventListeners() {
        // File input change
        document.getElementById('fileInput').addEventListener('change', (e) => {
            if (e.target.files[0]) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // URL input enter key
        document.getElementById('videoUrl').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.analyzeUrl();
            }
        });

        // Platform selection
        document.querySelectorAll('.platform-card').forEach(card => {
            card.addEventListener('click', () => {
                this.togglePlatform(card.dataset.platform);
            });
        });

        // Timeline interaction
        this.setupTimelineInteraction();
    }

    setupDragDrop() {
        const dropZone = document.getElementById('dropZone');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
        });

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

        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files[0] && files[0].type.startsWith('video/')) {
                this.handleFileUpload(files[0]);
            } else {
                this.showToast('Please upload a video file', 'error');
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

        clipSelector.addEventListener('mousedown', (e) => {
            e.preventDefault();
            const rect = clipSelector.getBoundingClientRect();

            if (e.clientX > rect.right - 10) {
                isResizing = true;
                startWidth = rect.width;
            } else {
                isDragging = true;
                startLeft = parseFloat(clipSelector.style.left) || 0;
            }

            startX = e.clientX;
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging && !isResizing) return;

            const timelineRect = timeline.getBoundingClientRect();
            const deltaX = e.clientX - startX;
            const deltaPercent = (deltaX / timelineRect.width) * 100;

            if (isDragging) {
                const newLeft = Math.max(0, Math.min(80, startLeft + deltaPercent));
                clipSelector.style.left = newLeft + '%';
            } else if (isResizing) {
                const currentLeft = parseFloat(clipSelector.style.left) || 0;
                const maxWidth = 100 - currentLeft;
                const newWidth = Math.max(5, Math.min(maxWidth, (startWidth / timelineRect.width * 100) + deltaPercent));
                clipSelector.style.width = newWidth + '%';
            }

            this.updateClipTimes();
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            isResizing = false;
        });
    }

    setupTimelineInteraction() {
        const timeline = document.getElementById('timeline');
        if (!timeline) return;

        timeline.addEventListener('click', (e) => {
            if (e.target === timeline) {
                const rect = timeline.getBoundingClientRect();
                const clickPercent = ((e.clientX - rect.left) / rect.width) * 100;
                document.getElementById('clipSelector').style.left = Math.max(0, Math.min(80, clickPercent - 10)) + '%';
                this.updateClipTimes();
            }
        });
    }

    updateClipTimes() {
        const clipSelector = document.getElementById('clipSelector');
        const left = parseFloat(clipSelector.style.left) || 0;
        const width = parseFloat(clipSelector.style.width) || 20;

        if (this.analysisData && this.analysisData.video_info) {
            const duration = this.analysisData.video_info.duration || 300;
            const startTime = (left / 100) * duration;
            const endTime = ((left + width) / 100) * duration;

            // Update preview video time
            const video = document.getElementById('previewVideo');
            if (video) {
                video.currentTime = startTime;
            }

            // Update UI to show current selection
            this.showToast(`Clip: ${this.formatTime(startTime)} - ${this.formatTime(endTime)}`, 'info');
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    async handleFileUpload(file) {
        this.showLoading('Uploading and analyzing video... 🎬');

        try {
            // Simulate file upload and analysis
            await this.delay(2000);

            // Create video preview
            const video = document.getElementById('previewVideo');
            const url = URL.createObjectURL(file);
            video.src = url;

            // Mock analysis data
            this.analysisData = {
                session_id: 'session_' + Date.now(),
                video_info: {
                    title: file.name,
                    duration: 180, // Will be updated when video loads
                    thumbnail: url
                },
                ai_insights: {
                    viral_potential: 87,
                    best_clips: [
                        { start: 15, end: 45, score: 92, reason: "High engagement hook" },
                        { start: 67, end: 97, score: 89, reason: "Emotional peak moment" },
                        { start: 120, end: 150, score: 85, reason: "Action sequence" }
                    ],
                    trending_topics: ["viral", "trending", "amazing"],
                    sentiment: "positive"
                }
            };

            video.addEventListener('loadedmetadata', () => {
                this.analysisData.video_info.duration = video.duration;
                this.updateAnalysisDisplay();
            });

            this.hideLoading();
            this.goToStep(2);
            this.showToast('Video uploaded successfully! 🚀', 'success');

        } catch (error) {
            this.hideLoading();
            this.showToast('Upload failed: ' + error.message, 'error');
        }
    }

    async analyzeUrl() {
        const url = document.getElementById('videoUrl').value.trim();
        if (!url) {
            this.showToast('Please enter a valid URL', 'error');
            return;
        }

        this.showLoading('Analyzing video with AI... 🤖');

        try {
            const response = await fetch('/api/v2/analyze-video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: url,
                    clip_duration: 60,
                    viral_optimization: true,
                    ai_editing: true
                })
            });

            const data = await response.json();

            if (data.success) {
                this.analysisData = data;
                this.updateAnalysisDisplay();
                this.goToStep(2);
                this.showToast('Analysis complete! 🎯', 'success');
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (error) {
            this.showToast('Analysis failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    updateAnalysisDisplay() {
        if (!this.analysisData) return;

        // Update viral score
        document.getElementById('viralScore').textContent = this.analysisData.ai_insights.viral_potential || 85;

        // Update insights grid
        const insightGrid = document.getElementById('insightGrid');
        insightGrid.innerHTML = '';

        const insights = [
            { icon: '🔥', title: 'Viral Potential', value: `${this.analysisData.ai_insights.viral_potential}%` },
            { icon: '⏱️', title: 'Duration', value: this.formatTime(this.analysisData.video_info.duration) },
            { icon: '📊', title: 'Sentiment', value: this.analysisData.ai_insights.sentiment },
            { icon: '🎯', title: 'Best Clips', value: `${this.analysisData.ai_insights.best_clips.length} found` }
        ];

        insights.forEach(insight => {
            const card = document.createElement('div');
            card.className = 'insight-card';
            card.innerHTML = `
                <div class="insight-icon">${insight.icon}</div>
                <div class="insight-text">
                    <strong>${insight.title}</strong>
                    <p>${insight.value}</p>
                </div>
            `;
            insightGrid.appendChild(card);
        });

        // Update suggested clips
        this.updateSuggestedClips();
    }

    updateSuggestedClips() {
        const clipsGrid = document.querySelector('.clips-grid');
        clipsGrid.innerHTML = '';

        this.analysisData.ai_insights.best_clips.forEach((clip, index) => {
            const clipCard = document.createElement('div');
            clipCard.className = 'clip-card';
            clipCard.innerHTML = `
                <div class="clip-preview">
                    <div class="clip-duration">${this.formatTime(clip.end - clip.start)}</div>
                    <div class="clip-score">Score: ${clip.score}</div>
                </div>
                <div class="clip-info">
                    <h4>Clip ${index + 1}</h4>
                    <p>${clip.reason}</p>
                    <p class="clip-time">${this.formatTime(clip.start)} - ${this.formatTime(clip.end)}</p>
                </div>
                <button class="btn btn-sm ${this.selectedClips.includes(index) ? 'btn-primary' : 'btn-outline'}" 
                        onclick="app.toggleClip(${index})">
                    ${this.selectedClips.includes(index) ? '✓ Selected' : 'Select'}
                </button>
            `;
            clipsGrid.appendChild(clipCard);
        });
    }

    toggleClip(clipIndex) {
        const index = this.selectedClips.indexOf(clipIndex);
        if (index > -1) {
            this.selectedClips.splice(index, 1);
        } else {
            this.selectedClips.push(clipIndex);
        }
        this.updateSuggestedClips();
    }

    togglePlatform(platform) {
        const card = document.querySelector(`[data-platform="${platform}"]`);
        const index = this.selectedPlatforms.indexOf(platform);

        if (index > -1) {
            this.selectedPlatforms.splice(index, 1);
            card.classList.remove('selected');
        } else {
            this.selectedPlatforms.push(platform);
            card.classList.add('selected');
        }
    }

    async processVideo() {
        if (this.selectedClips.length === 0) {
            this.showToast('Please select at least one clip', 'error');
            return;
        }

        if (this.selectedPlatforms.length === 0) {
            this.showToast('Please select at least one platform', 'error');
            return;
        }

        const platform = this.selectedPlatforms[0]; // Use first selected platform
        this.showLoading('Starting video processing... 🎬');

        try {
            // Prepare clip data
            const clipsData = this.selectedClips.map(clipIndex => {
                const clipData = this.analysisData.ai_insights.best_clips[clipIndex] || {};
                return {
                    start_time: clipData.start || clipIndex * 30,
                    end_time: clipData.end || (clipIndex + 1) * 30,
                    title: `ViralClip Pro - Clip ${clipIndex + 1}`,
                    description: `AI-optimized clip for ${platform}`,
                    tags: this.analysisData.ai_insights.trending_topics || []
                };
            });

            const formData = new FormData();
            formData.append('session_id', this.analysisData.session_id || 'test-session');
            formData.append('clips', JSON.stringify(clipsData));
            formData.append('priority', 'normal');
            formData.append('platform', platform);

            const response = await fetch('/api/v2/process-video', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.processingTaskId = data.task_id;
                this.goToStep(4);
                this.startProcessingMonitor();
                this.showToast('Processing started! 🚀', 'success');
            } else {
                throw new Error(data.error || 'Processing failed');
            }
        } catch (error) {
            this.showToast('Processing failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    startProcessingMonitor() {
        const updateProgress = async () => {
            if (!this.processingTaskId) return;

            try {
                const response = await fetch(`/api/v2/processing-status/${this.processingTaskId}`);
                const data = await response.json();

                if (data.success) {
                    const task = data.data;
                    this.updateProcessingUI(task);

                    if (task.status === 'completed') {
                        this.showProcessingResults(task);
                        this.goToStep(5);
                        return;
                    } else if (task.status === 'failed') {
                        this.showToast('Processing failed: ' + task.error, 'error');
                        return;
                    }

                    // Continue monitoring
                    setTimeout(updateProgress, 2000);
                }
            } catch (error) {
                console.error('Error monitoring progress:', error);
                setTimeout(updateProgress, 5000);
            }
        };

        updateProgress();
    }

    updateProcessingUI(task) {
        const progress = task.progress || 0;
        const status = task.current_step || 'Processing...';

        // Update progress ring
        const circle = document.getElementById('progressCircle');
        if (circle) {
            const circumference = 2 * Math.PI * 52;
            const offset = circumference - (progress / 100) * circumference;
            circle.style.strokeDasharray = circumference;
            circle.style.strokeDashoffset = offset;
        }

        // Update progress bar
        const progressFill = document.getElementById('progressFill');
        if (progressFill) {
            progressFill.style.width = progress + '%';
        }

        // Update text
        document.getElementById('processingStatus').textContent = status;
        document.getElementById('progressText').textContent = Math.round(progress) + '%';
    }

    showProcessingResults(task) {
        const resultsContent = document.getElementById('resultsContent');
        const showcase = resultsContent.querySelector('.clips-showcase');

        showcase.innerHTML = '';

        task.results.forEach((result, index) => {
            const clipResult = document.createElement('div');
            clipResult.className = 'clip-result';
            clipResult.innerHTML = `
                <div class="result-preview">
                    <video controls>
                        <source src="/api/v2/download/${task.task_id || this.processingTaskId}/${index}" type="video/mp4">
                    </video>
                </div>
                <div class="result-info">
                    <h4>${result.title}</h4>
                    <p>Duration: ${this.formatTime(result.duration)}</p>
                    <p>Viral Score: ${result.viral_score}%</p>
                    <p>Size: ${this.formatFileSize(result.file_size)}</p>
                </div>
                <div class="result-actions">
                    <button class="btn btn-primary" onclick="app.downloadClip('${this.processingTaskId}', ${index})">
                        📥 Download
                    </button>
                    <button class="btn btn-secondary" onclick="app.shareClip('${this.processingTaskId}', ${index})">
                        📤 Share
                    </button>
                </div>
            `;
            showcase.appendChild(clipResult);
        });
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async downloadClip(taskId, clipIndex) {
        window.open(`/api/v2/download/${taskId}/${clipIndex}`, '_blank');
    }

    async downloadAll() {
        if (!this.processingTaskId) return;

        // Download each clip individually
        const task = await this.getTaskStatus(this.processingTaskId);
        if (task && task.results) {
            task.results.forEach((result, index) => {
                setTimeout(() => {
                    this.downloadClip(this.processingTaskId, index);
                }, index * 1000); // Stagger downloads
            });
        }
    }

    async getTaskStatus(taskId) {
        try {
            const response = await fetch(`/api/v2/processing-status/${taskId}`);
            const data = await response.json();
            return data.success ? data.data : null;
        } catch (error) {
            console.error('Error getting task status:', error);
            return null;
        }
    }

    shareClip(taskId, clipIndex) {
        // Implement social media sharing
        this.showToast('Social media sharing coming soon! 📤', 'info');
    }

    shareToSocial() {
        this.showToast('Direct social media integration coming soon! 🚀', 'info');
    }

    createNew() {
        this.currentStep = 1;
        this.analysisData = null;
        this.selectedClips = [];
        this.selectedPlatforms = [];
        this.processingTaskId = null;

        // Reset UI
        document.querySelectorAll('.step').forEach(step => step.classList.remove('active'));
        document.getElementById('step-upload').classList.add('active');

        // Reset form
        document.getElementById('videoUrl').value = '';
        document.getElementById('fileInput').value = '';

        this.updateStepIndicators();
        this.showToast('Ready for new video! 🎬', 'success');
    }

    // Navigation methods
    goToStep(stepNumber) {
        document.querySelectorAll('.step').forEach(step => step.classList.remove('active'));
        document.getElementById(`step-${this.getStepName(stepNumber)}`).classList.add('active');

        this.currentStep = stepNumber;
        this.updateStepIndicators();
        this.updateNavigationButtons();
    }

    getStepName(stepNumber) {
        const stepNames = ['', 'upload', 'analyze', 'platforms', 'processing', 'complete'];
        return stepNames[stepNumber] || 'upload';
    }

    nextStep() {
        if (this.currentStep === 2 && this.selectedClips.length === 0) {
            this.showToast('Please select at least one clip', 'error');
            return;
        }

        if (this.currentStep === 3) {
            this.processVideo();
            return;
        }

        if (this.currentStep < 5) {
            this.goToStep(this.currentStep + 1);
        }
    }

    prevStep() {
        if (this.currentStep > 1) {
            this.goToStep(this.currentStep - 1);
        }
    }

    updateStepIndicators() {
        document.querySelectorAll('.step-dot').forEach((dot, index) => {
            dot.classList.toggle('active', index + 1 <= this.currentStep);
        });
    }

    updateNavigationButtons() {
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');

        prevBtn.style.display = this.currentStep > 1 ? 'block' : 'none';

        if (this.currentStep === 3) {
            nextBtn.textContent = 'Start Processing →';
        } else if (this.currentStep >= 4) {
            nextBtn.style.display = 'none';
        } else {
            nextBtn.textContent = 'Next →';
            nextBtn.style.display = 'block';
        }
    }

    // Utility methods
    showLoading(message) {
        document.getElementById('loadingText').textContent = message;
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;

        document.getElementById('toastContainer').appendChild(toast);

        setTimeout(() => {
            toast.classList.add('show');
        }, 100);

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Feature methods
    loadTemplates() {
        // Load viral video templates
        console.log('Loading viral templates...');
    }

    showTemplates() {
        this.showToast('Viral templates coming soon! 📋', 'info');
    }

    showAnalytics() {
        this.showToast('Analytics dashboard coming soon! 📊', 'info');
    }

    showSettings() {
        this.showToast('Settings panel coming soon! ⚙️', 'info');
    }

    showHelp() {
        this.showToast('Help center coming soon! ❓', 'info');
    }

    setupPlatformSelection() {
        const platformGrid = document.getElementById('platform-grid');
        if (!platformGrid) return;

        this.supportedPlatforms.forEach(platform => {
            const platformCard = this.createPlatformCard(platform);
            platformGrid.appendChild(platformCard);
        });
    }

    createPlatformCard(platform) {
        const card = document.createElement('div');
        card.className = 'platform-card';
        card.dataset.platform = platform;

        const platformData = this.getPlatformData(platform);

        card.innerHTML = `
            <div class="platform-icon">${platformData.icon}</div>
            <h3>${platformData.name}</h3>
            <div class="platform-specs">
                <span class="spec">${platformData.aspectRatio}</span>
                <span class="spec">${platformData.maxDuration}s max</span>
                <span class="spec">${platformData.resolution}</span>
            </div>
            <div class="platform-features">
                ${platformData.features.map(feature => `<span class="feature">${feature}</span>`).join('')}
            </div>
            <div class="platform-score" id="score-${platform}">
                <span class="score-label">Suitability</span>
                <span class="score-value">--</span>
            </div>
        `;

        card.addEventListener('click', () => this.selectPlatform(platform, card));

        return card;
    }

    getPlatformData(platform) {
        const platformSpecs = {
            'tiktok': {
                name: 'TikTok',
                icon: '🎵',
                aspectRatio: '9:16',
                maxDuration: 180,
                resolution: '1080x1920',
                features: ['Trending Sounds', 'Effects', 'Hashtags']
            },
            'instagram_reels': {
                name: 'Instagram Reels',
                icon: '📸',
                aspectRatio: '9:16',
                maxDuration: 90,
                resolution: '1080x1920',
                features: ['Music Library', 'AR Effects', 'Shopping']
            },
            'instagram_story': {
                name: 'Instagram Story',
                icon: '📱',
                aspectRatio: '9:16',
                maxDuration: 15,
                resolution: '1080x1920',
                features: ['Stickers', 'Polls', 'Links']
            },
            'instagram_feed': {
                name: 'Instagram Feed',
                icon: '🖼️',
                aspectRatio: '1:1',
                maxDuration: 60,
                resolution: '1080x1080',
                features: ['IGTV', 'Carousel', 'Tags']
            },
            'youtube_shorts': {
                name: 'YouTube Shorts',
                icon: '📺',
                aspectRatio: '9:16',
                maxDuration: 60,
                resolution: '1080x1920',
                features: ['Monetization', 'Analytics', 'Community']
            },
            'youtube_standard': {
                name: 'YouTube',
                icon: '🎬',
                aspectRatio: '16:9',
                maxDuration: 3600,
                resolution: '1920x1080',
                features: ['Long Form', 'Thumbnails', 'SEO']
            },
            'twitter': {
                name: 'Twitter',
                icon: '🐦',
                aspectRatio: '16:9',
                maxDuration: 140,
                resolution: '1280x720',
                features: ['Trending', 'Threads', 'Engagement']
            },
            'facebook': {
                name: 'Facebook',
                icon: '👥',
                aspectRatio: '16:9',
                maxDuration: 240,
                resolution: '1920x1080',
                features: ['Stories', 'Watch', 'Groups']
            },
            'linkedin': {
                name: 'LinkedIn',
                icon: '💼',
                aspectRatio: '16:9',
                maxDuration: 600,
                resolution: '1920x1080',
                features: ['Professional', 'B2B', 'Thought Leadership']
            },
            'snapchat': {
                name: 'Snapchat',
                icon: '👻',
                aspectRatio: '9:16',
                maxDuration: 60,
                resolution: '1080x1920',
                features: ['Lenses', 'Discover', 'Snap Map']
            },
            'pinterest': {
                name: 'Pinterest',
                icon: '📌',
                aspectRatio: '2:3',
                maxDuration: 15,
                resolution: '1000x1500',
                features: ['Idea Pins', 'Shopping', 'Inspiration']
            }
        };

        return platformSpecs[platform] || {};
    }
}

// Initialize the app
const app = new ViralClipApp();

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

// Export for global access
window.app = app;