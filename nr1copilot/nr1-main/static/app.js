/**
 * ViralClip Pro - Netflix-Level Frontend
 * Advanced video processing interface with real-time updates
 */

class ViralClipApp {
    constructor() {
        this.ws = null;
        this.currentTaskId = null;
        this.currentSessionId = null;
        this.clips = [];
        this.analysisData = null;
        this.processingStartTime = null;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.setupWebSocket();
        this.showNotification('ViralClip Pro loaded successfully!', 'success');
    }

    setupEventListeners() {
        // URL analysis
        const analyzeBtn = document.getElementById('analyze-btn');
        const urlInput = document.getElementById('video-url');

        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyzeVideo());
        }

        if (urlInput) {
            urlInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.analyzeVideo();
                }
            });

            urlInput.addEventListener('input', () => {
                this.validateUrl(urlInput.value);
            });
        }

        // File upload
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
    }

    setupDragAndDrop() {
        const uploadZone = document.getElementById('upload-zone');
        if (!uploadZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => {
                uploadZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => {
                uploadZone.classList.remove('dragover');
            }, false);
        });

        uploadZone.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    setupWebSocket() {
        // WebSocket will be initialized when needed
        console.log('WebSocket ready for initialization');
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    validateUrl(url) {
        const urlPattern = /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/;
        const youtubePattern = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/;

        const analyzeBtn = document.getElementById('analyze-btn');
        if (!analyzeBtn) return;

        if (url && (youtubePattern.test(url) || urlPattern.test(url))) {
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('disabled');
        } else {
            analyzeBtn.disabled = !url;
            analyzeBtn.classList.toggle('disabled', !url);
        }
    }

    async analyzeVideo() {
        const urlInput = document.getElementById('video-url');
        const url = urlInput?.value?.trim();

        if (!url) {
            this.showNotification('Please enter a valid video URL', 'error');
            return;
        }

        this.setButtonLoading('analyze-btn', true);

        try {
            const response = await fetch('/api/v2/analyze-video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: url,
                    clip_duration: 60,
                    output_format: 'mp4',
                    resolution: '1080p',
                    aspect_ratio: '9:16',
                    enable_captions: true,
                    enable_transitions: true,
                    ai_editing: true,
                    viral_optimization: true,
                    language: 'en',
                    priority: 'normal'
                })
            });

            const data = await response.json();

            if (data.success) {
                this.currentSessionId = data.session_id;
                this.analysisData = data;
                this.showAnalysisResults(data);
                this.showNotification('Video analysis complete!', 'success');
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.setButtonLoading('analyze-btn', false);
        }
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            await this.uploadFile(file);
        }
    }

    async handleDrop(event) {
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            await this.uploadFile(files[0]);
        }
    }

    async uploadFile(file) {
        // Validate file
        if (!file.type.startsWith('video/')) {
            this.showNotification('Please select a video file', 'error');
            return;
        }

        if (file.size > 2 * 1024 * 1024 * 1024) { // 2GB
            this.showNotification('File size must be less than 2GB', 'error');
            return;
        }

        const progressContainer = document.getElementById('upload-progress');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');

        if (progressContainer) {
            progressContainer.style.display = 'block';
        }

        try {
            const formData = new FormData();
            formData.append('file', file);

            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    if (progressFill) {
                        progressFill.style.width = percentComplete + '%';
                    }
                    if (progressText) {
                        progressText.textContent = Math.round(percentComplete) + '%';
                    }
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    const response = JSON.parse(xhr.responseText);
                    if (response.success) {
                        this.showNotification('File uploaded successfully!', 'success');
                        // Process the uploaded file
                        this.processUploadedFile(response);
                    } else {
                        throw new Error(response.message || 'Upload failed');
                    }
                } else {
                    throw new Error('Upload failed');
                }
            });

            xhr.addEventListener('error', () => {
                throw new Error('Upload failed');
            });

            xhr.open('POST', '/api/v2/upload-video');
            xhr.send(formData);

        } catch (error) {
            console.error('Upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
            if (progressContainer) {
                progressContainer.style.display = 'none';
            }
        }
    }

    processUploadedFile(uploadResponse) {
        // For uploaded files, we can proceed to clip creation
        this.showNotification('File ready for processing!', 'success');
        // Show clip creation interface
        this.showClipCreator({
            video_info: {
                title: uploadResponse.filename,
                duration: 300, // We'd need to extract this from the file
                thumbnail: null
            }
        });
    }

    showAnalysisResults(data) {
        const analysisSection = document.getElementById('analysis-section');
        const resultsContainer = document.getElementById('analysis-results');

        if (!analysisSection || !resultsContainer) return;

        const videoInfo = data.video_info;
        const aiInsights = data.ai_insights;

        resultsContainer.innerHTML = `
            <div class="analysis-grid">
                <div class="video-info-card">
                    <h3>📹 Video Information</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="label">Title:</span>
                            <span class="value">${videoInfo.title}</span>
                        </div>
                        <div class="info-item">
                            <span class="label">Duration:</span>
                            <span class="value">${this.formatDuration(videoInfo.duration)}</span>
                        </div>
                        <div class="info-item">
                            <span class="label">Views:</span>
                            <span class="value">${this.formatNumber(videoInfo.view_count)}</span>
                        </div>
                        <div class="info-item">
                            <span class="label">Likes:</span>
                            <span class="value">${this.formatNumber(videoInfo.like_count)}</span>
                        </div>
                    </div>
                </div>

                <div class="ai-insights-card">
                    <h3>🤖 AI Analysis</h3>
                    <div class="insights-grid">
                        <div class="insight-meter">
                            <span class="meter-label">Viral Potential</span>
                            <div class="meter">
                                <div class="meter-fill" style="width: ${aiInsights.viral_potential}%"></div>
                            </div>
                            <span class="meter-value">${aiInsights.viral_potential}%</span>
                        </div>
                        <div class="insight-meter">
                            <span class="meter-label">Engagement Score</span>
                            <div class="meter">
                                <div class="meter-fill" style="width: ${aiInsights.engagement_prediction}%"></div>
                            </div>
                            <span class="meter-value">${aiInsights.engagement_prediction}%</span>
                        </div>
                        <div class="insights-list">
                            <h4>Best Platforms:</h4>
                            <div class="platform-tags">
                                ${aiInsights.suggested_formats.map(format => `<span class="platform-tag">${format}</span>`).join('')}
                            </div>
                        </div>
                        <div class="insights-list">
                            <h4>Optimal Length:</h4>
                            <span class="optimal-length">${aiInsights.optimal_length} seconds</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        analysisSection.style.display = 'block';
        analysisSection.scrollIntoView({ behavior: 'smooth' });

        // Show clip creator
        setTimeout(() => {
            this.showClipCreator(data);
        }, 500);
    }

    showClipCreator(data) {
        const clipCreator = document.getElementById('clip-creator');
        if (!clipCreator) return;

        clipCreator.style.display = 'block';

        // Initialize with some default clips
        this.clips = [
            { start_time: 10, end_time: 70, title: 'Viral Clip 1', description: 'Main highlight' },
            { start_time: 30, end_time: 90, title: 'Viral Clip 2', description: 'Secondary moment' }
        ];

        this.renderClipTimeline();
    }

    renderClipTimeline() {
        const timeline = document.getElementById('clip-timeline');
        if (!timeline) return;

        timeline.innerHTML = `
            <div class="timeline-container">
                <h4>Clip Timeline</h4>
                <div class="clips-list">
                    ${this.clips.map((clip, index) => `
                        <div class="clip-item" data-index="${index}">
                            <div class="clip-header">
                                <h5>${clip.title}</h5>
                                <button onclick="app.removeClip(${index})" class="btn-remove">×</button>
                            </div>
                            <div class="clip-times">
                                <input type="number" value="${clip.start_time}" 
                                       onchange="app.updateClip(${index}, 'start_time', this.value)"
                                       placeholder="Start (seconds)" class="time-input">
                                <span>to</span>
                                <input type="number" value="${clip.end_time}" 
                                       onchange="app.updateClip(${index}, 'end_time', this.value)"
                                       placeholder="End (seconds)" class="time-input">
                            </div>
                            <input type="text" value="${clip.title}" 
                                   onchange="app.updateClip(${index}, 'title', this.value)"
                                   placeholder="Clip title" class="clip-title-input">
                            <textarea onchange="app.updateClip(${index}, 'description', this.value)"
                                      placeholder="Description (optional)" class="clip-description">${clip.description}</textarea>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    addClip() {
        const newClip = {
            start_time: 0,
            end_time: 60,
            title: `Viral Clip ${this.clips.length + 1}`,
            description: ''
        };
        this.clips.push(newClip);
        this.renderClipTimeline();
        this.showNotification('New clip added!', 'success');
    }

    removeClip(index) {
        this.clips.splice(index, 1);
        this.renderClipTimeline();
        this.showNotification('Clip removed', 'success');
    }

    updateClip(index, field, value) {
        if (this.clips[index]) {
            this.clips[index][field] = field.includes('time') ? parseFloat(value) : value;
        }
    }

    async processClips() {
        if (!this.currentSessionId) {
            this.showNotification('No session found. Please analyze a video first.', 'error');
            return;
        }

        if (this.clips.length === 0) {
            this.showNotification('Please add at least one clip', 'error');
            return;
        }

        this.setButtonLoading('process-btn', true);

        try {
            const formData = new FormData();
            formData.append('session_id', this.currentSessionId);
            formData.append('clips', JSON.stringify(this.clips));
            formData.append('priority', 'normal');

            const response = await fetch('/api/v2/process-video', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.currentTaskId = data.task_id;
                this.showProcessingStatus();
                this.initializeWebSocket(data.task_id);
                this.showNotification('Processing started!', 'success');
            } else {
                throw new Error(data.detail || 'Processing failed');
            }
        } catch (error) {
            console.error('Processing error:', error);
            this.showNotification(`Processing failed: ${error.message}`, 'error');
        } finally {
            this.setButtonLoading('process-btn', false);
        }
    }

    initializeWebSocket(taskId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v2/ws/${taskId}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.processingStartTime = Date.now();
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showNotification('Connection error', 'error');
        };

        // Send periodic pings
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'status_update':
            case 'progress_update':
                this.updateProcessingStatus(message.data);
                break;
            case 'processing_complete':
                this.handleProcessingComplete(message.data);
                break;
            case 'processing_error':
                this.handleProcessingError(message.data);
                break;
            case 'pong':
                // Heartbeat response
                break;
        }
    }

    showProcessingStatus() {
        const processingSection = document.getElementById('processing-section');
        if (processingSection) {
            processingSection.style.display = 'block';
            processingSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    updateProcessingStatus(data) {
        const statusContainer = document.getElementById('processing-status');
        if (!statusContainer) return;

        const elapsedTime = this.processingStartTime ? 
            Math.floor((Date.now() - this.processingStartTime) / 1000) : 0;

        statusContainer.innerHTML = `
            <div class="processing-card">
                <div class="processing-header">
                    <h3>🎬 Processing Your Clips</h3>
                    <span class="status-badge ${data.status}">${data.status.toUpperCase()}</span>
                </div>
                <div class="progress-section">
                    <div class="progress-bar-large">
                        <div class="progress-fill-large" style="width: ${data.progress || 0}%"></div>
                    </div>
                    <div class="progress-info">
                        <span class="progress-percentage">${data.progress || 0}%</span>
                        <span class="progress-time">Elapsed: ${this.formatDuration(elapsedTime)}</span>
                    </div>
                </div>
                <div class="current-step">
                    <strong>Current Step:</strong> ${data.current_step || data.message || 'Initializing...'}
                </div>
            </div>
        `;
    }

    handleProcessingComplete(data) {
        this.showNotification('All clips processed successfully!', 'success');
        this.showProcessingResults(data.results);
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    handleProcessingError(data) {
        this.showNotification(`Processing failed: ${data.error}`, 'error');
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    showProcessingResults(results) {
        const resultsContainer = document.getElementById('clip-results');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = `
            <div class="results-header">
                <h3>🎉 Your Viral Clips Are Ready!</h3>
                <p>Download your professionally processed clips below:</p>
            </div>
            <div class="clips-grid">
                ${results.map((result, index) => `
                    <div class="result-clip-card">
                        <div class="clip-preview">
                            ${result.thumbnail ? 
                                `<img src="${result.thumbnail}" alt="Clip ${index + 1}" class="clip-thumbnail">` :
                                `<div class="clip-placeholder">🎬</div>`
                            }
                        </div>
                        <div class="clip-info">
                            <h4>${result.title}</h4>
                            <div class="clip-stats">
                                <span class="stat">⏱️ ${this.formatDuration(result.duration)}</span>
                                <span class="stat">📊 ${result.viral_score}% viral potential</span>
                                <span class="stat">💾 ${this.formatFileSize(result.file_size)}</span>
                            </div>
                            ${result.ai_enhancements?.length ? `
                                <div class="enhancements">
                                    <strong>AI Enhancements:</strong>
                                    <ul>
                                        ${result.ai_enhancements.map(enhancement => `<li>${enhancement}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                        <div class="clip-actions">
                            <button onclick="app.downloadClip(${index})" class="btn btn-primary">
                                📥 Download
                            </button>
                            <button onclick="app.shareClip(${index})" class="btn btn-outline">
                                📤 Share
                            </button>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    async downloadClip(clipIndex) {
        if (!this.currentTaskId) return;

        try {
            const response = await fetch(`/api/v2/download/${this.currentTaskId}/${clipIndex}`);

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

                this.showNotification('Download started!', 'success');
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showNotification('Download failed', 'error');
        }
    }

    shareClip(clipIndex) {
        if (navigator.share) {
            navigator.share({
                title: 'Check out my viral clip from ViralClip Pro!',
                text: 'Created with the SendShort.ai killer - ViralClip Pro',
                url: window.location.href
            });
        } else {
            // Fallback: copy link to clipboard
            navigator.clipboard.writeText(window.location.href).then(() => {
                this.showNotification('Link copied to clipboard!', 'success');
            });
        }
    }

    // Utility methods
    setButtonLoading(buttonId, loading) {
        const button = document.getElementById(buttonId);
        if (!button) return;

        if (loading) {
            button.classList.add('loading');
            button.disabled = true;
        } else {
            button.classList.remove('loading');
            button.disabled = false;
        }
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        if (!container) return;

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        container.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    formatDuration(seconds) {
        if (!seconds) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatNumber(num) {
        if (!num) return '0';
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    formatFileSize(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Global app instance
const app = new ViralClipApp();

// Global functions for HTML onclick handlers
function analyzeVideo() {
    app.analyzeVideo();
}

function addClip() {
    app.addClip();
}

function processClips() {
    app.processClips();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ViralClipApp;
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