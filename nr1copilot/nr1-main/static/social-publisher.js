/**
 * ViralClip Pro - Social Media Publishing Hub
 * Netflix-level social publishing interface
 */

class SocialPublishingHub {
    constructor() {
        this.connectedPlatforms = new Set();
        this.publishingJobs = new Map();
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadConnectedPlatforms();
        this.setupRealtimeUpdates();
    }

    setupEventListeners() {
        // Platform authentication
        document.addEventListener('click', (e) => {
            if (e.target.matches('.connect-platform-btn')) {
                this.connectPlatform(e.target.dataset.platform);
            }

            if (e.target.matches('.publish-btn')) {
                this.publishContent();
            }

            if (e.target.matches('.generate-caption-btn')) {
                this.generateCaption();
            }

            if (e.target.matches('.generate-thumbnail-btn')) {
                this.generateThumbnail();
            }

            if (e.target.matches('.predict-performance-btn')) {
                this.predictPerformance();
            }
        });

        // Form submissions
        const publishForm = document.getElementById('publishForm');
        if (publishForm) {
            publishForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.publishContent();
            });
        }
    }

    async connectPlatform(platform) {
        try {
            this.showLoading(`Connecting to ${platform}...`);

            // Generate OAuth URL (simplified - in production use proper OAuth flow)
            const authUrl = this.generateAuthUrl(platform);

            // Open OAuth popup
            const popup = window.open(authUrl, 'oauth', 'width=500,height=600');

            // Listen for OAuth callback
            const authCode = await this.waitForAuthCallback(popup);

            // Exchange auth code for tokens
            const response = await fetch('/api/social/authenticate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    platform: platform,
                    auth_code: authCode,
                    user_id: this.getCurrentUserId(),
                    redirect_uri: `${window.location.origin}/oauth/callback`
                })
            });

            const result = await response.json();

            if (result.success) {
                this.connectedPlatforms.add(platform);
                this.updatePlatformUI(platform, true);
                this.showSuccess(`Successfully connected to ${result.display_name}!`);
            } else {
                this.showError(`Failed to connect to ${platform}: ${result.error}`);
            }

        } catch (error) {
            this.showError(`Connection failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async publishContent() {
        try {
            const formData = this.collectPublishingData();

            if (!this.validatePublishingData(formData)) {
                return;
            }

            this.showLoading('Publishing content...');

            const response = await fetch('/api/social/publish', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.publishingJobs.set(result.job_id, {
                    ...result,
                    startTime: Date.now()
                });

                this.updatePublishingStatus(result.job_id, 'queued');
                this.showSuccess(`Content queued for publishing! Job ID: ${result.job_id}`);
                this.trackPublishingProgress(result.job_id);

            } else {
                this.showError(`Publishing failed: ${result.error}`);
            }

        } catch (error) {
            this.showError(`Publishing error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async generateCaption() {
        try {
            const contentDescription = document.getElementById('contentDescription').value;
            const platform = document.getElementById('primaryPlatform').value;
            const tone = document.getElementById('captionTone').value || 'engaging';

            if (!contentDescription) {
                this.showError('Please describe your content first');
                return;
            }

            this.showLoading('Generating viral caption...');

            const formData = new FormData();
            formData.append('content_description', contentDescription);
            formData.append('platform', platform);
            formData.append('tone', tone);
            formData.append('target_audience', 'gen_z');
            formData.append('include_hashtags', 'true');

            const response = await fetch('/api/social/caption-generator', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                document.getElementById('description').value = result.caption;
                document.getElementById('hashtags').value = result.hashtags.join(', ');

                // Show viral score
                this.displayViralScore(result.viral_score);
                this.showSuccess(`Generated viral caption with ${(result.viral_score * 100).toFixed(1)}% viral score!`);

            } else {
                this.showError(`Caption generation failed: ${result.error}`);
            }

        } catch (error) {
            this.showError(`Caption generation error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async generateThumbnail() {
        try {
            const videoPath = document.getElementById('videoPath').value;
            const platform = document.getElementById('primaryPlatform').value;

            if (!videoPath) {
                this.showError('Please select a video first');
                return;
            }

            this.showLoading('Generating smart thumbnail...');

            const formData = new FormData();
            formData.append('video_path', videoPath);
            formData.append('platform', platform);
            formData.append('style', 'high_engagement');

            const response = await fetch('/api/social/thumbnail-generator', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.displayThumbnail(result.thumbnail_url, result.engagement_score);
                this.showSuccess(`Generated thumbnail with ${(result.engagement_score * 100).toFixed(1)}% engagement score!`);

            } else {
                this.showError(`Thumbnail generation failed: ${result.error}`);
            }

        } catch (error) {
            this.showError(`Thumbnail generation error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async predictPerformance() {
        try {
            const formData = this.collectPublishingData();

            this.showLoading('Predicting performance...');

            const predictionData = new FormData();
            predictionData.append('video_path', formData.get('video_path'));
            predictionData.append('caption', formData.get('description'));
            predictionData.append('hashtags', formData.get('hashtags'));
            predictionData.append('platform', formData.get('platforms').split(',')[0]);
            predictionData.append('posting_time', new Date().toISOString());

            const response = await fetch('/api/social/performance-predictor', {
                method: 'POST',
                body: predictionData
            });

            const result = await response.json();

            if (result.success) {
                this.displayPerformancePrediction(result);
                this.showSuccess(`Performance predicted with ${(result.confidence * 100).toFixed(1)}% confidence!`);

            } else {
                this.showError(`Performance prediction failed: ${result.error}`);
            }

        } catch (error) {
            this.showError(`Performance prediction error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async trackPublishingProgress(jobId) {
        const maxAttempts = 60; // 5 minutes max
        let attempts = 0;

        const checkStatus = async () => {
            try {
                const response = await fetch(`/api/social/job/${jobId}`);
                const result = await response.json();

                if (result.success) {
                    const job = result.job;
                    this.updatePublishingStatus(jobId, job.status);

                    if (job.status === 'published') {
                        this.showPublishingSuccess(job);
                        return;
                    } else if (job.status === 'failed') {
                        this.showPublishingError(job);
                        return;
                    }
                }

                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(checkStatus, 5000); // Check every 5 seconds
                }

            } catch (error) {
                console.error('Status check failed:', error);
            }
        };

        checkStatus();
    }

    collectPublishingData() {
        const formData = new FormData();

        // Get selected platforms
        const selectedPlatforms = Array.from(document.querySelectorAll('.platform-checkbox:checked'))
            .map(cb => cb.value);

        formData.append('session_id', this.generateSessionId());
        formData.append('user_id', this.getCurrentUserId());
        formData.append('platforms', selectedPlatforms.join(','));
        formData.append('video_path', document.getElementById('videoPath').value);
        formData.append('title', document.getElementById('title').value);
        formData.append('description', document.getElementById('description').value);
        formData.append('hashtags', document.getElementById('hashtags').value);
        formData.append('priority', document.getElementById('priority').value || '5');
        formData.append('optimization_level', 'netflix_grade');

        // Scheduled time if specified
        const scheduledTime = document.getElementById('scheduledTime').value;
        if (scheduledTime) {
            formData.append('scheduled_time', scheduledTime);
        }

        return formData;
    }

    validatePublishingData(formData) {
        const required = ['platforms', 'video_path', 'title', 'description'];

        for (const field of required) {
            if (!formData.get(field)) {
                this.showError(`Please fill in the ${field.replace('_', ' ')} field`);
                return false;
            }
        }

        const platforms = formData.get('platforms').split(',');
        const connectedPlatforms = Array.from(this.connectedPlatforms);

        for (const platform of platforms) {
            if (!connectedPlatforms.includes(platform)) {
                this.showError(`Please connect to ${platform} first`);
                return false;
            }
        }

        return true;
    }

    displayViralScore(score) {
        const scoreElement = document.getElementById('viralScore');
        if (scoreElement) {
            const percentage = (score * 100).toFixed(1);
            scoreElement.textContent = `${percentage}%`;
            scoreElement.className = `viral-score ${this.getScoreClass(score)}`;
        }
    }

    displayThumbnail(thumbnailUrl, engagementScore) {
        const thumbnailContainer = document.getElementById('thumbnailPreview');
        if (thumbnailContainer) {
            thumbnailContainer.innerHTML = `
                <img src="${thumbnailUrl}" alt="Generated thumbnail" class="thumbnail-preview">
                <div class="engagement-score">Engagement Score: ${(engagementScore * 100).toFixed(1)}%</div>
            `;
        }
    }

    displayPerformancePrediction(prediction) {
        const predictionContainer = document.getElementById('performancePrediction');
        if (predictionContainer) {
            predictionContainer.innerHTML = `
                <div class="prediction-card">
                    <h4>Performance Prediction</h4>
                    <div class="prediction-metrics">
                        <div class="metric">
                            <span class="label">Viral Probability:</span>
                            <span class="value">${(prediction.viral_probability * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span class="label">Estimated Reach:</span>
                            <span class="value">${prediction.estimated_reach.toLocaleString()}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Predicted Likes:</span>
                            <span class="value">${prediction.engagement_prediction.likes.toLocaleString()}</span>
                        </div>
                    </div>
                    ${prediction.recommendations.length > 0 ? `
                        <div class="recommendations">
                            <h5>Recommendations:</h5>
                            <ul>
                                ${prediction.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `;
        }
    }

    updatePublishingStatus(jobId, status) {
        const statusElement = document.getElementById(`status-${jobId}`);
        if (statusElement) {
            statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            statusElement.className = `status ${status}`;
        }
    }

    showPublishingSuccess(job) {
        const successHtml = `
            <div class="publishing-success">
                <h4>✅ Content Published Successfully!</h4>
                <div class="published-links">
                    ${Object.entries(job.published_urls).map(([platform, url]) => 
                        `<a href="${url}" target="_blank" class="platform-link">${platform}</a>`
                    ).join('')}
                </div>
                <div class="job-stats">
                    <span>Duration: ${job.duration}s</span>
                    <span>Success Rate: ${(job.success_rate * 100).toFixed(1)}%</span>
                </div>
            </div>
        `;

        this.showMessage(successHtml, 'success');
    }

    showPublishingError(job) {
        this.showError(`Publishing failed: ${job.error_details}`);
    }

    generateAuthUrl(platform) {
        // Simplified - in production, generate proper OAuth URLs
        return `https://oauth.${platform}.com/authorize?client_id=your_client_id&redirect_uri=${encodeURIComponent(window.location.origin + '/oauth/callback')}&scope=publish`;
    }

    async waitForAuthCallback(popup) {
        return new Promise((resolve, reject) => {
            const checkClosed = setInterval(() => {
                if (popup.closed) {
                    clearInterval(checkClosed);
                    reject(new Error('OAuth popup was closed'));
                }
            }, 1000);

            window.addEventListener('message', (event) => {
                if (event.origin !== window.location.origin) return;

                if (event.data.type === 'oauth-success') {
                    clearInterval(checkClosed);
                    popup.close();
                    resolve(event.data.code);
                } else if (event.data.type === 'oauth-error') {
                    clearInterval(checkClosed);
                    popup.close();
                    reject(new Error(event.data.error));
                }
            });
        });
    }

    updatePlatformUI(platform, connected) {
        const platformBtn = document.querySelector(`[data-platform="${platform}"]`);
        if (platformBtn) {
            platformBtn.textContent = connected ? '✅ Connected' : 'Connect';
            platformBtn.disabled = connected;
            platformBtn.classList.toggle('connected', connected);
        }

        const platformCheckbox = document.querySelector(`input[value="${platform}"]`);
        if (platformCheckbox) {
            platformCheckbox.disabled = !connected;
        }
    }

    loadConnectedPlatforms() {
        // Load from localStorage or API
        const stored = localStorage.getItem('connectedPlatforms');
        if (stored) {
            this.connectedPlatforms = new Set(JSON.parse(stored));
            this.connectedPlatforms.forEach(platform => {
                this.updatePlatformUI(platform, true);
            });
        }
    }

    setupRealtimeUpdates() {
        // WebSocket connection for real-time updates
        if (typeof io !== 'undefined') {
            const socket = io();

            socket.on('publishing-update', (data) => {
                if (this.publishingJobs.has(data.job_id)) {
                    this.updatePublishingStatus(data.job_id, data.status);
                }
            });
        }
    }

    getCurrentUserId() {
        return localStorage.getItem('userId') || 'user_' + Math.random().toString(36).substr(2, 9);
    }

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    getScoreClass(score) {
        if (score >= 0.9) return 'excellent';
        if (score >= 0.8) return 'good';
        if (score >= 0.7) return 'average';
        return 'poor';
    }

    showLoading(message) {
        const loader = document.getElementById('loadingIndicator');
        if (loader) {
            loader.textContent = message;
            loader.style.display = 'block';
        }
    }

    hideLoading() {
        const loader = document.getElementById('loadingIndicator');
        if (loader) {
            loader.style.display = 'none';
        }
    }

    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showMessage(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.socialPublisher = new SocialPublishingHub();
});