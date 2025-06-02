
// Social Media Integration UI Components
class SocialMediaManager {
    constructor() {
        this.connectedPlatforms = new Map();
        this.publishingJobs = new Map();
        this.platformInsights = new Map();
        this.socket = null;
        
        this.initializeUI();
        this.loadConnectedPlatforms();
        this.setupWebSocketConnection();
    }

    initializeUI() {
        // Add social media section to main interface
        const socialSection = document.createElement('div');
        socialSection.id = 'social-media-section';
        socialSection.className = 'section social-section';
        socialSection.innerHTML = `
            <div class="section-header">
                <h2>🚀 Social Media Publishing Hub</h2>
                <div class="header-controls">
                    <button id="refresh-platforms" class="btn btn-secondary">
                        <i class="icon-refresh"></i> Refresh
                    </button>
                    <button id="connect-platform" class="btn btn-primary">
                        <i class="icon-plus"></i> Connect Platform
                    </button>
                </div>
            </div>
            
            <div class="social-content">
                <!-- Connected Platforms -->
                <div class="platforms-grid" id="platforms-grid">
                    <div class="platform-placeholder">
                        <div class="placeholder-icon">📱</div>
                        <p>Connect your first social media platform to get started</p>
                        <button class="btn btn-primary connect-first-platform">Connect Platform</button>
                    </div>
                </div>
                
                <!-- Publishing Interface -->
                <div class="publishing-interface" id="publishing-interface" style="display: none;">
                    <div class="publish-form">
                        <h3>📋 Create Publishing Job</h3>
                        
                        <div class="form-group">
                            <label for="publish-title">Title</label>
                            <input type="text" id="publish-title" class="form-control" 
                                   placeholder="Enter your content title">
                        </div>
                        
                        <div class="form-group">
                            <label for="publish-description">Description</label>
                            <textarea id="publish-description" class="form-control" rows="4" 
                                      placeholder="Write your caption here..."></textarea>
                            <div class="character-count">
                                <span id="char-count">0</span> / <span id="char-limit">280</span>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="publish-hashtags">Hashtags</label>
                            <input type="text" id="publish-hashtags" class="form-control" 
                                   placeholder="#viral #trending #content">
                            <small class="form-text">Separate hashtags with spaces</small>
                        </div>
                        
                        <div class="form-group">
                            <label>Target Platforms</label>
                            <div class="platform-selector" id="platform-selector">
                                <!-- Platform checkboxes will be populated dynamically -->
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="optimization-level">Optimization Level</label>
                            <select id="optimization-level" class="form-control">
                                <option value="netflix_grade">Netflix Grade (Recommended)</option>
                                <option value="enterprise">Enterprise</option>
                                <option value="advanced">Advanced</option>
                                <option value="standard">Standard</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>
                                <input type="checkbox" id="auto-schedule"> 
                                Auto-schedule for optimal timing
                            </label>
                        </div>
                        
                        <div class="form-group" id="schedule-group" style="display: none;">
                            <label for="scheduled-time">Schedule Time</label>
                            <input type="datetime-local" id="scheduled-time" class="form-control">
                        </div>
                        
                        <div class="publish-actions">
                            <button id="predict-performance" class="btn btn-secondary">
                                🎯 Predict Performance
                            </button>
                            <button id="submit-publish" class="btn btn-primary">
                                🚀 Publish Now
                            </button>
                        </div>
                    </div>
                    
                    <!-- Performance Prediction -->
                    <div class="performance-prediction" id="performance-prediction" style="display: none;">
                        <h4>📊 Performance Prediction</h4>
                        <div class="prediction-grid" id="prediction-grid"></div>
                    </div>
                </div>
                
                <!-- Publishing Jobs -->
                <div class="jobs-section" id="jobs-section">
                    <h3>📋 Publishing Jobs</h3>
                    <div class="jobs-list" id="jobs-list">
                        <div class="no-jobs">
                            <p>No publishing jobs yet. Create your first job above!</p>
                        </div>
                    </div>
                </div>
                
                <!-- Analytics Dashboard -->
                <div class="analytics-section" id="analytics-section">
                    <h3>📈 Analytics Dashboard</h3>
                    <div class="analytics-grid" id="analytics-grid"></div>
                </div>
            </div>
        `;

        // Insert after video processing section
        const videoSection = document.querySelector('#video-processing');
        if (videoSection) {
            videoSection.parentNode.insertBefore(socialSection, videoSection.nextSibling);
        } else {
            document.querySelector('main').appendChild(socialSection);
        }

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Platform connection
        document.getElementById('connect-platform').addEventListener('click', () => {
            this.showPlatformConnectionModal();
        });

        document.querySelector('.connect-first-platform')?.addEventListener('click', () => {
            this.showPlatformConnectionModal();
        });

        // Publishing form
        document.getElementById('publish-description').addEventListener('input', (e) => {
            this.updateCharacterCount(e.target.value);
        });

        document.getElementById('auto-schedule').addEventListener('change', (e) => {
            const scheduleGroup = document.getElementById('schedule-group');
            scheduleGroup.style.display = e.target.checked ? 'none' : 'block';
        });

        document.getElementById('predict-performance').addEventListener('click', () => {
            this.predictPerformance();
        });

        document.getElementById('submit-publish').addEventListener('click', () => {
            this.submitPublishingJob();
        });

        // Refresh platforms
        document.getElementById('refresh-platforms').addEventListener('click', () => {
            this.loadConnectedPlatforms();
        });
    }

    async loadConnectedPlatforms() {
        try {
            showLoadingState('Loading connected platforms...');
            
            const response = await fetch('/api/v6/social/platforms', {
                headers: getAuthHeaders()
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.renderConnectedPlatforms(data.connected_platforms);
                this.updatePlatformSelector(data.connected_platforms);
                
                if (data.connected_platforms.length > 0) {
                    document.getElementById('publishing-interface').style.display = 'block';
                    document.querySelector('.platform-placeholder').style.display = 'none';
                }
            } else {
                throw new Error(data.error || 'Failed to load platforms');
            }
            
        } catch (error) {
            console.error('Failed to load platforms:', error);
            showNotification('Failed to load connected platforms', 'error');
        } finally {
            hideLoadingState();
        }
    }

    renderConnectedPlatforms(platforms) {
        const grid = document.getElementById('platforms-grid');
        
        if (platforms.length === 0) {
            grid.innerHTML = `
                <div class="platform-placeholder">
                    <div class="placeholder-icon">📱</div>
                    <p>Connect your first social media platform to get started</p>
                    <button class="btn btn-primary connect-first-platform">Connect Platform</button>
                </div>
            `;
            return;
        }

        grid.innerHTML = platforms.map(platform => `
            <div class="platform-card ${platform.needs_refresh ? 'needs-refresh' : ''}">
                <div class="platform-header">
                    <div class="platform-icon">${this.getPlatformIcon(platform.platform)}</div>
                    <div class="platform-info">
                        <h4>${this.getPlatformDisplayName(platform.platform)}</h4>
                        <p class="platform-username">${platform.account_username}</p>
                    </div>
                    <div class="platform-status">
                        ${platform.needs_refresh ? 
                            '<span class="status-badge refresh">Needs Refresh</span>' :
                            '<span class="status-badge connected">Connected</span>'
                        }
                    </div>
                </div>
                
                <div class="platform-stats">
                    <div class="stat">
                        <span class="stat-label">Account Type</span>
                        <span class="stat-value">${platform.is_business_account ? 'Business' : 'Personal'}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Connected</span>
                        <span class="stat-value">${this.formatDate(platform.connected_at)}</span>
                    </div>
                </div>
                
                <div class="platform-actions">
                    ${platform.needs_refresh ? 
                        `<button class="btn btn-warning refresh-tokens" data-platform="${platform.platform}">
                            <i class="icon-refresh"></i> Refresh
                        </button>` : ''
                    }
                    <button class="btn btn-secondary view-insights" data-platform="${platform.platform}">
                        <i class="icon-chart"></i> Insights
                    </button>
                    <button class="btn btn-danger disconnect-platform" data-platform="${platform.platform}">
                        <i class="icon-trash"></i> Disconnect
                    </button>
                </div>
            </div>
        `).join('');

        // Add event listeners to platform cards
        grid.querySelectorAll('.view-insights').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.showPlatformInsights(e.target.dataset.platform);
            });
        });

        grid.querySelectorAll('.disconnect-platform').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.disconnectPlatform(e.target.dataset.platform);
            });
        });
    }

    updatePlatformSelector(platforms) {
        const selector = document.getElementById('platform-selector');
        
        selector.innerHTML = platforms.map(platform => `
            <label class="platform-checkbox">
                <input type="checkbox" value="${platform.platform}" 
                       ${!platform.needs_refresh ? 'checked' : 'disabled'}>
                <span class="platform-label">
                    ${this.getPlatformIcon(platform.platform)} 
                    ${this.getPlatformDisplayName(platform.platform)}
                    ${platform.needs_refresh ? ' (Refresh Required)' : ''}
                </span>
            </label>
        `).join('');
    }

    showPlatformConnectionModal() {
        const modal = document.createElement('div');
        modal.className = 'modal social-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>🔗 Connect Social Media Platform</h3>
                    <button class="modal-close">&times;</button>
                </div>
                
                <div class="modal-body">
                    <p>Choose a platform to connect:</p>
                    
                    <div class="platform-options">
                        <div class="platform-option" data-platform="tiktok">
                            <div class="platform-icon">🎵</div>
                            <div class="platform-name">TikTok</div>
                            <div class="platform-description">Short viral videos</div>
                        </div>
                        
                        <div class="platform-option" data-platform="instagram">
                            <div class="platform-icon">📷</div>
                            <div class="platform-name">Instagram</div>
                            <div class="platform-description">Photos and Reels</div>
                        </div>
                        
                        <div class="platform-option" data-platform="youtube_shorts">
                            <div class="platform-icon">📺</div>
                            <div class="platform-name">YouTube Shorts</div>
                            <div class="platform-description">Short-form videos</div>
                        </div>
                        
                        <div class="platform-option" data-platform="twitter">
                            <div class="platform-icon">🐦</div>
                            <div class="platform-name">Twitter/X</div>
                            <div class="platform-description">Microblogging</div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close modal
        modal.querySelector('.modal-close').addEventListener('click', () => {
            modal.remove();
        });

        // Platform selection
        modal.querySelectorAll('.platform-option').forEach(option => {
            option.addEventListener('click', () => {
                this.initiateOAuthFlow(option.dataset.platform);
                modal.remove();
            });
        });
    }

    async initiateOAuthFlow(platform) {
        try {
            showNotification(`Connecting to ${this.getPlatformDisplayName(platform)}...`, 'info');
            
            // Simulate OAuth flow - in production, this would redirect to actual OAuth
            const authCode = `mock_auth_code_${Date.now()}`;
            const redirectUri = `${window.location.origin}/oauth/callback`;
            
            const response = await fetch('/api/v6/social/authenticate', {
                method: 'POST',
                headers: {
                    ...getAuthHeaders(),
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    platform: platform,
                    auth_code: authCode,
                    redirect_uri: redirectUri
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                showNotification(`Successfully connected ${data.display_name}!`, 'success');
                this.loadConnectedPlatforms(); // Refresh the platform list
            } else {
                throw new Error(data.error || 'Authentication failed');
            }
            
        } catch (error) {
            console.error('OAuth flow failed:', error);
            showNotification(`Failed to connect ${this.getPlatformDisplayName(platform)}: ${error.message}`, 'error');
        }
    }

    updateCharacterCount(text) {
        const selectedPlatforms = this.getSelectedPlatforms();
        let maxLength = 280; // Default Twitter limit
        
        // Find the most restrictive platform
        if (selectedPlatforms.includes('twitter')) {
            maxLength = 280;
        } else if (selectedPlatforms.includes('instagram') || selectedPlatforms.includes('tiktok')) {
            maxLength = 2200;
        } else if (selectedPlatforms.includes('youtube_shorts')) {
            maxLength = 5000;
        }
        
        document.getElementById('char-count').textContent = text.length;
        document.getElementById('char-limit').textContent = maxLength;
        
        const countElement = document.getElementById('char-count');
        if (text.length > maxLength) {
            countElement.style.color = '#ff4444';
        } else if (text.length > maxLength * 0.9) {
            countElement.style.color = '#ff8800';
        } else {
            countElement.style.color = '#4CAF50';
        }
    }

    getSelectedPlatforms() {
        const checkboxes = document.querySelectorAll('#platform-selector input[type="checkbox"]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }

    async predictPerformance() {
        try {
            const platforms = this.getSelectedPlatforms();
            const caption = document.getElementById('publish-description').value;
            const hashtags = document.getElementById('publish-hashtags').value.split(' ').filter(h => h.trim());
            
            if (platforms.length === 0) {
                showNotification('Please select at least one platform', 'warning');
                return;
            }
            
            showLoadingState('Predicting performance...');
            
            const response = await fetch('/api/v6/social/content/predict', {
                method: 'POST',
                headers: {
                    ...getAuthHeaders(),
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_path: getCurrentVideoPath(), // From main app state
                    platforms: platforms,
                    caption: caption,
                    hashtags: hashtags
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.renderPerformancePrediction(data);
                document.getElementById('performance-prediction').style.display = 'block';
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
            
        } catch (error) {
            console.error('Performance prediction failed:', error);
            showNotification('Performance prediction failed', 'error');
        } finally {
            hideLoadingState();
        }
    }

    renderPerformancePrediction(data) {
        const grid = document.getElementById('prediction-grid');
        
        grid.innerHTML = `
            <div class="prediction-summary">
                <h5>📊 Overall Prediction</h5>
                <div class="overall-metrics">
                    <div class="metric">
                        <span class="metric-value">${data.overall_metrics.average_engagement.toFixed(1)}%</span>
                        <span class="metric-label">Avg Engagement</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">${this.formatNumber(data.overall_metrics.total_predicted_views)}</span>
                        <span class="metric-label">Total Views</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">${(data.overall_metrics.confidence * 100).toFixed(0)}%</span>
                        <span class="metric-label">Confidence</span>
                    </div>
                </div>
            </div>
            
            <div class="platform-predictions">
                ${Object.entries(data.platform_predictions).map(([platform, prediction]) => {
                    if (!prediction.success) return '';
                    
                    const pred = prediction.predictions;
                    return `
                        <div class="platform-prediction">
                            <div class="platform-header">
                                ${this.getPlatformIcon(platform)} 
                                ${this.getPlatformDisplayName(platform)}
                            </div>
                            
                            <div class="prediction-metrics">
                                <div class="metric-row">
                                    <span class="metric-label">👀 Views</span>
                                    <span class="metric-value">${this.formatNumber(pred.predicted_views)}</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">❤️ Likes</span>
                                    <span class="metric-value">${this.formatNumber(pred.predicted_likes)}</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">🔄 Shares</span>
                                    <span class="metric-value">${this.formatNumber(pred.predicted_shares)}</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">💬 Comments</span>
                                    <span class="metric-value">${this.formatNumber(pred.predicted_comments)}</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">🔥 Viral Score</span>
                                    <span class="metric-value">${(pred.viral_probability * 100).toFixed(0)}%</span>
                                </div>
                            </div>
                            
                            ${pred.recommendations.length > 0 ? `
                                <div class="recommendations">
                                    <h6>💡 Recommendations</h6>
                                    <ul>
                                        ${pred.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    async submitPublishingJob() {
        try {
            const platforms = this.getSelectedPlatforms();
            const title = document.getElementById('publish-title').value;
            const description = document.getElementById('publish-description').value;
            const hashtags = document.getElementById('publish-hashtags').value.split(' ').filter(h => h.trim());
            const optimizationLevel = document.getElementById('optimization-level').value;
            const autoSchedule = document.getElementById('auto-schedule').checked;
            const scheduledTime = document.getElementById('scheduled-time').value;
            
            // Validation
            if (platforms.length === 0) {
                showNotification('Please select at least one platform', 'warning');
                return;
            }
            
            if (!title.trim() || !description.trim()) {
                showNotification('Please fill in title and description', 'warning');
                return;
            }
            
            showLoadingState('Submitting publishing job...');
            
            const jobData = {
                session_id: getCurrentSessionId(),
                platforms: platforms,
                video_path: getCurrentVideoPath(),
                title: title,
                description: description,
                hashtags: hashtags,
                optimization_level: optimizationLevel
            };
            
            if (!autoSchedule && scheduledTime) {
                jobData.scheduled_time = new Date(scheduledTime).toISOString();
            }
            
            const response = await fetch('/api/v6/social/publish', {
                method: 'POST',
                headers: {
                    ...getAuthHeaders(),
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(jobData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                showNotification(`Publishing job submitted! Job ID: ${data.job_id}`, 'success');
                this.trackPublishingJob(data.job_id);
                this.clearPublishingForm();
                this.loadPublishingJobs();
            } else {
                throw new Error(data.error || 'Job submission failed');
            }
            
        } catch (error) {
            console.error('Publishing job submission failed:', error);
            showNotification(`Publishing failed: ${error.message}`, 'error');
        } finally {
            hideLoadingState();
        }
    }

    trackPublishingJob(jobId) {
        // Start tracking the job status
        const trackingInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/v6/social/jobs/${jobId}`, {
                    headers: getAuthHeaders()
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const job = data.job;
                    this.updateJobStatus(jobId, job);
                    
                    // Stop tracking if job is complete
                    if (['published', 'failed', 'cancelled', 'partial_success'].includes(job.status)) {
                        clearInterval(trackingInterval);
                        
                        if (job.status === 'published') {
                            showNotification('Content published successfully! 🎉', 'success');
                        } else if (job.status === 'partial_success') {
                            showNotification('Content published to some platforms', 'warning');
                        } else if (job.status === 'failed') {
                            showNotification('Publishing failed', 'error');
                        }
                    }
                }
            } catch (error) {
                console.error('Job tracking error:', error);
            }
        }, 2000); // Check every 2 seconds
        
        // Store interval for cleanup
        this.publishingJobs.set(jobId, { interval: trackingInterval });
    }

    clearPublishingForm() {
        document.getElementById('publish-title').value = '';
        document.getElementById('publish-description').value = '';
        document.getElementById('publish-hashtags').value = '';
        document.getElementById('performance-prediction').style.display = 'none';
        this.updateCharacterCount('');
    }

    // Utility methods
    getPlatformIcon(platform) {
        const icons = {
            tiktok: '🎵',
            instagram: '📷',
            youtube_shorts: '📺',
            twitter: '🐦',
            facebook: '📘',
            linkedin: '💼',
            snapchat: '👻',
            pinterest: '📌'
        };
        return icons[platform] || '📱';
    }

    getPlatformDisplayName(platform) {
        const names = {
            tiktok: 'TikTok',
            instagram: 'Instagram',
            youtube_shorts: 'YouTube Shorts',
            twitter: 'Twitter/X',
            facebook: 'Facebook',
            linkedin: 'LinkedIn',
            snapchat: 'Snapchat',
            pinterest: 'Pinterest'
        };
        return names[platform] || platform.charAt(0).toUpperCase() + platform.slice(1);
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    formatDate(dateString) {
        return new Date(dateString).toLocaleDateString();
    }
}

// Initialize social media manager when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (typeof window.socialMediaManager === 'undefined') {
        window.socialMediaManager = new SocialMediaManager();
    }
});

// Helper functions to integrate with main app
function getCurrentVideoPath() {
    // Return current video path from main app state
    return window.currentVideoPath || '/mock/video/path.mp4';
}

function getCurrentSessionId() {
    // Return current session ID from main app state
    return window.currentSessionId || 'session_' + Date.now();
}
