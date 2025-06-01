
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

class ViralClipGenerator {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 5;
        this.videoData = null;
        this.analysisData = null;
        this.processingTaskId = null;
        this.processingInterval = null;
        this.selectedClips = [];
        this.supportedPlatforms = [
            'tiktok', 'instagram_reels', 'instagram_story', 'instagram_feed',
            'youtube_shorts', 'youtube_standard', 'twitter', 'facebook',
            'linkedin', 'snapchat', 'pinterest'
        ];
        
        this.init();
    }

    async init() {
        console.log('🎬 Initializing ViralClip Pro - Netflix Level');
        
        await this.setupEventListeners();
        await this.setupServiceWorker();
        await this.setupProgressiveEnhancement();
        await this.loadUserPreferences();
        await this.initializePerformanceMonitoring();
        
        // Show loading screen initially
        await this.hideLoadingScreen();
        
        console.log('✅ ViralClip Pro initialized successfully');
    }

    async setupEventListeners() {
        // Form submissions
        const urlForm = document.getElementById('url-form');
        if (urlForm) {
            urlForm.addEventListener('submit', (e) => this.handleUrlSubmit(e));
        }

        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        }

        // Navigation
        const nextBtn = document.getElementById('next-btn');
        const prevBtn = document.getElementById('prev-btn');
        const processBtn = document.getElementById('process-btn');

        if (nextBtn) nextBtn.addEventListener('click', () => this.nextStep());
        if (prevBtn) prevBtn.addEventListener('click', () => this.prevStep());
        if (processBtn) processBtn.addEventListener('click', () => this.startProcessing());

        // Platform selection
        this.setupPlatformSelection();

        // Clip management
        this.setupClipManagement();

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // Real-time updates
        this.setupWebSocketConnection();
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

    selectPlatform(platform, cardElement) {
        // Remove previous selections
        document.querySelectorAll('.platform-card').forEach(card => {
            card.classList.remove('selected');
        });

        // Select current platform
        cardElement.classList.add('selected');
        
        // Update clip suggestions based on platform
        this.updateClipSuggestions(platform);
        
        this.showToast(`Selected ${this.getPlatformData(platform).name}`, 'success');
    }

    setupClipManagement() {
        const addClipBtn = document.getElementById('add-clip-btn');
        if (addClipBtn) {
            addClipBtn.addEventListener('click', () => this.addCustomClip());
        }
    }

    async handleUrlSubmit(e) {
        e.preventDefault();
        
        const urlInput = document.getElementById('url-input');
        const url = urlInput.value.trim();
        
        if (!url) {
            this.showToast('Please enter a valid URL', 'error');
            return;
        }

        this.showLoading('Analyzing video... 🎯');
        
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
                this.videoData = data.video_info;
                this.analysisData = data.ai_insights;
                
                this.displayAnalysisResults(data);
                this.updatePlatformSuitability(data.ai_insights);
                this.generateClipSuggestions(data.ai_insights);
                
                this.goToStep(2);
                this.showToast('Video analyzed successfully! 🚀', 'success');
            } else {
                throw new Error(data.message || 'Analysis failed');
            }
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showToast(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayAnalysisResults(data) {
        const resultsContainer = document.getElementById('analysis-results');
        if (!resultsContainer) return;

        const videoInfo = data.video_info;
        const aiInsights = data.ai_insights;
        
        resultsContainer.innerHTML = `
            <div class="analysis-grid">
                <div class="analysis-card video-info-card">
                    <h3>📹 Video Information</h3>
                    <div class="video-meta">
                        <div class="video-thumbnail">
                            <img src="${videoInfo.thumbnail}" alt="Video thumbnail" loading="lazy">
                            <div class="duration-badge">${this.formatDuration(videoInfo.duration)}</div>
                        </div>
                        <div class="video-details">
                            <h4>${videoInfo.title}</h4>
                            <div class="meta-row">
                                <span class="meta-label">👤 Uploader:</span>
                                <span class="meta-value">${videoInfo.uploader}</span>
                            </div>
                            <div class="meta-row">
                                <span class="meta-label">👀 Views:</span>
                                <span class="meta-value">${this.formatNumber(videoInfo.view_count)}</span>
                            </div>
                            <div class="meta-row">
                                <span class="meta-label">👍 Likes:</span>
                                <span class="meta-value">${this.formatNumber(videoInfo.like_count)}</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="analysis-card ai-insights-card">
                    <h3>🤖 AI Insights</h3>
                    <div class="insights-grid">
                        <div class="insight-item">
                            <div class="insight-label">Viral Potential</div>
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${aiInsights.viral_potential}%"></div>
                                <span class="score-text">${aiInsights.viral_potential}%</span>
                            </div>
                        </div>
                        <div class="insight-item">
                            <div class="insight-label">Engagement Score</div>
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${aiInsights.engagement_prediction}%"></div>
                                <span class="score-text">${aiInsights.engagement_prediction}%</span>
                            </div>
                        </div>
                        <div class="insight-item">
                            <div class="insight-label">Optimal Length</div>
                            <div class="insight-value">${aiInsights.optimal_length}s</div>
                        </div>
                        <div class="insight-item">
                            <div class="insight-label">Sentiment</div>
                            <div class="sentiment-badge ${aiInsights.sentiment_analysis}">${aiInsights.sentiment_analysis}</div>
                        </div>
                    </div>
                </div>

                <div class="analysis-card trending-topics-card">
                    <h3>🔥 Trending Topics</h3>
                    <div class="trending-tags">
                        ${aiInsights.trending_topics.map(topic => 
                            `<span class="trending-tag">#${topic}</span>`
                        ).join('')}
                    </div>
                </div>

                <div class="analysis-card hook-moments-card">
                    <h3>🎣 Hook Moments</h3>
                    <div class="hook-timeline">
                        ${aiInsights.hook_moments.map(moment => 
                            `<div class="hook-point" style="left: ${(moment / videoInfo.duration) * 100}%">
                                <div class="hook-tooltip">
                                    <strong>Hook at ${this.formatDuration(moment)}</strong>
                                    <p>High engagement potential</p>
                                </div>
                            </div>`
                        ).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    updatePlatformSuitability(aiInsights) {
        this.supportedPlatforms.forEach(platform => {
            const scoreElement = document.getElementById(`score-${platform}`);
            if (scoreElement) {
                // Calculate platform-specific suitability score
                const score = this.calculatePlatformScore(platform, aiInsights);
                
                const scoreValue = scoreElement.querySelector('.score-value');
                scoreValue.textContent = `${score}%`;
                scoreValue.className = `score-value ${this.getScoreClass(score)}`;
            }
        });
    }

    calculatePlatformScore(platform, aiInsights) {
        let baseScore = aiInsights.viral_potential;
        
        // Platform-specific adjustments
        const platformBoosts = {
            'tiktok': aiInsights.viral_potential > 80 ? 15 : 5,
            'instagram_reels': aiInsights.engagement_prediction > 75 ? 12 : 3,
            'youtube_shorts': aiInsights.viral_potential > 70 ? 10 : 2,
            'twitter': aiInsights.sentiment_analysis === 'positive' ? 8 : 0,
            'facebook': baseScore > 60 ? 5 : 0,
            'linkedin': aiInsights.sentiment_analysis === 'professional' ? 15 : -5
        };
        
        const boost = platformBoosts[platform] || 0;
        return Math.min(Math.max(baseScore + boost, 0), 100);
    }

    getScoreClass(score) {
        if (score >= 80) return 'score-excellent';
        if (score >= 60) return 'score-good';
        if (score >= 40) return 'score-fair';
        return 'score-poor';
    }

    generateClipSuggestions(aiInsights) {
        const clipsContainer = document.getElementById('suggested-clips');
        if (!clipsContainer) return;

        const bestClips = aiInsights.best_clips || [];
        
        clipsContainer.innerHTML = `
            <div class="clips-header">
                <h3>✨ AI-Generated Clip Suggestions</h3>
                <p>Based on viral patterns and engagement analysis</p>
            </div>
            <div class="clips-grid">
                ${bestClips.map((clip, index) => `
                    <div class="clip-suggestion" data-clip-index="${index}">
                        <div class="clip-preview">
                            <div class="clip-timeline">
                                <div class="clip-range" style="left: ${(clip.start / this.videoData.duration) * 100}%; width: ${((clip.end - clip.start) / this.videoData.duration) * 100}%"></div>
                            </div>
                            <div class="clip-info">
                                <div class="clip-title">Clip ${index + 1}</div>
                                <div class="clip-duration">${this.formatDuration(clip.end - clip.start)}</div>
                                <div class="clip-score">Viral Score: ${clip.viral_score || 85 + index}%</div>
                            </div>
                        </div>
                        <div class="clip-actions">
                            <button class="btn-clip-action" onclick="app.selectClip(${index})">
                                <span class="icon">✓</span> Select
                            </button>
                            <button class="btn-clip-action" onclick="app.editClip(${index})">
                                <span class="icon">✏️</span> Edit
                            </button>
                            <button class="btn-clip-action" onclick="app.previewClip(${index})">
                                <span class="icon">👁️</span> Preview
                            </button>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    selectClip(clipIndex) {
        const clipElement = document.querySelector(`[data-clip-index="${clipIndex}"]`);
        if (clipElement) {
            clipElement.classList.toggle('selected');
            
            if (clipElement.classList.contains('selected')) {
                this.selectedClips.push(clipIndex);
                this.showToast(`Clip ${clipIndex + 1} added to processing queue`, 'success');
            } else {
                this.selectedClips = this.selectedClips.filter(index => index !== clipIndex);
                this.showToast(`Clip ${clipIndex + 1} removed from queue`, 'info');
            }
            
            this.updateProcessingQueue();
        }
    }

    updateProcessingQueue() {
        const queueContainer = document.getElementById('processing-queue');
        if (!queueContainer) return;
        
        queueContainer.innerHTML = `
            <div class="queue-header">
                <h3>📋 Processing Queue (${this.selectedClips.length} clips)</h3>
            </div>
            <div class="queue-items">
                ${this.selectedClips.map(clipIndex => `
                    <div class="queue-item">
                        <span class="queue-clip-name">Clip ${clipIndex + 1}</span>
                        <button class="btn-remove" onclick="app.removeFromQueue(${clipIndex})">❌</button>
                    </div>
                `).join('')}
            </div>
        `;
        
        // Update process button state
        const processBtn = document.getElementById('process-btn');
        if (processBtn) {
            processBtn.disabled = this.selectedClips.length === 0;
            processBtn.textContent = this.selectedClips.length === 0 ? 
                'Select clips to process' : 
                `Process ${this.selectedClips.length} clip${this.selectedClips.length === 1 ? '' : 's'}`;
        }
    }

    removeFromQueue(clipIndex) {
        this.selectedClips = this.selectedClips.filter(index => index !== clipIndex);
        
        // Update UI
        const clipElement = document.querySelector(`[data-clip-index="${clipIndex}"]`);
        if (clipElement) {
            clipElement.classList.remove('selected');
        }
        
        this.updateProcessingQueue();
        this.showToast(`Clip ${clipIndex + 1} removed from queue`, 'info');
    }

    async startProcessing() {
        if (this.selectedClips.length === 0) {
            this.showToast('Please select at least one clip to process', 'warning');
            return;
        }

        const selectedPlatform = document.querySelector('.platform-card.selected');
        if (!selectedPlatform) {
            this.showToast('Please select a target platform', 'warning');
            return;
        }

        const platform = selectedPlatform.dataset.platform;
        this.showLoading('Starting video processing... 🎬');
        
        try {
            // Prepare clip data
            const clipsData = this.selectedClips.map(clipIndex => {
                const clipData = this.analysisData.best_clips[clipIndex] || {};
                return {
                    start_time: clipData.start || clipIndex * 30,
                    end_time: clipData.end || (clipIndex + 1) * 30,
                    title: `ViralClip Pro - Clip ${clipIndex + 1}`,
                    description: `AI-optimized clip for ${platform}`,
                    tags: this.analysisData.trending_topics || []
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
                throw new Error(data.message || 'Processing failed to start');
            }
            
        } catch (error) {
            console.error('Processing error:', error);
            this.showToast(`Processing failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    startProcessingMonitor() {
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
        }
        
        this.processingInterval = setInterval(async () => {
            await this.checkProcessingStatus();
        }, 2000); // Check every 2 seconds
        
        // Initial check
        this.checkProcessingStatus();
    }

    async checkProcessingStatus() {
        if (!this.processingTaskId) return;
        
        try {
            const response = await fetch(`/api/v2/processing-status/${this.processingTaskId}`);
            const data = await response.json();
            
            if (data.success) {
                this.updateProcessingUI(data.data);
                
                if (data.data.status === 'completed') {
                    this.onProcessingComplete(data.data);
                } else if (data.data.status === 'failed') {
                    this.onProcessingFailed(data.data);
                }
            }
            
        } catch (error) {
            console.error('Status check error:', error);
        }
    }

    updateProcessingUI(taskData) {
        const progressContainer = document.getElementById('processing-progress');
        if (!progressContainer) return;
        
        const progress = taskData.progress || 0;
        const status = taskData.status || 'queued';
        const currentStep = taskData.current_step || 'initializing';
        
        progressContainer.innerHTML = `
            <div class="processing-header">
                <h3>🎬 Processing Your Clips</h3>
                <div class="processing-status ${status}">${this.getStatusText(status)}</div>
            </div>
            
            <div class="progress-bar-container">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${progress}%"></div>
                </div>
                <div class="progress-text">${progress}%</div>
            </div>
            
            <div class="processing-details">
                <div class="current-step">
                    <strong>Current Step:</strong> ${this.formatStepName(currentStep)}
                </div>
                
                ${taskData.queue_position ? `
                    <div class="queue-position">
                        <strong>Queue Position:</strong> #${taskData.queue_position}
                    </div>
                ` : ''}
                
                ${taskData.estimated_time ? `
                    <div class="estimated-time">
                        <strong>Estimated Time:</strong> ${this.formatDuration(taskData.estimated_time)}
                    </div>
                ` : ''}
            </div>
            
            <div class="processing-effects">
                <div class="effect-particle"></div>
                <div class="effect-particle"></div>
                <div class="effect-particle"></div>
            </div>
        `;
    }

    onProcessingComplete(taskData) {
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
        
        this.goToStep(5);
        this.displayResults(taskData.results);
        this.showToast('Processing completed successfully! 🎉', 'success');
        
        // Trigger confetti effect
        this.triggerConfetti();
    }

    onProcessingFailed(taskData) {
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
        
        this.showToast(`Processing failed: ${taskData.error || 'Unknown error'}`, 'error');
    }

    displayResults(results) {
        const resultsContainer = document.getElementById('results-container');
        if (!resultsContainer) return;
        
        resultsContainer.innerHTML = `
            <div class="results-header">
                <h2>🎉 Your Viral Clips Are Ready!</h2>
                <p>Processed ${results.length} clip${results.length === 1 ? '' : 's'} with Netflix-level quality</p>
            </div>
            
            <div class="results-grid">
                ${results.map((result, index) => `
                    <div class="result-card">
                        <div class="result-preview">
                            ${result.thumbnail ? `
                                <img src="${result.thumbnail}" alt="Clip ${index + 1} thumbnail" loading="lazy">
                            ` : `
                                <div class="placeholder-thumbnail">
                                    <span class="icon">🎬</span>
                                    <span>Clip ${index + 1}</span>
                                </div>
                            `}
                            <div class="result-overlay">
                                <button class="btn-play" onclick="app.previewResult(${index})">
                                    <span class="icon">▶️</span> Preview
                                </button>
                            </div>
                        </div>
                        
                        <div class="result-info">
                            <h3>${result.title || `Viral Clip ${index + 1}`}</h3>
                            <div class="result-stats">
                                <span class="stat">
                                    <span class="stat-icon">⏱️</span>
                                    ${this.formatDuration(result.duration)}
                                </span>
                                <span class="stat">
                                    <span class="stat-icon">📊</span>
                                    ${result.viral_score}% viral
                                </span>
                                <span class="stat">
                                    <span class="stat-icon">📦</span>
                                    ${this.formatFileSize(result.file_size)}
                                </span>
                            </div>
                            
                            ${result.ai_enhancements ? `
                                <div class="enhancements">
                                    <h4>🤖 AI Enhancements</h4>
                                    <ul>
                                        ${result.ai_enhancements.slice(0, 3).map(enhancement => 
                                            `<li>${enhancement}</li>`
                                        ).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                        
                        <div class="result-actions">
                            <button class="btn-primary" onclick="app.downloadResult(${index})">
                                <span class="icon">⬇️</span> Download
                            </button>
                            <button class="btn-secondary" onclick="app.shareResult(${index})">
                                <span class="icon">📤</span> Share
                            </button>
                            <button class="btn-secondary" onclick="app.optimizeForPlatforms(${index})">
                                <span class="icon">🔧</span> Optimize
                            </button>
                        </div>
                    </div>
                `).join('')}
            </div>
            
            <div class="results-actions">
                <button class="btn-large btn-primary" onclick="app.downloadAll()">
                    <span class="icon">📦</span> Download All Clips
                </button>
                <button class="btn-large btn-secondary" onclick="app.startNew()">
                    <span class="icon">➕</span> Create More Clips
                </button>
                <button class="btn-large btn-secondary" onclick="app.shareCollection()">
                    <span class="icon">🌟</span> Share Collection
                </button>
            </div>
        `;
    }

    async downloadResult(clipIndex) {
        if (!this.processingTaskId) return;
        
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
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showToast(`Clip ${clipIndex + 1} downloaded successfully!`, 'success');
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showToast('Download failed. Please try again.', 'error');
        }
    }

    // Navigation methods
    goToStep(step) {
        if (step < 1 || step > this.totalSteps) return;
        
        // Hide all steps
        document.querySelectorAll('.step-section').forEach(section => {
            section.classList.remove('active');
        });
        
        // Show target step
        const targetStep = document.getElementById(`step-${step}`);
        if (targetStep) {
            targetStep.classList.add('active');
            targetStep.scrollIntoView({ behavior: 'smooth' });
        }
        
        this.currentStep = step;
        this.updateProgressIndicator();
        this.updateNavigationButtons();
    }

    nextStep() {
        if (this.currentStep < this.totalSteps) {
            this.goToStep(this.currentStep + 1);
        }
    }

    prevStep() {
        if (this.currentStep > 1) {
            this.goToStep(this.currentStep - 1);
        }
    }

    updateProgressIndicator() {
        const progressSteps = document.querySelectorAll('.progress-step');
        progressSteps.forEach((step, index) => {
            if (index + 1 < this.currentStep) {
                step.classList.add('completed');
                step.classList.remove('active');
            } else if (index + 1 === this.currentStep) {
                step.classList.add('active');
                step.classList.remove('completed');
            } else {
                step.classList.remove('active', 'completed');
            }
        });
    }

    updateNavigationButtons() {
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        
        if (prevBtn) prevBtn.disabled = this.currentStep === 1;
        if (nextBtn) nextBtn.disabled = this.currentStep === this.totalSteps;
    }

    // Utility methods
    formatDuration(seconds) {
        if (!seconds) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatNumber(num) {
        if (!num) return '0';
        if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
        if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
        return num.toLocaleString();
    }

    formatFileSize(bytes) {
        if (!bytes) return '0 B';
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
    }

    getStatusText(status) {
        const statusTexts = {
            'queued': 'Queued for processing',
            'processing': 'Processing in progress',
            'completed': 'Processing completed',
            'failed': 'Processing failed'
        };
        return statusTexts[status] || status;
    }

    formatStepName(step) {
        const stepNames = {
            'downloading': 'Downloading video',
            'ai_analysis': 'AI analysis in progress',
            'processing_clip_1': 'Processing clip 1',
            'processing_clip_2': 'Processing clip 2',
            'processing_clip_3': 'Processing clip 3',
            'finalizing': 'Finalizing clips',
            'completed': 'Process completed'
        };
        return stepNames[step] || step.replace(/_/g, ' ');
    }

    // UI Enhancement methods
    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container');
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icons = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        };
        
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
            <span class="toast-message">${message}</span>
            <button class="toast-close" onclick="this.parentElement.remove()">×</button>
        `;
        
        toastContainer.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }

    showLoading(message = 'Loading...') {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            const loadingText = loadingOverlay.querySelector('.loading-text');
            if (loadingText) {
                loadingText.textContent = message;
            }
            loadingOverlay.classList.remove('hidden');
        }
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.classList.add('hidden');
        }
    }

    triggerConfetti() {
        // Simple confetti effect
        for (let i = 0; i < 50; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti-piece';
            confetti.style.left = Math.random() * 100 + '%';
            confetti.style.backgroundColor = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'][Math.floor(Math.random() * 5)];
            confetti.style.animationDelay = Math.random() * 3 + 's';
            
            document.body.appendChild(confetti);
            
            setTimeout(() => confetti.remove(), 3000);
        }
    }

    // Advanced features
    async setupServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register('/sw.js');
                console.log('Service Worker registered:', registration);
            } catch (error) {
                console.error('Service Worker registration failed:', error);
            }
        }
    }

    async loadUserPreferences() {
        try {
            const preferences = localStorage.getItem('viralclip-preferences');
            if (preferences) {
                const prefs = JSON.parse(preferences);
                this.applyPreferences(prefs);
            }
        } catch (error) {
            console.error('Failed to load preferences:', error);
        }
    }

    setupWebSocketConnection() {
        // Placeholder for real-time updates via WebSocket
        // In production, this would connect to a WebSocket server
        console.log('WebSocket connection setup placeholder');
    }

    async initializePerformanceMonitoring() {
        // Performance monitoring setup
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
    }

    handleKeyboard(e) {
        // Keyboard shortcuts
        if (e.key === 'Escape') {
            const modal = document.querySelector('.modal-overlay:not(.hidden)');
            if (modal) modal.classList.add('hidden');
        }
        
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (this.currentStep < this.totalSteps) {
                this.nextStep();
            }
        }
    }

    async setupProgressiveEnhancement() {
        // Setup progressive enhancement features
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

// Export for external access
window.app = app;
