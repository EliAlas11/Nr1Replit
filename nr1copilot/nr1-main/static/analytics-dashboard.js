class AdvancedAnalyticsDashboard {
    constructor() {
        this.currentTimeframe = '24h';
        this.websocket = null;
        this.dashboardData = null;
        this.updateInterval = null;
        this.charts = {};

        this.initializeEventListeners();
        this.connectWebSocket();
        this.loadDashboard();
    }

    initializeEventListeners() {
        // Timeframe selector
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.currentTimeframe = e.target.dataset.timeframe;
                this.loadDashboard();
            });
        });

        // Auto-refresh every 30 seconds
        this.updateInterval = setInterval(() => {
            this.loadDashboard();
        }, 30000);

        // Tab switching for detailed views
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('tab-btn')) {
                this.switchTab(e.target.dataset.tab);
            }
        });
    }

    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            this.websocket = new WebSocket(`${protocol}//${window.location.host}/ws/analytics/real-time`);

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleRealtimeUpdate(data);
            };

            this.websocket.onopen = () => {
                console.log('🔗 Real-time analytics WebSocket connected');
                this.sendMessage({
                    type: 'subscribe',
                    topics: ['engagement', 'viral_insights', 'roi_updates', 'trend_alerts']
                });
            };

            this.websocket.onerror = (error) => {
                console.error('Analytics WebSocket error:', error);
                this.showConnectionStatus('error');
            };

            this.websocket.onclose = () => {
                console.log('Analytics WebSocket disconnected');
                this.showConnectionStatus('disconnected');
                // Attempt reconnection after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }

    sendMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        }
    }

    async loadDashboard() {
        try {
            this.showLoading();

            const response = await fetch(`/api/v7/analytics/dashboard?timeframe=${this.currentTimeframe}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                this.dashboardData = data.dashboard;
                this.renderDashboard();
                this.showConnectionStatus('connected');
            } else {
                this.showError('Failed to load analytics data: ' + data.error);
            }
        } catch (error) {
            console.error('Dashboard loading error:', error);
            this.showError('Network error loading dashboard: ' + error.message);
        }
    }

    async loadVideoAnalysis(videoId) {
        try {
            const response = await fetch(`/api/v7/analytics/video/${videoId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (data.success) {
                this.showVideoAnalysis(data.analysis);
            } else {
                this.showError('Failed to load video analysis');
            }
        } catch (error) {
            console.error('Video analysis error:', error);
            this.showError('Failed to analyze video');
        }
    }

    async predictViralPotential(contentData) {
        try {
            const response = await fetch('/api/v7/analytics/viral-prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(contentData)
            });

            const data = await response.json();

            if (data.success) {
                this.showViralPrediction(data.prediction);
            } else {
                this.showError('Viral prediction failed');
            }
        } catch (error) {
            console.error('Viral prediction error:', error);
            this.showError('Failed to predict viral potential');
        }
    }

    async generateABComparison(videoAId, videoBId) {
        try {
            const response = await fetch('/api/v7/analytics/ab-comparison', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_a_id: videoAId,
                    video_b_id: videoBId,
                    metrics: ['views', 'engagement_rate', 'viral_score', 'roi']
                })
            });

            const data = await response.json();

            if (data.success) {
                this.showABComparison(data.comparison);
            } else {
                this.showError('A/B comparison failed');
            }
        } catch (error) {
            console.error('A/B comparison error:', error);
            this.showError('Failed to generate comparison');
        }
    }

    async trackROI(videoId, revenueData) {
        try {
            const response = await fetch(`/api/v7/analytics/roi/${videoId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(revenueData)
            });

            const data = await response.json();

            if (data.success) {
                this.updateROIDisplay(data.roi_metrics);
            } else {
                this.showError('ROI tracking failed');
            }
        } catch (error) {
            console.error('ROI tracking error:', error);
            this.showError('Failed to track ROI');
        }
    }

    async monitorUnderperformingContent() {
        try {
            const response = await fetch('/api/v7/analytics/underperforming-monitor', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (data.success) {
                this.showUnderperformanceAlerts(data.monitoring_report);
            } else {
                this.showError('Monitoring failed');
            }
        } catch (error) {
            console.error('Underperformance monitoring error:', error);
            this.showError('Failed to monitor content');
        }
    }

    async trackContentTrends() {
        try {
            const response = await fetch('/api/v7/analytics/content-trends', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (data.success) {
                this.showTrendAnalysis(data.trend_report);
            } else {
                this.showError('Trend tracking failed');
            }
        } catch (error) {
            console.error('Trend tracking error:', error);
            this.showError('Failed to track trends');
        }
    }

    renderDashboard() {
        const container = document.getElementById('dashboardContainer');
        const data = this.dashboardData;

        container.innerHTML = `
            ${this.renderNavigationTabs()}
            ${this.renderAlerts(data.alerts)}
            ${this.renderRealTimeMetrics(data)}
            ${this.renderDetailedAnalytics(data)}
        `;

        this.initializeInteractiveElements();
        this.initializeCharts();
    }

    renderNavigationTabs() {
        return `
            <div class="analytics-card" style="grid-column: 1 / -1;">
                <div class="nav-tabs">
                    <button class="tab-btn active" data-tab="overview">📊 Overview</button>
                    <button class="tab-btn" data-tab="viral">🚀 Viral Analysis</button>
                    <button class="tab-btn" data-tab="roi">💰 ROI Tracking</button>
                    <button class="tab-btn" data-tab="trends">📈 Trends</button>
                    <button class="tab-btn" data-tab="competitive">🏆 Competitive</button>
                </div>
            </div>
        `;
    }

    renderRealTimeMetrics(data) {
        return `
            <div class="analytics-card real-time-metrics">
                <div class="card-header">
                    <h3 class="card-title">⚡ Real-Time Metrics</h3>
                    <div class="status-indicator"></div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-value">${this.formatNumber(data.engagement_metrics?.total_views || 0)}</div>
                        <div class="metric-label">Live Views</div>
                        <div class="metric-change positive">+${((Math.random() * 10) + 2).toFixed(1)}%</div>
                    </div>

                    <div class="metric-item">
                        <div class="metric-value">${(data.engagement_metrics?.engagement_rate * 100 || 0).toFixed(1)}%</div>
                        <div class="metric-label">Engagement</div>
                        <div class="metric-change positive">+${((Math.random() * 5) + 1).toFixed(1)}%</div>
                    </div>

                    <div class="metric-item">
                        <div class="metric-value">${(data.viral_insights?.viral_probability * 100 || 0).toFixed(0)}</div>
                        <div class="metric-label">Viral Score</div>
                        <div class="metric-change neutral">${((Math.random() * 4) - 2).toFixed(1)}%</div>
                    </div>

                    <div class="metric-item">
                        <div class="metric-value">$${(data.roi_tracking?.total_revenue || 0).toFixed(0)}</div>
                        <div class="metric-label">Revenue</div>
                        <div class="metric-change positive">+${((Math.random() * 15) + 5).toFixed(1)}%</div>
                    </div>
                </div>

                <div class="live-activity-feed">
                    <div class="activity-item">🎯 New high engagement post detected</div>
                    <div class="activity-item">📈 Viral potential increasing for latest video</div>
                    <div class="activity-item">💰 Revenue milestone reached</div>
                </div>
            </div>
        `;
    }

    renderDetailedAnalytics(data) {
        return `
            <div class="tab-content active" data-tab-content="overview">
                ${this.renderEngagementOverview(data.engagement_metrics)}
                ${this.renderPerformanceCharts(data.performance_analytics)}
                ${this.renderQuickInsights(data)}
            </div>

            <div class="tab-content" data-tab-content="viral">
                ${this.renderViralAnalysis(data.viral_insights)}
                ${this.renderViralPredictionTools()}
            </div>

            <div class="tab-content" data-tab-content="roi">
                ${this.renderROIAnalysis(data.roi_tracking)}
                ${this.renderRevenueBreakdown(data.roi_tracking)}
            </div>

            <div class="tab-content" data-tab-content="trends">
                ${this.renderTrendDashboard(data.trend_analysis)}
                ${this.renderContentOpportunities(data.trend_analysis)}
            </div>

            <div class="tab-content" data-tab-content="competitive">
                ${this.renderCompetitiveAnalysis(data.competitive_insights)}
                ${this.renderMarketPosition(data.competitive_insights)}
            </div>
        `;
    }

    renderViralPredictionTools() {
        return `
            <div class="analytics-card">
                <div class="card-header">
                    <h3 class="card-title">🔮 Viral Prediction Tools</h3>
                </div>

                <div class="prediction-form">
                    <div class="form-group">
                        <label>Content Type</label>
                        <select id="contentType">
                            <option value="educational">Educational</option>
                            <option value="entertainment">Entertainment</option>
                            <option value="tutorial">Tutorial</option>
                            <option value="lifestyle">Lifestyle</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Target Platform</label>
                        <select id="targetPlatform">
                            <option value="tiktok">TikTok</option>
                            <option value="instagram">Instagram</option>
                            <option value="youtube">YouTube Shorts</option>
                        </select>
                    </div>

                    <button class="predict-btn" onclick="dashboard.predictContent()">
                        🚀 Predict Viral Potential
                    </button>
                </div>

                <div id="predictionResults" class="prediction-results"></div>
            </div>
        `;
    }

    renderABTestingTools() {
        return `
            <div class="analytics-card">
                <div class="card-header">
                    <h3 class="card-title">🧪 A/B Testing Suite</h3>
                </div>

                <div class="ab-test-form">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Video A ID</label>
                            <input type="text" id="videoAId" placeholder="Enter video ID">
                        </div>
                        <div class="form-group">
                            <label>Video B ID</label>
                            <input type="text" id="videoBId" placeholder="Enter video ID">
                        </div>
                    </div>

                    <div class="metrics-selector">
                        <label>Compare Metrics:</label>
                        <div class="checkbox-group">
                            <label><input type="checkbox" value="views" checked> Views</label>
                            <label><input type="checkbox" value="engagement" checked> Engagement</label>
                            <label><input type="checkbox" value="shares" checked> Shares</label>
                            <label><input type="checkbox" value="roi" checked> ROI</label>
                        </div>
                    </div>

                    <button class="compare-btn" onclick="dashboard.runABTest()">
                        📊 Run Comparison
                    </button>
                </div>

                <div id="abResults" class="ab-results"></div>
            </div>
        `;
    }

    initializeCharts() {
        this.initializeEngagementChart();
        this.initializeViralTrendChart();
        this.initializeROIChart();
        this.initializeRetentionCurve();
    }

    initializeEngagementChart() {
        const canvas = document.getElementById('engagementChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Sample data for demonstration
        const data = {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Engagement Rate',
                data: [8.2, 9.1, 7.8, 10.5, 12.3, 15.7, 14.2],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
                fill: true
            }]
        };

        this.charts.engagement = new Chart(ctx, {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
    }

    handleRealtimeUpdate(data) {
        switch (data.type) {
            case 'engagement_update':
                this.updateEngagementMetrics(data.metrics);
                break;
            case 'viral_insight':
                this.showViralAlert(data.insight);
                break;
            case 'roi_update':
                this.updateROIMetrics(data.roi);
                break;
            case 'trend_alert':
                this.showTrendAlert(data.trend);
                break;
            case 'performance_alert':
                this.showPerformanceAlert(data.alert);
                break;
        }
    }

    updateEngagementMetrics(metrics) {
        // Update real-time engagement display
        const engagementValue = document.querySelector('.metric-item:nth-child(2) .metric-value');
        if (engagementValue) {
            engagementValue.textContent = `${(metrics.engagement_rate * 100).toFixed(1)}%`;
        }
    }

    showViralAlert(insight) {
        const alertsContainer = document.querySelector('.live-activity-feed');
        if (alertsContainer) {
            const alertItem = document.createElement('div');
            alertItem.className = 'activity-item new';
            alertItem.textContent = `🚀 ${insight.message}`;
            alertsContainer.prepend(alertItem);

            // Remove oldest items if too many
            const items = alertsContainer.querySelectorAll('.activity-item');
            if (items.length > 5) {
                items[items.length - 1].remove();
            }
        }
    }

    showTrendAlert(trend) {
        this.showNotification(`📈 Trending: ${trend.topic} is gaining momentum!`, 'info');
    }

    showPerformanceAlert(alert) {
        this.showNotification(alert.message, alert.severity);
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    switchTab(tabName) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        // Remove active class from all tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });

        // Show selected tab content
        const targetContent = document.querySelector(`[data-tab-content="${tabName}"]`);
        if (targetContent) {
            targetContent.classList.add('active');
        }

        // Mark selected tab as active
        const targetTab = document.querySelector(`[data-tab="${tabName}"]`);
        if (targetTab) {
            targetTab.classList.add('active');
        }
    }

    async predictContent() {
        const contentType = document.getElementById('contentType')?.value;
        const targetPlatform = document.getElementById('targetPlatform')?.value;

        const contentData = {
            content_type: contentType,
            target_platform: targetPlatform,
            features: {
                duration: 30,
                has_audio: true,
                has_text_overlay: true,
                format: 'vertical'
            }
        };

        await this.predictViralPotential(contentData);
    }

    async runABTest() {
        const videoAId = document.getElementById('videoAId')?.value;
        const videoBId = document.getElementById('videoBId')?.value;

        if (!videoAId || !videoBId) {
            this.showError('Please enter both video IDs');
            return;
        }

        await this.generateABComparison(videoAId, videoBId);
    }

    showLoading() {
        const container = document.getElementById('dashboardContainer');
        container.innerHTML = `
            <div class="analytics-card">
                <div class="loading-spinner"></div>
                <p style="text-align: center; margin-top: 1rem;">Loading analytics data...</p>
            </div>
        `;
    }

    showConnectionStatus(status) {
        const indicators = document.querySelectorAll('.status-indicator');
        indicators.forEach(indicator => {
            indicator.className = `status-indicator ${status}`;
        });
    }

    showError(message) {
        const container = document.getElementById('dashboardContainer');
        container.innerHTML = `
            <div class="analytics-card error-card">
                <div class="card-header">
                    <h3 class="card-title">❌ Error</h3>
                </div>
                <p>${message}</p>
                <button onclick="dashboard.loadDashboard()" class="time-btn">
                    🔄 Retry
                </button>
            </div>
        `;
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    destroy() {
        // Cleanup when dashboard is destroyed
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        if (this.websocket) {
            this.websocket.close();
        }

        // Destroy charts
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) {
                chart.destroy();
            }
        });
    }
}

// Netflix-Level Analytics Dashboard v6.0
let socket = null;
let charts = {};
let metricsData = {};
let realTimeMetrics = new Map();
let performanceTracker = new Map();
let alertSystem = new AlertManager();

function createViralPredictionChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Viral Score',
                data: data.values,
                borderColor: '#FF6B6B',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#FF6B6B',
                pointBorderColor: '#FFFFFF',
                pointBorderWidth: 2,
                pointRadius: 6,
                pointHoverRadius: 8
            }, {
                label: 'Predicted Performance',
                data: data.predictions,
                borderColor: '#4ECDC4',
                backgroundColor: 'rgba(78, 205, 196, 0.1)',
                borderDash: [5, 5],
                fill: false,
                tension: 0.4
            }, {
                label: 'Industry Benchmark',
                data: Array(data.labels.length).fill(75),
                borderColor: '#95A5A6',
                backgroundColor: 'rgba(149, 165, 166, 0.05)',
                borderWidth: 1,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Netflix-Level Viral Prediction Analysis',
                    font: { size: 16, weight: 'bold' },
                    color: '#2C3E50'
                },
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#FFFFFF',
                    bodyColor: '#FFFFFF',
                    borderColor: '#4ECDC4',
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: true,
                    callbacks: {
                        afterBody: function(context) {
                            const dataIndex = context[0].dataIndex;
                            const score = data.values[dataIndex];
                            let performance = 'Low';
                            if (score >= 90) performance = 'Viral Potential';
                            else if (score >= 75) performance = 'High';
                            else if (score >= 60) performance = 'Medium';

                            return `Performance Level: ${performance}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        color: '#7F8C8D'
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        color: '#7F8C8D',
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// Global dashboard instance
let dashboard;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new AdvancedAnalyticsDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (dashboard) {
        dashboard.destroy();
    }
});