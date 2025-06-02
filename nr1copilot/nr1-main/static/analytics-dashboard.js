
/**
 * ViralClip Pro v8.0 - Netflix-Level Analytics Dashboard
 * Real-time analytics with enterprise performance monitoring
 */

class NetflixLevelAnalyticsDashboard {
    constructor() {
        this.websocket = null;
        this.charts = new Map();
        this.metrics = new Map();
        this.realTimeData = new Map();
        this.perfObserver = null;
        this.updateInterval = null;
        this.connectionRetries = 0;
        this.maxRetries = 5;
        
        // Performance monitoring
        this.performanceMetrics = {
            renderTime: [],
            updateLatency: [],
            memoryUsage: [],
            cpuUsage: []
        };
        
        this.initializeChartLibrary();
        this.setupPerformanceMonitoring();
        this.connectWebSocket();
        this.initializeDashboard();
        
        console.log('🚀 Netflix-Level Analytics Dashboard initialized');
    }

    initializeChartLibrary() {
        // Load Chart.js if not already loaded
        if (typeof Chart === 'undefined') {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js';
            script.onload = () => this.setupChartDefaults();
            document.head.appendChild(script);
        } else {
            this.setupChartDefaults();
        }
    }

    setupChartDefaults() {
        Chart.defaults.font.family = "'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
        Chart.defaults.color = '#6B7280';
        Chart.defaults.elements.point.radius = 0;
        Chart.defaults.elements.point.hoverRadius = 4;
        Chart.defaults.elements.line.borderWidth = 2;
        Chart.defaults.elements.line.tension = 0.4;
        Chart.defaults.plugins.legend.display = false;
        Chart.defaults.scales.x.grid.display = false;
        Chart.defaults.scales.y.grid.color = '#F3F4F6';
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;
    }

    setupPerformanceMonitoring() {
        // Performance Observer for monitoring dashboard performance
        if ('PerformanceObserver' in window) {
            this.perfObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (entry.entryType === 'measure') {
                        this.recordPerformanceMetric(entry.name, entry.duration);
                    }
                }
            });
            
            this.perfObserver.observe({ entryTypes: ['measure'] });
        }

        // Memory monitoring
        this.startMemoryMonitoring();
    }

    startMemoryMonitoring() {
        setInterval(() => {
            if ('memory' in performance) {
                const memory = performance.memory;
                this.performanceMetrics.memoryUsage.push({
                    timestamp: Date.now(),
                    used: memory.usedJSHeapSize,
                    total: memory.totalJSHeapSize,
                    limit: memory.jsHeapSizeLimit
                });
                
                // Keep only last 100 entries
                if (this.performanceMetrics.memoryUsage.length > 100) {
                    this.performanceMetrics.memoryUsage.shift();
                }
            }
        }, 5000);
    }

    recordPerformanceMetric(name, duration) {
        if (!this.performanceMetrics[name]) {
            this.performanceMetrics[name] = [];
        }
        
        this.performanceMetrics[name].push({
            timestamp: Date.now(),
            duration: duration
        });
        
        // Keep only last 50 measurements
        if (this.performanceMetrics[name].length > 50) {
            this.performanceMetrics[name].shift();
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/analytics/real-time`;
        
        try {
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected to analytics engine');
                this.connectionRetries = 0;
                this.showConnectionStatus('connected');
                this.startHeartbeat();
            };
            
            this.websocket.onmessage = (event) => {
                this.handleRealTimeUpdate(JSON.parse(event.data));
            };
            
            this.websocket.onclose = () => {
                console.log('🔄 WebSocket disconnected, attempting reconnection...');
                this.showConnectionStatus('disconnected');
                this.attemptReconnection();
            };
            
            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.showConnectionStatus('error');
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.showConnectionStatus('error');
        }
    }

    attemptReconnection() {
        if (this.connectionRetries < this.maxRetries) {
            this.connectionRetries++;
            const delay = Math.min(1000 * Math.pow(2, this.connectionRetries), 30000);
            
            setTimeout(() => {
                console.log(`🔄 Reconnection attempt ${this.connectionRetries}/${this.maxRetries}`);
                this.connectWebSocket();
            }, delay);
        } else {
            console.error('❌ Max reconnection attempts reached');
            this.showConnectionStatus('failed');
        }
    }

    startHeartbeat() {
        setInterval(() => {
            if (this.websocket?.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'heartbeat',
                    timestamp: Date.now()
                }));
            }
        }, 30000);
    }

    showConnectionStatus(status) {
        const statusElement = document.querySelector('.connection-status');
        if (statusElement) {
            statusElement.className = `connection-status ${status}`;
            statusElement.textContent = status === 'connected' ? '🟢 Connected' : 
                                      status === 'disconnected' ? '🟡 Reconnecting...' :
                                      status === 'failed' ? '🔴 Connection Failed' : '🟠 Error';
        }
    }

    handleRealTimeUpdate(data) {
        performance.mark('realtime-update-start');
        
        try {
            switch (data.type) {
                case 'metrics_update':
                    this.updateMetrics(data.payload);
                    break;
                case 'chart_data':
                    this.updateChartData(data.payload);
                    break;
                case 'viral_insight':
                    this.displayViralInsight(data.payload);
                    break;
                case 'performance_alert':
                    this.showPerformanceAlert(data.payload);
                    break;
                case 'system_health':
                    this.updateSystemHealth(data.payload);
                    break;
                default:
                    console.log('Unknown real-time update type:', data.type);
            }
        } catch (error) {
            console.error('Error handling real-time update:', error);
        }
        
        performance.mark('realtime-update-end');
        performance.measure('realtime-update', 'realtime-update-start', 'realtime-update-end');
    }

    async initializeDashboard() {
        performance.mark('dashboard-init-start');
        
        try {
            // Show loading state
            this.showLoadingState();
            
            // Load initial data
            const [metricsData, chartsData, systemHealth] = await Promise.all([
                this.fetchMetrics(),
                this.fetchChartData(),
                this.fetchSystemHealth()
            ]);
            
            // Initialize components
            await Promise.all([
                this.initializeMetricCards(metricsData),
                this.initializeCharts(chartsData),
                this.initializeSystemHealthDisplay(systemHealth),
                this.initializeViralInsights(),
                this.initializePerformanceMonitor()
            ]);
            
            // Hide loading state
            this.hideLoadingState();
            
            // Start auto-refresh
            this.startAutoRefresh();
            
            console.log('✅ Dashboard initialization completed');
            
        } catch (error) {
            console.error('❌ Dashboard initialization failed:', error);
            this.showErrorState(error);
        }
        
        performance.mark('dashboard-init-end');
        performance.measure('dashboard-init', 'dashboard-init-start', 'dashboard-init-end');
    }

    async fetchMetrics() {
        const response = await fetch('/api/v8/analytics/metrics', {
            headers: {
                'Accept': 'application/json',
                'X-Request-Source': 'dashboard'
            }
        });
        
        if (!response.ok) {
            throw new Error(`Failed to fetch metrics: ${response.status}`);
        }
        
        return response.json();
    }

    async fetchChartData() {
        const response = await fetch('/api/v8/analytics/charts', {
            headers: {
                'Accept': 'application/json',
                'X-Request-Source': 'dashboard'
            }
        });
        
        if (!response.ok) {
            throw new Error(`Failed to fetch chart data: ${response.status}`);
        }
        
        return response.json();
    }

    async fetchSystemHealth() {
        const response = await fetch('/api/v8/system/health', {
            headers: {
                'Accept': 'application/json',
                'X-Request-Source': 'dashboard'
            }
        });
        
        if (!response.ok) {
            throw new Error(`Failed to fetch system health: ${response.status}`);
        }
        
        return response.json();
    }

    initializeMetricCards(data) {
        const metrics = data.metrics || {};
        
        // Key Performance Indicators
        this.updateMetricCard('total-videos', {
            value: metrics.totalVideos || 0,
            change: metrics.videosChange || 0,
            format: 'number'
        });
        
        this.updateMetricCard('total-views', {
            value: metrics.totalViews || 0,
            change: metrics.viewsChange || 0,
            format: 'number'
        });
        
        this.updateMetricCard('viral-rate', {
            value: metrics.viralRate || 0,
            change: metrics.viralRateChange || 0,
            format: 'percentage'
        });
        
        this.updateMetricCard('engagement-rate', {
            value: metrics.engagementRate || 0,
            change: metrics.engagementChange || 0,
            format: 'percentage'
        });
        
        this.updateMetricCard('revenue', {
            value: metrics.revenue || 0,
            change: metrics.revenueChange || 0,
            format: 'currency'
        });
        
        this.updateMetricCard('active-users', {
            value: metrics.activeUsers || 0,
            change: metrics.activeUsersChange || 0,
            format: 'number'
        });
    }

    updateMetricCard(id, data) {
        const card = document.querySelector(`[data-metric="${id}"]`);
        if (!card) return;
        
        const valueElement = card.querySelector('.metric-value');
        const changeElement = card.querySelector('.metric-change');
        
        if (valueElement) {
            valueElement.textContent = this.formatValue(data.value, data.format);
        }
        
        if (changeElement) {
            const changeText = data.change >= 0 ? `+${data.change}%` : `${data.change}%`;
            const changeClass = data.change >= 0 ? 'positive' : 'negative';
            
            changeElement.textContent = changeText;
            changeElement.className = `metric-change ${changeClass}`;
        }
        
        // Add animation effect
        card.classList.add('updated');
        setTimeout(() => card.classList.remove('updated'), 500);
    }

    formatValue(value, format) {
        switch (format) {
            case 'number':
                return new Intl.NumberFormat().format(value);
            case 'percentage':
                return `${value.toFixed(1)}%`;
            case 'currency':
                return new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD'
                }).format(value);
            default:
                return value.toString();
        }
    }

    initializeCharts(data) {
        const chartConfigs = [
            {
                id: 'views-chart',
                type: 'line',
                data: data.viewsOverTime || [],
                options: this.getLineChartOptions('Video Views Over Time')
            },
            {
                id: 'viral-score-chart',
                type: 'line',
                data: data.viralScores || [],
                options: this.getLineChartOptions('Viral Score Trends')
            },
            {
                id: 'platform-distribution',
                type: 'doughnut',
                data: data.platformDistribution || [],
                options: this.getDoughnutChartOptions('Platform Distribution')
            },
            {
                id: 'engagement-metrics',
                type: 'bar',
                data: data.engagementMetrics || [],
                options: this.getBarChartOptions('Engagement Metrics')
            }
        ];
        
        chartConfigs.forEach(config => {
            this.createChart(config);
        });
    }

    createChart({ id, type, data, options }) {
        const canvas = document.getElementById(id);
        if (!canvas) {
            console.warn(`Chart canvas not found: ${id}`);
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.charts.has(id)) {
            this.charts.get(id).destroy();
        }
        
        const chart = new Chart(ctx, {
            type: type,
            data: this.formatChartData(data, type),
            options: options
        });
        
        this.charts.set(id, chart);
    }

    formatChartData(data, type) {
        switch (type) {
            case 'line':
                return {
                    labels: data.map(item => item.label),
                    datasets: [{
                        data: data.map(item => item.value),
                        borderColor: '#3B82F6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true
                    }]
                };
            case 'doughnut':
                return {
                    labels: data.map(item => item.label),
                    datasets: [{
                        data: data.map(item => item.value),
                        backgroundColor: [
                            '#3B82F6', '#10B981', '#F59E0B', 
                            '#EF4444', '#8B5CF6', '#06B6D4'
                        ]
                    }]
                };
            case 'bar':
                return {
                    labels: data.map(item => item.label),
                    datasets: [{
                        data: data.map(item => item.value),
                        backgroundColor: '#3B82F6'
                    }]
                };
            default:
                return { labels: [], datasets: [] };
        }
    }

    getLineChartOptions(title) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'hour'
                    }
                },
                y: {
                    beginAtZero: true
                }
            },
            animation: {
                duration: 750,
                easing: 'easeInOutQuart'
            }
        };
    }

    getDoughnutChartOptions(title) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title
                },
                legend: {
                    display: true,
                    position: 'bottom'
                }
            },
            animation: {
                animateRotate: true,
                duration: 1000
            }
        };
    }

    getBarChartOptions(title) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutBounce'
            }
        };
    }

    updateChartData(chartUpdates) {
        chartUpdates.forEach(update => {
            const chart = this.charts.get(update.chartId);
            if (chart) {
                chart.data = this.formatChartData(update.data, chart.config.type);
                chart.update('none'); // No animation for real-time updates
            }
        });
    }

    initializeSystemHealthDisplay(healthData) {
        const healthElement = document.querySelector('.system-health');
        if (!healthElement) return;
        
        const statusBadge = healthElement.querySelector('.health-status');
        const metricsContainer = healthElement.querySelector('.health-metrics');
        
        if (statusBadge) {
            statusBadge.textContent = healthData.status || 'Unknown';
            statusBadge.className = `health-status ${healthData.status || 'unknown'}`;
        }
        
        if (metricsContainer) {
            const metrics = [
                { label: 'CPU Usage', value: `${healthData.cpu_usage || 0}%` },
                { label: 'Memory Usage', value: `${healthData.memory_usage || 0}%` },
                { label: 'Response Time', value: `${healthData.response_time_ms || 0}ms` },
                { label: 'Active Connections', value: healthData.active_connections || 0 }
            ];
            
            metricsContainer.innerHTML = metrics.map(metric => `
                <div class="health-metric">
                    <span class="metric-label">${metric.label}</span>
                    <span class="metric-value">${metric.value}</span>
                </div>
            `).join('');
        }
    }

    updateSystemHealth(healthData) {
        this.initializeSystemHealthDisplay(healthData);
        
        // Update performance grade indicator
        const gradeElement = document.querySelector('.performance-grade');
        if (gradeElement) {
            gradeElement.textContent = healthData.performance_grade || '10/10';
        }
    }

    initializeViralInsights() {
        const container = document.querySelector('.viral-insights');
        if (!container) return;
        
        // Create insights display
        container.innerHTML = `
            <div class="insights-header">
                <h3>Real-Time Viral Insights</h3>
                <div class="insights-filter">
                    <select id="insights-timeframe">
                        <option value="1h">Last Hour</option>
                        <option value="24h" selected>Last 24 Hours</option>
                        <option value="7d">Last 7 Days</option>
                    </select>
                </div>
            </div>
            <div class="insights-container" id="insights-list">
                <div class="loading-insights">Loading viral insights...</div>
            </div>
        `;
        
        // Load initial insights
        this.loadViralInsights();
    }

    async loadViralInsights() {
        try {
            const response = await fetch('/api/v8/analytics/viral-insights');
            const insights = await response.json();
            this.displayViralInsights(insights.insights || []);
        } catch (error) {
            console.error('Failed to load viral insights:', error);
            this.showInsightsError();
        }
    }

    displayViralInsights(insights) {
        const container = document.getElementById('insights-list');
        if (!container) return;
        
        if (insights.length === 0) {
            container.innerHTML = '<div class="no-insights">No viral insights available</div>';
            return;
        }
        
        container.innerHTML = insights.map(insight => `
            <div class="insight-item ${insight.priority || 'normal'}">
                <div class="insight-icon">${this.getInsightIcon(insight.type)}</div>
                <div class="insight-content">
                    <div class="insight-title">${insight.title}</div>
                    <div class="insight-description">${insight.description}</div>
                    <div class="insight-meta">
                        <span class="confidence">Confidence: ${(insight.confidence * 100).toFixed(0)}%</span>
                        <span class="timestamp">${this.formatTimestamp(insight.timestamp)}</span>
                    </div>
                </div>
                <div class="insight-action">
                    <button class="apply-insight" data-insight-id="${insight.id}">Apply</button>
                </div>
            </div>
        `).join('');
        
        // Add event listeners for insight actions
        container.querySelectorAll('.apply-insight').forEach(button => {
            button.addEventListener('click', (e) => {
                this.applyInsight(e.target.dataset.insightId);
            });
        });
    }

    displayViralInsight(insight) {
        // Add new insight to the top of the list
        const container = document.getElementById('insights-list');
        if (!container) return;
        
        const insightElement = document.createElement('div');
        insightElement.className = `insight-item ${insight.priority || 'normal'} new-insight`;
        insightElement.innerHTML = `
            <div class="insight-icon">${this.getInsightIcon(insight.type)}</div>
            <div class="insight-content">
                <div class="insight-title">${insight.title}</div>
                <div class="insight-description">${insight.description}</div>
                <div class="insight-meta">
                    <span class="confidence">Confidence: ${(insight.confidence * 100).toFixed(0)}%</span>
                    <span class="timestamp">Just now</span>
                </div>
            </div>
            <div class="insight-action">
                <button class="apply-insight" data-insight-id="${insight.id}">Apply</button>
            </div>
        `;
        
        container.insertBefore(insightElement, container.firstChild);
        
        // Remove 'new' class after animation
        setTimeout(() => insightElement.classList.remove('new-insight'), 1000);
        
        // Limit to 10 insights
        const insights = container.querySelectorAll('.insight-item');
        if (insights.length > 10) {
            insights[insights.length - 1].remove();
        }
    }

    getInsightIcon(type) {
        const icons = {
            'viral_opportunity': '🚀',
            'engagement_boost': '📈',
            'trend_alert': '⚡',
            'optimization': '⚙️',
            'platform_specific': '📱',
            'content_suggestion': '💡'
        };
        return icons[type] || '💡';
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        return date.toLocaleDateString();
    }

    async applyInsight(insightId) {
        try {
            const response = await fetch(`/api/v8/analytics/insights/${insightId}/apply`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.showToast('Insight applied successfully!', 'success');
            } else {
                throw new Error('Failed to apply insight');
            }
        } catch (error) {
            console.error('Failed to apply insight:', error);
            this.showToast('Failed to apply insight', 'error');
        }
    }

    initializePerformanceMonitor() {
        const container = document.querySelector('.performance-monitor');
        if (!container) return;
        
        container.innerHTML = `
            <div class="performance-header">
                <h3>Performance Monitor</h3>
                <div class="performance-status excellent">
                    <span class="status-indicator"></span>
                    <span class="status-text">Excellent</span>
                </div>
            </div>
            <div class="performance-metrics">
                <div class="performance-metric">
                    <span class="metric-label">Response Time</span>
                    <span class="metric-value" id="response-time">--ms</span>
                </div>
                <div class="performance-metric">
                    <span class="metric-label">Memory Usage</span>
                    <span class="metric-value" id="memory-usage">--%</span>
                </div>
                <div class="performance-metric">
                    <span class="metric-label">CPU Usage</span>
                    <span class="metric-value" id="cpu-usage">--%</span>
                </div>
                <div class="performance-metric">
                    <span class="metric-label">Cache Hit Rate</span>
                    <span class="metric-value" id="cache-hit-rate">--%</span>
                </div>
            </div>
        `;
    }

    updateMetrics(metricsData) {
        // Update metric cards
        Object.entries(metricsData).forEach(([key, value]) => {
            const element = document.querySelector(`[data-metric="${key}"] .metric-value`);
            if (element) {
                element.textContent = this.formatMetricValue(key, value);
            }
        });
        
        // Update performance monitor
        this.updatePerformanceMetrics(metricsData);
    }

    updatePerformanceMetrics(data) {
        const updates = {
            'response-time': `${data.responseTime || 0}ms`,
            'memory-usage': `${data.memoryUsage || 0}%`,
            'cpu-usage': `${data.cpuUsage || 0}%`,
            'cache-hit-rate': `${data.cacheHitRate || 0}%`
        };
        
        Object.entries(updates).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }

    formatMetricValue(key, value) {
        switch (key) {
            case 'totalVideos':
            case 'totalViews':
            case 'activeUsers':
                return new Intl.NumberFormat().format(value);
            case 'viralRate':
            case 'engagementRate':
                return `${value.toFixed(1)}%`;
            case 'revenue':
                return new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD'
                }).format(value);
            default:
                return value.toString();
        }
    }

    showPerformanceAlert(alert) {
        this.showToast(
            `Performance Alert: ${alert.message}`,
            alert.level === 'critical' ? 'error' : 'warning'
        );
    }

    startAutoRefresh() {
        // Refresh dashboard data every 30 seconds
        this.updateInterval = setInterval(async () => {
            try {
                const [metricsData, systemHealth] = await Promise.all([
                    this.fetchMetrics(),
                    this.fetchSystemHealth()
                ]);
                
                this.updateMetrics(metricsData.metrics);
                this.updateSystemHealth(systemHealth);
                
            } catch (error) {
                console.error('Auto-refresh failed:', error);
            }
        }, 30000);
    }

    showLoadingState() {
        const dashboard = document.querySelector('.dashboard-container');
        if (dashboard) {
            dashboard.classList.add('loading');
        }
    }

    hideLoadingState() {
        const dashboard = document.querySelector('.dashboard-container');
        if (dashboard) {
            dashboard.classList.remove('loading');
        }
    }

    showErrorState(error) {
        const errorContainer = document.querySelector('.error-container');
        if (errorContainer) {
            errorContainer.innerHTML = `
                <div class="error-message">
                    <h3>Dashboard Error</h3>
                    <p>${error.message}</p>
                    <button onclick="location.reload()" class="retry-button">Retry</button>
                </div>
            `;
            errorContainer.style.display = 'block';
        }
    }

    showInsightsError() {
        const container = document.getElementById('insights-list');
        if (container) {
            container.innerHTML = `
                <div class="insights-error">
                    <p>Failed to load viral insights</p>
                    <button onclick="dashboard.loadViralInsights()" class="retry-button">Retry</button>
                </div>
            `;
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        // Animate in
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Remove after 5 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 5000);
    }

    destroy() {
        // Clean up resources
        if (this.websocket) {
            this.websocket.close();
        }
        
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        if (this.perfObserver) {
            this.perfObserver.disconnect();
        }
        
        // Destroy charts
        this.charts.forEach(chart => chart.destroy());
        this.charts.clear();
        
        console.log('🔄 Dashboard destroyed and resources cleaned up');
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new NetflixLevelAnalyticsDashboard();
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
});

// Export for use in other modules
window.NetflixLevelAnalyticsDashboard = NetflixLevelAnalyticsDashboard;
