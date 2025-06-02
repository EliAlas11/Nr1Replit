
/*
ViralClip Pro v6.0 - Netflix-Level Upload System
Modular, scalable, and maintainable upload architecture
*/

// ================================
// Core Upload System Architecture
// ================================

class UploadSystemCore {
    constructor() {
        this.modules = new Map();
        this.eventBus = new EventBus();
        this.performance = new PerformanceMonitor();
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        // Initialize core modules in dependency order
        await this.initializeModules();
        await this.setupGlobalEventHandlers();
        await this.validateSystemReadiness();
        
        this.initialized = true;
        this.eventBus.emit('system:ready');
    }

    async initializeModules() {
        const moduleInitOrder = [
            'validator',
            'storage',
            'uploader',
            'ui',
            'analytics',
            'realtime'
        ];

        for (const moduleKey of moduleInitOrder) {
            const moduleClass = this.getModuleClass(moduleKey);
            this.modules.set(moduleKey, new moduleClass(this.eventBus, this.performance));
            await this.modules.get(moduleKey).initialize();
        }
    }

    getModuleClass(key) {
        const moduleMap = {
            validator: FileValidationModule,
            storage: StorageModule,
            uploader: ChunkedUploaderModule,
            ui: UIModule,
            analytics: AnalyticsModule,
            realtime: RealtimeModule
        };
        return moduleMap[key];
    }

    getModule(key) {
        return this.modules.get(key);
    }

    async setupGlobalEventHandlers() {
        // Global error handling
        window.addEventListener('error', this.handleGlobalError.bind(this));
        window.addEventListener('unhandledrejection', this.handleUnhandledRejection.bind(this));
        
        // Performance monitoring
        this.performance.startMonitoring();
    }

    handleGlobalError(event) {
        this.getModule('analytics')?.trackError(event.error);
        console.error('Global error:', event.error);
    }

    handleUnhandledRejection(event) {
        this.getModule('analytics')?.trackError(event.reason);
        console.error('Unhandled rejection:', event.reason);
    }

    async validateSystemReadiness() {
        const requiredFeatures = [
            'File API',
            'WebSocket',
            'IndexedDB',
            'Web Workers'
        ];

        for (const feature of requiredFeatures) {
            if (!this.isFeatureSupported(feature)) {
                throw new Error(`Required feature not supported: ${feature}`);
            }
        }
    }

    isFeatureSupported(feature) {
        const featureTests = {
            'File API': () => !!(window.File && window.FileReader && window.Blob),
            'WebSocket': () => !!window.WebSocket,
            'IndexedDB': () => !!window.indexedDB,
            'Web Workers': () => !!window.Worker
        };
        
        return featureTests[feature]?.() || false;
    }
}

// ================================
// Event Bus for Decoupled Communication
// ================================

class EventBus {
    constructor() {
        this.listeners = new Map();
        this.onceListeners = new Map();
    }

    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
        return () => this.off(event, callback);
    }

    once(event, callback) {
        if (!this.onceListeners.has(event)) {
            this.onceListeners.set(event, new Set());
        }
        this.onceListeners.get(event).add(callback);
    }

    off(event, callback) {
        this.listeners.get(event)?.delete(callback);
        this.onceListeners.get(event)?.delete(callback);
    }

    emit(event, data) {
        // Regular listeners
        this.listeners.get(event)?.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`Error in event listener for ${event}:`, error);
            }
        });

        // Once listeners
        const onceCallbacks = this.onceListeners.get(event);
        if (onceCallbacks) {
            onceCallbacks.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in once listener for ${event}:`, error);
                }
            });
            this.onceListeners.delete(event);
        }
    }
}

// ================================
// Performance Monitor
// ================================

class PerformanceMonitor {
    constructor() {
        this.metrics = new Map();
        this.observers = [];
        this.startTime = performance.now();
    }

    startMonitoring() {
        this.setupPerformanceObservers();
        this.startMetricsCollection();
    }

    setupPerformanceObservers() {
        if ('PerformanceObserver' in window) {
            const observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    this.recordMetric(entry.entryType, entry);
                }
            });
            
            observer.observe({ entryTypes: ['measure', 'navigation', 'resource'] });
            this.observers.push(observer);
        }
    }

    startMetricsCollection() {
        setInterval(() => {
            this.collectSystemMetrics();
        }, 5000); // Every 5 seconds
    }

    collectSystemMetrics() {
        const metrics = {
            memory: this.getMemoryUsage(),
            timing: this.getTimingMetrics(),
            network: this.getNetworkMetrics(),
            timestamp: Date.now()
        };
        
        this.recordMetric('system', metrics);
    }

    getMemoryUsage() {
        if ('memory' in performance) {
            return {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit
            };
        }
        return null;
    }

    getTimingMetrics() {
        const navigation = performance.getEntriesByType('navigation')[0];
        if (navigation) {
            return {
                domComplete: navigation.domComplete,
                loadComplete: navigation.loadEventEnd,
                firstPaint: this.getFirstPaint()
            };
        }
        return null;
    }

    getFirstPaint() {
        const paintEntries = performance.getEntriesByType('paint');
        const firstPaint = paintEntries.find(entry => entry.name === 'first-paint');
        return firstPaint?.startTime || null;
    }

    getNetworkMetrics() {
        const resources = performance.getEntriesByType('resource');
        return {
            resourceCount: resources.length,
            totalTransferSize: resources.reduce((sum, entry) => sum + (entry.transferSize || 0), 0)
        };
    }

    recordMetric(type, data) {
        if (!this.metrics.has(type)) {
            this.metrics.set(type, []);
        }
        
        const typeMetrics = this.metrics.get(type);
        typeMetrics.push({
            ...data,
            timestamp: performance.now()
        });

        // Keep only last 100 entries per type
        if (typeMetrics.length > 100) {
            typeMetrics.splice(0, typeMetrics.length - 100);
        }
    }

    getMetrics(type) {
        return this.metrics.get(type) || [];
    }

    mark(name) {
        performance.mark(name);
    }

    measure(name, startMark, endMark) {
        performance.measure(name, startMark, endMark);
    }
}

// ================================
// File Validation Module
// ================================

class FileValidationModule {
    constructor(eventBus, performance) {
        this.eventBus = eventBus;
        this.performance = performance;
        this.config = this.getValidationConfig();
    }

    async initialize() {
        this.eventBus.on('file:validate', this.validateFile.bind(this));
    }

    getValidationConfig() {
        return {
            maxFileSize: 500 * 1024 * 1024, // 500MB
            supportedFormats: new Set(['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.3gp', '.mp3', '.wav']),
            maxConcurrentFiles: 10,
            chunkSize: 5 * 1024 * 1024 // 5MB
        };
    }

    async validateFile(file) {
        this.performance.mark('validation:start');
        
        try {
            const result = await this.performValidation(file);
            this.performance.mark('validation:end');
            this.performance.measure('validation:duration', 'validation:start', 'validation:end');
            
            this.eventBus.emit('file:validated', { file, result });
            return result;
        } catch (error) {
            this.eventBus.emit('file:validation:error', { file, error });
            throw error;
        }
    }

    async performValidation(file) {
        const validations = [
            this.validateFileSize.bind(this),
            this.validateFileType.bind(this),
            this.validateFileName.bind(this),
            this.validateFileIntegrity.bind(this)
        ];

        for (const validation of validations) {
            const result = await validation(file);
            if (!result.valid) {
                return result;
            }
        }

        return {
            valid: true,
            metadata: await this.extractMetadata(file)
        };
    }

    validateFileSize(file) {
        if (file.size > this.config.maxFileSize) {
            return {
                valid: false,
                error: `File too large. Maximum size: ${this.formatFileSize(this.config.maxFileSize)}`
            };
        }
        
        if (file.size === 0) {
            return {
                valid: false,
                error: 'File is empty'
            };
        }

        return { valid: true };
    }

    validateFileType(file) {
        const extension = this.getFileExtension(file.name);
        
        if (!this.config.supportedFormats.has(extension)) {
            return {
                valid: false,
                error: `Unsupported format: ${extension}. Supported: ${Array.from(this.config.supportedFormats).join(', ')}`
            };
        }

        return { valid: true };
    }

    validateFileName(file) {
        const invalidChars = /[<>:"/\\|?*]/;
        
        if (invalidChars.test(file.name)) {
            return {
                valid: false,
                error: 'Filename contains invalid characters'
            };
        }

        if (file.name.length > 255) {
            return {
                valid: false,
                error: 'Filename is too long (max 255 characters)'
            };
        }

        return { valid: true };
    }

    async validateFileIntegrity(file) {
        // Basic integrity check by reading first few bytes
        try {
            const slice = file.slice(0, 1024);
            const arrayBuffer = await slice.arrayBuffer();
            
            if (arrayBuffer.byteLength === 0) {
                return {
                    valid: false,
                    error: 'File appears to be corrupted'
                };
            }

            return { valid: true };
        } catch (error) {
            return {
                valid: false,
                error: 'Unable to read file'
            };
        }
    }

    async extractMetadata(file) {
        const metadata = {
            name: file.name,
            size: file.size,
            type: file.type,
            lastModified: file.lastModified,
            extension: this.getFileExtension(file.name),
            estimatedDuration: this.estimateVideoDuration(file.size),
            chunksCount: Math.ceil(file.size / this.config.chunkSize)
        };

        // Extract additional metadata for video files
        if (file.type.startsWith('video/')) {
            metadata.thumbnail = await this.generateThumbnail(file);
            metadata.videoMetadata = await this.extractVideoMetadata(file);
        }

        return metadata;
    }

    async generateThumbnail(file) {
        return new Promise((resolve) => {
            const video = document.createElement('video');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            video.addEventListener('loadedmetadata', () => {
                canvas.width = 160;
                canvas.height = 90;
                video.currentTime = Math.min(2, video.duration / 4);
            });

            video.addEventListener('seeked', () => {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const thumbnailUrl = canvas.toDataURL('image/jpeg', 0.8);
                resolve(thumbnailUrl);
                URL.revokeObjectURL(video.src);
            });

            video.addEventListener('error', () => resolve(null));
            
            video.src = URL.createObjectURL(file);
            video.load();
        });
    }

    async extractVideoMetadata(file) {
        return new Promise((resolve) => {
            const video = document.createElement('video');
            
            video.addEventListener('loadedmetadata', () => {
                resolve({
                    duration: video.duration,
                    videoWidth: video.videoWidth,
                    videoHeight: video.videoHeight,
                    aspectRatio: video.videoWidth / video.videoHeight
                });
                URL.revokeObjectURL(video.src);
            });

            video.addEventListener('error', () => resolve({}));
            
            video.src = URL.createObjectURL(file);
        });
    }

    getFileExtension(filename) {
        return '.' + filename.split('.').pop().toLowerCase();
    }

    estimateVideoDuration(fileSize) {
        // Rough estimation: 1MB ≈ 1 second for typical video
        return Math.max(1, fileSize / (1024 * 1024));
    }

    formatFileSize(bytes) {
        const units = ['B', 'KB', 'MB', 'GB'];
        let size = bytes;
        let unitIndex = 0;

        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }

        return `${size.toFixed(2)} ${units[unitIndex]}`;
    }
}

// ================================
// Storage Module
// ================================

class StorageModule {
    constructor(eventBus, performance) {
        this.eventBus = eventBus;
        this.performance = performance;
        this.db = null;
        this.cache = new Map();
    }

    async initialize() {
        await this.initializeIndexedDB();
        this.setupEventHandlers();
    }

    async initializeIndexedDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('ViralClipUploadDB', 1);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                if (!db.objectStoreNames.contains('uploads')) {
                    const uploadStore = db.createObjectStore('uploads', { keyPath: 'id' });
                    uploadStore.createIndex('status', 'status', { unique: false });
                    uploadStore.createIndex('timestamp', 'timestamp', { unique: false });
                }
                
                if (!db.objectStoreNames.contains('chunks')) {
                    const chunkStore = db.createObjectStore('chunks', { keyPath: 'id' });
                    chunkStore.createIndex('uploadId', 'uploadId', { unique: false });
                }
            };
        });
    }

    setupEventHandlers() {
        this.eventBus.on('upload:save', this.saveUploadData.bind(this));
        this.eventBus.on('upload:load', this.loadUploadData.bind(this));
        this.eventBus.on('chunk:save', this.saveChunkData.bind(this));
        this.eventBus.on('chunk:load', this.loadChunkData.bind(this));
    }

    async saveUploadData(uploadData) {
        try {
            const transaction = this.db.transaction(['uploads'], 'readwrite');
            const store = transaction.objectStore('uploads');
            await store.put(uploadData);
            
            // Also cache in memory for quick access
            this.cache.set(`upload:${uploadData.id}`, uploadData);
            
            this.eventBus.emit('upload:saved', uploadData);
        } catch (error) {
            this.eventBus.emit('storage:error', { operation: 'saveUpload', error });
        }
    }

    async loadUploadData(uploadId) {
        try {
            // Check cache first
            const cacheKey = `upload:${uploadId}`;
            if (this.cache.has(cacheKey)) {
                return this.cache.get(cacheKey);
            }
            
            const transaction = this.db.transaction(['uploads'], 'readonly');
            const store = transaction.objectStore('uploads');
            const result = await store.get(uploadId);
            
            if (result) {
                this.cache.set(cacheKey, result);
            }
            
            return result;
        } catch (error) {
            this.eventBus.emit('storage:error', { operation: 'loadUpload', error });
            return null;
        }
    }

    async saveChunkData(chunkData) {
        try {
            const transaction = this.db.transaction(['chunks'], 'readwrite');
            const store = transaction.objectStore('chunks');
            await store.put(chunkData);
        } catch (error) {
            this.eventBus.emit('storage:error', { operation: 'saveChunk', error });
        }
    }

    async loadChunkData(uploadId) {
        try {
            const transaction = this.db.transaction(['chunks'], 'readonly');
            const store = transaction.objectStore('chunks');
            const index = store.index('uploadId');
            const result = await index.getAll(uploadId);
            return result;
        } catch (error) {
            this.eventBus.emit('storage:error', { operation: 'loadChunks', error });
            return [];
        }
    }

    async clearUploadData(uploadId) {
        try {
            // Clear from cache
            this.cache.delete(`upload:${uploadId}`);
            
            // Clear from IndexedDB
            const transaction = this.db.transaction(['uploads', 'chunks'], 'readwrite');
            
            await transaction.objectStore('uploads').delete(uploadId);
            
            const chunkStore = transaction.objectStore('chunks');
            const chunkIndex = chunkStore.index('uploadId');
            const chunks = await chunkIndex.getAllKeys(uploadId);
            
            for (const chunkKey of chunks) {
                await chunkStore.delete(chunkKey);
            }
            
            this.eventBus.emit('upload:cleared', uploadId);
        } catch (error) {
            this.eventBus.emit('storage:error', { operation: 'clearUpload', error });
        }
    }
}

// ================================
// Chunked Uploader Module
// ================================

class ChunkedUploaderModule {
    constructor(eventBus, performance) {
        this.eventBus = eventBus;
        this.performance = performance;
        this.activeUploads = new Map();
        this.uploadQueue = [];
        this.maxConcurrentUploads = 3;
        this.chunkSize = 5 * 1024 * 1024; // 5MB
    }

    async initialize() {
        this.setupEventHandlers();
        this.startQueueProcessor();
    }

    setupEventHandlers() {
        this.eventBus.on('file:validated', this.handleValidatedFile.bind(this));
        this.eventBus.on('upload:pause', this.pauseUpload.bind(this));
        this.eventBus.on('upload:resume', this.resumeUpload.bind(this));
        this.eventBus.on('upload:cancel', this.cancelUpload.bind(this));
    }

    async handleValidatedFile({ file, result }) {
        if (!result.valid) return;

        const uploadSession = this.createUploadSession(file, result.metadata);
        this.uploadQueue.push(uploadSession);
        
        this.eventBus.emit('upload:queued', uploadSession);
    }

    createUploadSession(file, metadata) {
        const sessionId = this.generateSessionId();
        
        return {
            id: sessionId,
            file,
            metadata,
            status: 'queued',
            progress: 0,
            uploadedChunks: new Set(),
            totalChunks: metadata.chunksCount,
            speed: 0,
            startTime: null,
            pausedAt: null,
            retryCount: 0,
            maxRetries: 3,
            errors: []
        };
    }

    generateSessionId() {
        return `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    startQueueProcessor() {
        setInterval(() => {
            this.processQueue();
        }, 1000);
    }

    async processQueue() {
        if (this.activeUploads.size >= this.maxConcurrentUploads) return;
        
        const nextUpload = this.uploadQueue.find(upload => upload.status === 'queued');
        if (!nextUpload) return;

        nextUpload.status = 'uploading';
        this.activeUploads.set(nextUpload.id, nextUpload);
        
        try {
            await this.startUpload(nextUpload);
        } catch (error) {
            this.handleUploadError(nextUpload, error);
        }
    }

    async startUpload(uploadSession) {
        this.performance.mark(`upload:${uploadSession.id}:start`);
        uploadSession.startTime = Date.now();
        
        // Initialize upload session on server
        const sessionResponse = await this.initializeServerSession(uploadSession);
        if (!sessionResponse.success) {
            throw new Error(sessionResponse.error);
        }

        // Start chunked upload
        await this.uploadChunks(uploadSession);
    }

    async initializeServerSession(uploadSession) {
        const response = await fetch('/api/v6/upload/init', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: uploadSession.file.name,
                file_size: uploadSession.file.size,
                total_chunks: uploadSession.totalChunks,
                upload_id: uploadSession.id
            })
        });

        return await response.json();
    }

    async uploadChunks(uploadSession) {
        const { file, totalChunks, uploadedChunks } = uploadSession;
        
        for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
            if (uploadedChunks.has(chunkIndex)) continue;
            if (uploadSession.status === 'paused') break;
            if (uploadSession.status === 'cancelled') break;

            await this.uploadChunk(uploadSession, chunkIndex);
        }

        if (uploadedChunks.size === totalChunks) {
            await this.completeUpload(uploadSession);
        }
    }

    async uploadChunk(uploadSession, chunkIndex) {
        const { file } = uploadSession;
        const start = chunkIndex * this.chunkSize;
        const end = Math.min(start + this.chunkSize, file.size);
        const chunk = file.slice(start, end);

        const maxRetries = 3;
        let attempt = 0;

        while (attempt < maxRetries) {
            try {
                const startTime = Date.now();
                await this.sendChunk(uploadSession, chunk, chunkIndex);
                
                const uploadTime = (Date.now() - startTime) / 1000;
                const speed = chunk.size / uploadTime;
                
                uploadSession.uploadedChunks.add(chunkIndex);
                uploadSession.speed = speed;
                uploadSession.progress = (uploadSession.uploadedChunks.size / uploadSession.totalChunks) * 100;
                
                this.eventBus.emit('upload:progress', {
                    uploadId: uploadSession.id,
                    progress: uploadSession.progress,
                    speed: speed,
                    chunkIndex,
                    totalChunks: uploadSession.totalChunks
                });

                break; // Success, exit retry loop
                
            } catch (error) {
                attempt++;
                if (attempt >= maxRetries) {
                    throw error;
                }
                
                // Exponential backoff
                await this.delay(Math.pow(2, attempt) * 1000);
            }
        }
    }

    async sendChunk(uploadSession, chunk, chunkIndex) {
        const formData = new FormData();
        formData.append('file', chunk, `chunk_${chunkIndex}`);
        formData.append('upload_id', uploadSession.id);
        formData.append('chunk_index', chunkIndex);
        formData.append('total_chunks', uploadSession.totalChunks);

        const response = await fetch('/api/v6/upload/chunk', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error);
        }

        return result;
    }

    async completeUpload(uploadSession) {
        uploadSession.status = 'completed';
        
        this.performance.mark(`upload:${uploadSession.id}:end`);
        this.performance.measure(
            `upload:${uploadSession.id}:duration`,
            `upload:${uploadSession.id}:start`,
            `upload:${uploadSession.id}:end`
        );

        this.activeUploads.delete(uploadSession.id);
        
        this.eventBus.emit('upload:completed', uploadSession);
    }

    handleUploadError(uploadSession, error) {
        uploadSession.errors.push({
            error: error.message,
            timestamp: Date.now(),
            chunkIndex: uploadSession.uploadedChunks.size
        });

        if (uploadSession.retryCount < uploadSession.maxRetries) {
            uploadSession.retryCount++;
            uploadSession.status = 'queued'; // Re-queue for retry
            this.eventBus.emit('upload:retry', uploadSession);
        } else {
            uploadSession.status = 'failed';
            this.activeUploads.delete(uploadSession.id);
            this.eventBus.emit('upload:failed', uploadSession);
        }
    }

    pauseUpload(uploadId) {
        const uploadSession = this.activeUploads.get(uploadId);
        if (uploadSession) {
            uploadSession.status = 'paused';
            uploadSession.pausedAt = Date.now();
            this.eventBus.emit('upload:paused', uploadSession);
        }
    }

    resumeUpload(uploadId) {
        const uploadSession = this.activeUploads.get(uploadId);
        if (uploadSession && uploadSession.status === 'paused') {
            uploadSession.status = 'uploading';
            uploadSession.pausedAt = null;
            this.eventBus.emit('upload:resumed', uploadSession);
        }
    }

    cancelUpload(uploadId) {
        const uploadSession = this.activeUploads.get(uploadId);
        if (uploadSession) {
            uploadSession.status = 'cancelled';
            this.activeUploads.delete(uploadId);
            this.eventBus.emit('upload:cancelled', uploadSession);
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// ================================
// UI Module
// ================================

class UIModule {
    constructor(eventBus, performance) {
        this.eventBus = eventBus;
        this.performance = performance;
        this.elements = new Map();
        this.templates = new Map();
    }

    async initialize() {
        await this.createUploadInterface();
        this.setupEventHandlers();
        this.setupDragAndDrop();
        this.setupMobileOptimizations();
    }

    async createUploadInterface() {
        const uploadSection = this.createElement('div', {
            id: 'enhanced-upload-section',
            className: 'netflix-upload-section'
        });

        uploadSection.innerHTML = await this.renderMainTemplate();
        
        this.insertIntoDOM(uploadSection);
        this.cacheElements();
    }

    async renderMainTemplate() {
        return `
            <div class="upload-header">
                <h1 class="upload-title">🎬 Netflix-Grade Upload Experience</h1>
                <div class="upload-stats">
                    <span class="stat-item">
                        <span class="stat-icon">⚡</span>
                        <span class="stat-label">Speed: <span id="upload-speed">0 MB/s</span></span>
                    </span>
                    <span class="stat-item">
                        <span class="stat-icon">📊</span>
                        <span class="stat-label">Queue: <span id="queue-count">0</span></span>
                    </span>
                    <span class="stat-item">
                        <span class="stat-icon">✅</span>
                        <span class="stat-label">Success: <span id="success-rate">100%</span></span>
                    </span>
                </div>
            </div>

            <div class="netflix-drop-zone" id="mainDropZone">
                <div class="drop-zone-content">
                    <div class="drop-zone-icon" id="dropZoneIcon">
                        <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <polyline points="14,2 14,8 20,8"/>
                            <line x1="12" y1="18" x2="12" y2="12"/>
                            <line x1="9" y1="15" x2="12" y2="12"/>
                            <line x1="15" y1="15" x2="12" y2="12"/>
                        </svg>
                    </div>
                    <div class="drop-zone-text">
                        <h2 id="dropZoneTitle">Drop your video files here</h2>
                        <p id="dropZoneSubtitle">or click to browse files</p>
                        <div class="supported-formats">
                            <span class="format-label">Supported formats:</span>
                            <span class="format-list">MP4, MOV, AVI, MKV, WEBM, M4V, 3GP, MP3, WAV</span>
                        </div>
                        <div class="file-limits">
                            <span class="limit-item">📐 Max size: 500MB</span>
                            <span class="limit-item">🎯 Best quality: 1080p+</span>
                            <span class="limit-item">⚡ Lightning fast processing</span>
                        </div>
                    </div>
                    <div class="drop-zone-actions">
                        <button class="netflix-btn primary-btn" id="browseFilesBtn">
                            <span class="btn-icon">📁</span>
                            <span class="btn-text">Browse Files</span>
                        </button>
                        <button class="netflix-btn secondary-btn" id="pasteUrlBtn">
                            <span class="btn-icon">🔗</span>
                            <span class="btn-text">Paste URL</span>
                        </button>
                    </div>
                </div>
            </div>

            <div class="file-preview-section" id="filePreviewSection" style="display: none;">
                <h3 class="preview-title">📋 Upload Queue</h3>
                <div class="file-previews" id="filePreviews"></div>
            </div>

            <div class="queue-manager" id="queueManager" style="display: none;">
                <div class="queue-header">
                    <h3>🎬 Upload Manager</h3>
                    <div class="queue-controls">
                        <button class="queue-btn" id="pauseAllBtn">⏸️ Pause All</button>
                        <button class="queue-btn" id="resumeAllBtn">▶️ Resume All</button>
                        <button class="queue-btn danger" id="clearQueueBtn">🗑️ Clear Queue</button>
                    </div>
                </div>
                <div class="upload-items" id="uploadItems"></div>
            </div>

            <input type="file" id="fileInput" multiple 
                   accept=".mp4,.mov,.avi,.mkv,.webm,.m4v,.3gp,.mp3,.wav" 
                   style="display: none;">
        `;
    }

    setupEventHandlers() {
        // File browser
        this.getElement('browseFilesBtn').addEventListener('click', () => {
            this.getElement('fileInput').click();
        });

        this.getElement('fileInput').addEventListener('change', (e) => {
            this.handleFileSelection(Array.from(e.target.files));
        });

        // Queue controls
        this.getElement('pauseAllBtn')?.addEventListener('click', () => {
            this.eventBus.emit('queue:pauseAll');
        });

        this.getElement('resumeAllBtn')?.addEventListener('click', () => {
            this.eventBus.emit('queue:resumeAll');
        });

        this.getElement('clearQueueBtn')?.addEventListener('click', () => {
            this.eventBus.emit('queue:clear');
        });

        // Event bus listeners
        this.eventBus.on('upload:queued', this.renderUploadItem.bind(this));
        this.eventBus.on('upload:progress', this.updateUploadProgress.bind(this));
        this.eventBus.on('upload:completed', this.handleUploadCompleted.bind(this));
        this.eventBus.on('upload:failed', this.handleUploadFailed.bind(this));
    }

    setupDragAndDrop() {
        const dropZone = this.getElement('mainDropZone');
        let dragCounter = 0;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dragCounter++;
                this.highlightDropZone(true);
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dragCounter--;
                if (dragCounter === 0) {
                    this.highlightDropZone(false);
                }
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const files = Array.from(e.dataTransfer.files);
            this.handleFileSelection(files);
        }, false);

        dropZone.addEventListener('click', () => {
            this.getElement('fileInput').click();
        });
    }

    setupMobileOptimizations() {
        if ('ontouchstart' in window) {
            document.body.classList.add('touch-device');
            
            const dropZone = this.getElement('mainDropZone');
            let touchStartTime = 0;
            
            // Enhanced touch feedback with haptic response
            dropZone.addEventListener('touchstart', (e) => {
                e.preventDefault();
                touchStartTime = Date.now();
                this.highlightDropZone(true);
                
                // Haptic feedback if available
                if (navigator.vibrate) {
                    navigator.vibrate(50);
                }
                
                // Visual feedback for touch
                dropZone.style.transform = 'scale(0.98)';
            });

            dropZone.addEventListener('touchmove', (e) => {
                e.preventDefault();
                // Prevent scrolling while touching upload zone
            });

            dropZone.addEventListener('touchend', (e) => {
                e.preventDefault();
                const touchDuration = Date.now() - touchStartTime;
                
                // Reset visual feedback
                dropZone.style.transform = 'scale(1)';
                this.highlightDropZone(false);
                
                // Only trigger file selection if it was a quick tap
                if (touchDuration < 500) {
                    setTimeout(() => {
                        this.getElement('fileInput').click();
                    }, 100);
                }
            });

            // Improved file input for mobile
            const fileInput = this.getElement('fileInput');
            fileInput.addEventListener('change', () => {
                // Provide immediate feedback on mobile
                if (fileInput.files.length > 0) {
                    this.showMobileUploadFeedback(fileInput.files.length);
                }
            });
        }

        this.setupResponsiveLayout();
        this.setupMobileProgressOptimizations();
    }

    showMobileUploadFeedback(fileCount) {
        const feedback = document.createElement('div');
        feedback.className = 'mobile-upload-feedback';
        feedback.innerHTML = `
            <div class="feedback-content">
                <span class="feedback-icon">📱</span>
                <span class="feedback-text">${fileCount} file${fileCount > 1 ? 's' : ''} selected</span>
            </div>
        `;
        
        document.body.appendChild(feedback);
        
        setTimeout(() => {
            feedback.remove();
        }, 2000);
    }

    setupMobileProgressOptimizations() {
        // Optimize progress updates for mobile performance
        let lastProgressUpdate = 0;
        const originalUpdateProgress = this.updateUploadProgress.bind(this);
        
        this.updateUploadProgress = (progressData) => {
            const now = Date.now();
            // Throttle updates on mobile to 100ms intervals
            if (document.body.classList.contains('touch-device')) {
                if (now - lastProgressUpdate < 100) {
                    return;
                }
                lastProgressUpdate = now;
            }
            
            originalUpdateProgress(progressData);
        };
    }

    setupResponsiveLayout() {
        const mediaQuery = window.matchMedia('(max-width: 768px)');
        
        const handleMobileLayout = (e) => {
            const dropZone = this.getElement('mainDropZone');
            
            if (e.matches) {
                dropZone.classList.add('mobile-layout');
            } else {
                dropZone.classList.remove('mobile-layout');
            }
        };

        mediaQuery.addListener(handleMobileLayout);
        handleMobileLayout(mediaQuery);
    }

    handleFileSelection(files) {
        if (files.length === 0) return;

        this.showSections(['filePreviewSection', 'queueManager']);

        files.forEach(file => {
            this.eventBus.emit('file:validate', file);
        });
    }

    renderUploadItem(uploadSession) {
        const previewsContainer = this.getElement('filePreviews');
        const uploadItemsContainer = this.getElement('uploadItems');

        // File preview card
        const previewCard = this.createElement('div', {
            className: 'file-preview-card'
        });

        previewCard.innerHTML = this.renderPreviewCardTemplate(uploadSession);
        previewsContainer.appendChild(previewCard);

        // Upload manager item
        const uploadItem = this.createElement('div', {
            className: 'upload-item',
            id: `upload-${uploadSession.id}`
        });

        uploadItem.innerHTML = this.renderUploadItemTemplate(uploadSession);
        uploadItemsContainer.appendChild(uploadItem);

        this.setupUploadItemControls(uploadSession);
    }

    renderPreviewCardTemplate(uploadSession) {
        const { metadata } = uploadSession;
        
        return `
            <div class="preview-thumbnail">
                ${metadata.thumbnail ? 
                    `<img src="${metadata.thumbnail}" alt="Video thumbnail">` :
                    `<div class="thumbnail-placeholder">
                        <span class="file-icon">${this.getFileIcon(uploadSession.file.type)}</span>
                    </div>`
                }
                <div class="file-format">.${metadata.extension.toUpperCase().slice(1)}</div>
            </div>
            <div class="preview-info">
                <div class="file-name" title="${uploadSession.file.name}">${uploadSession.file.name}</div>
                <div class="file-details">
                    <span class="file-size">${this.formatFileSize(uploadSession.file.size)}</span>
                    <span class="file-duration">~${metadata.estimatedDuration}s</span>
                    <span class="file-chunks">${metadata.chunksCount} chunks</span>
                </div>
            </div>
        `;
    }

    renderUploadItemTemplate(uploadSession) {
        const { metadata } = uploadSession;
        
        return `
            <div class="upload-item-header">
                <div class="item-thumbnail">
                    ${metadata.thumbnail ? 
                        `<img src="${metadata.thumbnail}" alt="Thumbnail">` :
                        `<div class="placeholder-thumb">${this.getFileIcon(uploadSession.file.type)}</div>`
                    }
                </div>
                <div class="item-info">
                    <div class="item-name">${uploadSession.file.name}</div>
                    <div class="item-meta">
                        <span class="item-size">${this.formatFileSize(uploadSession.file.size)}</span>
                        <span class="item-status" id="status-${uploadSession.id}">Queued</span>
                    </div>
                </div>
                <div class="item-controls">
                    <button class="control-btn pause-btn" id="pause-${uploadSession.id}" title="Pause">⏸️</button>
                    <button class="control-btn retry-btn" id="retry-${uploadSession.id}" title="Retry" style="display: none;">🔄</button>
                    <button class="control-btn remove-btn" id="remove-${uploadSession.id}" title="Remove">🗑️</button>
                </div>
            </div>
            
            <div class="upload-progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-${uploadSession.id}" style="width: 0%"></div>
                    <div class="progress-text" id="progress-text-${uploadSession.id}">0%</div>
                </div>
                <div class="upload-stats">
                    <span class="upload-speed" id="speed-${uploadSession.id}">0 MB/s</span>
                    <span class="upload-eta" id="eta-${uploadSession.id}">--:--</span>
                    <span class="upload-chunks" id="chunks-${uploadSession.id}">0/${metadata.chunksCount}</span>
                </div>
            </div>

            <div class="chunk-progress" id="chunk-progress-${uploadSession.id}">
                ${Array(Math.min(metadata.chunksCount, 20)).fill(0).map((_, i) => 
                    `<div class="chunk-indicator" id="chunk-${uploadSession.id}-${i}"></div>`
                ).join('')}
                ${metadata.chunksCount > 20 ? '<span class="chunk-more">...</span>' : ''}
            </div>
        `;
    }

    setupUploadItemControls(uploadSession) {
        const pauseBtn = document.getElementById(`pause-${uploadSession.id}`);
        const retryBtn = document.getElementById(`retry-${uploadSession.id}`);
        const removeBtn = document.getElementById(`remove-${uploadSession.id}`);

        pauseBtn?.addEventListener('click', () => {
            this.eventBus.emit('upload:pause', uploadSession.id);
        });

        retryBtn?.addEventListener('click', () => {
            this.eventBus.emit('upload:retry', uploadSession.id);
        });

        removeBtn?.addEventListener('click', () => {
            this.eventBus.emit('upload:cancel', uploadSession.id);
            this.removeUploadItem(uploadSession.id);
        });
    }

    updateUploadProgress({ uploadId, progress, speed, chunkIndex, totalChunks, bytesUploaded, totalBytes }) {
        const progressFill = document.getElementById(`progress-${uploadId}`);
        const progressText = document.getElementById(`progress-text-${uploadId}`);
        const speedElement = document.getElementById(`speed-${uploadId}`);
        const chunksElement = document.getElementById(`chunks-${uploadId}`);
        const etaElement = document.getElementById(`eta-${uploadId}`);

        if (progressFill) {
            progressFill.style.width = `${progress}%`;
            
            // Add speed-based color coding
            const speedClass = this.getSpeedClass(speed);
            progressFill.className = `progress-fill ${speedClass}`;
        }
        
        if (progressText) progressText.textContent = `${Math.round(progress)}%`;
        
        if (speedElement) {
            const formattedSpeed = this.formatSpeed(speed);
            const speedTrend = this.getSpeedTrend(uploadId, speed);
            speedElement.innerHTML = `
                <span class="speed-value">${formattedSpeed}</span>
                <span class="speed-trend ${speedTrend.direction}">${speedTrend.icon}</span>
            `;
        }
        
        if (chunksElement) {
            chunksElement.textContent = `${chunkIndex + 1}/${totalChunks || 'N/A'}`;
        }

        // Enhanced ETA calculation
        if (etaElement && speed > 0 && totalBytes) {
            const remainingBytes = totalBytes - (bytesUploaded || 0);
            const etaSeconds = remainingBytes / speed;
            etaElement.textContent = this.formatETA(etaSeconds);
        }

        // Update chunk indicator with animation
        const chunkIndicator = document.getElementById(`chunk-${uploadId}-${chunkIndex}`);
        if (chunkIndicator && !chunkIndicator.classList.contains('completed')) {
            chunkIndicator.classList.add('uploading');
            setTimeout(() => {
                chunkIndicator.classList.remove('uploading');
                chunkIndicator.classList.add('completed');
            }, 300);
        }

        // Update global stats
        this.updateGlobalStats();
    }

    getSpeedClass(speed) {
        const mbps = speed / (1024 * 1024);
        if (mbps > 10) return 'speed-excellent';
        if (mbps > 5) return 'speed-good';
        if (mbps > 1) return 'speed-average';
        return 'speed-slow';
    }

    getSpeedTrend(uploadId, currentSpeed) {
        if (!this.speedHistory) this.speedHistory = new Map();
        
        const history = this.speedHistory.get(uploadId) || [];
        history.push(currentSpeed);
        
        if (history.length > 5) history.shift();
        this.speedHistory.set(uploadId, history);
        
        if (history.length < 2) return { direction: 'stable', icon: '━' };
        
        const recent = history.slice(-3);
        const avgRecent = recent.reduce((a, b) => a + b, 0) / recent.length;
        const prevAvg = history.slice(-5, -2).reduce((a, b) => a + b, 0) / Math.max(1, history.length - 2);
        
        if (avgRecent > prevAvg * 1.1) return { direction: 'up', icon: '↗' };
        if (avgRecent < prevAvg * 0.9) return { direction: 'down', icon: '↘' };
        return { direction: 'stable', icon: '━' };
    }

    formatETA(seconds) {
        if (!isFinite(seconds) || seconds <= 0) return '--:--';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }

    updateGlobalStats() {
        const speedElement = this.getElement('upload-speed');
        const queueElement = this.getElement('queue-count');
        
        if (speedElement && this.speedHistory) {
            const allSpeeds = Array.from(this.speedHistory.values()).flat();
            const avgSpeed = allSpeeds.length > 0 ? 
                allSpeeds.reduce((a, b) => a + b, 0) / allSpeeds.length : 0;
            speedElement.textContent = this.formatSpeed(avgSpeed);
        }
        
        if (queueElement) {
            const activeUploads = document.querySelectorAll('.upload-item').length;
            queueElement.textContent = activeUploads.toString();
        }
    }

    handleUploadCompleted(uploadSession) {
        const statusElement = document.getElementById(`status-${uploadSession.id}`);
        if (statusElement) {
            statusElement.textContent = 'Completed';
            statusElement.classList.add('status-completed');
        }
    }

    handleUploadFailed(uploadSession) {
        const statusElement = document.getElementById(`status-${uploadSession.id}`);
        if (statusElement) {
            statusElement.textContent = 'Failed';
            statusElement.classList.add('status-error');
        }

        const retryBtn = document.getElementById(`retry-${uploadSession.id}`);
        if (retryBtn) {
            retryBtn.style.display = 'block';
        }
    }

    removeUploadItem(uploadId) {
        const uploadElement = document.getElementById(`upload-${uploadId}`);
        if (uploadElement) {
            uploadElement.remove();
        }
    }

    // Utility methods
    createElement(tagName, attributes = {}) {
        const element = document.createElement(tagName);
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else {
                element.setAttribute(key, value);
            }
        });
        return element;
    }

    insertIntoDOM(element) {
        const mainContent = document.querySelector('main') || document.body;
        const existingUpload = document.querySelector('#upload-section, #enhanced-upload-section');
        
        if (existingUpload) {
            existingUpload.replaceWith(element);
        } else {
            mainContent.insertBefore(element, mainContent.firstChild);
        }
    }

    cacheElements() {
        const elementIds = [
            'mainDropZone', 'dropZoneIcon', 'dropZoneTitle', 'dropZoneSubtitle',
            'browseFilesBtn', 'pasteUrlBtn', 'fileInput', 'filePreviewSection',
            'filePreviews', 'queueManager', 'uploadItems', 'pauseAllBtn',
            'resumeAllBtn', 'clearQueueBtn', 'upload-speed', 'queue-count', 'success-rate'
        ];

        elementIds.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                this.elements.set(id, element);
            }
        });
    }

    getElement(id) {
        return this.elements.get(id) || document.getElementById(id);
    }

    showSections(sectionIds) {
        sectionIds.forEach(id => {
            const section = this.getElement(id);
            if (section) {
                section.style.display = 'block';
            }
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlightDropZone(highlight) {
        const dropZone = this.getElement('mainDropZone');
        const icon = this.getElement('dropZoneIcon');
        const title = this.getElement('dropZoneTitle');
        const subtitle = this.getElement('dropZoneSubtitle');

        if (highlight) {
            dropZone.classList.add('drag-active');
            title.textContent = '🎯 Drop files to upload';
            subtitle.textContent = 'Release to start uploading';
            icon.style.transform = 'scale(1.1) rotate(5deg)';
        } else {
            dropZone.classList.remove('drag-active');
            title.textContent = 'Drop your video files here';
            subtitle.textContent = 'or click to browse files';
            icon.style.transform = 'scale(1) rotate(0deg)';
        }
    }

    getFileIcon(mimeType) {
        if (mimeType.startsWith('video/')) return '🎬';
        if (mimeType.startsWith('audio/')) return '🎵';
        return '📄';
    }

    formatFileSize(bytes) {
        const units = ['B', 'KB', 'MB', 'GB'];
        let size = bytes;
        let unitIndex = 0;

        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }

        return `${size.toFixed(2)} ${units[unitIndex]}`;
    }

    formatSpeed(bytesPerSecond) {
        if (bytesPerSecond === 0) return '0 MB/s';
        const mbps = bytesPerSecond / (1024 * 1024);
        return `${mbps.toFixed(1)} MB/s`;
    }
}

// ================================
// Analytics Module
// ================================

class AnalyticsModule {
    constructor(eventBus, performance) {
        this.eventBus = eventBus;
        this.performance = performance;
        this.analytics = {
            totalUploads: 0,
            successfulUploads: 0,
            failedUploads: 0,
            totalBytes: 0,
            averageSpeed: 0,
            errors: []
        };
    }

    async initialize() {
        this.setupEventHandlers();
        this.startPeriodicReporting();
    }

    setupEventHandlers() {
        this.eventBus.on('upload:queued', this.trackUploadStart.bind(this));
        this.eventBus.on('upload:completed', this.trackUploadSuccess.bind(this));
        this.eventBus.on('upload:failed', this.trackUploadFailure.bind(this));
        this.eventBus.on('upload:progress', this.trackUploadProgress.bind(this));
    }

    trackUploadStart(uploadSession) {
        this.analytics.totalUploads++;
        this.reportMetric('upload_started', {
            fileSize: uploadSession.file.size,
            fileType: uploadSession.file.type,
            chunks: uploadSession.totalChunks
        });
    }

    trackUploadSuccess(uploadSession) {
        this.analytics.successfulUploads++;
        this.analytics.totalBytes += uploadSession.file.size;
        
        const duration = (Date.now() - uploadSession.startTime) / 1000;
        const speed = uploadSession.file.size / duration;
        
        this.updateAverageSpeed(speed);
        
        this.reportMetric('upload_completed', {
            fileSize: uploadSession.file.size,
            duration,
            speed,
            retryCount: uploadSession.retryCount
        });
    }

    trackUploadFailure(uploadSession) {
        this.analytics.failedUploads++;
        
        this.reportMetric('upload_failed', {
            fileSize: uploadSession.file.size,
            errors: uploadSession.errors,
            retryCount: uploadSession.retryCount
        });
    }

    trackUploadProgress({ uploadId, progress, speed }) {
        this.reportMetric('upload_progress', {
            uploadId,
            progress,
            speed
        });
    }

    trackError(error) {
        this.analytics.errors.push({
            error: error.message || error,
            timestamp: Date.now(),
            stack: error.stack
        });

        this.reportMetric('error', {
            message: error.message || error,
            stack: error.stack
        });
    }

    updateAverageSpeed(speed) {
        const count = this.analytics.successfulUploads;
        this.analytics.averageSpeed = (
            (this.analytics.averageSpeed * (count - 1) + speed) / count
        );
    }

    reportMetric(type, data) {
        // In production, send to analytics service
        console.log(`Analytics: ${type}`, data);
    }

    startPeriodicReporting() {
        setInterval(() => {
            this.reportSystemMetrics();
        }, 30000); // Every 30 seconds
    }

    reportSystemMetrics() {
        const metrics = {
            ...this.analytics,
            performance: this.performance.getMetrics('system').slice(-1)[0],
            timestamp: Date.now()
        };

        this.reportMetric('system_metrics', metrics);
    }

    getAnalytics() {
        return {
            ...this.analytics,
            errorRate: this.analytics.totalUploads > 0 ? 
                this.analytics.failedUploads / this.analytics.totalUploads : 0,
            successRate: this.analytics.totalUploads > 0 ? 
                this.analytics.successfulUploads / this.analytics.totalUploads : 1
        };
    }
}

// ================================
// Realtime Module
// ================================

class RealtimeModule {
    constructor(eventBus, performance) {
        this.eventBus = eventBus;
        this.performance = performance;
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    async initialize() {
        await this.setupWebSocketConnection();
        this.setupEventHandlers();
    }

    async setupWebSocketConnection() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/v6/ws/enterprise/upload_manager`;
            
            this.socket = new WebSocket(wsUrl);
            
            this.socket.onopen = () => {
                console.log('🔗 WebSocket connected');
                this.reconnectAttempts = 0;
                this.eventBus.emit('realtime:connected');
            };
            
            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.socket.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                this.eventBus.emit('realtime:disconnected');
                this.handleReconnection();
            };
            
            this.socket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.eventBus.emit('realtime:error', error);
            };
            
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
        }
    }

    setupEventHandlers() {
        this.eventBus.on('upload:progress', this.sendProgressUpdate.bind(this));
        this.eventBus.on('upload:completed', this.sendCompletionUpdate.bind(this));
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'upload_progress':
                this.eventBus.emit('realtime:progress', data);
                break;
            case 'upload_complete':
                this.eventBus.emit('realtime:complete', data);
                break;
            case 'system_stats':
                this.eventBus.emit('realtime:stats', data);
                break;
        }
    }

    sendProgressUpdate(progressData) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({
                type: 'upload_progress',
                data: progressData
            }));
        }
    }

    sendCompletionUpdate(uploadSession) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({
                type: 'upload_complete',
                data: {
                    uploadId: uploadSession.id,
                    fileSize: uploadSession.file.size,
                    duration: Date.now() - uploadSession.startTime
                }
            }));
        }
    }

    handleReconnection() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.pow(2, this.reconnectAttempts) * 1000;
            
            setTimeout(() => {
                console.log(`🔄 Reconnecting... (attempt ${this.reconnectAttempts})`);
                this.setupWebSocketConnection();
            }, delay);
        }
    }
}

// ================================
// Global Initialization
// ================================

// Initialize upload system when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    if (typeof window.uploadSystem === 'undefined') {
        try {
            window.uploadSystem = new UploadSystemCore();
            await window.uploadSystem.initialize();
            console.log('🚀 Netflix-level upload system initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize upload system:', error);
            
            // Fallback to basic upload functionality
            document.body.innerHTML = `
                <div style="text-align: center; padding: 50px;">
                    <h2>Upload System Unavailable</h2>
                    <p>Please refresh the page or try again later.</p>
                    <button onclick="location.reload()">Refresh Page</button>
                </div>
            `;
        }
    }
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        UploadSystemCore,
        FileValidationModule,
        ChunkedUploaderModule,
        UIModule,
        AnalyticsModule,
        RealtimeModule
    };
}
