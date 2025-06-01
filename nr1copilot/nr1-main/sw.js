
/**
 * ViralClip Pro Service Worker
 * Enhanced offline support and caching strategy
 */

const CACHE_NAME = 'viralclip-pro-v2.0.0';
const STATIC_CACHE = 'viralclip-static-v2.0.0';
const DYNAMIC_CACHE = 'viralclip-dynamic-v2.0.0';

// Files to cache immediately
const STATIC_FILES = [
    '/',
    '/index.html',
    '/public/styles.css',
    '/static/app.js',
    '/public/manifest.json',
    '/public/offline.html'
];

// API endpoints to cache
const API_CACHE_PATTERNS = [
    /^https?:\/\/.*\/api\/v2\/health$/,
    /^https?:\/\/.*\/api\/v2\/metrics$/
];

// Install event
self.addEventListener('install', (event) => {
    console.log('SW: Installing Service Worker');
    
    event.waitUntil(
        Promise.all([
            // Cache static files
            caches.open(STATIC_CACHE).then((cache) => {
                console.log('SW: Caching static files');
                return cache.addAll(STATIC_FILES.map(url => new Request(url, {
                    cache: 'reload'
                })));
            }),
            
            // Skip waiting to activate immediately
            self.skipWaiting()
        ])
    );
});

// Activate event
self.addEventListener('activate', (event) => {
    console.log('SW: Activating Service Worker');
    
    event.waitUntil(
        Promise.all([
            // Clean up old caches
            caches.keys().then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => {
                        if (cacheName !== STATIC_CACHE && 
                            cacheName !== DYNAMIC_CACHE && 
                            cacheName !== CACHE_NAME) {
                            console.log('SW: Deleting old cache:', cacheName);
                            return caches.delete(cacheName);
                        }
                    })
                );
            }),
            
            // Take control of all clients
            self.clients.claim()
        ])
    );
});

// Fetch event
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Skip non-GET requests
    if (request.method !== 'GET') {
        return;
    }
    
    // Skip WebSocket connections
    if (request.headers.get('upgrade') === 'websocket') {
        return;
    }
    
    // Skip chrome-extension and other non-http(s) schemes
    if (!url.protocol.startsWith('http')) {
        return;
    }
    
    event.respondWith(
        handleFetchRequest(request)
    );
});

async function handleFetchRequest(request) {
    const url = new URL(request.url);
    
    try {
        // Static files - cache first strategy
        if (STATIC_FILES.some(staticUrl => url.pathname === staticUrl || url.pathname.endsWith(staticUrl))) {
            return await cacheFirstStrategy(request, STATIC_CACHE);
        }
        
        // API endpoints - network first with cache fallback
        if (url.pathname.startsWith('/api/')) {
            if (API_CACHE_PATTERNS.some(pattern => pattern.test(request.url))) {
                return await networkFirstStrategy(request, DYNAMIC_CACHE);
            } else {
                // Don't cache other API endpoints (real-time data)
                return await fetch(request);
            }
        }
        
        // Images and assets - cache first
        if (request.destination === 'image' || 
            url.pathname.match(/\.(jpg|jpeg|png|gif|webp|svg|ico)$/)) {
            return await cacheFirstStrategy(request, DYNAMIC_CACHE);
        }
        
        // Other requests - network first
        return await networkFirstStrategy(request, DYNAMIC_CACHE);
        
    } catch (error) {
        console.warn('SW: Fetch failed:', error);
        
        // Return offline page for navigation requests
        if (request.mode === 'navigate') {
            const offlineResponse = await caches.match('/public/offline.html');
            if (offlineResponse) {
                return offlineResponse;
            }
        }
        
        // Return a basic offline response
        return new Response('Offline', {
            status: 503,
            statusText: 'Service Unavailable',
            headers: new Headers({
                'Content-Type': 'text/plain'
            })
        });
    }
}

async function cacheFirstStrategy(request, cacheName) {
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
        console.log('SW: Serving from cache:', request.url);
        
        // Update cache in background for static files
        if (cacheName === STATIC_CACHE) {
            fetch(request).then(response => {
                if (response.ok) {
                    cache.put(request, response.clone());
                }
            }).catch(() => {
                // Ignore network errors in background update
            });
        }
        
        return cachedResponse;
    }
    
    // Not in cache, fetch from network
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
        console.log('SW: Caching new resource:', request.url);
        cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
}

async function networkFirstStrategy(request, cacheName) {
    try {
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, networkResponse.clone());
            console.log('SW: Updated cache from network:', request.url);
        }
        
        return networkResponse;
        
    } catch (error) {
        console.log('SW: Network failed, trying cache:', request.url);
        
        const cache = await caches.open(cacheName);
        const cachedResponse = await cache.match(request);
        
        if (cachedResponse) {
            console.log('SW: Serving stale from cache:', request.url);
            return cachedResponse;
        }
        
        throw error;
    }
}

// Background sync for failed uploads
self.addEventListener('sync', (event) => {
    if (event.tag === 'background-upload') {
        console.log('SW: Background sync triggered');
        event.waitUntil(handleBackgroundSync());
    }
});

async function handleBackgroundSync() {
    // Handle failed uploads when connection is restored
    try {
        const failedUploads = await getFailedUploads();
        
        for (const upload of failedUploads) {
            try {
                await retryUpload(upload);
                await removeFailedUpload(upload.id);
            } catch (error) {
                console.warn('SW: Failed to retry upload:', upload.id, error);
            }
        }
    } catch (error) {
        console.error('SW: Background sync error:', error);
    }
}

async function getFailedUploads() {
    // Get failed uploads from IndexedDB or localStorage
    const failedUploads = localStorage.getItem('failed-uploads');
    return failedUploads ? JSON.parse(failedUploads) : [];
}

async function retryUpload(upload) {
    // Retry the upload
    const response = await fetch(upload.url, {
        method: upload.method,
        body: upload.body,
        headers: upload.headers
    });
    
    if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
    }
    
    return response;
}

async function removeFailedUpload(uploadId) {
    const failedUploads = await getFailedUploads();
    const filtered = failedUploads.filter(upload => upload.id !== uploadId);
    localStorage.setItem('failed-uploads', JSON.stringify(filtered));
}

// Push notifications (for future use)
self.addEventListener('push', (event) => {
    console.log('SW: Push notification received');
    
    const options = {
        body: event.data ? event.data.text() : 'Your video processing is complete!',
        icon: '/public/icons/icon-192x192.png',
        badge: '/public/icons/badge-72x72.png',
        tag: 'viralclip-notification',
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'View Results',
                icon: '/public/icons/checkmark.png'
            },
            {
                action: 'close',
                title: 'Close',
                icon: '/public/icons/xmark.png'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('ViralClip Pro', options)
    );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
    console.log('SW: Notification click received');
    
    event.notification.close();
    
    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

// Message handler for communication with main thread
self.addEventListener('message', (event) => {
    console.log('SW: Message received:', event.data);
    
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
    
    if (event.data && event.data.type === 'CACHE_URLS') {
        event.waitUntil(
            cacheUrls(event.data.payload)
        );
    }
});

async function cacheUrls(urls) {
    const cache = await caches.open(DYNAMIC_CACHE);
    await cache.addAll(urls);
    console.log('SW: Cached additional URLs:', urls);
}

// Error handler
self.addEventListener('error', (event) => {
    console.error('SW: Service Worker error:', event.error);
});

// Unhandled rejection handler
self.addEventListener('unhandledrejection', (event) => {
    console.error('SW: Unhandled promise rejection:', event.reason);
});

console.log('SW: Service Worker loaded successfully');
