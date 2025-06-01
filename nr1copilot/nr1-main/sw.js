
/**
 * ViralClip Pro Service Worker
 * Provides offline functionality and caching
 */

const CACHE_NAME = 'viralclip-pro-v1.0.0';
const STATIC_CACHE = 'viralclip-static-v1.0.0';
const DYNAMIC_CACHE = 'viralclip-dynamic-v1.0.0';

// Files to cache immediately
const STATIC_FILES = [
    '/',
    '/public/styles.css',
    '/static/app.js',
    '/public/manifest.json',
    '/public/offline.html',
    '/favicon.ico'
];

// Install event - cache static files
self.addEventListener('install', (event) => {
    console.log('Service Worker: Installing...');
    
    event.waitUntil(
        caches.open(STATIC_CACHE).then((cache) => {
            console.log('Service Worker: Caching static files');
            return cache.addAll(STATIC_FILES);
        }).catch((error) => {
            console.error('Service Worker: Failed to cache static files', error);
        })
    );
    
    // Force the waiting service worker to become the active service worker
    self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('Service Worker: Activating...');
    
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cache) => {
                    if (cache !== STATIC_CACHE && cache !== DYNAMIC_CACHE) {
                        console.log('Service Worker: Deleting old cache', cache);
                        return caches.delete(cache);
                    }
                })
            );
        })
    );
    
    // Ensure the service worker takes control immediately
    return self.clients.claim();
});

// Fetch event - serve cached files or fetch from network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    
    // Skip non-GET requests
    if (request.method !== 'GET') {
        return;
    }
    
    // Skip Chrome extension requests
    if (request.url.startsWith('chrome-extension://')) {
        return;
    }
    
    // Handle API requests differently
    if (request.url.includes('/api/')) {
        event.respondWith(handleApiRequest(request));
        return;
    }
    
    // Handle static file requests
    event.respondWith(handleStaticRequest(request));
});

// Handle API requests with network-first strategy
async function handleApiRequest(request) {
    try {
        // Always try network first for API requests
        const networkResponse = await fetch(request);
        
        // Cache successful responses
        if (networkResponse.ok) {
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.warn('Service Worker: Network request failed, trying cache', error);
        
        // Try to get from cache
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Return offline response for API failures
        return new Response(
            JSON.stringify({
                success: false,
                error: 'Network unavailable',
                offline: true
            }),
            {
                status: 503,
                statusText: 'Service Unavailable',
                headers: {
                    'Content-Type': 'application/json'
                }
            }
        );
    }
}

// Handle static file requests with cache-first strategy
async function handleStaticRequest(request) {
    try {
        // Try cache first
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // If not in cache, fetch from network
        const networkResponse = await fetch(request);
        
        // Cache the response for future use
        if (networkResponse.ok) {
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.warn('Service Worker: Failed to fetch', request.url, error);
        
        // If it's a page request and we're offline, show offline page
        if (request.mode === 'navigate') {
            const offlineResponse = await caches.match('/public/offline.html');
            if (offlineResponse) {
                return offlineResponse;
            }
        }
        
        // Return a generic offline response
        return new Response(
            'Offline - Content not available',
            {
                status: 503,
                statusText: 'Service Unavailable',
                headers: {
                    'Content-Type': 'text/plain'
                }
            }
        );
    }
}

// Handle background sync
self.addEventListener('sync', (event) => {
    if (event.tag === 'background-sync') {
        console.log('Service Worker: Background sync triggered');
        event.waitUntil(doBackgroundSync());
    }
});

async function doBackgroundSync() {
    // Implement background sync logic here
    // For example, retry failed uploads or API calls
    console.log('Service Worker: Performing background sync');
}

// Handle push notifications
self.addEventListener('push', (event) => {
    console.log('Service Worker: Push message received');
    
    const options = {
        body: event.data ? event.data.text() : 'New update available',
        icon: '/favicon.ico',
        badge: '/favicon.ico',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'Open ViralClip Pro',
                icon: '/favicon.ico'
            },
            {
                action: 'close',
                title: 'Close notification',
                icon: '/favicon.ico'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('ViralClip Pro', options)
    );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
    console.log('Service Worker: Notification click received');
    
    event.notification.close();
    
    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

// Handle message events from the main thread
self.addEventListener('message', (event) => {
    console.log('Service Worker: Message received', event.data);
    
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
    
    if (event.data && event.data.type === 'GET_VERSION') {
        event.ports[0].postMessage({ version: CACHE_NAME });
    }
});

// Utility function to clean up old caches
async function cleanupCaches() {
    const cacheNames = await caches.keys();
    const currentCaches = [STATIC_CACHE, DYNAMIC_CACHE];
    
    return Promise.all(
        cacheNames
            .filter(cacheName => !currentCaches.includes(cacheName))
            .map(cacheName => caches.delete(cacheName))
    );
}

// Periodically clean up caches
setInterval(cleanupCaches, 24 * 60 * 60 * 1000); // Once per day

console.log('Service Worker: Loaded successfully');
