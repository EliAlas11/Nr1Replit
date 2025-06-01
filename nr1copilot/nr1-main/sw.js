
const CACHE_NAME = 'viralclip-pro-v2.1.0';
const STATIC_CACHE = 'viralclip-static-v2.1.0';
const DYNAMIC_CACHE = 'viralclip-dynamic-v2.1.0';

// Assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/public/styles.css',
  '/static/app.js',
  '/public/manifest.json',
  '/favicon.ico',
  '/public/offline.html'
];

// API endpoints to cache dynamically
const DYNAMIC_CACHE_URLS = [
  '/api/v2/health',
  '/api/v2/stats'
];

// Install event - cache static assets
self.addEventListener('install', event => {
  console.log('📦 Service Worker installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('📦 Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('✅ Static assets cached successfully');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('❌ Failed to cache static assets:', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('🔄 Service Worker activating...');
  
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (cacheName !== STATIC_CACHE && 
                cacheName !== DYNAMIC_CACHE && 
                cacheName !== CACHE_NAME) {
              console.log('🗑️ Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('✅ Service Worker activated');
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Skip WebSocket requests
  if (url.protocol === 'ws:' || url.protocol === 'wss:') {
    return;
  }

  // Skip external requests
  if (url.origin !== location.origin) {
    return;
  }

  // Handle different types of requests
  if (isStaticAsset(request.url)) {
    event.respondWith(handleStaticAsset(request));
  } else if (isAPIRequest(request.url)) {
    event.respondWith(handleAPIRequest(request));
  } else {
    event.respondWith(handleNavigationRequest(request));
  }
});

// Check if request is for a static asset
function isStaticAsset(url) {
  return STATIC_ASSETS.some(asset => url.includes(asset)) ||
         url.includes('/public/') ||
         url.includes('/static/') ||
         url.includes('.css') ||
         url.includes('.js') ||
         url.includes('.ico');
}

// Check if request is for an API endpoint
function isAPIRequest(url) {
  return url.includes('/api/') || 
         DYNAMIC_CACHE_URLS.some(endpoint => url.includes(endpoint));
}

// Handle static assets - cache first strategy
async function handleStaticAsset(request) {
  try {
    const cache = await caches.open(STATIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      // Return cached version and update cache in background
      updateCacheInBackground(request, cache);
      return cachedResponse;
    }
    
    // If not in cache, fetch and cache
    const networkResponse = await fetch(request);
    if (networkResponse.status === 200) {
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.error('Static asset fetch failed:', error);
    
    // Return offline page for HTML requests
    if (request.destination === 'document') {
      return caches.match('/public/offline.html');
    }
    
    throw error;
  }
}

// Handle API requests - network first with cache fallback
async function handleAPIRequest(request) {
  try {
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.status === 200) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('Network failed, trying cache for:', request.url);
    
    const cache = await caches.open(DYNAMIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline response for critical API endpoints
    if (request.url.includes('/health')) {
      return new Response(JSON.stringify({
        status: 'offline',
        cached: true,
        timestamp: new Date().toISOString()
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    throw error;
  }
}

// Handle navigation requests - network first with offline fallback
async function handleNavigationRequest(request) {
  try {
    const networkResponse = await fetch(request);
    return networkResponse;
  } catch (error) {
    console.log('Navigation failed, serving offline page');
    return caches.match('/public/offline.html');
  }
}

// Update cache in background
async function updateCacheInBackground(request, cache) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.status === 200) {
      cache.put(request, networkResponse.clone());
    }
  } catch (error) {
    // Silently fail - we already have cached version
  }
}

// Handle background sync
self.addEventListener('sync', event => {
  console.log('🔄 Background sync event:', event.tag);
  
  if (event.tag === 'upload-retry') {
    event.waitUntil(retryFailedUploads());
  }
});

// Retry failed uploads when back online
async function retryFailedUploads() {
  try {
    // Get failed uploads from IndexedDB or localStorage
    const failedUploads = getFailedUploads();
    
    for (const upload of failedUploads) {
      try {
        await fetch('/api/v2/upload-video', {
          method: 'POST',
          body: upload.formData
        });
        
        // Remove from failed uploads
        removeFailedUpload(upload.id);
        
        // Notify user of successful retry
        self.registration.showNotification('Upload Successful', {
          body: 'Your video upload was completed successfully',
          icon: '/public/icon-192x192.png',
          badge: '/public/icon-72x72.png',
          tag: 'upload-success'
        });
        
      } catch (error) {
        console.error('Failed to retry upload:', error);
      }
    }
  } catch (error) {
    console.error('Background sync failed:', error);
  }
}

// Handle push notifications
self.addEventListener('push', event => {
  console.log('📱 Push notification received');
  
  const options = {
    body: 'Your video processing is complete!',
    icon: '/public/icon-192x192.png',
    badge: '/public/icon-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'view',
        title: 'View Results',
        icon: '/public/icon-72x72.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/public/icon-72x72.png'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('ViralClip Pro', options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  console.log('📱 Notification clicked:', event.action);
  
  event.notification.close();
  
  if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Message handling for communication with main app
self.addEventListener('message', event => {
  console.log('📨 Message received:', event.data);
  
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data && event.data.type === 'CACHE_URLS') {
    event.waitUntil(
      cacheUrls(event.data.urls)
    );
  }
});

// Cache specific URLs on demand
async function cacheUrls(urls) {
  const cache = await caches.open(DYNAMIC_CACHE);
  return Promise.all(
    urls.map(url => {
      return fetch(url).then(response => {
        if (response.status === 200) {
          cache.put(url, response.clone());
        }
        return response;
      }).catch(error => {
        console.error('Failed to cache URL:', url, error);
      });
    })
  );
}

// Utility functions for managing failed uploads
function getFailedUploads() {
  // Implementation would use IndexedDB or localStorage
  return [];
}

function removeFailedUpload(id) {
  // Implementation would remove from IndexedDB or localStorage
}

// Periodic cache cleanup
function cleanupOldCaches() {
  const maxAge = 7 * 24 * 60 * 60 * 1000; // 7 days
  const now = Date.now();
  
  caches.keys().then(cacheNames => {
    cacheNames.forEach(cacheName => {
      if (cacheName.includes('dynamic')) {
        caches.open(cacheName).then(cache => {
          cache.keys().then(requests => {
            requests.forEach(request => {
              cache.match(request).then(response => {
                const dateHeader = response.headers.get('date');
                if (dateHeader) {
                  const cacheDate = new Date(dateHeader).getTime();
                  if (now - cacheDate > maxAge) {
                    cache.delete(request);
                  }
                }
              });
            });
          });
        });
      }
    });
  });
}

// Run cleanup periodically
setInterval(cleanupOldCaches, 24 * 60 * 60 * 1000); // Daily
