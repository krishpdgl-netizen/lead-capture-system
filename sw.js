// Event Lead Capture — Service Worker
// -------------------------------------------------------------------------
// Caches the app shell (login + dashboard pages) so that if the page ever
// reloads while offline — a pull-to-refresh gesture, an accidental tap on
// the browser's refresh icon, etc. — it still renders the app instead of
// the browser's native offline page. Without this, IndexedDB-queued leads
// are untouched, but the JS that reads/writes that queue never gets a
// chance to run again until the page can load from the network.
//
// Bump CACHE_NAME whenever index.html/dashboard.html change so returning
// online users pick up the new version instead of a stale cached shell.
const CACHE_NAME = "lead-capture-shell-v14";
const APP_SHELL = [
  "./index.html",
  "./dashboard.html",
  "./manifest.json",
  "./icon-192.png",
  "./icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
      )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  // Page loads/reloads: cache-first, so an offline reload always renders
  // the app shell instead of failing. Falls back to network to pick up a
  // fresher copy when available, and updates the cache with it.
  if (req.mode === "navigate") {
    event.respondWith(
      caches.match(req).then((cached) => {
        if (cached) return cached;
        return fetch(req)
          .then((res) => {
            const clone = res.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
            return res;
          })
          .catch(() => caches.match("./dashboard.html"));
      })
    );
    return;
  }

  // Same-origin static assets: cache-first, network fallback.
  const url = new URL(req.url);
  if (url.origin === self.location.origin) {
    event.respondWith(caches.match(req).then((cached) => cached || fetch(req)));
  }
});
