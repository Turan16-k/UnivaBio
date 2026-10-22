// Service worker: keeps LittleHeart working without internet after the first visit.
// Files are served from the cache straight away and refreshed in the background.

const CACHE = "littleheart-v3";
importScripts("model/metrics.js");   // provides METRICS, which lists the demo and quiz recordings

const FILES = [
  "./", "index.html", "manifest.webmanifest", "css/style.css",
  "js/i18n.js", "js/features.js", "js/model.js", "js/draw.js", "js/app.js", "js/guided.js",
  "js/quiz.js", "js/log.js", "js/validation.js", "js/main.js",
  "vendor/ort/ort.wasm.min.js", "vendor/ort/ort-wasm-simd-threaded.wasm", "vendor/ort/ort-wasm-simd-threaded.mjs",
  "model/murmur_cnn.onnx", "model/config.json", "model/heartbeat_check.json", "model/metrics.js", "model/external.json",
  "icons/icon.svg", "icons/icon-192.png", "icons/icon-512.png",
  ...METRICS.demo_cases.flatMap(c => c.recordings.map(r => "samples/" + r.file)),
  ...(METRICS.quiz || []).map(q => "quiz/" + q.file),
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== location.origin) return;
  event.respondWith(caches.open(CACHE).then(async cache => {
    const cached = await cache.match(event.request, { ignoreSearch: true });
    const fresh = fetch(event.request).then(res => {
      if (res.ok) cache.put(event.request, res.clone());
      return res;
    }).catch(() => cached);
    return cached || fresh;
  }));
});
