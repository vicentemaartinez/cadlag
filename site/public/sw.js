// càdlàg sin conexión: muestra lo último que se vio.
// Páginas y datos: primero la red (para tener lo más nuevo); si falla, la copia guardada.
// Archivos con hash (_next/static): primero la copia guardada, nunca cambian.
const CACHE = "cadlag-v1";
const ROOT = new URL(self.registration.scope).pathname;
self.addEventListener("install", (e) => { self.skipWaiting(); e.waitUntil(caches.open(CACHE).then((c) => c.add(ROOT)).catch(() => {})); });
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
const put = (req, res) => { if (res.ok) { const cp = res.clone(); caches.open(CACHE).then((c) => c.put(req, cp)); } return res; };
self.addEventListener("fetch", (e) => {
  const req = e.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith(ROOT + "_next/static/")) {
    e.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => put(req, res))));
    return;
  }
  e.respondWith(fetch(req).then((res) => put(req, res)).catch(() =>
    caches.match(req, { ignoreSearch: url.pathname.endsWith("/") }).then((hit) => hit || caches.match(ROOT))));
});
