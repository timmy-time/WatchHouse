// ============================================================================
// CCTV Console — web client
// Live: hero stage + filmstrip + event rail, client-side canvas overlays.
// ============================================================================

let currentRoute = "";
let sseSource = null;
let statusInterval = null;
let detectionsInterval = null;
let clockInterval = null;
let showClientOverlays = true;

// Live-view state
let liveCameras = {};
let liveHeroSlug = null;
let liveDetections = {}; // slug -> { detections, zones, slots }

// ============================================================================
// Routing
// ============================================================================
function initRouter() {
  window.addEventListener("hashchange", handleRoute);
  if (!window.location.hash) {
    window.location.hash = "#/live";
  } else {
    handleRoute();
  }
}

function handleRoute() {
  const hash = window.location.hash || "#/live";
  currentRoute = hash;

  const inLive = hash.startsWith("#/live");
  if (!inLive) {
    if (statusInterval) { clearInterval(statusInterval); statusInterval = null; }
    if (detectionsInterval) { clearInterval(detectionsInterval); detectionsInterval = null; }
    stopOverlayLoop();
  }

  document.querySelectorAll(".nav-link").forEach((link) => {
    const route = link.getAttribute("href");
    link.classList.toggle("active", hash.startsWith(route));
  });

  document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));

  if (hash === "#/live" || hash === "#/") {
    document.getElementById("view-live").classList.remove("hidden");
    loadLiveView();
  } else if (hash.startsWith("#/events/")) {
    const eventId = hash.replace("#/events/", "");
    document.getElementById("view-events").classList.remove("hidden");
    loadEventsView();
    openEventModal(eventId);
  } else if (hash.startsWith("#/events")) {
    document.getElementById("view-events").classList.remove("hidden");
    loadEventsView();
  } else if (hash.startsWith("#/faces")) {
    document.getElementById("view-faces").classList.remove("hidden");
    loadFacesView();
  } else if (hash.startsWith("#/scenery")) {
    document.getElementById("view-scenery").classList.remove("hidden");
    loadSceneryView();
  } else if (hash.startsWith("#/archive")) {
    document.getElementById("view-archive").classList.remove("hidden");
    loadArchiveView();
  }
}

// ============================================================================
// Global SSE, clock, toasts
// ============================================================================
function initSSE() {
  if (sseSource) sseSource.close();
  sseSource = new EventSource("/api/stream");

  sseSource.addEventListener("notification", (e) => {
    try {
      const data = JSON.parse(e.data);
      showToast(data.title || "Notification", data.body || "");
      if (Notification.permission === "granted") {
        new Notification(data.title, { body: data.body });
      }
    } catch (err) {
      console.error("SSE notification parse error:", err);
    }
  });

  sseSource.addEventListener("event_update", (e) => {
    try {
      const data = JSON.parse(e.data);
      addLiveFeedItem(data);
      if (currentRoute.startsWith("#/events")) loadEventsView();
    } catch (err) {
      console.error("SSE event parse error:", err);
    }
  });

  sseSource.onerror = () => console.warn("SSE interrupted; browser will retry");
}

function initClock() {
  const el = document.getElementById("clock");
  const tick = () => {
    const now = new Date();
    el.textContent = now.toLocaleTimeString([], { hour12: false });
    el.setAttribute("datetime", now.toISOString());
  };
  tick();
  clearInterval(clockInterval);
  clockInterval = setInterval(tick, 1000);
}

function showToast(title, body) {
  const container = document.getElementById("toasts-container");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(body)}</span>`;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 5200);
}

// ============================================================================
// LIVE VIEW
// ============================================================================
async function loadLiveView() {
  await fetchLiveStatus();
  await pollLiveDetections();
  await prefillFeed();

  startOverlayLoop();
  if (!statusInterval) statusInterval = setInterval(fetchLiveStatus, 5000);
  if (!detectionsInterval) detectionsInterval = setInterval(pollLiveDetections, 250);
}

// --- system chips in the top bar ---
async function fetchLiveStatus() {
  try {
    const res = await fetch("/api/live/status");
    const data = await res.json();
    liveCameras = data.cameras || {};
    renderSysChips(data.gpus || {}, liveCameras);
    renderLiveTiles(liveCameras);
  } catch (err) {
    console.error("live status error:", err);
  }
}

function renderSysChips(gpus, cameras) {
  const host = document.getElementById("sys-chips");
  const parts = [];

  const gpuKeys = Object.keys(gpus).sort();
  gpuKeys.forEach((k) => {
    const g = gpus[k];
    const util = g.utilization_pct ?? 0;
    parts.push(`<span class="chip chip-accent" title="${escapeHtml(g.name || "")} — VRAM ${g.memory_used_mb || 0}/${g.memory_total_mb || "?"} MB">GPU${escapeHtml(k)} ${util}%</span>`);
  });

  const camList = Object.values(cameras);
  const connected = camList.filter((c) => c.connected).length;
  parts.push(`<span class="chip ${connected === camList.length && camList.length ? "chip-ok" : "chip-warn"}">${connected}/${camList.length} cams</span>`);

  const boosting = camList.filter((c) => c.mode === "boost").length;
  if (boosting > 0) parts.push(`<span class="chip chip-warn">${boosting} boost</span>`);

  host.innerHTML = parts.join("");
}

// --- tile construction ---
function tileInner(name, cam, isHero) {
  const slug = encodeURIComponent(cam.slug);
  const mode = (cam.mode || "idle").toUpperCase();
  const modeCls = cam.mode === "boost" ? "chip-warn" : "";
  return `
    <div class="cam-head">
      <span class="cam-name">${escapeHtml(name)}</span>
      <span class="spacer"></span>
      <span class="chip js-recinfo" style="display:none" title="No recordings are written for this camera">LIVE-ONLY</span>
      <span class="chip js-event ${cam.open_event_id ? "chip-accent" : ""}" ${cam.open_event_id ? "" : 'style="display:none"'}>rec #${cam.open_event_id || ""}</span>
      <span class="chip js-mode ${modeCls}">${mode}</span>
    </div>
    <div class="cam-media">
      <img src="/api/live/${slug}/stream.mjpg"
           alt="${escapeHtml(name)} live stream"
           onerror="this.onerror=null;this.src='/api/live/${slug}/snapshot.jpg'">
      <canvas class="camera-live-overlay"></canvas>
      <div class="cam-hud">
        <span class="badge js-conn">…</span>
        <span class="badge badge-decode js-decode">${escapeHtml((cam.decode || "cpu").toUpperCase())}</span>
        <span class="badge js-src">${escapeHtml((cam.source || "main").toUpperCase())}</span>
        <span class="badge js-fps">0 fps</span>
        <span class="badge js-infer">0 infer</span>
      </div>
      ${isHero ? `<div class="cam-actions">
        <button class="icon-btn js-full" title="Fullscreen">⛶</button>
      </div>` : ""}
    </div>`;
}

function buildTile(name, cam, isHero) {
  const tile = document.createElement("article");
  tile.className = "cam-tile" + (isHero ? " is-hero" : "");
  tile.dataset.slug = cam.slug;
  tile.innerHTML = tileInner(name, cam, isHero);

  if (isHero) {
    tile.querySelector(".js-full").onclick = (e) => {
      e.stopPropagation();
      if (document.fullscreenElement) document.exitFullscreen();
      else tile.requestFullscreen && tile.requestFullscreen();
    };
  } else {
    tile.onclick = () => {
      liveHeroSlug = cam.slug;
      try { localStorage.setItem("liveHeroSlug", cam.slug); } catch (_) {}
      renderLiveTiles(liveCameras);
      drawAllOverlays();
    };
  }
  return tile;
}

function renderLiveTiles(cameras) {
  const names = Object.keys(cameras);
  if (names.length === 0) return;

  // liveHeroSlug holds a slug; cameras is keyed by NAME — resolve via slug map.
  const bySlug = camListByName(cameras);
  if (!liveHeroSlug) {
    try { liveHeroSlug = localStorage.getItem("liveHeroSlug") || null; } catch (_) {}
  }
  if (!liveHeroSlug || !bySlug[liveHeroSlug]) liveHeroSlug = cameras[names[0]].slug;

  const stage = document.getElementById("stage");
  const strip = document.getElementById("filmstrip");

  // Rebuild tiles only when the camera set or hero selection changes
  const signature = `${liveHeroSlug}::${names.map((n) => cameras[n].slug).sort().join("|")}`;
  if (stage.dataset.signature !== signature) {
    stage.dataset.signature = signature;
    stage.innerHTML = "";
    strip.innerHTML = "";
    names.forEach((n) => {
      const cam = cameras[n];
      if (cam.slug === liveHeroSlug) stage.appendChild(buildTile(n, cam, true));
      else strip.appendChild(buildTile(n, cam, false));
    });
  }

  // Live-update chips (without touching canvases)
  document.querySelectorAll(".cam-tile").forEach((tile) => {
    const cam = camListByName(cameras)[tile.dataset.slug];
    if (!cam) return;
    const mode = (cam.mode || "idle").toUpperCase();
    const modeEl = tile.querySelector(".js-mode");
    modeEl.textContent = mode;
    modeEl.className = "chip js-mode" + (cam.mode === "boost" ? " chip-warn" : "");

    const evEl = tile.querySelector(".js-event");
    if (cam.open_event_id) {
      evEl.style.display = "";
      evEl.textContent = "rec #" + cam.open_event_id;
      evEl.className = "chip js-event chip-accent";
    } else {
      evEl.style.display = "none";
    }

    const recEl = tile.querySelector(".js-recinfo");
    if (recEl) recEl.style.display = cam.record === false ? "" : "none";

    const conn = tile.querySelector(".js-conn");
    if (cam.connected) {
      conn.className = "badge js-conn badge-connected";
      conn.textContent = "LIVE";
    } else {
      conn.className = "badge js-conn badge-disconnected";
      conn.textContent = "OFFLINE";
    }

    const dec = tile.querySelector(".js-decode");
    if (dec) dec.textContent = (cam.decode || "cpu").toUpperCase();
    const src = tile.querySelector(".js-src");
    if (src) src.textContent = (cam.source || "main").toUpperCase();
    const fps = tile.querySelector(".js-fps");
    if (fps) fps.textContent = `${cam.fps_in ?? 0} fps in`;
    const infer = tile.querySelector(".js-infer");
    if (infer) infer.textContent = `${cam.fps_analyzed ?? 0} infer`;
    tile.classList.toggle("is-active", tile.dataset.slug === liveHeroSlug);
  });
}

function camListByName(cameras) {
  const map = {};
  Object.values(cameras).forEach((c) => { map[c.slug] = c; });
  return map;
}

// --- detections polling + predicted overlay rendering ---
let overlayRaf = null;
let lastOverlayDraw = 0;
const renderedBoxes = new Map(); // "slug:track_id" -> {x1,y1,x2,y2,lastSeen}

function startOverlayLoop() {
  if (overlayRaf === null) overlayRaf = requestAnimationFrame(overlayTick);
}

function stopOverlayLoop() {
  if (overlayRaf !== null) cancelAnimationFrame(overlayRaf);
  overlayRaf = null;
}

function overlayTick(ts) {
  overlayRaf = requestAnimationFrame(overlayTick);
  if (ts - lastOverlayDraw < 33) return; // ~30 fps is plenty and keeps CPU low
  lastOverlayDraw = ts;
  if (document.getElementById("view-live").classList.contains("hidden")) return;

  // Use the Unix clock: detection timestamps come from the server in epoch seconds
  const nowSec = Date.now() / 1000;
  drawAllOverlays(nowSec);
  pruneRenderedBoxes(nowSec);
}

function pruneRenderedBoxes(now) {
  for (const [key, st] of renderedBoxes) {
    if (now - st.lastSeen > 2.5) renderedBoxes.delete(key);
  }
}

async function pollLiveDetections() {
  if (document.getElementById("view-live").classList.contains("hidden")) return;
  const cams = Object.values(liveCameras);
  await Promise.all(cams.map(async (cam) => {
    try {
      const r = await fetch(`/api/live/${encodeURIComponent(cam.slug)}/detections`);
      if (r.ok) liveDetections[cam.slug] = await r.json();
    } catch (_) { /* ignore transient errors */ }
  }));
}

function drawAllOverlays(now = Date.now() / 1000) {
  document.querySelectorAll(".cam-tile").forEach((tile) => {
    drawTileOverlay(tile, liveDetections[tile.dataset.slug], now);
  });
}

/**
 * Letterbox-correct overlay. The <img> uses object-fit: contain, so boxes must
 * be mapped into the *rendered image rect*, not the raw canvas rect.
 */
function drawTileOverlay(tile, data, now = Date.now() / 1000) {
  const media = tile.querySelector(".cam-media");
  const canvas = tile.querySelector(".camera-live-overlay");
  const img = tile.querySelector(".cam-media img");
  if (!media || !canvas || !img) return;

  const cw = media.clientWidth;
  const ch = media.clientHeight;
  if (cw === 0 || ch === 0) return;
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw;
    canvas.height = ch;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  if (!showClientOverlays || !data) return;

  const isHero = tile.classList.contains("is-hero");
  const fs = isHero ? 12 : 10;

  // Rendered image rect (account for object-fit: contain letterboxing)
  const iw = img.naturalWidth || 16;
  const ih = img.naturalHeight || 9;
  const scale = Math.min(cw / iw, ch / ih);
  const dw = iw * scale;
  const dh = ih * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;
  const px = (n) => [ox + n[0] * dw, oy + n[1] * dh];

  // 1. Zones
  (data.zones || []).forEach((z) => {
    const poly = z.polygon || [];
    if (poly.length < 3) return;
    const pts = poly.map(px);
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
    ctx.setLineDash([6, 5]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = z.type === "ignore" ? "rgba(150,160,175,0.55)" : "rgba(91,200,232,0.6)";
    ctx.stroke();
    ctx.setLineDash([]);
    if (isHero && poly[0]) {
      const [lx, ly] = px(poly[0]);
      drawPill(ctx, `${z.type === "ignore" ? "ignore" : ""} ${z.name}`.trim(), lx, ly - 4, "#26303c", "rgba(200,215,230,0.85)", fs - 1);
    }
  });

  // 2. Vehicle scenery slots
  (data.slots || []).forEach((s) => {
    const b = s.slot_box || [];
    if (b.length !== 4) return;
    const [x1, y1] = px([b[0], b[1]]);
    const [x2, y2] = px([b[2], b[3]]);
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = s.is_friendly ? "rgba(227,165,69,0.75)" : "rgba(229,96,79,0.75)";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    if (isHero) drawPill(ctx, s.name, x1, y1 - 4, "#2a2416", "rgba(240,205,140,0.95)", fs - 1);
  });

  // 3. Detections — extrapolated with per-track velocity between polls and
  //    smoothed toward each new measurement so boxes glide instead of jumping.
  const dataAge = Math.max(0, now - (data.updated_at || now));
  const coast = Math.min(dataAge, 0.6); // seconds of forward prediction
  const slug = tile.dataset.slug;

  (data.detections || []).forEach((d) => {
    const b = d.box_norm || [];
    if (b.length !== 4) return;

    const vel = d.vel || [0, 0];
    // Predicted target in normalized units
    const tx1 = b[0] + vel[0] * coast;
    const ty1 = b[1] + vel[1] * coast;
    const tx2 = b[2] + vel[0] * coast;
    const ty2 = b[3] + vel[1] * coast;

    const key = `${slug}:${d.track_id !== undefined ? d.track_id : `${b[0]},${b[1]}`}`;
    let st = renderedBoxes.get(key);
    if (!st) {
      st = { x1: tx1, y1: ty1, x2: tx2, y2: ty2, lastSeen: now };
      renderedBoxes.set(key, st);
    } else {
      const a = 0.35; // EMA correction rate
      st.x1 += (tx1 - st.x1) * a;
      st.y1 += (ty1 - st.y1) * a;
      st.x2 += (tx2 - st.x2) * a;
      st.y2 += (ty2 - st.y2) * a;
      st.lastSeen = now;
    }

    const [x1, y1] = px([st.x1, st.y1]);
    const [x2, y2] = px([st.x2, st.y2]);
    const w = x2 - x1;
    const h = y2 - y1;

    let color = "#5bc8e8"; // default marker
    if (d.is_anchored) color = "#e3a545";
    else if (d.class_name === "person") color = "#3ecf9a";
    else if (["car", "truck", "bus"].includes(d.class_name)) color = "#e5604f";

    ctx.strokeStyle = color;
    ctx.lineWidth = isHero ? 2 : 1.5;
    ctx.strokeRect(x1, y1, w, h);

    const label = d.anchored_name
      ? `${d.anchored_name} #${d.track_id} ${d.conf}`
      : `${d.class_name} #${d.track_id} ${d.conf}`;
    drawPill(ctx, label, x1, y1 - 2, tintDark(color), "#ffffff", fs);
  });
}

function drawPill(ctx, text, x, y, bg, fg, fontSize) {
  ctx.font = `600 ${fontSize}px "IBM Plex Mono", monospace`;
  const tw = ctx.measureText(text).width;
  const pad = 5;
  const h = fontSize + 8;
  const px = Math.max(0, x);
  const py = Math.max(0, y - h);
  ctx.fillStyle = bg;
  ctx.globalAlpha = 0.88;
  ctx.fillRect(px, py, tw + pad * 2, h);
  ctx.globalAlpha = 1;
  ctx.fillStyle = fg;
  ctx.fillText(text, px + pad, py + fontSize + 3);
}

function tintDark(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${Math.round(r * 0.22)},${Math.round(g * 0.22)},${Math.round(b * 0.22)},0.9)`;
}

// --- event feed ---
async function prefillFeed() {
  const feed = document.getElementById("live-feed-list");
  if (feed.dataset.prefilled === "1") return;
  feed.dataset.prefilled = "1";
  try {
    const res = await fetch("/api/events?limit=8");
    const data = await res.json();
    const items = (data.items || []).slice().reverse();
    items.forEach((ev) => addLiveFeedItem(ev, true));
  } catch (_) { /* keep empty state */ }
}

function addLiveFeedItem(ev, quiet = false) {
  const feed = document.getElementById("live-feed-list");
  if (!feed) return;
  const empty = feed.querySelector(".empty");
  if (empty) empty.remove();

  const sev = ev.primary_class === "person" ? "sev-person"
    : ev.primary_class === "vehicle" ? "sev-vehicle" : "sev-alert";

  const item = document.createElement("div");
  item.className = `feed-item ${sev}`;

  const t = ev.started_at
    ? new Date(ev.started_at * 1000).toLocaleTimeString([], { hour12: false })
    : "";
  const chips = (ev.behaviors || []).map((b) => `<span class="chip">${escapeHtml(b)}</span>`).join("");

  item.innerHTML = `
    <div class="feed-meta">
      <span class="feed-cam">${escapeHtml(ev.camera || "")}</span>
      <span class="feed-time">${t}</span>
    </div>
    <div class="feed-title">${escapeHtml(ev.primary_class || "event")} <span class="id">#${ev.id}</span></div>
    ${chips ? `<div class="feed-chips">${chips}</div>` : ""}`;

  item.onclick = () => { window.location.hash = `#/events/${ev.id}`; };

  if (quiet) feed.appendChild(item);
  else feed.prepend(item);

  while (feed.children.length > 40) feed.removeChild(feed.lastChild);
}

// ============================================================================
// EVENTS VIEW
// ============================================================================
let eventsOffset = 0;
const EVENTS_LIMIT = 30;

async function loadEventsView() {
  await populateEventFilters();
  await fetchEvents();
}

async function populateEventFilters() {
  try {
    const statRes = await fetch("/api/live/status");
    const statData = await statRes.json();
    const camSelect = document.getElementById("event-filter-camera");
    if (camSelect && camSelect.options.length <= 1) {
      Object.keys(statData.cameras || {}).forEach((camName) => {
        const opt = document.createElement("option");
        opt.value = camName;
        opt.textContent = camName;
        camSelect.appendChild(opt);
      });
    }

    const idRes = await fetch("/api/identities");
    const idData = await idRes.json();
    const idSelect = document.getElementById("event-filter-identity");
    if (idSelect && idSelect.options.length <= 1) {
      idData.forEach((ident) => {
        const opt = document.createElement("option");
        opt.value = ident.id;
        opt.textContent = ident.name;
        idSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.error("filter populate error:", err);
  }
}

async function fetchEvents(offset = 0) {
  eventsOffset = offset;
  const camera = document.getElementById("event-filter-camera").value;
  const behavior = document.getElementById("event-filter-behavior").value;
  const identityId = document.getElementById("event-filter-identity").value;
  const dateVal = document.getElementById("event-filter-date").value;

  const params = new URLSearchParams({
    offset: offset.toString(),
    limit: EVENTS_LIMIT.toString(),
  });
  if (camera) params.set("camera", camera);
  if (behavior) params.set("behavior", behavior);
  if (identityId) params.set("identity_id", identityId);
  if (dateVal) {
    const startEpoch = new Date(`${dateVal}T00:00:00Z`).getTime() / 1000;
    params.set("since", startEpoch.toString());
    params.set("until", (startEpoch + 86400).toString());
  }

  try {
    const res = await fetch(`/api/events?${params.toString()}`);
    const data = await res.json();
    renderEventsGrid(data.items || []);
    renderPagination("events-pagination", data.total || 0, offset, EVENTS_LIMIT, fetchEvents);
  } catch (err) {
    console.error("fetch events error:", err);
  }
}

function renderEventsGrid(items) {
  const grid = document.getElementById("events-grid");
  grid.innerHTML = "";

  if (items.length === 0) {
    grid.innerHTML = `<div class="empty" style="grid-column: 1/-1;"><strong>No events match</strong>Loosen the filters or check back later.</div>`;
    return;
  }

  items.forEach((ev) => {
    const card = document.createElement("div");
    card.className = "event-card";

    const dateStr = ev.started_at ? new Date(ev.started_at * 1000).toLocaleString() : "";
    const chips = (ev.behaviors || []).map((b) => `<span class="chip">${escapeHtml(b)}</span>`).join("");
    const identities = (ev.identities || []).map((n) => `<span class="chip chip-identity">${escapeHtml(n)}</span>`).join("");
    const cls = (ev.primary_class || "event");
    const clsChip = cls === "person" ? "chip-person" : cls === "vehicle" ? "chip-vehicle" : "";
    const status = ev.status === "finalized" ? "" : `<span class="chip chip-warn">${escapeHtml(ev.status || "")}</span>`;

    card.innerHTML = `
      <div class="card-thumb ${ev.thumb_url ? "" : "no-thumb"}">
        ${ev.thumb_url ? `<img src="${ev.thumb_url}" alt="Event ${ev.id} thumbnail" loading="lazy">` : ""}
      </div>
      <div class="card-body">
        <div class="card-meta">
          <span class="card-cam">${escapeHtml(ev.camera || "")}</span>
          <span>${dateStr}</span>
        </div>
        <div class="card-title">
          <span class="chip ${clsChip}">${escapeHtml(cls)}</span>
          <span class="id">#${ev.id}</span>
          ${status}
        </div>
        <div class="chips">${chips}${identities}</div>
      </div>`;

    card.onclick = () => { window.location.hash = `#/events/${ev.id}`; };
    grid.appendChild(card);
  });
}

// ============================================================================
// EVENT / CLIP MODAL
// ============================================================================
async function openEventModal(eventId) {
  const modal = document.getElementById("modal");
  const body = document.getElementById("modal-body");
  body.innerHTML = `<div class="empty"><strong>Loading event ${escapeHtml(String(eventId))}</strong>Fetching clip and detections…</div>`;
  modal.classList.remove("hidden");

  try {
    const res = await fetch(`/api/events/${eventId}`);
    if (!res.ok) {
      body.innerHTML = `<div class="empty"><strong>Event ${escapeHtml(String(eventId))} not found</strong>It may have been removed.</div>`;
      return;
    }
    const ev = await res.json();
    const dateStr = ev.started_at ? new Date(ev.started_at * 1000).toLocaleString() : "";

    const videoHtml = ev.clip_url
      ? `<video controls autoplay muted style="width:100%;max-height:480px;background:#05070a;border-radius:10px;" src="${ev.clip_url}"></video>`
      : `<div class="empty" style="height:220px;display:grid;place-content:center;">${ev.thumb_url ? `<img src="${ev.thumb_url}" style="max-height:180px;object-fit:contain;border-radius:8px;" alt="Event still">` : ""}<strong style="margin-top:10px;">${ev.status === "finalized" ? "Not recorded" : "Clip processing"}</strong>${ev.status === "finalized" ? "This camera is live-inference only — no video is kept." : "The video will appear once finalized."}</div>`;

    const behaviors = (ev.behaviors || []).map((b) => `<span class="chip">${escapeHtml(b)}</span>`).join("");

    let facesHtml = "";
    if (ev.faces && ev.faces.length > 0) {
      facesHtml = `
        <div style="margin-top:18px;">
          <div class="eyebrow" style="margin-bottom:8px;">Faces · ${ev.faces.length}</div>
          <div style="display:flex;gap:10px;flex-wrap:wrap;">
            ${ev.faces.map((f) => `
              <figure style="margin:0;display:flex;flex-direction:column;gap:5px;align-items:center;">
                <img src="${f.crop_url || f.context_url}" alt="Face crop" style="width:76px;height:76px;object-fit:cover;border-radius:8px;border:1px solid var(--line);">
                <figcaption style="font-size:0.72rem;color:var(--text-dim);text-align:center;">
                  ${escapeHtml(f.identity_name || `cluster ${f.cluster_id ?? "?"}`)}<br>
                  <span class="mono" style="color:var(--text-faint);">${Math.round(f.face_width || 0)}px</span>
                </figcaption>
              </figure>`).join("")}
          </div>
        </div>`;
    }

    body.innerHTML = `
      <div class="eyebrow">${escapeHtml(ev.camera || "")} · ${dateStr} · ${escapeHtml(ev.status || "")}</div>
      <h2 style="margin:4px 0 12px;">${escapeHtml(ev.primary_class || "event")} <span class="mono" style="color:var(--text-faint);font-size:0.9rem;">#${ev.id}</span></h2>
      ${videoHtml}
      ${behaviors ? `<div class="chips" style="margin-top:14px;">${behaviors}</div>` : ""}
      ${facesHtml}`;
  } catch (err) {
    body.innerHTML = `<div class="empty"><strong>Failed to load event</strong>${escapeHtml(String(err))}</div>`;
  }
}

function closeModal() {
  const modal = document.getElementById("modal");
  document.getElementById("modal-body").innerHTML = "";
  modal.classList.add("hidden");
  if (window.location.hash.startsWith("#/events/")) window.location.hash = "#/events";
}

// ============================================================================
// FACES VIEW
// ============================================================================
async function loadFacesView() {
  await fetchIdentities();
  await fetchClusters();
}

async function fetchIdentities() {
  try {
    const res = await fetch("/api/identities");
    const identities = await res.json();
    renderIdentities(identities);

    const dl = document.getElementById("identities-datalist");
    dl.innerHTML = identities.map((i) => `<option value="${escapeHtml(i.name)}"></option>`).join("");
  } catch (err) {
    console.error("fetch identities error:", err);
  }
}

function renderIdentities(items) {
  const grid = document.getElementById("identities-grid");
  grid.innerHTML = "";

  if (items.length === 0) {
    grid.innerHTML = `<div class="empty" style="grid-column: 1/-1;"><strong>No identities yet</strong>Create one, or label an unknown cluster from the list below.</div>`;
    return;
  }

  items.forEach((ident) => {
    const card = document.createElement("div");
    card.className = "face-card";
    const samples = (ident.sample_urls || [])
      .map((url) => `<img src="${url}" class="face-sample-img" alt="${escapeHtml(ident.name)} sample">`)
      .join("");

    card.innerHTML = `
      <div class="face-card-header">
        <strong style="font-size:0.98rem;">${escapeHtml(ident.name)}</strong>
        <span class="chip">${ident.face_count} faces</span>
      </div>
      <div class="face-samples">${samples || '<span style="color:var(--text-faint);font-size:0.8rem;">No photo samples</span>'}</div>
      <div style="display:flex;gap:6px;margin-top:auto;">
        <button class="btn btn-ghost btn-xs btn-rename">Rename</button>
        <button class="btn btn-ghost btn-xs btn-upload">Upload photo</button>
        <button class="btn btn-danger btn-xs btn-delete" style="margin-left:auto;">Delete</button>
      </div>`;

    card.querySelector(".btn-rename").onclick = () => renameIdentity(ident.id, ident.name);
    card.querySelector(".btn-delete").onclick = () => deleteIdentity(ident.id);
    card.querySelector(".btn-upload").onclick = () => uploadIdentityPhoto(ident.id);

    grid.appendChild(card);
  });
}

async function renameIdentity(id, currentName) {
  const newName = prompt("New name for this identity:", currentName);
  if (!newName || newName.trim() === currentName) return;
  try {
    const res = await fetch(`/api/identities/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: newName.trim() }),
    });
    if (res.ok) loadFacesView();
    else {
      const err = await res.json();
      showToast("Rename failed", err.detail || "Error");
    }
  } catch (err) {
    showToast("Rename failed", String(err));
  }
}

async function deleteIdentity(id) {
  if (!confirm("Delete this identity? Its faces become one unknown cluster.")) return;
  try {
    const res = await fetch(`/api/identities/${id}`, { method: "DELETE" });
    if (res.ok) loadFacesView();
    else showToast("Delete failed", "Identity could not be removed");
  } catch (err) {
    showToast("Delete failed", String(err));
  }
}

function uploadIdentityPhoto(id) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.onchange = async () => {
    if (!input.files || input.files.length === 0) return;
    const formData = new FormData();
    formData.append("file", input.files[0]);
    try {
      const res = await fetch(`/api/identities/${id}/photos`, { method: "POST", body: formData });
      if (res.ok) {
        showToast("Photo added", "Face embedded into this identity.");
        loadFacesView();
      } else {
        const err = await res.json();
        showToast("Photo rejected", err.detail || "No face detected");
      }
    } catch (err) {
      showToast("Upload error", String(err));
    }
  };
  input.click();
}

async function fetchClusters() {
  try {
    const res = await fetch("/api/faces/clusters");
    renderClusters(await res.json());
  } catch (err) {
    console.error("fetch clusters error:", err);
  }
}

function renderClusters(items) {
  const grid = document.getElementById("clusters-grid");
  grid.innerHTML = "";

  if (items.length === 0) {
    grid.innerHTML = `<div class="empty" style="grid-column: 1/-1;"><strong>No unknown clusters</strong>Unrecognised faces will accumulate here for labelling.</div>`;
    return;
  }

  items.forEach((c) => {
    const card = document.createElement("div");
    card.className = "face-card";
    const dateStr = c.last_seen ? new Date(c.last_seen * 1000).toLocaleString() : "";
    const samples = (c.sample_urls || [])
      .map((url) => `<img src="${url}" class="face-sample-img" alt="Cluster ${c.cluster_id} sample">`)
      .join("");

    card.innerHTML = `
      <div class="face-card-header">
        <strong style="font-size:0.95rem;">Cluster #${c.cluster_id}</strong>
        <span class="chip">${c.count} sightings</span>
      </div>
      <div style="font-size:0.74rem;color:var(--text-faint);">Last seen ${dateStr}</div>
      <div class="face-samples">${samples}</div>
      <div style="display:flex;gap:6px;margin-top:auto;">
        <input type="text" class="form-control cluster-name-input" list="identities-datalist" placeholder="Assign a name…" style="flex:1;">
        <button class="btn btn-primary btn-xs btn-assign">Assign</button>
      </div>`;

    card.querySelector(".btn-assign").onclick = async () => {
      const name = card.querySelector(".cluster-name-input").value.trim();
      if (!name) { showToast("Name required", "Type or pick an identity first."); return; }
      try {
        const res = await fetch(`/api/faces/clusters/${c.cluster_id}/assign`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name }),
        });
        if (res.ok) {
          showToast("Cluster labelled", `Assigned to ${name}.`);
          loadFacesView();
        } else showToast("Assign failed", "Cluster could not be labelled");
      } catch (err) {
        showToast("Assign error", String(err));
      }
    };

    grid.appendChild(card);
  });
}

// ============================================================================
// SCENERY VIEW
// ============================================================================
let currentSceneryCam = "";
let currentScenerySlots = [];
let drawingSlot = false;
let drawStartX = 0;
let drawStartY = 0;

async function loadSceneryView() {
  await fetchVehicles();          // vehicles first: slot cards need them for linking
  await populateSceneryCameras();
  initSceneryCanvas();
}

let liveVehicles = [];

async function fetchVehicles() {
  try {
    const res = await fetch("/api/vehicles");
    liveVehicles = await res.json();
    renderVehicles(liveVehicles);
    syncVehicleDatalist(liveVehicles);
  } catch (err) {
    console.error("fetch vehicles error:", err);
  }
}

function syncVehicleDatalist(vehicles) {
  let dl = document.getElementById("vehicles-datalist");
  if (!dl) {
    dl = document.createElement("datalist");
    dl.id = "vehicles-datalist";
    document.body.appendChild(dl);
  }
  dl.innerHTML = vehicles.map((v) => `<option value="${escapeHtml(v.name)}"></option>`).join("");
  const nameInput = document.getElementById("slot-name-input");
  if (nameInput) {
    nameInput.setAttribute("list", "vehicles-datalist");
    nameInput.placeholder = "pick an existing vehicle or type a new name";
  }
}

function timeAgo(ts) {
  if (!ts) return "—";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return `${Math.round(s)}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

function renderVehicles(vehicles) {
  const host = document.getElementById("vehicles-list");
  if (!host) return;
  host.innerHTML = "";

  if (vehicles.length === 0) {
    host.innerHTML = `<div class="empty"><strong>No vehicles yet</strong>Create a slot to register a vehicle.</div>`;
    return;
  }

  vehicles.forEach((v) => {
    const row = document.createElement("div");
    row.className = "feed-item";

    const camChips = (v.cameras || [])
      .map((c) => `<span class="chip chip-accent">${escapeHtml(c)}</span>`)
      .join(" ") || `<span class="chip">no camera yet</span>`;

    const slotLines = (v.slots || []).map((s) => {
      const when = (v.sightings || []).find((x) => x.camera === s.camera);
      const seen = when ? `${when.hits} hits · ${timeAgo(when.last_seen)}` : "not seen yet";
      return `<div class="feed-meta"><span>${escapeHtml(s.camera)}</span><span class="mono">${seen}</span></div>`;
    }).join("");

    row.innerHTML = `
      <div class="feed-meta">
        <span class="feed-cam">${escapeHtml(v.name)}</span>
        <span class="mono">${escapeHtml(v.color_name || "")}</span>
      </div>
      <div class="feed-chips" style="margin-bottom:6px;">${camChips}</div>
      ${slotLines}
      <div style="display:flex;gap:6px;margin-top:8px;">
        <button class="btn btn-ghost btn-xs btn-veh-rename">Rename</button>
        <button class="btn btn-danger btn-xs btn-veh-delete" style="margin-left:auto;">Delete</button>
      </div>`;

    row.querySelector(".btn-veh-rename").onclick = async () => {
      const name = prompt("Vehicle name (slots on every camera follow this name):", v.name);
      if (!name || !name.trim() || name.trim() === v.name) return;
      try {
        const res = await fetch(`/api/vehicles/${v.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name.trim() }),
        });
        if (res.ok) { showToast("Vehicle renamed", name.trim()); fetchVehicles(); fetchScenerySlots(); }
        else showToast("Rename failed", "Name may already exist");
      } catch (err) { showToast("Rename error", String(err)); }
    };

    row.querySelector(".btn-veh-delete").onclick = async () => {
      if (!confirm(`Delete vehicle "${v.name}"? Its slots stay but become unlinked.`)) return;
      try {
        const res = await fetch(`/api/vehicles/${v.id}`, { method: "DELETE" });
        if (res.ok) { showToast("Vehicle deleted", v.name); fetchVehicles(); fetchScenerySlots(); }
      } catch (err) { showToast("Delete error", String(err)); }
    };

    host.appendChild(row);
  });
}

async function populateSceneryCameras() {
  try {
    const res = await fetch("/api/live/status");
    const data = await res.json();
    const cameras = data.cameras || {};
    const select = document.getElementById("scenery-camera-select");
    select.innerHTML = "";

    const camNames = Object.keys(cameras);
    if (camNames.length === 0) {
      select.innerHTML = "<option value=''>No active cameras</option>";
      return;
    }

    camNames.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    });

    if (!currentSceneryCam || !cameras[currentSceneryCam]) currentSceneryCam = camNames[0];
    select.value = currentSceneryCam;

    select.onchange = () => {
      currentSceneryCam = select.value;
      updateSceneryView();
    };

    updateSceneryView();
  } catch (err) {
    console.error("scenery cameras error:", err);
  }
}

async function updateSceneryView() {
  if (!currentSceneryCam) return;
  document.getElementById("scenery-cam-title").textContent = currentSceneryCam;

  const slug = currentSceneryCam.replace(/[^A-Za-z0-9_-]+/g, "_");
  const img = document.getElementById("scenery-snap-img");
  img.src = `/api/live/${encodeURIComponent(slug)}/snapshot.jpg?t=${Date.now()}`;
  img.onload = () => drawSceneryOverlay();

  await fetchScenerySlots();
}

async function fetchScenerySlots() {
  if (!currentSceneryCam) return;
  try {
    const res = await fetch(`/api/scenery/slots?camera=${encodeURIComponent(currentSceneryCam)}`);
    currentScenerySlots = await res.json();
    renderScenerySlots(currentScenerySlots);
    drawSceneryOverlay();
  } catch (err) {
    console.error("fetch slots error:", err);
  }
}

function renderScenerySlots(slots) {
  const host = document.getElementById("scenery-slots-grid");
  host.innerHTML = "";

  if (slots.length === 0) {
    host.innerHTML = `<div class="empty"><strong>No slots for ${escapeHtml(currentSceneryCam)}</strong>Drag a box over a parked car to anchor it.</div>`;
    return;
  }

  slots.forEach((s) => {
    const row = document.createElement("div");
    row.className = "feed-item";
    const boxStr = Array.isArray(s.slot_box) ? s.slot_box.map((v) => (typeof v === "number" ? v.toFixed(2) : v)).join(", ") : "";

    const options = [
      `<option value="">— unlinked —</option>`,
      ...liveVehicles.map((v) =>
        `<option value="${v.id}" ${s.vehicle_id === v.id ? "selected" : ""}>${escapeHtml(v.name)}</option>`
      ),
    ].join("");

    row.innerHTML = `
      <div class="feed-meta">
        <span class="feed-cam">${escapeHtml(s.name)}</span>
        <span class="chip ${s.is_friendly ? "chip-ok" : "chip-warn"}">${s.is_friendly ? "friendly" : "alert"}</span>
      </div>
      <div class="feed-chips" style="margin-bottom:6px;">
        <span class="chip">${escapeHtml(s.color_name || "unknown")}</span>
        <span class="chip mono">[${boxStr}]</span>
      </div>
      <div class="field" style="margin-bottom:6px;">
        <label>Vehicle (cross-camera)</label>
        <select class="form-control js-vehicle-link" style="width:100%;">${options}</select>
      </div>
      <div style="display:flex;gap:6px;">
        <button class="btn btn-ghost btn-xs btn-toggle-friendly">${s.is_friendly ? "Mark alert" : "Mark friendly"}</button>
        <button class="btn btn-danger btn-xs btn-delete-slot" style="margin-left:auto;">Delete</button>
      </div>`;

    row.querySelector(".js-vehicle-link").onchange = async (e) => {
      const val = e.target.value;
      const vehicle_id = val === "" ? null : parseInt(val, 10);
      try {
        const res = await fetch(`/api/scenery/slots/${s.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ vehicle_id }),
        });
        if (res.ok) {
          showToast("Slot linked", vehicle_id === null ? "Unlinked" : "Vehicle updated");
          fetchScenerySlots();
          fetchVehicles();
        } else {
          showToast("Link failed", "Vehicle not found");
        }
      } catch (err) {
        showToast("Link error", String(err));
      }
    };

    row.querySelector(".btn-toggle-friendly").onclick = async () => {
      try {
        const res = await fetch(`/api/scenery/slots/${s.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ is_friendly: !s.is_friendly }),
        });
        if (res.ok) fetchScenerySlots();
      } catch (err) {
        showToast("Update failed", String(err));
      }
    };

    row.querySelector(".btn-delete-slot").onclick = async () => {
      if (!confirm(`Delete slot "${s.name}"?`)) return;
      try {
        const res = await fetch(`/api/scenery/slots/${s.id}`, { method: "DELETE" });
        if (res.ok) {
          showToast("Slot removed", s.name);
          fetchScenerySlots();
        }
      } catch (err) {
        showToast("Delete error", String(err));
      }
    };

    host.appendChild(row);
  });
}

function initSceneryCanvas() {
  const canvas = document.getElementById("scenery-overlay-canvas");
  const container = document.querySelector(".snap-frame");
  if (!canvas || !container) return;

  const resize = () => {
    if (container.clientWidth > 0) {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      drawSceneryOverlay();
    }
  };
  window.addEventListener("resize", resize);
  setTimeout(resize, 80);

  let startX = 0;
  let startY = 0;

  canvas.onmousedown = (e) => {
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    drawingSlot = true;
  };

  canvas.onmousemove = (e) => {
    if (!drawingSlot) return;
    const rect = canvas.getBoundingClientRect();
    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;
    drawSceneryOverlay();
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = "#5bc8e8";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(Math.min(startX, curX), Math.min(startY, curY), Math.abs(curX - startX), Math.abs(curY - startY));
    ctx.setLineDash([]);
  };

  canvas.onmouseup = (e) => {
    if (!drawingSlot) return;
    drawingSlot = false;
    const rect = canvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;

    const x1 = Math.min(startX, endX) / canvas.width;
    const y1 = Math.min(startY, endY) / canvas.height;
    const x2 = Math.max(startX, endX) / canvas.width;
    const y2 = Math.max(startY, endY) / canvas.height;

    if (x2 - x1 > 0.02 && y2 - y1 > 0.02) {
      document.getElementById("slot-x1").value = x1.toFixed(3);
      document.getElementById("slot-y1").value = y1.toFixed(3);
      document.getElementById("slot-x2").value = x2.toFixed(3);
      document.getElementById("slot-y2").value = y2.toFixed(3);
    }
    drawSceneryOverlay();
  };

  const refreshBtn = document.getElementById("btn-refresh-scenery-snap");
  if (refreshBtn) refreshBtn.onclick = () => updateSceneryView();

  const saveBtn = document.getElementById("btn-save-slot");
  if (saveBtn) {
    saveBtn.onclick = async () => {
      const name = document.getElementById("slot-name-input").value.trim();
      const x1 = parseFloat(document.getElementById("slot-x1").value);
      const y1 = parseFloat(document.getElementById("slot-y1").value);
      const x2 = parseFloat(document.getElementById("slot-x2").value);
      const y2 = parseFloat(document.getElementById("slot-y2").value);
      const isFriendly = document.getElementById("slot-friendly-check").checked;

      if (!name) { showToast("Name required", "Give the slot a vehicle name."); return; }
      if ([x1, y1, x2, y2].some((v) => isNaN(v))) {
        showToast("Box required", "Drag on the snapshot to define the slot area.");
        return;
      }

      try {
        const res = await fetch("/api/scenery/slots", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ camera: currentSceneryCam, name, slot_box: [x1, y1, x2, y2], is_friendly: isFriendly }),
        });
        if (res.ok) {
          showToast("Slot saved", name);
          document.getElementById("slot-name-input").value = "";
          fetchScenerySlots();
          fetchVehicles();
        } else {
          const err = await res.json();
          showToast("Save failed", err.detail || "Error");
        }
      } catch (err) {
        showToast("Save error", String(err));
      }
    };
  }
}

function drawSceneryOverlay() {
  const canvas = document.getElementById("scenery-overlay-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  currentScenerySlots.forEach((slot) => {
    if (!Array.isArray(slot.slot_box) || slot.slot_box.length !== 4) return;
    const x1 = slot.slot_box[0] * canvas.width;
    const y1 = slot.slot_box[1] * canvas.height;
    const w = (slot.slot_box[2] - slot.slot_box[0]) * canvas.width;
    const h = (slot.slot_box[3] - slot.slot_box[1]) * canvas.height;

    ctx.strokeStyle = slot.is_friendly ? "rgba(227,165,69,0.85)" : "rgba(229,96,79,0.85)";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(x1, y1, w, h);
    ctx.setLineDash([]);

    ctx.font = '600 11px "IBM Plex Mono", monospace';
    const tw = ctx.measureText(slot.name).width;
    ctx.fillStyle = "rgba(20,18,12,0.85)";
    ctx.fillRect(x1, Math.max(0, y1 - 17), tw + 10, 16);
    ctx.fillStyle = "rgba(240,205,140,0.95)";
    ctx.fillText(slot.name, x1 + 5, Math.max(11, y1 - 5));
  });
}

// ============================================================================
// ARCHIVE VIEW
// ============================================================================
let archiveOffset = 0;
const ARCHIVE_LIMIT = 30;

async function loadArchiveView() {
  await fetchArchiveSummary();
  await fetchArchiveList();
}

async function fetchArchiveSummary() {
  try {
    const res = await fetch("/api/archive/summary");
    if (!res.ok) return;
    const s = await res.json();
    document.getElementById("archive-summary").innerHTML = `
      <div class="summary-card"><div class="summary-value">${(s.total_clips || 0).toLocaleString()}</div><div class="summary-label">Clips scanned</div></div>
      <div class="summary-card"><div class="summary-value" style="color:var(--ok)">${(s.kept_count || 0).toLocaleString()}</div><div class="summary-label">Kept events</div></div>
      <div class="summary-card"><div class="summary-value" style="color:var(--text-dim)">${(s.discarded_count || 0).toLocaleString()}</div><div class="summary-label">Suppressed</div></div>
      <div class="summary-card"><div class="summary-value" style="color:var(--accent)">${s.reduction_percentage || 0}%</div><div class="summary-label">Noise reduction</div></div>`;

    fillArchiveSelect("archive-filter-camera", s.cameras, "All cameras");
    fillArchiveSelect("archive-filter-class", s.classes, "All events");
    const from = document.getElementById("archive-filter-date-from");
    const to = document.getElementById("archive-filter-date-to");
    if (from && s.date_min) { from.min = s.date_min.slice(0, 10); from.max = (s.date_max || "").slice(0, 10) || ""; }
    if (to && s.date_max) { to.min = (s.date_min || "").slice(0, 10) || ""; to.max = s.date_max.slice(0, 10); }
  } catch (err) {
    console.error("archive summary error:", err);
  }
}

function fillArchiveSelect(id, counts, allLabel) {
  const sel = document.getElementById(id);
  if (!sel) return;
  const previous = sel.value;
  sel.innerHTML = `<option value="">${allLabel}</option>`;
  Object.keys(counts || {}).forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = `${name} (${Number(counts[name] || 0).toLocaleString()})`;
    sel.appendChild(opt);
  });
  if (previous) sel.value = previous;
}

function formatArchiveTime(item) {
  if (item.datetime) return item.datetime.replace("T", " ");
  const raw = item.timestamp;
  if (raw && String(raw).length >= 14) {
    const t = String(raw);
    return `${t.slice(0, 4)}-${t.slice(4, 6)}-${t.slice(6, 8)} ${t.slice(8, 10)}:${t.slice(10, 12)}:${t.slice(12, 14)}`;
  }
  return item.filename || "";
}

const ARCHIVE_VEHICLE_CLASSES = new Set(["car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "tractor"]);

function archiveClassChips(item) {
  const counts = item.class_counts || {};
  return Object.keys(counts)
    .map((name) => {
      const chip = ARCHIVE_VEHICLE_CLASSES.has(name) ? "chip chip-vehicle" : name === "person" ? "chip chip-person" : "chip";
      const n = Number(counts[name] || 0);
      return `<span class="${chip}">${escapeHtml(name)}${n > 1 ? ` &times;${n}` : ""}</span>`;
    })
    .join("");
}

async function fetchArchiveList(offset = 0) {
  archiveOffset = offset;
  const verdict = document.getElementById("archive-filter-verdict").value;
  const camera = document.getElementById("archive-filter-camera").value;
  const className = document.getElementById("archive-filter-class").value;
  const dateFrom = document.getElementById("archive-filter-date-from").value;
  const dateTo = document.getElementById("archive-filter-date-to").value;
  const reason = document.getElementById("archive-filter-reason").value;
  const sort = document.getElementById("archive-filter-sort").value;

  const params = new URLSearchParams({ offset: offset.toString(), limit: ARCHIVE_LIMIT.toString() });
  if (verdict) params.set("verdict", verdict);
  if (camera) params.set("camera", camera);
  if (className) params.set("class_name", className);
  if (dateFrom) params.set("date_from", dateFrom);
  if (dateTo) params.set("date_to", dateTo);
  if (reason) params.set("reason", reason);
  if (sort) params.set("sort", sort);

  try {
    const res = await fetch(`/api/archive?${params.toString()}`);
    if (!res.ok) return;
    const data = await res.json();
    renderArchiveGrid(data.items || []);
    renderPagination("archive-pagination", data.total || 0, offset, ARCHIVE_LIMIT, fetchArchiveList);
  } catch (err) {
    console.error("fetch archive error:", err);
  }
}

function renderArchiveGrid(items) {
  const grid = document.getElementById("archive-grid");
  grid.innerHTML = "";

  if (items.length === 0) {
    grid.innerHTML = `<div class="empty" style="grid-column:1/-1;"><strong>Nothing here</strong>Adjust the filters to widen the search.</div>`;
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "archive-card";
    const isKeep = item.verdict === "KEEP";

    card.innerHTML = `
      <div class="card-thumb ${item.thumb_url ? "" : "no-thumb"}">
        ${item.thumb_url ? `<img src="${item.thumb_url}" alt="Clip ${item.idx} thumbnail" loading="lazy">` : ""}
      </div>
      <div class="card-body">
        <div class="card-meta">
          <span class="card-cam">${escapeHtml(item.camera || "unknown")}</span>
          <span class="${isKeep ? "verdict-keep" : "verdict-discard"}">${escapeHtml(item.verdict || "")}</span>
        </div>
        <div class="mono archive-card-time">${escapeHtml(formatArchiveTime(item))}</div>
        <div class="chip-row">${archiveClassChips(item)}</div>
        <div style="font-size:0.86rem;font-weight:600;">${escapeHtml(item.primary_reason || item.reason || "")}</div>
        <div class="card-meta"><span>confidence</span><span class="mono">${Math.round((item.confidence || 0) * 100)}%</span></div>
      </div>`;

    card.onclick = () => openArchiveModal(item);
    grid.appendChild(card);
  });
}

function openArchiveModal(item) {
  const modal = document.getElementById("modal");
  const body = document.getElementById("modal-body");

  const videoHtml = item.clip_url
    ? `<video controls autoplay muted style="width:100%;max-height:480px;background:#05070a;border-radius:10px;" src="${item.clip_url}"></video>`
    : `<div class="empty" style="height:200px;display:grid;place-content:center;"><strong>Clip unavailable</strong>The file is not accessible from this server.</div>`;

  body.innerHTML = `
    <div class="eyebrow">${escapeHtml(item.camera || "archive")} · ${escapeHtml(item.verdict || "")}</div>
    <h2 style="margin:4px 0 12px;">${escapeHtml(item.primary_reason || item.reason || "clip")}</h2>
    <div class="modal-meta-row">
      <span class="mono">${escapeHtml(formatArchiveTime(item))}</span>
      <span class="chip-row">${archiveClassChips(item)}</span>
    </div>
    ${videoHtml}`;
  modal.classList.remove("hidden");
}

// ============================================================================
// Pagination
// ============================================================================
function renderPagination(containerId, total, currentOffset, limit, onPage) {
  const bar = document.getElementById(containerId);
  bar.innerHTML = "";
  if (total <= limit) return;

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(currentOffset / limit) + 1;

  if (currentPage > 1) {
    const prev = document.createElement("button");
    prev.className = "btn btn-ghost btn-sm";
    prev.textContent = "← Newer";
    prev.onclick = () => onPage((currentPage - 2) * limit);
    bar.appendChild(prev);
  }

  const info = document.createElement("span");
  info.className = "mono";
  info.textContent = `${currentPage} / ${totalPages} · ${total} total`;
  bar.appendChild(info);

  if (currentPage < totalPages) {
    const next = document.createElement("button");
    next.className = "btn btn-ghost btn-sm";
    next.textContent = "Older →";
    next.onclick = () => onPage(currentPage * limit);
    bar.appendChild(next);
  }
}

// ============================================================================
// Helpers
// ============================================================================
function escapeHtml(text) {
  if (text === null || text === undefined) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ============================================================================
// Boot
// ============================================================================
document.addEventListener("DOMContentLoaded", () => {
  initRouter();
  initSSE();
  initClock();

  document.getElementById("modal-close").onclick = closeModal;
  document.getElementById("modal-backdrop").onclick = closeModal;

  document.getElementById("btn-browser-notifications").onclick = async () => {
    if ("Notification" in window) {
      const perm = await Notification.requestPermission();
      if (perm === "granted") showToast("Alerts enabled", "Desktop notifications are on.");
    }
  };

  // Overlay toggle
  const toggleBoxesBtn = document.getElementById("btn-toggle-client-boxes");
  toggleBoxesBtn.onclick = () => {
    showClientOverlays = !showClientOverlays;
    toggleBoxesBtn.textContent = `Boxes ${showClientOverlays ? "ON" : "OFF"}`;
    drawAllOverlays();
  };

  // Vehicles refresh (scenery view)
  const refreshVehiclesBtn = document.getElementById("btn-refresh-vehicles");
  if (refreshVehiclesBtn) {
    refreshVehiclesBtn.onclick = () => { fetchVehicles(); fetchScenerySlots(); };
  }

  // Event filters
  document.getElementById("event-filter-apply").onclick = () => fetchEvents(0);
  document.getElementById("event-filter-reset").onclick = () => {
    document.getElementById("event-filter-camera").value = "";
    document.getElementById("event-filter-behavior").value = "";
    document.getElementById("event-filter-identity").value = "";
    document.getElementById("event-filter-date").value = "";
    fetchEvents(0);
  };

  // Archive filters
  document.getElementById("archive-filter-apply").onclick = () => fetchArchiveList(0);
  ["archive-filter-verdict", "archive-filter-camera", "archive-filter-class", "archive-filter-sort", "archive-filter-date-from", "archive-filter-date-to"].forEach((id) => {
    document.getElementById(id).onchange = () => fetchArchiveList(0);
  });
  document.getElementById("archive-filter-reason").onkeydown = (ev) => {
    if (ev.key === "Enter") fetchArchiveList(0);
  };
  document.getElementById("archive-filter-reset").onclick = () => {
    ["archive-filter-verdict", "archive-filter-camera", "archive-filter-class", "archive-filter-date-from", "archive-filter-date-to", "archive-filter-reason"].forEach((id) => {
      document.getElementById(id).value = "";
    });
    document.getElementById("archive-filter-sort").value = "date_desc";
    fetchArchiveList(0);
  };

  // New identity
  document.getElementById("btn-new-identity").onclick = async () => {
    const name = prompt("Name for the new identity:");
    if (!name || !name.trim()) return;
    try {
      const res = await fetch("/api/identities", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() }),
      });
      if (res.ok) {
        showToast("Identity created", name.trim());
        loadFacesView();
      } else {
        const err = await res.json();
        showToast("Create failed", err.detail || "Error");
      }
    } catch (err) {
      showToast("Create error", String(err));
    }
  };

  // Re-layout canvases on resize
  let rT;
  window.addEventListener("resize", () => {
    clearTimeout(rT);
    rT = setTimeout(drawAllOverlays, 120);
  });
});
