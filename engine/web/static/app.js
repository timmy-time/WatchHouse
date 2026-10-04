// CCTV Analytics Web Dashboard Application

let currentRoute = "";
let sseSource = null;
let statusInterval = null;
let showClientOverlays = true;

// Routing
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

  // Clear live status polling when navigating away from live
  if (statusInterval && !hash.startsWith("#/live")) {
    clearInterval(statusInterval);
    statusInterval = null;
  }

  // Update navigation links
  document.querySelectorAll(".nav-link").forEach((link) => {
    const route = link.getAttribute("href");
    if (hash.startsWith(route)) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });

  // Switch view containers
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

// Global SSE & Notifications
function initSSE() {
  if (sseSource) sseSource.close();
  sseSource = new EventSource("/api/stream");

  sseSource.addEventListener("notification", (e) => {
    try {
      const data = JSON.parse(e.data);
      showToast(data.title || "New Notification", data.body || "");
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
      if (currentRoute.startsWith("#/events")) {
        loadEventsView();
      }
    } catch (err) {
      console.error("SSE event parse error:", err);
    }
  });

  sseSource.onerror = (err) => {
    console.warn("SSE connection interrupted, retrying...", err);
  };
}

function showToast(title, body) {
  const container = document.getElementById("toasts-container");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `<strong>${escapeHtml(title)}</strong><div>${escapeHtml(body)}</div>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 5000);
}

// Live View
async function loadLiveView() {
  await fetchLiveStatus();
  if (!statusInterval) {
    statusInterval = setInterval(fetchLiveStatus, 5000);
  }
}

async function fetchLiveStatus() {
  try {
    const res = await fetch("/api/live/status");
    const data = await res.json();
    renderLiveStreams(data.cameras || {});
  } catch (err) {
    console.error("Error fetching live status:", err);
  }
}

function renderLiveStreams(cameras) {
  const grid = document.getElementById("live-streams-grid");
  const existingTiles = new Set();

  Object.entries(cameras).forEach(([name, cam]) => {
    existingTiles.add(cam.slug);
    let tile = document.getElementById(`cam-tile-${cam.slug}`);
    if (!tile) {
      tile = document.createElement("div");
      tile.id = `cam-tile-${cam.slug}`;
      tile.className = "camera-tile";
      tile.innerHTML = `
        <div class="camera-header">
          <span class="camera-name">${escapeHtml(name)}</span>
          <div class="camera-badges">
            <span class="badge badge-conn">...</span>
            <span class="badge badge-decode">${escapeHtml(cam.decode || "decode")}</span>
          </div>
        <div class="camera-media" style="position: relative;">
          <img src="/api/live/${encodeURIComponent(cam.slug)}/stream.mjpg" alt="${escapeHtml(name)}" onerror="this.src='/api/live/${encodeURIComponent(cam.slug)}/snapshot.jpg'">
          <canvas class="camera-live-overlay" style="position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>
        </div>
        <div class="camera-footer">
          <span class="cam-fps">FPS: ${cam.fps_analyzed || 0}</span>
          <span class="cam-restarts">Restarts: ${cam.restarts || 0}</span>
        </div>
      `;
      grid.appendChild(tile);
    }

    // Update status badges
    const badgeConn = tile.querySelector(".badge-conn");
    if (cam.connected) {
      badgeConn.className = "badge badge-conn badge-connected";
      badgeConn.textContent = "CONNECTED";
    } else {
      badgeConn.className = "badge badge-conn badge-disconnected";
      badgeConn.textContent = "DISCONNECTED";
    }

    const badgeDecode = tile.querySelector(".badge-decode");
    badgeDecode.textContent = (cam.decode || "cpu").toUpperCase();

    tile.querySelector(".cam-fps").textContent = `FPS: ${cam.fps_analyzed || 0}`;
    tile.querySelector(".cam-restarts").textContent = `Restarts: ${cam.restarts || 0}`;

    // Draw client-side overlays if available
    drawLiveCameraDetections(tile, cam.detections || []);
  });
}
function drawLiveCameraDetections(tile, detections) {
  const canvas = tile.querySelector(".camera-live-overlay");
  if (!canvas) return;
  const container = tile.querySelector(".camera-media");
  if (!container) return;

  if (canvas.width !== container.clientWidth || canvas.height !== container.clientHeight) {
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!showClientOverlays) return;

  detections.forEach((d) => {
    if (!Array.isArray(d.box_norm) || d.box_norm.length !== 4) return;
    const x1 = d.box_norm[0] * canvas.width;
    const y1 = d.box_norm[1] * canvas.height;
    const w = (d.box_norm[2] - d.box_norm[0]) * canvas.width;
    const h = (d.box_norm[3] - d.box_norm[1]) * canvas.height;

    let color = "#38bdf8"; // blue vehicle
    if (d.is_anchored) {
      color = "#eab308"; // gold anchored
    } else if (d.class_name === "person") {
      color = "#10b981"; // green person
    } else if (d.class_name === "car" || d.class_name === "truck" || d.class_name === "bus") {
      color = "#ef4444"; // red vehicle
    }

    // Bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, w, h);

    // High-contrast pill label
    const text = d.anchored_name ? `${d.anchored_name} #${d.track_id} ${d.conf}` : `${d.class_name} #${d.track_id} ${d.conf}`;
    ctx.font = "bold 12px sans-serif";
    const textWidth = ctx.measureText(text).width;
    const pad = 4;

    const pillX = Math.max(0, x1);
    const pillY = Math.max(0, y1 - 20);
    ctx.fillStyle = "rgba(18, 18, 18, 0.85)";
    ctx.fillRect(pillX, pillY, textWidth + pad * 2, 18);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(pillX, pillY, textWidth + pad * 2, 18);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, pillX + pad, pillY + 13);
  });
}


function addLiveFeedItem(ev) {
  const feed = document.getElementById("live-feed-list");
  if (!feed) return;

  const item = document.createElement("div");
  item.className = "feed-item";
  const dateStr = ev.started_at ? new Date(ev.started_at * 1000).toLocaleTimeString() : "";
  const behaviors = Array.isArray(ev.behaviors) ? ev.behaviors.join(", ") : "";

  item.innerHTML = `
    <div class="feed-item-header">
      <span>${escapeHtml(ev.camera || "")}</span>
      <span>${dateStr}</span>
    </div>
    <div class="feed-item-title">${escapeHtml(ev.primary_class || "event")}</div>
    <div style="font-size: 0.8rem; color: var(--muted);">${escapeHtml(behaviors)}</div>
  `;
  item.onclick = () => {
    window.location.hash = `#/events/${ev.id}`;
  };

  feed.prepend(item);
  // Keep only latest 30 items
  while (feed.children.length > 30) {
    feed.removeChild(feed.lastChild);
  }
}

// Events View
let eventsOffset = 0;
const EVENTS_LIMIT = 30;

async function loadEventsView() {
  populateEventFilters();
  await fetchEvents();
}

async function populateEventFilters() {
  // Populate cameras
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

    // Populate identities
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
    console.error("Filter populate error:", err);
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
    const endEpoch = startEpoch + 86400;
    params.set("since", startEpoch.toString());
    params.set("until", endEpoch.toString());
  }

  try {
    const res = await fetch(`/api/events?${params.toString()}`);
    const data = await res.json();
    renderEventsGrid(data.items || []);
    renderPagination("events-pagination", data.total || 0, offset, EVENTS_LIMIT, fetchEvents);
  } catch (err) {
    console.error("Fetch events error:", err);
  }
}

function renderEventsGrid(items) {
  const grid = document.getElementById("events-grid");
  grid.innerHTML = "";

  if (items.length === 0) {
    grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--muted); padding: 40px;">No events matching filters</div>`;
    return;
  }

  items.forEach((ev) => {
    const card = document.createElement("div");
    card.className = "event-card";
    const dateStr = ev.started_at ? new Date(ev.started_at * 1000).toLocaleString() : "";

    const chipsHtml = (ev.behaviors || [])
      .map((b) => `<span class="chip">${escapeHtml(b)}</span>`)
      .join("");

    const identityChips = (ev.identities || [])
      .map((idName) => `<span class="chip chip-identity">${escapeHtml(idName)}</span>`)
      .join("");

    const thumbSrc = ev.thumb_url || "/static/placeholder.jpg";

    card.innerHTML = `
      <div class="card-thumb">
        <img src="${thumbSrc}" alt="Event ${ev.id}" loading="lazy">
      </div>
      <div class="card-body">
        <div class="card-meta">
          <span>${escapeHtml(ev.camera || "")}</span>
          <span>${dateStr}</span>
        </div>
        <div style="font-weight: 600; font-size: 1rem; margin-bottom: 4px;">
          ${escapeHtml(ev.primary_class || "event").toUpperCase()}
          <span style="font-size: 0.8rem; color: var(--muted); font-weight: normal;">#${ev.id} (${ev.status})</span>
        </div>
        <div class="chips">${chipsHtml} ${identityChips}</div>
      </div>
    `;

    card.onclick = () => {
      window.location.hash = `#/events/${ev.id}`;
    };

    grid.appendChild(card);
  });
}

// Modal View
async function openEventModal(eventId) {
  const modal = document.getElementById("modal");
  const modalBody = document.getElementById("modal-body");
  modalBody.innerHTML = `<div style="text-align: center; padding: 40px;">Loading event ${eventId}...</div>`;
  modal.classList.remove("hidden");

  try {
    const res = await fetch(`/api/events/${eventId}`);
    if (!res.ok) {
      modalBody.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--accent-error);">Event ${eventId} not found</div>`;
      return;
    }
    const ev = await res.json();
    const dateStr = ev.started_at ? new Date(ev.started_at * 1000).toLocaleString() : "";

    let videoHtml = "";
    if (ev.clip_url) {
      videoHtml = `<video controls autoplay style="width: 100%; max-height: 480px; background: #000; border-radius: 6px;" src="${ev.clip_url}"></video>`;
    } else {
      videoHtml = `
        <div style="width: 100%; height: 260px; background: #000; display: flex; align-items: center; justify-content: center; flex-direction: column; gap: 8px; border-radius: 6px;">
          ${ev.thumb_url ? `<img src="${ev.thumb_url}" style="max-height: 200px; object-fit: contain;">` : ""}
          <div style="color: var(--muted); font-size: 0.9rem;">Clip processing or finalized...</div>
        </div>
      `;
    }

    let facesHtml = "";
    if (ev.faces && ev.faces.length > 0) {
      facesHtml = `
        <div style="margin-top: 20px;">
          <h4 style="margin-bottom: 10px;">Detected Faces (${ev.faces.length})</h4>
          <div style="display: flex; gap: 12px; flex-wrap: wrap;">
            ${ev.faces
              .map(
                (f) => `
              <div style="display: flex; flex-direction: column; align-items: center; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px; gap: 6px;">
                <img src="${f.crop_url || f.context_url}" style="width: 80px; height: 80px; object-fit: cover; border-radius: 4px;">
                <span style="font-size: 0.8rem; font-weight: 600;">${escapeHtml(f.identity_name || `Cluster #${f.cluster_id || "unknown"}`)}</span>
                <span style="font-size: 0.7rem; color: var(--muted);">Width: ${Math.round(f.face_width || 0)}px</span>
              </div>
            `
              )
              .join("")}
          </div>
        </div>
      `;
    }

    modalBody.innerHTML = `
      <h3 style="margin-bottom: 8px;">${escapeHtml(ev.camera)} - ${escapeHtml(ev.primary_class).toUpperCase()} #${ev.id}</h3>
      <div style="color: var(--muted); font-size: 0.85rem; margin-bottom: 14px;">${dateStr} &bull; Status: ${escapeHtml(ev.status)}</div>
      ${videoHtml}
      <div style="margin-top: 14px;">
        <strong>Behaviors:</strong> ${(ev.behaviors || []).join(", ") || "None"}
      </div>
      ${facesHtml}
    `;
  } catch (err) {
    modalBody.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--accent-error);">Error loading event: ${err}</div>`;
  }
}

function closeModal() {
  const modal = document.getElementById("modal");
  const modalBody = document.getElementById("modal-body");
  modal.classList.add("hidden");
  modalBody.innerHTML = "";
  if (window.location.hash.startsWith("#/events/")) {
    window.location.hash = "#/events";
  }
}

// Faces & Identities View
async function loadFacesView() {
  await fetchIdentities();
  await fetchClusters();
}

async function fetchIdentities() {
  try {
    const res = await fetch("/api/identities");
    const identities = await res.json();
    renderIdentities(identities);

    // Update datalist
    const dl = document.getElementById("identities-datalist");
    dl.innerHTML = identities.map((i) => `<option value="${escapeHtml(i.name)}"></option>`).join("");
  } catch (err) {
    console.error("Fetch identities error:", err);
  }
}

function renderIdentities(items) {
  const grid = document.getElementById("identities-grid");
  grid.innerHTML = "";

  if (items.length === 0) {
    grid.innerHTML = `<div style="grid-column: 1 / -1; color: var(--muted);">No registered identities yet. Create one or assign an unknown cluster!</div>`;
    return;
  }

  items.forEach((ident) => {
    const card = document.createElement("div");
    card.className = "face-card";

    const samplesHtml = (ident.sample_urls || [])
      .map((url) => `<img src="${url}" class="face-sample-img" alt="${ident.name}">`)
      .join("");

    card.innerHTML = `
      <div class="face-card-header">
        <strong style="font-size: 1.05rem;">${escapeHtml(ident.name)}</strong>
        <span class="badge" style="background: var(--bg);">${ident.face_count} faces</span>
      </div>
      <div class="face-samples">${samplesHtml || '<span style="color: var(--muted); font-size: 0.8rem;">No photo samples</span>'}</div>
      <div style="display: flex; gap: 6px; margin-top: auto;">
        <button class="btn btn-outline btn-sm btn-rename">Rename</button>
        <button class="btn btn-outline btn-sm btn-upload">Upload Photo</button>
        <button class="btn btn-danger btn-sm btn-delete" style="margin-left: auto;">Delete</button>
      </div>
    `;

    card.querySelector(".btn-rename").onclick = () => renameIdentity(ident.id, ident.name);
    card.querySelector(".btn-delete").onclick = () => deleteIdentity(ident.id);
    card.querySelector(".btn-upload").onclick = () => uploadIdentityPhoto(ident.id);

    grid.appendChild(card);
  });
}

async function renameIdentity(id, currentName) {
  const newName = prompt("Enter new name for identity:", currentName);
  if (!newName || newName.trim() === currentName) return;

  try {
    const res = await fetch(`/api/identities/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: newName.trim() }),
    });
    if (res.ok) {
      loadFacesView();
    } else {
      const err = await res.json();
      alert(`Rename failed: ${err.detail || "Error"}`);
    }
  } catch (err) {
    alert(`Rename failed: ${err}`);
  }
}

async function deleteIdentity(id) {
  if (!confirm("Are you sure you want to delete this identity? Its faces will become an unknown cluster.")) return;

  try {
    const res = await fetch(`/api/identities/${id}`, { method: "DELETE" });
    if (res.ok) {
      loadFacesView();
    } else {
      alert("Failed to delete identity");
    }
  } catch (err) {
    alert(`Delete failed: ${err}`);
  }
}

function uploadIdentityPhoto(id) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.onchange = async () => {
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`/api/identities/${id}/photos`, {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        showToast("Success", "Photo uploaded and face embedded!");
        loadFacesView();
      } else {
        const err = await res.json();
        alert(`Photo upload failed: ${err.detail || "No face detected"}`);
      }
    } catch (err) {
      alert(`Upload error: ${err}`);
    }
  };
  input.click();
}

async function fetchClusters() {
  try {
    const res = await fetch("/api/faces/clusters");
    const clusters = await res.json();
    renderClusters(clusters);
  } catch (err) {
    console.error("Fetch clusters error:", err);
  }
}

function renderClusters(items) {
  const grid = document.getElementById("clusters-grid");
  grid.innerHTML = "";

  if (items.length === 0) {
    grid.innerHTML = `<div style="grid-column: 1 / -1; color: var(--muted);">No unknown face clusters detected.</div>`;
    return;
  }

  items.forEach((c) => {
    const card = document.createElement("div");
    card.className = "face-card";
    const dateStr = c.last_seen ? new Date(c.last_seen * 1000).toLocaleString() : "";

    const samplesHtml = (c.sample_urls || [])
      .map((url) => `<img src="${url}" class="face-sample-img" alt="Cluster ${c.cluster_id}">`)
      .join("");

    card.innerHTML = `
      <div class="face-card-header">
        <strong style="font-size: 1rem;">Cluster #${c.cluster_id}</strong>
        <span class="badge" style="background: var(--bg);">${c.count} sightings</span>
      </div>
      <div style="font-size: 0.75rem; color: var(--muted);">Last seen: ${dateStr}</div>
      <div class="face-samples">${samplesHtml}</div>
      <div style="display: flex; gap: 6px; margin-top: auto;">
        <input type="text" class="form-control form-control-sm cluster-name-input" list="identities-datalist" placeholder="Assign identity name..." style="flex: 1;">
        <button class="btn btn-primary btn-sm btn-assign">Assign</button>
      </div>
    `;

    card.querySelector(".btn-assign").onclick = async () => {
      const input = card.querySelector(".cluster-name-input");
      const name = input.value.trim();
      if (!name) return alert("Please enter an identity name");

      try {
        const res = await fetch(`/api/faces/clusters/${c.cluster_id}/assign`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name }),
        });
        if (res.ok) {
          showToast("Success", `Assigned cluster to ${name}`);
          loadFacesView();
        } else {
          alert("Failed to assign cluster");
        }
      } catch (err) {
        alert(`Error assigning cluster: ${err}`);
      }
    };

    grid.appendChild(card);
  });
}

// Scenery & Vehicle Anchors View
let currentSceneryCam = "";
let currentScenerySlots = [];
let drawingSlot = false;
let drawStartX = 0;
let drawStartY = 0;

async function loadSceneryView() {
  await populateSceneryCameras();
  initSceneryCanvas();
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
      opt.textContent = `${name} (${cameras[name].slug})`;
      select.appendChild(opt);
    });

    if (!currentSceneryCam || !cameras[currentSceneryCam]) {
      currentSceneryCam = camNames[0];
    }
    select.value = currentSceneryCam;

    select.onchange = () => {
      currentSceneryCam = select.value;
      updateSceneryView();
    };

    updateSceneryView();
  } catch (err) {
    console.error("Scenery cameras load error:", err);
  }
}

async function updateSceneryView() {
  if (!currentSceneryCam) return;
  document.getElementById("scenery-cam-title").textContent = `Camera: ${currentSceneryCam}`;

  const slug = currentSceneryCam.replace(/[^A-Za-z0-9_-]+/g, "_");
  const img = document.getElementById("scenery-snap-img");
  img.src = `/api/live/${encodeURIComponent(slug)}/snapshot.jpg?t=${Date.now()}`;

  img.onload = () => {
    drawSceneryOverlay();
  };

  await fetchScenerySlots();
}

async function fetchScenerySlots() {
  if (!currentSceneryCam) return;
  try {
    const res = await fetch(`/api/scenery/slots?camera=${encodeURIComponent(currentSceneryCam)}`);
    const slots = await res.json();
    currentScenerySlots = slots;
    renderScenerySlots(slots);
    drawSceneryOverlay();
  } catch (err) {
    console.error("Fetch slots error:", err);
  }
}

function renderScenerySlots(slots) {
  const grid = document.getElementById("scenery-slots-grid");
  grid.innerHTML = "";

  if (slots.length === 0) {
    grid.innerHTML = `<div style="grid-column: 1 / -1; color: var(--muted); padding: 20px 0;">No vehicle slots registered for ${escapeHtml(currentSceneryCam)}. Click and drag on the snapshot above to define a persistent slot anchor!</div>`;
    return;
  }

  slots.forEach((s) => {
    const card = document.createElement("div");
    card.className = "face-card";
    const boxStr = Array.isArray(s.slot_box) ? s.slot_box.map(v => typeof v === 'number' ? v.toFixed(2) : v).join(", ") : "";

    card.innerHTML = `
      <div class="face-card-header">
        <strong style="font-size: 1rem;">${escapeHtml(s.name)}</strong>
        <span class="badge" style="background: ${s.is_friendly ? 'rgba(16, 185, 129, 0.2); color: #34d399;' : 'rgba(239, 68, 68, 0.2); color: #f87171;'}">
          ${s.is_friendly ? 'Friendly Anchor' : 'Alert Anchor'}
        </span>
      </div>
      <div style="font-size: 0.8rem; color: var(--muted);">
        <div>Box [x1, y1, x2, y2]: [${boxStr}]</div>
        <div>Dominant Color: <strong>${escapeHtml(s.color_name || 'unknown')}</strong></div>
      </div>
      <div style="display: flex; gap: 8px; margin-top: auto;">
        <button class="btn btn-outline btn-sm btn-toggle-friendly">${s.is_friendly ? 'Mark Alert' : 'Mark Friendly'}</button>
        <button class="btn btn-danger btn-sm btn-delete-slot" style="margin-left: auto;">Delete</button>
      </div>
    `;

    card.querySelector(".btn-toggle-friendly").onclick = async () => {
      try {
        const res = await fetch(`/api/scenery/slots/${s.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ is_friendly: !s.is_friendly }),
        });
        if (res.ok) fetchScenerySlots();
      } catch (err) {
        alert(`Error updating slot: ${err}`);
      }
    };

    card.querySelector(".btn-delete-slot").onclick = async () => {
      if (!confirm(`Delete vehicle slot "${s.name}"?`)) return;
      try {
        const res = await fetch(`/api/scenery/slots/${s.id}`, { method: "DELETE" });
        if (res.ok) {
          showToast("Deleted", `Removed slot "${s.name}"`);
          fetchScenerySlots();
        }
      } catch (err) {
        alert(`Error deleting slot: ${err}`);
      }
    };

    grid.appendChild(card);
  });
}

function initSceneryCanvas() {
  const canvas = document.getElementById("scenery-overlay-canvas");
  const container = document.getElementById("scenery-canvas-container");
  if (!canvas) return;

  function resizeCanvas() {
    if (container && container.clientWidth > 0) {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      drawSceneryOverlay();
    }
  }

  window.addEventListener("resize", resizeCanvas);
  setTimeout(resizeCanvas, 100);

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
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(
      Math.min(startX, curX),
      Math.min(startY, curY),
      Math.abs(curX - startX),
      Math.abs(curY - startY)
    );
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

    if ((x2 - x1) > 0.02 && (y2 - y1) > 0.02) {
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

      if (!name) return alert("Please enter a vehicle name");
      if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) {
        return alert("Please define valid slot coordinates by clicking and dragging on the snapshot");
      }

      try {
        const res = await fetch("/api/scenery/slots", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            camera: currentSceneryCam,
            name: name,
            slot_box: [x1, y1, x2, y2],
            is_friendly: isFriendly,
          }),
        });
        if (res.ok) {
          showToast("Saved", `Registered slot "${name}"`);
          document.getElementById("slot-name-input").value = "";
          fetchScenerySlots();
        } else {
          const err = await res.json();
          alert(`Failed to save slot: ${err.detail || "Error"}`);
        }
      } catch (err) {
        alert(`Error saving slot: ${err}`);
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
    const sx1 = slot.slot_box[0] * canvas.width;
    const sy1 = slot.slot_box[1] * canvas.height;
    const sw = (slot.slot_box[2] - slot.slot_box[0]) * canvas.width;
    const sh = (slot.slot_box[3] - slot.slot_box[1]) * canvas.height;

    ctx.strokeStyle = slot.is_friendly ? "#10b981" : "#f59e0b";
    ctx.lineWidth = 2;
    ctx.strokeRect(sx1, sy1, sw, sh);

    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(sx1, Math.max(0, sy1 - 20), ctx.measureText(slot.name).width + 12, 18);

    ctx.fillStyle = slot.is_friendly ? "#34d399" : "#fbbf24";
    ctx.font = "12px sans-serif";
    ctx.fillText(slot.name, sx1 + 6, Math.max(14, sy1 - 6));
  });
}

// Archive View
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
    const container = document.getElementById("archive-summary");
    container.innerHTML = `
      <div class="summary-card">
        <div class="summary-value">${s.total_analyzed || 0}</div>
        <div class="summary-label">Total Clips</div>
      </div>
      <div class="summary-card">
        <div class="summary-value" style="color: var(--accent-keep);">${s.kept_clips || 0}</div>
        <div class="summary-label">Kept Events</div>
      </div>
      <div class="summary-card">
        <div class="summary-value" style="color: var(--accent-discard);">${s.discarded_clips || 0}</div>
        <div class="summary-label">Suppressed / Discarded</div>
      </div>
      <div class="summary-card">
        <div class="summary-value" style="color: #38bdf8;">${s.reduction_percentage || 0}%</div>
        <div class="summary-label">Nuisance Reduction</div>
      </div>
    `;
  } catch (err) {
    console.error("Archive summary error:", err);
  }
}

async function fetchArchiveList(offset = 0) {
  archiveOffset = offset;
  const verdict = document.getElementById("archive-filter-verdict").value;
  const camera = document.getElementById("archive-filter-camera").value;
  const reason = document.getElementById("archive-filter-reason").value;

  const params = new URLSearchParams({
    offset: offset.toString(),
    limit: ARCHIVE_LIMIT.toString(),
  });
  if (verdict) params.set("verdict", verdict);
  if (camera) params.set("camera", camera);
  if (reason) params.set("reason", reason);

  try {
    const res = await fetch(`/api/archive?${params.toString()}`);
    if (!res.ok) return;
    const data = await res.json();
    renderArchiveGrid(data.items || []);
    renderPagination("archive-pagination", data.total || 0, offset, ARCHIVE_LIMIT, fetchArchiveList);
  } catch (err) {
    console.error("Fetch archive error:", err);
  }
}

function renderArchiveGrid(items) {
  const grid = document.getElementById("archive-grid");
  grid.innerHTML = "";

  if (items.length === 0) {
    grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--muted); padding: 40px;">No archive clips found</div>`;
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "archive-card";

    const isKeep = item.verdict === "KEEP";
    const badgeColor = isKeep ? "var(--accent-keep)" : "var(--accent-discard)";
    const thumbSrc = item.thumb_url || "/static/placeholder.jpg";

    card.innerHTML = `
      <div class="card-thumb">
        <img src="${thumbSrc}" alt="Clip ${item.idx}" loading="lazy">
      </div>
      <div class="card-body">
        <div class="card-meta">
          <span>${escapeHtml(item.camera || "Unknown")}</span>
          <span style="font-weight: 700; color: ${badgeColor};">${item.verdict}</span>
        </div>
        <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 4px;">
          ${escapeHtml(item.primary_reason || item.reason || "")}
        </div>
        <div style="font-size: 0.75rem; color: var(--muted);">
          Confidence: ${Math.round((item.confidence || 0) * 100)}%
        </div>
      </div>
    `;

    card.onclick = () => openArchiveModal(item);
    grid.appendChild(card);
  });
}

function openArchiveModal(item) {
  const modal = document.getElementById("modal");
  const modalBody = document.getElementById("modal-body");

  let videoHtml = "";
  if (item.clip_url) {
    videoHtml = `<video controls autoplay style="width: 100%; max-height: 480px; background: #000; border-radius: 6px;" src="${item.clip_url}"></video>`;
  } else {
    videoHtml = `<div style="text-align: center; padding: 40px; color: var(--muted);">Clip video file not accessible</div>`;
  }

  modalBody.innerHTML = `
    <h3 style="margin-bottom: 8px;">${escapeHtml(item.camera || "Archive Clip")}</h3>
    <div style="color: var(--muted); font-size: 0.85rem; margin-bottom: 14px;">Verdict: <strong>${item.verdict}</strong> &bull; Reason: ${escapeHtml(item.primary_reason || item.reason || "")}</div>
    ${videoHtml}
  `;
  modal.classList.remove("hidden");
}

// Pagination Component
function renderPagination(containerId, total, currentOffset, limit, onPage) {
  const bar = document.getElementById(containerId);
  bar.innerHTML = "";
  if (total <= limit) return;

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(currentOffset / limit) + 1;

  if (currentPage > 1) {
    const prevBtn = document.createElement("button");
    prevBtn.className = "btn btn-outline btn-sm";
    prevBtn.textContent = "Previous";
    prevBtn.onclick = () => onPage((currentPage - 2) * limit);
    bar.appendChild(prevBtn);
  }

  const pageInfo = document.createElement("span");
  pageInfo.style.alignSelf = "center";
  pageInfo.style.fontSize = "0.85rem";
  pageInfo.textContent = `Page ${currentPage} of ${totalPages} (${total} total)`;
  bar.appendChild(pageInfo);

  if (currentPage < totalPages) {
    const nextBtn = document.createElement("button");
    nextBtn.className = "btn btn-outline btn-sm";
    nextBtn.textContent = "Next";
    nextBtn.onclick = () => onPage(currentPage * limit);
    bar.appendChild(nextBtn);
  }
}

// Helpers
function escapeHtml(text) {
  if (text === null || text === undefined) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Initialization & Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  initRouter();
  initSSE();

  // Modal close
  document.getElementById("modal-close").onclick = closeModal;
  document.getElementById("modal-backdrop").onclick = closeModal;

  // Browser notifications button
  document.getElementById("btn-browser-notifications").onclick = async () => {
    if ("Notification" in window) {
      const perm = await Notification.requestPermission();
      if (perm === "granted") {
        showToast("Enabled", "Browser notifications are enabled!");
      }
    }
  };
  // Toggle Client Overlays button
  const toggleBoxesBtn = document.getElementById("btn-toggle-client-boxes");
  if (toggleBoxesBtn) {
    toggleBoxesBtn.onclick = () => {
      showClientOverlays = !showClientOverlays;
      toggleBoxesBtn.textContent = `Toggle Overlays (${showClientOverlays ? "ON" : "OFF"})`;
      document.querySelectorAll(".camera-tile").forEach((tile) => {
        const canvas = tile.querySelector(".camera-live-overlay");
        if (canvas && !showClientOverlays) {
          const ctx = canvas.getContext("2d");
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      });
    };
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
  document.getElementById("archive-filter-reset").onclick = () => {
    document.getElementById("archive-filter-verdict").value = "";
    document.getElementById("archive-filter-camera").value = "";
    document.getElementById("archive-filter-reason").value = "";
    fetchArchiveList(0);
  };

  // Create Identity button
  document.getElementById("btn-new-identity").onclick = async () => {
    const name = prompt("Enter name for new identity:");
    if (!name || !name.trim()) return;

    try {
      const res = await fetch("/api/identities", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() }),
      });
      if (res.ok) {
        showToast("Created", `Identity '${name}' created`);
        loadFacesView();
      } else {
        const err = await res.json();
        alert(`Failed to create identity: ${err.detail || "Error"}`);
      }
    } catch (err) {
      alert(`Error creating identity: ${err}`);
    }
  };
});
