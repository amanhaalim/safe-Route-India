/**
 * frontend/map.js
 * ══════════════════════════════════════════════════════════════════
 * SafeRoute India — Leaflet.js Map Engine
 * ══════════════════════════════════════════════════════════════════
 * Handles:
 *   - Map initialisation and tile layer setup
 *   - Route drawing with colour-coded risk segments
 *   - Crime / flood heatmap overlay
 *   - Origin / destination marker management
 *   - Risk panel rendering
 *   - Route tab switching
 *   - User incident reporting
 *   - City switching and map re-centering
 *
 * Used by: frontend/index.html
 * Calls:   /route, /cities, /heatmap/{city}, /report  (api/main.py)
 */

"use strict";

// ═══════════════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════════════

const API_BASE = "";   // same origin; change to "http://localhost:8000" for dev

const CITY_CENTERS = {
  chennai:   { lat: 13.0827, lon: 80.2707, zoom: 13 },
  mumbai:    { lat: 19.0760, lon: 72.8777, zoom: 13 },
  delhi:     { lat: 28.6139, lon: 77.2090, zoom: 12 },
  bengaluru: { lat: 12.9716, lon: 77.5946, zoom: 13 },
  hyderabad: { lat: 17.3850, lon: 78.4867, zoom: 13 },
};

const ROUTE_STYLES = {
  safest:   { color: "#22c55e", weight: 7,  opacity: 0.90, dashArray: null },
  balanced: { color: "#f59e0b", weight: 5,  opacity: 0.80, dashArray: null },
  fastest:  { color: "#ef4444", weight: 4,  opacity: 0.65, dashArray: "8 4" },
};

const DIM_OPACITY = 0.18;   // opacity for inactive routes

// ═══════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════

const state = {
  map:           null,
  activeCity:    "chennai",
  activeRoute:   "safest",
  routeLayers:   {},          // { safest: L.LayerGroup, ... }
  markerLayer:   null,
  heatmapLayer:  null,
  routeData:     {},          // { safest: { coordinates, summary, segments }, ... }
  originCoords:  null,
  destCoords:    null,
  reportMode:    false,
  reportMarker:  null,
};

// ═══════════════════════════════════════════════════════════════
// INITIALISATION
// ═══════════════════════════════════════════════════════════════

function initMap() {
  const center = CITY_CENTERS[state.activeCity];

  state.map = L.map("map", {
    center:          [center.lat, center.lon],
    zoom:            center.zoom,
    zoomControl:     true,
    attributionControl: true,
  });

  // Base tile layers
  const tiles = {
    "Dark (CartoDB)": L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      { attribution: "© OpenStreetMap © CartoDB", maxZoom: 19 }
    ),
    "Light (CartoDB)": L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      { attribution: "© OpenStreetMap © CartoDB", maxZoom: 19 }
    ),
    "OSM Standard": L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      { attribution: "© OpenStreetMap contributors", maxZoom: 19 }
    ),
  };

  tiles["Dark (CartoDB)"].addTo(state.map);
  L.control.layers(tiles, {}, { position: "topright", collapsed: true }).addTo(state.map);

  // Marker layer group (origin/dest markers)
  state.markerLayer = L.layerGroup().addTo(state.map);

  // Empty route layer groups
  for (const name of ["safest", "balanced", "fastest"]) {
    state.routeLayers[name] = L.layerGroup().addTo(state.map);
  }

  // Scale control
  L.control.scale({ metric: true, imperial: false, position: "bottomleft" }).addTo(state.map);

  // Click handler for report mode
  state.map.on("click", onMapClick);

  // Populate city dropdown
  populateCityDropdown();

  console.log("SafeRoute India map initialised.");
}

// ═══════════════════════════════════════════════════════════════
// CITY MANAGEMENT
// ═══════════════════════════════════════════════════════════════

async function populateCityDropdown() {
  try {
    const res  = await fetch(`${API_BASE}/cities`);
    const data = await res.json();
    const sel  = document.getElementById("citySelect");

    // API returns { cities: { chennai: {...}, mumbai: {...} } }
    const cityKeys = data.cities
      ? Object.keys(data.cities)
      : Object.keys(CITY_CENTERS);

    sel.innerHTML = "";
    cityKeys.forEach(city => {
      const opt      = document.createElement("option");
      opt.value      = city;
      opt.textContent = city.charAt(0).toUpperCase() + city.slice(1);
      if (city === state.activeCity) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.warn("Could not load city list from API — using defaults.", e);
  }
}

function onCityChange() {
  const sel          = document.getElementById("citySelect");
  state.activeCity   = sel.value;
  const center       = CITY_CENTERS[state.activeCity] || { lat: 20, lon: 78, zoom: 11 };
  state.map.setView([center.lat, center.lon], center.zoom);
  clearAll();
}

// ═══════════════════════════════════════════════════════════════
// ROUTE REQUEST
// ═══════════════════════════════════════════════════════════════

async function findRoutes() {
  const origin      = document.getElementById("origin").value.trim();
  const destination = document.getElementById("destination").value.trim();
  const hour        = parseInt(document.getElementById("travelHour").value) || 12;
  const profile     = document.getElementById("profileSelect")?.value || "default";

  if (!origin || !destination) {
    setStatus("⚠️ Please enter both origin and destination.", "warn");
    return;
  }

  setStatus("🔍 Geocoding addresses and computing routes...", "info");
  setLoading(true);
  clearAll();

  try {
    const res = await fetch(`${API_BASE}/route`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        origin,
        destination,
        city:        state.activeCity,
        travel_hour: hour,
        profile,
        include_segments: true,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Server error" }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data        = await res.json();
    state.routeData   = data.routes  || {};
    state.originCoords = data.origin_coords;
    state.destCoords   = data.destination_coords;

    drawAllRoutes();
    placeMarkers();
    showRouteTabs();
    activateRoute("safest");
    fitMapToRoutes();
    setStatus("", "");

  } catch (e) {
    setStatus(`❌ ${e.message}`, "error");
    console.error(e);
  } finally {
    setLoading(false);
  }
}

// ═══════════════════════════════════════════════════════════════
// ROUTE DRAWING
// ═══════════════════════════════════════════════════════════════

function drawAllRoutes() {
  // Draw fastest first (bottom), safest last (top)
  for (const name of ["fastest", "balanced", "safest"]) {
    const route = state.routeData[name];
    if (!route) continue;

    const layerGroup = state.routeLayers[name];
    layerGroup.clearLayers();

    const segments = route.segments;

    if (segments && segments.length > 0) {
      // Draw per-segment risk gradient
      segments.forEach(seg => {
        L.polyline(seg.coords, {
          color:     seg.color,
          weight:    ROUTE_STYLES[name].weight,
          opacity:   DIM_OPACITY,          // start dimmed; activate() brightens
          dashArray: ROUTE_STYLES[name].dashArray,
          lineCap:   "round",
          lineJoin:  "round",
        })
        .bindTooltip(buildSegmentTooltip(name, seg), { sticky: true, opacity: 0.92 })
        .addTo(layerGroup);
      });
    } else {
      // Fallback: solid route colour
      L.polyline(route.coordinates, {
        color:     ROUTE_STYLES[name].color,
        weight:    ROUTE_STYLES[name].weight,
        opacity:   DIM_OPACITY,
        dashArray: ROUTE_STYLES[name].dashArray,
      }).addTo(layerGroup);
    }
  }
}

function buildSegmentTooltip(routeName, seg) {
  return `
    <div style="font-family:Arial;font-size:12px;line-height:1.6">
      <b>${routeName.toUpperCase()} route</b><br>
      <span style="color:#94a3b8">Road:</span> ${seg.highway}<br>
      <span style="color:#ef4444">Crime:</span> ${(seg.crime_score * 100).toFixed(0)}%
      &nbsp;<span style="color:#f59e0b">Accident:</span> ${(seg.accident_score * 100).toFixed(0)}%
      &nbsp;<span style="color:#3b82f6">Flood:</span> ${(seg.flood_score * 100).toFixed(0)}%<br>
      <b>Risk: ${(seg.risk * 100).toFixed(0)}%</b>
    </div>`;
}

function activateRoute(name) {
  state.activeRoute = name;

  // Update tab button styles
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.route === name);
  });

  // Dim all, then highlight active
  for (const [rName, layerGroup] of Object.entries(state.routeLayers)) {
    const isActive = rName === name;
    layerGroup.eachLayer(layer => {
      if (layer.setStyle) {
        layer.setStyle({
          opacity: isActive ? ROUTE_STYLES[rName].opacity : DIM_OPACITY,
          weight:  isActive ? ROUTE_STYLES[rName].weight  : 2,
        });
      }
    });
    // Bring active to front
    if (isActive) layerGroup.eachLayer(l => l.bringToFront?.());
  }

  renderRiskPanel(name);
}

// ═══════════════════════════════════════════════════════════════
// RISK PANEL
// ═══════════════════════════════════════════════════════════════

function renderRiskPanel(routeName) {
  const panel = document.getElementById("riskPanel");
  const route = state.routeData[routeName];

  if (!route) {
    panel.innerHTML = `<p style="color:#94a3b8;font-size:13px">No ${routeName} route found.</p>`;
    panel.style.display = "block";
    return;
  }

  const s    = route.summary;
  const icon = { safest: "🟢", balanced: "🟡", fastest: "🔴" }[routeName];

  panel.style.display = "block";
  panel.innerHTML = `
    <div class="route-header" style="color:${ROUTE_STYLES[routeName].color}">
      ${icon} ${routeName.toUpperCase()} ROUTE
    </div>

    <div class="stat-grid">
      ${statRow("📏 Distance",        `${s.distance_km} km`)}
      ${statRow("⏱ Est. Time",       `~${s.estimated_minutes} min`)}
      ${statRow("🛡 Safety Score",    `${s.safety_pct}%`,   safetyColor(s.safety_pct))}
      ${statRow("⚠️ Avg Risk",        s.avg_risk_score.toFixed(3), riskColor(s.avg_risk_score))}
      ${statRow("🔴 High-Risk Segs",  `${s.high_risk_segments} / ${s.total_segments}`)}
      ${statRow("📊 Risk Level",      s.risk_label,          s.risk_color)}
    </div>

    <div class="bars-section">
      ${riskBar("🔴 Crime",    s.avg_crime_score,    "#ef4444")}
      ${riskBar("🟠 Accident", s.avg_accident_score, "#f59e0b")}
      ${riskBar("🔵 Flood",    s.avg_flood_score,    "#3b82f6")}
      ${riskBar("🟤 Road Type",s.avg_road_score,     "#a78bfa")}
    </div>

    <div class="seg-dist">
      ${segBar("Low",    s.low_risk_segments,    s.total_segments, "#22c55e")}
      ${segBar("Medium", s.medium_risk_segments, s.total_segments, "#f59e0b")}
      ${segBar("High",   s.high_risk_segments,   s.total_segments, "#ef4444")}
    </div>
  `;
}

function statRow(label, value, color = "#ffffff") {
  return `
    <div class="stat-row">
      <span class="stat-label">${label}</span>
      <span class="stat-value" style="color:${color}">${value}</span>
    </div>`;
}

function riskBar(label, score, color) {
  const pct = Math.round(score * 100);
  return `
    <div class="risk-bar-wrap">
      <div class="risk-bar-header">
        <span>${label}</span><span>${pct}%</span>
      </div>
      <div class="risk-bar-outer">
        <div class="risk-bar-inner" style="width:${pct}%;background:${color}"></div>
      </div>
    </div>`;
}

function segBar(label, count, total, color) {
  const pct = total > 0 ? Math.round(100 * count / total) : 0;
  return `
    <div class="seg-bar-wrap">
      <span style="color:${color};font-size:11px">${label}: ${count} (${pct}%)</span>
      <div class="risk-bar-outer" style="height:5px;margin-top:2px">
        <div class="risk-bar-inner" style="width:${pct}%;background:${color};height:5px"></div>
      </div>
    </div>`;
}

function safetyColor(pct) {
  if (pct >= 75)  return "#22c55e";
  if (pct >= 50)  return "#f59e0b";
  return "#ef4444";
}

function riskColor(score) {
  if (score < 0.25) return "#22c55e";
  if (score < 0.50) return "#f59e0b";
  return "#ef4444";
}

// ═══════════════════════════════════════════════════════════════
// MARKERS
// ═══════════════════════════════════════════════════════════════

function placeMarkers() {
  state.markerLayer.clearLayers();
  if (!state.originCoords || !state.destCoords) return;

  const makeIcon = (emoji, label, bg) => L.divIcon({
    className: "",
    iconAnchor: [16, 16],
    html: `
      <div style="background:${bg};color:#fff;padding:5px 10px;
                  border-radius:20px;font-size:12px;font-weight:700;
                  white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,0.4)">
        ${emoji} ${label}
      </div>`,
  });

  L.marker(state.originCoords, { icon: makeIcon("🟢", "Start", "#1d4ed8"), zIndexOffset: 1000 })
    .bindPopup(`<b>Origin</b><br>${state.originCoords[0].toFixed(4)}, ${state.originCoords[1].toFixed(4)}`)
    .addTo(state.markerLayer);

  L.marker(state.destCoords, { icon: makeIcon("🔴", "End", "#dc2626"), zIndexOffset: 1000 })
    .bindPopup(`<b>Destination</b><br>${state.destCoords[0].toFixed(4)}, ${state.destCoords[1].toFixed(4)}`)
    .addTo(state.markerLayer);
}

// ═══════════════════════════════════════════════════════════════
// HEATMAP OVERLAY
// ═══════════════════════════════════════════════════════════════

async function toggleHeatmap() {
  const btn = document.getElementById("heatmapBtn");

  if (state.heatmapLayer) {
    state.map.removeLayer(state.heatmapLayer);
    state.heatmapLayer = null;
    btn.textContent = "🌡 Show Crime Heatmap";
    btn.classList.remove("active");
    return;
  }

  setStatus("Loading crime heatmap...", "info");
  btn.disabled = true;

  try {
    const res  = await fetch(`${API_BASE}/heatmap/${state.activeCity}`);
    const data = await res.json();

    if (!data.points || data.points.length === 0) {
      setStatus("⚠️ No heatmap data available for this city.", "warn");
      return;
    }

    state.heatmapLayer = L.heatLayer(data.points, {
      radius:   20,
      blur:     14,
      maxZoom:  16,
      max:      1.0,
      gradient: { 0.2: "#3b82f6", 0.4: "#22c55e", 0.65: "#f59e0b", 0.85: "#f97316", 1.0: "#ef4444" },
    }).addTo(state.map);

    btn.textContent = "🌡 Hide Crime Heatmap";
    btn.classList.add("active");
    setStatus("", "");
  } catch (e) {
    setStatus(`❌ Heatmap load failed: ${e.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════
// INCIDENT REPORTING
// ═══════════════════════════════════════════════════════════════

function toggleReportMode() {
  state.reportMode = !state.reportMode;
  const btn = document.getElementById("reportBtn");

  if (state.reportMode) {
    btn.textContent  = "❌ Cancel Report";
    btn.style.background = "#dc2626";
    setStatus("📍 Click anywhere on the map to place your incident report.", "info");
    state.map.getContainer().style.cursor = "crosshair";
  } else {
    btn.textContent  = "📢 Report Incident";
    btn.style.background = "";
    setStatus("", "");
    state.map.getContainer().style.cursor = "";
    if (state.reportMarker) {
      state.map.removeLayer(state.reportMarker);
      state.reportMarker = null;
    }
  }
}

function onMapClick(e) {
  if (!state.reportMode) return;
  const { lat, lng } = e.latlng;

  if (state.reportMarker) state.map.removeLayer(state.reportMarker);

  state.reportMarker = L.marker([lat, lng], {
    icon: L.divIcon({
      className: "",
      html: `<div style="font-size:24px;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.5))">⚠️</div>`,
    }),
  }).addTo(state.map);

  // Show report form popup
  const form = buildReportPopup(lat, lng);
  state.reportMarker.bindPopup(form, { maxWidth: 260 }).openPopup();
}

function buildReportPopup(lat, lng) {
  return `
    <div style="font-family:Arial;font-size:13px">
      <b>Report Incident</b><br>
      <small style="color:#6b7280">${lat.toFixed(5)}, ${lng.toFixed(5)}</small><br><br>
      <label>Type:<br>
        <select id="rType" style="width:100%;margin-top:3px;padding:4px">
          <option value="crime">Crime</option>
          <option value="accident">Accident</option>
          <option value="flood">Flood / Waterlogging</option>
          <option value="road_damage">Road Damage</option>
          <option value="other">Other</option>
        </select>
      </label><br><br>
      <label>Severity:<br>
        <select id="rSev" style="width:100%;margin-top:3px;padding:4px">
          <option value="1">Low</option>
          <option value="2">Medium</option>
          <option value="3" selected>High</option>
        </select>
      </label><br><br>
      <label>Notes (optional):<br>
        <input id="rNotes" type="text" style="width:100%;margin-top:3px;padding:4px"
               placeholder="Brief description">
      </label><br><br>
      <button onclick="submitReport(${lat}, ${lng})"
              style="background:#1d4ed8;color:#fff;border:none;padding:7px 16px;
                     border-radius:6px;cursor:pointer;width:100%;font-size:13px">
        Submit Report
      </button>
    </div>`;
}

async function submitReport(lat, lng) {
  const type     = document.getElementById("rType").value;
  const severity = parseInt(document.getElementById("rSev").value);
  const notes    = document.getElementById("rNotes").value.trim();

  try {
    const res = await fetch(`${API_BASE}/report`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ lat, lon: lng, incident_type: type, severity, notes, city: state.activeCity }),
    });
    const data = await res.json();
    if (state.reportMarker) state.reportMarker.closePopup();
    setStatus("✅ Report submitted. Thank you!", "success");
    toggleReportMode();
  } catch (e) {
    setStatus(`❌ Report failed: ${e.message}`, "error");
  }
}

// ═══════════════════════════════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════════════════════════════

function showRouteTabs() {
  document.getElementById("routeTabs").style.display  = "flex";
}

function fitMapToRoutes() {
  const allCoords = [];
  for (const route of Object.values(state.routeData)) {
    if (route?.coordinates) allCoords.push(...route.coordinates);
  }
  if (allCoords.length > 0) {
    state.map.fitBounds(L.latLngBounds(allCoords), { padding: [40, 40] });
  }
}

function clearAll() {
  for (const lg of Object.values(state.routeLayers)) lg.clearLayers();
  state.markerLayer.clearLayers();
  state.routeData  = {};
  state.activeRoute = "safest";
  document.getElementById("routeTabs").style.display  = "none";
  document.getElementById("riskPanel").style.display  = "none";
  if (state.heatmapLayer) {
    state.map.removeLayer(state.heatmapLayer);
    state.heatmapLayer = null;
  }
}

function setStatus(msg, type = "") {
  const el = document.getElementById("statusMsg");
  if (!el) return;
  el.textContent = msg;
  el.className   = "status " + type;
}

function setLoading(on) {
  const btn = document.getElementById("searchBtn");
  if (!btn) return;
  btn.disabled    = on;
  btn.textContent = on ? "⏳ Computing routes..." : "🔍 Find Safe Route";
}

// ═══════════════════════════════════════════════════════════════
// KEYBOARD SHORTCUT
// ═══════════════════════════════════════════════════════════════

document.addEventListener("keydown", e => {
  if ((e.key === "Enter") && (e.target.id === "origin" || e.target.id === "destination")) {
    findRoutes();
  }
});

// ═══════════════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════════════

window.addEventListener("DOMContentLoaded", initMap);

// Expose functions needed by inline HTML onclick handlers
window.findRoutes     = findRoutes;
window.onCityChange   = onCityChange;
window.activateRoute  = activateRoute;
window.toggleHeatmap  = toggleHeatmap;
window.toggleReportMode = toggleReportMode;
window.submitReport   = submitReport;