/* Melbourne CBD Real-Time Pedestrian Crowd & Sensory Map logic. */
(function () {
  "use strict";

  let map = null;
  let sensorsLayer = null;
  let routeLayerGroup = null;
  let origMarker = null;
  let destMarker = null;

  const state = {
    orig: null, // { lat, lon, name }
    dest: null, // { lat, lon, name }
    sensors: [],
  };

  const $ = (id) => document.getElementById(id);

  document.addEventListener("DOMContentLoaded", () => {
    initMap();
    bindEvents();
    loadSensorsMap();
  });

  function initMap() {
    if (!$("map")) return;

    map = L.map("map", {
      center: [-37.815, 144.965],
      zoom: 14,
      minZoom: 13,
      maxBounds: [[-37.840, 144.920], [-37.780, 145.010]],
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 18,
    }).addTo(map);

    sensorsLayer = L.layerGroup().addTo(map);
    routeLayerGroup = L.layerGroup().addTo(map);

    // Zoom-dependent information density
    map.on("zoomend", () => {
      renderSensorsOnMap(state.sensors);
    });

    // Map click sets origin / destination
    map.on("click", (e) => {
      const { lat, lng } = e.latlng;
      if (!state.orig) {
        setOrigin(lat, lng, `Point (${lat.toFixed(4)}, ${lng.toFixed(4)})`);
      } else if (!state.dest) {
        setDestination(lat, lng, `Point (${lat.toFixed(4)}, ${lng.toFixed(4)})`);
        calculateRoute();
      } else {
        setOrigin(lat, lng, `Point (${lat.toFixed(4)}, ${lng.toFixed(4)})`);
        setDestination(null, null, "");
      }
    });
  }

  async function loadSensorsMap() {
    const statusEl = $("mapStatus");

    try {
      const r = await fetch("/api/map");
      if (!r.ok) throw new Error("API error");
      const data = await r.json();
      state.sensors = data.sensors || [];

      renderSensorsOnMap(state.sensors);

      if (statusEl) {
        statusEl.textContent = `Live Feed: ${state.sensors.length} CBD sensors active`;
        statusEl.className = "badge badge-ok";
      }
    } catch (e) {
      if (statusEl) {
        statusEl.textContent = "Offline / failed to load live sensors";
        statusEl.className = "badge badge-warn";
      }
    }
  }

  function renderSensorsOnMap(sensors) {
    if (!sensorsLayer || !map) return;
    sensorsLayer.clearLayers();

    const zoom = map.getZoom();
    const isHighDetail = zoom >= 15;

    sensors.forEach((s) => {
      const color = s.level === "HIGH" ? "#dc3545" : s.level === "MEDIUM" ? "#ffc107" : "#28a745";
      const radius = isHighDetail ? 9 : 7;

      const marker = L.circleMarker([s.latitude, s.longitude], {
        radius: radius,
        fillColor: color,
        color: "#000000",
        weight: isHighDetail ? 2 : 1.5,
        opacity: 1,
        fillOpacity: 0.85,
      });

      if (isHighDetail) {
        marker.bindTooltip(
          `<strong>${escapeHtml(s.description || s.name)}</strong><br>${s.sensory_label} (${Math.round(s.current_count)}/hr)`,
          { permanent: false, direction: "top", offset: [0, -8] }
        );
      }

      const popupHtml = `
        <div style="font-family: system-ui, sans-serif; min-width: 190px;">
          <strong style="font-size: 1.05rem;">${escapeHtml(s.description || s.name)}</strong>
          <p style="margin: 0.3rem 0;">Status: <span class="badge badge-${s.level.toLowerCase()}">${escapeHtml(s.sensory_label || s.level)}</span></p>
          <p style="margin: 0.3rem 0;">Live Count: <strong>${Math.round(s.current_count)}</strong> people/hr</p>
          <p style="margin: 0.3rem 0; font-size: 0.88rem; color: #555;">${escapeHtml(s.sensory_advice || '')}</p>
          <div style="margin-top: 0.6rem; display: flex; flex-direction: column; gap: 0.3rem;">
            <button onclick="window.setOrigFromPopup(${s.latitude}, ${s.longitude}, '${escapeHtml(s.name)}')">Start Route Here</button>
            <button onclick="window.setDestFromPopup(${s.latitude}, ${s.longitude}, '${escapeHtml(s.name)}')">End Route Here</button>
          </div>
        </div>
      `;

      marker.bindPopup(popupHtml);
      sensorsLayer.addLayer(marker);
    });
  }

  window.setOrigFromPopup = (lat, lon, name) => {
    setOrigin(lat, lon, name);
    map.closePopup();
    if (state.dest) calculateRoute();
  };

  window.setDestFromPopup = (lat, lon, name) => {
    setDestination(lat, lon, name);
    map.closePopup();
    if (state.orig) calculateRoute();
  };

  function setOrigin(lat, lon, name) {
    if (lat === null) {
      state.orig = null;
      if (origMarker) map.removeLayer(origMarker);
      origMarker = null;
      if ($("origInput")) $("origInput").value = "";
      return;
    }
    state.orig = { lat, lon, name };
    if ($("origInput")) $("origInput").value = name;

    if (origMarker) map.removeLayer(origMarker);
    origMarker = L.marker([lat, lon], {
      title: "Origin",
      alt: "Start point",
    }).addTo(map).bindPopup(`<strong>Start:</strong> ${escapeHtml(name)}`).openPopup();
  }

  function setDestination(lat, lon, name) {
    if (lat === null) {
      state.dest = null;
      if (destMarker) map.removeLayer(destMarker);
      destMarker = null;
      if ($("destInput")) $("destInput").value = "";
      return;
    }
    state.dest = { lat, lon, name };
    if ($("destInput")) $("destInput").value = name;

    if (destMarker) map.removeLayer(destMarker);
    destMarker = L.marker([lat, lon], {
      title: "Destination",
      alt: "End point",
    }).addTo(map).bindPopup(`<strong>Destination:</strong> ${escapeHtml(name)}`).openPopup();
  }

  function bindEvents() {
    setupGeocodeInput("origInput", "origSuggestions", (item) => {
      setOrigin(item.latitude, item.longitude, item.display_name);
      if (state.dest) calculateRoute();
    });

    setupGeocodeInput("destInput", "destSuggestions", (item) => {
      setDestination(item.latitude, item.longitude, item.display_name);
      if (state.orig) calculateRoute();
    });

    if ($("planRouteBtn")) {
      $("planRouteBtn").addEventListener("click", calculateRoute);
    }

    if ($("clearRouteBtn")) {
      $("clearRouteBtn").addEventListener("click", () => {
        setOrigin(null, null, "");
        setDestination(null, null, "");
        if (routeLayerGroup) routeLayerGroup.clearLayers();
        if ($("routeResultsSection")) $("routeResultsSection").style.display = "none";
        if ($("routeStatus")) {
          $("routeStatus").textContent = "Enter locations to view route";
          $("routeStatus").className = "badge badge-unknown";
        }
      });
    }
  }

  function setupGeocodeInput(inputId, suggestionsId, onSelect) {
    const input = $(inputId);
    const box = $(suggestionsId);
    if (!input || !box) return;

    let timer = null;

    input.addEventListener("input", () => {
      clearTimeout(timer);
      const q = input.value.trim();
      if (q.length < 2) {
        box.hidden = true;
        box.innerHTML = "";
        return;
      }

      timer = setTimeout(async () => {
        try {
          const r = await fetch(`/api/geocode?q=${encodeURIComponent(q)}`);
          if (!r.ok) return;
          const data = await r.json();
          renderSuggestions(data.results || [], box, onSelect);
        } catch (e) {}
      }, 300);
    });

    document.addEventListener("click", (e) => {
      if (!input.contains(e.target) && !box.contains(e.target)) {
        box.hidden = true;
      }
    });
  }

  function renderSuggestions(results, box, onSelect) {
    box.innerHTML = "";
    if (!results.length) {
      box.hidden = true;
      return;
    }

    results.forEach((item) => {
      const div = document.createElement("div");
      div.className = "suggestion-item";
      div.tabIndex = 0;
      div.textContent = item.display_name;
      div.addEventListener("click", () => {
        onSelect(item);
        box.hidden = true;
      });
      div.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          onSelect(item);
          box.hidden = true;
        }
      });
      box.appendChild(div);
    });

    box.hidden = false;
  }

  async function calculateRoute() {
    const routeStatus = $("routeStatus");
    if (!state.orig || !state.dest) {
      if (routeStatus) {
        routeStatus.textContent = "Please select start and end points.";
        routeStatus.className = "badge badge-warn";
      }
      return;
    }

    if (routeStatus) {
      routeStatus.textContent = "Comparing live route sensory loads…";
      routeStatus.className = "badge badge-unknown";
    }

    const params = new URLSearchParams({
      orig_lat: state.orig.lat,
      orig_lon: state.orig.lon,
      dest_lat: state.dest.lat,
      dest_lon: state.dest.lon,
      mode: "rule",
    });

    try {
      const r = await fetch(`/api/route?${params.toString()}`);
      if (!r.ok) throw new Error("Route calculation failed");
      const data = await r.json();

      displayRoutesOnMap(data.routes);
      renderRouteCards(data);

      if (routeStatus) {
        if (data.is_fallback) {
          routeStatus.textContent = "Offline mode: Straight-line path shown";
          routeStatus.className = "badge badge-warn";
        } else {
          routeStatus.textContent = `Compared ${data.routes.length} route options`;
          routeStatus.className = "badge badge-ok";
        }
      }
    } catch (e) {
      if (routeStatus) {
        routeStatus.textContent = "Failed to calculate route";
        routeStatus.className = "badge badge-err";
      }
    }
  }

  function displayRoutesOnMap(routes) {
    if (!routeLayerGroup) return;
    routeLayerGroup.clearLayers();

    const bounds = L.latLngBounds();

    routes.forEach((r) => {
      const isLeastCrowded = r.is_least_crowded;
      const isFastest = r.is_fastest;

      const color = isLeastCrowded ? "#28a745" : isFastest ? "#0b5cad" : "#6c757d";
      const weight = isLeastCrowded || isFastest ? 6 : 4;
      const dashArray = r.is_fallback ? "8, 8" : null;

      const polyline = L.polyline(r.coordinates, {
        color: color,
        weight: weight,
        opacity: 0.85,
        dashArray: dashArray,
      });

      routeLayerGroup.addLayer(polyline);
      r.coordinates.forEach((c) => bounds.extend(c));
    });

    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [40, 40] });
    }
  }

  function renderRouteCards(data) {
    const sec = $("routeResultsSection");
    const container = $("routeCards");
    if (!sec || !container) return;

    container.innerHTML = "";
    sec.style.display = "block";

    data.routes.forEach((r, idx) => {
      const card = document.createElement("div");
      card.className = `route-card ${r.is_least_crowded ? "least-crowded" : r.is_fastest ? "fastest" : ""}`;

      let tag = "";
      if (r.is_least_crowded && r.is_fastest) {
        tag = '<span class="badge badge-ok">⭐ FASTEST & CALMEST PATH</span>';
      } else if (r.is_least_crowded) {
        tag = '<span class="badge badge-ok">⭐ RECOMMENDED CALM ROUTE</span>';
      } else if (r.is_fastest) {
        tag = '<span class="badge badge-warn">SHORTEST / FASTEST PATH</span>';
      } else {
        tag = `<span class="badge badge-unknown">OPTION ${idx + 1}</span>`;
      }

      const crowdBadge = `<span class="badge badge-${r.crowd_score.toLowerCase()}">${escapeHtml(r.sensory_tag || r.crowd_score)}</span>`;

      let remarksHtml = "";
      if (r.remarks && r.remarks.length > 0) {
        remarksHtml = `
          <ul class="remarks-list">
            ${r.remarks.map((rem) => `<li>${escapeHtml(rem)}</li>`).join("")}
          </ul>
        `;
      } else {
        remarksHtml = "<p class='hint'>Peaceful path with low crowd density.</p>";
      }

      const fallbackNote = r.is_fallback ? `<p class="badge badge-warn">${escapeHtml(r.fallback_note)}</p>` : "";

      card.innerHTML = `
        <div class="card-header">
          ${tag}
          ${crowdBadge}
        </div>
        <div class="route-metrics">
          ⏱️ <strong>${r.duration_min} mins</strong> (${(r.distance_m / 1000).toFixed(2)} km)
        </div>
        ${fallbackNote}
        <div>
          <strong>Sensory Breakdown:</strong>
          ${remarksHtml}
        </div>
      `;

      container.appendChild(card);
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
})();
