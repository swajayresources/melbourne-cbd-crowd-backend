/* Predict Future Crowd & Area Calmness page logic. */
(function () {
  "use strict";

  let predictMap = null;
  let routeLayerGroup = null;

  const state = {
    dest: null, // { lat, lon, name }
    orig: null, // { lat, lon, name }
    datetime: "",
    lastPredData: null,
  };

  const $ = (id) => document.getElementById(id);

  document.addEventListener("DOMContentLoaded", () => {
    initDateTimeInput();
    bindEvents();
  });

  function initDateTimeInput() {
    const timeInput = $("predTime");
    if (timeInput) {
      const now = new Date();
      now.setHours(now.getHours() + 1); // Default to next hour
      now.setMinutes(0);
      now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
      timeInput.value = now.toISOString().slice(0, 16);
    }
  }

  function bindEvents() {
    setupGeocodeInput("destSearch", "destPredictSuggestions", (item) => {
      state.dest = { lat: item.latitude, lon: item.longitude, name: item.display_name };
      if ($("destSearch")) $("destSearch").value = item.display_name;
    });

    setupGeocodeInput("travelOrigInput", "travelOrigSuggestions", (item) => {
      state.orig = { lat: item.latitude, lon: item.longitude, name: item.display_name };
      if ($("travelOrigInput")) $("travelOrigInput").value = item.display_name;
    });

    if ($("predictBtn")) {
      $("predictBtn").addEventListener("click", runPrediction);
    }

    if ($("toggleTravelBtn")) {
      $("toggleTravelBtn").addEventListener("click", () => {
        const sec = $("futureTravelSection");
        if (sec) {
          sec.style.display = "block";
          sec.scrollIntoView({ behavior: "smooth" });
        }
      });
    }

    if ($("calcFutureRouteBtn")) {
      $("calcFutureRouteBtn").addEventListener("click", calculateFutureRoute);
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

  async function runPrediction() {
    const destInput = $("destSearch");
    const timeInput = $("predTime");
    const sec = $("predictResultsSection");

    const query = destInput ? destInput.value.trim() : "";
    const dtVal = timeInput ? timeInput.value : "";

    if (!query) {
      alert("Please type a location to check.");
      return;
    }

    state.datetime = dtVal;

    const params = new URLSearchParams({ q: query });
    if (dtVal) params.append("datetime", dtVal);

    try {
      const r = await fetch(`/api/predict?${params.toString()}`);
      if (!r.ok) throw new Error("Prediction API error");
      const data = await r.json();

      state.lastPredData = data;
      state.dest = {
        lat: data.latitude,
        lon: data.longitude,
        name: data.display_name || query,
      };

      renderPredictionResults(data);
      if (sec) sec.style.display = "block";
    } catch (e) {
      alert("Could not retrieve prediction for this location.");
    }
  }

  function renderPredictionResults(data) {
    const sName = data.display_name || (data.sensor ? (data.sensor.description || data.sensor.name) : "Location");
    const th = data.thresholds || {};

    if ($("resLocationName")) $("resLocationName").textContent = sName;
    if ($("travelTimeTitle")) $("travelTimeTitle").textContent = `${sName} at ${data.datetime}`;

    if ($("resPoint")) $("resPoint").textContent = `${Math.round(data.point)} people/hr`;
    if ($("resTimeLabel")) $("resTimeLabel").textContent = `Predicted for: ${data.datetime}`;

    if ($("resBand")) {
      $("resBand").textContent = `${Math.round(data.band_cal.lo)} – ${Math.round(data.band_cal.hi)} people/hr`;
    }

    if ($("resLevel")) {
      const badgeKind = data.level.toLowerCase();
      $("resLevel").innerHTML = `<span class="badge badge-${badgeKind}">${escapeHtml(data.sensory_label || data.level)}</span>`;
    }

    if ($("resAdviceNote")) {
      $("resAdviceNote").textContent = escapeHtml(data.sensory_advice || "");
    }

    if ($("resProbSentence")) {
      const prob = data.p75_exceed_prob_pct;
      const p75 = th.p75;

      $("resProbSentence").innerHTML = `
        Machine learning models estimate a <strong>${prob}% chance</strong> of high crowd / overstimulation exceeding <strong>${p75} people/hr</strong>.
      `;
    }
  }

  async function calculateFutureRoute() {
    if (!state.orig || !state.dest) {
      alert("Please enter a starting location to calculate the route.");
      return;
    }

    const mapContainer = $("futureMapContainer");
    const resultsContainer = $("futureRouteResults");

    if (mapContainer) mapContainer.style.display = "block";

    initPredictMap();

    const params = new URLSearchParams({
      orig_lat: state.orig.lat,
      orig_lon: state.orig.lon,
      dest_lat: state.dest.lat,
      dest_lon: state.dest.lon,
      mode: "ml",
    });
    if (state.datetime) params.append("datetime", state.datetime);

    try {
      const r = await fetch(`/api/route?${params.toString()}`);
      if (!r.ok) throw new Error("Route API error");
      const data = await r.json();

      displayRoutesOnMap(data.routes);
      renderRouteCards(data, resultsContainer);
    } catch (e) {
      alert("Failed to calculate route.");
    }
  }

  function initPredictMap() {
    if (predictMap) return;

    predictMap = L.map("predictMap", {
      center: [-37.815, 144.965],
      zoom: 14,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(predictMap);

    routeLayerGroup = L.layerGroup().addTo(predictMap);
  }

  function displayRoutesOnMap(routes) {
    if (!routeLayerGroup || !predictMap) return;
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
      predictMap.fitBounds(bounds, { padding: [30, 30] });
    }
  }

  function renderRouteCards(data, container) {
    if (!container) return;
    container.innerHTML = "";

    data.routes.forEach((r, idx) => {
      const card = document.createElement("div");
      card.className = `route-card ${r.is_least_crowded ? "least-crowded" : r.is_fastest ? "fastest" : ""}`;

      let tag = "";
      if (r.is_least_crowded) {
        tag = '<span class="badge badge-ok">⭐ RECOMMENDED CALM ROUTE AT TARGET TIME</span>';
      } else if (r.is_fastest) {
        tag = '<span class="badge badge-warn">SHORTEST ROUTE</span>';
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
        remarksHtml = "<p class='hint'>Quiet route predicted.</p>";
      }

      card.innerHTML = `
        <div class="card-header">
          ${tag}
          ${crowdBadge}
        </div>
        <div class="route-metrics">
          ⏱️ <strong>${r.duration_min} mins</strong> (${(r.distance_m / 1000).toFixed(2)} km)
        </div>
        <div>
          <strong>Predicted Sensory Remarks:</strong>
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
