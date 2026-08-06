/* Shared UI helpers (display preferences, fetch, badges) + dashboard logic. */
(function () {
  "use strict";

  /* ---------- display preferences (remembered in localStorage) ---------- */
  const prefs = (() => {
    try { return JSON.parse(localStorage.getItem("mpl_prefs") || "{}"); }
    catch (e) { return {}; }
  })();

  function savePrefs() {
    try { localStorage.setItem("mpl_prefs", JSON.stringify(prefs)); } catch (e) {}
  }

  const fontMinus = document.getElementById("fontMinus");
  const fontPlus = document.getElementById("fontPlus");
  const contrast = document.getElementById("contrastToggle");

  function applyContrast() {
    if (contrast) contrast.setAttribute("aria-pressed", String(!!prefs.hc));
    document.body.classList.toggle("hc", !!prefs.hc);
  }
  if (prefs.hc) applyContrast();
  if (prefs.size) document.body.classList.add("size-" + prefs.size);

  if (fontMinus) fontMinus.addEventListener("click", () => {
    document.body.classList.remove("size-20", "size-16");
    prefs.size = prefs.size === "16" ? undefined : "16";
    if (prefs.size) document.body.classList.add("size-" + prefs.size);
    savePrefs();
  });
  if (fontPlus) fontPlus.addEventListener("click", () => {
    document.body.classList.remove("size-16", "size-20");
    prefs.size = prefs.size === "20" ? undefined : "20";
    if (prefs.size) document.body.classList.add("size-" + prefs.size);
    savePrefs();
  });
  if (contrast) contrast.addEventListener("click", () => {
    prefs.hc = !prefs.hc;
    applyContrast();
    savePrefs();
  });

  /* ---------- tiny fetch wrapper + badges ---------- */
  const API = {
    async get(path) {
      const r = await fetch(path);
      if (!r.ok) throw new Error(path + " -> " + r.status);
      return r.json();
    },
    async post(path) {
      const r = await fetch(path, { method: "POST" });
      if (!r.ok) throw new Error(path + " -> " + r.status);
      return r.json();
    },
  };
  window.API = API;
  window.escapeHtml = function(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  };
  const $ = (id) => document.getElementById(id);
  const setBadge = (id, text, kind) => {
    const el = $(id);
    if (!el) return;
    el.textContent = text;
    el.className = "badge badge-" + (kind || "unknown");
  };

  /* =====================================================================
     Dashboard logic (pages that include #controls).
     ===================================================================== */
  const state = { sensors: [], loc: null, h: 1, fw: "lgb", cal: true };

  function markStatusBadges(data) {
    const last = new Date((data.data_as_of || "").replace(" ", "T") + "Z");
    const ageDays = isNaN(last) ? 0 : (Date.now() - last.getTime()) / 86400000;
    setBadge("dataStatus",
      ageDays > 2 ? "Hourly data is stale (last: " + data.data_as_of + ")"
                  : "Hourly data up to date (" + data.data_as_of + ")",
      ageDays > 2 ? "warn" : "ok");
    const f = data.feed || {};
    if (f.status === "live") setBadge("feedStatus", "Live feed: fresh (" + f.last_update + ")", "ok");
    else if (f.status && f.status.indexOf("demo") === 0)
      setBadge("feedStatus", "Live feed unavailable — demo feed in use", "warn");
    else setBadge("feedStatus", "Live feed: not loaded yet", "unknown");
  }

  async function loadSensors() {
    const data = await API.get("/api/sensors");
    state.sensors = data.sensors;
    markStatusBadges(data);
    const sel = $("sensor");
    const add = (label, list) => {
      if (!list.length) return;
      const og = document.createElement("optgroup");
      og.label = label;
      list.forEach(s => {
        const o = document.createElement("option");
        o.value = s.location_id;
        o.textContent = s.name + " (id " + s.location_id + ")";
        og.appendChild(o);
      });
      sel.appendChild(og);
    };
    add("New sensors — less history", state.sensors.filter(s => s.group === "short"));
    add("Established sensors — full history", state.sensors.filter(s => s.group === "long"));
    state.loc = state.sensors[0].location_id;
    sel.value = String(state.loc);
    sel.addEventListener("change", () => {
      state.loc = parseInt(sel.value, 10);
      showSensorHint();
      update();
    });
    showSensorHint();
    update();
  }

  function showSensorHint() {
    const s = state.sensors.find(x => x.location_id === state.loc);
    if (!s) return;
    const note = s.group === "short"
      ? "This sensor is new (installed " + (s.install_date || "recently") +
        "), so the model has less history for it — expect a wider range."
      : "Full history since " + s.history_start + ".";
    $("sensorHint").textContent = s.description ? s.description + ". " + note : note;
  }

  function setupPills() {
    document.querySelectorAll("#horizonPills .pill").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll("#horizonPills .pill").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      state.h = parseInt(b.dataset.h, 10);
      update();
    }));
    document.querySelectorAll("#modelPills .pill").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll("#modelPills .pill").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      state.fw = b.dataset.fw;
      update();
    }));
    if ($("calToggle")) $("calToggle").addEventListener("change", () => {
      state.cal = $("calToggle").checked;
      update();
    });
    if ($("refresh")) $("refresh").addEventListener("click", update);
    if ($("refreshFeed")) $("refreshFeed").addEventListener("click", () => refreshFeed($("demoFeed").checked));
    if ($("demoFeed")) $("demoFeed").addEventListener("change", () => refreshFeed($("demoFeed").checked));
    if ($("autoRefresh")) $("autoRefresh").addEventListener("change", () => {
      clearInterval(window.__autoTimer);
      if ($("autoRefresh").checked) {
        window.__autoTimer = setInterval(update, 5 * 60 * 1000);
        setBadge("dataStatus", "Auto-refresh is ON — page updates every 5 minutes", "warn");
      } else {
        setBadge("dataStatus", "Auto-refresh is off", "unknown");
      }
    });
  }

  async function refreshFeed(demo) {
    setBadge("feedStatus", "Fetching live feed…", "unknown");
    const f = await API.post("/api/feed" + (demo ? "?demo=1" : ""));
    $("feedDetail").textContent =
      (f.status === "live" ? "Live feed loaded" : "Demo feed loaded") +
      " — " + f.sensors + " sensors, " + f.rows + " hourly rows. Last check " + f.last_update + ".";
    $("feedHint").textContent = f.dups_removed > 0
      ? "Removed " + f.dups_removed + " duplicate rows (the known issue on sensors 67/68/69)."
      : "No duplicates found in this capture.";
    setBadge("feedStatus", f.status === "live" ? "Live feed: fresh" : "Demo feed (network unavailable)",
      f.status === "live" ? "ok" : "warn");
    await update();
  }

  function sensorName() {
    const s = state.sensors.find(x => x.location_id === state.loc);
    return s ? s.name : ("sensor " + state.loc);
  }

  async function update() {
    if (state.loc == null) return;
    setBadge("dataStatus", "Updating forecast…", "unknown");
    let data;
    try {
      data = await API.get("/api/forecast?location_id=" + state.loc);
    } catch (e) {
      setBadge("dataStatus", "Error: " + e.message, "err");
      return;
    }
    markStatusBadges({ data_as_of: data.sensor.history_end, feed: data.feed });
    const f = data.forecast[String(state.h)];
    const m = f[state.fw];
    const band = state.cal ? m.band_cal : m.band_raw;
    const madeAt = new Date(f.at);
    const target = new Date(madeAt.getTime() + state.h * 3600 * 1000);
    const fmtT = d => d.toLocaleString("en-AU", {
      weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"});

    $("pointPred").textContent = Math.round(m.point).toLocaleString();
    $("pointWhen").textContent = "at " + fmtT(target) + " (forecast made at " + fmtT(madeAt) + ")";
    $("bandPred").textContent = Math.round(band.lo).toLocaleString() + " – " +
      Math.round(band.hi).toLocaleString() + " people";
    $("bandNote").textContent = state.cal
      ? "Calibrated to 80% coverage. Without calibration: " +
        Math.round(m.band_raw.lo).toLocaleString() + " – " + Math.round(m.band_raw.hi).toLocaleString() + "."
      : "Raw model band (measured coverage ~74% on real data).";
    $("sentence").textContent =
      "In plain words: for " + sensorName() + " we expect about " + Math.round(m.point).toLocaleString() +
      " people in the hour starting " + fmtT(target) + ", and there is an 80% chance the real count is between " +
      Math.round(band.lo).toLocaleString() + " and " + Math.round(band.hi).toLocaleString() + ".";
    setBadge("dataStatus", "Forecast updated at " + fmtT(madeAt), "ok");
    drawChart(data);
  }

  /* ---------- minimal SVG chart (no external libraries, offline-safe) ---------- */
  function drawChart(data) {
    const f = data.forecast[String(state.h)];
    const m = f[state.fw];
    const band = state.cal ? m.band_cal : m.band_raw;
    const hist = data.history || [];
    const N = hist.length;
    const W = 980, H = 320, padL = 56, padR = 16, padT = 12, padB = 28;
    const x = i => padL + (i / Math.max(N - 1, 1)) * (W - padL - padR);
    const vmax = Math.max(...hist.map(p => p.v), m.point, band.hi, 1) * 1.1;
    const y = v => padT + (1 - v / vmax) * (H - padT - padB);
    const fx = x(N - 1) + (x(1) - x(0)) * state.h;

    let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + W + " " + H + '">';
    for (let g = 0; g <= 4; g++) {
      const v = vmax * g / 4, yy = y(v);
      svg += '<line x1="' + padL + '" y1="' + yy + '" x2="' + (W - padR) + '" y2="' + yy +
        '" stroke="currentColor" stroke-opacity="0.15"/>' +
        '<text x="' + (padL - 8) + '" y="' + (yy + 4) + '" text-anchor="end">' + Math.round(v) + "</text>";
    }
    const path = hist.map((p, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p.v).toFixed(1)).join("");
    svg += '<path d="' + path + '" fill="none" stroke="currentColor" stroke-width="2.5"/>';
    const bw = Math.max(x(1) - x(0), 24);
    svg += '<rect x="' + fx.toFixed(1) + '" y="' + y(band.hi).toFixed(1) + '" width="' + bw.toFixed(1) +
      '" height="' + Math.max(y(band.lo) - y(band.hi), 2).toFixed(1) +
      '" fill="var(--band-fill)" stroke="currentColor" stroke-opacity="0.35"/>' +
      '<circle cx="' + fx.toFixed(1) + '" cy="' + y(m.point).toFixed(1) + '" r="5" fill="currentColor"/>';
    for (let i = 0; i < N; i += 24) {
      const d = new Date(hist[i].t);
      svg += '<text x="' + x(i).toFixed(1) + '" y="' + (H - 8) + '" text-anchor="middle">' +
        d.toLocaleDateString("en-AU", { day: "numeric", month: "short" }) + "</text>";
    }
    svg += '<text x="' + fx.toFixed(1) + '" y="' + (H - 8) + '" text-anchor="middle" font-weight="bold">forecast</text>';
    svg += "</svg>";
    $("chart").innerHTML = svg;
  }

  /* ---------- boot ---------- */
  if ($("controls")) {
    setupPills();
    loadSensors().catch(e => {
      $("oneLiner").textContent = "Could not load the forecast data. Is the server running with results/ present?";
      setBadge("dataStatus", "Error: " + e.message, "err");
    });
  }
})();
