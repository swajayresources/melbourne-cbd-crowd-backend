/* Models page: renders the experiment head-to-head tables from /api/experiment. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  function mdTableToHtml(md) {
    const lines = md.split("\n").filter(l => l.trim().startsWith("|"));
    if (!lines.length) return md;
    const rows = lines.map(l =>
      l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim()));
    let h = "<table><thead><tr>" + rows[0].map(c => "<th>" + window.escapeHtml(c) + "</th>").join("") + "</tr></thead><tbody>";
    rows.slice(2).forEach(r => {
      h += "<tr>" + r.map(c => "<td>" + window.escapeHtml(c) + "</td>").join("") + "</tr>";
    });
    return h + "</tbody></table>";
  }

  function envHtml(meta) {
    return "<table><tbody>" +
      "<tr><td>Python</td><td>" + window.escapeHtml(meta.python + " | " + meta.platform) + "</td></tr>" +
      "<tr><td>Hardware</td><td>" + window.escapeHtml(meta.cpus + " CPUs | " + meta.gpu) + "</td></tr>" +
      "<tr><td>Libraries</td><td>" + window.escapeHtml("xgboost " + meta.xgboost + " | lightgbm " + meta.lightgbm) + "</td></tr>" +
      "<tr><td>Data</td><td>" + window.escapeHtml(meta.data + " | " + meta.n_rows.toLocaleString() + " rows | " + meta.n_sensors + " sensors") + "</td></tr>" +
      "</tbody></table>";
  }

  function pointHtml(d) {
    let h = "<table><thead><tr><th>Horizon</th><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE %</th><th>WMAPE %</th></tr></thead><tbody>";
    [1, 6, 24].forEach(hh => {
      ["xgb", "lgb"].forEach(fw => {
        const m = d["point_" + hh][fw];
        if (!m) return;
        h += "<tr><td>" + hh + "h</td><td>" + fw.toUpperCase() + "</td><td>" + m.mae.toFixed(2) +
          "</td><td>" + m.rmse.toFixed(2) + "</td><td>" + (m.mape * 100).toFixed(2) +
          "</td><td>" + (m.wmape === undefined ? "-" : (m.wmape * 100).toFixed(2)) + "</td></tr>";
      });
    });
    return h + "</tbody></table>";
  }

  function intervalHtml(d) {
    let h = "<table><thead><tr><th>Horizon</th><th>Model</th><th>80% interval pinball</th><th>Coverage %</th><th>Mean width</th></tr></thead><tbody>";
    [1, 6, 24].forEach(hh => {
      const row = d["interval_" + hh] || {};
      ["xgb", "lgb"].forEach(fw => {
        const m = row[fw];
        if (!m) return;
        h += "<tr><td>" + hh + "h</td><td>" + fw.toUpperCase() + "</td><td>" + m.pinball.toFixed(2) +
          "</td><td>" + (m.coverage * 100).toFixed(1) + "</td><td>" + m.width.toFixed(1) + "</td></tr>";
      });
    });
    return h + "</tbody></table>";
  }

  function trainHtml(d) {
    const t = d.train_totals || {};
    const f = x => x === undefined ? "-" : Math.round(x) + " s";
    let h = "<table><thead><tr><th>Training device</th><th>XGBoost total (12 models)</th><th>LightGBM total (12 models)</th></tr></thead><tbody>";
    h += "<tr><td>CPU</td><td>" + f(t.xgb_cpu) + "</td><td>" + f(t.lgb_cpu) + "</td></tr>";
    h += "<tr><td>GPU (CUDA)</td><td>" + f(t.xgb_gpu) + (t.xgb_cpu && t.xgb_gpu ? " (" + (t.xgb_cpu / t.xgb_gpu).toFixed(1) + "x speedup)" : "") +
      "</td><td>N/A — official pip wheels have no GPU build</td></tr></tbody></table>";
    return h;
  }

  function serveHtml(d) {
    let h = "<table><thead><tr><th>Horizon</th><th>Model</th><th>Single request (ms)</th><th>File size (MB)</th></tr></thead><tbody>";
    [1, 6, 24].forEach(hh => {
      ["xgb", "lgb"].forEach(fw => {
        const m = d["point_" + hh][fw];
        if (!m) return;
        h += "<tr><td>" + hh + "h</td><td>" + fw.toUpperCase() + "</td><td>" + m.single_ms.toFixed(2) +
          "</td><td>" + m.file_mb.toFixed(2) + "</td></tr>";
      });
    });
    return h + "</tbody></table>";
  }

  function calHtml(d) {
    let h = "<table><thead><tr><th>Model</th><th>Horizon</th><th>Band adjustment (pedestrians)</th></tr></thead><tbody>";
    Object.entries(d.calibration || {}).forEach(([k, v]) => {
      const [fw, hh] = k.split("_");
      h += "<tr><td>" + fw.toUpperCase() + "</td><td>" + hh + "h</td><td>+" + v.toFixed(1) +
        " / -" + v.toFixed(1) + "</td></tr>";
    });
    return h + "</tbody></table>";
  }

  API.get("/api/experiment").then(d => {
    $("env").innerHTML = envHtml(d.meta);
    $("pointTable").innerHTML = pointHtml(d);
    $("intervalTable").innerHTML = intervalHtml(d);
    $("trainTable").innerHTML = trainHtml(d);
    $("serveTable").innerHTML = serveHtml(d);
    $("calTable").innerHTML = calHtml(d);
  }).catch(e => {
    $("env").textContent = "Error: " + e.message;
  });
})();
