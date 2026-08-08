/* Features page: renders the importance comparison from /api/importance. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  API.get("/api/importance").then(d => {
    const tbl = d.table.split("\n").filter(l => l.trim().startsWith("|"));
    const rows = tbl.map(l =>
      l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim()));
    let h = "<table><thead><tr>" + rows[0].map(c => "<th>" + window.escapeHtml(c) + "</th>").join("") + "</tr></thead><tbody>";
    rows.slice(2).forEach(r => {
      h += "<tr>" + r.map(c => "<td>" + window.escapeHtml(c) + "</td>").join("") + "</tr>";
    });
    $("impTable").innerHTML = h + "</tbody></table>";
    $("impStats").textContent =
      "Rank correlation between the two models' importance: " + d.stats.rho.toFixed(2) +
      ". Top-10 feature overlap: " + d.stats.top10_jaccard + ".";
  }).catch(e => {
    $("impTable").textContent = "Error: " + e.message;
  });
})();
