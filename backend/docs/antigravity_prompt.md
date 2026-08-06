# Antigravity prompt: Melbourne CBD Pedestrian Crowd Map

Copy-paste this into Antigravity.

---

PROJECT: "Melbourne CBD Pedestrian Crowd Map" — a local Flask web app in
C:\Users\Swajay\Downloads\trade\indexp that I must evolve into a Google-Maps-like
view of Melbourne's CBD. You are the ONLY engineer; iterate in the browser.

CONTEXT — what already exists (do not rebuild):
- Offline experiment: src/config.py, src/features.py, src/models.py,
  src/data/load.py, src/data/synthetic.py, src/report.py (XGBoost 3.3 vs
  LightGBM 4.7, point + quantile q10/q50/q90 models, 3 horizons 1h/6h/24h).
- Trained models: results/real/*.model (24 files, incl. lgb_cpu_* and
  xgb_cpu_*). Load via src.models.load_booster + predict.
- Working Flask app: app/server.py (port 8000) with routes /, /models,
  /features, /help and APIs /api/sensors, /api/forecast, /api/history,
  /api/feed, /api/importance, /api/experiment. Service class in
  app/forecast_service.py (feature builder make_features_row, conformal
  calibration results/calibration.json, live per-minute feed with dedupe).
  Templates in app/templates, assets in app/static (vanilla JS + SVG charts,
  NO build step, NO external JS libs so far).
- Real data: data/raw/hourly_counts.csv — 1.61M hourly rows, 103 sensors,
  2024-08..2026-08; sensor_locations.csv has latitude, longitude,
  sensor_description, installation_date, status. All CC BY 4.0 (City of
  Melbourne Open Data). Live feed: past-hour per-minute counts (dedupe on
  location_id+sensing_datetime, sensors 67/68/69 known to duplicate).

NEW FEATURES TO BUILD (this is the main work):

A) MAP PAGE (/map) — Google-Maps-like, Melbourne CBD only:
   1. Use Leaflet 1.9 from CDN + OpenStreetMap tiles (no API keys).
      Center ~(-37.815, 144.965), zoom 14, maxBounds to the CBD.
   2. Markers for all 103 sensors, colored by current crowd level
      (LOW=green, MEDIUM=amber, HIGH=red) computed from the LIVE feed /
      latest hourly count vs the sensor's own p50/p75 percentile thresholds
      (thresholds computed from the hourly CSV at startup). Marker popup:
      name, street, current count, level, link to forecast.
   3. Search box ("Where do you want to go?"): geocode via Nominatim
      (https://nominatim.openstreetmap.org/search?q=...&format=json&limit=5,
      set a real User-Agent), ALSO match against sensor names/descriptions
      locally as fallback. Click-to-set origin and destination on the map.
   4. Route suggestions: call public OSRM
      (https://router.project-osrm.org/route/v1/foot/{lon1},{lat1};{lon2},{lat2}
      ?alternatives=3&steps=true&overview=full) with walking profile.
      For each candidate route: sample points along the polyline, find nearby
      sensors (<200m), annotate segments with crowd remarks, e.g.
      "High crowd: Bourke St Mall North (~450/hr at 10:00)".
      Show two recommendations: FASTEST route and LEAST-CROWDED route
      (lowest average/peak crowd level, only if detour < 25% extra time),
      with a crowd score (LOW/MEDIUM/HIGH) per route.
   5. All route/crowd logic can be RULE-BASED (no ML needed): per-sensor
      expected[hour][dow] historical table + weekend/holiday factors,
      level = count vs p50/p75. Use the ML models only where exact numbers
      matter (see B). Both implementations must exist behind one interface;
      flag --mode rule|ml on the server (default rule).

B) PREDICT PAGE (/predict) — separate from the map:
   - Search a location (geocode or pick a sensor from a dropdown), pick a
     date+time (datetime-local input, default now), click Predict.
   - Show: expected pedestrian count, 80% likely range (calibrated
     interval), crowd level, and a probability number "X% chance the count
     exceeds Y people" — estimate by interpolating the CDF through the
     trained q10/q50/q90 predictions (piecewise-linear), Y = the sensor's
     p75 threshold. Uses the LightGBM models + /api/predict endpoint.

C) KEEP all existing pages (/ , /models, /features, /help) working.

BACKEND (extend, don't rewrite):
   - Add routes to app/server.py: /map, /predict, /api/geocode,
     /api/route (returns routes + remarks + recommendation),
     /api/predict (lat/lon or sensor_id + ISO datetime),
     /api/map (sensors + lat/lon + current level).
   - New module app/crowd.py: rule-based crowd engine + routing helpers
     (thresholds from CSV, nearest-sensor lookup via simple lat/lon
     distance — no heavy GIS libs).
   - Network calls (Nominatim/OSRM) MUST have short timeouts and graceful
     fallbacks; if the network is down, routes degrade to "sensor-to-sensor
     straight line" with a visible note, and geocoding falls back to local
     sensor names.
   - Python 3.14, Flask 3.1, pandas 3.0. No new pip dependencies unless
     truly necessary (prefer urllib over requests).

ACCESSIBILITY (hard requirement — user is neurodivergent):
   - No animation, no flashing, no auto-refresh by default (opt-in only,
     clearly labelled).
   - High-contrast toggle + text-size A-/A+ controls (existing pattern in
     app/static/app.js + style.css, persisted in localStorage).
   - Plain language everywhere; explain every term inline ("80% range means
     8 out of 10 times the real count lands inside").
   - Colour is never the only cue: levels also have text labels.
   - All controls keyboard-accessible with visible focus.

CONSTRAINTS:
   - Keep files under 500 lines; follow existing code style (no comments
     unless needed, type hints, no speculative abstractions).
   - Never fake data or pass synthetic data off as real; label demo/offline
     states explicitly in the UI.
   - Attribution: "Map data © OpenStreetMap contributors" on the map.
   - Verify in the browser after every change (server runs
     python app/server.py --port 8000). Test the happy path AND the
     offline path (turn off network) before finishing.
   - Acceptance: (1) map shows 103 colored sensors; (2) typing a CBD
     destination returns 2 route options with crowd remarks; (3) /predict
     returns count + range + probability for any past or future hour;
     (4) every page works with keyboard only; (5) no console errors.
