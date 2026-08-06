"""Tests for Flask app legacy endpoints (/map, /predict, /api/map, /api/geocode, /api/route, /api/predict).
Run: python tests/test_server_endpoints.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app

def check(name: str, cond: bool, detail: str = ""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(1)

def test_endpoints():
    app = create_app("testing")
    client = app.test_client()

    # HTML pages
    r_index = client.get("/")
    check("GET / returns 200", r_index.status_code == 200)

    r_map = client.get("/map")
    check("GET /map returns 200", r_map.status_code == 200)

    r_predict = client.get("/predict")
    check("GET /predict returns 200", r_predict.status_code == 200)

    # API Map
    r_api_map = client.get("/api/map")
    check("GET /api/map returns 200", r_api_map.status_code == 200)
    data_map = r_api_map.get_json()
    check("api/map returns 103 sensors", len(data_map["sensors"]) >= 100)
    first_s = data_map["sensors"][0]
    check("sensor has level", first_s["level"] in ("LOW", "MEDIUM", "HIGH"))

    # API Geocode
    r_geo = client.get("/api/geocode?q=Bourke")
    check("GET /api/geocode returns 200", r_geo.status_code == 200)
    data_geo = r_geo.get_json()
    check("api/geocode returns results", len(data_geo["results"]) > 0)

    # API Route
    r_route = client.get("/api/route?orig_lat=-37.813&orig_lon=144.963&dest_lat=-37.818&dest_lon=144.968")
    check("GET /api/route returns 200", r_route.status_code == 200)
    data_route = r_route.get_json()
    check("api/route returns route list", len(data_route["routes"]) > 0)
    check("api/route contains recommendation", "fastest_id" in data_route["recommendation"])

    # API Predict
    r_pred = client.get("/api/predict?location_id=1")
    check("GET /api/predict returns 200", r_pred.status_code == 200)
    data_pred = r_pred.get_json()
    check("api/predict returns point prediction", data_pred["point"] >= 0)
    check("api/predict returns 80% band", "lo" in data_pred["band_cal"] and "hi" in data_pred["band_cal"])
    check("api/predict returns probability pct", 0.0 <= data_pred["p75_exceed_prob_pct"] <= 100.0)

if __name__ == "__main__":
    test_endpoints()
    print("ALL SERVER ENDPOINT TESTS PASSED")
