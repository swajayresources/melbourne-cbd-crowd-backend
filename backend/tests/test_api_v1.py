"""Integration test suite for API V1 endpoints."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app


def check(name: str, cond: bool, detail: str = ""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(1)


def test_api_v1():
    app = create_app("testing")
    client = app.test_client()

    print("--- Testing /api/v1/sensors ---")
    r_sensors = client.get("/api/v1/sensors")
    check("GET /api/v1/sensors returns 200", r_sensors.status_code == 200)
    data_sensors = r_sensors.get_json()
    check("sensors list returned", len(data_sensors["sensors"]) > 0)

    print("--- Testing /api/v1/map ---")
    r_map = client.get("/api/v1/map")
    check("GET /api/v1/map returns 200", r_map.status_code == 200)
    data_map = r_map.get_json()
    check("103 map sensors returned", len(data_map["sensors"]) >= 100)

    print("--- Testing /api/v1/geocode ---")
    r_geo = client.get("/api/v1/geocode?q=Elizabeth")
    check("GET /api/v1/geocode returns 200", r_geo.status_code == 200)
    data_geo = r_geo.get_json()
    check("geocode results returned", len(data_geo["results"]) > 0)

    print("--- Testing /api/v1/route ---")
    r_route = client.get("/api/v1/route?orig_lat=-37.813&orig_lon=144.963&dest_lat=-37.818&dest_lon=144.968")
    check("GET /api/v1/route returns 200", r_route.status_code == 200)
    data_route = r_route.get_json()
    check("routes returned", len(data_route["routes"]) > 0)
    check("recommendations present", "fastest_id" in data_route["recommendation"])

    print("--- Testing /api/v1/predict ---")
    r_pred = client.get("/api/v1/predict?location_id=1")
    check("GET /api/v1/predict returns 200", r_pred.status_code == 200)
    data_pred = r_pred.get_json()
    check("prediction point returned", data_pred["point"] >= 0)
    check("calibrated 80% band present", "lo" in data_pred["band_cal"])

    print("--- Testing /api/v1/auth/session ---")
    r_auth = client.post("/api/v1/auth/session", json={"public_key": "ecdsa_p256_dummy_key"})
    check("POST /api/v1/auth/session returns 200", r_auth.status_code == 200)
    data_auth = r_auth.get_json()
    check("token returned", "token" in data_auth)
    check("compliance header present", data_auth["compliance"] == "APP_1988_ZERO_PII")

    token = data_auth["token"]

    print("--- Testing /api/v1/user/prefs ---")
    r_prefs_save = client.post(
        "/api/v1/user/prefs",
        headers={"Authorization": f"Bearer {token}"},
        json={"preferences": {"high_contrast": True, "text_scale": 1.25}},
    )
    check("POST /api/v1/user/prefs returns 200", r_prefs_save.status_code == 200)

    r_prefs_get = client.get(
        "/api/v1/user/prefs",
        headers={"Authorization": f"Bearer {token}"},
    )
    check("GET /api/v1/user/prefs returns 200", r_prefs_get.status_code == 200)
    data_prefs = r_prefs_get.get_json()
    check("saved preferences retrieved", data_prefs["preferences"].get("high_contrast") is True)


if __name__ == "__main__":
    test_api_v1()
    print("ALL API V1 TESTS PASSED")
