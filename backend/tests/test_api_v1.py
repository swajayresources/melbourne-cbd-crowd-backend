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

    print("--- Testing /api/v1/predict with Modal Offloading ---")
    import unittest.mock
    import json
    app.config["MODAL_API_URL"] = "https://dummy-modal-app.modal.run"
    mock_modal_response = json.dumps({
        "location_id": 1,
        "point": 850.0,
        "q50": 850.0,
        "band_cal": {"lo": 700.0, "hi": 1000.0},
        "band_raw": {"lo": 650.0, "hi": 1050.0},
        "level": "LOW",
        "sensory_label": "Calm (Quiet)",
        "sensory_advice": "Low crowd density",
        "p75_exceed_prob_pct": 15.0,
        "thresholds": {"p50": 888.0, "p75": 1989.0},
        "mode": "modal_serverless",
        "provider": "modal_serverless",
    }).encode("utf-8")
    
    with unittest.mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_cm = unittest.mock.MagicMock()
        mock_cm.__enter__.return_value.read.return_value = mock_modal_response
        mock_urlopen.return_value = mock_cm
        r_pred_modal = client.get("/api/v1/predict?location_id=1")
        check("GET /api/v1/predict with Modal returns 200", r_pred_modal.status_code == 200)
        data_pred_modal = r_pred_modal.get_json()
        check("Modal prediction mode returned", data_pred_modal.get("mode") == "modal_serverless")
        check("Modal prediction probability returned", data_pred_modal.get("p75_exceed_prob_pct") == 15.0)
    app.config["MODAL_API_URL"] = ""

    print("--- Testing /api/v1/predict with Modal ONNX Inference ---")
    import unittest.mock
    import json
    app.config["MODAL_ML_API_URL"] = "https://dummy-modal-onnx.modal.run"
    mock_onnx_response = json.dumps({
        "forecast": {
            "1": {"point": 2109.0, "q50": 2075.7,
                  "band_raw": {"lo": 1597.1, "hi": 2527.2},
                  "band_cal": {"lo": 1551.9, "hi": 2572.4}},
            "6": {"point": 1542.0, "q50": 1481.8,
                  "band_raw": {"lo": 1032.9, "hi": 1912.3},
                  "band_cal": {"lo": 947.9, "hi": 1997.3}},
            "24": {"point": 2019.5, "q50": 1893.1,
                   "band_raw": {"lo": 1594.8, "hi": 2363.0},
                   "band_cal": {"lo": 1454.3, "hi": 2503.5}},
        },
        "mode": "modal_onnx",
        "provider": "modal_onnx",
    }).encode("utf-8")

    with unittest.mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_cm = unittest.mock.MagicMock()
        mock_cm.__enter__.return_value.read.return_value = mock_onnx_response
        mock_urlopen.return_value = mock_cm
        r_pred_onnx = client.get("/api/v1/predict?location_id=1")
        check("GET /api/v1/predict with Modal ONNX returns 200", r_pred_onnx.status_code == 200)
        data_pred_onnx = r_pred_onnx.get_json()
        check("ONNX prediction mode returned", data_pred_onnx.get("mode") == "modal_onnx")
        check("ONNX forecast present", "forecast" in data_pred_onnx)
        check("ONNX point returned", data_pred_onnx.get("point") == 2109.0)
    app.config["MODAL_ML_API_URL"] = ""

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
