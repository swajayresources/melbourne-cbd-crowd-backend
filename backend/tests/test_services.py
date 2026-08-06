"""Unit test suite for modular backend services."""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.crowd_service import CrowdService
from app.services.routing_service import RoutingService
from app.services.forecast_service import ForecastService
from app.services.auth_service import AuthService
from app.services.cache_service import CacheService
from app.services.db_service import DatabaseService


def check(name: str, cond: bool, detail: str = ""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(1)


def test_services():
    print("--- Testing CrowdService ---")
    crowd_svc = CrowdService()
    check("crowd_svc loads locations", len(crowd_svc.locations_df) > 0)
    th = crowd_svc.get_thresholds(1)
    check("thresholds contain p50 and p75", "p50" in th and "p75" in th)
    lvl, label, advice = crowd_svc.classify_sensory_level(1, th["p50"] - 1.0)
    check("below p50 is LOW", lvl == "LOW")

    print("--- Testing RoutingService ---")
    routing_svc = RoutingService(crowd_svc.engine)
    geo_res = routing_svc.geocode("Bourke")
    check("geocoding returns results", len(geo_res) > 0)
    routes, is_fallback = routing_svc.fetch_routes(-37.813, 144.963, -37.818, 144.968)
    check("fetch_routes returns candidate routes", len(routes) > 0)

    print("--- Testing ForecastService ---")
    forecast_svc = ForecastService("real")
    sensors = forecast_svc.sensor_list()
    check("forecast_svc lists sensors", len(sensors) > 0)

    print("--- Testing AuthService ---")
    auth_svc = AuthService(secret_key="test-secret")
    session_hash = auth_svc.hash_public_key("sample_public_key_ecdsa_p256")
    check("session hash derived", len(session_hash) == 32)
    token = auth_svc.create_anonymous_token(session_hash)
    check("token generated", token.count(".") == 2)
    payload = auth_svc.verify_token(token)
    check("token payload valid", payload is not None and payload["session_hash"] == session_hash)

    print("--- Testing CacheService ---")
    cache_svc = CacheService()
    cache_svc.set("test_key", {"status": "ok"}, ttl=10)
    val = cache_svc.get("test_key")
    check("cache get returns stored value", val == {"status": "ok"})

    print("--- Testing DatabaseService ---")
    db_svc = DatabaseService()
    saved = db_svc.save_user_preferences(session_hash, {"high_contrast": True, "text_scale": 1.2})
    check("user preferences saved", saved)
    prefs = db_svc.get_user_preferences(session_hash)
    check("user preferences retrieved", prefs.get("high_contrast") is True)


if __name__ == "__main__":
    test_services()
    print("ALL SERVICE TESTS PASSED")
