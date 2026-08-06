"""Unit tests for app/crowd.py crowd engine and routing logic.
Run: python tests/test_crowd.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.crowd import (
    CrowdEngine,
    geocode_location,
    fetch_osrm_route,
    evaluate_route_crowds,
    estimate_exceedance_probability,
    decode_polyline,
)

def check(name: str, cond: bool, detail: str = ""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(1)

def test_crowd_engine():
    engine = CrowdEngine()
    check("crowd engine loads locations", len(engine.locations_df) > 0)
    check("thresholds loaded", len(engine.thresholds) > 0)
    
    first_id = int(engine.locations_df.index[0])
    th = engine.get_thresholds(first_id)
    check("thresholds contain p50 and p75", "p50" in th and "p75" in th)
    check("p50 <= p75", th["p50"] <= th["p75"])
    
    lvl = engine.classify_level(first_id, th["p50"] - 1.0)
    check("below p50 is LOW", lvl == "LOW")
    lvl_mid = engine.classify_level(first_id, (th["p50"] + th["p75"]) / 2)
    check("between p50 and p75 is MEDIUM", lvl_mid == "MEDIUM")
    lvl_hi = engine.classify_level(first_id, th["p75"] + 10.0)
    check("above p75 is HIGH", lvl_hi == "HIGH")

    # Rule prediction
    dt = pd.Timestamp("2026-08-04 10:00:00")
    pred = engine.predict_rule_count(first_id, dt)
    check("rule count non-negative", pred >= 0.0)

def test_nearest_sensor():
    engine = CrowdEngine()
    nearest = engine.find_nearest_sensor(-37.815, 144.965)
    check("finds nearest sensor ID", nearest is not None and nearest in engine.locations_df.index)

def test_geocoding_fallback():
    engine = CrowdEngine()
    res = geocode_location("Bourke St", engine)
    check("geocoding returns results", len(res) > 0)
    check("geocoding item contains lat/lon", "latitude" in res[0] and "longitude" in res[0])

def test_osrm_and_evaluation():
    engine = CrowdEngine()
    orig = (-37.813, 144.963)
    dest = (-37.818, 144.968)
    routes, is_fallback = fetch_osrm_route(orig[0], orig[1], dest[0], dest[1])
    check("fetch_osrm_route returns routes", len(routes) > 0)
    
    eval_res = evaluate_route_crowds(routes, pd.Timestamp.now(), engine, mode="rule")
    check("evaluates routes", len(eval_res["routes"]) > 0)
    check("recommendation present", "fastest_id" in eval_res["recommendation"])

def test_cdf_interpolation():
    prob1 = estimate_exceedance_probability(q10=100, q50=200, q90=400, p75_threshold=200)
    check("CDF at q50 is ~50%", abs(prob1 - 50.0) < 1.0, f"got {prob1}%")
    
    prob_hi = estimate_exceedance_probability(q10=100, q50=200, q90=400, p75_threshold=50)
    check("CDF below q10 is >90% exceedance", prob_hi > 90.0, f"got {prob_hi}%")
    
    prob_lo = estimate_exceedance_probability(q10=100, q50=200, q90=400, p75_threshold=500)
    check("CDF above q90 is <10% exceedance", prob_lo < 10.0, f"got {prob_lo}%")

if __name__ == "__main__":
    test_crowd_engine()
    test_nearest_sensor()
    test_geocoding_fallback()
    test_osrm_and_evaluation()
    test_cdf_interpolation()
    print("ALL CROWD TESTS PASSED")
