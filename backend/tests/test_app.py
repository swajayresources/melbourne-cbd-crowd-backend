"""App service tests. Run:  python tests/test_app.py

Verifies that the app's single-row feature builder is value-identical to the
training pipeline, calibration widens bands, and forecast endpoints respond.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg
from src.features import build_features
from src.data.load import get_counts
from src.models import location_code_map
from app.forecast_service import Service, make_features_row

QUIET = True


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(1)


def test_feature_consistency():
    counts = get_counts("real")
    codes = location_code_map(counts)
    feats, _ = build_features(counts, codes)
    sensor = int(counts["location_id"].iloc[0])
    sub = feats[feats.location_id == sensor]
    at = sub.iloc[len(sub) // 2]["datetime"]  # mid-series hour that exists
    row = make_features_row(counts, codes, sensor, at, feed_hourly=None)
    train_row = sub[sub["datetime"] == at].iloc[0]
    bad = []
    for c in cfg.FEATURES:
        a, b = row.iloc[0][c], train_row[c]
        if not (pd.isna(a) and pd.isna(b)) and not np.isclose(a, b, rtol=1e-4, atol=1e-4):
            bad.append((c, a, b))
    check("app row == training row (no feed)", not bad, str(bad[:3]))


def test_calibration_and_forecast():
    svc = Service("real")
    svc.refresh_feed(use_demo=True)
    f = svc.forecast(int(svc.counts["location_id"].iloc[0]))
    h1 = f["1"]["lgb"]
    check("calibration widens the band", h1["band_cal"]["hi"] - h1["band_cal"]["lo"]
          >= h1["band_raw"]["hi"] - h1["band_raw"]["lo"])
    check("band contains median", h1["band_cal"]["lo"] <= h1["q50"] <= h1["band_cal"]["hi"])
    check("point non-negative", h1["point"] >= 0)
    check("forecast has all horizons", all(str(h) in f for h in cfg.HORIZONS))
    check("feed dedupe meta present", svc.feed_meta["dups_removed"] >= 0)
    try:
        svc.forecast(99999)
        check("unknown sensor raises", False)
    except KeyError:
        check("unknown sensor raises", True)


if __name__ == "__main__":
    test_feature_consistency()
    test_calibration_and_forecast()
    print("ALL APP TESTS PASSED")
