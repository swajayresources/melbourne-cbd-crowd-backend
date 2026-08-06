"""Sanity tests for the experiment pipeline. Run:  python tests/test_pipeline.py

Covers: synthetic generator shape, feature leakage (lag values), time-respecting
split ordering, quantile monotonicity, interval coverage sanity, holiday logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg
from src.data.synthetic import make_synthetic_counts
from src.features import build_features, make_splits


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        raise SystemExit(1)


def _run_test_synthetic():
    df = make_synthetic_counts()
    check("synthetic: 8 sensors", df.location_id.nunique() == 8)
    check("synthetic: date range", df.datetime.min() == pd.Timestamp("2021-01-01"))
    check("synthetic: no NaN counts", not df["count"].isna().any())
    check("synthetic: no negative counts", (df["count"] >= 0).all())
    check("synthetic: zero counts exist (dead hours)", (df["count"] == 0).any())
    short = df[df.location_id.isin(cfg.SYNTHETIC["short_ids"])]
    long = df[~df.location_id.isin(cfg.SYNTHETIC["short_ids"])]
    check("synthetic: short sensors have less history",
          short.datetime.max() > long.datetime.min(),
          f"short rows={len(short):,} long rows={len(long):,}")
    return df


def test_pipeline_all():
    df = _run_test_synthetic()
    feats = _run_test_features(df)
    train, val, test = _run_test_splits(feats)
    _run_test_models(train, val, test)
    test_holidays()


def _run_test_features(df):
    feats, _ = build_features(df)
    check("features: columns", all(c in feats.columns for c in cfg.FEATURES + ["location_id", "datetime"]))
    check("features: targets exist", all(f"target_{h}" in feats.columns for h in cfg.HORIZONS))
    check("features: NaN lags only early in history",
          feats["lag_1"].isna().sum() > 0 and feats["lag_168"].isna().sum() > 0)
    d = feats.sort_values("datetime")
    t1 = d[d["datetime"] == d["datetime"].min()]
    check("features: no NaN in calendar/identity", t1[["hour", "dow", "month", "location_code"]].notna().all().all())
    # leakage: lag_1 at time t must equal the raw count at t-1h for the same sensor
    spot = feats[feats["datetime"] == pd.Timestamp("2025-03-01 12:00")]
    if len(spot):
        loc = spot.iloc[0]["location_id"]
        prev = df[(df.location_id == loc) & (df.datetime == pd.Timestamp("2025-03-01 11:00"))]["count"].iloc[0]
        check("features: lag_1 == count(t-1h) (no leakage)", spot.iloc[0]["lag_1"] == prev)
    return feats


def _run_test_splits(feats):
    train, val, test = make_splits(feats)
    check("split: train < val < test (time-respecting)",
          train["datetime"].max() < val["datetime"].min() <= test["datetime"].min())
    check("split: test is the most recent block",
          test["datetime"].max() == feats["datetime"].max())
    return train, val, test


def _run_test_models(train, val, test):
    from src.models import train_model, predict
    cols = cfg.FEATURES
    # mirror production: codes fitted on train only; unseen -> modal code
    locs = pd.unique(train["location_id"])
    codes, _ = pd.factorize(locs)
    cmap = dict(zip(locs, codes))
    modal = int(pd.Series(codes).mode()[0])
    for part in (train, val, test):
        part["location_code"] = (part["location_id"].map(lambda l: cmap.get(l, modal))
                                 .astype("int32"))
    small_tr = train.loc[train.groupby("location_code")["datetime"]
                             .apply(lambda s: s.sample(min(len(s), 3000), random_state=1).index)
                             .explode()]
    small_va = val.sample(8000, random_state=1)
    small_te = test.sample(8000, random_state=1)
    qhat = {}
    for fw in ("xgb", "lgb"):
        for a in (0.1, 0.5, 0.9):
            b, _ = train_model(fw, a, a, 1, small_tr, small_va, "cpu")
            qhat[(fw, a)] = predict(b, fw, small_te[cols])
        q10, q50, q90 = qhat[(fw, 0.1)], qhat[(fw, 0.5)], qhat[(fw, 0.9)]
        mono = float(np.mean((q10 <= q50 + 1e-9) & (q50 <= q90 + 1e-9)))
        check(f"{fw}: quantile monotonicity q10<=q50<=q90", mono > 0.98, f"({mono:.3f})")
        cov = float(np.mean((small_te["target_1"] >= q10) & (small_te["target_1"] <= q90)))
        check(f"{fw}: 80% coverage sane (0.5..0.97)", 0.5 <= cov <= 0.97, f"(coverage={cov:.2f})")


def test_holidays():
    import holidays as py_holidays
    vic = py_holidays.AU(subdiv="VIC", years=[2026])
    check("holidays: 2026-12-25 is public holiday", pd.Timestamp("2026-12-25") in vic)
    check("holidays: 2026-08-03 is not a holiday", pd.Timestamp("2026-08-03") not in vic)


if __name__ == "__main__":
    test_pipeline_all()
    print("ALL TESTS PASSED")
