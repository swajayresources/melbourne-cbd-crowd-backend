"""Feature engineering - identical for both frameworks (fair comparison).

All features at row t use information available strictly at-or-before t, so
predictions for y(t+h) never leak the future. Features:

  calendar : hour, dow, month, is_weekend, is_public_holiday (Australia/VIC)
  lags     : count at t-1h, t-24h, t-168h        (NaN early in a sensor's life;
             LightGBM & XGBoost route NaNs natively -> model learns sparsity)
  rolling  : trailing means over 24h / 7d / 28d, min_periods=1 (no NaN)
  identity : location_code (categorical)

Targets  : count at t+1, t+6, t+24 (one per horizon, direct multi-horizon).
"""
from __future__ import annotations

import pandas as pd
import holidays as py_holidays

from . import config as cfg


def _calendar_features(dt: pd.Series) -> pd.DataFrame:
    dt = pd.DatetimeIndex(dt)
    vic = py_holidays.AU(prov="VIC", years=range(dt.year.min(), dt.year.max() + 1))
    is_hol = pd.Series(dt.isin(vic), index=dt, name="is_public_holiday")
    return pd.DataFrame({
        "hour": dt.hour,
        "dow": dt.dayofweek,
        "month": dt.month,
        "is_weekend": (dt.dayofweek >= 5).astype(int),
        "is_public_holiday": is_hol.astype(int).to_numpy(),
    }, index=dt)


def build_features(counts: pd.DataFrame, location_codes: pd.Series | None = None) -> pd.DataFrame:
    """counts: [location_id, datetime, count] sorted by (location_id, datetime).
    Returns one row per (location, hour) with features + target_{1,6,24}."""
    df = counts.sort_values(["location_id", "datetime"]).reset_index(drop=True)
    g = df.groupby("location_id", sort=False)["count"]

    feats = pd.DataFrame(index=df.index)
    feats["datetime"] = df["datetime"]
    feats["location_id"] = df["location_id"]

    cal = _calendar_features(df["datetime"])
    for c in cal.columns:
        feats[c] = cal[c].to_numpy()

    for lag in (1, 24, 168):
        feats[f"lag_{lag}"] = g.shift(lag).astype("float32")

    for win, name in ((24, "roll_24_mean"), (168, "roll_168_mean"), (672, "roll_672_mean")):
        feats[name] = g.transform(lambda s: s.rolling(win, min_periods=1).mean()).astype("float32")

    if location_codes is None:
        codes, _ = pd.factorize(pd.unique(df["location_id"]))
        location_codes = pd.Series(codes, index=pd.unique(df["location_id"]))
    feats["location_code"] = feats["location_id"].map(location_codes).astype("int32")

    for h in cfg.HORIZONS:
        feats[f"{cfg.TARGET_PREFIX}{h}"] = g.shift(-h).astype("float32")

    feats = feats.dropna(subset=[f"{cfg.TARGET_PREFIX}{h}" for h in cfg.HORIZONS])
    return feats.reset_index(drop=True), location_codes


def make_splits(feats: pd.DataFrame):
    """Chronological, time-respecting split. No shuffling anywhere."""
    val_start = pd.Timestamp(cfg.VAL_START)
    test_start = pd.Timestamp(cfg.TEST_START)
    t = feats["datetime"]
    train = feats[t < val_start]
    val = feats[(t >= val_start) & (t < test_start)]
    test = feats[t >= test_start]
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:5s}: {len(part):>9,} rows  {part['datetime'].min()} .. {part['datetime'].max()}")
    return train, val, test
