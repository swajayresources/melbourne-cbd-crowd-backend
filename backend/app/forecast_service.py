"""Forecast service: serves single-sensor forecasts from the trained models.

Bridges the offline experiment artifacts to a live web app:

  * loads the 12 trained LightGBM models (3 horizons x point + q10/q50/q90)
    - plus the 12 XGBoost models for the comparison toggle
  * builds single-row features for (sensor, time) that are value-identical to
    the training pipeline (verified by tests)
  * applies split-conformal calibration (CQR) computed on the validation block
    to fix the ~74% -> ~80% under-coverage found in the experiment
  * consumes the live per-minute "Past Hour" feed: dedupe on
    (location_id, sensing_datetime), aggregate to hourly Melbourne-local time,
    and uses it as the freshest observations (lag_1 / current hour)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg
from src.models import location_code_map, load_booster, predict
from src.features import _calendar_features, build_features
from src.data.load import get_counts, fetch_past_hour_feed, simulate_feed_with_known_dups

FEATURE_COLS = cfg.FEATURES
CALIBRATION_CACHE = cfg.RESULTS_DIR / "calibration.json"


# ------------------------------------------------------------------ data ---

class Service:
    def __init__(self, data: str = "real"):
        self.data = data
        self.counts = get_counts(data)
        self.groups = self._sensor_groups()
        self.locations = self._load_locations()
        self.codes = location_code_map(self.counts)
        self.boosters = self._load_boosters()
        self.calibration = self._load_calibration()
        self.feed = None          # latest live feed snapshot
        self.feed_meta = dict(status="not_fetched", last_update=None, dups_removed=0)

    # -- sensor metadata ------------------------------------------------
    def _sensor_groups(self) -> dict:
        first = self.counts.groupby("location_id")["datetime"].min()
        thr = pd.Timestamp(cfg.SHORT_HISTORY_THRESHOLD)
        return {int(loc): ("short" if t >= thr else "long") for loc, t in first.items()}

    def _load_locations(self) -> pd.DataFrame:
        try:
            from src.data.load import load_sensor_locations
            loc = load_sensor_locations()
            loc["location_id"] = loc["location_id"].astype(int)
            return loc.set_index("location_id")
        except FileNotFoundError:
            return pd.DataFrame()

    def sensor_list(self) -> list[dict]:
        rows = []
        first = self.counts.groupby("location_id")["datetime"].min()
        last = self.counts.groupby("location_id")["datetime"].max()
        for loc in sorted(self.counts["location_id"].unique()):
            loc = int(loc)
            name = self.locations.at[loc, "sensor_name"] if loc in self.locations.index else f"Sensor {loc}"
            desc = self.locations.at[loc, "sensor_description"] if loc in self.locations.index else ""
            rows.append(dict(
                location_id=loc, name=name, description=desc,
                group=self.groups.get(loc, "long"),
                install_date=str(self.locations.at[loc, "installation_date"].date())
                if loc in self.locations.index and pd.notna(self.locations.at[loc, "installation_date"]) else None,
                history_start=str(first[loc].date()), history_end=str(last[loc].date()),
                history_days=(last[loc] - first[loc]).days + 1,
            ))
        return rows

    # -- models ----------------------------------------------------------
    def _load_boosters(self) -> dict:
        dirs = cfg.RESULTS_DIR / self.data
        out = {}
        for fw in ("lgb", "xgb"):
            for h in cfg.HORIZONS:
                out[(fw, h, "point")] = load_booster(fw, dirs / f"{fw}_cpu_point_None_{h}.model")
                for a in cfg.ALPHAS:
                    out[(fw, h, a)] = load_booster(fw, dirs / f"{fw}_cpu_{a}_{a}_{h}.model")
        return out

    # -- calibration -------------------------------------------------------
    def _load_calibration(self) -> dict:
        """Split-conformal (CQR) adjustment: on the val block, conformity score
        = max(q10 - y, y - q90); the 80th percentile of scores is the width
        adjustment added to the raw band."""
        if CALIBRATION_CACHE.exists():
            return json.loads(CALIBRATION_CACHE.read_text())
        print("computing conformal calibration on validation block (one-time)...")
        t0 = time.perf_counter()
        feats, _ = build_features(self.counts, self.codes)
        val = feats[(feats["datetime"] >= pd.Timestamp(cfg.VAL_START)) &
                    (feats["datetime"] < pd.Timestamp(cfg.TEST_START))]
        cal = {}
        for fw in ("lgb", "xgb"):
            for h in cfg.HORIZONS:
                y = val[f"target_{h}"].to_numpy()
                X = val[FEATURE_COLS]
                q10 = np.clip(predict(self.boosters[(fw, h, 0.1)], fw, X), 0, None)
                q90 = np.clip(predict(self.boosters[(fw, h, 0.9)], fw, X), 0, None)
                scores = np.maximum(q10 - y, y - q90).clip(0, None)
                n = len(scores)
                k = int(np.ceil(0.8 * (n + 1)))
                cal[f"{fw}_{h}"] = float(np.sort(scores)[min(k, n) - 1])
        CALIBRATION_CACHE.write_text(json.dumps(cal, indent=1))
        print(f"calibration computed in {time.perf_counter() - t0:.0f}s")
        return cal

    # -- live feed ----------------------------------------------------------
    def refresh_feed(self, use_demo: bool = False) -> dict:
        """Fetch per-minute feed, dedupe, aggregate to hourly. Returns meta + the
        hourly frame. Falls back to a deterministic demo feed with the known
        67/68/69 duplicate bug if the network is unavailable."""
        try:
            if use_demo:
                raise OSError("demo mode")
            hourly, dups = fetch_past_hour_feed()
            status = "live"
        except Exception as e:
            minute, dups = simulate_feed_with_known_dups()
            hourly = minute.groupby([minute["datetime"].dt.floor("h"), "location_id"])["count"] \
                           .sum().rename("count").reset_index().rename(columns={"datetime": "hour"})
            status = f"demo (network unavailable: {str(e)[:60]})"
        self.feed = hourly
        self.feed_meta = dict(
            status=status,
            last_update=str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
            dups_removed=int(dups),
            sensors=len(hourly),
            rows=len(hourly),
        )
        return self.feed_meta

    # -- forecasting ---------------------------------------------------------
    def forecast(self, location_id: int, at: pd.Timestamp | None = None,
                 frameworks: tuple = ("lgb", "xgb")) -> dict:
        """Point + q10/q50/q90 + raw & calibrated 80% bands for all horizons."""
        if location_id not in set(self.counts["location_id"]):
            raise KeyError(f"unknown sensor {location_id}")
        at = pd.Timestamp(at) if at is not None else pd.Timestamp.now().floor("h")
        feed_hourly = self.feed if self.feed is not None else None
        row = make_features_row(self.counts, self.codes, location_id, at, feed_hourly)

        def band(fw, h, mode):
            q10 = float(np.clip(predict(self.boosters[(fw, h, 0.1)], fw, row), 0, None)[0])
            q90 = float(np.clip(predict(self.boosters[(fw, h, 0.9)], fw, row), 0, None)[0])
            if mode == "raw":
                return dict(lo=round(q10, 1), hi=round(q90, 1))
            adj = float(self.calibration.get(f"{fw}_{h}", 0.0))
            return dict(lo=round(max(q10 - adj, 0), 1), hi=round(q90 + adj, 1))

        out = dict(at=str(at))
        for h in cfg.HORIZONS:
            out[str(h)] = {}
            for fw in frameworks:
                out[str(h)][fw] = dict(
                    point=round(float(np.clip(
                        predict(self.boosters[(fw, h, "point")], fw, row), 0, None)[0]), 1),
                    q50=round(float(np.clip(
                        predict(self.boosters[(fw, h, 0.5)], fw, row), 0, None)[0]), 1),
                    band_raw=band(fw, h, "raw"),
                    band_cal=band(fw, h, "cal"),
                )
        return out

    def history(self, location_id: int, hours: int = 168) -> list[dict]:
        s = self.counts[self.counts["location_id"] == location_id].sort_values("datetime")
        out = [dict(t=str(t), v=float(v)) for t, v in s[["datetime", "count"]].itertuples(index=False)]
        if self.feed is not None:
            f = self.feed[self.feed["location_id"] == location_id]
            for t, v in f[["hour", "count"]].itertuples(index=False):
                out.append(dict(t=str(t), v=float(v)))
        return out[-hours:]


# --------------------------------------------------------- feature builder ---

def make_features_row(counts: pd.DataFrame, codes: pd.Series, location_id: int,
                      at: pd.Timestamp, feed_hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Single-row features for (sensor, at), value-identical to build_features.

    at = the prediction reference time. The 'current' count (inclusive of `at`)
    comes from the live feed when available, else from the hourly history.
    """
    s = counts[counts["location_id"] == location_id].sort_values("datetime")
    hour = pd.Timestamp(at).floor("h")

    def value_at(t: pd.Timestamp) -> float:
        if feed_hourly is not None:
            hit = feed_hourly[(feed_hourly["location_id"] == location_id) &
                              (feed_hourly["hour"] == t)]["count"]
            if len(hit):
                return float(hit.iloc[0])
        hit = s[s["datetime"] == t]["count"]
        return float(hit.iloc[0]) if len(hit) else np.nan

    now_val = value_at(hour)
    feed_has_hour = (feed_hourly is not None and
                     len(feed_hourly[(feed_hourly["location_id"] == location_id) &
                                     (feed_hourly["hour"] == hour)]) > 0)

    def roll_mean(win: int) -> float:
        lo = hour - pd.Timedelta(hours=win - 1)
        if feed_has_hour:
            # feed replaces the row at `hour` (history may be stale at that hour)
            window = s[(s["datetime"] >= lo) & (s["datetime"] < hour)]
            vals = window["count"].astype(float).tolist()
            vals.append(now_val)
        else:
            window = s[(s["datetime"] >= lo) & (s["datetime"] <= hour)]
            vals = window["count"].astype(float).tolist()
        return float(np.mean(vals)) if vals else float("nan")

    cal = _calendar_features(pd.Series([hour]))
    feat = {}
    for c in cal.columns:
        feat[c] = float(cal[c].iloc[0])
    feat["lag_1"] = value_at(hour - pd.Timedelta(hours=1))
    feat["lag_24"] = value_at(hour - pd.Timedelta(hours=24))
    feat["lag_168"] = value_at(hour - pd.Timedelta(hours=168))
    feat["roll_24_mean"] = roll_mean(24)
    feat["roll_168_mean"] = roll_mean(168)
    feat["roll_672_mean"] = roll_mean(672)
    code = codes.get(location_id, int(codes.mode()[0]))
    feat["location_code"] = float(code)

    row = pd.DataFrame([feat])[FEATURE_COLS]
    row[cfg.CATEGORICAL_FEATURE] = row[cfg.CATEGORICAL_FEATURE].astype("int32")
    return row
