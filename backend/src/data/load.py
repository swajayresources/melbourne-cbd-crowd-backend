"""Real data loading + live (per-minute "Past Hour") feed handling.

Normalises the City of Melbourne exports into one long format:
    location_id, datetime, count, sensor_name

The portal field names differ from the docs users know:
    sensing_date + hourday  -> datetime   (was: sensing_datetime)
    pedestriancount         -> count      (was: total_of_directions)
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

from .. import config as cfg

MIN_OBS = 48  # sensors with fewer hourly records than this are dropped

PAST_HOUR_URL = (
    "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
    "pedestrian-counting-system-past-hour-counts-per-minute/exports/csv?limit=-1"
)


def load_real_counts() -> pd.DataFrame:
    """Read the raw hourly-counts export -> long format with clean types."""
    if not cfg.RAW_HOURLY_CSV.exists():
        print(f"Dataset {cfg.RAW_HOURLY_CSV.name} not present — falling back to synthetic counts...")
        from .synthetic import make_synthetic_counts
        return make_synthetic_counts(all_sensors=True)

    df = pd.read_csv(
        cfg.RAW_HOURLY_CSV,
        sep=";",
        parse_dates=["sensing_date"],
        low_memory=False,
    )
    df = df.rename(columns={"pedestriancount": "count"})
    df["datetime"] = df["sensing_date"] + pd.to_timedelta(df["hourday"], unit="h")
    df = df.rename(columns={"location_id": "location_id"})[["location_id", "datetime", "count", "sensor_name"]]
    df = df.dropna(subset=["count"])
    df = df[df["count"] >= 0]
    df = df.drop_duplicates(subset=["location_id", "datetime"])
    df = df.sort_values(["location_id", "datetime"]).reset_index(drop=True)
    sizes = df.groupby("location_id")["datetime"].count()
    keep = sizes[sizes >= MIN_OBS].index
    df = df[df["location_id"].isin(keep)]
    print(f"real counts: {len(df):,} rows, {df.location_id.nunique()} sensors "
          f"({df.datetime.min()} .. {df.datetime.max()})")
    return df


def load_sensor_locations() -> pd.DataFrame:
    """Sensor metadata: street description, installation date, status."""
    json_path = cfg.RAW_LOCATIONS_CSV.parent.parent / "sensor_locations.json"
    if json_path.exists():
        try:
            df = pd.read_json(json_path)
            if "installation_date" in df.columns:
                df["installation_date"] = pd.to_datetime(df["installation_date"], errors="coerce")
            return df[
                ["location_id", "sensor_description", "sensor_name",
                 "installation_date", "status", "latitude", "longitude"]
            ]
        except Exception:
            pass

    if cfg.RAW_LOCATIONS_CSV.exists():
        loc = pd.read_csv(cfg.RAW_LOCATIONS_CSV, sep=";", parse_dates=["installation_date"])
        return loc[
            ["location_id", "sensor_description", "sensor_name",
             "installation_date", "status", "latitude", "longitude"]
        ]
    return pd.DataFrame()


PAST_HOUR_JSON_URL = (
    "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
    "pedestrian-counting-system-past-hour-counts-per-minute/records?limit=1000"
)


def fetch_past_hour_feed() -> tuple[pd.DataFrame, int]:
    """Live per-minute feed (NOT used for training).

    Queries City of Melbourne Open Data API for live minute records, dedupes
    on (location_id, sensing_datetime), and aggregates to hourly counts in local time.
    """
    raw = pd.DataFrame()
    # 1. Try fast JSON REST API (<200ms)
    try:
        req = urllib.request.Request(PAST_HOUR_JSON_URL, headers={"User-Agent": "MelbournePedestrianCrowdMap/1.0"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if results:
                rows = []
                for item in results:
                    dt = item.get("sensing_datetime") or item.get("sensing_date")
                    cnt = item.get("total_of_directions") if "total_of_directions" in item else item.get("pedestriancount", 0)
                    loc = item.get("location_id")
                    if dt and loc is not None:
                        rows.append({"location_id": int(loc), "datetime": dt, "count": float(cnt or 0)})
                raw = pd.DataFrame(rows)
    except Exception as err:
        print(f"live feed JSON API fetch failed, trying CSV export: {err}")

    # 2. Fallback to CSV export URL if JSON failed or returned empty
    if raw.empty:
        try:
            req = urllib.request.Request(PAST_HOUR_URL, headers={"User-Agent": "MelbournePedestrianCrowdMap/1.0"})
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                content = resp.read().decode("utf-8")
            raw = pd.read_csv(io.StringIO(content), sep=";")
            dt_col = "sensing_datetime" if "sensing_datetime" in raw.columns else "sensing_date"
            count_col = next((c for c in ("pedestriancount", "total_of_directions") if c in raw.columns), None)
            if count_col:
                raw = raw.rename(columns={dt_col: "datetime", count_col: "count"})
        except Exception as err:
            print(f"live feed CSV export fetch failed: {err}")
            return pd.DataFrame(), 0

    if raw.empty or "datetime" not in raw.columns or "count" not in raw.columns:
        return pd.DataFrame(), 0

    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True, errors="coerce")
    raw["datetime"] = raw["datetime"].dt.tz_convert("Australia/Melbourne").dt.tz_localize(None)
    n_before = len(raw)
    raw = raw.drop_duplicates(subset=["location_id", "datetime"])
    n_after = len(raw)
    hourly = (raw.groupby([raw["datetime"].dt.floor("h"), "location_id"])["count"]
                  .sum().rename("count").reset_index())
    hourly = hourly.rename(columns={"datetime": "hour"})
    print(f"live feed: {n_before} raw minute rows -> {n_after} after dedupe "
          f"(removed {n_before - n_after} dups) -> {len(hourly)} hourly aggregates (local time)")
    return hourly, n_before - n_after


def simulate_feed_with_known_dups() -> tuple[pd.DataFrame, int]:
    """Deterministic demo of the documented 67/68/69 duplicate bug, so the
    dedupe path is testable offline. Returns (deduped_minute_df, dups_removed)."""
    rng = pd.date_range("2026-08-03 00:00", "2026-08-03 23:59", freq="min")
    rows = []
    for loc in (67, 68, 69, 10, 20):
        base = 300.0 if loc in (67, 68, 69) else 150.0
        vals = (base * (1 + 0.5 * rng.hour / 24)).astype(int)
        for t, v in zip(rng, vals):
            rows.append((loc, t, v))
            if loc in (67, 68, 69) and t.minute % 7 == 0:
                rows.append((loc, t, v))  # duplicate minute row
    n_before = len(rows)
    df = pd.DataFrame(rows, columns=["location_id", "datetime", "count"])
    df = df.drop_duplicates(subset=["location_id", "datetime"])
    return df, n_before - len(df)


def get_counts(source: str) -> pd.DataFrame:
    """Unified entry point: 'real' reads the downloaded CSV, 'synthetic' calls the generator."""
    if source == "real":
        return load_real_counts()
    if source == "synthetic":
        from .synthetic import make_synthetic_counts
        return make_synthetic_counts()
    raise ValueError(f"unknown data source: {source!r}")
