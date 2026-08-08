"""Real data loading + live (per-minute "Past Hour") feed handling.

Normalises the City of Melbourne exports into one long format:
    location_id, datetime, count, sensor_name

The portal field names differ from the docs users know:
    sensing_date + hourday  -> datetime   (was: sensing_datetime)
    pedestriancount         -> count      (was: total_of_directions)
"""
from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .. import config as cfg

MIN_OBS = 48  # sensors with fewer hourly records than this are dropped

# Recent-history window pulled from Supabase when the local CSV is absent.
# Must cover roll_672 (28 days of lag features); 45 days leaves a margin.
SUPABASE_COUNTS_DAYS = int(os.getenv("SUPABASE_COUNTS_DAYS", "45"))
SUPABASE_FETCH_PAGE = 1000  # PostgREST max rows per request
SUPABASE_FETCH_WORKERS = 10

PAST_HOUR_URL = (
    "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
    "pedestrian-counting-system-past-hour-counts-per-minute/exports/csv?limit=-1"
)


def _supabase_counts_config() -> tuple[str, str]:
    """Resolve Supabase REST credentials from env (falls back to app config)."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if url and key:
        return url, key
    try:
        from flask import current_app
        return (
            current_app.config.get("SUPABASE_URL", ""),
            current_app.config.get("SUPABASE_KEY", ""),
        )
    except Exception:
        return "", ""


def fetch_counts_from_supabase(days: int = SUPABASE_COUNTS_DAYS) -> pd.DataFrame:
    """Pull recent real hourly counts from Supabase (threaded pagination).

    Returns the long format (location_id, datetime, count, sensor_name) with
    the same normalisation as load_real_counts, or an empty DataFrame when
    Supabase is not configured / unreachable / has no rows.
    """
    raw_url, key = _supabase_counts_config()
    if not raw_url or not key:
        print("supabase counts: not configured, skipping")
        return pd.DataFrame()

    rest = raw_url.rstrip("/")
    if not rest.endswith("/rest/v1"):
        rest = f"{rest}/rest/v1"

    since = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    since_enc = urllib.parse.quote(since, safe="")
    table_url = f"{rest}/hourly_counts"

    def fetch_page(offset: int) -> list[dict]:
        url = f"{table_url}?select=location_id,datetime,count,sensor_name&datetime=gte.{since_enc}&order=location_id.asc,datetime.asc&limit={SUPABASE_FETCH_PAGE}&offset={offset}"
        req = urllib.request.Request(
            url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Range": f"{offset}-{offset + SUPABASE_FETCH_PAGE - 1}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"supabase counts: page offset={offset} failed: {e}")
            return []

    frames = []
    offset = 0
    pages_fetched = 0
    max_pages = 400
    reached_end = False
    with ThreadPoolExecutor(max_workers=SUPABASE_FETCH_WORKERS) as pool:
        while pages_fetched < max_pages and not reached_end:
            wave_offsets = [offset + i * SUPABASE_FETCH_PAGE for i in range(SUPABASE_FETCH_WORKERS * 2)]
            futures = {o: pool.submit(fetch_page, o) for o in wave_offsets}
            results = {o: f.result() for o, f in futures.items()}

            for o in sorted(results):
                rows = results[o]
                if rows:
                    frames.append(pd.DataFrame(rows))
                    pages_fetched += 1
                if len(rows) < SUPABASE_FETCH_PAGE:
                    reached_end = True
                    break
            if not reached_end:
                offset = wave_offsets[-1] + SUPABASE_FETCH_PAGE

    if not frames:
        print("supabase counts: no rows returned")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["location_id", "datetime"])
    df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce")
    # The seed writes naive Melbourne-local datetimes (same as the CSV /
    # training pipeline). Only convert if the column is stored tz-aware.
    dt_str = df["datetime"].astype(str)
    if dt_str.str.endswith("Z").any() or dt_str.str.contains("+", regex=False).any():
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["datetime"] = df["datetime"].dt.tz_convert("Australia/Melbourne").dt.tz_localize(None)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    if "sensor_name" not in df.columns:
        df["sensor_name"] = ""
    df = df.dropna(subset=["count", "location_id", "datetime"])
    df = df[df["count"] >= 0]
    df = df.sort_values(["location_id", "datetime"]).reset_index(drop=True)
    df["location_id"] = df["location_id"].astype(int)
    print(f"supabase counts: {len(df):,} rows, {df.location_id.nunique()} sensors "
          f"({df.datetime.min()} .. {df.datetime.max()})")
    return df


def load_real_counts() -> pd.DataFrame:
    """Read real hourly counts: local CSV -> Supabase -> synthetic fallback."""
    if not cfg.RAW_HOURLY_CSV.exists():
        print(f"Dataset {cfg.RAW_HOURLY_CSV.name} not present — trying Supabase real counts...")
        supabase_df = fetch_counts_from_supabase()
        if not supabase_df.empty:
            return supabase_df
        print("Supabase counts unavailable — falling back to synthetic counts...")
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
