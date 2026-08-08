"""Generates static JSON snapshots of sensor locations and lookup statistics.

These files are committed directly to Git so that deployed backend services
(e.g., Render) have instant, zero-latency access to sensor metadata and precomputed
crowd threshold statistics without needing the 122MB raw CSV files.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data"

LOC_CSV = RAW_DIR / "sensor_locations.csv"
COUNTS_CSV = RAW_DIR / "hourly_counts.csv"


def generate_sensor_locations_json():
    if not LOC_CSV.exists():
        print(f"[ERROR] {LOC_CSV} not found.")
        return

    df_loc = pd.read_csv(LOC_CSV, sep=";")
    records_loc = []
    for _, r in df_loc.iterrows():
        records_loc.append({
            "location_id": int(r["location_id"]),
            "sensor_description": str(r.get("sensor_description", "")),
            "sensor_name": str(r.get("sensor_name", "")),
            "status": str(r.get("status", "A")),
            "latitude": float(r.get("latitude", 0)),
            "longitude": float(r.get("longitude", 0)),
            "installation_date": str(r.get("installation_date", ""))
        })

    out_file = DATA_DIR / "sensor_locations.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records_loc, f, indent=2)

    print(f"[SUCCESS] Saved {len(records_loc)} sensors to {out_file}")


def generate_lookup_stats_json():
    if not COUNTS_CSV.exists():
        print(f"[ERROR] {COUNTS_CSV} not found.")
        return

    print("Reading hourly counts CSV to compute lookup stats...")
    df_c = pd.read_csv(COUNTS_CSV, sep=";", parse_dates=["sensing_date"], low_memory=False)
    df_c = df_c.rename(columns={"pedestriancount": "count"})
    df_c["datetime"] = df_c["sensing_date"] + pd.to_timedelta(df_c["hourday"], unit="h")
    df_c["location_id"] = df_c["location_id"].astype(int)
    df_c["hour"] = df_c["datetime"].dt.hour
    df_c["dow"] = df_c["datetime"].dt.dayofweek

    thresholds = {}
    grouped = df_c.groupby("location_id")["count"]
    for loc_id, series in grouped:
        vals = series.dropna().to_numpy()
        if len(vals) > 0:
            p50 = float(np.percentile(vals, 50))
            p75 = float(np.percentile(vals, 75))
            thresholds[str(int(loc_id))] = {"p50": round(p50, 1), "p75": round(p75, 1)}

    expected = {}
    agg = df_c.groupby(["location_id", "hour", "dow"])["count"].mean().reset_index()
    for _, row in agg.iterrows():
        key = f"{int(row['location_id'])}_{int(row['hour'])}_{int(row['dow'])}"
        expected[key] = round(float(row["count"]), 1)

    first_dt = df_c.groupby("location_id")["datetime"].min()
    last_dt = df_c.groupby("location_id")["datetime"].max()
    history_bounds = {}
    for loc_id in df_c["location_id"].unique():
        loc_id = int(loc_id)
        history_bounds[str(loc_id)] = {
            "start": str(first_dt[loc_id].date()),
            "end": str(last_dt[loc_id].date()),
            "days": int((last_dt[loc_id] - first_dt[loc_id]).days + 1)
        }

    lookup_stats = {
        "thresholds": thresholds,
        "expected": expected,
        "history": history_bounds,
        "max_datetime": str(df_c["datetime"].max())
    }

    out_file = DATA_DIR / "sensor_lookup_stats.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(lookup_stats, f)

    print(f"[SUCCESS] Saved lookup stats to {out_file} (Size: {out_file.stat().st_size:,} bytes)")


if __name__ == "__main__":
    generate_sensor_locations_json()
    generate_lookup_stats_json()
