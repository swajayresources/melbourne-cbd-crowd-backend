"""Database seeder script to stream historical hourly counts into Supabase PostgreSQL."""
from __future__ import annotations

import os
import sys
import json
import urllib.request
from pathlib import Path
import pandas as pd

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path("backend").resolve()
sys.path.insert(0, str(BASE_DIR))

from app.config import BaseConfig

SUPABASE_URL = os.getenv("SUPABASE_URL", BaseConfig.SUPABASE_URL).rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", BaseConfig.SUPABASE_KEY)


def seed_hourly_counts(batch_size: int = 500):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[ERROR] SUPABASE_URL and SUPABASE_KEY environment variables are required.")
        return

    csv_path = BASE_DIR / "data" / "raw" / "hourly_counts.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found.")
        return

    print(f"Reading raw hourly counts from {csv_path}...")
    df = pd.read_csv(csv_path, sep=";", parse_dates=["sensing_date"], low_memory=False)
    df = df.rename(columns={"pedestriancount": "count"})
    df["datetime"] = (df["sensing_date"] + pd.to_timedelta(df["hourday"], unit="h")).dt.strftime("%Y-%m-%d %H:%M:%S")

    subset = df[["location_id", "datetime", "count", "sensor_name"]].dropna(subset=["count"])
    subset = subset[subset["count"] >= 0].drop_duplicates(subset=["location_id", "datetime"])

    print(f"Total historical records to stream: {len(subset):,}")

    url = f"{SUPABASE_URL}/rest/v1/hourly_counts"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    records = []
    seeded_count = 0

    for idx, r in enumerate(subset.itertuples(index=False), 1):
        records.append({
            "location_id": int(r.location_id),
            "datetime": str(r.datetime),
            "count": float(r.count),
            "sensor_name": str(r.sensor_name if pd.notna(r.sensor_name) else ""),
        })

        if len(records) >= batch_size:
            req = urllib.request.Request(url, data=json.dumps(records).encode("utf-8"), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    seeded_count += len(records)
                    print(f"Seeded batch {seeded_count:,} / {len(subset):,} records (Status: {resp.status})")
            except Exception as e:
                print(f"[WARNING] Failed to seed batch around row {idx}: {e}")
            records = []

    if records:
        req = urllib.request.Request(url, data=json.dumps(records).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                seeded_count += len(records)
                print(f"[SUCCESS] Final batch seeded! Total: {seeded_count:,} records.")
        except Exception as e:
            print(f"[ERROR] Final batch failed: {e}")


if __name__ == "__main__":
    seed_hourly_counts()
