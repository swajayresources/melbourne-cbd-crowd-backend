"""Database seeder script to populate Supabase PostgreSQL with sensor metadata and historical counts."""
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


def seed_sensor_locations():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[ERROR] SUPABASE_URL and SUPABASE_KEY environment variables are required.")
        print("Usage: SUPABASE_URL='https://...supabase.co' SUPABASE_KEY='...' python scripts/seed_supabase.py")
        return

    csv_path = BASE_DIR / "data" / "raw" / "sensor_locations.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found.")
        return

    print(f"Reading sensor metadata from {csv_path}...")
    df = pd.read_csv(csv_path, sep=";")
    records = []
    for _, r in df.iterrows():
        records.append({
            "location_id": int(r["location_id"]),
            "sensor_description": str(r.get("sensor_description", "")),
            "sensor_name": str(r.get("sensor_name", "")),
            "status": str(r.get("status", "A")),
            "latitude": float(r.get("latitude", 0)),
            "longitude": float(r.get("longitude", 0)),
        })

    print(f"Seeding {len(records)} sensor records into Supabase...")
    url = f"{SUPABASE_URL}/rest/v1/sensor_locations?on_conflict=location_id"
    req = urllib.request.Request(
        url,
        data=json.dumps(records).encode("utf-8"),
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            print(f"[SUCCESS] Supabase sensor locations seeded! Status: {resp.status}")
    except Exception as e:
        print(f"[ERROR] Failed to seed Supabase: {e}")


if __name__ == "__main__":
    seed_sensor_locations()
