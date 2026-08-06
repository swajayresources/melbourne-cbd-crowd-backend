"""Supabase PostgreSQL integration service with local dataset fallback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Dict, List
import urllib.request
import urllib.error
import pandas as pd


class DatabaseService:
    def __init__(self, supabase_url: str = "", supabase_key: str = "", data_dir: Optional[Path] = None):
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.data_dir = data_dir or (Path(__file__).resolve().parent.parent.parent / "data")
        self._user_prefs_store: Dict[str, Dict[str, Any]] = {}

    def is_supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    def fetch_sensors(self) -> List[Dict[str, Any]]:
        """Fetch sensor metadata from Supabase or fallback to sensor_locations.csv."""
        if self.is_supabase_configured():
            try:
                url = f"{self.supabase_url}/rest/v1/sensor_locations?select=*"
                req = urllib.request.Request(
                    url,
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                    }
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception:
                pass  # Fallback to CSV

        # Local CSV Fallback
        csv_path = self.data_dir / "raw" / "sensor_locations.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return df.to_dict(orient="records")
        return []

    def get_user_preferences(self, session_hash: str) -> Dict[str, Any]:
        """Fetch user accessibility preferences by session hash."""
        if self.is_supabase_configured():
            try:
                url = f"{self.supabase_url}/rest/v1/user_sessions?session_hash=eq.{session_hash}&select=*"
                req = urllib.request.Request(
                    url,
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                    }
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    records = json.loads(resp.read().decode("utf-8"))
                    if records:
                        return records[0].get("preferences", {})
            except Exception:
                pass

        return self._user_prefs_store.get(session_hash, {
            "high_contrast": False,
            "text_scale": 1.0,
            "theme": "default",
        })

    def save_user_preferences(self, session_hash: str, prefs: Dict[str, Any]) -> bool:
        """Store user accessibility preferences without any PII."""
        self._user_prefs_store[session_hash] = prefs

        if self.is_supabase_configured():
            try:
                url = f"{self.supabase_url}/rest/v1/user_sessions"
                payload = json.dumps({
                    "session_hash": session_hash,
                    "preferences": prefs,
                }).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": "application/json",
                        "Prefer": "resolution=merge-duplicates",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    return resp.status in (200, 201)
            except Exception:
                pass
        return True
