"""Crowd engine with neurodivergent sensory-friendly classification, geocoding, OSRM foot routing, and CDF probability interpolation.

Translates raw counts into plain-language sensory status:
  * LOW -> "Calm (Quiet)"
  * MEDIUM -> "Moderate Activity"
  * HIGH -> "Busy / Overstimulating"
"""
from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config as cfg

# CBD Bounding box and default center
CBD_CENTER = (-37.815, 144.965)
CBD_BOUNDS = [[-37.835, 144.935], [-37.795, 144.995]]


class CrowdEngine:
    def __init__(self, data_path: Optional[Path] = None, loc_path: Optional[Path] = None):
        self.raw_hourly_csv = data_path or cfg.RAW_HOURLY_CSV
        self.raw_loc_csv = loc_path or cfg.RAW_LOCATIONS_CSV
        
        self.locations_df = self._load_locations()
        self.thresholds, self.expected_table = self._build_lookup_tables()

    def _load_locations(self) -> pd.DataFrame:
        # 1. Embedded JSON snapshot (committed to Git - works on Render / serverless)
        json_path = self.raw_loc_csv.parent.parent / "sensor_locations.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data:
                    df = pd.DataFrame(data)
                    df["location_id"] = df["location_id"].astype(int)
                    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
                    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
                    return df.set_index("location_id")
            except Exception:
                pass

        # 2. Local CSV fallback
        if self.raw_loc_csv.exists():
            df = pd.read_csv(self.raw_loc_csv, sep=";")
            df["location_id"] = df["location_id"].astype(int)
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            return df.set_index("location_id")
        
        # 3. Supabase REST fallback if local files are missing
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_KEY", "")
        if supabase_url and supabase_key:
            try:
                url = f"{supabase_url.rstrip('/')}/rest/v1/sensor_locations?select=*"
                req = urllib.request.Request(
                    url,
                    headers={
                        "apikey": supabase_key,
                        "Authorization": f"Bearer {supabase_key}",
                    }
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data:
                        df = pd.DataFrame(data)
                        df["location_id"] = df["location_id"].astype(int)
                        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
                        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
                        return df.set_index("location_id")
            except Exception:
                pass
        return pd.DataFrame()

    def _build_lookup_tables(self) -> Tuple[Dict[int, Dict[str, float]], Dict[Tuple[int, int, int], float]]:
        thresholds: Dict[int, Dict[str, float]] = {}
        expected: Dict[Tuple[int, int, int], float] = {}

        # 1. Embedded JSON lookup stats snapshot (committed to Git)
        json_path = self.raw_hourly_csv.parent.parent / "sensor_lookup_stats.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                th_data = stats.get("thresholds", {})
                for k, v in th_data.items():
                    thresholds[int(k)] = {"p50": float(v["p50"]), "p75": float(v["p75"])}
                
                exp_data = stats.get("expected", {})
                for k, v in exp_data.items():
                    parts = k.split("_")
                    if len(parts) == 3:
                        expected[(int(parts[0]), int(parts[1]), int(parts[2]))] = float(v)
                return thresholds, expected
            except Exception:
                pass

        # 2. Raw hourly CSV calculation
        if self.raw_hourly_csv.exists():
            df = pd.read_csv(self.raw_hourly_csv, sep=";", parse_dates=["sensing_date"], low_memory=False)
            df = df.rename(columns={"pedestriancount": "count"})
            df["datetime"] = df["sensing_date"] + pd.to_timedelta(df["hourday"], unit="h")
            df["location_id"] = df["location_id"].astype(int)
            df["hour"] = df["datetime"].dt.hour
            df["dow"] = df["datetime"].dt.dayofweek
            
            grouped = df.groupby("location_id")["count"]
            for loc_id, series in grouped:
                vals = series.dropna().to_numpy()
                if len(vals) > 0:
                    p50 = float(np.percentile(vals, 50))
                    p75 = float(np.percentile(vals, 75))
                    thresholds[int(loc_id)] = {"p50": round(p50, 1), "p75": round(p75, 1)}
                else:
                    thresholds[int(loc_id)] = {"p50": 100.0, "p75": 300.0}

            agg = df.groupby(["location_id", "hour", "dow"])["count"].mean().reset_index()
            for _, row in agg.iterrows():
                loc_id = int(row["location_id"])
                hr = int(row["hour"])
                dow = int(row["dow"])
                expected[(loc_id, hr, dow)] = float(row["count"])

        return thresholds, expected

    def get_thresholds(self, location_id: int) -> Dict[str, float]:
        return self.thresholds.get(location_id, {"p50": 100.0, "p75": 300.0})

    def classify_sensory_level(self, location_id: int, count: float) -> Tuple[str, str, str]:
        """Returns (level_code, sensory_label, sensory_advice)"""
        th = self.get_thresholds(location_id)
        if count < th["p50"]:
            return "LOW", "Calm (Quiet)", "Low crowd density — low noise & movement."
        elif count <= th["p75"]:
            return "MEDIUM", "Moderate Activity", "Moderate movement — noticeable activity."
        else:
            return "HIGH", "Busy / Overstimulating", "High crowd density — potential sound & visual stimulation."

    def classify_level(self, location_id: int, count: float) -> str:
        code, _, _ = self.classify_sensory_level(location_id, count)
        return code

    def predict_rule_count(self, location_id: int, dt: pd.Timestamp) -> float:
        hr = dt.hour
        dow = dt.dayofweek
        base = self.expected_table.get((location_id, hr, dow), 150.0)
        
        factor = 1.0
        if dow >= 5:
            factor *= 0.75
        
        month, day = dt.month, dt.day
        if (month == 1 and day in (1, 26)) or (month == 4 and day == 25) or (month == 12 and day in (25, 26)):
            factor *= 0.3

        return max(0.0, round(base * factor, 1))

    def find_nearest_sensor(self, lat: float, lon: float) -> Optional[int]:
        if self.locations_df.empty:
            return None
        valid = self.locations_df.dropna(subset=["latitude", "longitude"])
        if valid.empty:
            return None
        
        dists = (valid["latitude"] - lat) ** 2 + (valid["longitude"] - lon) ** 2
        return int(dists.idxmin())

    def resolve_location_query(self, query: str) -> Optional[Tuple[int, str, float, float]]:
        """Resolves any address (e.g. '455 Elizabeth Street') or landmark query."""
        query = query.strip()
        if not query:
            return None
        
        geo_results = geocode_location(query, self)
        if geo_results:
            first = geo_results[0]
            lat, lon = first["latitude"], first["longitude"]
            nearest_id = self.find_nearest_sensor(lat, lon)
            if nearest_id is not None:
                row = self.locations_df.loc[nearest_id]
                desc = str(row.get("sensor_description", "")) or str(row.get("sensor_name", ""))
                return nearest_id, first.get("display_name", desc), lat, lon
        
        # Local fallback search matching street names/numbers
        clean_q = re.sub(r'[^\w\s]', '', query).lower()
        words = [w for w in clean_q.split() if w not in ('st', 'street', 'rd', 'road', 'ave', 'avenue', 'ln', 'lane')]
        
        if not self.locations_df.empty:
            for loc_id, row in self.locations_df.iterrows():
                name = str(row.get("sensor_name", "")).lower()
                desc = str(row.get("sensor_description", "")).lower()
                lat = row.get("latitude")
                lon = row.get("longitude")
                if pd.isna(lat) or pd.isna(lon):
                    continue
                if any(w in name or w in desc for w in words if len(w) > 2):
                    return int(loc_id), f"{query} (near {desc or name})", float(lat), float(lon)

        return None

    def get_all_sensors_map_data(self, current_counts: Dict[int, float]) -> List[Dict[str, Any]]:
        results = []
        for loc_id, row in self.locations_df.iterrows():
            loc_id = int(loc_id)
            lat = row.get("latitude")
            lon = row.get("longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue
            
            cnt = current_counts.get(loc_id, 0.0)
            th = self.get_thresholds(loc_id)
            level_code, sensory_label, advice = self.classify_sensory_level(loc_id, cnt)
            
            results.append({
                "location_id": loc_id,
                "name": str(row.get("sensor_name", f"Sensor {loc_id}")),
                "description": str(row.get("sensor_description", "")),
                "latitude": float(lat),
                "longitude": float(lon),
                "current_count": round(float(cnt), 1),
                "level": level_code,
                "sensory_label": sensory_label,
                "sensory_advice": advice,
                "p50": th["p50"],
                "p75": th["p75"],
            })
        return results


# ----------------------------------------------------------- Geocoding -----

def format_clean_address(raw_name: str) -> str:
    """Formats raw Nominatim output into clean, human-readable address."""
    parts = [p.strip() for p in raw_name.split(",") if p.strip()]
    if not parts:
        return raw_name
    
    # Keep street number + street + suburb (e.g. 455 Elizabeth Street, Melbourne 3000)
    filtered = []
    for p in parts:
        if any(bad in p for bad in ("Koreatown", "Greek Precinct", "City of Melbourne")):
            continue
        filtered.append(p)
    
    if len(filtered) >= 3:
        return f"{filtered[0]}, {filtered[1]}, {filtered[2]}"
    elif len(filtered) >= 2:
        return f"{filtered[0]}, {filtered[1]}"
    return parts[0]


def geocode_location(query: str, crowd_engine: CrowdEngine) -> List[Dict[str, Any]]:
    query = query.strip()
    if not query:
        return []

    # 1. Try Nominatim with explicit Melbourne Victoria bounds
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query + ', Melbourne Victoria Australia')}&format=json&limit=5"
    req = urllib.request.Request(url, headers={"User-Agent": "MelbournePedestrianCrowdMap/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data:
                return [{
                    "display_name": format_clean_address(item.get("display_name", query)),
                    "latitude": float(item["lat"]),
                    "longitude": float(item["lon"]),
                    "source": "nominatim"
                } for item in data if "lat" in item and "lon" in item]
    except Exception:
        pass

    # 2. Fallback: Search local sensors matching street name
    results = []
    clean_q = re.sub(r'[^\w\s]', '', query).lower()
    num_match = re.search(r'\d+', query)
    house_num = num_match.group(0) if num_match else ""
    words = [w for w in clean_q.split() if w not in ('st', 'street', 'rd', 'road', 'ave', 'avenue', 'ln', 'lane') and not w.isdigit()]

    df = crowd_engine.locations_df
    if not df.empty:
        for loc_id, row in df.iterrows():
            name = str(row.get("sensor_name", ""))
            desc = str(row.get("sensor_description", ""))
            lat = row.get("latitude")
            lon = row.get("longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue
            
            combined = f"{name} {desc}".lower()
            if any(w in combined for w in words if len(w) > 2):
                clean_desc = desc or name
                display_label = f"{house_num} {query.title()} (near {clean_desc})" if house_num else clean_desc
                results.append({
                    "display_name": display_label,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "source": "local_sensor"
                })
            if len(results) >= 5:
                break

    return results


# ------------------------------------------------------------- Routing -----

def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    index = 0
    lat = 0
    lng = 0
    coordinates = []

    while index < len(polyline_str):
        shift = 0
        result = 0
        while True:
            byte = ord(polyline_str[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = 0
        result = 0
        while True:
            byte = ord(polyline_str[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append((lat * 1e-5, lng * 1e-5))

    return coordinates


def fetch_osrm_route(orig_lat: float, orig_lon: float, dest_lat: float, dest_lon: float) -> Tuple[List[Dict[str, Any]], bool]:
    url = f"https://router.project-osrm.org/route/v1/foot/{orig_lon},{orig_lat};{dest_lon},{dest_lat}?alternatives=3&steps=true&overview=full"
    req = urllib.request.Request(url, headers={"User-Agent": "MelbournePedestrianCrowdMap/1.0"})
    
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("code") == "Ok" and data.get("routes"):
                routes = []
                for idx, r in enumerate(data["routes"]):
                    geom = r.get("geometry", "")
                    coords = decode_polyline(geom) if isinstance(geom, str) else []
                    routes.append({
                        "id": idx,
                        "duration": float(r.get("duration", 0)),
                        "distance": float(r.get("distance", 0)),
                        "coordinates": coords,
                        "is_fallback": False,
                    })
                return routes, False
    except Exception:
        pass

    dist_m = haversine_distance_m(orig_lat, orig_lon, dest_lat, dest_lon)
    walk_sec = dist_m / 1.35
    fallback_route = [{
        "id": 0,
        "duration": round(walk_sec, 1),
        "distance": round(dist_m, 1),
        "coordinates": [(orig_lat, orig_lon), (dest_lat, dest_lon)],
        "is_fallback": True,
        "fallback_note": "Routing service offline — straight-line estimate shown",
    }]
    return fallback_route, True


def evaluate_route_crowds(
    routes: List[Dict[str, Any]],
    dt: pd.Timestamp,
    crowd_engine: CrowdEngine,
    mode: str = "rule",
    service: Any = None
) -> Dict[str, Any]:
    if not routes:
        return {"routes": [], "recommendation": None}

    sensors = crowd_engine.locations_df.dropna(subset=["latitude", "longitude"])
    annotated_routes = []

    for r in routes:
        coords = r["coordinates"]
        nearby_sensors: Dict[int, Dict[str, Any]] = {}
        
        step = max(1, len(coords) // 20)
        sampled = coords[::step] if coords else []

        for lat, lon in sampled:
            for loc_id, row in sensors.iterrows():
                loc_id = int(loc_id)
                slat, slon = float(row["latitude"]), float(row["longitude"])
                d_m = haversine_distance_m(lat, lon, slat, slon)
                if d_m <= 200.0:
                    if loc_id not in nearby_sensors or d_m < nearby_sensors[loc_id]["dist"]:
                        nearby_sensors[loc_id] = {
                            "dist": d_m,
                            "name": str(row.get("sensor_name", f"Sensor {loc_id}")),
                            "desc": str(row.get("sensor_description", "")),
                        }

        remarks = []
        crowd_levels = []
        sensor_details = []

        for loc_id, info in nearby_sensors.items():
            if mode == "ml" and service is not None:
                f = service.forecast_ml_modal(loc_id, dt)
                if f and f.get("1", {}).get("lgb"):
                    pred_count = f["1"]["lgb"]["point"]
                else:
                    pred_count = crowd_engine.predict_rule_count(loc_id, dt)
            else:
                pred_count = crowd_engine.predict_rule_count(loc_id, dt)

            code, sensory_label, advice = crowd_engine.classify_sensory_level(loc_id, pred_count)
            crowd_levels.append(code)
            sensor_name = info["desc"] or info["name"]
            
            sensor_details.append({
                "location_id": loc_id,
                "name": sensor_name,
                "count": pred_count,
                "level": code,
                "sensory_label": sensory_label,
            })
            
            if code == "HIGH":
                remarks.append(f"Busy area near {sensor_name} (~{int(pred_count)} people/hr)")
            elif code == "MEDIUM":
                remarks.append(f"Moderate activity near {sensor_name}")
            else:
                remarks.append(f"Calm zone near {sensor_name}")

        if "HIGH" in crowd_levels:
            route_score = "HIGH"
            sensory_tag = "Busy / Higher Sensory Load"
        elif "MEDIUM" in crowd_levels:
            route_score = "MEDIUM"
            sensory_tag = "Moderate Sensory Load"
        else:
            route_score = "LOW"
            sensory_tag = "Calm (Quiet Path)"

        annotated_routes.append({
            "id": r["id"],
            "duration_min": round(r["duration"] / 60.0, 1),
            "distance_m": round(r["distance"]),
            "coordinates": r["coordinates"],
            "is_fallback": r.get("is_fallback", False),
            "fallback_note": r.get("fallback_note"),
            "crowd_score": route_score,
            "sensory_tag": sensory_tag,
            "remarks": remarks[:4],
            "nearby_sensors": sensor_details,
        })

    fastest = min(annotated_routes, key=lambda x: x["duration_min"])
    score_weights = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    least_crowded = min(annotated_routes, key=lambda x: (score_weights[x["crowd_score"]], x["duration_min"]))

    for ar in annotated_routes:
        ar["is_fastest"] = (ar["id"] == fastest["id"])
        ar["is_least_crowded"] = (ar["id"] == least_crowded["id"])

    return {
        "routes": annotated_routes,
        "recommendation": {
            "fastest_id": fastest["id"],
            "least_crowded_id": least_crowded["id"],
        }
    }


# ----------------------------------------------- Probability CDF Interpolator -

def estimate_exceedance_probability(q10: float, q50: float, q90: float, p75_threshold: float) -> float:
    y = float(p75_threshold)
    q10, q50, q90 = float(q10), float(q50), float(q90)

    q50 = max(q50, q10 + 1e-3)
    q90 = max(q90, q50 + 1e-3)

    if y <= q10:
        slope = 0.10 / max(q10, 1.0)
        cdf = max(0.01, slope * y)
    elif y <= q50:
        cdf = 0.10 + 0.40 * ((y - q10) / (q50 - q10))
    elif y <= q90:
        cdf = 0.50 + 0.40 * ((y - q50) / (q90 - q50))
    else:
        extra_span = max(q90 * 0.5, 50.0)
        cdf = min(0.99, 0.90 + 0.09 * ((y - q90) / extra_span))

    prob_exceed = (1.0 - cdf) * 100.0
    return float(round(np.clip(prob_exceed, 0.1, 99.9), 1))
