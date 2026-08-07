"""Forecast API V1 routes."""
from __future__ import annotations

import os
import json
import urllib.request
import pandas as pd
from flask import Blueprint, jsonify, request, current_app

from app.forecast_service import make_features_row

forecast_bp = Blueprint("forecast", __name__)


def get_services():
    return (
        current_app.extensions["forecast_service"],
        current_app.extensions["crowd_service"],
    )


@forecast_bp.get("/forecast")
def api_forecast():
    svc, _ = get_services()
    try:
        loc = int(request.args.get("location_id", 1))
    except ValueError:
        return jsonify(dict(error="location_id must be an integer")), 400
    try:
        f = svc.get_forecast(loc, frameworks=("lgb",))
    except KeyError as e:
        return jsonify(dict(error=str(e))), 404
    meta = next((s for s in svc.sensor_list() if s["location_id"] == loc), None)
    if not meta:
        return jsonify(dict(error=f"sensor {loc} metadata not found")), 404
    return jsonify(dict(
        forecast=f,
        sensor=meta,
        feed=svc.service.feed_meta,
        history=svc.history(loc, 168),
    ))


@forecast_bp.get("/history")
def api_history():
    svc, _ = get_services()
    loc = int(request.args.get("location_id", 1))
    hours = int(request.args.get("hours", 168))
    return jsonify(dict(points=svc.history(loc, hours), feed=svc.service.feed_meta))


@forecast_bp.post("/feed")
def api_feed():
    svc, _ = get_services()
    demo = request.args.get("demo") == "1"
    return jsonify(svc.refresh_feed(use_demo=demo))


@forecast_bp.get("/predict")
def api_predict():
    svc, crowd = get_services()
    q_param = request.args.get("q", "").strip()
    loc_param = request.args.get("location_id")
    lat_param = request.args.get("lat")
    lon_param = request.args.get("lon")
    dt_str = request.args.get("datetime")

    try:
        dt = pd.Timestamp(dt_str) if dt_str else pd.Timestamp.now().floor("h")
    except Exception:
        dt = pd.Timestamp.now().floor("h")

    display_name = ""
    resolved_lat, resolved_lon = None, None

    if q_param:
        resolved = crowd.resolve_location_query(q_param)
        if resolved:
            loc_id, display_name, resolved_lat, resolved_lon = resolved
        else:
            loc_id = 1
            display_name = q_param
    elif loc_param:
        try:
            loc_id = int(loc_param)
        except ValueError:
            return jsonify(dict(error="location_id must be an integer")), 400
    elif lat_param and lon_param:
        try:
            lat, lon = float(lat_param), float(lon_param)
            nearest = crowd.find_nearest_sensor(lat, lon)
            if nearest is None:
                return jsonify(dict(error="could not find nearest sensor")), 404
            loc_id = nearest
            resolved_lat, resolved_lon = lat, lon
        except ValueError:
            return jsonify(dict(error="lat and lon must be numbers")), 400
    else:
        loc_id = 1

    meta = next((s for s in svc.sensor_list() if s["location_id"] == loc_id), None)
    if not meta:
        if loc_id in crowd.locations_df.index:
            row = crowd.locations_df.loc[loc_id]
            meta = {
                "location_id": loc_id,
                "name": str(row.get("sensor_name", f"Sensor {loc_id}")),
                "description": str(row.get("sensor_description", "")),
                "group": "long",
            }
        else:
            meta = {"location_id": loc_id, "name": f"Location {loc_id}", "description": "", "group": "long"}

    if not display_name:
        display_name = meta.get("description") or meta.get("name") or f"Location {loc_id}"
    if resolved_lat is None or resolved_lon is None:
        if loc_id in crowd.locations_df.index:
            row = crowd.locations_df.loc[loc_id]
            resolved_lat = float(row.get("latitude", 0))
            resolved_lon = float(row.get("longitude", 0))

    th = crowd.get_thresholds(loc_id)
    p75_val = th["p75"]

    # Real ONNX inference via Modal (offloads heavy ML from Render free tier).
    modal_ml_url = current_app.config.get("MODAL_ML_API_URL") or os.getenv("MODAL_ML_API_URL", "")
    if modal_ml_url:
        try:
            row = make_features_row(svc.service.counts, svc.service.codes, loc_id, dt, svc.service.feed)
            payload = json.dumps({
                "features": [float(v) for v in row.iloc[0].tolist()],
                "calibration": {str(h): float(svc.service.calibration.get(f"lgb_{h}", 0.0))
                                for h in (1, 6, 24)},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{modal_ml_url.rstrip('/')}",
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "MelbournePedestrianCrowdMap/1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                modal_data = json.loads(resp.read().decode("utf-8"))
            fc = modal_data.get("forecast", {})
            h1 = fc.get("1", {})
            if h1:
                point_pred = float(h1["point"])
                band_cal = h1["band_cal"]
                q10 = band_cal["lo"]
                q50 = h1["q50"]
                q90 = band_cal["hi"]
                prob_exceed = svc.calculate_exceedance_prob(q10, q50, q90, p75_val)
                level_code, sensory_label, advice = crowd.classify_sensory_level(loc_id, point_pred)
                return jsonify(dict(
                    sensor=meta,
                    display_name=display_name,
                    latitude=resolved_lat,
                    longitude=resolved_lon,
                    datetime=str(dt),
                    point=point_pred,
                    q50=q50,
                    band_cal=band_cal,
                    band_raw=h1.get("band_raw", band_cal),
                    thresholds=th,
                    level=level_code,
                    sensory_label=sensory_label,
                    sensory_advice=advice,
                    p75_exceed_prob_pct=prob_exceed,
                    mode="modal_onnx",
                    forecast=fc,
                ))
        except Exception:
            pass

    # Optional Modal Serverless Offloading if MODAL_API_URL is configured
    modal_url = current_app.config.get("MODAL_API_URL") or os.getenv("MODAL_API_URL", "")
    if modal_url:
        try:
            req_url = f"{modal_url.rstrip('/')}?location_id={loc_id}&hour={dt.hour}&dow={dt.dayofweek}"
            req = urllib.request.Request(req_url, headers={"User-Agent": "MelbournePedestrianCrowdMap/1.0"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                modal_data = json.loads(resp.read().decode("utf-8"))
                modal_data["sensor"] = meta
                modal_data["display_name"] = display_name
                modal_data["latitude"] = resolved_lat
                modal_data["longitude"] = resolved_lon
                modal_data["datetime"] = str(dt)
                if "mode" not in modal_data:
                    modal_data["mode"] = modal_data.get("provider", "modal_serverless")
                if "p75_exceed_prob_pct" not in modal_data:
                    modal_data["p75_exceed_prob_pct"] = 50.0 if modal_data.get("point", 0) >= p75_val else 20.0
                return jsonify(modal_data)
        except Exception:
            pass

    try:
        f = svc.get_forecast(loc_id, at=dt, frameworks=("lgb",))
        h1 = f["1"]["lgb"]
        point_pred = h1["point"]
        band_cal = h1["band_cal"]
        q10 = band_cal["lo"]
        q50 = h1["q50"]
        q90 = band_cal["hi"]

        prob_exceed = svc.calculate_exceedance_prob(q10, q50, q90, p75_val)
        level_code, sensory_label, advice = crowd.classify_sensory_level(loc_id, point_pred)

        return jsonify(dict(
            sensor=meta,
            display_name=display_name,
            latitude=resolved_lat,
            longitude=resolved_lon,
            datetime=str(dt),
            point=point_pred,
            q50=q50,
            band_cal=band_cal,
            band_raw=h1["band_raw"],
            thresholds=th,
            level=level_code,
            sensory_label=sensory_label,
            sensory_advice=advice,
            p75_exceed_prob_pct=prob_exceed,
            mode="ml",
        ))
    except Exception as e:
        rule_count = crowd.predict_rule_count(loc_id, dt)
        level_code, sensory_label, advice = crowd.classify_sensory_level(loc_id, rule_count)
        return jsonify(dict(
            sensor=meta,
            datetime=str(dt),
            point=rule_count,
            q50=rule_count,
            band_cal=dict(lo=round(max(0, rule_count * 0.7), 1), hi=round(rule_count * 1.3, 1)),
            band_raw=dict(lo=round(max(0, rule_count * 0.8), 1), hi=round(rule_count * 1.2, 1)),
            thresholds=th,
            level=level_code,
            sensory_label=sensory_label,
            sensory_advice=advice,
            p75_exceed_prob_pct=50.0 if rule_count >= p75_val else 20.0,
            mode="rule_fallback",
            note=str(e),
        ))
