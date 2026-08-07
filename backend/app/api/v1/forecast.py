"""Forecast API V1 routes."""
from __future__ import annotations

import pandas as pd
from flask import Blueprint, jsonify, request, current_app

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
        return jsonify(dict(error=f"location {loc_id} not found")), 404

    if not display_name:
        display_name = meta.get("description") or meta.get("name") or f"Location {loc_id}"
    if resolved_lat is None or resolved_lon is None:
        if loc_id in crowd.locations_df.index:
            row = crowd.locations_df.loc[loc_id]
            resolved_lat = float(row.get("latitude", 0))
            resolved_lon = float(row.get("longitude", 0))

    th = crowd.get_thresholds(loc_id)
    p75_val = th["p75"]

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
