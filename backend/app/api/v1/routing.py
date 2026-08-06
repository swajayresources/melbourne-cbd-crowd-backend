"""Routing API V1 routes."""
from __future__ import annotations

import pandas as pd
from flask import Blueprint, jsonify, request, current_app

routing_bp = Blueprint("routing", __name__)


def get_services():
    return (
        current_app.extensions["routing_service"],
        current_app.extensions["forecast_service"],
        current_app.extensions["cache_service"],
    )


@routing_bp.get("/geocode")
def geocode():
    routing_svc, _, cache = get_services()
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify(dict(results=[]))
    
    cache_key = f"geocode_{q.lower()}"
    cached_geo = cache.get(cache_key)
    if cached_geo:
        return jsonify(dict(results=cached_geo))

    res = routing_svc.geocode(q)
    cache.set(cache_key, res, ttl=3600)
    return jsonify(dict(results=res))


@routing_bp.get("/route")
def route():
    routing_svc, forecast_svc, _ = get_services()
    try:
        orig_lat = float(request.args.get("orig_lat", 0))
        orig_lon = float(request.args.get("orig_lon", 0))
        dest_lat = float(request.args.get("dest_lat", 0))
        dest_lon = float(request.args.get("dest_lon", 0))
    except (TypeError, ValueError):
        return jsonify(dict(error="invalid latitude/longitude parameters")), 400

    mode = request.args.get("mode", current_app.config.get("DEFAULT_SERVER_MODE", "rule"))
    dt_str = request.args.get("datetime")
    try:
        dt = pd.Timestamp(dt_str) if dt_str else pd.Timestamp.now()
    except Exception:
        dt = pd.Timestamp.now()

    routes, is_fallback = routing_svc.fetch_routes(orig_lat, orig_lon, dest_lat, dest_lon)
    evaluated = routing_svc.evaluate_routes(
        routes=routes,
        dt=dt,
        mode=mode,
        forecast_service=forecast_svc.service,
    )

    return jsonify(dict(
        orig=dict(lat=orig_lat, lon=orig_lon),
        dest=dict(lat=dest_lat, lon=dest_lon),
        datetime=str(dt),
        mode=mode,
        is_fallback=is_fallback,
        routes=evaluated["routes"],
        recommendation=evaluated["recommendation"],
    ))
