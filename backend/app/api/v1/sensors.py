"""Sensors API V1 routes."""
from __future__ import annotations

from flask import Blueprint, jsonify, current_app

sensors_bp = Blueprint("sensors", __name__)


def get_services():
    return (
        current_app.extensions["forecast_service"],
        current_app.extensions["crowd_service"],
        current_app.extensions["cache_service"],
    )


@sensors_bp.get("/sensors")
def get_sensors():
    svc, crowd, cache = get_services()
    cached_sensors = cache.get("api_sensors_list")
    if cached_sensors:
        return jsonify(cached_sensors)

    data = {
        "sensors": svc.sensor_list(),
        "feed": svc.service.feed_meta,
        "data_as_of": str(svc.service.counts["datetime"].max()),
    }
    cache.set("api_sensors_list", data, ttl=60)
    return jsonify(data)


@sensors_bp.get("/map")
def get_map_data():
    svc, crowd, cache = get_services()
    cached_map = cache.get("api_map_data")
    if cached_map:
        return jsonify(cached_map)

    current_counts = {}
    if svc.service.feed is not None and not svc.service.feed.empty:
        for _, r in svc.service.feed.iterrows():
            current_counts[int(r["location_id"])] = float(r["count"])
    else:
        latest = svc.service.counts.groupby("location_id").last()
        for loc_id, r in latest.iterrows():
            current_counts[int(loc_id)] = float(r["count"])

    sensor_list = crowd.get_all_sensors_map_data(current_counts)
    mode = current_app.config.get("DEFAULT_SERVER_MODE", "rule")
    data = {
        "sensors": sensor_list,
        "count": len(sensor_list),
        "data_as_of": str(svc.service.counts["datetime"].max()),
        "mode": mode,
    }
    cache.set("api_map_data", data, ttl=60)
    return jsonify(data)
