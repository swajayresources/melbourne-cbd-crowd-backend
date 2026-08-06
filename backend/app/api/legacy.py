"""Legacy unversioned API routes (/api/*) for full backward compatibility."""
from __future__ import annotations

from flask import Blueprint

legacy_api = Blueprint("legacy_api", __name__, url_prefix="/api")


@legacy_api.get("/sensors")
def legacy_sensors():
    from app.api.v1.sensors import get_sensors
    return get_sensors()


@legacy_api.get("/map")
def legacy_map():
    from app.api.v1.sensors import get_map_data
    return get_map_data()


@legacy_api.get("/geocode")
def legacy_geocode():
    from app.api.v1.routing import geocode
    return geocode()


@legacy_api.get("/route")
def legacy_route():
    from app.api.v1.routing import route
    return route()


@legacy_api.get("/predict")
def legacy_predict():
    from app.api.v1.forecast import api_predict
    return api_predict()


@legacy_api.get("/forecast")
def legacy_forecast():
    from app.api.v1.forecast import api_forecast
    return api_forecast()


@legacy_api.get("/history")
def legacy_history():
    from app.api.v1.forecast import api_history
    return api_history()


@legacy_api.post("/feed")
def legacy_feed():
    from app.api.v1.forecast import api_feed
    return api_feed()


@legacy_api.get("/importance")
def legacy_importance():
    from app.api.v1.experiments import api_importance
    return api_importance()


@legacy_api.get("/experiment")
def legacy_experiment():
    from app.api.v1.experiments import api_experiment
    return api_experiment()
