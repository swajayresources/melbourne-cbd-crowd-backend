"""Flask app: Melbourne CBD Pedestrian Sensory Load Platform server.

Provides HTML views and maintains complete backward compatibility for unversioned /api/*
endpoints by delegating to the modular V1 service architecture.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from flask import render_template, jsonify, request, redirect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app

# Create application instance using Application Factory
app = create_app()

# Global compatibility accessors for legacy scripts & test suites
SERVICE = app.extensions["forecast_service"].service
CROWD_ENGINE = app.extensions["crowd_service"].engine
SERVER_MODE = app.config.get("DEFAULT_SERVER_MODE", "rule")

# Frontend is served from Vercel; HTML routes redirect there.
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://cbd-calm-route.vercel.app")


def svc():
    return app.extensions["forecast_service"].service


def crowd():
    return app.extensions["crowd_service"].engine


# ------------------------------------------------------------- HTML Views -

def _frontend(path: str = ""):
    return redirect(f"{FRONTEND_URL.rstrip('/')}/{path.lstrip('/')}")


@app.get("/")
def index():
    return _frontend("")


@app.get("/forecast")
def dashboard():
    return _frontend("")


@app.get("/map")
def map_page():
    return _frontend("map")


@app.get("/predict")
def predict_page():
    return _frontend("predict")


@app.get("/models")
def models():
    return _frontend("models")


@app.get("/features")
def features():
    return _frontend("features")


@app.get("/help")
def help_page():
    return _frontend("help")


# ------------------------------------------------ Legacy API Compatibility -

@app.get("/api/sensors")
def legacy_api_sensors():
    from app.api.v1.sensors import get_sensors
    return get_sensors()


@app.get("/api/map")
def legacy_api_map():
    from app.api.v1.sensors import get_map_data
    return get_map_data()


@app.get("/api/geocode")
def legacy_api_geocode():
    from app.api.v1.routing import geocode
    return geocode()


@app.get("/api/route")
def legacy_api_route():
    from app.api.v1.routing import route
    return route()


@app.get("/api/predict")
def legacy_api_predict():
    from app.api.v1.forecast import api_predict
    return api_predict()


@app.get("/api/forecast")
def legacy_api_forecast():
    from app.api.v1.forecast import api_forecast
    return api_forecast()


@app.get("/api/history")
def legacy_api_history():
    from app.api.v1.forecast import api_history
    return api_history()


@app.post("/api/feed")
def legacy_api_feed():
    from app.api.v1.forecast import api_feed
    return api_feed()


@app.get("/api/importance")
def legacy_api_importance():
    from app.api.v1.experiments import api_importance
    return api_importance()


@app.get("/api/experiment")
def legacy_api_experiment():
    from app.api.v1.experiments import api_experiment
    return api_experiment()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--mode", choices=["rule", "ml"], default="rule")
    args = ap.parse_args()

    app.config["DEFAULT_SERVER_MODE"] = args.mode
    SERVER_MODE = args.mode

    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)
    except OSError:
        print(f"port {args.port} busy; trying {args.port + 1}")
        app.run(host="0.0.0.0", port=args.port + 1, threaded=True, debug=False)
