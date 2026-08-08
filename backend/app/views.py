"""HTML Web View routes for Melbourne CBD Pedestrian Platform.

The frontend is served from Vercel; these routes redirect there.
"""
from __future__ import annotations

import os

from flask import Blueprint, redirect

views_bp = Blueprint("views", __name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://cbd-calm-route.vercel.app")


def _frontend(path: str = ""):
    return redirect(f"{FRONTEND_URL.rstrip('/')}/{path.lstrip('/')}")


@views_bp.get("/")
def index():
    return _frontend("")


@views_bp.get("/forecast")
def dashboard():
    return _frontend("")


@views_bp.get("/map")
def map_page():
    return _frontend("map")


@views_bp.get("/predict")
def predict_page():
    return _frontend("predict")


@views_bp.get("/models")
def models():
    return _frontend("models")


@views_bp.get("/features")
def features():
    return _frontend("features")


@views_bp.get("/help")
def help_page():
    return _frontend("help")
