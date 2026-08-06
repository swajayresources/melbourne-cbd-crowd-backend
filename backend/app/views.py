"""HTML Web View routes for Melbourne CBD Pedestrian Platform."""
from __future__ import annotations

from flask import Blueprint, render_template

views_bp = Blueprint("views", __name__)


@views_bp.get("/")
def index():
    return render_template("map.html")


@views_bp.get("/forecast")
def dashboard():
    return render_template("index.html")


@views_bp.get("/map")
def map_page():
    return render_template("map.html")


@views_bp.get("/predict")
def predict_page():
    return render_template("predict.html")


@views_bp.get("/models")
def models():
    return render_template("models.html")


@views_bp.get("/features")
def features():
    return render_template("features.html")


@views_bp.get("/help")
def help_page():
    return render_template("help.html")
