"""API V1 package initialization and Blueprint registration."""
from __future__ import annotations

from flask import Blueprint

from .sensors import sensors_bp
from .routing import routing_bp
from .forecast import forecast_bp
from .auth import auth_bp
from .experiments import experiments_bp

api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")

api_v1.register_blueprint(sensors_bp)
api_v1.register_blueprint(routing_bp)
api_v1.register_blueprint(forecast_bp)
api_v1.register_blueprint(auth_bp)
api_v1.register_blueprint(experiments_bp)
