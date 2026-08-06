"""Flask application factory module."""
from __future__ import annotations

import os
from pathlib import Path
from flask import Flask, jsonify

from app.config import config_by_name, BaseConfig
from app.services.forecast_service import ForecastService
from app.services.crowd_service import CrowdService
from app.services.routing_service import RoutingService
from app.services.auth_service import AuthService
from app.services.cache_service import CacheService
from app.services.db_service import DatabaseService


def create_app(config_name: str | None = None) -> Flask:
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "production")

    app = Flask(__name__, template_folder="templates", static_folder="static")

    cfg_cls = config_by_name.get(config_name, BaseConfig)
    app.config.from_object(cfg_cls)

    # Initialize Services
    data_type = os.getenv("DATA_TYPE", "real")
    forecast_svc = ForecastService(data=data_type)
    crowd_svc = CrowdService()
    routing_svc = RoutingService(crowd_svc.engine)
    auth_svc = AuthService(secret_key=app.config["SECRET_KEY"])
    cache_svc = CacheService(
        redis_url=app.config["UPSTASH_REDIS_URL"],
        redis_token=app.config["UPSTASH_REDIS_TOKEN"],
        default_ttl=app.config["CACHE_TTL_SECONDS"],
    )
    db_svc = DatabaseService(
        supabase_url=app.config["SUPABASE_URL"],
        supabase_key=app.config["SUPABASE_KEY"],
        data_dir=app.config["DATA_DIR"],
    )

    # Refresh live feed initial snapshot (only if not in testing mode)
    if not app.config.get("TESTING"):
        try:
            forecast_svc.refresh_feed()
        except Exception:
            pass

    # Store services in app extensions
    app.extensions["forecast_service"] = forecast_svc
    app.extensions["crowd_service"] = crowd_svc
    app.extensions["routing_service"] = routing_svc
    app.extensions["auth_service"] = auth_svc
    app.extensions["cache_service"] = cache_svc
    app.extensions["db_service"] = db_svc

    # Register Blueprints
    from app.api.v1 import api_v1
    from app.api.legacy import legacy_api
    from app.views import views_bp
    app.register_blueprint(api_v1)
    app.register_blueprint(legacy_api)
    app.register_blueprint(views_bp)

    # Security Headers & CORS
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

    # Global Error Handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify(dict(error="Resource not found", status=404)), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify(dict(error="Internal server error", status=500)), 500

    return app
