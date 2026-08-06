"""Auth API V1 routes enforcing APP 1988 Zero-PII compliance."""
from __future__ import annotations

from flask import Blueprint, jsonify, request, current_app

auth_bp = Blueprint("auth", __name__)


def get_services():
    return (
        current_app.extensions["auth_service"],
        current_app.extensions["db_service"],
    )


@auth_bp.post("/auth/session")
def create_session():
    auth_svc, db_svc = get_services()
    body = request.get_json(silent=True) or {}
    public_key = body.get("public_key", "").strip()

    if not public_key:
        session_hash = "anon_default_session"
    else:
        session_hash = auth_svc.hash_public_key(public_key)

    token = auth_svc.create_anonymous_token(session_hash)
    prefs = db_svc.get_user_preferences(session_hash)

    return jsonify(dict(
        token=token,
        session_hash=session_hash,
        preferences=prefs,
        compliance="APP_1988_ZERO_PII",
    ))


@auth_bp.get("/user/prefs")
def get_user_prefs():
    auth_svc, db_svc = get_services()
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "").strip()

    payload = auth_svc.verify_token(token) if token else None
    if not payload:
        return jsonify(dict(error="Unauthorized or expired anonymous session")), 401

    session_hash = payload.get("session_hash", "")
    prefs = db_svc.get_user_preferences(session_hash)
    return jsonify(dict(preferences=prefs, session_hash=session_hash))


@auth_bp.post("/user/prefs")
def save_user_prefs():
    auth_svc, db_svc = get_services()
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "").strip()

    payload = auth_svc.verify_token(token) if token else None
    if not payload:
        return jsonify(dict(error="Unauthorized or expired anonymous session")), 401

    session_hash = payload.get("session_hash", "")
    body = request.get_json(silent=True) or {}
    prefs = body.get("preferences", {})

    success = db_svc.save_user_preferences(session_hash, prefs)
    return jsonify(dict(success=success, preferences=prefs))
