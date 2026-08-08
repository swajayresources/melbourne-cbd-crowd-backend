"""Modal.com Serverless ML Inference Service for Melbourne CBD Crowd Platform.

Offloads ONNX model inference to Modal's cloud infrastructure, eliminating
CPU/memory pressure on Render's free tier (512MB limit).

Endpoints:
  GET  /predict         heuristic fallback (kept for backwards compatibility)
  POST /predict_onnx    real LightGBM ONNX inference over the 12 trained models

The POST endpoint accepts the 12-feature row built by Render
(make_features_row) and returns the forecast structure identical to
Service.forecast():

    {"1": {"point", "q50", "band_raw": {"lo","hi"}, "band_cal": {"lo","hi"}}, ...}

Usage:
  modal deploy backend/scripts/modal_app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# Model assets live in backend/results/onnx (12 LightGBM ONNX binaries).
ONNX_DIR = Path(__file__).resolve().parent.parent / "results" / "onnx"

app = modal.App("melbourne-cbd-crowd-ml")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "onnxruntime", "pydantic", "fastapi[standard]")
    .add_local_dir(ONNX_DIR, "/models")
)

HORIZONS = (1, 6, 24)
QUANTILES = ("point", 0.1, 0.5, 0.9)
CALIBRATION = {"1": 45.2, "6": 85.0, "24": 140.5}

# Fast p50/p75 lookup thresholds (same as the rule engine on Render)
THRESHOLDS = {
    1: {"p50": 888.0, "p75": 1989.0},
    2: {"p50": 745.0, "p75": 1620.0},
    3: {"p50": 1120.0, "p75": 2450.0},
}


def _model_name(h: int, key: Any) -> str:
    if key == "point":
        return f"lgb_cpu_point_None_{h}"
    return f"lgb_cpu_{key}_{key}_{h}"


def _model_path(h: int, key: Any) -> Path:
    return Path("/models") / f"{_model_name(h, key)}.onnx"


def _estimate_exceedance_probability(q10: float, q50: float, q90: float, p75_threshold: float) -> float:
    import numpy as np

    y = float(p75_threshold)
    q10, q50, q90 = float(q10), float(q50), float(q90)
    q50 = max(q50, q10 + 1e-3)
    q90 = max(q90, q50 + 1e-3)

    if y <= q10:
        slope = 0.10 / max(q10, 1.0)
        cdf = max(0.01, slope * y)
    elif y <= q50:
        cdf = 0.10 + 0.40 * ((y - q10) / (q50 - q10))
    elif y <= q90:
        cdf = 0.50 + 0.40 * ((y - q50) / (q90 - q50))
    else:
        extra_span = max(q90 * 0.5, 50.0)
        cdf = min(0.99, 0.90 + 0.09 * ((y - q90) / extra_span))

    prob_exceed = (1.0 - cdf) * 100.0
    return float(round(float(np.clip(prob_exceed, 0.1, 99.9)), 1))


@app.function(image=image, min_containers=1)
@modal.fastapi_endpoint(method="GET")
def predict_endpoint(location_id: int = 1, hour: int = 12, dow: int = 0):
    """Fast heuristic web endpoint (legacy fallback, sub-10ms)."""
    th = THRESHOLDS.get(location_id, {"p50": 300.0, "p75": 750.0})
    base_count = round(th["p50"] * (1 + 0.3 * (12 - abs(hour - 12)) / 12), 1)

    if base_count < th["p50"]:
        level, sensory_label = "LOW", "Calm (Quiet)"
        advice = "Low crowd density — low noise & movement."
        prob_exceed = 15.0
    elif base_count <= th["p75"]:
        level, sensory_label = "MEDIUM", "Moderate Activity"
        advice = "Moderate movement — noticeable activity."
        prob_exceed = 45.0
    else:
        level, sensory_label = "HIGH", "Busy / Overstimulating"
        advice = "High crowd density — potential sound & visual stimulation."
        prob_exceed = 85.0

    return {
        "location_id": location_id,
        "point": base_count,
        "q50": base_count,
        "band_cal": {"lo": round(max(0, base_count * 0.85), 1), "hi": round(base_count * 1.15, 1)},
        "band_raw": {"lo": round(max(0, base_count * 0.80), 1), "hi": round(base_count * 1.20, 1)},
        "level": level,
        "sensory_label": sensory_label,
        "sensory_advice": advice,
        "p75_exceed_prob_pct": prob_exceed,
        "thresholds": th,
        "mode": "modal_serverless",
        "provider": "modal_serverless",
    }


@app.function(image=image, min_containers=1, timeout=60)
@modal.fastapi_endpoint(method="POST")
def predict_onnx(body: dict):
    """Real LightGBM ONNX inference on the 12 trained models.

    body = {
        "features": [12 floats in cfg.FEATURES order],
        "calibration": {"1": .., "6": .., "24": ..},   # optional overrides
    }
    """
    import numpy as np
    import onnxruntime as ort

    features = body.get("features")
    if not features or len(features) != 12:
        raise ValueError("features must be a 12-element list")
    calibration = dict(CALIBRATION)
    calibration.update(body.get("calibration") or {})

    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    sessions: Dict[tuple, Any] = {}

    def run(key: Any, h: int) -> float:
        cache_key = (h, key)
        if cache_key not in sessions:
            sess = ort.InferenceSession(str(_model_path(h, key)), providers=["CPUExecutionProvider"])
            sessions[cache_key] = sess
        sess = sessions[cache_key]
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        preds = sess.run([output_name], {input_name: x})[0]
        return float(np.clip(np.asarray(preds).flatten()[0], 0, None))

    out: Dict[str, Any] = {}
    for h in HORIZONS:
        pt = run("point", h)
        q10 = run(0.1, h)
        q50 = run(0.5, h)
        q90 = run(0.9, h)
        adj = float(calibration.get(str(h), 0.0))
        out[str(h)] = {
            "point": round(pt, 1),
            "q50": round(q50, 1),
            "band_raw": {"lo": round(q10, 1), "hi": round(q90, 1)},
            "band_cal": {"lo": round(max(q10 - adj, 0), 1), "hi": round(q90 + adj, 1)},
        }
    return {"forecast": out, "mode": "modal_onnx", "provider": "modal_onnx"}


@app.function(image=image, min_containers=1, timeout=90)
@modal.fastapi_endpoint(method="POST")
def predict_onnx_batch(body: dict):
    """Batched real ONNX inference: one HTTP round-trip for many sensors.

    body = {
        "sensors": {
            "<location_id>": [12 features],
            ...
        },
        "calibration": {"1": .., "6": .., "24": ..},
    }
    Returns {"forecasts": {"<location_id>": {"1": {...}, "6": {...}, "24": {...}}}}
    """
    import numpy as np
    import onnxruntime as ort

    sensors = body.get("sensors") or {}
    if not sensors:
        return {"forecasts": {}}
    calibration = dict(CALIBRATION)
    calibration.update(body.get("calibration") or {})

    session_cache: Dict[tuple, Any] = {}

    def get_session(key: Any, h: int):
        cache_key = (h, key)
        if cache_key not in session_cache:
            sess = ort.InferenceSession(str(_model_path(h, key)), providers=["CPUExecutionProvider"])
            session_cache[cache_key] = sess
        return session_cache[cache_key]

    def run_single(features: List[float], key: Any, h: int) -> float:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        sess = get_session(key, h)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        preds = sess.run([output_name], {input_name: x})[0]
        return float(np.clip(np.asarray(preds).flatten()[0], 0, None))

    forecasts: Dict[str, Any] = {}
    for loc_str, features in sensors.items():
        if len(features) != 12:
            continue
        out: Dict[str, Any] = {}
        for h in HORIZONS:
            pt = run_single(features, "point", h)
            q10 = run_single(features, 0.1, h)
            q50 = run_single(features, 0.5, h)
            q90 = run_single(features, 0.9, h)
            adj = float(calibration.get(str(h), 0.0))
            out[str(h)] = {
                "point": round(pt, 1),
                "q50": round(q50, 1),
                "band_raw": {"lo": round(q10, 1), "hi": round(q90, 1)},
                "band_cal": {"lo": round(max(q10 - adj, 0), 1), "hi": round(q90 + adj, 1)},
            }
        forecasts[loc_str] = out
    return {"forecasts": forecasts, "mode": "modal_onnx_batch", "provider": "modal_onnx"}
