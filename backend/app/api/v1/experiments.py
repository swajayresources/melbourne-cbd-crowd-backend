"""Experiments and model evaluation API V1 routes."""
from __future__ import annotations

import json
from flask import Blueprint, jsonify, current_app

experiments_bp = Blueprint("experiments", __name__)


def get_services():
    return current_app.extensions["forecast_service"]


@experiments_bp.get("/importance")
def api_importance():
    svc = get_services()
    from src.report import importance_table
    tbl, stats = importance_table(svc.service.data)
    return jsonify(dict(table=tbl, stats=stats))


@experiments_bp.get("/experiment")
def api_experiment():
    svc = get_services()
    from src import config as cfg
    res = {}
    d = cfg.RESULTS_DIR / svc.service.data
    for p in d.glob("*.json"):
        if p.stem != "meta":
            res[p.stem] = json.loads(p.read_text())
    meta = json.loads((d / "meta.json").read_text())
    out = dict(meta=meta)
    for h in cfg.HORIZONS:
        row = {}
        for fw in ("xgb", "lgb"):
            key = f"{fw}_cpu_point_None_{h}"
            if key in res:
                m = res[key]
                row[fw] = dict(mae=m["mae"], rmse=m["rmse"], mape=m["mape"],
                               file_mb=round(m["file_size_bytes"] / 1e6, 3),
                               single_ms=m["single_ms_mean"])
        out[f"point_{h}"] = row
        for fw in ("xgb", "lgb"):
            q = res.get(f"{fw}_cpu_0.1_0.1_{h}")
            if q:
                out.setdefault(f"interval_{h}", {})[fw] = dict(
                    coverage=q["coverage"], pinball=q["interval_pinball"],
                    width=q["interval_width"])
    out["calibration"] = svc.service.calibration
    totals = {}
    for m in res.values():
        if "train_time_s" not in m:
            continue
        k = f"{m['framework']}_{m['device']}"
        totals[k] = totals.get(k, 0) + m["train_time_s"]
    out["train_totals"] = totals
    return jsonify(out)
