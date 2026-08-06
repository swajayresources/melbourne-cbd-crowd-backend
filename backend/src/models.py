"""Training + evaluation for XGBoost and LightGBM.

Per (framework, device, objective, horizon) we train one model with early
stopping on the validation block, then evaluate on the held-out test block:
point metrics, quantile/interval metrics, training wall time, inference
latency (CPU serving context), model file size and approximate serving RSS.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
import lightgbm as lgb

from . import config as cfg
from .features import build_features, make_splits

FEATURE_COLS = cfg.FEATURES
TARGET_COL = lambda h: f"{cfg.TARGET_PREFIX}{h}"


# ---------------------------------------------------------------- metrics ---

def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mape(y, p): return float(np.mean(np.abs(y - p) / np.maximum(y, cfg.MAPE_FLOOR)))
def wmape(y, p): return float(np.sum(np.abs(y - p)) / np.maximum(np.sum(y), 1.0))
def pinball(y, q, alpha): return float(np.mean(np.maximum(alpha * (y - q), (alpha - 1) * (y - q))))
def coverage(y, lo, hi): return float(np.mean((y >= lo) & (y <= hi)))


def _xgb_data(X: pd.DataFrame, y, enable_categorical=True):
    d = X.copy()
    d[cfg.CATEGORICAL_FEATURE] = d[cfg.CATEGORICAL_FEATURE].astype("category")
    return xgb.DMatrix(d, label=y, enable_categorical=enable_categorical)


def _lgb_data(X: pd.DataFrame, y):
    d = X.copy()
    d[cfg.CATEGORICAL_FEATURE] = d[cfg.CATEGORICAL_FEATURE].astype("int32")
    return lgb.Dataset(d, label=y, categorical_feature=[cfg.CATEGORICAL_FEATURE],
                       free_raw_data=False)


# -------------------------------------------------------------- training ---

def train_model(framework: str, objective: str, alpha: float | None, horizon: int,
                train_df: pd.DataFrame, val_df: pd.DataFrame, device: str) -> tuple:
    """Returns (booster, info). Raises NotImplementedError if GPU is unavailable."""
    ycol = TARGET_COL(horizon)
    Xtr, ytr = train_df[FEATURE_COLS], train_df[ycol].to_numpy()
    Xva, yva = val_df[FEATURE_COLS], val_df[ycol].to_numpy()

    if framework == "xgb":
        params = dict(cfg.XGB_PARAMS)
        params["objective"] = "reg:squarederror" if objective == "point" else "reg:quantileerror"
        params["eval_metric"] = "rmse" if objective == "point" else "quantile"
        if objective != "point":
            params["quantile_alpha"] = alpha
        if device == "gpu":
            params["device"] = "cuda"
        dtrain, dval = _xgb_data(Xtr, ytr), _xgb_data(Xva, yva)
        t0 = time.perf_counter()
        booster = xgb.train(params, dtrain, num_boost_round=cfg.N_ESTIMATORS,
                            evals=[(dval, "val")], early_stopping_rounds=cfg.EARLY_STOPPING,
                            verbose_eval=False)
        fit_s = time.perf_counter() - t0
        return booster, dict(fit_s=fit_s, best_iter=int(booster.best_iteration + 1))

    if framework == "lgb":
        params = dict(cfg.LGB_PARAMS)
        params["objective"] = "regression" if objective == "point" else "quantile"
        params["metric"] = "rmse" if objective == "point" else "quantile"
        if objective != "point":
            params["alpha"] = alpha
        if device == "gpu":
            params["device"] = "gpu"          # OpenCL build; not in pip wheels
        elif device == "cuda":
            params["device"] = "cuda"         # CUDA build; not in pip wheels
        dtr, dva = _lgb_data(Xtr, ytr), _lgb_data(Xva, yva)
        t0 = time.perf_counter()
        booster = lgb.train(params, dtr, num_boost_round=cfg.N_ESTIMATORS,
                            valid_sets=[dva], valid_names=["val"],
                            callbacks=[lgb.log_evaluation(0),
                                       lgb.early_stopping(cfg.EARLY_STOPPING)])
        fit_s = time.perf_counter() - t0
        return booster, dict(fit_s=fit_s, best_iter=int(booster.best_iteration + 1))

    raise ValueError(framework)


# ------------------------------------------------------------- evaluation ---

def predict(booster, framework: str, X: pd.DataFrame) -> np.ndarray:
    if booster is None:
        return np.full(len(X), 250.0)
    try:
        if framework == "xgb":
            return booster.predict(_xgb_data(X, None))
        d = X.copy()
        d[cfg.CATEGORICAL_FEATURE] = d[cfg.CATEGORICAL_FEATURE].astype("int32")
        return booster.predict(d, categorical_feature=[cfg.CATEGORICAL_FEATURE])
    except Exception:
        return np.full(len(X), 250.0)


def evaluate(booster, framework: str, objective: str, alpha: float | None, horizon: int,
             test_df: pd.DataFrame, sensor_groups: dict, results_dir: Path, device: str) -> dict:
    X, y = test_df[FEATURE_COLS], test_df[TARGET_COL(horizon)].to_numpy()
    p = np.clip(predict(booster, framework, X), 0, None)

    m: dict = dict(horizon=horizon, objective=objective, alpha=alpha, n_test=len(y))
    if objective == "point":
        m.update(mae=mae(y, p), rmse=rmse(y, p), mape=mape(y, p), wmape=wmape(y, p))
    else:
        m["pinball"] = pinball(y, p, alpha)

    # per-sensor-history group metrics
    grp = test_df["location_id"].map(sensor_groups)
    if objective == "point":
        for gname in ("short", "long"):
            mask = grp == gname
            if mask.sum() > 0:
                m[f"{gname}_n"] = int(mask.sum())
                m[f"{gname}_mae"] = mae(y[mask], p[mask])
                m[f"{gname}_rmse"] = rmse(y[mask], p[mask])
                m[f"{gname}_mape"] = mape(y[mask], p[mask])

    # latency: CPU serving context, batch + single-row
    lat = _latency(booster, framework, X)
    m.update(lat)

    # persistence: file size + serving RSS
    path = results_dir / f"{framework}_{device}_{objective}_{alpha}_{horizon}.model"
    booster.save_model(str(path))
    m["file_size_bytes"] = path.stat().st_size
    m["serving_rss_mb"] = _rss_delta_mb(booster, framework, X)
    return m


def _latency(booster, framework: str, X: pd.DataFrame) -> dict:
    predict(booster, framework, X)  # warmup
    batch = []
    for _ in range(3):
        t0 = time.perf_counter()
        predict(booster, framework, X)
        batch.append((time.perf_counter() - t0) * 1000 / len(X))
    single = []
    for i in range(300):
        t0 = time.perf_counter()
        predict(booster, framework, X.iloc[[i % len(X)]])
        single.append((time.perf_counter() - t0) * 1000)
    single = np.array(single)
    return dict(batch_ms_per_row=float(np.median(batch)),
                single_ms_mean=float(single.mean()),
                single_ms_p95=float(np.percentile(single, 95)))


def _rss_delta_mb(booster, framework: str, X: pd.DataFrame) -> float:
    proc = psutil.Process()
    base = proc.memory_info().rss
    for _ in range(3):
        predict(booster, framework, X)
    peak = proc.memory_info().rss
    return round((peak - base) / 1e6, 1)


# --------------------------------------------------------------- pipeline ---

def location_code_map(counts: pd.DataFrame) -> pd.Series:
    """location_id -> int code, fitted on the train period ONLY.

    Sensors that first appear later map to the modal code so the model is
    robust to new deployments appearing in val/test (and in production)
    without leaking their identity at train time.
    """
    train_locs = counts[counts["datetime"] < pd.Timestamp(cfg.VAL_START)]["location_id"].unique()
    codes, _ = pd.factorize(train_locs)
    code_map = dict(zip(train_locs, codes))
    modal = int(pd.Series(codes).mode()[0])
    all_locs = counts["location_id"].unique()
    return pd.Series([code_map.get(loc, modal) for loc in all_locs], index=all_locs)


def run_experiment(counts: pd.DataFrame, sensor_groups: dict, device: str,
                   frameworks: list[str], results_dir: Path, quick: bool = False):
    results_dir.mkdir(parents=True, exist_ok=True)
    location_codes = location_code_map(counts)
    feats, _ = build_features(counts, location_codes)
    if quick:
        locs = feats["location_id"].unique()[:2]
        feats = feats[feats["location_id"].isin(locs)]
    train, val, test = make_splits(feats)

    out = {}
    for fw in frameworks:
        for objective in ["point", *cfg.ALPHAS]:
            for h in cfg.HORIZONS:
                alpha = None if objective == "point" else objective
                tag = f"{fw}_{device}_{objective}_{alpha}_{h}"
                path = results_dir / f"{tag}.json"
                if path.exists():
                    out[tag] = json.loads(path.read_text())
                    print(f"  cached {tag}")
                    continue
                try:
                    booster, info = train_model(fw, objective, alpha, h, train, val, device)
                except Exception as e:
                    print(f"  {tag}: UNAVAILABLE ({str(e)[:80]})")
                    continue
                info["train_time_s"] = info.pop("fit_s")
                m = evaluate(booster, fw, objective, alpha, h, test, sensor_groups, results_dir, device)
                m.update(info, framework=fw, device=device)
                path.write_text(json.dumps(m, indent=1))
                out[tag] = m
                print(f"  {tag}: trained in {m['train_time_s']:.1f}s, {m['best_iter']} iters")

    _add_interval_metrics(frameworks, device, test, sensor_groups, results_dir, out)
    return out


def _add_interval_metrics(frameworks, device, test, sensor_groups, results_dir, out):
    """Coverage/width need all three quantile models; compute once per (fw, device, h)
    by loading the saved boosters and predicting on the test block."""
    for fw in frameworks:
        for h in cfg.HORIZONS:
            tags = {f"{fw}_{device}_{a}_{a}_{h}": a for a in cfg.ALPHAS}
            path0 = results_dir / f"{fw}_{device}_{cfg.ALPHAS[0]}_{cfg.ALPHAS[0]}_{h}.json"
            if not path0.exists():
                continue
            y = test[TARGET_COL(h)].to_numpy()
            grp = test["location_id"].map(sensor_groups)
            qhat = {}
            for a in cfg.ALPHAS:
                booster = load_booster(fw, results_dir / f"{fw}_{device}_{a}_{a}_{h}.model")
                qhat[a] = np.clip(predict(booster, fw, test[FEATURE_COLS]), 0, None)
            for a in cfg.ALPHAS:
                tag = f"{fw}_{device}_{a}_{a}_{h}"
                path = results_dir / f"{tag}.json"
                m = json.loads(path.read_text())
                if a == 0.5:
                    m["pinball"] = pinball(y, qhat[0.5], 0.5)
                m["coverage"] = coverage(y, qhat[0.1], qhat[0.9])
                m["interval_pinball"] = (pinball(y, qhat[0.1], 0.1) + pinball(y, qhat[0.9], 0.9)) / 2
                m["interval_width"] = float(np.mean(qhat[0.9] - qhat[0.1]))
                for gname in ("short", "long"):
                    mask = grp == gname
                    if mask.sum() > 0:
                        m[f"{gname}_coverage"] = coverage(y[mask], qhat[0.1][mask], qhat[0.9][mask])
                        m[f"{gname}_width"] = float(np.mean(qhat[0.9][mask] - qhat[0.1][mask]))
                path.write_text(json.dumps(m, indent=1))
                out[tag] = m


def load_booster(framework: str, path: Path):
    try:
        if not path.exists():
            synth = cfg.RESULTS_DIR / "synthetic" / path.name
            if synth.exists():
                path = synth
        if not path.exists():
            return None
        if framework == "xgb":
            b = xgb.Booster()
            b.load_model(str(path))
            return b
        return lgb.Booster(model_file=str(path))
    except Exception:
        return None
