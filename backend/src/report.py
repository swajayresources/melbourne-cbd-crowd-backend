"""Report generator: reads results/<data>/*.json -> clean paste-ready tables.

Usage:  python -m src.report --data real
Writes results/<data>/report.md + CSVs and prints everything to stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from . import config as cfg

FMT = "{:.1f}"
def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def load_results(data: str) -> dict:
    d = cfg.RESULTS_DIR / data
    return {p.stem: json.loads(p.read_text()) for p in d.glob("*.json") if p.stem != "meta"}


def md_table(rows, headers, aligns=None) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def point_table(res: dict, data: str) -> str:
    headers = ["Horizon", "Model", "MAE", "RMSE", "MAPE %", "WMAPE %"]
    rows = []
    for h in cfg.HORIZONS:
        for fw in ("xgb", "lgb"):
            m = res.get(f"{fw}_cpu_point_None_{h}")
            if not m:
                continue
            rows.append([f"{h}h", fw.upper(), fmt(m["mae"]), fmt(m["rmse"]),
                         fmt(m["mape"] * 100, 2), fmt(m["wmape"] * 100, 2)])
    return md_table(rows, headers)


def interval_table(res: dict) -> str:
    headers = ["Horizon", "Model", "Pinball@0.1", "Pinball@0.5", "Pinball@0.9",
               "80% interval pinball", "Coverage %", "Mean width"]
    rows = []
    for h in cfg.HORIZONS:
        for fw in ("xgb", "lgb"):
            q1 = res.get(f"{fw}_cpu_0.1_0.1_{h}")
            q5 = res.get(f"{fw}_cpu_0.5_0.5_{h}")
            q9 = res.get(f"{fw}_cpu_0.9_0.9_{h}")
            if not (q1 and q5 and q9):
                continue
            rows.append([f"{h}h", fw.upper(),
                         fmt(q1["pinball"]), fmt(q5["pinball"]), fmt(q9["pinball"]),
                         fmt(q1["interval_pinball"]),
                         fmt(q1["coverage"] * 100, 1), fmt(q1["interval_width"])])
    return md_table(rows, headers)


def train_time_table(res: dict, meta: dict) -> str:
    rows = []
    per_model = {"xgb_cpu": [], "xgb_gpu": [], "lgb_cpu": [], "lgb_gpu": []}
    for m in res.values():
        if "train_time_s" not in m:
            continue
        key = f"{m['framework']}_{m['device']}"
        per_model.setdefault(key, []).append(m["train_time_s"])
    headers = ["Training device", "XGBoost total (s)", "XGBoost mean/model (s)",
               "LightGBM total (s)", "LightGBM mean/model (s)"]
    rows.append(["CPU", fmt(sum(per_model["xgb_cpu"]), 0), fmt(np.mean(per_model["xgb_cpu"]), 1),
                 fmt(sum(per_model["lgb_cpu"]), 0), fmt(np.mean(per_model["lgb_cpu"]), 1)])
    gx, gl = per_model.get("xgb_gpu", []), per_model.get("lgb_gpu", [])
    rows.append(["GPU", fmt(sum(gx), 0), fmt(np.mean(gx), 1) if gx else "-",
                 fmt(sum(gl), 0) if gl else "N/A (no GPU build)", "-"])
    rows.append(["GPU speedup (XGB)", fmt(sum(per_model["xgb_cpu"]) / sum(gx), 1) if gx else "-",
                 "-", "-", "-"])
    return md_table(rows, headers)


def latency_table(res: dict) -> str:
    headers = ["Horizon", "Model", "Batch us/row", "Single ms (mean)", "Single ms (p95)",
               "File size (MB)", "Serving RSS (MB)"]
    rows = []
    for h in cfg.HORIZONS:
        for fw in ("xgb", "lgb"):
            m = res.get(f"{fw}_cpu_point_None_{h}")
            if not m or m.get("device") != "cpu":
                continue
            rows.append([f"{h}h", fw.upper(), fmt(m["batch_ms_per_row"], 4),
                         fmt(m["single_ms_mean"], 3), fmt(m["single_ms_p95"], 3),
                         fmt(m["file_size_bytes"] / 1e6, 3), fmt(m["serving_rss_mb"], 1)])
    return md_table(rows, headers)


def group_table(res: dict) -> str:
    headers = ["Horizon", "Model", "Group", "n rows", "MAE", "RMSE", "MAPE %", "Coverage %", "Width"]
    rows = []
    for h in cfg.HORIZONS:
        for fw in ("xgb", "lgb"):
            p = res.get(f"{fw}_cpu_point_None_{h}")
            q1 = res.get(f"{fw}_cpu_0.1_0.1_{h}")
            if not (p and q1):
                continue
            for g in ("short", "long"):
                rows.append([f"{h}h", fw.upper(), g,
                             p.get(f"{g}_n", "-"), fmt(p.get(f"{g}_mae")), fmt(p.get(f"{g}_rmse")),
                             fmt(p.get(f"{g}_mape", 0) * 100, 2) if f"{g}_mape" in p else "-",
                             fmt(q1.get(f"{g}_coverage", 0) * 100, 1) if f"{g}_coverage" in q1 else "-",
                             fmt(q1.get(f"{g}_width"))])
    return md_table(rows, headers)


def importance_table(data: str, top: int = 15) -> str:
    """Gain importance from the h=1 point models trained on CPU."""
    def load(fw):
        path = cfg.RESULTS_DIR / data / f"{fw}_cpu_point_None_1.model"
        if fw == "xgb":
            b = xgb.Booster(); b.load_model(str(path))
            sc = b.get_score(importance_type="gain")
        else:
            b = lgb.Booster(model_file=str(path))
            sc = dict(zip(b.feature_name(), b.feature_importance("gain")))
        s = pd.Series(sc).sort_values(ascending=False)
        return s / s.sum()

    x, l = load("xgb"), load("lgb")
    common = sorted(set(x.index) & set(l.index))
    rho = x[common].rank().corr(l[common].rank()) if len(common) > 1 else np.nan
    jaccard = len(set(x.head(10).index) & set(l.head(10).index)) / 10
    rows, names = [], sorted(set(x.index) | set(l.index), key=lambda n: x.get(n, 0), reverse=True)
    for i, n in enumerate(names[:top], 1):
        rows.append([i, n, fmt(x.get(n, 0) * 100, 2), fmt(l.get(n, 0) * 100, 2)])
    tbl = md_table(rows, ["#", "Feature", "XGBoost gain %", "LightGBM gain %"])
    return tbl, dict(rho=float(rho), top10_jaccard=jaccard,
                     xgb_top10=list(x.head(10).index), lgb_top10=list(l.head(10).index))


def head_to_head(res: dict, imp: dict, meta: dict) -> str:
    rows = []
    for h in cfg.HORIZONS:
        x, l = res.get(f"xgb_cpu_point_None_{h}"), res.get(f"lgb_cpu_point_None_{h}")
        rows.append([f"MAE @ {h}h", fmt(x["mae"]), fmt(l["mae"]), "XGB" if x["mae"] < l["mae"] else "LGB"])
        rows.append([f"RMSE @ {h}h", fmt(x["rmse"]), fmt(l["rmse"]), "XGB" if x["rmse"] < l["rmse"] else "LGB"])
        xq, lq = res.get(f"xgb_cpu_0.1_0.1_{h}"), res.get(f"lgb_cpu_0.1_0.1_{h}")
        rows.append([f"Coverage @ {h}h (%)", fmt(xq["coverage"] * 100, 1), fmt(lq["coverage"] * 100, 1),
                     "XGB" if abs(xq["coverage"] - 0.8) < abs(lq["coverage"] - 0.8) else "LGB"])
        rows.append([f"Interval pinball @ {h}h", fmt(xq["interval_pinball"]),
                     fmt(lq["interval_pinball"]), "XGB" if xq["interval_pinball"] < lq["interval_pinball"] else "LGB"])
    x = [m for m in res.values() if m.get("framework") == "xgb" and m.get("device") == "cpu"]
    l = [m for m in res.values() if m.get("framework") == "lgb" and m.get("device") == "cpu"]
    xg = [m for m in res.values() if m.get("framework") == "xgb" and m.get("device") == "gpu"]
    rows += [
        ["CPU training, all 12 models (s)", fmt(sum(m["train_time_s"] for m in x), 0),
         fmt(sum(m["train_time_s"] for m in l), 0),
         "XGB" if sum(m["train_time_s"] for m in x) < sum(m["train_time_s"] for m in l) else "LGB"],
        ["GPU training, all 12 models (s)",
         fmt(sum(m["train_time_s"] for m in xg), 0) if xg else "-",
         "N/A (pip wheel has no GPU)", "-"],
        ["Batch inference us/row",
         fmt(min(m["batch_ms_per_row"] for m in x), 4), fmt(min(m["batch_ms_per_row"] for m in l), 4), "-"],
        ["Single-row inference ms (mean)",
         fmt(np.mean([m["single_ms_mean"] for m in x]), 3), fmt(np.mean([m["single_ms_mean"] for m in l]), 3), "-"],
        ["Model file size (point @1h, MB)", fmt(x[0]["file_size_bytes"] / 1e6, 3),
         fmt(l[0]["file_size_bytes"] / 1e6, 3), "-"],
        ["Serving RSS delta (MB)", fmt(np.mean([m["serving_rss_mb"] for m in x]), 1),
         fmt(np.mean([m["serving_rss_mb"] for m in l]), 1), "-"],
        ["GPU support", "CUDA (pip)", "N/A (needs source build)", "-"],
    ]
    return md_table(rows, ["Metric", "XGBoost", "LightGBM", "Winner"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    args = ap.parse_args()
    data = args.data
    res = load_results(data)
    meta = json.loads((cfg.RESULTS_DIR / data / "meta.json").read_text())
    imp, imp_stats = importance_table(data)

    blocks = {
        "environment": f"- python {meta['python']} | {meta['platform']}\n"
                       f"- CPUs: {meta['cpus']} | GPU: {meta['gpu']}\n"
                       f"- xgboost {meta['xgboost']} | lightgbm {meta['lightgbm']}\n"
                       f"- data: {meta['data']} | {meta['n_rows']:,} rows | {meta['n_sensors']} sensors\n"
                       f"- short-history sensors: {meta['short_sensors']}",
        "T1_split": "train: < 2025-11-01 | val: 2025-11-01..2026-02-28 | test: >= 2026-03-01 (most recent block)",
        "T2_point": point_table(res, data),
        "T3_intervals": interval_table(res),
        "T4_traintime": train_time_table(res, meta),
        "T5_latency": latency_table(res),
        "T6_groups": group_table(res),
        "T7_importance": f"{imp}\n\nSpearman rank corr (gain, all features): {imp_stats['rho']:.3f}  "
                         f"| Top-10 feature overlap (Jaccard): {imp_stats['top10_jaccard']:.2f}\n"
                         f"XGBoost top10: {list(imp_stats['xgb_top10'])}\n"
                         f"LightGBM top10: {list(imp_stats['lgb_top10'])}",
        "T8_head2head": head_to_head(res, imp_stats, meta),
    }

    out = "\n\n".join(f"## {k}\n\n{v}" for k, v in blocks.items())
    (cfg.RESULTS_DIR / data / "report.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
