"""Experiment orchestrator: builds data/features, runs all models, persists results.

Usage:
    python -m src.experiment --data real --device both --framework both
    python -m src.experiment --data synthetic --device cpu --quick
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import config as cfg
from .data.load import get_counts
from .models import run_experiment


def _env_meta() -> dict:
    meta = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        cpus=__import__("os").cpu_count(),
        xgboost=__import__("xgboost").__version__,
        lightgbm=__import__("lightgbm").__version__,
    )
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        meta["gpu"] = out
    except Exception:
        meta["gpu"] = "unavailable"
    return meta


def sensor_groups(counts: pd.DataFrame) -> dict:
    """location_id -> 'short' | 'long' based on first observed record."""
    first = counts.groupby("location_id")["datetime"].min()
    thr = pd.Timestamp(cfg.SHORT_HISTORY_THRESHOLD)
    return {loc: ("short" if t >= thr else "long") for loc, t in first.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--device", choices=["cpu", "gpu", "both"], default="both")
    ap.add_argument("--framework", choices=["xgb", "lgb", "both"], default="both")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    out_dir = cfg.RESULTS_DIR / args.data
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = get_counts(args.data)
    groups = sensor_groups(counts)
    print("short-history sensors:", {k for k, v in groups.items() if v == "short"})
    print("full-history sensors:", sum(1 for v in groups.values() if v == "long"))
    print("building features (full series, then chronological split)...")
    (out_dir / "meta.json").write_text(json.dumps(
        {**_env_meta(), "data": args.data, "n_rows": len(counts),
         "n_sensors": counts.location_id.nunique(),
         "short_sensors": sorted(k for k, v in groups.items() if v == "short")},
        indent=1))

    frameworks = ["xgb", "lgb"] if args.framework == "both" else [args.framework]
    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]
    # LightGBM GPU is not compiled into official pip wheels; probe once and record.
    if "gpu" in devices and "lgb" in frameworks:
        _probe_lgb_gpu(out_dir)

    for device in devices:
        if device == "gpu" and not _gpu_ok():
            print(f"[warn] skipping gpu: no CUDA device detected")
            continue
        print(f"\n=== device={device} ===")
        run_experiment(counts, groups, device, frameworks, out_dir, quick=args.quick)
    print("\ndone. results in", out_dir)
    return 0


def _gpu_ok() -> bool:
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=30)
        return True
    except Exception:
        return False


def _probe_lgb_gpu(out_dir: Path):
    import lightgbm as lgb
    import numpy as np
    X = np.random.rand(1000, 5).astype("float32")
    y = np.random.rand(1000).astype("float32")
    probe = dict(gpu="ok", cuda="ok")
    for dev in ("gpu", "cuda"):
        try:
            lgb.train({"objective": "regression", "device": dev, "verbosity": -1},
                      lgb.Dataset(X, label=y), num_boost_round=2)
        except Exception as e:
            probe[dev] = str(e).strip().splitlines()[0][:120]
    (out_dir / "lgb_gpu_probe.json").write_text(json.dumps(probe, indent=1))
    print("lgb gpu probe:", probe)


if __name__ == "__main__":
    sys.exit(main())
