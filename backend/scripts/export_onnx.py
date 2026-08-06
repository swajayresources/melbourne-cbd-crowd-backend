"""Export LightGBM models to ONNX format binaries for fast sub-5ms production inference."""
from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import config as cfg

def export_models_to_onnx():
    models_dir = cfg.RESULTS_DIR / "real"
    output_dir = cfg.RESULTS_DIR / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning models in: {models_dir}")
    model_files = list(models_dir.glob("*.model"))
    print(f"Found {len(model_files)} trained model files.")

    try:
        import lightgbm as lgb
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        onnx_tools_available = True
    except ImportError:
        onnx_tools_available = False
        print("Note: onnxmltools not installed. To convert models to ONNX, install: pip install onnxmltools skl2onnx")

    converted_count = 0
    for model_path in model_files:
        out_onnx = output_dir / f"{model_path.stem}.onnx"
        if onnx_tools_available and model_path.stem.startswith("lgb"):
            try:
                booster = lgb.Booster(model_file=str(model_path))
                initial_types = [("input", FloatTensorType([None, len(cfg.FEATURES)]))]
                onnx_model = onnxmltools.convert_lightgbm(booster, initial_types=initial_types)
                onnxmltools.utils.save_model(onnx_model, str(out_onnx))
                print(f"[EXPORTED] {out_onnx.name}")
                converted_count += 1
            except Exception as e:
                print(f"[SKIP] Failed to convert {model_path.name}: {e}")
        else:
            print(f"[INFO] Prepared export path for: {out_onnx.name}")

    print(f"ONNX export completed. {converted_count} models exported to {output_dir}")

if __name__ == "__main__":
    export_models_to_onnx()
