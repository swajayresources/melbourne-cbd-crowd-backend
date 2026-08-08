"""ONNX Runtime inference engine module for fast sub-5ms predictions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


import src.config as cfg


class ONNXInferenceEngine:
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or (cfg.RESULTS_DIR / "onnx")
        self._sessions: Dict[str, Any] = {}
        self.onnx_available = False

        try:
            import onnxruntime as ort
            self.ort = ort
            self.onnx_available = True
        except ImportError:
            self.ort = None

    def is_available(self) -> bool:
        return self.onnx_available

    def load_model(self, model_name: str) -> bool:
        if not self.onnx_available:
            return False
        model_path = self.models_dir / f"{model_name}.onnx"
        if not model_path.exists():
            return False
        try:
            session = self.ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self._sessions[model_name] = session
            return True
        except Exception:
            return False

    def predict(self, model_name: str, features_array: np.ndarray) -> Optional[np.ndarray]:
        if model_name not in self._sessions:
            if not self.load_model(model_name):
                return None

        session = self._sessions[model_name]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        feats_float32 = features_array.astype(np.float32)
        if feats_float32.ndim == 1:
            feats_float32 = feats_float32.reshape(1, -1)

        try:
            preds = session.run([output_name], {input_name: feats_float32})[0]
            return np.array(preds).flatten()
        except Exception:
            return None
