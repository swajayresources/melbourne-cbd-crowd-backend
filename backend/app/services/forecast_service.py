"""ForecastService module providing ML predictions, CQR calibration, and exceedance probabilities."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from app.forecast_service import Service
from app.crowd import estimate_exceedance_probability
from app.services.onnx_service import ONNXInferenceEngine


class ForecastService:
    def __init__(self, data: str = "real"):
        self.service = Service(data=data)
        self.onnx_engine = ONNXInferenceEngine()

    def get_forecast(self, location_id: int, at: Optional[pd.Timestamp] = None, frameworks: Tuple[str, ...] = ("lgb",)) -> Dict[str, Any]:
        return self.service.forecast(location_id=location_id, at=at, frameworks=frameworks)

    def sensor_list(self) -> List[Dict[str, Any]]:
        return self.service.sensor_list()

    def history(self, location_id: int, hours: int = 168) -> List[Dict[str, Any]]:
        return self.service.history(location_id=location_id, hours=hours)

    def refresh_feed(self, use_demo: bool = False) -> Dict[str, Any]:
        return self.service.refresh_feed(use_demo=use_demo)

    def calculate_exceedance_prob(self, q10: float, q50: float, q90: float, p75_threshold: float) -> float:
        return estimate_exceedance_probability(q10, q50, q90, p75_threshold)
