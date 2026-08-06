"""CrowdService module managing sensor thresholding, sensory classification, and live feeds."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from app.crowd import CrowdEngine, CBD_CENTER, CBD_BOUNDS


class CrowdService:
    def __init__(self, data_path: Optional[Path] = None, loc_path: Optional[Path] = None):
        self.engine = CrowdEngine(data_path=data_path, loc_path=loc_path)

    @property
    def locations_df(self) -> pd.DataFrame:
        return self.engine.locations_df

    def get_thresholds(self, location_id: int) -> Dict[str, float]:
        return self.engine.get_thresholds(location_id)

    def classify_sensory_level(self, location_id: int, count: float) -> Tuple[str, str, str]:
        return self.engine.classify_sensory_level(location_id, count)

    def classify_level(self, location_id: int, count: float) -> str:
        return self.engine.classify_level(location_id, count)

    def predict_rule_count(self, location_id: int, dt: pd.Timestamp) -> float:
        return self.engine.predict_rule_count(location_id, dt)

    def find_nearest_sensor(self, lat: float, lon: float) -> Optional[int]:
        return self.engine.find_nearest_sensor(lat, lon)

    def resolve_location_query(self, query: str) -> Optional[Tuple[int, str, float, float]]:
        return self.engine.resolve_location_query(query)

    def get_all_sensors_map_data(self, current_counts: Dict[int, float]) -> List[Dict[str, Any]]:
        return self.engine.get_all_sensors_map_data(current_counts)
