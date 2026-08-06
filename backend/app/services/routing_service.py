"""RoutingService providing address geocoding, OSRM foot routing, and sensory route evaluation."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd

from app.crowd import (
    CrowdEngine,
    geocode_location,
    fetch_osrm_route,
    evaluate_route_crowds,
)


class RoutingService:
    def __init__(self, crowd_engine: CrowdEngine):
        self.crowd_engine = crowd_engine

    def geocode(self, query: str) -> List[Dict[str, Any]]:
        return geocode_location(query, self.crowd_engine)

    def fetch_routes(
        self, orig_lat: float, orig_lon: float, dest_lat: float, dest_lon: float
    ) -> Tuple[List[Dict[str, Any]], bool]:
        return fetch_osrm_route(orig_lat, orig_lon, dest_lat, dest_lon)

    def evaluate_routes(
        self,
        routes: List[Dict[str, Any]],
        dt: pd.Timestamp,
        mode: str = "rule",
        forecast_service: Any = None,
    ) -> Dict[str, Any]:
        return evaluate_route_crowds(
            routes=routes,
            dt=dt,
            crowd_engine=self.crowd_engine,
            mode=mode,
            service=forecast_service,
        )
