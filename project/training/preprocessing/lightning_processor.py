"""
Lightning Processor for Convective Nowcasting Training Pipeline (SIH26084 Part 5).
Converts point strike detections into 15-minute spatio-temporal density grids and trend maps.
Adheres to Section 3 & 16 of Part 5 Specification.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class LightningProcessor:
    """
    Transforms discrete lightning stroke records into continuous density and trend fields.
    """

    def __init__(
        self,
        grid_resolution_km: float = 3.0,
        max_flashes_per_cell: float = 250.0
    ):
        self.resolution_km = grid_resolution_km
        self.max_flashes = max_flashes_per_cell
        self.cell_area_km2 = max(self.resolution_km ** 2, 1.0)

    def aggregate_strikes_to_density(
        self,
        strikes: List[Dict[str, Any]],
        grid_shape: Tuple[int, int],
        bbox: Dict[str, float]
    ) -> np.ndarray:
        """
        Bins strike coordinates into 2D grid and calculates flash density (flashes / km^2 / 15-min).
        """
        H, W = grid_shape
        counts = np.zeros((H, W), dtype=np.float32)
        min_lat = bbox["min_lat"]
        max_lat = bbox["max_lat"]
        min_lon = bbox["min_lon"]
        max_lon = bbox["max_lon"]

        for s in strikes:
            lat = s.get("lat", 0.0)
            lon = s.get("lon", 0.0)
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                y = int(np.clip((max_lat - lat) / (max_lat - min_lat) * H, 0, H - 1))
                x = int(np.clip((lon - min_lon) / (max_lon - min_lon) * W, 0, W - 1))
                counts[y, x] += 1.0

        density = counts / self.cell_area_km2
        return density.astype(np.float32)

    def compute_lightning_trend(
        self,
        current_density: np.ndarray,
        previous_density: np.ndarray
    ) -> np.ndarray:
        """
        Calculates spatial gradient of lightning intensification (trend map).
        """
        return (current_density - previous_density).astype(np.float32)
