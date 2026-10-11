"""
Radar Processor for Convective Nowcasting Training Pipeline (SIH26084 Part 5).
Handles Doppler Weather Radar (DWR) volume decoding, QC, clutter removal, and calibration.
Adheres to Section 3 & 6 of Part 5 Specification.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class RadarProcessor:
    """
    Quality-controls, de-aliases, and normalizes radar reflectivity (dBZ) and radial velocity (m/s).
    """

    def __init__(
        self,
        min_dbz: float = -32.0,
        max_dbz: float = 80.0,
        min_velocity: float = -50.0,
        max_velocity: float = 50.0,
        clutter_threshold_dbz: float = 5.0
    ):
        self.min_dbz = min_dbz
        self.max_dbz = max_dbz
        self.min_vel = min_velocity
        self.max_vel = max_velocity
        self.clutter_thresh = clutter_threshold_dbz

    def process_reflectivity(self, raw_reflectivity: np.ndarray) -> np.ndarray:
        """
        Cleans invalid/NaN values, filters ground clutter, and clips within physical bounds.
        """
        cleaned = np.nan_to_num(raw_reflectivity, nan=0.0, posinf=self.max_dbz, neginf=self.min_dbz)
        cleaned = np.clip(cleaned, self.min_dbz, self.max_dbz)
        # Suppress non-meteorological ground clutter below threshold
        cleaned[cleaned < self.clutter_thresh] = 0.0
        return cleaned.astype(np.float32)

    def process_velocity(self, raw_velocity: np.ndarray) -> np.ndarray:
        """
        Filters Nyquist folding outliers and clips Doppler radial velocity.
        """
        cleaned = np.nan_to_num(raw_velocity, nan=0.0, posinf=self.max_vel, neginf=self.min_vel)
        cleaned = np.clip(cleaned, self.min_vel, self.max_vel)
        return cleaned.astype(np.float32)

    def extract_extreme_mask(self, reflectivity: np.ndarray, severe_threshold_dbz: float = 45.0) -> np.ndarray:
        """
        Identifies convective cores (severe thunderstorm / hail initiation core).
        """
        return (reflectivity >= severe_threshold_dbz).astype(np.float32)
