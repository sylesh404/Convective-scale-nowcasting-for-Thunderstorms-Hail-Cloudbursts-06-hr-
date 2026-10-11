"""
Physically Meaningful Normalization Engine for Convective Nowcasting (SIH26084 Part 5).
Adheres strictly to Section 7 of Part 5 Specification:
- Variable-specific scaling (Z-score and physical min/max scaling)
- Log-transformation for sparse, heavy-tailed lightning flash densities
- Generation and export of `normalization.json` fit EXCLUSIVELY on training data
- Exact statistical reuse during validation and production inference
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional


class NormalizationEngine:
    """
    Computes, saves, and applies variable-specific meteorological normalization parameters.
    """

    def __init__(self, stats_path: Optional[str] = None):
        self.stats_path = stats_path
        self.stats: Dict[str, Dict[str, float]] = {}

        if stats_path and os.path.exists(stats_path):
            self.load(stats_path)
        else:
            self._init_default_stats()

    def _init_default_stats(self):
        """
        Calibrated physical meteorological parameter distributions (default prior).
        """
        self.stats = {
            "radar_reflectivity": {
                "min": 0.0,
                "max": 75.0,
                "mean": 18.5,
                "std": 14.2,
                "type": "min_max"
            },
            "radar_velocity": {
                "min": -40.0,
                "max": 40.0,
                "mean": 0.0,
                "std": 12.0,
                "type": "symmetric_min_max"
            },
            "satellite_ir": {
                "min": 190.0,
                "max": 310.0,
                "mean": 265.0,
                "std": 28.0,
                "type": "inverted_ir"  # colder tops -> higher normalized value
            },
            "satellite_wv": {
                "min": 210.0,
                "max": 270.0,
                "mean": 242.0,
                "std": 14.0,
                "type": "min_max"
            },
            "lightning_density": {
                "min": 0.0,
                "max": 20.0,
                "mean": 0.35,
                "std": 1.25,
                "type": "log_transform"
            },
            "rainfall_qpe": {
                "min": 0.0,
                "max": 150.0,
                "mean": 4.8,
                "std": 12.5,
                "type": "log_transform"
            },
            "relative_humidity": {
                "min": 0.0,
                "max": 100.0,
                "mean": 72.0,
                "std": 18.0,
                "type": "min_max"
            },
            "temperature": {
                "min": -20.0,
                "max": 50.0,
                "mean": 24.0,
                "std": 9.5,
                "type": "z_score"
            },
            "nwp_precipitation": {
                "min": 0.0,
                "max": 100.0,
                "mean": 3.2,
                "std": 8.0,
                "type": "log_transform"
            },
            "nwp_cape": {
                "min": 0.0,
                "max": 5000.0,
                "mean": 1850.0,
                "std": 1100.0,
                "type": "min_max"
            },
            "wind_u": {
                "min": -40.0,
                "max": 40.0,
                "mean": 4.5,
                "std": 10.0,
                "type": "symmetric_min_max"
            },
            "wind_v": {
                "min": -40.0,
                "max": 40.0,
                "mean": -2.0,
                "std": 9.0,
                "type": "symmetric_min_max"
            }
        }

    def fit_from_training_arrays(self, channel_data: Dict[str, np.ndarray]):
        """
        Fits empirical empirical statistics EXCLUSIVELY on the training dataset split.
        """
        for var_name, arr in channel_data.items():
            if arr is None or arr.size == 0:
                continue
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                continue

            v_min = float(np.min(valid))
            v_max = float(np.max(valid))
            v_mean = float(np.mean(valid))
            v_std = max(float(np.std(valid)), 1e-4)

            existing_type = self.stats.get(var_name, {}).get("type", "min_max")
            self.stats[var_name] = {
                "min": v_min,
                "max": v_max,
                "mean": v_mean,
                "std": v_std,
                "type": existing_type
            }

    def normalize(self, var_name: str, data: np.ndarray) -> np.ndarray:
        """
        Applies variable-specific normalization to array -> returns [0.0, 1.0] or standardized values.
        """
        st = self.stats.get(var_name, {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 1.0, "type": "min_max"})
        t = st.get("type", "min_max")

        if t == "min_max":
            span = max(st["max"] - st["min"], 1e-4)
            return np.clip((data - st["min"]) / span, 0.0, 1.0).astype(np.float32)

        elif t == "symmetric_min_max":
            max_abs = max(abs(st["max"]), abs(st["min"]), 1e-4)
            # Map [-max_abs, +max_abs] -> [0.0, 1.0] with 0.5 as zero
            return np.clip((data + max_abs) / (2.0 * max_abs), 0.0, 1.0).astype(np.float32)

        elif t == "inverted_ir":
            # Colder cloud tops (e.g. 190 K) -> close to 1.0; warmer (310 K) -> 0.0
            span = max(st["max"] - st["min"], 1e-4)
            return np.clip((st["max"] - data) / span, 0.0, 1.0).astype(np.float32)

        elif t == "log_transform":
            # Log1p transform for heavy-tailed precipitation and lightning
            log_data = np.log1p(np.maximum(data, 0.0))
            max_log = np.log1p(max(st["max"], 1.0))
            return np.clip(log_data / max_log, 0.0, 1.0).astype(np.float32)

        elif t == "z_score":
            std = max(st["std"], 1e-4)
            z = (data - st["mean"]) / std
            # Sigmoid squashing to [0, 1] range for unified tensor
            return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

        return np.clip(data, 0.0, 1.0).astype(np.float32)

    def denormalize(self, var_name: str, norm_data: np.ndarray) -> np.ndarray:
        """
        Inverts normalized model outputs back to physical meteorological units.
        """
        st = self.stats.get(var_name, {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 1.0, "type": "min_max"})
        t = st.get("type", "min_max")

        if t == "min_max":
            span = st["max"] - st["min"]
            return (norm_data * span) + st["min"]

        elif t == "symmetric_min_max":
            max_abs = max(abs(st["max"]), abs(st["min"]))
            return (norm_data * 2.0 * max_abs) - max_abs

        elif t == "inverted_ir":
            span = st["max"] - st["min"]
            return st["max"] - (norm_data * span)

        elif t == "log_transform":
            max_log = np.log1p(max(st["max"], 1.0))
            return np.expm1(norm_data * max_log)

        return norm_data

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            self.stats = json.load(f)
