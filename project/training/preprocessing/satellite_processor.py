"""
Satellite Processor for Convective Nowcasting Training Pipeline (SIH26084 Part 5).
Decodes and quality-controls INSAT-3D/3DR/3DS channels (IR, Water Vapour, Visible, Cloud products).
Adheres to Section 3 & 15 of Part 5 Specification.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class SatelliteProcessor:
    """
    Standardizes geostationary brightness temperatures and radiance fields.
    """

    def __init__(
        self,
        min_tb_kelvin: float = 180.0,
        max_tb_kelvin: float = 330.0
    ):
        self.min_tb = min_tb_kelvin
        self.max_tb = max_tb_kelvin

    def process_ir(self, raw_ir: np.ndarray) -> np.ndarray:
        """
        Processes Infrared / Thermal IR brightness temperatures.
        Deep convective cloud tops are characteristically cold (< 210 K).
        """
        cleaned = np.nan_to_num(raw_ir, nan=self.max_tb, posinf=self.max_tb, neginf=self.min_tb)
        cleaned = np.clip(cleaned, self.min_tb, self.max_tb)
        return cleaned.astype(np.float32)

    def process_water_vapour(self, raw_wv: np.ndarray) -> np.ndarray:
        """
        Processes upper-tropospheric water vapour (6.7 um band).
        """
        cleaned = np.nan_to_num(raw_wv, nan=self.max_tb, posinf=self.max_tb, neginf=self.min_tb)
        cleaned = np.clip(cleaned, self.min_tb, self.max_tb)
        return cleaned.astype(np.float32)

    def compute_convective_cloud_top_mask(self, ir_tb: np.ndarray, overshoot_threshold_k: float = 210.0) -> np.ndarray:
        """
        Identifies overshooting convective cloud tops penetrating the tropopause.
        """
        return (ir_tb <= overshoot_threshold_k).astype(np.float32)
