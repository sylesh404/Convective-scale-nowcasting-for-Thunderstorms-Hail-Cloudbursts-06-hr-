"""
NWP Processor for Convective Nowcasting Training Pipeline (SIH26084 Part 5).
Ingests and aligns thermodynamic and kinematic atmospheric environment fields:
- CAPE (Convective Available Potential Energy)
- CIN (Convective Inhibition)
- Temperature & Relative Humidity
- U & V wind vectors (850 hPa & deep shear)
- Background NWP Precipitation
Adheres to Section 3 & 17 of Part 5 Specification.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class NWPProcessor:
    """
    Standardizes gridded numerical weather prediction fields into model channels.
    """

    def __init__(self):
        pass

    def process_thermodynamics(
        self,
        cape: np.ndarray,
        cin: np.ndarray,
        temp_c: np.ndarray,
        rh_pct: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Cleans and physically bounds thermodynamic sounding parameters.
        """
        cape_clean = np.clip(np.nan_to_num(cape, nan=0.0), 0.0, 7000.0).astype(np.float32)
        cin_clean = np.clip(np.nan_to_num(cin, nan=0.0), 0.0, 500.0).astype(np.float32)
        temp_clean = np.clip(np.nan_to_num(temp_c, nan=25.0), -60.0, 60.0).astype(np.float32)
        rh_clean = np.clip(np.nan_to_num(rh_pct, nan=50.0), 0.0, 100.0).astype(np.float32)

        return {
            "cape": cape_clean,
            "cin": cin_clean,
            "temperature": temp_clean,
            "relative_humidity": rh_clean
        }

    def process_kinematics(
        self,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        precip_mm: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Cleans wind vector components and background NWP precipitation forecast.
        """
        u_clean = np.clip(np.nan_to_num(u_wind, nan=0.0), -80.0, 80.0).astype(np.float32)
        v_clean = np.clip(np.nan_to_num(v_wind, nan=0.0), -80.0, 80.0).astype(np.float32)
        precip_clean = np.clip(np.nan_to_num(precip_mm, nan=0.0), 0.0, 300.0).astype(np.float32)

        return {
            "u_wind": u_clean,
            "v_wind": v_clean,
            "nwp_precipitation": precip_clean
        }
