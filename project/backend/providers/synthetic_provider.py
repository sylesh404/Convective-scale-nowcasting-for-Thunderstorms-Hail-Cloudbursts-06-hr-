"""
Synthetic Weather Data Provider for SIH26084.
Implements Section 2 & 3 of Part 4 Specification:
- Fallback provider reading from synthetic_nowcasting_dataset.npz
- Provides individual source extractions:
  get_latest_satellite()
  get_latest_radar()
  get_latest_lightning()
  get_latest_nwp()
  get_latest_aws()
  get_latest_rainfall()
- Preserves get_latest_sequence() for existing ConvLSTM execution
"""

import os
import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base_provider import WeatherDataProvider
from .base import DataProvider


class SyntheticDataProvider(WeatherDataProvider, DataProvider):
    """
    Demonstration provider reading from synthetic_nowcasting_dataset.npz.
    Used while awaiting MOSDAC account approval and live IMD DWR feeds.
    """

    def __init__(self, dataset_path: Optional[str] = None):
        super().__init__(mode="demo")
        if dataset_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidates = [
                os.path.join(base_dir, "data", "synthetic_nowcasting_dataset.npz"),
                os.path.join(os.path.dirname(base_dir), "demo_pkg", "synthetic_nowcasting_dataset.npz"),
                os.path.join(os.getcwd(), "backend", "data", "synthetic_nowcasting_dataset.npz"),
                os.path.join(os.getcwd(), "data", "synthetic_nowcasting_dataset.npz")
            ]
            for c in candidates:
                if os.path.exists(c):
                    dataset_path = c
                    break

        if not dataset_path or not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Synthetic dataset file 'synthetic_nowcasting_dataset.npz' not found. "
                f"Searched: {candidates if 'candidates' in locals() else dataset_path}"
            )

        self.dataset_path = dataset_path
        self._data = np.load(dataset_path)
        self.X = self._data["X"]  # Shape: (N, 4, 4, 16, 16)
        self.num_sequences = len(self.X)
        self.Y = self._data["Y"] if "Y" in self._data else None

        # Update last simulated update timestamps to present
        now = datetime.datetime.now(datetime.timezone.utc)
        for k in self.last_update_times:
            self.last_update_times[k] = now

    def get_latest_sequence(self, sequence_idx: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        idx = sequence_idx % self.num_sequences
        seq = self.X[idx].astype(np.float32)  # Shape (4, 4, 16, 16)

        metadata = {
            "source": "SYNTHETIC DEMONSTRATION DATASET",
            "sequence_index": idx,
            "total_available_sequences": self.num_sequences,
            "is_synthetic": True,
            "is_live": False,
            "channel_labels": [
                "Synthetic Radar Reflectivity Proxy (0-1)",
                "Synthetic INSAT-3DR Thermal Proxy (0-1)",
                "Synthetic Lightning Strike Density Proxy (0-1)",
                "Synthetic Relative Humidity Proxy (0-1)"
            ],
            "input_frames": 4,
            "grid_dimensions": [16, 16],
            "interval_minutes": 30,
            "disclaimer": "Prototype demonstration data. MOSDAC live credentials pending."
        }
        return seq, metadata

    def get_latest_satellite(self) -> Dict[str, Any]:
        """
        Synthesizes INSAT-3D/3DR IR, Thermal IR, and Water Vapour proxy channels.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        self.last_update_times["satellite"] = now
        # Channel 1 in synthetic dataset is thermal IR proxy
        raw_slice = self.X[0, -1, 1].copy()
        return {
            "source": "INSAT-3DR (Synthetic Proxy)",
            "timestamp": now.isoformat(),
            "channels": {
                "ir": raw_slice.tolist(),
                "thermal_ir": (raw_slice * 1.05).clip(0, 1).tolist(),
                "water_vapour": (raw_slice * 0.85).clip(0, 1).tolist(),
                "visible": (raw_slice * 0.5).clip(0, 1).tolist()
            },
            "cloud_motion_vectors": {"u": 12.4, "v": -8.2, "unit": "m/s"},
            "extent": {"min_lat": 28.0, "max_lat": 32.0, "min_lon": 76.0, "max_lon": 80.0},
            "quality": "GOOD",
            "is_synthetic": True
        }

    def get_latest_radar(self) -> Dict[str, Any]:
        """
        Synthesizes DWR Max Reflectivity (dBZ) and Radial Velocity (m/s).
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        self.last_update_times["radar"] = now
        # Channel 0 in synthetic dataset is radar reflectivity proxy normalized 0-1
        norm_ref = self.X[0, -1, 0].copy()
        # Scale to realistic meteorological dBZ (0 - 70 dBZ)
        dbz_grid = (norm_ref * 70.0).round(1)
        velocity_grid = ((norm_ref - 0.5) * 40.0).round(1)  # -20 to +20 m/s
        return {
            "source": "IMD Doppler Weather Radar (Synthetic Proxy)",
            "station": "DWR Dehradun / Delhi Network",
            "timestamp": now.isoformat(),
            "reflectivity_dbz": dbz_grid.tolist(),
            "radial_velocity_ms": velocity_grid.tolist(),
            "max_reflectivity_dbz": float(np.max(dbz_grid)),
            "grid_dimensions": [16, 16],
            "quality": "GOOD",
            "is_synthetic": True
        }

    def get_latest_lightning(self) -> Dict[str, Any]:
        """
        Synthesizes 15-minute aggregated lightning strikes and density grid.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        self.last_update_times["lightning"] = now
        # Channel 2 is lightning strike density proxy
        density_grid = self.X[0, -1, 2].copy()
        total_strikes = int(np.sum(density_grid * 150))
        return {
            "source": "Lightning Detection Network (Synthetic Proxy)",
            "timestamp": now.isoformat(),
            "aggregation_window_minutes": 15,
            "density_grid": density_grid.tolist(),
            "total_strikes_15min": total_strikes,
            "peak_density_strikes_km2": round(float(np.max(density_grid) * 4.5), 2),
            "trend": "INCREASING" if np.mean(density_grid) > 0.3 else "STABLE",
            "quality": "GOOD",
            "is_synthetic": True
        }

    def get_latest_nwp(self) -> Dict[str, Any]:
        """
        Synthesizes NWP thermodynamic and kinematic boundary variables.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        self.last_update_times["nwp"] = now
        # Channel 3 is RH proxy
        rh_grid = (self.X[0, -1, 3] * 100.0).round(1)
        return {
            "source": "NCMRWF Unified Model / WRF (Synthetic Proxy)",
            "cycle_utc": now.strftime("%Y%m%d_%H00"),
            "timestamp": now.isoformat(),
            "cape_j_kg": 2850.0,
            "cin_j_kg": 35.0,
            "temperature_2m_c": 28.4,
            "relative_humidity_pct": rh_grid.tolist(),
            "wind_shear_0_6km_kts": 38.5,
            "u_wind_850hpa_ms": 14.2,
            "v_wind_850hpa_ms": 6.8,
            "geopotential_height_500hpa_gpm": 5860,
            "quality": "GOOD",
            "is_synthetic": True
        }

    def get_latest_aws(self) -> Dict[str, Any]:
        """
        Synthesizes Automatic Weather Station surface observations for ground QC.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        self.last_update_times["aws"] = now
        return {
            "source": "IMD Automatic Weather Station Network (Synthetic Proxy)",
            "timestamp": now.isoformat(),
            "station_count": 14,
            "observations": [
                {"station_id": "AWS-DED-01", "name": "Dehradun IMD", "temp_c": 27.2, "rh_pct": 88.0, "pressure_hpa": 932.4, "rain_rate_mm_hr": 34.0, "wind_speed_kmh": 42.0},
                {"station_id": "AWS-RSH-02", "name": "Rishikesh", "temp_c": 28.5, "rh_pct": 82.5, "pressure_hpa": 965.1, "rain_rate_mm_hr": 18.5, "wind_speed_kmh": 28.0},
                {"station_id": "AWS-SHM-03", "name": "Shimla Ridge", "temp_c": 19.4, "rh_pct": 92.0, "pressure_hpa": 782.0, "rain_rate_mm_hr": 45.0, "wind_speed_kmh": 54.0}
            ],
            "quality": "GOOD",
            "is_synthetic": True
        }

    def get_latest_rainfall(self) -> Dict[str, Any]:
        """
        Synthesizes fused radar/satellite Quantitative Precipitation Estimation (QPE).
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        self.last_update_times["rainfall"] = now
        ref_norm = self.X[0, -1, 0].copy()
        # Marshall-Palmer Z-R relation proxy: R = (Z/200)^(1/1.6)
        z = np.power(10.0, (ref_norm * 70.0) / 10.0)
        rain_rate = np.power(z / 200.0, 1.0 / 1.6).clip(0, 150.0).round(1)
        return {
            "source": "Multi-Sensor QPE (Synthetic Proxy)",
            "timestamp": now.isoformat(),
            "rain_rate_mm_hr": rain_rate.tolist(),
            "peak_rain_rate_mm_hr": float(np.max(rain_rate)),
            "mean_rain_rate_mm_hr": round(float(np.mean(rain_rate)), 2),
            "quality": "GOOD",
            "is_synthetic": True
        }

    def is_live(self) -> bool:
        return False
