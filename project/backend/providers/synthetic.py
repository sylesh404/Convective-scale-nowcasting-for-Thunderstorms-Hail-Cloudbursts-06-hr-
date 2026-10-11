"""
SIH26084 Convective Scale Nowcasting - Synthetic Demonstration Data Provider
Loads the offline calibrated synthetic demonstration dataset (synthetic_nowcasting_dataset.npz).
Clearly flags all data as synthetic demonstration proxies.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
from .base import DataProvider


class SyntheticDataProvider(DataProvider):
    """
    Demonstration provider reading from synthetic_nowcasting_dataset.npz.
    Used while awaiting MOSDAC account approval and live IMD DWR feeds.
    """

    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            # Search candidate paths relative to this file or project root
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

    def get_provider_status(self) -> Dict[str, Any]:
        return {
            "provider_name": "SyntheticDataProvider",
            "status": "ACTIVE_DEMO",
            "is_live": False,
            "dataset_file": os.path.basename(self.dataset_path),
            "total_sequences": self.num_sequences,
            "channels": 4,
            "grid": "16x16",
            "mosdac_approval_status": "Awaiting account approval"
        }

    def is_live(self) -> bool:
        return False
