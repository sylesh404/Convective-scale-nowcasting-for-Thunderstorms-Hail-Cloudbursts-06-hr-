"""
Lightweight Reproducible Experiment Tracker & Manifest Registry (SIH26084 Part 7).
Adheres strictly to Sections 3, 30, 31, and 32 of Part 7 Specification:
- Generates reproducible experiment manifest (seed, dataset hash, software/hardware metadata)
- Maintains experiments/index.json registry
- Tracks runs, parameters, metrics, and artifact references
- Integrates with Model Registry status workflow
"""

import os
import sys
import json
import time
import hashlib
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ExperimentTracker:
    """
    Local JSON-based experiment tracking system ensuring scientific provenance and full reproducibility.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or (PROJECT_ROOT / "experiments"))
        self.runs_dir = self.base_dir / "runs"
        self.results_dir = self.base_dir / "results"
        self.plots_dir = self.base_dir / "plots"
        self.index_file = self.base_dir / "index.json"
        self.manifest_file = self.base_dir / "experiment_manifest.json"

        # Ensure directory structure
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_index()

    def _ensure_index(self):
        if not self.index_file.exists():
            initial_index = {
                "system": "NOWCAST AI — SIH26084",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "experiments": {}
            }
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(initial_index, f, indent=2)

    def compute_dataset_hash(self, dataset_path: str) -> str:
        """Computes SHA-256 fingerprint of dataset file for tamper-proof provenance."""
        if not dataset_path:
            return "simulated-dataset-hash-demo-v1.0"
        p = Path(dataset_path)
        if not p.exists() or p.is_dir():
            return "simulated-dataset-hash-demo-v1.0"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()[:16]

    def create_manifest(
        self,
        experiment_id: str,
        model_name: str,
        model_version: str,
        config: Dict[str, Any],
        dataset_name: str = "convective_nowcast_archive",
        dataset_version: str = "v1.0-demo",
        seed: int = 42,
        duration_seconds: float = 0.0
    ) -> Dict[str, Any]:
        """
        Creates and writes the exact Section 3 experiment_manifest.json.
        """
        manifest = {
            "experiment_id": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "dataset": {
                "name": dataset_name,
                "version": dataset_version,
                "dataset_hash": self.compute_dataset_hash(config.get("dataset_path", "")),
                "chronological_split": True,
                "synthetic_mode": True
            },
            "model": {
                "name": model_name,
                "version": model_version,
                "resolution_km": config.get("grid", {}).get("resolution_km", 3.0),
                "forecast_hours": config.get("temporal", {}).get("forecast_hours", 6.0),
                "input_frames": config.get("temporal", {}).get("input_frames", 4),
                "forecast_frames": config.get("temporal", {}).get("forecast_frames", 12)
            },
            "configuration": config,
            "environment": {
                "software_version": "NOWCAST-AI-v1.0-SIH26084",
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "operating_system": platform.platform(),
                "processor": platform.processor() or "x86_64",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "cuda_available": torch.cuda.is_available()
            },
            "execution": {
                "training_duration_seconds": round(duration_seconds, 2),
                "reproducible_command": f"python scripts/evaluate_model.py --config experiments/configs/{config.get('config_name', 'full_fusion.yaml')}"
            }
        }

        # Write latest experiment manifest
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def log_experiment_run(
        self,
        experiment_id: str,
        model_version: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        artifacts: Dict[str, str],
        status: str = "DEMO"
    ) -> Dict[str, Any]:
        """
        Logs complete experiment run into index.json and runs/<experiment_id>.json.
        """
        run_record = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version,
            "status": status,
            "parameters": parameters,
            "metrics": metrics,
            "artifacts": artifacts
        }

        # Save individual run file
        run_file = self.runs_dir / f"{experiment_id}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2)

        # Update index.json
        with open(self.index_file, "r", encoding="utf-8") as f:
            idx = json.load(f)

        idx["experiments"][experiment_id] = {
            "timestamp": run_record["timestamp"],
            "model_version": model_version,
            "status": status,
            "lead_30m_csi": metrics.get("csi_30m", metrics.get("csi", "N/A")),
            "lead_60m_csi": metrics.get("csi_60m", "N/A"),
            "mean_rmse": metrics.get("mean_rmse", "N/A"),
            "run_file": str(run_file.relative_to(PROJECT_ROOT))
        }

        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=2)

        return run_record

    def get_experiment_history(self) -> Dict[str, Any]:
        """Loads index.json history."""
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"experiments": {}}
