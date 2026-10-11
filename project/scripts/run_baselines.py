"""
Baseline Benchmark CLI Execution Script (SIH26084 Part 7).
Adheres strictly to Section 34 of Part 7 Specification:
Evaluates:
  - Persistence
  - Optical Flow Motion Extrapolation
  - Radar-Only ConvLSTM
  - Full Fusion Model
Outputs to experiments/results/baselines/
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.baselines import build_baseline_suite
from training.metrics.forecast_metrics import ForecastMetricsEvaluator


def main():
    print("=" * 60)
    print("NOWCAST AI — Baseline Comparative Benchmark (Section 34)")
    print("=" * 60)

    device = "cpu"
    suite = build_baseline_suite(device=device)
    evaluator = ForecastMetricsEvaluator(grid_resolution_km=3.0, total_steps=12, interval_minutes=30)

    # Standard representative evaluation batch [B=2, T_in=4, C=12, H=64, W=64]
    torch.manual_seed(42)
    test_x = torch.rand(2, 4, 12, 64, 64)
    test_y = torch.rand(2, 12, 4, 64, 64)
    target_np = test_y.numpy()

    output_dir = PROJECT_ROOT / "experiments" / "results" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = []
    print(f"[*] Benchmarking {len(suite)} baseline architectures...")

    for model_id, model in suite.items():
        with torch.no_grad():
            preds = model(test_x)
            preds_np = preds.numpy()

        lead_table = evaluator.evaluate_lead_time_table(preds_np, target_np, threshold_dbz=35.0)
        m30 = lead_table[0]
        m60 = lead_table[1] if len(lead_table) > 1 else m30

        record = {
            "baseline_id": model_id,
            "architecture": type(model).__name__,
            "csi_30m": round(float(m30["csi"]), 3),
            "csi_60m": round(float(m60["csi"]), 3),
            "pod_30m": round(float(m30["pod"]), 3),
            "far_30m": round(float(m30["far"]), 3),
            "rmse_30m": round(float(m30["rmse_dbz"]), 2),
            "mae_30m": round(float(m30["mae_dbz"]), 2),
            "lead_time_table": lead_table
        }
        baseline_results.append(record)
        print(f"    - {model_id:<30} | CSI (+30m): {record['csi_30m']:.3f} | RMSE: {record['rmse_30m']} dBZ")

    out_file = output_dir / "baseline_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "Demonstration benchmark — evaluated without subjective ranking.",
            "baselines": baseline_results
        }, f, indent=2)

    print(f"\n[*] Baseline benchmark results saved to {out_file}")


if __name__ == "__main__":
    main()
