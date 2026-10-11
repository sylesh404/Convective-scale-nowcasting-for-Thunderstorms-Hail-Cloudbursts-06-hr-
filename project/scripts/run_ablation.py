"""
Ablation Study CLI Execution Script (SIH26084 Part 7).
Adheres strictly to Section 35 of Part 7 Specification:
Evaluates:
  A: Radar only
  B: Radar + Satellite
  C: Radar + Lightning
  D: Radar + NWP
  E: Radar + Satellite + Lightning
  F: Radar + Satellite + Lightning + NWP
  G: Full model
Outputs to experiments/results/ablation/
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.ablation_engine import AblationEngine


def main():
    print("=" * 60)
    print("NOWCAST AI — Multi-Source Sensor Ablation Study (Section 35)")
    print("=" * 60)

    device = "cpu"
    engine = AblationEngine(device=device)

    torch.manual_seed(42)
    test_x = torch.rand(2, 4, 12, 64, 64)
    test_y = torch.rand(2, 12, 4, 64, 64)

    print("[*] Running multi-sensor ablation forward passes (Exps A -> G)...")
    ablation_res = engine.run_ablation_study(test_x=test_x, test_y=test_y)
    complexity_res = engine.run_model_complexity_study()

    out_dir = PROJECT_ROOT / "experiments" / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "Demonstration evaluation only — synthetic distribution benchmark.",
            "ablation": ablation_res,
            "complexity": complexity_res
        }, f, indent=2)

    print(f"\n[*] Ablation study complete. Results saved to {out_file}")
    for row in ablation_res["ablation_table"]:
        print(f"    - {row['configuration']:<30} | CSI (+30m): {row['csi_30m']:.3f} | Latency: {row['latency_ms']} ms")


if __name__ == "__main__":
    main()
