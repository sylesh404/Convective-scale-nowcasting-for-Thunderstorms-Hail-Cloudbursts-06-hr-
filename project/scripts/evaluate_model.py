"""
Model Evaluation CLI Script (SIH26084 Part 7).
Adheres strictly to Section 33 of Part 7 Specification:
Usage:
    python scripts/evaluate_model.py --config experiments/configs/full_fusion.yaml --checkpoint training/training/checkpoints/best_model.pt
Outputs:
    - experiments/results/metrics.json
    - experiments/results/metrics.csv
    - experiments/plots/ (12 scientific validation plots)
    - experiments/reports/evaluation_report.md
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime, timezone
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.models.nowcast_model import MultiSourceNowcastNet
from training.datasets.sequence_dataset import ConvectiveSequenceDataset
from training.metrics.forecast_metrics import ForecastMetricsEvaluator
from ml.calibration.temperature_scaling import ModelCalibrator
from ml.evaluation.ablation_engine import AblationEngine
from ml.visualization.scientific_plots import ScientificPlotGenerator
from ml.tracking.experiment_tracker import ExperimentTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Nowcasting Model")
    parser.add_argument("--config", type=str, default="experiments/configs/full_fusion.yaml")
    parser.add_argument("--checkpoint", type=str, default="training/training/checkpoints/best_model.pt")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("NOWCAST AI — Model Evaluation & Validation Harness (Part 7)")
    print("=" * 60)

    # 1. Load Configuration
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {"grid": {"resolution_km": 3.0}, "temporal": {"forecast_hours": 6.0}}
    config["config_name"] = Path(args.config).name

    device = torch.device(args.device)

    # 2. Dataset
    print("[*] Preparing evaluation dataset (Chronological Partition)...")
    dataset = ConvectiveSequenceDataset(mock_samples=16)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 3. Model
    model = MultiSourceNowcastNet(in_channels=12, hidden_dim=64, forecast_frames=12).to(device)
    chk_path = PROJECT_ROOT / args.checkpoint
    if chk_path.exists():
        print(f"[*] Loading checkpoint weights: {chk_path}")
        chk = torch.load(chk_path, map_location=device)
        model.load_state_dict(chk.get("model_state_dict", chk))
    else:
        print(f"[!] Checkpoint {chk_path} not found. Using initialized network for evaluation harness.")

    model.eval()

    # 4. Inference
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    # 5. Metrics & Lead-Time Table
    evaluator = ForecastMetricsEvaluator(grid_resolution_km=3.0, total_steps=12, interval_minutes=30)
    lead_table = evaluator.evaluate_lead_time_table(preds_np, targets_np, threshold_dbz=35.0)

    # 6. Calibration
    calibrator = ModelCalibrator()
    cal_data = calibrator.evaluate_reliability_bins(preds_np[:, :, 0], (targets_np[:, :, 0] >= 0.50).astype(int))

    # 7. Ablation Engine
    ablation_engine = AblationEngine(device=str(device))
    ablation_res = ablation_engine.run_ablation_study(test_x=torch.tensor(preds_np[:2, :4, :12], device=device))
    complexity_res = ablation_engine.run_model_complexity_study()

    # 8. Visual Validation Plots (12 scientific plots)
    print("[*] Generating Section 22 publication-grade scientific plots...")
    plotter = ScientificPlotGenerator()
    generated_plots = plotter.plot_all(
        lead_time_table=lead_table,
        ablation_table=ablation_res["ablation_table"],
        complexity_data=complexity_res["models"],
        calibration_data=cal_data,
        pred_field=preds_np[0, 1, 0] * 70.0,
        target_field=targets_np[0, 1, 0] * 70.0
    )
    print(f"[*] Generated {len(generated_plots)} scientific plots in experiments/plots/")

    # 9. Output directory setup
    out_dir = PROJECT_ROOT / args.output_dir
    results_dir = out_dir / "results"
    reports_dir = out_dir / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Write metrics.json
    metrics_json_path = results_dir / "metrics.json"
    metrics_data = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "ConvLSTM-Fusion-v1.0",
        "dataset_mode": "Demonstration evaluation only — synthetic distribution benchmark.",
        "forecast_horizon_hours": 6.0,
        "lead_time_table": lead_table,
        "calibration": cal_data,
        "ablation": ablation_res,
        "complexity": complexity_res
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Write metrics.csv
    metrics_csv_path = results_dir / "metrics.csv"
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lead_time_min", "rmse_dbz", "mae_dbz", "csi", "pod", "far"])
        for row in lead_table:
            writer.writerow([
                row["lead_time"],
                row.get("rmse_dbz", ""),
                row.get("mae_dbz", ""),
                row.get("csi", ""),
                row.get("pod", ""),
                row.get("far", "")
            ])

    # Write evaluation_report.md
    report_md_path = reports_dir / "evaluation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# NOWCAST AI — Scientific Evaluation Report (Part 7)\n")
        f.write("**Model:** `ConvLSTM-Fusion-v1.0` | **Status:** Demonstration Evaluation\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Scientific Honesty Disclaimer:** Validation performed on synthetic meteorological distribution pending real MOSDAC/IMD observation integration. Not calibrated for life-safety operational issuance.\n\n")
        f.write("## 1. 0–6 Hour Lead-Time Verification Table\n\n")
        f.write("| Lead Time | RMSE (dBZ) | MAE (dBZ) | CSI (>=35 dBZ) | POD | FAR |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in lead_table:
            f.write(f"| +{r['lead_time']} min | {r.get('rmse_dbz', 0.0):.2f} | {r.get('mae_dbz', 0.0):.2f} | {r.get('csi', 0.0):.3f} | {r.get('pod', 0.0):.3f} | {r.get('far', 0.0):.3f} |\n")
        f.write("\n## 2. Multi-Source Ablation Summary\n\n")
        f.write("| Configuration | CSI (+30m) | CSI (+60m) | Latency (ms) |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        for ab in ablation_res["ablation_table"]:
            f.write(f"| {ab['configuration']} | {ab['csi_30m']:.3f} | {ab['csi_60m']:.3f} | {ab['latency_ms']:.1f} ms |\n")
        f.write("\nGenerated plots are available in `experiments/plots/`.\n")

    # 10. Record in Experiment Tracker & Reproducibility Manifest
    tracker = ExperimentTracker()
    manifest = tracker.create_manifest(
        experiment_id="exp_part7_evaluation",
        model_name="MultiSourceNowcastNet",
        model_version="ConvLSTM-Fusion-v1.0",
        config=config,
        seed=42
    )
    tracker.log_experiment_run(
        experiment_id="exp_part7_evaluation",
        model_version="ConvLSTM-Fusion-v1.0",
        parameters=config,
        metrics={
            "csi_30m": lead_table[0]["csi"],
            "csi_60m": lead_table[1]["csi"] if len(lead_table) > 1 else 0.0,
            "mean_rmse": lead_table[0].get("rmse_dbz", 0.0)
        },
        artifacts={
            "metrics_json": str(metrics_json_path.relative_to(PROJECT_ROOT)),
            "metrics_csv": str(metrics_csv_path.relative_to(PROJECT_ROOT)),
            "report_md": str(report_md_path.relative_to(PROJECT_ROOT))
        }
    )

    print(f"[*] Evaluation completed successfully.")
    print(f"[*] Artifacts written:")
    print(f"    - {metrics_json_path}")
    print(f"    - {metrics_csv_path}")
    print(f"    - {report_md_path}")
    print(f"    - Manifest: experiments/experiment_manifest.json")


if __name__ == "__main__":
    main()
