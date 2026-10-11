"""
Validation Report Generator
Part 5 — SIH26084 MoES / NCMRWF Convective-Scale Nowcasting (0-6 hr)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Validation Report Markdown")
    parser.add_argument("--json", type=str, default="training/validation/validation_results.json", help="Path to validation results JSON")
    parser.add_argument("--output", type=str, default="training/validation/VALIDATION_REPORT.md", help="Path to markdown output")
    return parser.parse_args()


def generate_markdown(data: Dict[str, Any]) -> str:
    version = data.get("model_version", "ConvLSTM-Fusion-v1.0")
    timestamp = data.get("timestamp", "N/A")
    normal = data.get("normal_events", {})
    extreme = data.get("extreme_events", {})

    status_note = normal.get("scientific_honesty_note", "Validation pending real dataset.")

    md = []
    md.append(f"# Convective Nowcasting Model Validation Report")
    md.append(f"**Model Version:** `{version}`  ")
    md.append(f"**Generated:** {timestamp}  ")
    md.append(f"**Status:** {normal.get('status', 'DEMO_SYNTHETIC')}  \n")

    md.append("> [!IMPORTANT]")
    md.append(f"> **Scientific Honesty & Operational Integrity:**  ")
    md.append(f"> {status_note}\n")

    md.append("## 1. 0–6 Hour Lead-Time Performance (Normal Weather vs Extreme Events)\n")
    md.append("| Lead Time | RMSE (Norm) | MAE (Norm) | CSI (Norm) | POD (Norm) | FAR (Norm) | CSI (Extreme) | POD (Extreme) |")
    md.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    norm_leads = normal.get("lead_time_metrics", [])
    extr_leads = extreme.get("lead_time_metrics", [])
    extr_map = {row["lead_step"]: row for row in extr_leads}

    for row in norm_leads:
        step = row["lead_step"]
        mins = row["lead_time_minutes"]
        ex_row = extr_map.get(step, {})
        md.append(
            f"| +{mins} min | {row.get('rmse', 0):.4f} | {row.get('mae', 0):.4f} | "
            f"{row.get('csi', 0):.3f} | {row.get('pod', 0):.3f} | {row.get('far', 0):.3f} | "
            f"{ex_row.get('csi', 0):.3f} | {ex_row.get('pod', 0):.3f} |"
        )
    md.append("")

    md.append("## 2. Storm Tracking & Spatial Displacement Error\n")
    disp_norm = normal.get("spatial_displacement", {})
    md.append(f"- **Mean Storm Centroid Error (3.0 km grid):** {disp_norm.get('mean_displacement_km', 0.0):.2f} km")
    md.append(f"- **Max Displacement Error:** {disp_norm.get('max_displacement_km', 0.0):.2f} km\n")

    md.append("### Storm Track Divergence Across Forecast Horizon\n")
    md.append("| Forecast Horizon | Mean Track Error (km) |")
    md.append("|:---:|:---:|")
    for trk in normal.get("track_divergence_km", []):
        md.append(f"| +{trk['lead_time_minutes']} min | {trk['mean_error_km']:.2f} km |")
    md.append("")

    md.append("## 3. High-Risk Hazard Overlap (Thunderstorm & Hail Core)\n")
    haz = normal.get("hazard_overlap", {})
    md.append(f"- **Hazard IoU (Intersection over Union):** {haz.get('iou', 0.0):.4f}")
    md.append(f"- **Hazard CSI (Critical Success Index):** {haz.get('csi', 0.0):.4f}")
    md.append(f"- **Hazard Precision:** {haz.get('precision', 0.0):.4f}")
    md.append(f"- **Hazard Recall:** {haz.get('recall', 0.0):.4f}\n")

    md.append("## 4. A/B Model Comparison (Across Lead Times)\n")
    ab = normal.get("ab_comparison", {})
    md.append("| Lead Time | Persistence Baseline (RMSE) | ConvLSTM Baseline (RMSE) | Multi-Source Fusion (RMSE) |")
    md.append("|:---:|:---:|:---:|:---:|")
    pers_data = ab.get("Persistence", [])
    cl_data = ab.get("ConvLSTM_Baseline", [])
    ms_data = ab.get("MultiSource_ConvLSTM", [])

    cl_map = {r["lead_step"]: r.get("rmse", 0) for r in cl_data} if cl_data else {}
    ms_map = {r["lead_step"]: r.get("rmse", 0) for r in ms_data} if ms_data else {}

    for r in pers_data:
        st = r["lead_step"]
        m = r["lead_time_minutes"]
        p_rmse = r.get("rmse", 0)
        c_rmse = cl_map.get(st, "N/A")
        m_rmse = ms_map.get(st, "N/A")
        c_str = f"{c_rmse:.4f}" if isinstance(c_rmse, (int, float)) else str(c_rmse)
        m_str = f"{m_rmse:.4f}" if isinstance(m_rmse, (int, float)) else str(m_rmse)
        md.append(f"| +{m} min | {p_rmse:.4f} | {c_str} | **{m_str}** |")
    md.append("")

    md.append("## 5. Probability Calibration & Reliability\n")
    cal = normal.get("hail_calibration", {})
    md.append(f"- **Hail Brier Score:** {cal.get('brier_score', 0.0):.4f}")
    md.append(f"- **Calibration Status:** {cal.get('status', 'Calibration unavailable')}\n")

    return "\n".join(md)


def main():
    args = parse_args()
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"[!] Results JSON {json_path} does not exist. Please run validate.py first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report_md = generate_markdown(data)
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[*] Validation report written to {out_file}")


if __name__ == "__main__":
    main()
