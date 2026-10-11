# NOWCAST AI — Scientific Evaluation Report (Part 7)
**Model:** `ConvLSTM-Fusion-v1.0` | **Status:** Demonstration Evaluation

> [!IMPORTANT]
> **Scientific Honesty Disclaimer:** Validation performed on synthetic meteorological distribution pending real MOSDAC/IMD observation integration. Not calibrated for life-safety operational issuance.

## 1. 0–6 Hour Lead-Time Verification Table

| Lead Time | RMSE (dBZ) | MAE (dBZ) | CSI (>=35 dBZ) | POD | FAR |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ++30 min min | 7.04 | 4.34 | 0.728 | 1.000 | 0.272 |
| ++60 min min | 6.90 | 4.24 | 0.760 | 1.000 | 0.240 |
| ++90 min min | 8.00 | 5.01 | 0.000 | 0.000 | 0.000 |
| ++120 min min | 10.99 | 6.26 | 0.000 | 0.000 | 0.000 |
| ++150 min min | 11.53 | 6.32 | 0.000 | 0.000 | 0.000 |
| ++180 min min | 10.87 | 5.92 | 0.000 | 0.000 | 0.000 |
| ++210 min min | 9.94 | 5.42 | 0.000 | 0.000 | 0.000 |
| ++240 min min | 8.94 | 4.90 | 0.000 | 0.000 | 0.000 |
| ++270 min min | 7.93 | 4.37 | 0.000 | 0.000 | 0.000 |
| ++300 min min | 6.91 | 3.84 | 0.000 | 0.000 | 0.000 |
| ++330 min min | 5.89 | 3.32 | 0.000 | 0.000 | 0.000 |
| ++360 min min | 4.89 | 2.80 | 0.000 | 0.000 | 0.000 |

## 2. Multi-Source Ablation Summary

| Configuration | CSI (+30m) | CSI (+60m) | Latency (ms) |
|:---|:---:|:---:|:---:|
| Radar Only | 0.531 | 0.528 | 169.3 ms |
| Radar + Satellite | 0.531 | 0.528 | 244.1 ms |
| Radar + Lightning | 0.531 | 0.528 | 236.9 ms |
| Radar + NWP | 0.000 | 0.000 | 341.5 ms |
| Radar + Sat + Lightning | 0.216 | 0.000 | 313.8 ms |
| Radar + Sat + Ltg + NWP | 0.531 | 0.528 | 429.2 ms |
| Full Multi-Source Model | 0.531 | 0.528 | 547.4 ms |

Generated plots are available in `experiments/plots/`.
