# Convective Nowcasting Model Validation Report
**Model Version:** `ConvLSTM-Fusion-v1.0`  
**Generated:** 2026-09-27T16:00:00Z  
**Status:** DEMO_SYNTHETIC  

> [!IMPORTANT]
> **Scientific Honesty & Operational Integrity:**  
> Validation pending real dataset. Results generated on synthetic test distribution — not suitable for operational meteorology.

## 1. 0–6 Hour Lead-Time Performance (Normal Weather vs Extreme Events)

| Lead Time | RMSE (Norm) | MAE (Norm) | CSI (Norm) | POD (Norm) | FAR (Norm) | CSI (Extreme) | POD (Extreme) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| +30 min | 7.0663 | 4.3775 | 0.729 | 1.000 | 0.271 | 0.730 | 1.000 |
| +60 min | 6.9345 | 4.2664 | 0.760 | 1.000 | 0.240 | 0.760 | 1.000 |
| +90 min | 8.0268 | 5.0352 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| +120 min | 11.0007 | 6.2884 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| +150 min | 11.5454 | 6.3457 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| +180 min | 10.8822 | 5.9423 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| +210 min | 9.9445 | 5.4385 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| +240 min | 8.9458 | 4.9131 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| +270 min | 7.9319 | 4.3834 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| +300 min | 6.9150 | 3.8543 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| +330 min | 5.8995 | 3.3280 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| +360 min | 4.8888 | 2.8068 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## 2. Storm Tracking & Spatial Displacement Error

- **Mean Storm Centroid Error (3.0 km grid):** 0.50 km
- **Max Displacement Error:** 1.10 km

### Storm Track Divergence Across Forecast Horizon

| Forecast Horizon | Mean Track Error (km) |
|:---:|:---:|
| +30 min | 0.40 km |
| +60 min | 0.60 km |
| +90 min | 14.80 km |
| +120 min | 16.20 km |
| +150 min | 17.60 km |
| +180 min | 19.00 km |
| +210 min | 20.40 km |
| +240 min | 21.80 km |
| +270 min | 23.20 km |
| +300 min | 24.60 km |
| +330 min | 26.00 km |
| +360 min | 27.40 km |

## 3. High-Risk Hazard Overlap (Thunderstorm & Hail Core)

- **Hazard IoU (Intersection over Union):** 0.1430
- **Hazard CSI (Critical Success Index):** 0.1430
- **Hazard Precision:** 0.5050
- **Hazard Recall:** 0.1670

## 4. A/B Model Comparison (Across Lead Times)

| Lead Time | Persistence Baseline (RMSE) | ConvLSTM Baseline (RMSE) | Multi-Source Fusion (RMSE) |
|:---:|:---:|:---:|:---:|
| +30 min | 0.0000 | 28.7556 | **7.0663** |
| +60 min | 1.0305 | 28.8605 | **6.9345** |
| +90 min | 2.0611 | 28.9516 | **8.0268** |
| +120 min | 3.0916 | 29.0630 | **11.0007** |
| +150 min | 4.1221 | 29.2093 | **11.5454** |
| +180 min | 5.1526 | 29.4021 | **10.8822** |
| +210 min | 6.1832 | 29.6385 | **9.9445** |
| +240 min | 7.2137 | 29.9145 | **8.9458** |
| +270 min | 8.2442 | 30.2272 | **7.9319** |
| +300 min | 9.2747 | 30.5739 | **6.9150** |
| +330 min | 10.3053 | 30.9526 | **5.8995** |
| +360 min | 11.3358 | 31.3615 | **4.8888** |

## 5. Probability Calibration & Reliability

- **Hail Brier Score:** 0.0445
- **Calibration Status:** Calibration pending real dataset
