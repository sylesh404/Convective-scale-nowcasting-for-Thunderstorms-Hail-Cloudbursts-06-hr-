# Model Card — MultiSourceNowcastNet (v1.0)
**Product:** NOWCAST AI — Convective Scale Weather Intelligence  
**Problem Statement:** SIH26084 (Ministry of Earth Sciences / NCMRWF)  
**Standard:** Adheres to Mitchell et al. (2019) Model Cards for Model Reporting

---

## 1. Model Details
- **Model Name:** `MultiSourceNowcastNet`
- **Model Version:** `ConvLSTM-Fusion-v1.0`
- **Model Type:** Spatiotemporal Recurrent Neural Network (Convolutional LSTM with Channel Attention Fusion)
- **Framework:** PyTorch (>= 2.0.0)
- **License:** Open Research / Government Demonstration License
- **Contact:** SIH26084 MoES Development Team

---

## 2. Intended Use & Scope
- **Primary Domain:** High-resolution (3.0 km) convective nowcasting for thunderstorms, severe lightning, hail risk, and cloudburst proxies across India.
- **Forecast Horizon:** 0 to 6 hours (+30 min, +60 min, ..., +360 min) at 30-minute intervals.
- **Intended Users:** State disaster management authorities (SDMA), airport meteorological offices, agricultural advisory boards, weather researchers.
- **Out-of-Scope Use:** MUST NOT be used as the sole automated trigger for civil evacuations without human meteorologist in the loop. Not calibrated for long-range synoptic weather (>24 hours).

---

## 3. Architecture & Data Inputs
### Spatial Input Grid
- $64 \times 64$ common grid (extensible to sub-domains) at 3.0 km horizontal spatial resolution.
- Bounding Coordinates: Lat 8.0°N–38.0°N, Lon 68.0°E–98.0°E.

### 12 Ingested Observation Channels
1. **Radar Reflectivity ($Z$, dBZ):** IMD Doppler Weather Radar (0–75 dBZ).
2. **Doppler Radial Velocity ($V_r$, m/s):** Inbound/outbound hydrometeor kinematics.
3. **INSAT-3D Thermal Infrared (TIR-1, K):** Deep convective cloud-top brightness temperature.
4. **INSAT-3D Water Vapor (WV, K):** Upper-tropospheric moisture transport.
5. **Lightning Flash Density (flashes/100 $\text{km}^2$):** Optical ground/satellite flash rate.
6. **Lightning Flash Jump Index:** Rapid rate of electrical intensification.
7. **NWP CAPE (J/kg):** Convective Available Potential Energy (NCMRWF NCUM).
8. **NWP CIN (J/kg):** Convective Inhibition barrier.
9. **NWP PWAT (mm):** Precipitable Water column depth.
10. **NWP 500 hPa U-wind (m/s):** Mid-tropospheric steering wind component.
11. **NWP 500 hPa V-wind (m/s):** Mid-tropospheric steering wind component.
12. **NWP 850 hPa Temperature (K):** Low-level thermal advection.

### 4 Operational Prediction Heads
- **Head 1 (Rainfall Intensity):** Calibrated surface radar reflectivity [0, 75 dBZ].
- **Head 2 (Lightning Flash Density):** Severe electrical hazard density [0, 1].
- **Head 3 (Hail Risk Probability):** Binary/multiclass severe hail probability [0.0, 1.0].
- **Head 4 (Convective Intensity Index):** Severe updraft vigor composite index [0, 100].
- **Head 5 (Downburst Velocity):** *Explicit placeholder* — currently reported as "Downburst model unavailable — awaiting Doppler downdraft ground truth."

---

## 4. Training & Validation Setup
- **Dataset Partitioning:** Strict chronological split (70% train, 15% validation, 15% testing) to eliminate future temporal data leakage.
- **Loss Function:** MultiHazardLoss = $1.5 \cdot \mathcal{L}_{\text{rain}} + 1.0 \cdot \mathcal{L}_{\text{ltg}} + 1.2 \cdot \mathcal{L}_{\text{hail}} + 0.8 \cdot \mathcal{L}_{\text{conv}}$, boosted by $2.5\times$ on severe convective extremes.
- **Hardware:** Optimized for CPU demonstration fallback and NVIDIA CUDA GPU acceleration.
- **Inference Latency:** 14.8 ms per 6-hour forecast (64x64 grid on CPU).

---

## 5. Metrics & Performance (Synthetic Demonstration Benchmark)
| Horizon | RMSE (dBZ) | MAE (dBZ) | CSI (>=35 dBZ) | POD | FAR | Mean Track Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| +30 min | 7.07 | 4.38 | 0.729 | 1.000 | 0.271 | 0.40 km |
| +60 min | 6.93 | 4.27 | 0.760 | 1.000 | 0.240 | 0.60 km |
| +180 min | 10.88 | 5.94 | 0.000* | 0.000* | 0.000* | 19.00 km |

*\*Note: Metrics at extended horizons reflect rapid cellular dissipation in synthetic test events.*

---

## 6. Known Limitations & Failure Cases
1. **Radar Attenuation:** Heavy precipitation cores cause beam attenuation in C/X-band DWR, requiring satellite IR gap-filling.
2. **Rapid Convective Initiation:** Explosive cells forming in <20 minutes may lag by one forecast cycle (15-min cadence).
3. **Sparse Lightning in Warm Rain:** Mountainous warm-cloud cloudbursts with low mixed-phase ice produce negligible lightning.
4. **Hail Label Verification:** Ground hail verification networks are sparse across India; hail probabilities are model-derived proxies.

---

## 7. Ethical & Responsible AI Considerations
- **Non-Replacement:** Model outputs supplement, but do not override, official warnings issued by the India Meteorological Department (IMD).
- **Scientific Honesty:** Clear indicators are displayed when synthetic data is in use or when specific hazards lack calibrated ground truth.
