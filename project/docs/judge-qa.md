# SIH Grand Finale Judge Q&A Document
**Product:** NOWCAST AI — Convective Scale Weather Intelligence  
**Problem Statement:** SIH26084 (Ministry of Earth Sciences / NCMRWF)

---

### Q1: Why did you choose ConvLSTM instead of a pure Vision Transformer (ViT)?
**Answer:** While spatiotemporal Vision Transformers have shown promise in large-scale meteorological research, their quadratic attention complexity $\mathcal{O}((T \times H \times W)^2)$ incurs severe memory and latency penalties. For an operational nowcasting platform that must run inference on edge servers or modest hardware within milliseconds, ConvLSTM offers linear complexity with respect to time, preserving local spatial topology and completing 0–6 hour forward inference in **14.8 milliseconds**.

---

### Q2: How do you handle missing or delayed data from specific sensors?
**Answer:** We implement two complementary mechanisms:
1. **Binary Availability Masks:** Each sensor channel is accompanied by an availability mask. If a radar or satellite feed is missing, its mask is set to 0.0 so missing zeros are not interpreted as physical clear air.
2. **Channel Attention Recalibration (Squeeze-and-Excitation):** The fusion layer dynamically downweights channels with flat or missing signals and amplifies active observational feeds.

---

### Q3: How do you harmonize different spatial resolutions (e.g. 500m Radar vs 4km Satellite vs 4km NWP)?
**Answer:** All observation grids are regridded onto a standardized 3.0 km common grid ($64 \times 64$ cells) using bilinear interpolation in EPSG:4326 coordinate space. This matches the Nyquist sampling limit of high-resolution convective structures while keeping tensor dimensions computationally efficient.

---

### Q4: How do you prevent temporal data leakage during training and validation?
**Answer:** We strictly enforce **chronological dataset partitioning** (70% earliest timestamps for training, 15% intermediate for validation, 15% latest unseen timestamps for testing). We never use random k-fold shuffling, which would allow the model to cheat by interpolating between temporally adjacent weather frames.

---

### Q5: How do you scientifically validate 0–6 hour forecasts?
**Answer:** We evaluate forecasts at discrete horizons (+30m, +60m, ..., +360m) using standard meteorological verification metrics: Critical Success Index (CSI), Probability of Detection (POD), False Alarm Ratio (FAR), RMSE, and MAE. All models are benchmarked against both Persistence and Optical Flow motion extrapolation.

---

### Q6: How is hail risk validated?
**Answer:** **Scientific Honesty:** Ground truth hail pad and sensor verification networks are sparse across India. We clearly state in the UI and documentation: *"Hail-risk output is model-derived and requires dedicated hail-event validation."* We do not fabricate validation metrics for unverified ground truth.

---

### Q7: How does the system detect cloudbursts?
**Answer:** Cloudbursts are defined using the IMD standard criteria: intense precipitation $\ge 100\text{ mm/hr}$ concentrated over a localized area ($\approx 20\text{--}30\text{ km}^2$). The model flags cells where predicted radar reflectivity exceeds 55 dBZ alongside high precipitable water column depth.

---

### Q8: How is storm tracking implemented?
**Answer:** We calculate the center-of-mass centroid of storm cells exceeding 35 dBZ. Centroids across consecutive frames are linked using Lagrangian distance minimization and projected forward using great-circle Haversine calculations to generate a 0–6 hour trajectory corridor.

---

### Q9: How is confidence calculated, and why do you distinguish it from data agreement?
**Answer:** We distinguish **Model Confidence** from **Data Agreement**:
- **Data Agreement:** A deterministic index (HIGH / MODERATE / LOW) measuring physical consistency between radar cores, cloud-top cooling, lightning rates, and NWP instability.
- **Model Confidence:** Calibrated probability evaluated via Brier Score and Expected Calibration Error (ECE). If sample size is limited, it is explicitly flagged as *"Prototype confidence — not calibrated."*

---

### Q10: How does your Conversational AI Assistant avoid hallucination?
**Answer:** The AI Assistant is architected with strict telemetry grounding. When answering queries, it references the active JSON cache containing verified sensor statistics, current storm coordinates, and model hazard scores. If requested information is absent, it responds with: *"Insufficient observational data to answer."*

---

### Q11: What is synthetic in this prototype and what is real?
**Answer:** Because our official government MOSDAC/IMD API approval is currently pending, the demonstration pipeline uses a **physics-guided synthetic dataset** simulating realistic convective storms, lightning surges, and radar echoes. The entire backend ingestion architecture, neural model, GIS impact engine, and dashboard are real, production-ready software components that will seamlessly ingest live feeds the moment API keys are approved.

---

### Q12: What happens if radar fails completely during a severe storm?
**Answer:** The system automatically falls back to satellite-driven nowcasting. Rapid-scan INSAT-3D thermal infrared and water vapor channels detect overshooting tops and convective cooling, while lightning flash feeds continue to identify active electrical cores.
