# 2-Minute Technical Pitch
**Problem Statement:** SIH26084 (Ministry of Earth Sciences / NCMRWF)  
**Product:** NOWCAST AI — Convective Scale Weather Intelligence

---

### Minute 1: The Problem & Data Architecture
Respected Judges and MoES Officials:

Every pre-monsoon and monsoon season, localized convective storms like Kalbaisakhi in Bengal and cloudbursts in the Himalayan foothills strike with little warning. Traditional Numerical Weather Prediction models require 3 to 6 hours for data assimilation and execution, leaving duty forecasters blind in the crucial 0 to 6-hour nowcasting window. Meanwhile, existing radar optical flow extrapolation fails beyond 60 minutes because it cannot model storm initiation or decay.

We created **NOWCAST AI** to solve this exact gap.

Our system ingests 12 channels of multi-source data:
- IMD Doppler Weather Radar for hydrometeor reflectivity ($Z$) and velocity ($V_r$).
- INSAT-3D Thermal IR and Water Vapor channels to observe cloud-top cooling.
- Lightning flash density and flash jump indices from ground/satellite networks.
- NCMRWF Unified Model NWP fields for thermodynamic instability (CAPE, CIN, and vertical wind shear).

All feeds are harmonized onto a standard 3.0 km common grid with binary missing-data masks.

---

### Minute 2: The Neural Engine, Impact Analysis & Provenance
At the core of NOWCAST AI is **MultiSourceNowcastNet** — a deep neural architecture combining 4 dedicated sensor encoders, Squeeze-and-Excitation channel attention recalibration, and a recurrent ConvLSTM spatiotemporal backbone.

In just **14.8 milliseconds**, the model forecasts 12 future frames out to +360 minutes across 4 prediction heads:
1. Calibrated radar rainfall intensity
2. Lightning flash density
3. Hail risk probability
4. Convective intensity index

Next, our Lagrangian storm tracker computes storm centroids, trajectory vectors, and spatial displacement corridors. Our GIS impact engine automatically intersects storm hazard polygons with civil infrastructure layers, identifying specific schools, hospitals, and highway corridors in harm's way. Forecasters can also query our grounded Conversational AI Assistant, which references active telemetry tensors without hallucinating.

Crucially, we maintain rigorous scientific honesty: we benchmarked against Persistence and Optical Flow, recorded full provenance in reproducible manifests, and clearly report demonstration status until official live MOSDAC API credentials are approved.

Thank you.
