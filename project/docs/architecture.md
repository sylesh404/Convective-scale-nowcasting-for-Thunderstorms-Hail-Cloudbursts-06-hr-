# System Architecture Documentation — NOWCAST AI
**Problem Statement:** SIH26084 (Ministry of Earth Sciences / NCMRWF)  
**System:** NOWCAST AI — Convective Scale Weather Intelligence Platform

---

## 1. High-Level Architectural Layers

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                              │
│  - Meteorological Mission Control Dashboard (index.html)               │
│  - Live Interactive GIS Map & Radar Playback Timeline                  │
│  - Storm Tracking & Infrastructure Impact Panel                        │
│  - Grounded Conversational AI Assistant                                │
│  - Model Validation & Scientific Benchmark Dashboard                   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ HTTPS / REST / WebSockets
┌──────────────────────────────────▼─────────────────────────────────────┐
│                        API & BACKEND SERVICES                          │
│  - FastAPI Microservice (backend/app.py)                               │
│  - Ingestion Orchestrator & Multi-Source Quality Control               │
│  - 0-6 Hour Forecast Cycle Engine (13 steps)                           │
│  - GIS Spatial Analysis & Infrastructure Topology Intersect            │
│  - Validation API & Scientific Transparency Endpoints                  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                        MACHINE LEARNING SUBSYSTEM                      │
│  - Preprocessing: 3km Spatial Regridder & Temporal Aligner             │
│  - MultiSourceNowcastNet (Multi-Encoder ConvLSTM + Channel Attention)  │
│  - Inference Engine (inference/run_nowcast.py)                         │
│  - Baseline Suite & Multi-Source Ablation Engine                       │
│  - Confidence Calibration & Meteorological Attribution Engine          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                        DATA & STORAGE LAYER                            │
│  - Primary: Synthetic Meteorological Demo Archive                      │
│  - Future: MOSDAC/IMD Secure Ingestion Pipeline                        │
│  - File Cache, Redis State Cache & Experiment Tracking Registry        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Directory Layout
- `backend/app.py`: FastAPI server exposing `/api/nowcast`, `/api/run-forecast`, and validation routes.
- `training/models/nowcast_model.py`: MultiSourceNowcastNet and baseline neural definitions.
- `inference/run_nowcast.py`: Production inference service generating 12 forecast frames.
- `ml/`: Part 7 scientific validation, baselines, calibration, and explainability modules.
- `experiments/`: Reproducibility manifests, configs, results, and publication plots.
- `scripts/`: Operational evaluation and quality gate verification commands.
- `docs/`: Technical reports, model cards, data cards, and pitch decks.
