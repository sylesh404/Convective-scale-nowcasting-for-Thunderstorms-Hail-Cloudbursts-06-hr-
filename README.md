# SIH26084 - Convective scale nowcasting for Thunderstorms, Hail & Cloudbursts (06 hr)

[![Smart India Hackathon 2026](https://img.shields.io/badge/SIH-2026-blue.svg)](https://sih.gov.in)
[![Category](https://img.shields.io/badge/Category-Software-emerald.svg)](https://sih.gov.in)
[![Ministry / Org](https://img.shields.io/badge/Organization-Ministry%20of%20Earth%20Sciences%20(MoES)-indigo.svg)]()
[![Theme](https://img.shields.io/badge/Theme-Disaster%20Management-purple.svg)]()
[![Domain](https://img.shields.io/badge/Domain-Landslide%20&%20Slope%20Stability%20GIS-orange.svg)]()

---

## 🎯 Problem Statement Overview
- **Problem Statement ID:** `SIH26084`
- **Title:** Convective scale nowcasting for Thunderstorms, Hail & Cloudbursts (06 hr)
- **Sponsoring Organization:** Ministry of Earth Sciences (MoES)
- **Department:** National Centre for Medium Range Weather Forecasting (NCMRWF)
- **Category:** Software
- **Theme:** Disaster Management

### 📖 Official Description
Convective storms, such as severe thunderstorms, hail, downburst winds, and cloudbursts are among Indiaâ€™s deadliest natural hazards, especially during the pre-monsoon and monsoon seasons.Despite advancements in Numerical Weather Prediction (NWP) models, traditional systems often fail to accurately capture these mesoscale extreme weather events.The primary limitation stems from spatial and temporal constraints: these violent storms develop rapidly within a window of minutes and occur at localized scales that slip through coarse grid resolutions. Current early warning infrastructures struggle to provide high-resolution, short-term forecasts (0â€“6 hours), leaving local administrations, aviation sectors, and rural farming communities vulnerable to sudden, devastating impacts.The challenge is to build a real-time, convective-scale Nowcasting System (0â€“6 hour lead time) operating at a hyper-local 1â€“3 km spatial resolution.Because traditional physics-based models are too computationally slow to simulate these rapid developments in real-time, participants must design a system rooted in Multi-Source Data Fusion architectures. The core objective is to ingest high-frequency, heterogeneous meteorological streams, automatically detect early convective initiation, and dynamically forecast severe storm parameters (including lightning strike density, hail probability, downburst velocity, and cloudburst thresholds).Design a robust, real-time ingestion engine that fuses data streams from multiple sources: Doppler Weather Radars (DWR - reflectivity and velocity fields), geostationary satellite imagery (INSAT-3D/3DR thermal/infrared bands), and ground-based lightning detection networks. A real-time,interactive GIS-mapped dashboard showcasing high-resolution (1â€“3 km) hazard zones with live countdown clocks for storm arrivals.

---

## 💡 Proposed Solution Architecture
Our team has engineered a comprehensive, production-ready solution tailored specifically for **Ministry of Earth Sciences (MoES)**:
1. **Interactive Mission Control Dashboard (`project/index.html`):** Glassmorphic dark-mode web application featuring real-time Chart.js telemetry, interactive parameter tuning, automated simulation triggers, and exportable audit logs.
2. **FastAPI Microservice Engine (`project/app.py`):** High-throughput Python REST backend with Pydantic v2 validation, domain-specific AI anomaly scoring, cryptographic audit logs, and OpenAPI Swagger documentation.
3. **Comprehensive Technical Whitepaper (`project/solution.md`):** Complete architectural breakdown, mathematical formulations, PostgreSQL + PostGIS DDL schemas, and deployment topologies.
4. **Automated Test Suite (`project/test_app.py`):** Built-in unit and integration tests verifying all REST endpoints.
5. **Turnkey Containerization (`project/Dockerfile` & `project/docker-compose.yml`):** Ready for one-command deployment on Docker / Kubernetes.

---

## 🚀 Quick Start Guide

### Option 1: Instant Browser Demo (Zero Setup)
Simply open `project/index.html` in any modern web browser or serve it locally:
```bash
cd "SIH26084 - Convective scale nowcasting for Thunderstorms, Hail & Cloudbursts (06 hr)/project"
python -m http.server 8080
```
Open [http://localhost:8080](http://localhost:8080) to access the command center.

### Option 2: Run Full Python FastAPI Microservice
```bash
cd "SIH26084 - Convective scale nowcasting for Thunderstorms, Hail & Cloudbursts (06 hr)/project"
pip install -r requirements.txt
python app.py
```
- API Server: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive OpenAPI Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Option 3: Run Automated Tests
```bash
cd "SIH26084 - Convective scale nowcasting for Thunderstorms, Hail & Cloudbursts (06 hr)/project"
pytest test_app.py -v
```

---

## 📂 Project Repository Structure
```plaintext
SIH26084 - Convective scale nowcasting for Thunderstorms, Hail & Cloudbursts (06 hr)/
├── README.md                           # Main problem statement pitch & guide
├── problem_statement.json              # Official SIH 2026 metadata
└── project/
    ├── index.html                      # Interactive Dark-Mode Glassmorphism Web App
    ├── app.py                          # FastAPI REST API Microservice
    ├── test_app.py                     # Pytest automated test suite
    ├── solution.md                     # Deep-dive 8-section technical whitepaper
    ├── requirements.txt                # Python backend dependencies
    ├── Dockerfile                      # Production Docker container definition
    ├── docker-compose.yml              # Multi-container orchestration
    └── README.md                       # Project execution manual
```
