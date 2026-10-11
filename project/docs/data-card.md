# Data Card — Meteorological Ingestion & Demonstration Datasets
**Product:** NOWCAST AI — Convective Scale Weather Intelligence  
**Problem Statement:** SIH26084 (Ministry of Earth Sciences / NCMRWF)  
**Standard:** Adheres to Gebru et al. (2021) Datasheets for Datasets

---

## 1. Overview & Data Architecture Separation
NOWCAST AI supports a dual-pipeline ingestion architecture designed to transition seamlessly from **Synthetic Demonstration Mode** to **Operational Government Data Mode** without altering downstream inference pipelines or user interfaces:

```
[Synthetic Mode]               [Operational Mode]
Physics-Guided Simulator       MOSDAC / IMD Secure API (Pending Approval)
       │                                     │
       └──────────────┬──────────────────────┘
                      ▼
            Unified Preprocessing
       (Regridding, Masking, Normalization)
                      ▼
               12-Channel Tensor
```

---

## 2. Ingestion Pipeline Sources

### Source 1: Doppler Weather Radar (DWR)
- **Primary Agency:** India Meteorological Department (IMD)
- **Coverage:** Indian sub-continent radar network (S-band & C-band installations)
- **Spatial Resolution:** 500 m to 1 km polar coordinates, regridded to 3.0 km Cartesian grid
- **Temporal Resolution:** 10–15 minutes (volume scan cycle)
- **Variables:** Equivalent radar reflectivity factor ($Z$, dBZ), Doppler radial velocity ($V_r$, m/s), spectrum width ($\sigma_v$, m/s)
- **Format:** HDF5, NetCDF4, or raw UF
- **Quality Limitations:** Ground clutter, beam attenuation in torrential rain, mountainous beam blockage.
- **Missing Data Handling:** Binary availability mask (0.0), spatial fallback to satellite brightness temperature.

### Source 2: Geostationary Satellite Imagery
- **Primary Agency:** Space Applications Centre (SAC) / ISRO / MOSDAC
- **Satellites:** INSAT-3D, INSAT-3DR (Rapid Scan Mode)
- **Coverage:** $45^\circ\text{E} - 105^\circ\text{E}$, $10^\circ\text{S} - 45^\circ\text{N}$
- **Spatial Resolution:** 4.0 km (Thermal IR & Water Vapor)
- **Temporal Resolution:** 15 minutes (routine), 4.5 minutes (rapid-scan over India)
- **Variables:** Thermal Infrared 1 (TIR-1, $10.8\,\mu\text{m}$, Brightness Temp K), Water Vapor (WV, $6.8\,\mu\text{m}$, K)
- **Missing Data Handling:** Inverted IR normalization ($300\text{ K} \to 0.0$, $200\text{ K} \to 1.0$), temporal forward filling up to 30 min.

### Source 3: Lightning Detection Networks
- **Primary Sources:** IITM Damini Ground Lightning Network / INSAT-3DS Imager Proxy
- **Coverage:** Pan-India continental domain
- **Spatial Resolution:** Point events clustered into $3.0\text{ km} \times 3.0\text{ km}$ spatial bins
- **Temporal Resolution:** Continuous event streaming aggregated into 15-minute integration windows
- **Variables:** Total Lightning Flash Count (IC + CG), Peak Flash Rate of Change ($\Delta F/\Delta t$)
- **Missing Data Handling:** Log-transformed density normalization; absence of signal correctly treated as physical zero (no electrical discharge).

### Source 4: Numerical Weather Prediction (NWP)
- **Primary Agency:** National Centre for Medium Range Weather Forecasting (NCMRWF)
- **Models:** NCMRWF Unified Model (NCUM, 4.0 km convection-permitting) & Unified Ensemble (NEPS)
- **Temporal Cadence:** 6-hourly cycle initialization with hourly forecast output
- **Variables:** CAPE (J/kg), CIN (J/kg), Precipitable Water (PWAT, mm), 500 hPa U/V Wind Shears (m/s), 850 hPa Temp (K)
- **Format:** GRIB2 / NetCDF4

---

## 3. Synthetic Demonstration Dataset Specification
- **Storage Location:** `training/datasets/` & internal simulator
- **Status:** **DEMO_SYNTHETIC**
- **Generation Method:** 2D advection-diffusion reaction model simulating rotating supercellular convective storms, electrical lightning surge kinematics, and cold-pool advection.
- **Provenance Tag:** `is_synthetic: true` permanently encoded into every forecast payload to guarantee scientific honesty and avoid mistaking simulated storms for live observations.
