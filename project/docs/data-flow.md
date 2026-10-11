# Meteorological Data Flow Architecture
**System:** NOWCAST AI — Convective Scale Weather Intelligence (SIH26084)

---

## 1. End-to-End Data Pipeline Flow

```
[ RAW METEOROLOGICAL INGESTION ]
  ├─ Radar: Volume Reflectivity (Z, dBZ) & Doppler Velocity (Vr, m/s)
  ├─ Satellite: INSAT-3D Thermal IR (TIR-1) & Water Vapor (WV)
  ├─ Lightning: Flash Densities (flashes/100 km²) & Flash Rate of Change
  └─ NWP: NCUM Gridded Convective Fields (CAPE, CIN, PWAT, Shears)
                   │
                   ▼
[ STAGE 1: INGESTION QUALITY CONTROL & SANITIZATION ]
  ├─ Outlier threshold clipping (Reflectivity: [0, 75 dBZ], IR: [180, 320 K])
  ├─ Radar ground-clutter & non-meteorological echo filtering
  └─ Generation of per-channel Binary Availability Masks
                   │
                   ▼
[ STAGE 2: SPATIAL REGRIDDING & TEMPORAL SYNCHRONIZATION ]
  ├─ Projection to EPSG:4326 Geographical Lat/Lon
  ├─ Bilinear interpolation to 3.0 km common grid (64x64 cells)
  └─ 15-minute sliding temporal window alignment (T-45, T-30, T-15, T0)
                   │
                   ▼
[ STAGE 3: PHYSICAL NORMALIZATION ]
  ├─ Inverted IR normalization: Colder cloud tops -> higher positive activations
  ├─ Logarithmic lightning flash density scaling
  └─ Linear min-max scaling of thermodynamic fields to [0.0, 1.0]
                   │
                   ▼
[ STAGE 4: NEURAL TENSOR CONSTRUCTION ]
  └─ Shape: [Batch=1, Time=4, Channels=12, Height=64, Width=64]
                   │
                   ▼
[ STAGE 5: NEURAL INFERENCE (MultiSourceNowcastNet) ]
  ├─ Parallel multi-sensor spatial feature extraction (16-dim latent per sensor)
  ├─ Squeeze-and-Excitation channel attention recalibration
  ├─ Recurrent ConvLSTM temporal state propagation
  └─ Autoregressive decoding across 12 future frames (+30m to +360m)
                   │
                   ▼
[ STAGE 6: MULTI-HAZARD SYNTHESIS & IMPACT GEOMETRY ]
  ├─ Channel 0: Radar Reflectivity & Rain Rate
  ├─ Channel 1: Lightning Flash Density & Surge Risk
  ├─ Channel 2: Hail Probability
  ├─ Channel 3: Convective Intensity Index
  ├─ Lagrangian storm centroid tracking & trajectory extrapolation
  └─ GIS spatial intersection with Schools, Hospitals, Highways, Population
                   │
                   ▼
[ STAGE 7: API DISSEMINATION & MISSION CONTROL DASHBOARD ]
  ├─ GeoJSON storm polygons and hazard contour layers
  ├─ Grounded conversational AI assistant responses
  └─ Automated structured early-warning bulletins
```
