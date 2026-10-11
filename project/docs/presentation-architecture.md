# Architecture Presentation Slide Content
**Product:** NOWCAST AI — Convective Scale Weather Intelligence (SIH26084)

---

## Slide 1: End-to-End Operational Architecture

### 1. Data Ingestion Layer
- **Multi-Source Feeds:** IMD Doppler Weather Radar, INSAT-3D/3DR Geostationary Imager, Ground/Satellite Lightning Networks, NCMRWF NWP Convective Indices.
- **Unified Harmonization:** 3.0 km spatial regridding, 15-minute temporal sliding synchronization, inverted IR normalization, and binary missing-data masks.

### 2. AI / Neural Modeling Layer
- **MultiSourceNowcastNet:** 4 parallel sensor encoders (16-dim latent spaces) with Squeeze-and-Excitation channel attention recalibration.
- **Spatiotemporal Engine:** Recurrent ConvLSTM backbone maintaining 2D spatial memory topology across 12 autoregressive decoding steps.
- **Multi-Hazard Heads:** Simultaneous inference for Surface Reflectivity, Lightning Density, Hail Risk, and Convective Intensity Index.

### 3. Forecast & GIS Impact Layer
- **Kinematic Storm Tracker:** Center-of-mass centroid tracking with Lagrangian velocity extrapolation and trajectory dispersion corridors.
- **GIS Topology Engine:** Spatial polygon intersection against Schools, Hospitals, Highway Corridors, and Gridded Population Density.

### 4. Application & Interface Layer
- **Mission Control Dashboard:** High-contrast dark-mode WebGL/Leaflet GIS interface with interactive timeline playback.
- **Grounded AI Assistant:** Conversational query engine strictly bounded to active observation and forecast telemetry.
- **Automated Reporting:** CAP-compliant meteorological bulletin generator with instant PDF/Markdown export.
