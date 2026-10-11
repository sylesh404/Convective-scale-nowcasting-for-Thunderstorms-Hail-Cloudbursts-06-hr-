# Scientific Limitations & Boundary Conditions
**Product:** NOWCAST AI — Convective Scale Weather Intelligence  
**Problem Statement:** SIH26084 (Ministry of Earth Sciences / NCMRWF)

---

## 1. Physical & Sensor Limitations

### 1.1 Radar Beam Geometry & Coverage Horizons
- **Earth Curvature & Beam Broadening:** Doppler Weather Radar beams ascend with distance. At 150–200 km from the radar tower, the center of the beam is often 3–5 km above ground level, overshooting shallow or low-topped convective storms.
- **Topographic Beam Blockage:** In mountainous terrain (Himalayas, Western Ghats), prominent ridgelines cause complete or partial beam blockage, resulting in radar blind shadows.
- **Heavy Precipitation Attenuation:** In C-band and X-band radars, torrential precipitation cores (>= 55 dBZ) absorb and scatter electromagnetic pulses, severely attenuating echoes behind the leading storm wall.

### 1.2 Satellite Infrared Limitations
- **Cirrus Canopy Obscuration:** Expanding anvil clouds and upper-tropospheric cirrus shields can mask low-level convective reinvigoration occurring beneath the cloud deck.
- **Parallax Error:** Geostationary observation angles from 74°E (INSAT-3D) introduce spatial displacement of high cloud tops relative to their true surface coordinate, requiring trigonometric parallax correction.

### 1.3 Lightning Detection Constraints
- **Mixed-Phase Microphysics Requirement:** Cloud electrification requires non-inductive collisions between graupel, ice crystals, and supercooled water. Purely warm-rain convective cloudbursts without ice phase produce little to no lightning despite catastrophic rainfall rates.

---

## 2. Modeling & AI Algorithmic Limitations

### 2.1 Spatiotemporal Blurring in Autoregressive Models
- Like all recurrent convolutional architectures, ConvLSTM experiences progressive spatial diffusion beyond 2 to 3 hours lead time due to $L_2/L_1$ minimization. Fine-scale convective cores tend to spread out and decrease in peak intensity.

### 2.2 Rapid Convective Initiation (RCI)
- Convective storms initiating from clear-air boundary layer convergence within a 15-minute sampling window cannot be anticipated prior to the first detection of cloud-top cooling or radar echoes.

### 2.3 Non-Linear Cell Mergers & Splitting
- Complex multicell dynamics—such as cold pool collisions, rear-inflow jet development, and supercell splitting—are governed by 3D non-hydrostatic Navier-Stokes thermodynamics that 2D recurrent networks can only approximate empirically.

---

## 3. Data & Validation Limitations

### 3.1 Sparse Ground Truth for Hail and Downbursts
- India lacks dense automated hailpad networks and surface Doppler anemometer rings. Consequently, hail risk is a model-derived proxy, and downburst forecasting is explicitly marked as unavailable until labeled downdraft velocities are integrated.

### 3.2 Cloudburst Event Rarity
- Cloudbursts (>= 100 mm/h over limited areas) are extreme outliers in historical meteorological archives, resulting in extreme class imbalance during model training.

### 3.3 Domain & Seasonal Generalization
- A model trained on pre-monsoon thunderstorm events (Nor'westers / Kalbaisakhi) in Northeast India exhibits domain shift when applied to winter western disturbances in North India or monsoon depressions in Central India.
