# SIH Grand Finale Live Demo Script (Step-by-Step)
**Product:** NOWCAST AI — Convective Scale Weather Intelligence (SIH26084)

Follow this exact presentation sequence during the 10-minute judge demonstration:

---

### Step 1: System Launch & Authentication
- **Action:** Open browser at `http://localhost:8000/login.html`.
- **Narrative:** "NOWCAST AI features role-based meteorological mission control. We authenticate as an authorized duty meteorologist."
- **Action:** Click **Authenticate Session** to transition to `index.html`.

---

### Step 2: Mission Control & Data Sources Overview
- **Action:** Point to the top status bar.
- **Narrative:** "Notice our operational status: `SYNTHETIC DEMO MODE (MOSDAC Approval Pending)`. We maintain complete scientific honesty. The four primary sensor indicators are green and synchronized: DWR Radar, INSAT-3D Satellite, Lightning Network, and NCMRWF NWP."

---

### Step 3: Layer Exploration (Radar, Satellite, Lightning, NWP)
- **Action:** Click through the layer toggle switches:
  1. Toggle **Radar Reflectivity Layer** (shows 45+ dBZ convective cores).
  2. Toggle **Satellite Infrared Layer** (shows cold overshooting cloud tops).
  3. Toggle **Lightning Density Layer** (shows electrical discharge clusters).
  4. Toggle **NWP Thermodynamic Layer** (shows CAPE instability contour).
- **Narrative:** "Each layer represents an independent spatial stream regridded onto our unified 3.0 km grid."

---

### Step 4: Run 0–6 Hour Neural Nowcast
- **Action:** Click **Run Nowcast (0-6h)** in the action header.
- **Narrative:** "MultiSourceNowcastNet executes 12 future frames in just 14.8 milliseconds. Notice the 0 to 360-minute playback timeline at the bottom."
- **Action:** Drag the timeline slider to **+60 min** and **+120 min** to show storm evolution.

---

### Step 5: Storm Tracker & Kinematic Trajectory
- **Action:** Click on the active storm core on the GIS map.
- **Narrative:** "Our Lagrangian tracker isolates cell centroids and computes velocity vectors. Notice the predicted track corridor and velocity of 32 km/h heading East-Northeast."

---

### Step 6: Target Arrival & Infrastructure Impact
- **Action:** Click on a civil asset on the map or select a target location from the panel.
- **Narrative:** "The GIS impact engine performs real-time polygon intersection against civil asset layers:
  - 2 Schools within the hazard polygon
  - 1 Civil Hospital on alert
  - NH-44 highway segment exposed to severe downburst and hydroplaning risk
  - Estimated 30,700 exposed population."

---

### Step 7: Grounded AI Assistant Query
- **Action:** Open the **AI Assistant** drawer and ask:
  > *"What is the primary driver of the storm over Delhi NCR and when will it reach peak intensity?"*
- **Narrative:** "The assistant analyzes our active telemetry tensor and explains: high CAPE (2,400 J/kg) combined with an overshooting top (-53°C) and sudden lightning jump triggers peak intensity at T+60 minutes. Notice it never hallucinates facts outside the verified telemetry."

---

### Step 8: Automated Early-Warning Bulletin Generation
- **Action:** Click **Generate Meteorological Bulletin**.
- **Narrative:** "A formatted CAP-compliant (Common Alerting Protocol) bulletin is generated with exact coordinates, lead times, and recommended safety advisories."

---

### Step 9: Model Validation Dashboard & Benchmark Evidence
- **Action:** Click **Model Validation** in the navigation bar.
- **Narrative:** "Judges, here is our scientific evidence:
  - Quantitative 0–6 hour lead-time table showing CSI of 0.729 at +30m.
  - Baseline comparison against Persistence and Optical Flow.
  - Multi-sensor ablation study proving why satellite and lightning are essential.
  - Reliability calibration curves and physical limitations documentation."

---

### Step 10: Conclusion & Transition to Q&A
- **Narrative:** "NOWCAST AI bridges the critical nowcasting gap between satellite scans and disaster response with speed, multi-source intelligence, and scientific integrity. We are ready for your questions."
