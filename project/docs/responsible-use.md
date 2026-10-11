# Responsible Use & Operational Governance Policy
**Product:** NOWCAST AI — Convective Scale Weather Intelligence  
**Problem Statement:** SIH26084 (Ministry of Earth Sciences / NCMRWF)

---

## 1. Operational Mandate & Non-Replacement Disclaimer
> [!IMPORTANT]
> **NOWCAST AI is an AI-powered meteorological decision-support tool.**  
> It is designed to assist meteorologists, researchers, and authorized disaster response planners. **IT DOES NOT REPLACE OFFICIAL WARNINGS ISSUED BY THE INDIA METEOROLOGICAL DEPARTMENT (IMD), THE NATIONAL DISASTER MANAGEMENT AUTHORITY (NDMA), OR STATE DISASTER MANAGEMENT AUTHORITIES (SDMA).**

Under Indian statutory law (Disaster Management Act, 2005), official weather warnings, colored alerts (Red, Orange, Yellow), and evacuation advisories may only be issued by legally designated government authorities.

---

## 2. Permitted & Recommended Uses
1. **Decision Support for Duty Forecasters:** Assisting operational meteorologists by synthesizing high-volume radar, satellite, lightning, and NWP feeds into a unified 0–6 hour situational picture.
2. **Infrastructure Impact Assessment:** Providing localized risk projections for airport runway operations, power transmission lines, highway corridors, and school districts.
3. **Research & Algorithm Benchmarking:** Evaluating multi-source neural fusion against standard persistence and optical flow extrapolation baselines.

---

## 3. Prohibited & High-Risk Uses
1. **Autonomous Evacuation Commands:** The system must never be configured to issue automated public evacuation orders without professional human meteorologist verification.
2. **Safety-of-Life Sole Decision Engine:** Aviation air-traffic routing, dam water discharge triggers, and school closure orders must not be made solely on raw model inference.
3. **Misrepresentation of Simulated Data:** In demonstration or sandbox mode, outputs must always remain labeled as `DEMO / SYNTHETIC` to prevent panic or misinterpretation.

---

## 4. Human-in-the-Loop Forecaster Override
The platform architecture provides forecasters with the explicit ability to:
- Override model-derived hazard levels.
- Annotate false alarms or radar ground-clutter artifacts.
- Issue calibrated advisory bulletins with verified source attribution.
