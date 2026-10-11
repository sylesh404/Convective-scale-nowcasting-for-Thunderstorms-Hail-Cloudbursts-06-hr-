# SIH26084 Synthetic Nowcasting Demo

## Run
pip install -r requirements.txt
python run_forecast.py

Open `demo_dashboard.html` in a browser.

## Included
- `synthetic_nowcasting_dataset.npz` — synthetic multi-source sequences
- `nowcast_demo.pt` — lightweight pretrained ConvLSTM demo model
- `nowcast_model.py` — model architecture
- `train_model.py` — retraining script
- `run_forecast.py` — inference script
- `forecast_demo.npz` — generated prediction
- `demo_dashboard.html` — visual prototype

## Synthetic channels
1. Radar reflectivity proxy
2. INSAT thermal/cloud proxy
3. Lightning density proxy
4. Humidity proxy

## Targets
1. Rain/radar intensity
2. Lightning density
3. Hail-risk proxy

The data are synthetic and are ONLY for demonstrating the software/model pipeline. They are not real weather observations or an operational warning system.

Production upgrade:
DWR reflectivity + radial velocity + INSAT + lightning + AWS + NWP
→ temporal/spatial alignment → ConvLSTM/U-Net fusion
→ 0–6 hour multi-hazard forecast → GIS dashboard.
