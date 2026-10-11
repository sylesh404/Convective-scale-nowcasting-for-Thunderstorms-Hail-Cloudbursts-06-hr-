# Deployment & Operations Guide
**System:** NOWCAST AI — Convective Scale Weather Intelligence (SIH26084)

---

## 1. Local Development Quickstart

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.10)
- PyTorch 2.0+
- Modern Web Browser (Chrome, Firefox, Edge)

### Step-by-Step Launch
1. **Navigate to the project root:**
   ```bash
   cd project
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the FastAPI application server:**
   ```bash
   python app.py
   ```
   *The server initializes and binds to `http://127.0.0.1:8000` (or `8080` if port 8000 is occupied).*
4. **Access the Mission Control Dashboard:**
   Open your browser to:
   `http://localhost:8000` or open `index.html` directly.

---

## 2. Docker Container Deployment

### Build the Docker Image
```bash
docker build -t nowcast-ai:v1.0 .
```

### Run Container via Docker Compose
```bash
docker-compose up -d
```
The microservice will be accessible at `http://localhost:8000` with automated health checks at `http://localhost:8000/`.

---

## 3. Automated Validation & Quality Gate Commands
- Run Part 5 neural training and inference test suite:
  ```bash
  pytest test_training_pipeline.py
  ```
- Run core FastAPI service and telemetry test suite:
  ```bash
  pytest test_app.py
  ```
- Run full scientific model evaluation harness:
  ```bash
  python scripts/evaluate_model.py
  ```
- Run multi-baseline benchmark:
  ```bash
  python scripts/run_baselines.py
  ```
- Run multi-sensor ablation study:
  ```bash
  python scripts/run_ablation.py
  ```
- Run operational readiness audit:
  ```bash
  python scripts/readiness_check.py
  ```
- Run final automated quality gate:
  ```bash
  python scripts/final_quality_gate.py
  ```
