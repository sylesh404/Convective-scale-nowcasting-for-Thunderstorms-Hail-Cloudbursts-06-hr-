"""
Automated Pytest Suite for SIH26084 FastAPI Microservice
Tests NOWCAST AI PyTorch ConvLSTM nowcasting endpoints, impact engine,
top risk locations, storm lifecycle, active alerts, and legacy telemetry.
"""

import os
import sys
import importlib.util
import pytest
from fastapi.testclient import TestClient

app_path = os.path.join(os.path.dirname(__file__), "app.py")
spec = importlib.util.spec_from_file_location("app_sih26084", app_path)
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
app = app_module.app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["problem_id"] == "SIH26084"
    assert data["status"] == "OPERATIONAL"
    assert "timestamp" in data
    assert data["model_status"] == "loaded"

def test_model_status_endpoint():
    response = client.get("/api/model-status")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "ConvLSTM"
    assert data["status"] == "loaded"
    assert data["device"] == "cpu"
    assert data["input_channels"] == 4
    assert data["forecast_steps"] == 6
    assert data["forecast_interval_minutes"] == 30
    assert data["data_mode"] == "synthetic_demo"

def test_run_forecast_endpoint():
    response = client.post("/api/run-forecast", json={"sequence_index": 0})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["mode"] == "synthetic_demo"
    assert "forecast_steps" in data
    assert len(data["forecast_steps"]) == 6

    # Test Step 0
    step0 = data["forecast_steps"][0]
    assert step0["lead_minutes"] == 0
    assert "rain_intensity" in step0
    assert "lightning_density" in step0
    assert "hail_probability" in step0
    assert "hazard_level" in step0
    assert step0["hazard_level"] in ["LOW", "MODERATE", "HIGH", "SEVERE"]
    assert "hazard_score" in step0
    assert 0.0 <= step0["hazard_score"] <= 1.0

    # Test 2D Grids
    assert len(step0["radar_grid"]) == 16
    assert len(step0["radar_grid"][0]) == 16
    assert len(step0["lightning_grid"]) == 16
    assert len(step0["hail_grid"]) == 16

    # Test Storm Track
    assert "storm_track" in data
    assert len(data["storm_track"]) == 6
    assert data["storm_track"][0]["lead_minutes"] == 0
    assert "norm_x" in data["storm_track"][0]
    assert "norm_y" in data["storm_track"][0]

    # Test Impact and Risk Locations inside step
    assert "impact" in step0
    assert "schools" in step0["impact"]
    assert "hospitals" in step0["impact"]
    assert "population_k" in step0["impact"]
    assert "top_risk_locations" in step0
    assert len(step0["top_risk_locations"]) <= 5

    # Test Lifecycle and Confidence in root
    assert "lifecycle" in data
    assert "current_stage" in data["lifecycle"]
    assert "confidence" in data
    assert "overall" in data["confidence"]
    assert "active_alerts" in data

def test_cached_forecast_endpoint():
    response = client.get("/api/forecast")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["forecast_steps"]) == 6

def test_impact_endpoint():
    response = client.get("/api/impact?step_index=0")
    assert response.status_code == 200
    data = response.json()
    assert "schools" in data
    assert "hospitals" in data
    assert "population_k" in data

def test_risk_locations_endpoint():
    response = client.get("/api/risk-locations?step_index=0")
    assert response.status_code == 200
    data = response.json()
    assert "top_risk_locations" in data
    assert isinstance(data["top_risk_locations"], list)

def test_storm_lifecycle_endpoint():
    response = client.get("/api/storm-lifecycle")
    assert response.status_code == 200
    data = response.json()
    assert "current_stage" in data
    assert "stages" in data

def test_alerts_endpoint():
    response = client.get("/api/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)

def test_confidence_endpoint():
    response = client.get("/api/confidence")
    assert response.status_code == 200
    data = response.json()
    assert "overall" in data
    assert "radar_agreement" in data

def test_reload_model_endpoint():
    response = client.post("/api/reload-model")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"

def test_data_sources_endpoint():
    response = client.get("/api/data-sources")
    assert response.status_code == 200
    data = response.json()
    assert data["active_mode"] == "DEMO_SYNTHETIC"
    assert "MOSDAC" in data["live_government_data_status"]
    assert "approval" in data["live_government_data_status"]["MOSDAC"].lower()

def test_stats_endpoint():
    response = client.get("/api/v1/telemetry/stats")
    assert response.status_code == 200
    data = response.json()
    assert "active_streams" in data
    assert "avg_latency_ms" in data
    assert data["system_health"] == "OPTIMAL"

def test_telemetry_ingest():
    payload = {
        "node_id": "TEST_NODE_01",
        "metric_value": 85.5,
        "attributes": {"sensor_model": "GEN_IV", "calibration": "OK"}
    }
    response = client.post("/api/v1/telemetry/ingest", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["ps_id"] == "SIH26084"
    assert data["metric_value"] == 85.5
    assert 0.0 <= data["risk_score"] <= 1.0
    assert isinstance(data["is_anomaly"], bool)
    assert "sha256_hash" in data

def test_audit_logs():
    response = client.get("/api/v1/audit/logs")
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data
    assert isinstance(data["records"], list)

def test_dispatch_action():
    payload = {
        "event_id": "EVT-999888",
        "protocol_type": "URGENT_OPERATIONAL_ESCALATION",
        "notes": "Automated pipeline trigger"
    }
    response = client.post("/api/v1/action/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["event_id"] == "EVT-999888"
    assert data["status"] == "DISPATCHED_TO_FIELD_TEAMS"

def test_nowcast_cells_endpoint():
    response = client.get("/api/v1/nowcast/cells")
    assert response.status_code == 200
    data = response.json()
    assert "active_cells" in data
    assert len(data["active_cells"]) >= 1
    cell = data["active_cells"][0]
    assert "cell_id" in cell
    assert "max_reflectivity_dbz" in cell
    assert "cloudburst_risk" in cell

def test_nowcast_timeline_endpoint():
    response = client.get("/api/v1/nowcast/timeline")
    assert response.status_code == 200
    data = response.json()
    assert "horizons" in data
    assert len(data["horizons"]) == 8
    assert data["horizons"][0]["horizon"] == "T+00m"
    assert data["horizons"][-1]["horizon"] == "T+06h"


# ==============================================================================
# Part 4 Real Meteorological Pipeline Tests
# ==============================================================================
def test_providers_status_endpoint():
    response = client.get("/api/providers/status")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert "overall_mode" in data
    assert "confidence_indicator" in data
    providers = data["providers"]
    for src in ["satellite", "radar", "lightning", "nwp", "aws"]:
        assert src in providers
        assert providers[src]["status"] in ["GOOD", "DEGRADED", "UNAVAILABLE"]
        assert providers[src]["mode"] in ["LIVE", "DEMO"]
        assert "last_received" in providers[src]
        assert "update_interval_min" in providers[src]


def test_ai_chatbot_endpoint():
    payload = {"message": "What is the current storm track and expected impact on schools?"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "category" in data
    assert "disclaimer" in data
    assert len(data["response"]) > 20


def test_reports_bulletin_endpoint():
    response = client.get("/api/reports/generate")
    assert response.status_code == 200
    data = response.json()
    assert "bulletin_id" in data
    assert "markdown_bulletin" in data
    assert "timeline" in data
    assert "impact_summary" in data
    assert "confidence" in data
    assert len(data["timeline"]) == 6


def test_pipeline_structured_logs():
    response = client.get("/api/logs?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert len(data["logs"]) >= 1
    sample = data["logs"][-1]
    assert "formatted" in sample
    assert sample["formatted"].startswith("[")


def test_forecast_cycle_run():
    response = client.post("/api/forecast-cycle/run")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert "cycle_id" in data
    assert "peak_hazard_level" in data


# ==============================================================================
# PART 7: SCIENTIFIC VALIDATION & EXPLAINABILITY ENDPOINT TESTS
# ==============================================================================

def test_evaluation_summary():
    res = client.get("/api/v1/evaluation/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "MultiSourceNowcastNet"
    assert "scientific_honesty_disclaimer" in data
    assert "headline_metrics" in data
    assert data["headline_metrics"]["csi_lead_30m"] > 0.0


def test_evaluation_lead_time():
    res = client.get("/api/v1/evaluation/lead-time")
    assert res.status_code == 200
    data = res.json()
    assert "lead_time_table" in data
    assert len(data["lead_time_table"]) >= 4
    first_step = data["lead_time_table"][0]
    assert "lead_time" in first_step
    assert "csi" in first_step
    assert "rmse_dbz" in first_step


def test_evaluation_baselines():
    res = client.get("/api/v1/evaluation/baselines")
    assert res.status_code == 200
    data = res.json()
    assert "baselines" in data
    assert len(data["baselines"]) >= 4
    names = [b["baseline_id"] for b in data["baselines"]]
    assert any("persistence" in n for n in names)


def test_evaluation_ablation():
    res = client.get("/api/v1/evaluation/ablation")
    assert res.status_code == 200
    data = res.json()
    assert "ablation_table" in data
    assert len(data["ablation_table"]) >= 3


def test_evaluation_storm_tracking():
    res = client.get("/api/v1/evaluation/storm-tracking")
    assert res.status_code == 200
    data = res.json()
    assert "overall_metrics" in data
    assert "mean_displacement_km" in data["overall_metrics"]


def test_evaluation_calibration():
    res = client.get("/api/v1/evaluation/calibration")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "bins" in data


def test_model_card_endpoint():
    res = client.get("/api/v1/model-card")
    assert res.status_code == 200
    data = res.json()
    assert "markdown" in data
    assert "MultiSourceNowcastNet" in data["markdown"]


def test_data_card_endpoint():
    res = client.get("/api/v1/data-card")
    assert res.status_code == 200
    data = res.json()
    assert "markdown" in data
    assert "Doppler Weather Radar" in data["markdown"]


def test_why_forecast_explanation():
    res = client.get("/api/v1/explainability/why-forecast")
    assert res.status_code == 200
    data = res.json()
    assert data["title"] == "Why This Forecast?"
    assert "evidence_list" in data
    assert len(data["evidence_list"]) >= 3


