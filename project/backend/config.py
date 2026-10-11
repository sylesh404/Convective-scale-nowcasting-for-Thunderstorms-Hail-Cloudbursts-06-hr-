"""
Configuration and Structured Logging for SIH26084 Convective Nowcasting Platform.
Supports config.yaml and environment variable overrides.
Enforces Part 4 configuration rules:
- DATA_MODE=demo/live
- GRID_RESOLUTION_KM=3
- FORECAST_HOURS=6
- FORECAST_INTERVAL_MINUTES=30
- Structured logging: [HH:MM:SS] Event description
"""

import os
import sys
import datetime
from typing import Dict, Any, List
from collections import deque

try:
    import yaml
except ImportError:
    yaml = None

# Base directories
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
CONFIG_YAML_PATH = os.path.join(PROJECT_DIR, "config.yaml")

# Load configuration file if present
_raw_config: Dict[str, Any] = {}
if yaml and os.path.exists(CONFIG_YAML_PATH):
    try:
        with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
            _raw_config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[CONFIG] Warning loading config.yaml: {e}", file=sys.stderr)

# Core Settings
DATA_MODE = os.environ.get("DATA_MODE", _raw_config.get("data_mode", "demo")).lower()
GRID_RESOLUTION_KM = float(os.environ.get("GRID_RESOLUTION_KM", _raw_config.get("grid_resolution_km", 3.0)))
FORECAST_HOURS = int(os.environ.get("FORECAST_HOURS", _raw_config.get("forecast_hours", 6)))
FORECAST_INTERVAL_MINUTES = int(os.environ.get("FORECAST_INTERVAL_MINUTES", _raw_config.get("forecast_interval_minutes", 30)))

# Provider Activation Flags
_providers = _raw_config.get("providers", {})
RADAR_ENABLED = os.environ.get("RADAR_ENABLED", str(_providers.get("radar_enabled", True))).lower() in ("true", "1", "yes")
SATELLITE_ENABLED = os.environ.get("SATELLITE_ENABLED", str(_providers.get("satellite_enabled", True))).lower() in ("true", "1", "yes")
LIGHTNING_ENABLED = os.environ.get("LIGHTNING_ENABLED", str(_providers.get("lightning_enabled", True))).lower() in ("true", "1", "yes")
NWP_ENABLED = os.environ.get("NWP_ENABLED", str(_providers.get("nwp_enabled", True))).lower() in ("true", "1", "yes")
AWS_ENABLED = os.environ.get("AWS_ENABLED", str(_providers.get("aws_enabled", True))).lower() in ("true", "1", "yes")

# MOSDAC Official Configuration
_mosdac_cfg = _raw_config.get("mosdac", {})
MOSDAC_USERNAME = os.environ.get("MOSDAC_USERNAME", "")
MOSDAC_PASSWORD = os.environ.get("MOSDAC_PASSWORD", "")
MOSDAC_API_KEY = os.environ.get("MOSDAC_API_KEY", "")
MOSDAC_BASE_URL = os.environ.get("MOSDAC_BASE_URL", _mosdac_cfg.get("base_url", "https://www.mosdac.gov.in"))
MOSDAC_API_ENDPOINT = os.environ.get("MOSDAC_API_ENDPOINT", _mosdac_cfg.get("api_endpoint", "https://api.mosdac.gov.in/v1"))
MOSDAC_SATELLITE_DATASET_ID = os.environ.get("MOSDAC_SATELLITE_DATASET_ID", _mosdac_cfg.get("satellite_dataset_id", "INSAT3D_L2B_HEM"))
MOSDAC_RADAR_DATASET_ID = os.environ.get("MOSDAC_RADAR_DATASET_ID", _mosdac_cfg.get("radar_dataset_id", "DWR_MAXZ"))
MOSDAC_LIGHTNING_DATASET_ID = os.environ.get("MOSDAC_LIGHTNING_DATASET_ID", _mosdac_cfg.get("lightning_dataset_id", "GLD_STRIKES"))

# Spatial Extent
_extent = _raw_config.get("spatial_extent", {})
SPATIAL_EXTENT = {
    "name": _extent.get("name", "Himalayan Foothills & Indo-Gangetic Plain"),
    "min_lat": float(_extent.get("min_lat", 28.0)),
    "max_lat": float(_extent.get("max_lat", 32.0)),
    "min_lon": float(_extent.get("min_lon", 76.0)),
    "max_lon": float(_extent.get("max_lon", 80.0)),
    "grid_height": int(_extent.get("grid_height", 16)),
    "grid_width": int(_extent.get("grid_width", 16)),
}


# ==============================================================================
# Structured Pipeline Logging (Master Prompt Section 29)
# Format: [HH:MM:SS] Message
# ==============================================================================
class StructuredPipelineLogger:
    """
    Maintains a thread-safe circular log buffer formatted according to
    Master Prompt Section 29: [HH:MM:SS] Event description.
    """
    def __init__(self, max_entries: int = 150):
        self.buffer: deque = deque(maxlen=max_entries)

    def log(self, message: str) -> str:
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{now_str}] {message}"
        self.buffer.append({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "time_str": now_str,
            "message": message,
            "formatted": entry
        })
        print(entry)
        return entry

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        entries = list(self.buffer)
        return entries[-limit:]


# Global logger instance
pipeline_logger = StructuredPipelineLogger()
log_event = pipeline_logger.log
