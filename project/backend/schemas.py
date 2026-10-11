"""
SIH26084 Convective Scale Nowcasting - Pydantic Schemas
Strict typing for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ModelStatusResponse(BaseModel):
    model: str = Field("ConvLSTM", description="Neural network architecture family")
    status: str = Field("loaded", description="Operational memory status")
    device: str = Field("cpu", description="Compute runtime device")
    input_channels: int = Field(4, description="Count of physical input channels")
    forecast_steps: int = Field(6, description="Output prediction horizons")
    forecast_interval_minutes: int = Field(30, description="Lead interval per step")
    data_mode: str = Field("synthetic_demo", description="Active ingestion stream mode")
    model_file: Optional[str] = None
    dataset_file: Optional[str] = None
    last_inference_timestamp: Optional[str] = None


class CentroidCoord(BaseModel):
    grid_row: float
    grid_col: float
    norm_x: float
    norm_y: float
    max_intensity: float
    core_active: bool


class StormTrackPoint(BaseModel):
    lead_minutes: int
    label: str
    grid_row: float
    grid_col: float
    norm_x: float
    norm_y: float
    estimated_lat: float
    estimated_lon: float
    core_intensity: float
    core_active: bool
    marker_label: str


class ForecastStep(BaseModel):
    lead_minutes: int
    label: str
    rain_intensity: float
    rain_intensity_mean: float
    max_reflectivity_dbz: float
    mean_reflectivity_dbz: float
    lightning_density: float
    lightning_density_mean: float
    estimated_strikes_per_min: int
    lightning_trend: str
    hail_probability: float
    hail_probability_pct: float
    hail_probability_mean: float
    hazard_score: float
    hazard_level: str
    narrative: str
    centroid: Dict[str, Any]
    impact: Optional[Dict[str, Any]] = None
    top_risk_locations: Optional[List[Dict[str, Any]]] = None
    live_weather: Optional[Dict[str, Any]] = None
    hail_stats: Optional[Dict[str, Any]] = None
    radar_grid: List[List[float]]
    lightning_grid: List[List[float]]
    hail_grid: List[List[float]]


class ForecastRequest(BaseModel):
    sequence_index: Optional[int] = Field(0, description="Dataset sample sequence index [0..79]")
    demo_mode: Optional[bool] = Field(True, description="Force synthetic demonstration pipeline")


class ForecastResponse(BaseModel):
    success: bool
    mode: str = "synthetic_demo"
    message: str
    timestamp: str
    data_source: str
    input_metadata: Dict[str, Any]
    model_info: Dict[str, Any]
    forecast_steps: List[ForecastStep]
    storm_track: List[StormTrackPoint]
    lifecycle: Optional[Dict[str, Any]] = None
    confidence: Optional[Dict[str, Any]] = None
    active_alerts: Optional[List[Dict[str, Any]]] = None
    summary: Dict[str, Any]
    extension_note: str = "0-3h active prototype. 3-6h extension requires trained long-horizon model."


class ReloadModelResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    model_device: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="Operator question for meteorological assistant")
    context_override: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    query: str
    category: str
    response: str
    timestamp: str
    disclaimer: str


class ProviderStatusItem(BaseModel):
    source: str
    status: str
    mode: str
    last_received: str
    data_age_min: Optional[float]
    update_interval_min: int
    fallback_retained: bool
    warning: Optional[str] = None


class ProvidersStatusResponse(BaseModel):
    overall_mode: str
    operational_status: str
    grid_resolution_km: float
    confidence_indicator: Dict[str, Any]
    providers: Dict[str, ProviderStatusItem]
