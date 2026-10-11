"""
Base Weather Data Provider Interface for SIH26084.
Enforces Section 2 of Part 4 Specification:
class WeatherDataProvider:
    get_latest_satellite()
    get_latest_radar()
    get_latest_lightning()
    get_latest_nwp()
    get_latest_aws()
    get_latest_rainfall()
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import datetime


class WeatherDataProvider(ABC):
    """
    Unified abstract interface for real meteorological and synthetic fallback data providers.
    All data returned is georeferenced, QC-validated, and prepared for spatial regridding.
    """

    def __init__(self, mode: str = "demo"):
        self.mode = mode  # "demo" or "live"
        self.last_update_times: Dict[str, Optional[datetime.datetime]] = {
            "satellite": None,
            "radar": None,
            "lightning": None,
            "nwp": None,
            "aws": None,
            "rainfall": None
        }

    @abstractmethod
    def get_latest_satellite(self) -> Dict[str, Any]:
        """
        Retrieves latest satellite observations (INSAT-3D/3DR/3DS).
        Variables: IR, Thermal IR, Water Vapour, Visible, Cloud products, CMV, QPE.
        """
        pass

    @abstractmethod
    def get_latest_radar(self) -> Dict[str, Any]:
        """
        Retrieves latest Doppler Weather Radar (DWR) volume.
        Variables: Reflectivity (dBZ), Radial velocity (m/s).
        """
        pass

    @abstractmethod
    def get_latest_lightning(self) -> Dict[str, Any]:
        """
        Retrieves 15-minute aggregated lightning strike network data.
        Variables: strike timestamps, lat/lon, peak current, polarity, spatial density.
        """
        pass

    @abstractmethod
    def get_latest_nwp(self) -> Dict[str, Any]:
        """
        Retrieves latest Numerical Weather Prediction model fields (WRF / NCMRWF Unified Model).
        Variables: Temperature, Relative Humidity, Pressure, U/V wind, CAPE, CIN, Precip, Shear.
        """
        pass

    @abstractmethod
    def get_latest_aws(self) -> Dict[str, Any]:
        """
        Retrieves Automatic Weather Station (AWS) surface observations.
        Variables: Temperature, Humidity, Pressure, Wind speed/direction, Rainfall rate.
        Used for ground validation, QC, and boundary context.
        """
        pass

    @abstractmethod
    def get_latest_rainfall(self) -> Dict[str, Any]:
        """
        Retrieves fused surface rainfall product / QPE estimate (mm/hr).
        """
        pass

    def get_provider_status(self) -> Dict[str, Any]:
        """
        Returns telemetry on provider connectivity, latency, and update intervals.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        source_status = {}
        for src, dt in self.last_update_times.items():
            age_min = round((now - dt).total_seconds() / 60.0, 1) if dt else None
            source_status[src] = {
                "last_received": dt.isoformat() if dt else "Never",
                "age_minutes": age_min,
                "status": "GOOD" if dt and age_min is not None and age_min < 60 else "DEGRADED" if dt else "UNAVAILABLE",
                "mode": "LIVE" if self.mode == "live" else "DEMO"
            }
        return {
            "provider_name": self.__class__.__name__,
            "mode": self.mode,
            "sources": source_status
        }
