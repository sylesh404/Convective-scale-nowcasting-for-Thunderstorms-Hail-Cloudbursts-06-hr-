"""
SIH26084 Meteorological Data Providers Package.
Exposes both modern Part 4 modular providers and backward-compatible references.
"""

from .base_provider import WeatherDataProvider
from .base import DataProvider
from .synthetic_provider import SyntheticDataProvider
from .mosdac_provider import MOSDACDataProvider
from .radar_provider import RadarProcessor
from .satellite_provider import SatelliteProcessor
from .lightning_provider import LightningProcessor
from .nwp_provider import NWPProcessor
from .aws_provider import AWSProcessor

__all__ = [
    "WeatherDataProvider",
    "DataProvider",
    "SyntheticDataProvider",
    "MOSDACDataProvider",
    "RadarProcessor",
    "SatelliteProcessor",
    "LightningProcessor",
    "NWPProcessor",
    "AWSProcessor",
]
