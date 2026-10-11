"""
SIH26084 Convective Scale Nowcasting - Abstract Data Provider Interface
Defines the contract for multi-sensor meteorological data feeding the ConvLSTM model.
Enables plug-and-play switching between Synthetic demonstration data and real MOSDAC/IMD streams.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class DataProvider(ABC):
    """
    Abstract base class for all meteorological data providers.
    Provides standard 4-channel input tensors:
      Channel 0: Radar reflectivity proxy (dBZ / normalized)
      Channel 1: INSAT thermal/cloud proxy (TIR1 / brightness temp)
      Channel 2: Lightning density proxy (flash count / normalized)
      Channel 3: Humidity proxy (Relative humidity / Vapour proxy)
    Grid: 16x16, Historical sequence length: 4 frames (30m intervals)
    """

    @abstractmethod
    def get_latest_sequence(self, sequence_idx: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Retrieves the latest 4-frame input sequence.
        Returns:
            sequence_tensor: numpy array of shape (4, 4, 16, 16) with values in [0, 1]
            metadata: dictionary describing the source, timestamp, and channel metadata
        """
        pass

    @abstractmethod
    def get_provider_status(self) -> Dict[str, Any]:
        """
        Returns connection and operational telemetry status for this provider.
        """
        pass

    @abstractmethod
    def is_live(self) -> bool:
        """
        Returns True if provider supplies actual live government/instrument feeds,
        False if synthetic / simulation.
        """
        pass
