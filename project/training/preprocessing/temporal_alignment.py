"""
Temporal Alignment and Sequence Synchronizer (SIH26084 Part 5).
Adheres strictly to Section 5 of Part 5 Specification:
- Configurable INPUT_INTERVAL_MINUTES (15 min)
- Configurable FORECAST_INTERVAL_MINUTES (30 min)
- Configurable INPUT_HISTORY_HOURS (1 hr -> 4 frames: T-45, T-30, T-15, T0)
- Configurable FORECAST_HOURS (6 hr -> 12 horizons: +30m to +360m)
- Temporal gap validation & sequence sanity checks
"""

import datetime
from typing import Dict, Any, List, Optional, Tuple


class TemporalAligner:
    """
    Handles temporal indexing, alignment, and sequence timestamp validation
    across disparate radar, satellite, lightning, and NWP cadences.
    """

    def __init__(
        self,
        input_interval_minutes: int = 15,
        forecast_interval_minutes: int = 30,
        input_frames: int = 4,
        forecast_steps: int = 12,
        input_interval_min: Optional[int] = None,
        forecast_interval_min: Optional[int] = None,
        input_history_hours: Optional[int] = None,
        forecast_hours: Optional[int] = None
    ):
        self.input_interval = input_interval_min if input_interval_min is not None else input_interval_minutes
        self.forecast_interval = forecast_interval_min if forecast_interval_min is not None else forecast_interval_minutes
        
        if input_history_hours is not None:
            self.input_frames = int((input_history_hours * 60) / self.input_interval)
        else:
            self.input_frames = input_frames

        if forecast_hours is not None:
            self.forecast_steps = int((forecast_hours * 60) / self.forecast_interval)
        else:
            self.forecast_steps = forecast_steps

    def get_sequence_lengths(self) -> Tuple[int, int]:
        return self.input_frames, self.forecast_steps

    def generate_input_timestamps(self, ref_time: datetime.datetime) -> List[datetime.datetime]:
        """
        Produces history frames [T-45m, T-30m, T-15m, T0].
        """
        timestamps = []
        for i in range(self.input_frames - 1, -1, -1):
            t = ref_time - datetime.timedelta(minutes=i * self.input_interval)
            timestamps.append(t)
        return timestamps

    def generate_forecast_timestamps(self, ref_time: datetime.datetime) -> List[datetime.datetime]:
        """
        Produces target future frames [T+30m, T+60m, ..., T+360m].
        """
        timestamps = []
        for i in range(1, self.forecast_steps + 1):
            t = ref_time + datetime.timedelta(minutes=i * self.forecast_interval)
            timestamps.append(t)
        return timestamps

    def check_temporal_continuity(
        self,
        frame_timestamps: List[datetime.datetime],
        expected_interval_minutes: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies that observations arrive within tolerance without large missing temporal gaps.
        """
        if len(frame_timestamps) < 2:
            return True, None

        max_allowed_gap = expected_interval_minutes * 1.5
        for i in range(1, len(frame_timestamps)):
            delta_min = (frame_timestamps[i] - frame_timestamps[i - 1]).total_seconds() / 60.0
            if delta_min > max_allowed_gap:
                return False, f"Temporal gap of {delta_min:.1f} min exceeds max allowed {max_allowed_gap} min"
        return True, None
