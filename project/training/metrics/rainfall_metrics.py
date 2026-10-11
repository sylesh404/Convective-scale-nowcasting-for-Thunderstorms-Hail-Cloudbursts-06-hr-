"""
Meteorological Rainfall and Reflectivity Verification Metrics (SIH26084 Part 5).
Adheres strictly to Section 27 of Part 5 Specification:
Computes:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
  - POD (Probability of Detection / Hit Rate) = Hits / (Hits + Misses)
  - FAR (False Alarm Ratio) = False Alarms / (Hits + False Alarms)
  - CSI (Critical Success Index / Threat Score) = Hits / (Hits + Misses + False Alarms)
  - HSS (Heidke Skill Score)
Thresholds:
  Moderate Rain (>= 20 dBZ / ~2.5 mm/hr)
  Heavy Rain (>= 35 dBZ / ~15 mm/hr)
  Extreme Convective Cloudburst (>= 45 dBZ / ~45 mm/hr)
"""

import numpy as np
from typing import Dict, Any, List


class RainfallMetrics:
    """
    Standard atmospheric verification suite for radar precipitation fields.
    """

    def __init__(self, thresholds_dbz: List[float] = [20.0, 35.0, 45.0]):
        self.thresholds_dbz = thresholds_dbz
        # Normalized threshold conversion (0-75 dBZ range)
        self.norm_thresholds = [t / 75.0 for t in thresholds_dbz]

    def compute_continuous_errors(
        self,
        pred_ref_dbz: np.ndarray,
        target_ref_dbz: np.ndarray
    ) -> Dict[str, float]:
        """
        MAE and RMSE in physical reflectivity (dBZ) units.
        """
        diff = pred_ref_dbz - target_ref_dbz
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        return {
            "mae_dbz": round(mae, 2),
            "rmse_dbz": round(rmse, 2)
        }

    def compute_contingency_scores(
        self,
        pred_norm: np.ndarray,
        target_norm: np.ndarray,
        thresh_norm: float
    ) -> Dict[str, float]:
        """
        Calculates POD, FAR, CSI, and HSS for a specific intensity threshold.
        """
        pred_bin = pred_norm >= thresh_norm
        target_bin = target_norm >= thresh_norm

        hits = int(np.sum(pred_bin & target_bin))
        misses = int(np.sum((~pred_bin) & target_bin))
        false_alarms = int(np.sum(pred_bin & (~target_bin)) )
        correct_negatives = int(np.sum((~pred_bin) & (~target_bin)))

        total = hits + misses + false_alarms + correct_negatives

        pod = float(hits / (hits + misses)) if (hits + misses) > 0 else 0.0
        far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
        csi = float(hits / (hits + misses + false_alarms)) if (hits + misses + false_alarms) > 0 else 0.0

        # Heidke Skill Score
        expected_hits = ((hits + misses) * (hits + false_alarms)) / total if total > 0 else 0
        denom = (total * (hits + misses) - expected_hits)
        hss = float((hits - expected_hits) / denom) if denom > 0 else 0.0

        return {
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "pod": round(pod, 3),
            "far": round(far, 3),
            "csi": round(csi, 3),
            "hss": round(hss, 3)
        }

    def evaluate_batch(
        self,
        pred_norm: np.ndarray,
        target_norm: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates full verification suite across all thresholds.
        """
        pred_dbz = pred_norm * 75.0
        target_dbz = target_norm * 75.0

        cont = self.compute_continuous_errors(pred_dbz, target_dbz)
        thresh_results = {}

        for t_dbz, t_norm in zip(self.thresholds_dbz, self.norm_thresholds):
            label = f"{int(t_dbz)}dBZ"
            thresh_results[label] = self.compute_contingency_scores(pred_norm, target_norm, t_norm)

        return {
            "continuous": cont,
            "thresholds": thresh_results
        }
