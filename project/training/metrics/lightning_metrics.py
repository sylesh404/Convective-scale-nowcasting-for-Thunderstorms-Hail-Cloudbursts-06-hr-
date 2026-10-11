"""
Lightning Verification Metrics (SIH26084 Part 5).
Adheres strictly to Section 27 of Part 5 Specification:
Computes:
  - MAE on continuous flash density
  - Precision, Recall, and F1 Score on lightning occurrence detection (active flash cells)
"""

import numpy as np
from typing import Dict, Any


class LightningMetrics:
    """
    Verification metrics for convective stroke density and lightning occurrence forecasts.
    """

    def __init__(self, occurrence_threshold: float = 0.05):
        self.occ_thresh = occurrence_threshold

    def evaluate(self, pred_norm: np.ndarray, target_norm: np.ndarray) -> Dict[str, float]:
        """
        pred_norm, target_norm: in [0, 1]
        """
        mae = float(np.mean(np.abs(pred_norm - target_norm)))

        pred_bin = pred_norm >= self.occ_thresh
        target_bin = target_norm >= self.occ_thresh

        tp = int(np.sum(pred_bin & target_bin))
        fp = int(np.sum(pred_bin & (~target_bin)))
        fn = int(np.sum((~pred_bin) & target_bin))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "mae": round(mae, 4),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)
        }
