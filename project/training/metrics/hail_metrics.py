"""
Hail Probability Verification and Calibration Metrics (SIH26084 Part 5).
Adheres strictly to Sections 27 and 33 of Part 5 Specification:
- Precision, Recall, and F1 Score for hail threshold
- Brier Score (mean squared error of probabilistic forecast)
- Probability Calibration evaluation (Reliability evaluation)
- If insufficient validation data exists: reports "Calibration unavailable" (Section 33)
"""

import numpy as np
from typing import Dict, Any


class HailMetrics:
    """
    Evaluates probabilistic hail risk predictions against ground/radar hail signatures.
    """

    @staticmethod
    def brier_score(pred_prob: np.ndarray, target_binary: np.ndarray) -> float:
        return float(np.mean((pred_prob - target_binary.astype(float)) ** 2))

    def __init__(self, probability_threshold: float = 0.50):
        self.prob_thresh = probability_threshold

    def evaluate(self, pred_prob: np.ndarray, target_binary: np.ndarray) -> Dict[str, Any]:
        """
        pred_prob: continuous hail probability [0, 1]
        target_binary: binary truth [0, 1]
        """
        pred_bin = pred_prob >= self.prob_thresh
        target_bin = target_binary >= 0.50

        tp = int(np.sum(pred_bin & target_bin))
        fp = int(np.sum(pred_bin & (~target_bin)))
        fn = int(np.sum((~pred_bin) & target_bin))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Brier Score: BS = (1/N) * sum((p_i - o_i)^2)
        brier_score = float(np.mean((pred_prob - target_binary.astype(float)) ** 2))

        # Calibration assessment (Section 33)
        total_positives = int(np.sum(target_bin))
        if total_positives < 10:
            calibration_status = "Calibration unavailable (insufficient severe hail samples in validation split)"
            reliability_bins = None
        else:
            # 5 probability bins [0-0.2, 0.2-0.4, ..., 0.8-1.0]
            bin_edges = np.linspace(0.0, 1.0, 6)
            reliability_bins = []
            for b in range(5):
                bin_mask = (pred_prob >= bin_edges[b]) & (pred_prob < bin_edges[b + 1])
                count = int(np.sum(bin_mask))
                if count > 0:
                    observed_freq = float(np.mean(target_bin[bin_mask]))
                    mean_pred_prob = float(np.mean(pred_prob[bin_mask]))
                    reliability_bins.append({
                        "bin_range": f"{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}",
                        "sample_count": count,
                        "mean_predicted_prob": round(mean_pred_prob, 3),
                        "observed_frequency": round(observed_freq, 3)
                    })
            calibration_status = "Calibrated on validation split"

        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "brier_score": round(brier_score, 4),
            "calibration_status": calibration_status,
            "reliability_bins": reliability_bins
        }
