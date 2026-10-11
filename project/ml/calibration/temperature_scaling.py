"""
Confidence Calibration & Reliability Assessment (SIH26084 Part 7).
Adheres strictly to Section 16 of Part 7 Specification:
- Temperature Scaling for probability recalibration
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Reliability Diagram binning (Predicted Confidence vs Observed Event Frequency)
- Clear indicator: "Prototype confidence — not calibrated" if sample size insufficient
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ModelCalibrator:
    """
    Evaluates probability calibration and computes empirical reliability diagrams.
    Applies learned Temperature Scaling parameter T > 0 on hazard logits: p_cal = sigmoid(z / T).
    """

    def __init__(self, temperature: float = 1.0, min_samples_for_calibration: int = 500):
        self.temperature = float(temperature)
        self.min_samples = min_samples_for_calibration
        self.is_calibrated = False

    def fit_temperature(
        self,
        uncalibrated_probs: np.ndarray,
        binary_ground_truth: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Optimizes scalar temperature T to minimize negative log-likelihood (BCE) on validation set.
        """
        eps = 1e-7
        probs = np.clip(uncalibrated_probs.flatten(), eps, 1.0 - eps)
        targets = binary_ground_truth.flatten().astype(np.float32)

        if len(probs) < self.min_samples:
            # Mark as uncalibrated prototype if data is insufficient
            self.is_calibrated = False
            return self.temperature

        # Invert sigmoid to retrieve logits
        logits = np.log(probs / (1.0 - probs))

        t_logits = torch.tensor(logits, dtype=torch.float32)
        t_targets = torch.tensor(targets, dtype=torch.float32)

        T_param = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = optim.LBFGS([T_param], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = t_logits / torch.clamp(T_param, 0.1, 10.0)
            loss = criterion(scaled_logits, t_targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.clamp(T_param, 0.1, 10.0).item())
        self.is_calibrated = True
        return self.temperature

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Applies temperature scaling to raw probability array.
        """
        if not self.is_calibrated or abs(self.temperature - 1.0) < 1e-4:
            return probs
        eps = 1e-7
        p = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p))
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))

    def compute_brier_score(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """
        Mean Squared Error in probability space: Brier = mean((p - y)^2).
        """
        return float(np.mean((probs.flatten() - targets.flatten()) ** 2))

    def evaluate_reliability_bins(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Computes Expected Calibration Error (ECE) and binning table for reliability diagrams.
        """
        p = probs.flatten()
        y = targets.flatten().astype(float)
        N = len(p)

        if N == 0:
            return {
                "ece": 0.0,
                "mce": 0.0,
                "brier_score": 0.0,
                "bins": [],
                "status": "Prototype confidence — not calibrated."
            }

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        bins_data = []
        ece = 0.0
        mce = 0.0

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            mask = (p >= bin_lower) & (p < bin_upper) if i < num_bins - 1 else (p >= bin_lower) & (p <= bin_upper)
            bin_count = int(np.sum(mask))

            if bin_count > 0:
                bin_conf = float(np.mean(p[mask]))
                bin_acc = float(np.mean(y[mask]))
                gap = abs(bin_acc - bin_conf)
                ece += (bin_count / N) * gap
                mce = max(mce, gap)
            else:
                bin_conf = float((bin_lower + bin_upper) / 2.0)
                bin_acc = 0.0
                gap = 0.0

            bins_data.append({
                "bin_index": i + 1,
                "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": bin_count,
                "predicted_confidence": round(bin_conf, 4),
                "observed_event_frequency": round(bin_acc, 4),
                "calibration_gap": round(gap, 4)
            })

        brier = self.compute_brier_score(p, y)

        return {
            "status": "Calibrated" if self.is_calibrated else "Prototype confidence — not calibrated.",
            "temperature": round(self.temperature, 4),
            "expected_calibration_error": round(float(ece), 4),
            "maximum_calibration_error": round(float(mce), 4),
            "brier_score": round(brier, 4),
            "sample_count": N,
            "bins": bins_data
        }
