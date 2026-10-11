"""
Lightning Loss Function for Sparse Convective Strokes (SIH26084 Part 5).
Adheres strictly to Section 21 of Part 5 Specification:
- Focal-style loss formulation for sparse flash density occurrences
- Suppresses overwhelming background zeros while emphasizing lightning bursts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLightningLoss(nn.Module):
    """
    Focal loss adapted for continuous and binary-thresholded lightning flash densities.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, T, 1, H, W] in [0, 1]
        """
        eps = 1e-6
        pred_clamped = torch.clamp(pred, eps, 1.0 - eps)

        # Occurrence indicator (non-zero strikes)
        target_binary = (target > 0.05).float()

        # Binary cross entropy
        bce = - (target_binary * torch.log(pred_clamped) + (1.0 - target_binary) * torch.log(1.0 - pred_clamped))

        # Modulating factor (p_t error)
        p_t = target_binary * pred_clamped + (1.0 - target_binary) * (1.0 - pred_clamped)
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Alpha class balancing
        alpha_t = target_binary * self.alpha + (1.0 - target_binary) * (1.0 - self.alpha)

        focal_loss = alpha_t * modulating_factor * bce

        # Add Smooth L1 continuous density regression for active flash cells
        continuous_loss = F.smooth_l1_loss(pred, target, reduction='none')
        total_loss = focal_loss + continuous_loss

        return total_loss.mean()
