"""
Multi-Hazard Combined Loss Engine (SIH26084 Part 5).
Adheres strictly to Sections 21 and 22 of Part 5 Specification:
- Combined Multi-Task Loss:
    L_total = w_rain * L_rain + w_ltg * L_ltg + w_hail * L_hail + w_conv * L_conv
- Configurable loss weights
- Event weighting emphasizing severe convective extremes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple, Optional
from .precipitation_loss import WeightedPrecipitationLoss
from .lightning_loss import FocalLightningLoss


class MultiHazardLoss(nn.Module):
    """
    Joint loss function optimizing all 4 convective prediction heads simultaneously.
    """

    def __init__(
        self,
        weight_rain: float = 1.0,
        weight_lightning: float = 0.8,
        weight_hail: float = 1.2,
        weight_convective: float = 0.6,
        precip_weight: Optional[float] = None,
        lightning_weight: Optional[float] = None,
        hail_weight: Optional[float] = None,
        convective_weight: Optional[float] = None,
        extreme_event_boost: float = 2.5
    ):
        super().__init__()
        self.w_rain = precip_weight if precip_weight is not None else weight_rain
        self.w_ltg = lightning_weight if lightning_weight is not None else weight_lightning
        self.w_hail = hail_weight if hail_weight is not None else weight_hail
        self.w_conv = convective_weight if convective_weight is not None else weight_convective
        self.extreme_boost = extreme_event_boost

        self.rain_loss_fn = WeightedPrecipitationLoss(extreme_threshold=0.50, extreme_weight=extreme_event_boost)
        self.ltg_loss_fn = FocalLightningLoss(gamma=2.0, alpha=0.75)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        predictions: [B, T, 4, H, W]
        targets: [B, T, 4, H, W]
        Returns: total_loss, loss_breakdown_dict
        """
        # Split prediction heads
        pred_rain = predictions[:, :, 0:1]
        pred_ltg = predictions[:, :, 1:2]
        pred_hail = predictions[:, :, 2:3]
        pred_conv = predictions[:, :, 3:4]

        # Split target heads
        target_rain = targets[:, :, 0:1]
        target_ltg = targets[:, :, 1:2]
        target_hail = targets[:, :, 2:3]
        target_conv = targets[:, :, 3:4]

        # 1. Rainfall / Radar loss
        loss_rain = self.rain_loss_fn(pred_rain, target_rain)

        # 2. Lightning flash density loss
        loss_ltg = self.ltg_loss_fn(pred_ltg, target_ltg)

        # 3. Hail risk BCE loss
        eps = 1e-6
        hail_clamped = torch.clamp(pred_hail, eps, 1.0 - eps)
        # Weight positive hail occurrences 4x due to severe class imbalance
        pos_weight = torch.tensor([4.0], device=predictions.device)
        loss_hail = F.binary_cross_entropy_with_logits(
            torch.logit(hail_clamped),
            target_hail,
            pos_weight=pos_weight
        )

        # 4. Convective intensity Huber loss
        loss_conv = F.smooth_l1_loss(pred_conv, target_conv)

        # Total Weighted Multi-Hazard Loss
        total_loss = (
            self.w_rain * loss_rain +
            self.w_ltg * loss_ltg +
            self.w_hail * loss_hail +
            self.w_conv * loss_conv
        )

        breakdown = {
            "loss_total": float(total_loss.item()),
            "loss_rain": float(loss_rain.item()),
            "loss_lightning": float(loss_ltg.item()),
            "loss_hail": float(loss_hail.item()),
            "loss_convective": float(loss_conv.item()),
            "rainfall": float(loss_rain.item()),
            "lightning": float(loss_ltg.item()),
            "hail": float(loss_hail.item()),
            "convective": float(loss_conv.item())
        }

        return total_loss, breakdown
