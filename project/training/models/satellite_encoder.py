"""
Satellite Spatial Feature Encoder (SIH26084 Part 5).
Adheres strictly to Section 15 of Part 5 Specification:
Input:
  IR + Water Vapour + Cloud products [B, 2, H, W]
Architecture:
  Lightweight multi-scale CNN
Output:
  Satellite latent feature representation.
"""

import torch
import torch.nn as nn


class SatelliteEncoder(nn.Module):
    """
    Encodes geostationary cloud top brightness temperatures and water vapour distributions.
    """

    def __init__(self, in_channels: int = 2, feature_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(2, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels=2, H, W] (IR, Water Vapour)
        Returns: [B, feature_dim, H, W]
        """
        return self.encoder(x)
