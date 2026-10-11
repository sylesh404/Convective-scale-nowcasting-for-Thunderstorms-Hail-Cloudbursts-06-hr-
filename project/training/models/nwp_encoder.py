"""
NWP Atmospheric Environment Feature Encoder (SIH26084 Part 5).
Adheres strictly to Section 17 of Part 5 Specification:
Input:
  CAPE, CIN, RH, Temperature, Wind U, Wind V, Precip [B, 6, H, W]
Architecture:
  Spatial 1x1 Conv + Deep Atmospheric Projection
Output:
  Atmospheric instability and steering environment features.
"""

import torch
import torch.nn as nn


class NWPEncoder(nn.Module):
    """
    Projects gridded NWP thermodynamic and kinematic boundary conditions into the fusion latent space.
    """

    def __init__(self, in_channels: int = 6, feature_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=1),
            nn.GroupNorm(4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels=6, H, W]
        Returns: [B, feature_dim, H, W]
        """
        return self.net(x)
