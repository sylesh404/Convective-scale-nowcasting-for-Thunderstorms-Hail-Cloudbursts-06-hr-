"""
Lightning Spatiotemporal Feature Encoder (SIH26084 Part 5).
Adheres strictly to Section 16 of Part 5 Specification:
Input:
  Lightning density map + trend map [B, 2, H, W]
Architecture:
  CNN + spatial feature projection
Output:
  Lightning vigor and convective electrification feature map.
"""

import torch
import torch.nn as nn


class LightningEncoder(nn.Module):
    """
    Encodes lightning strike flash densities and temporal intensification jumps.
    """

    def __init__(self, in_channels: int = 2, feature_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(2, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_dim),
            nn.ReLU(inplace=True)
        )
        self.proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels=2, H, W] (Density, Trend)
        Returns: [B, feature_dim, H, W]
        """
        return self.proj(self.conv(x))
