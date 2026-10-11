"""
Radar Spatial Feature Encoder (SIH26084 Part 5).
Adheres strictly to Section 14 of Part 5 Specification:
Input:
  Reflectivity (dBZ) + Radial Velocity (m/s) [B, 2, H, W]
Architecture:
  Conv2D -> GroupNorm -> ReLU -> Downsample blocks
Output:
  Radar feature map representation.
Optimized for high compute efficiency on standard CPU / limited GPU memory.
"""

import torch
import torch.nn as nn


class RadarEncoder(nn.Module):
    """
    Extracts deep kinematic and reflectivity spatial signatures from radar volumes.
    """

    def __init__(self, in_channels: int = 2, feature_dim: int = 16):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(2, feature_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_dim),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels=2, H, W] (Reflectivity, Velocity)
        Returns: [B, feature_dim, H, W]
        """
        f1 = self.conv_block1(x)
        f2 = self.conv_block2(f1)
        out = self.out_conv(f2)
        return out
