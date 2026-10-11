"""
Multi-Source Feature Fusion Module (SIH26084 Part 5).
Adheres strictly to Section 18 of Part 5 Specification:
Fuses:
  Radar features + Satellite features + Lightning features + NWP features
Method:
  Concatenation + Squeeze-and-Excitation / Channel Attention + 1x1 Convolution
"""

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Combines latent spatial features from Radar, Satellite, Lightning, and NWP encoders.
    Applies channel recalibration weights to prioritize active observational feeds.
    """

    def __init__(self, feature_dim: int = 16, fused_dim: int = 16):
        super().__init__()
        # Total incoming concatenated channels: 4 encoders * feature_dim = 64
        in_channels = feature_dim * 4

        # Channel Attention recalibration
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Dimension reduction and non-linear mixing
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, fused_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, fused_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_dim * 2, fused_dim, kernel_size=1),
            nn.GroupNorm(4, fused_dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        radar_feat: torch.Tensor,
        sat_feat: torch.Tensor,
        ltg_feat: torch.Tensor,
        nwp_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs: each [B, feature_dim, H, W]
        Returns: [B, fused_dim, H, W]
        """
        cat = torch.cat([radar_feat, sat_feat, ltg_feat, nwp_feat], dim=1)
        # Apply channel attention weights
        gate = self.channel_gate(cat)
        attended = cat * gate
        fused = self.fusion_conv(attended)
        return fused
