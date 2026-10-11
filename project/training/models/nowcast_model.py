"""
Complete Convective Nowcasting Model Suite (SIH26084 Part 5).
Adheres strictly to Sections 12, 13, 19, and 20 of Part 5 Specification:
- Baseline Models:
  1. PersistenceBaseline (latest observation repeated across future horizons)
  2. ConvLSTMBaseline (single-stream 4-channel ConvLSTM)
- Production Multi-Source Fusion Model (MultiSourceNowcastNet):
  Radar Encoder + Satellite Encoder + Lightning Encoder + NWP Encoder
          ↓
    Feature Fusion
          ↓
  ConvLSTM Spatiotemporal Backbone
          ↓
  Multi-Head Output:
    Head 1: Rainfall / Radar Reflectivity Intensity
    Head 2: Lightning Density
    Head 3: Hail Probability
    Head 4: Convective Intensity
    Head 5: Downburst ("Downburst model unavailable" until labeled data exists)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from .radar_encoder import RadarEncoder
from .satellite_encoder import SatelliteEncoder
from .lightning_encoder import LightningEncoder
from .nwp_encoder import NWPEncoder
from .fusion_model import FeatureFusion


# ==============================================================================
# 1. Baseline Model Implementations (Section 12)
# ==============================================================================
class PersistenceBaseline(nn.Module):
    """
    Persistence Baseline: Predicts that the future atmosphere remains identical
    to the latest observed observation frame. Essential benchmark for nowcasting.
    """

    def __init__(self, future_steps: int = 12, forecast_frames: Optional[int] = None):
        super().__init__()
        self.future_steps = forecast_frames if forecast_frames is not None else future_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T_in, C_in, H, W]
        Returns: [B, future_steps, C_out=4, H, W]
        """
        latest = x[:, -1]  # [B, C_in, H, W]
        B, C_in, H, W = latest.shape
        out = torch.zeros(B, self.future_steps, 4, H, W, device=x.device, dtype=x.dtype)

        # Ch 0: Radar
        out[:, :, 0] = latest[:, 0:1]
        # Ch 1: Lightning
        ltg_idx = 4 if C_in > 4 else (2 if C_in > 2 else 0)
        out[:, :, 1] = latest[:, ltg_idx:ltg_idx + 1]
        # Ch 2: Hail proxy
        out[:, :, 2] = torch.clamp(latest[:, 0:1] * 1.2 - 0.2, 0.0, 1.0)
        # Ch 3: Convective intensity
        out[:, :, 3] = (out[:, :, 0] * 0.6 + out[:, :, 1] * 0.4)
        return out


class ConvLSTMCell(nn.Module):
    """
    Standard Spatiotemporal ConvLSTM Cell preserving 2D spatial topology.
    """

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMBaseline(nn.Module):
    """
    4-Channel ConvLSTM Baseline model for comparative evaluation (Section 12).
    """

    def __init__(self, in_channels: int = 4, hidden_dim: int = 16, future_steps: int = 12, forecast_frames: Optional[int] = None):
        super().__init__()
        self.future = forecast_frames if forecast_frames is not None else future_steps
        self.hidden_dim = hidden_dim
        self.in_channels = 4
        # Channel adapter if input is not 4 channels
        self.input_adapter = nn.Conv2d(in_channels, 4, kernel_size=1) if in_channels != 4 else nn.Identity()
        self.cell = ConvLSTMCell(4, hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T_in, C, H, W]
        Returns: [B, future_steps, C_out=4, H, W]
        """
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)

        for t in range(T):
            x_t = self.input_adapter(x[:, t])
            h, c = self.cell(x_t, h, c)

        outputs = []
        feedback = x_t
        for _ in range(self.future):
            y = self.head(h)
            outputs.append(y)
            # Autoregressive feedback loop with 4 channels
            h, c = self.cell(y, h, c)

        return torch.stack(outputs, dim=1)


# ==============================================================================
# 2. Production Multi-Source Fusion Model (Section 13, 18, 19, 20)
# ==============================================================================
class MultiSourceNowcastNet(nn.Module):
    """
    State-of-the-Art Multi-Source Fusion Spatiotemporal Convective Nowcasting Network.
    Fuses Radar, Satellite, Lightning, and NWP into unified recurrent representations.
    """

    def __init__(
        self,
        in_channels: int = 12,
        feature_dim: int = 16,
        hidden_dim: int = 24,
        future_steps: int = 12,
        forecast_frames: Optional[int] = None,
        model_version: str = "ConvLSTM-Fusion-v1.0"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_version = model_version
        self.future_steps = forecast_frames if forecast_frames is not None else future_steps
        self.hidden_dim = hidden_dim

        # 1. Multi-Sensor Spatial Encoders (Sections 14, 15, 16, 17)
        self.radar_encoder = RadarEncoder(in_channels=2, feature_dim=feature_dim)
        self.satellite_encoder = SatelliteEncoder(in_channels=2, feature_dim=feature_dim)
        self.lightning_encoder = LightningEncoder(in_channels=2, feature_dim=feature_dim)
        self.nwp_encoder = NWPEncoder(in_channels=6, feature_dim=feature_dim)

        # 2. Feature Fusion Layer (Section 18)
        self.fusion = FeatureFusion(feature_dim=feature_dim, fused_dim=hidden_dim)

        # 3. ConvLSTM Spatiotemporal Recurrent Backbone (Section 19)
        self.temporal_cell = ConvLSTMCell(in_channels=hidden_dim, hidden_dim=hidden_dim)

        # 4. Multi-Hazard Prediction Heads (Section 20)
        # Head 1: Rainfall / Radar Intensity [0, 1]
        self.head_rain = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Head 2: Lightning Density [0, 1]
        self.head_lightning = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Head 3: Hail Probability [0, 1]
        self.head_hail = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Head 4: Convective Intensity [0, 1]
        self.head_convective = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Feedback projector back to fused latent space
        self.feedback_proj = nn.Conv2d(4, hidden_dim, kernel_size=1)

    def extract_sensor_slices(
        self,
        x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits 12-channel input at timestep t into dedicated sensor components:
        Ch 0,1: Radar (Ref, Vel)
        Ch 2,3: Satellite (IR, WV)
        Ch 4,5: Lightning (Density, Trend / Rainfall proxy)
        Ch 6-11: NWP (RH, Temp, Precip, CAPE, Wind U, Wind V)
        """
        B, C, H, W = x_t.shape
        if C >= 12:
            radar_in = x_t[:, 0:2]
            sat_in = x_t[:, 2:4]
            ltg_in = x_t[:, 4:6]
            nwp_in = x_t[:, 6:12]
        else:
            # Fallback for 4-channel input
            radar_in = torch.cat([x_t[:, 0:1], x_t[:, 0:1] * 0.1], dim=1)
            sat_in = torch.cat([x_t[:, 1:2], x_t[:, 1:2] * 0.9], dim=1)
            ltg_in = torch.cat([x_t[:, 2:3], x_t[:, 2:3] * 0.5], dim=1)
            nwp_in = torch.cat([x_t[:, 3:4]] * 6, dim=1)

        return radar_in, sat_in, ltg_in, nwp_in

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [B, T_in=4, C=12, H, W]
        mask / masks: Optional availability mask [B, T_in, C=12, H, W]
        Returns: [B, future_steps=12, C_out=4, H, W]
        """
        B, T_in, C, H, W = x.shape
        active_mask = mask if mask is not None else masks

        # Zero out unavailable channels if mask is provided
        if active_mask is not None:
            x = x * active_mask

        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)

        # 1. Historical Encoding & Temporal Updates (T_in frames)
        for t in range(T_in):
            x_t = x[:, t]
            r_in, s_in, l_in, n_in = self.extract_sensor_slices(x_t)

            r_feat = self.radar_encoder(r_in)
            s_feat = self.satellite_encoder(s_in)
            l_feat = self.lightning_encoder(l_in)
            n_feat = self.nwp_encoder(n_in)

            # Fuse multi-sensor features
            fused_t = self.fusion(r_feat, s_feat, l_feat, n_feat)
            h, c = self.temporal_cell(fused_t, h, c)

        # 2. Multi-Head Horizons Forecasting (future_steps)
        predictions = []
        feedback_feat = h

        for step in range(self.future_steps):
            # Evaluate multi-hazard prediction heads
            pred_rain = self.head_rain(feedback_feat)
            pred_ltg = self.head_lightning(feedback_feat)
            pred_hail = self.head_hail(feedback_feat)
            pred_conv = self.head_convective(feedback_feat)

            # Combined multi-hazard step: [B, 4, H, W]
            step_output = torch.cat([pred_rain, pred_ltg, pred_hail, pred_conv], dim=1)
            predictions.append(step_output)

            # Autoregressive feedback projection into temporal cell
            step_proj = self.feedback_proj(step_output)
            feedback_feat, c = self.temporal_cell(step_proj, feedback_feat, c)

        return torch.stack(predictions, dim=1)
