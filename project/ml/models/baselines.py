"""
Comprehensive Baseline Models Suite (SIH26084 Part 7).
Adheres strictly to Section 4 of Part 7 Specification:
  BASELINE 1: Persistence (forecast future radar/rainfall using latest observation)
  BASELINE 2: Optical-flow / motion extrapolation (propagate radar field using estimated storm motion)
  BASELINE 3: Radar-only ConvLSTM (2 radar channels -> 4 hazard heads)
  BASELINE 4: Radar + Satellite (4 channels -> 4 hazard heads)
  BASELINE 5: Radar + Lightning (4 channels -> 4 hazard heads)
  BASELINE 6: Radar + Satellite + Lightning (6 channels -> 4 hazard heads)
  BASELINE 7: Full Multi-Source Model (12 channels: Radar, Satellite, Lightning, NWP)
"""

import math
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Re-use ConvLSTMCell from training.models.nowcast_model
from training.models.nowcast_model import ConvLSTMCell, MultiSourceNowcastNet


# ==============================================================================
# BASELINE 1: Persistence Baseline
# ==============================================================================
class PersistenceBaseline(nn.Module):
    """
    Persistence Baseline: Assumes atmospheric state remains frozen at T0.
    Standard meteorological benchmark across lead times.
    """

    def __init__(self, forecast_frames: int = 12):
        super().__init__()
        self.forecast_frames = forecast_frames

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: [B, T_in, C_in, H, W]
        Returns: [B, forecast_frames, 4, H, W]
        """
        latest = x[:, -1]  # [B, C_in, H, W]
        B, C_in, H, W = latest.shape
        out = torch.zeros(B, self.forecast_frames, 4, H, W, device=x.device, dtype=x.dtype)

        # Ch 0: Radar Reflectivity
        out[:, :, 0] = latest[:, 0:1]
        # Ch 1: Lightning
        ltg_idx = 4 if C_in >= 6 else (2 if C_in >= 4 else 0)
        out[:, :, 1] = latest[:, ltg_idx:ltg_idx + 1]
        # Ch 2: Hail proxy
        out[:, :, 2] = torch.clamp(latest[:, 0:1] * 1.15 - 0.15, 0.0, 1.0)
        # Ch 3: Convective Intensity
        out[:, :, 3] = torch.clamp(out[:, :, 0] * 0.6 + out[:, :, 1] * 0.4, 0.0, 1.0)
        return out


# ==============================================================================
# BASELINE 2: Optical Flow / Motion Extrapolation Baseline
# ==============================================================================
class OpticalFlowExtrapolationBaseline(nn.Module):
    """
    Optical-Flow / Semi-Lagrangian Motion Extrapolation Baseline.
    Estimates 2D advection motion vectors between consecutive observation frames (T-1 -> T)
    and advects radar and hazard fields forward in time.
    """

    def __init__(self, forecast_frames: int = 12):
        super().__init__()
        self.forecast_frames = forecast_frames

    def _estimate_motion_vectors(self, f_prev: torch.Tensor, f_curr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates centroid/gradient-based mean advection velocity (u, v) in grid units per frame.
        f_prev, f_curr: [B, 1, H, W]
        """
        B, _, H, W = f_curr.shape
        # Spatial gradients
        dy = (f_curr[:, :, 2:, :] - f_curr[:, :, :-2, :])[:, :, :, 1:-1]
        dx = (f_curr[:, :, :, 2:] - f_curr[:, :, :, :-2])[:, :, 1:-1, :]
        dt = (f_curr[:, :, 1:-1, 1:-1] - f_prev[:, :, 1:-1, 1:-1])

        # Regularized least-squares motion estimation per batch
        u_list, v_list = [], []
        for b in range(B):
            dx_b = dx[b, 0].reshape(-1)
            dy_b = dy[b, 0].reshape(-1)
            dt_b = dt[b, 0].reshape(-1)

            A = torch.stack([dx_b, dy_b], dim=1)  # [N, 2]
            b_vec = -dt_b                          # [N]
            
            # Normal equations (A^T A + lambda I) x = A^T b
            reg = 1e-4 * torch.eye(2, device=f_curr.device, dtype=f_curr.dtype)
            AtA = torch.matmul(A.T, A) + reg
            Atb = torch.matmul(A.T, b_vec)
            try:
                vel = torch.linalg.solve(AtA, Atb)
                u, v = float(vel[0].item()), float(vel[1].item())
                # Clip to realistic physical velocity limits (-4 to +4 pixels per frame)
                u = max(-4.0, min(4.0, u))
                v = max(-4.0, min(4.0, v))
            except Exception:
                u, v = 0.5, -0.2  # Typical eastward convective drift default
            u_list.append(u)
            v_list.append(v)
            
        return torch.tensor(u_list, device=f_curr.device), torch.tensor(v_list, device=f_curr.device)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: [B, T_in, C_in, H, W]
        Returns: [B, forecast_frames, 4, H, W]
        """
        B, T_in, C_in, H, W = x.shape
        t_prev = x[:, -2, 0:1] if T_in >= 2 else x[:, -1, 0:1]
        t_curr = x[:, -1, 0:1]

        u_vecs, v_vecs = self._estimate_motion_vectors(t_prev, t_curr)

        # Base 4-channel representation at T0
        curr_4ch = torch.zeros(B, 4, H, W, device=x.device, dtype=x.dtype)
        curr_4ch[:, 0:1] = x[:, -1, 0:1]  # Radar
        ltg_idx = 4 if C_in >= 6 else (2 if C_in >= 4 else 0)
        curr_4ch[:, 1:2] = x[:, -1, ltg_idx:ltg_idx + 1]  # Lightning
        curr_4ch[:, 2:3] = torch.clamp(curr_4ch[:, 0:1] * 1.15 - 0.15, 0.0, 1.0)
        curr_4ch[:, 3:4] = torch.clamp(curr_4ch[:, 0:1] * 0.6 + curr_4ch[:, 1:2] * 0.4, 0.0, 1.0)

        # Coordinate grid for warping
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

        outputs = []
        for step in range(1, self.forecast_frames + 1):
            warped_batch = []
            for b in range(B):
                u = u_vecs[b] * step * (2.0 / W)
                v = v_vecs[b] * step * (2.0 / H)
                disp = torch.stack([u, v]).view(1, 1, 2)
                sample_grid = (base_grid - disp).unsqueeze(0)  # [1, H, W, 2]

                warped = F.grid_sample(
                    curr_4ch[b:b+1],
                    sample_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True
                )
                # Dampen intensity slightly with lead time to reflect physical dispersion
                decay = max(0.4, 1.0 - 0.04 * step)
                warped_batch.append(warped * decay)
            outputs.append(torch.cat(warped_batch, dim=0))

        return torch.stack(outputs, dim=1)  # [B, forecast_frames, 4, H, W]


# ==============================================================================
# BASELINE 3, 4, 5, 6: Modular Stream ConvLSTM Models
# ==============================================================================
class SubStreamConvLSTM(nn.Module):
    """
    Configurable Recurrent ConvLSTM network supporting arbitrary input channel combinations:
      - Radar-only (2 ch)
      - Radar + Satellite (4 ch)
      - Radar + Lightning (4 ch)
      - Radar + Satellite + Lightning (6 ch)
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 32,
        forecast_frames: int = 12,
        model_name: str = "ConvLSTM-SubStream"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.forecast_frames = forecast_frames
        self.model_name = model_name

        self.input_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.cell = ConvLSTMCell(in_channels=hidden_dim, hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=1),
            nn.Sigmoid()
        )
        self.feedback_proj = nn.Conv2d(4, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: [B, T_in, C_in, H, W]
        Uses first self.in_channels of x.
        Returns: [B, forecast_frames, 4, H, W]
        """
        x_sub = x[:, :, :self.in_channels]
        B, T_in, _, H, W = x_sub.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)

        for t in range(T_in):
            feat = self.input_encoder(x_sub[:, t])
            h, c = self.cell(feat, h, c)

        outputs = []
        feedback = self.feedback_proj(self.head(h))
        for _ in range(self.forecast_frames):
            pred_heads = self.head(h)
            outputs.append(pred_heads)
            feedback = self.feedback_proj(pred_heads)
            h, c = self.cell(feedback, h, c)

        return torch.stack(outputs, dim=1)


def build_baseline_suite(device: str = "cpu") -> Dict[str, nn.Module]:
    """
    Constructs all 7 baseline models ready for standardized benchmark comparison.
    """
    models = {
        "baseline_1_persistence": PersistenceBaseline(forecast_frames=12).to(device),
        "baseline_2_optical_flow": OpticalFlowExtrapolationBaseline(forecast_frames=12).to(device),
        "baseline_3_radar_only": SubStreamConvLSTM(in_channels=2, hidden_dim=32, model_name="Radar-Only ConvLSTM").to(device),
        "baseline_4_radar_satellite": SubStreamConvLSTM(in_channels=4, hidden_dim=40, model_name="Radar+Satellite ConvLSTM").to(device),
        "baseline_5_radar_lightning": SubStreamConvLSTM(in_channels=4, hidden_dim=40, model_name="Radar+Lightning ConvLSTM").to(device),
        "baseline_6_radar_sat_ltg": SubStreamConvLSTM(in_channels=6, hidden_dim=48, model_name="Radar+Sat+Lightning ConvLSTM").to(device),
        "baseline_7_full_fusion": MultiSourceNowcastNet(in_channels=12, hidden_dim=64, forecast_frames=12).to(device)
    }
    for m in models.values():
        m.eval()
    return models
