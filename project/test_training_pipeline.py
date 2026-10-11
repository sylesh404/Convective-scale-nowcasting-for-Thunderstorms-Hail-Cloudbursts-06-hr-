"""
Comprehensive Pytest Test Suite for Part 5 AI Model Training, Multi-Source Fusion & Validation
SIH26084 MoES / NCMRWF Convective-Scale Nowcasting (0-6 hr)
"""

import os
import sys
import json
import pytest
import numpy as np
import torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.preprocessing.spatial_regridding import SpatialRegridder
from training.preprocessing.temporal_alignment import TemporalAligner
from training.preprocessing.normalization import NormalizationEngine
from training.datasets.split_dataset import ChronologicalSplitter
from training.datasets.sequence_dataset import ConvectiveSequenceDataset
from training.datasets.dataset_builder import DatasetBuilder
from training.models.radar_encoder import RadarEncoder
from training.models.satellite_encoder import SatelliteEncoder
from training.models.lightning_encoder import LightningEncoder
from training.models.nwp_encoder import NWPEncoder
from training.models.fusion_model import FeatureFusion
from training.models.nowcast_model import MultiSourceNowcastNet, PersistenceBaseline, ConvLSTMBaseline
from training.losses.precipitation_loss import WeightedPrecipitationLoss
from training.losses.lightning_loss import FocalLightningLoss
from training.losses.hazard_loss import MultiHazardLoss
from training.metrics.rainfall_metrics import RainfallMetrics
from training.metrics.lightning_metrics import LightningMetrics
from training.metrics.hail_metrics import HailMetrics
from training.metrics.forecast_metrics import ForecastMetricsEvaluator
from inference.run_nowcast import NowcastInferenceService


class TestPart5Preprocessing:
    """Test Spatial Regridding, Temporal Alignment, and Normalization."""

    def test_spatial_regridding(self):
        regridder = SpatialRegridder(grid_resolution_km=3.0, grid_size=(64, 64))
        grid_info = regridder.get_grid_metadata()
        assert grid_info["resolution_km"] == 3.0
        assert grid_info["grid_shape"] == [64, 64]
        assert "lat_bounds" in grid_info
        assert "lon_bounds" in grid_info

        # Test interpolation
        dummy = np.random.rand(32, 32).astype(np.float32)
        regridded = regridder.regrid_to_common_grid(dummy, target_shape=(64, 64))
        assert regridded.shape == (64, 64)

    def test_temporal_alignment(self):
        aligner = TemporalAligner(
            input_interval_min=15,
            forecast_interval_min=30,
            input_history_hours=1,
            forecast_hours=6
        )
        t_in, t_out = aligner.get_sequence_lengths()
        assert t_in == 4  # T-45, T-30, T-15, T
        assert t_out == 12  # +30m to +360m at 30m steps

    def test_normalization_engine(self, tmp_path):
        norm_file = str(tmp_path / "test_norm.json")
        engine = NormalizationEngine()
        
        # Test physical normalization
        refl = np.array([0.0, 35.0, 75.0], dtype=np.float32)
        norm_refl = engine.normalize("radar_reflectivity", refl)
        assert 0.0 <= norm_refl.min() <= norm_refl.max() <= 1.0

        # Inverted IR (colder cloud top -> higher normalized value)
        ir_warm = np.array([290.0], dtype=np.float32)
        ir_cold = np.array([210.0], dtype=np.float32)
        assert engine.normalize("satellite_ir", ir_cold) > engine.normalize("satellite_ir", ir_warm)

        # Save and reload stats
        engine.save(norm_file)
        assert os.path.exists(norm_file)
        new_engine = NormalizationEngine(stats_path=norm_file)
        assert "radar_reflectivity" in new_engine.stats


class TestPart5Datasets:
    """Test Chronological Splitting, Sequence Streaming, and Missing Data Masks."""

    def test_chronological_split(self):
        splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        N = 100
        indices = list(range(N))
        splits = splitter.split_indices(indices)
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15
        # Ensure chronological ordering (no future leakage)
        assert max(splits["train"]) < min(splits["val"])
        assert max(splits["val"]) < min(splits["test"])

    def test_sequence_dataset_mock(self):
        ds = ConvectiveSequenceDataset(mock_samples=8)
        assert len(ds) == 8
        item = ds[0]
        assert "x" in item and "y" in item and "masks" in item
        assert item["x"].shape == (4, 12, 64, 64)
        assert item["y"].shape == (12, 4, 64, 64)
        assert item["masks"].shape == (4, 12, 64, 64)

    def test_missing_data_mask_integrity(self):
        # Section 8: Value channel + availability mask
        x = np.ones((1, 4, 12, 32, 32), dtype=np.float32)
        masks = np.ones((1, 4, 12, 32, 32), dtype=np.float32)
        
        # Simulate missing Radar (Channel 0 and 1)
        x[0, :, 0:2] = 0.0
        masks[0, :, 0:2] = 0.0

        ds = ConvectiveSequenceDataset(X=x, masks=masks)
        sample = ds[0]
        assert torch.all(sample["x"][:, 0:2] == 0.0)
        assert torch.all(sample["masks"][:, 0:2] == 0.0)
        assert torch.all(sample["masks"][:, 2:] == 1.0)


class TestPart5Models:
    """Test Multi-Source Encoders, Feature Fusion, Temporal ConvLSTM, and Baselines."""

    def test_sensor_encoders(self):
        B, H, W = 2, 32, 32
        
        r_enc = RadarEncoder(in_channels=2, feature_dim=16)
        r_out = r_enc(torch.randn(B, 2, H, W))
        assert r_out.shape == (B, 16, H, W)

        s_enc = SatelliteEncoder(in_channels=2, feature_dim=16)
        s_out = s_enc(torch.randn(B, 2, H, W))
        assert s_out.shape == (B, 16, H, W)

        l_enc = LightningEncoder(in_channels=2, feature_dim=16)
        l_out = l_enc(torch.randn(B, 2, H, W))
        assert l_out.shape == (B, 16, H, W)

        n_enc = NWPEncoder(in_channels=6, feature_dim=16)
        n_out = n_enc(torch.randn(B, 6, H, W))
        assert n_out.shape == (B, 16, H, W)

    def test_feature_fusion(self):
        B, C, H, W = 2, 16, 32, 32
        fusion = FeatureFusion(feature_dim=C, fused_dim=32)
        f_out = fusion(
            torch.randn(B, C, H, W),
            torch.randn(B, C, H, W),
            torch.randn(B, C, H, W),
            torch.randn(B, C, H, W)
        )
        assert f_out.shape == (B, 32, H, W)

    def test_multisource_nowcast_forward_shape(self):
        B, T_in, C_in, H, W = 2, 4, 12, 32, 32
        model = MultiSourceNowcastNet(
            in_channels=C_in,
            feature_dim=8,
            hidden_dim=16,
            future_steps=12
        )
        x = torch.randn(B, T_in, C_in, H, W)
        masks = torch.ones(B, T_in, C_in, H, W)
        
        out = model(x, masks=masks)
        # Expected Output: [B, 12 future frames, 4 heads, H, W]
        assert out.shape == (B, 12, 4, H, W)
        assert torch.all((out >= 0.0) & (out <= 1.0))  # Heads use Sigmoid

    def test_baselines(self):
        B, T_in, C_in, H, W = 2, 4, 12, 32, 32
        x = torch.rand(B, T_in, C_in, H, W)

        # Persistence Baseline
        pers = PersistenceBaseline(future_steps=12)
        pers_out = pers(x)
        assert pers_out.shape == (B, 12, 4, H, W)

        # ConvLSTM Baseline
        conv_base = ConvLSTMBaseline(in_channels=12, hidden_dim=16, future_steps=12)
        cb_out = conv_base(x)
        assert cb_out.shape == (B, 12, 4, H, W)


class TestPart5LossesAndMetrics:
    """Test Multi-Hazard Loss, Lead-Time Metrics Table, Storm Tracking, and Hazard IoU."""

    def test_loss_computation(self):
        B, T, C, H, W = 2, 6, 4, 32, 32
        pred = torch.rand(B, T, C, H, W)
        target = torch.rand(B, T, C, H, W)
        
        criterion = MultiHazardLoss(
            precip_weight=1.5,
            lightning_weight=1.2,
            hail_weight=1.0,
            convective_weight=1.0
        )
        loss, components = criterion(pred, target)
        assert loss.item() > 0.0
        assert "rainfall" in components
        assert "lightning" in components
        assert "hail" in components
        assert "convective" in components

    def test_lead_time_metrics_table(self):
        evaluator = ForecastMetricsEvaluator(grid_resolution_km=3.0, total_steps=6)
        preds = np.random.rand(4, 6, 4, 32, 32).astype(np.float32)
        targets = np.random.rand(4, 6, 4, 32, 32).astype(np.float32)
        
        table = evaluator.evaluate_lead_time_table(preds, targets, threshold_dbz=35.0)
        assert len(table) == 6
        for row in table:
            assert "lead_time" in row
            assert "rmse_dbz" in row
            assert "csi" in row
            assert "pod" in row
            assert "far" in row

    def test_storm_tracking_kinematics(self):
        evaluator = ForecastMetricsEvaluator(grid_resolution_km=3.0, total_steps=4)
        preds = np.zeros((2, 4, 4, 32, 32), dtype=np.float32)
        targets = np.zeros((2, 4, 4, 32, 32), dtype=np.float32)
        
        # Place storm core in center
        preds[:, :, 0, 16, 16] = 0.8
        targets[:, :, 0, 16, 18] = 0.8  # 2 pixels displacement = 6.0 km

        kinematics = evaluator.evaluate_storm_track_kinematics(preds, targets)
        assert "mean_storm_displacement_error_km" in kinematics
        assert kinematics["mean_storm_displacement_error_km"] is not None
        assert abs(kinematics["mean_storm_displacement_error_km"] - 6.0) < 1.0

    def test_hazard_masks_iou(self):
        evaluator = ForecastMetricsEvaluator()
        pred_hail = np.zeros((2, 4, 32, 32), dtype=np.float32)
        target_hail = np.zeros((2, 4, 32, 32), dtype=np.float32)
        pred_hail[:, :, 10:20, 10:20] = 0.7
        target_hail[:, :, 10:20, 10:20] = 0.7

        metrics = evaluator.evaluate_hazard_masks(pred_hail, target_hail, threshold=0.50)
        assert metrics["iou"] == 1.0
        assert metrics["csi"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0


class TestPart5InferenceAndRegistry:
    """Test Production NowcastInferenceService, Model Registry, and Downburst Placeholder."""

    def test_nowcast_inference_service(self):
        service = NowcastInferenceService.get_instance()
        res = service.run_inference(is_synthetic=True)
        
        assert res["model_version"] == "ConvLSTM-Fusion-v1.0"
        assert res["forecast_horizon_hours"] == 6
        assert len(res["outputs"]) == 12
        assert "confidence" in res
        assert "confidence_score" in res["confidence"]
        assert "storm_tracks" in res
        # Downburst capability check (Section 20 & 40)
        assert "downburst" in res["hazard_capabilities"]
        assert "unavailable" in res["hazard_capabilities"]["downburst"].lower()

    def test_model_registry_json(self):
        reg_file = PROJECT_ROOT / "training" / "models" / "model_registry.json"
        assert reg_file.exists()
        with open(reg_file, "r", encoding="utf-8") as f:
            registry = json.load(f)
        assert "ConvLSTM-Fusion-v1.0" in registry["models"]
        info = registry["models"]["ConvLSTM-Fusion-v1.0"]
        assert info["grid_resolution_km"] == 3.0
        assert info["forecast_horizon_hours"] == 6.0
        assert len(info["input_channels"]) == 12
        assert "downburst_head" in info["placeholders"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
