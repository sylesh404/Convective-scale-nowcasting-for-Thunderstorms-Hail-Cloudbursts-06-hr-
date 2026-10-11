"""
Training Script for Multi-Source Convective Nowcasting Model
Part 5 — SIH26084 MoES / NCMRWF Convective-Scale Nowcasting (0-6 hr)
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.models.nowcast_model import MultiSourceNowcastNet, ConvLSTMBaseline, PersistenceBaseline
from training.losses.hazard_loss import MultiHazardLoss
from training.datasets.sequence_dataset import ConvectiveSequenceDataset
from training.metrics.forecast_metrics import ForecastMetricsEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Source Nowcasting Model")
    parser.add_argument("--config", type=str, default="training/training/config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataset .npz files")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--experiment-name", type=str, default=None, help="Name for this experiment run")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_experiment_dir(base_dir: str, exp_name: Optional[str] = None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    if exp_name:
        exp_dir = base / exp_name
    else:
        existing = list(base.glob("experiment_*"))
        next_idx = len(existing) + 1
        exp_dir = base / f"experiment_{next_idx:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    loss_components = {"rainfall": 0.0, "lightning": 0.0, "hail": 0.0, "convective": 0.0}
    num_batches = 0

    for batch in dataloader:
        x = batch["x"].to(device)  # [B, T_in, C, H, W]
        y = batch["y"].to(device)  # [B, T_out, 4, H, W]
        masks = batch.get("masks")
        if masks is not None:
            masks = masks.to(device)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                preds = model(x, masks=masks)
                loss, comps = criterion(preds, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(x, masks=masks)
            loss, comps = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += comps.get(k, 0.0)
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_comps = {f"{k}_loss": v / max(1, num_batches) for k, v in loss_components.items()}
    avg_comps["train_loss"] = avg_loss
    return avg_comps


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    loss_components = {"rainfall": 0.0, "lightning": 0.0, "hail": 0.0, "convective": 0.0}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            masks = batch.get("masks")
            if masks is not None:
                masks = masks.to(device)

            preds = model(x, masks=masks)
            loss, comps = criterion(preds, y)

            total_loss += loss.item()
            for k in loss_components:
                loss_components[k] += comps.get(k, 0.0)
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_comps = {f"val_{k}_loss": v / max(1, num_batches) for k, v in loss_components.items()}
    avg_comps["val_loss"] = avg_loss
    return avg_comps


def main():
    args = parse_args()
    config = load_config(args.config)

    # Overrides
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.learning_rate or config["training"]["learning_rate"]
    data_dir = Path(args.data_dir or config["paths"]["data_dir"])
    chkpt_dir = Path(args.checkpoint_dir or config["paths"]["checkpoint_dir"])
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Experiment setup
    exp_dir = setup_experiment_dir(config["paths"]["experiments_dir"], args.experiment_name)
    print(f"[*] Experiment directory: {exp_dir}")

    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # Datasets
    train_npz = data_dir / "train.npz"
    val_npz = data_dir / "validation.npz"

    # Fallback to mock dataset if .npz does not exist yet
    if train_npz.exists():
        print(f"[*] Loading training set from {train_npz}")
        train_ds = ConvectiveSequenceDataset(npz_path=str(train_npz))
    else:
        print(f"[!] {train_npz} not found. Generating mock sequences for pipeline test.")
        train_ds = ConvectiveSequenceDataset(mock_samples=32)

    if val_npz.exists():
        print(f"[*] Loading validation set from {val_npz}")
        val_ds = ConvectiveSequenceDataset(npz_path=str(val_npz))
    else:
        val_ds = ConvectiveSequenceDataset(mock_samples=16)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Instantiate Model
    model = MultiSourceNowcastNet(
        in_channels=config["channels"]["input_total"],
        hidden_dim=config["model"]["hidden_dim"],
        forecast_frames=config["temporal"]["forecast_frames"]
    ).to(device)

    # Instantiate Loss
    loss_weights = config.get("loss_weights", {})
    criterion = MultiHazardLoss(
        precip_weight=loss_weights.get("precipitation", 1.5),
        lightning_weight=loss_weights.get("lightning", 1.2),
        hail_weight=loss_weights.get("hail", 1.0),
        convective_weight=loss_weights.get("convective_intensity", 1.0)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler() if (config["training"]["mixed_precision"] and device.type == "cuda") else None

    start_epoch = 0
    best_val_loss = float("inf")
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0
    history = []

    if args.resume and Path(args.resume).exists():
        print(f"[*] Resuming from checkpoint: {args.resume}")
        chk = torch.load(args.resume, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk.get("epoch", 0) + 1
        best_val_loss = chk.get("best_val_loss", float("inf"))

    print(f"[*] Starting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config["training"]["grad_clip_norm"],
            scaler=scaler
        )

        val_stats = validate(model=model, dataloader=val_loader, criterion=criterion, device=device)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_stats["val_loss"])

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_stats["train_loss"], 4),
            "val_loss": round(val_stats["val_loss"], 4),
            "lr": current_lr,
            "duration_sec": round(elapsed, 2),
            **{k: round(v, 4) for k, v in train_stats.items() if k != "train_loss"},
            **{k: round(v, 4) for k, v in val_stats.items() if k != "val_loss"}
        }
        history.append(epoch_record)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
              f"Train Loss: {train_stats['train_loss']:.4f} | "
              f"Val Loss: {val_stats['val_loss']:.4f} | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.2f}s")

        # Save latest checkpoint
        latest_path = chkpt_dir / "latest_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_stats["val_loss"],
            "config": config,
            "model_version": config["model"]["version"]
        }, latest_path)

        # Save best model
        if val_stats["val_loss"] < best_val_loss:
            best_val_loss = val_stats["val_loss"]
            patience_counter = 0
            best_path = chkpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
                "model_version": config["model"]["version"]
            }, best_path)
            # Copy to experiment dir
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
            print(f"  >>> New best validation loss: {best_val_loss:.4f} — Checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[*] Early stopping triggered after {patience} epochs without improvement.")
                break

    # Save training history
    with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(exp_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"SIH26084 Model Training Summary\n")
        f.write(f"Model: {config['model']['name']} ({config['model']['version']})\n")
        f.write(f"Completed Epochs: {len(history)}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Final LR: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Timestamp: {datetime.utcnow().isoformat()}Z\n")

    print(f"[*] Training finished. Artifacts saved in {exp_dir} and {chkpt_dir}")


if __name__ == "__main__":
    main()
