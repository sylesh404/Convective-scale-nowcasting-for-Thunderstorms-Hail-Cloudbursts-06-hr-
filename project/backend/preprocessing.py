"""
SIH26084 Convective Scale Nowcasting - Preprocessing Module
Validates, normalizes, and packages multi-channel meteorological grids into PyTorch ConvLSTM tensors.
Expected Tensor Format: [B=1, T=4, C=4, H=16, W=16]
"""

import os
from typing import Tuple, Dict, Any, Union
import numpy as np
import torch
from fastapi import HTTPException, status


def load_synthetic_data(data_path: str = None) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Loads raw numpy arrays from synthetic_nowcasting_dataset.npz.
    """
    if data_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "data", "synthetic_nowcasting_dataset.npz"),
            os.path.join(os.path.dirname(base_dir), "demo_pkg", "synthetic_nowcasting_dataset.npz"),
            os.path.join(os.getcwd(), "backend", "data", "synthetic_nowcasting_dataset.npz"),
            os.path.join(os.getcwd(), "data", "synthetic_nowcasting_dataset.npz")
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    if not data_path or not os.path.exists(data_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthetic dataset not found at '{data_path}'. Please ensure synthetic_nowcasting_dataset.npz is available."
        )

    try:
        archive = np.load(data_path)
        X = archive["X"]
        Y = archive["Y"] if "Y" in archive else None
        return X, Y
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse synthetic nowcasting dataset: {str(e)}"
        )


def normalize_channels(tensor_np: np.ndarray) -> np.ndarray:
    """
    Ensures input channels are bounded in [0.0, 1.0].
    Channels:
      0: Radar Reflectivity proxy
      1: INSAT thermal/cloud proxy
      2: Lightning density proxy
      3: Humidity proxy
    """
    if not isinstance(tensor_np, np.ndarray):
        tensor_np = np.asarray(tensor_np, dtype=np.float32)
    # Clip any physical sensor overshoots
    return np.clip(tensor_np, 0.0, 1.0).astype(np.float32)


def validate_input_shape(tensor: Union[np.ndarray, torch.Tensor]) -> None:
    """
    Validates tensor dimensions: [B=1, T=4, C=4, H=16, W=16] or [T=4, C=4, H=16, W=16].
    Raises HTTPException if shape or values are invalid.
    """
    shape = tuple(tensor.shape)
    
    if len(shape) == 4:
        # [T, C, H, W]
        t, c, h, w = shape
        if t != 4 or c != 4 or h != 16 or w != 16:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid unbatched input shape {shape}. Expected [T=4, C=4, H=16, W=16]."
            )
    elif len(shape) == 5:
        # [B, T, C, H, W]
        b, t, c, h, w = shape
        if b != 1 or t != 4 or c != 4 or h != 16 or w != 16:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid batched input shape {shape}. Expected [B=1, T=4, C=4, H=16, W=16]."
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unexpected tensor rank {len(shape)} with shape {shape}. Required rank 5: [B, T, C, H, W]."
        )


def prepare_model_tensor(
    sequence_data: Union[np.ndarray, torch.Tensor],
    device: str = "cpu"
) -> torch.Tensor:
    """
    Prepares a clean, validated PyTorch float32 tensor of shape [1, 4, 4, 16, 16] on target device.
    """
    if isinstance(sequence_data, torch.Tensor):
        arr = sequence_data.detach().cpu().numpy()
    else:
        arr = np.asarray(sequence_data, dtype=np.float32)

    validate_input_shape(arr)

    # Normalize values to [0, 1]
    arr = normalize_channels(arr)

    # Ensure batch dimension [1, 4, 4, 16, 16]
    if arr.ndim == 4:
        arr = np.expand_dims(arr, axis=0)

    validate_input_shape(arr)

    tensor = torch.from_numpy(arr).float().to(device)
    return tensor
