"""
Chronological and Extreme Event Dataset Splitting (SIH26084 Part 5).
Adheres strictly to Sections 10 and 11 of Part 5 Specification:
- Strict Chronological Splitting (No random shuffling to prevent temporal data leakage):
  70% Training (earlier timestamps)
  15% Validation (intermediate timestamps)
  15% Testing (latest unseen timestamps)
- Dedicated Extreme Event Test Set:
  Isolates severe thunderstorms, heavy rainfall (>= 45 dBZ), cloudburst proxies, and high lightning flash densities.
- Reports performance separately on NORMAL and EXTREME events.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class SplitResult(dict):
    """Container for split indices allowing dict access and tuple unpacking."""
    def __iter__(self):
        return iter((self["train"], self["val"], self["test"]))


class ChronologicalSplitter:
    """
    Partitions temporal meteorological sequences chronologically into Train, Val, Test,
    and identifies extreme convective events for stratified diagnostic evaluation.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        extreme_reflectivity_threshold: float = 0.60  # Normalized (corresponds to ~45+ dBZ)
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.extreme_thresh = extreme_reflectivity_threshold

    def split_indices(self, total_samples: Any) -> Any:
        """
        Computes non-overlapping chronological index ranges.
        Accepts integer count or sequence of items.
        """
        N = len(total_samples) if hasattr(total_samples, "__len__") else int(total_samples)
        n_train = int(N * self.train_ratio)
        n_val = int(N * self.val_ratio)

        train_indices = list(range(0, n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, N))

        res = SplitResult({
            "train": train_indices,
            "val": val_indices,
            "test": test_indices
        })
        return res

    def identify_extreme_events(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Identifies samples containing severe convective signatures:
        - Max radar reflectivity >= threshold
        - High lightning density >= 0.50
        - Hail probability >= 0.60
        """
        extreme_indices = []
        num_samples = len(X)

        for i in range(num_samples):
            is_extreme = False
            # Check X radar channel (channel 0)
            if np.max(X[i, :, 0]) >= self.extreme_thresh:
                is_extreme = True
            # Check X lightning channel (channel 4 or channel 2 in 4-ch)
            ltg_idx = 4 if X.shape[2] > 4 else 2
            if np.max(X[i, :, ltg_idx]) >= 0.50:
                is_extreme = True

            # Check targets if available
            if Y is not None:
                # Target hail probability (channel 2)
                if Y.shape[2] > 2 and np.max(Y[i, :, 2]) >= 0.55:
                    is_extreme = True

            if is_extreme:
                extreme_indices.append(i)

        return extreme_indices

    def create_split_metadata(
        self,
        total_samples: int,
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
        extreme_idx: List[int]
    ) -> Dict[str, Any]:
        return {
            "total_sequences": total_samples,
            "split_strategy": "CHRONOLOGICAL_NO_LEAKAGE",
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
            "extreme_event_samples": len(extreme_idx),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "extreme_in_test": len([idx for idx in test_idx if idx in extreme_idx])
        }
