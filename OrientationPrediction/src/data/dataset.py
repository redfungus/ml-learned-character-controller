"""PyTorch Dataset for orientation prediction."""

from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class OrientationDataset(Dataset):
    """
    Dataset for orientation prediction. Loads CSV with quaternion features/targets.
    Expects preprocessed data with continuous quaternions.
    """

    # Column names for features (matching generate_data.py output)
    ORIENTATION_COLS = [
        "current_orientation_w",
        "current_orientation_x",
        "current_orientation_y",
        "current_orientation_z",
    ]
    CONTROL_COLS = ["pad_input_x", "pad_input_y"]
    ANGULAR_VEL_COLS = [
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
    ]
    DT_COL = ["timestep_seconds"]

    # Column names for targets
    TARGET_COLS = [
        "orientation_change_w",
        "orientation_change_x",
        "orientation_change_y",
        "orientation_change_z",
    ]

    def __init__(
        self,
        data_path: str,
        rollout_steps: int = 1,
        transform: Optional[Callable[[dict], dict]] = None,
        predict_angular_velocity: bool = False,
    ):
        self.rollout_steps = rollout_steps
        self.transform = transform
        self.predict_angular_velocity = predict_angular_velocity

        self.df = pd.read_csv(data_path)
        self._validate_columns()
        self._prepare_data()

    def _validate_columns(self):
        """Validate that required columns exist."""
        required_cols = (
            self.ORIENTATION_COLS
            + self.CONTROL_COLS
            + self.ANGULAR_VEL_COLS
            + self.DT_COL
            + self.TARGET_COLS
        )

        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

    def _prepare_data(self):
        """Prepare features and targets tensors from dataframe."""
        feature_cols = (
            self.CONTROL_COLS
            + self.ANGULAR_VEL_COLS
            + self.DT_COL
        )
        features = self.df[feature_cols].values.astype(np.float32)
        quat_targets = self.df[self.TARGET_COLS].values.astype(np.float32)
        target_norms = np.linalg.norm(quat_targets, axis=1, keepdims=True)
        quat_targets = quat_targets / np.maximum(target_norms, 1e-8)

        if self.predict_angular_velocity:
            # Target includes next row's angular velocity
            next_angular_vel = (
                self.df[self.ANGULAR_VEL_COLS].values[1:].astype(np.float32)
            )
            features = features[:-1]
            quat_targets = quat_targets[:-1]
            targets = np.concatenate([quat_targets, next_angular_vel], axis=1)
        else:
            targets = quat_targets

        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        self.valid_indices = list(
            range(
                0,
                len(self.features) - self.rollout_steps + 1,
            )
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get sample with features and targets [rollout_steps, dim]."""
        actual_idx = self.valid_indices[idx]
        features = self.features[actual_idx : actual_idx + self.rollout_steps]
        target = self.targets[actual_idx : actual_idx + self.rollout_steps]

        sample = {"features": features, "targets": target}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @property
    def input_dim(self) -> int:
        """Return input feature dimension."""
        return self.features.shape[-1]

    @property
    def output_dim(self) -> int:
        """Return output dimension (accounts for transform if applied)."""
        # Get a sample to determine actual output dim after transform
        if self.transform is not None:
            sample = self[0]
            return sample["targets"].shape[-1]
        return self.targets.shape[-1]
