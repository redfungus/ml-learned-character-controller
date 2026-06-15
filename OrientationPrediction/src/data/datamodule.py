"""
PyTorch Lightning DataModule for orientation prediction.
"""

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import OrientationDataset


class OrientationDataModule(pl.LightningDataModule):
    """DataModule for orientation prediction. Handles splits and DataLoader creation."""

    def __init__(
        self,
        data_path: str,
        val_split: float = 0.15,
        test_split: float = 0.15,
        batch_size: int = 64,
        num_workers: int = 4,
        rollout_steps: int = 1,
        seed: int = 42,
        transform: Optional[Callable[[dict], dict]] = None,
        predict_angular_velocity: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])

        self.data_path = data_path
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rollout_steps = rollout_steps
        self.seed = seed
        self.transform = transform
        self.predict_angular_velocity = predict_angular_velocity

        self.train_dataset: Optional[OrientationDataset] = None
        self.val_dataset: Optional[OrientationDataset] = None
        self.test_dataset: Optional[OrientationDataset] = None

    def prepare_data(self):
        """Verify data file exists."""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                f"Run generate_data.py first."
            )

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.data_path)
        n_total = len(df)

        n_test = int(n_total * self.test_split)
        n_val = int(n_total * self.val_split)
        n_train = n_total - n_val - n_test

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train : n_train + n_val]
        test_df = df.iloc[n_train + n_val :]

        data_dir = Path(self.data_path).parent
        train_path = data_dir / "train_split.csv"
        val_path = data_dir / "val_split.csv"
        test_path = data_dir / "test_split.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        self.train_dataset = OrientationDataset(
            data_path=str(train_path),
            rollout_steps=self.rollout_steps,
            transform=self.transform,
            predict_angular_velocity=self.predict_angular_velocity,
        )
        self.val_dataset = OrientationDataset(
            data_path=str(val_path),
            rollout_steps=self.rollout_steps,
            transform=self.transform,
            predict_angular_velocity=self.predict_angular_velocity,
        )
        self.test_dataset = OrientationDataset(
            data_path=str(test_path),
            rollout_steps=self.rollout_steps,
            transform=self.transform,
            predict_angular_velocity=self.predict_angular_velocity,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def input_dim(self) -> int:
        """Return input feature dimension."""
        if self.train_dataset is not None:
            return self.train_dataset.input_dim
        # Default based on expected features
        return 6  # 2 (control) + 3 (ang_vel) + 1 (dt)

    @property
    def output_dim(self) -> int:
        """Return output dimension."""
        if self.train_dataset is not None:
            return self.train_dataset.output_dim
        # Default: 7 if predicting angular velocity, else 4
        return 7 if self.predict_angular_velocity else 4
