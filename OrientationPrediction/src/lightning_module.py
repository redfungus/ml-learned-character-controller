"""
PyTorch Lightning module for orientation prediction training.
"""

from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.losses import get_loss_function
from src.models import get_model
from src.training.schedulers import ExponentialDecayScheduler
from src.utils.quaternion import (angular_error,
                                  delta_quaternion_to_angular_velocity,
                                  ensure_same_hemisphere, normalize_quaternion,
                                  quaternion_conjugate, quaternion_multiply)


class OrientationPredictionModule(pl.LightningModule):
    """
    Lightning module for orientation prediction training.
    Supports single-step and multi-step rollout with teacher forcing.
    """

    def __init__(
        self,
        model_name: str = "mlp",
        model_kwargs: Optional[dict] = None,
        loss_name: str = "antipodal_mse",
        loss_kwargs: Optional[dict] = None,
        learning_rate: float = 1e-3,
        rollout_config: Optional[dict] = None,
        norm_loss_weight: float = 0.01,
        predict_angular_velocity: bool = False,
        angular_velocity_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.model_name = model_name

        # Angular velocity prediction config
        self.predict_angular_velocity = predict_angular_velocity
        self.angular_velocity_loss_weight = angular_velocity_loss_weight

        # Build model
        model_kwargs = model_kwargs or {}
        self.model = get_model(model_name, **model_kwargs)

        # Build loss function
        loss_kwargs = loss_kwargs or {}
        self.loss_fn = get_loss_function(loss_name, **loss_kwargs)
        self.loss_name = loss_name

        # Norm regularization weight (QuaterNet-style: penalize deviation from unit norm)
        self.norm_loss_weight = norm_loss_weight

        # Multi-step rollout configuration
        self.rollout_config = rollout_config or {}
        self.rollout_enabled = self.rollout_config.get("enabled", False)
        if self.rollout_enabled:
            self.rollout_steps = self.rollout_config.get("steps", 1)
        else:
            self.rollout_steps = 1  # Single-step mode when rollout disabled

        # Teacher forcing configuration (for rollout training)
        self.teacher_forcing_start = self.rollout_config.get(
            "teacher_forcing_start", 1.0
        )
        self.teacher_forcing_end = self.rollout_config.get("teacher_forcing_end", 0.2)
        self.teacher_forcing_ratio = self.teacher_forcing_start
        self.sampling_scheduler: Optional[ExponentialDecayScheduler] = None

        if self.rollout_enabled and self.rollout_steps > 1:
            # Setup exponential decay teacher forcing scheduler
            decay_rate = self.rollout_config.get("decay_rate", 0.005)
            self.sampling_scheduler = ExponentialDecayScheduler(
                start_ratio=self.teacher_forcing_start,
                decay_rate=decay_rate,
                min_ratio=self.teacher_forcing_end,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _compute_loss(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss with optional multi-step rollout.
        Teacher forcing: randomly use ground truth vs predicted state for next step.
        """
        features = batch["features"]  # [B, rollout_steps, 6]
        targets = batch["targets"]  # [B, rollout_steps, 4 or 7]

        device = features.device
        batch_size = features.shape[0]

        # Initialize from first step
        # features layout: [pad_x, pad_y, vel_x, vel_y, vel_z, dt]
        current_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(
            batch_size, -1
        )  # Identity quaternion [B, 4]
        pad_input = features[:, 0, 0:2]  # [B, 2]
        angular_vel = features[:, 0, 2:5]  # [B, 3]
        dt = features[:, 0, 5:6]  # [B, 1]

        total_loss = torch.tensor(0.0, device=device)
        total_norm_loss = torch.tensor(0.0, device=device)
        total_ang_error = torch.tensor(0.0, device=device)
        max_ang_error = torch.tensor(0.0, device=device)
        total_vel_loss = torch.tensor(0.0, device=device)

        for step in range(self.rollout_steps):
            # Build input for current step (6D: no orientation)
            step_input = torch.cat([pad_input, angular_vel, dt], dim=-1)

            # Predict delta quaternion (and optionally angular velocity)
            output = self.model(step_input)

            # Split output based on prediction mode
            if self.predict_angular_velocity:
                pred_delta_q = output[..., :4]
                pred_vel = output[..., 4:]
                target_delta = targets[:, step, :4]
                target_vel = targets[:, step, 4:]
            else:
                pred_delta_q = output
                target_delta = targets[:, step]

            # Accumulate norm loss if enabled
            if self.norm_loss_weight > 0:
                total_norm_loss = total_norm_loss + self.model.get_norm_loss()

            # Compute quaternion loss for this step
            step_loss = self.loss_fn(pred_delta_q, target_delta)
            total_loss = total_loss + step_loss

            # Compute angular velocity loss if enabled
            if self.predict_angular_velocity:
                vel_loss = F.mse_loss(pred_vel, target_vel)
                total_vel_loss = total_vel_loss + vel_loss

            # Compute angular error
            with torch.no_grad():
                step_ang_error = angular_error(pred_delta_q, target_delta, degrees=True)
                total_ang_error = total_ang_error + step_ang_error.mean()
                max_ang_error = torch.max(max_ang_error, step_ang_error.max())

            # Prepare state for next step (if not last step)
            if step < self.rollout_steps - 1:
                # Teacher forcing: use ground truth angular velocity; otherwise use prediction
                if (
                    stage == "train"
                    and torch.rand(1, device=device).item() < self.teacher_forcing_ratio
                ):
                    # Use ground truth angular velocity from features
                    angular_vel = features[:, step + 1, 2:5]
                else:
                    pred_new_quat = quaternion_multiply(pred_delta_q, current_quat)
                    pred_new_quat = normalize_quaternion(pred_new_quat)
                    pred_new_quat = ensure_same_hemisphere(pred_new_quat, current_quat)

                    if self.predict_angular_velocity:
                        angular_vel = pred_vel
                    else:
                        angular_vel = delta_quaternion_to_angular_velocity(
                            pred_delta_q, dt
                        )
                    current_quat = pred_new_quat

                pad_input = features[:, step + 1, 0:2]
                dt = features[:, step + 1, 5:6]

        # Average losses over steps
        quat_loss = total_loss / self.rollout_steps
        norm_loss = total_norm_loss / self.rollout_steps
        mean_ang_error = total_ang_error / self.rollout_steps

        # Start with quaternion loss
        loss = quat_loss

        # Add angular velocity loss if enabled
        if self.predict_angular_velocity:
            vel_loss = total_vel_loss / self.rollout_steps
            loss = loss + self.angular_velocity_loss_weight * vel_loss
            self.log(f"{stage}/vel_loss", vel_loss, on_step=False, on_epoch=True)

        # Add norm regularization
        if self.norm_loss_weight > 0:
            loss = loss + self.norm_loss_weight * norm_loss

        # Log metrics
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/quat_loss", quat_loss, on_step=False, on_epoch=True)
        self.log(
            f"{stage}/angular_error_mean",
            mean_ang_error,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}/angular_error_max", max_ang_error, on_step=False, on_epoch=True
        )
        if self.norm_loss_weight > 0:
            self.log(f"{stage}/norm_loss", norm_loss, on_step=False, on_epoch=True)
        if stage == "train" and self.rollout_steps > 1:
            self.log(
                "train/teacher_forcing_ratio",
                self.teacher_forcing_ratio,
                on_step=False,
                on_epoch=True,
            )

        return {
            "loss": loss,
            "angular_error_mean": mean_ang_error,
            "angular_error_max": max_ang_error,
        }

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        result = self._compute_loss(batch, "train")
        return result["loss"]

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        result = self._compute_loss(batch, "val")
        return result["loss"]

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        result = self._compute_loss(batch, "test")
        return result["loss"]

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.999,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_train_epoch_start(self):
        """Update teacher forcing ratio at the start of each epoch."""
        if self.sampling_scheduler is not None:
            self.teacher_forcing_ratio = (
                self.sampling_scheduler.get_teacher_forcing_ratio(self.current_epoch)
            )

    def on_train_epoch_end(self):
        """Log learning rate at end of epoch."""
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=False, on_epoch=True)
