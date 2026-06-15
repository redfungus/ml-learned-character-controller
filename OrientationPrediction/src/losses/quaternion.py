"""
Loss functions for quaternion orientation prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


def antipodal_mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute per-sample antipodal MSE: min(||pred - q||², ||pred + q||²)."""
    d_pos = torch.sum((pred - target) ** 2, dim=-1)
    d_neg = torch.sum((pred + target) ** 2, dim=-1)
    return torch.minimum(d_pos, d_neg)


class AntipodalMSELoss(nn.Module):
    """MSE with antipodal handling: min(||pred - q||², ||pred + q||²)."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(antipodal_mse_per_sample(pred, target))


class GeodesicLoss(nn.Module):

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        dot = torch.abs((pred * target).sum(dim=-1))
        dot = torch.clamp(dot, 0.0, 1.0 - self.eps)
        angle = 2.0 * torch.acos(dot)
        return angle.mean()


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """Factory function to get loss by name."""
    loss_functions = {
        "mse": MSELoss,
        "antipodal_mse": AntipodalMSELoss,
        "geodesic": GeodesicLoss,
    }

    if name not in loss_functions:
        raise ValueError(
            f"Unknown loss function: {name}. "
            f"Available: {list(loss_functions.keys())}"
        )

    return loss_functions[name](**kwargs)
