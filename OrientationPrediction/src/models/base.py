"""
Base model class for quaternion orientation prediction.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for orientation prediction models.

    All models should inherit from this class and implement the forward method.

    Args:
        input_dim: Dimension of input features
        output_dim: Dimension of output (4 for quaternion)
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 4,
        predict_angular_velocity: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.predict_angular_velocity = predict_angular_velocity
        self.last_output_norm: Optional[torch.Tensor] = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, input_dim] or [batch, seq_len, input_dim]

        Returns:
            Output tensor [batch, output_dim]
        """
        pass

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_norm_loss(self) -> torch.Tensor:
        """
        Get the norm regularization loss (QuaterNet-style).

        Must be called immediately after forward() to get the loss for the last batch.
        The norm is stored as instance state, so calling forward() again before
        get_norm_loss() will overwrite the previous norm.

        Returns:
            Mean squared difference from unit norm: mean((||output|| - 1)^2)
        """
        if self.last_output_norm is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        return ((self.last_output_norm - 1.0) ** 2).mean()
