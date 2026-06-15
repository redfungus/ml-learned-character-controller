"""MLP model for quaternion orientation prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class MLPModel(BaseModel):
    """
    MLP for orientation prediction. Output normalized to unit quaternion.
    Pre-normalization norm stored for QuaterNet-style regularization.
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        normalize_output: bool = True,
        predict_angular_velocity: bool = False,
    ):
        super().__init__(input_dim, output_dim, predict_angular_velocity)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.normalize_output = normalize_output

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns normalized quaternion delta."""
        output = self.network(x)

        if self.predict_angular_velocity:
            quat = output[..., :4]
            vel = output[..., 4:]
            self.last_output_norm = torch.norm(quat, p=2, dim=-1)
            if self.normalize_output:
                quat = F.normalize(quat, p=2, dim=-1)
            output = torch.cat([quat, vel], dim=-1)
        else:
            self.last_output_norm = torch.norm(output, p=2, dim=-1)
            if self.normalize_output:
                output = F.normalize(output, p=2, dim=-1)

        return output
