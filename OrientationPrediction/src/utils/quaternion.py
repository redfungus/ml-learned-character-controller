"""
Quaternion utility functions for rotation representation and conversion.

Quaternion convention: [w, x, y, z] where w is the scalar component.

References:
- Zhou et al. (2019) - "On the Continuity of Rotation Representations in Neural Networks"
- Daniel Holden's blog (theorangeduck.com) - Practical rotation handling
"""

import numpy as np
import torch
import torch.nn.functional as F


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length."""
    return F.normalize(q, p=2, dim=-1)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions [..., 4]."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (inverse for unit quaternions)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def angular_error(
    q1: torch.Tensor, q2: torch.Tensor, degrees: bool = True
) -> torch.Tensor:
    """Geodesic distance on SO(3) between two quaternions."""
    dot = torch.abs((q1 * q2).sum(dim=-1)).clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(dot)

    if degrees:
        angle = angle * 180.0 / np.pi

    return angle


def ensure_positive_w(q: torch.Tensor) -> torch.Tensor:
    """Flip quaternion to positive w hemisphere (canonical form)."""
    w_negative = q[..., 0:1] < 0
    return torch.where(w_negative, -q, q)


def ensure_same_hemisphere(q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    """Flip q if needed to maintain positive dot product with q_ref (for continuity)."""
    dot = (q * q_ref).sum(dim=-1, keepdim=True)
    return torch.where(dot < 0, -q, q)


def delta_quaternion_to_angular_velocity(
    delta_q: torch.Tensor, dt: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Derive angular velocity from delta quaternion (inverse of Euler integration).
    For small rotations: delta_q ≈ [1, 0.5*ω*dt], so ω = 2*delta_q[1:4]/dt.
    Note: delta_q should have positive w for correct sign.
    """

    # Ensure we take the short path (w > 0 means rotation < 180°)
    delta_q = ensure_positive_w(delta_q)

    return 2.0 * delta_q[..., 1:4] / torch.clamp(dt, min=eps)
