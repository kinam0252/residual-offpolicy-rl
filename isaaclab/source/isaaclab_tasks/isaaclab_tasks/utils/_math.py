from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional
from typing import Literal

# customized functions for batched quaternion operations
def quat_slerp_batch(q1: torch.Tensor, q2: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Batched SLERP between two sets of quaternions in (w, x, y, z) format.

    Args:
        q1: Tensor of shape (N, 4)
        q2: Tensor of shape (N, 4)
        tau: Scalar interpolation factor between 0 and 1

    Returns:
        Interpolated quaternions of shape (N, 4)
    """
    # Normalize to ensure unit quaternions
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)

    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)  # (N, 1)

    # Flip sign to ensure shortest path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)

    eps = torch.finfo(q1.dtype).eps
    angle = torch.acos(torch.clamp(dot, -1.0 + eps, 1.0 - eps))  # (N, 1)

    sin_angle = torch.sin(angle)
    factor1 = torch.sin((1.0 - tau) * angle) / sin_angle
    factor2 = torch.sin(tau * angle) / sin_angle

    # Avoid division by zero for nearly identical quaternions
    factor1 = torch.where(sin_angle < eps, 1.0 - tau, factor1)
    factor2 = torch.where(sin_angle < eps, tau, factor2)

    result = factor1 * q1 + factor2 * q2
    return torch.nn.functional.normalize(result, dim=-1)
