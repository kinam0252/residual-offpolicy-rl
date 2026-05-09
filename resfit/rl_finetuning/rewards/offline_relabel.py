"""
Per-task offline reward relabeling functions.

Each function takes a dict of numpy arrays (from an .npz file) and a
reward_type string, and returns a numpy array of per-step rewards.
Returns None if required features are missing (caller falls back to stored reward).
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def relabel_cup(data: dict, reward_type: str) -> Optional[np.ndarray]:
    """Recompute Cup reward from stored features."""
    required = ["tcp_cup_dist", "grasped", "uprightness", "cup_vel", "cup_z"]
    if any(k not in data for k in required):
        return None

    tcp_cup_dist = data["tcp_cup_dist"].astype(np.float32)
    grasped = data["grasped"].astype(np.float32)
    uprightness = data["uprightness"].astype(np.float32)
    cup_vel = data["cup_vel"].astype(np.float32)
    cup_z = data["cup_z"].astype(np.float32)

    approach_r = 1.0 - np.tanh(tcp_cup_dist / 0.10)
    grasp_r = grasped
    upright_r = np.clip(uprightness, 0.0, 1.0) * grasped
    is_success = (
        (uprightness > 0.82) & (cup_vel < 0.5) & (cup_z > 0.0) & (cup_z < 0.15)
    ).astype(np.float32)

    if reward_type == "dense_v2":
        reward = 0.10 * approach_r + 0.10 * grasp_r + 0.30 * upright_r + 0.50 * is_success
    elif reward_type == "dense_v3":
        reward = 0.25 * approach_r + 0.15 * grasp_r + 0.25 * upright_r + 0.35 * is_success
    elif reward_type in ("dense_bonus", "dense_equal_bonus"):
        w_a, w_g, w_u = (0.33, 0.34, 0.33) if "equal" in reward_type else (0.20, 0.15, 0.65)
        reward = w_a * approach_r + w_g * grasp_r + w_u * upright_r + 2.0 * is_success
    else:  # dense / dense_equal
        w_a, w_g, w_u = (0.33, 0.34, 0.33) if "equal" in reward_type else (0.20, 0.15, 0.65)
        reward = w_a * approach_r + w_g * grasp_r + w_u * upright_r

    return reward


def relabel_pnp(data: dict, reward_type: str) -> Optional[np.ndarray]:
    """Recompute PnP reward from stored features."""
    required = ["tcp_cube_dist", "cube_bowl_xy_dist", "grasped", "cube_z", "alignment"]
    if any(k not in data for k in required):
        return None

    tcp_cube_dist = data["tcp_cube_dist"].astype(np.float32)
    cube_bowl_xy = data["cube_bowl_xy_dist"].astype(np.float32)
    grasped = data["grasped"].astype(np.float32)
    cube_z = data["cube_z"].astype(np.float32)
    alignment = data["alignment"].astype(np.float32)

    approach_r = 1.0 - np.tanh(tcp_cube_dist / 0.05)
    grasp_r = grasped
    lift_r = np.clip((cube_z - 0.02) / 0.08, 0.0, 1.0) * grasped
    transport_r = (1.0 - np.tanh(cube_bowl_xy / 0.05)) * grasped
    place_r = alignment * grasped

    if reward_type == "dense_v2":
        reward = 0.10 * approach_r + 0.10 * grasp_r + 0.20 * lift_r + 0.30 * transport_r + 0.30 * place_r
    elif reward_type == "dense_v3":
        is_above = (cube_bowl_xy < 0.03).astype(np.float32)
        reward = 0.10 * approach_r + 0.10 * grasp_r + 0.15 * lift_r + 0.25 * transport_r + 0.20 * place_r + 0.20 * is_above
    else:  # dense / dense_clipped
        reward = 0.15 * approach_r + 0.15 * grasp_r + 0.20 * lift_r + 0.25 * transport_r + 0.25 * place_r

    return reward


def relabel_lift(data: dict, reward_type: str) -> Optional[np.ndarray]:
    """Recompute Lift reward from stored features."""
    required = ["tcp_cube_dist", "grasped", "lift_delta"]
    if any(k not in data for k in required):
        return None

    tcp_cube_dist = data["tcp_cube_dist"].astype(np.float32)
    grasped = data["grasped"].astype(np.float32)
    lift_delta = data["lift_delta"].astype(np.float32)

    approach_r = 1.0 - np.tanh(tcp_cube_dist / 0.10)
    grasp_r = grasped
    success_threshold = 0.03
    height_r = np.clip(lift_delta / success_threshold, 0.0, 1.0) * grasped

    if reward_type == "dense_clipped":
        height_bonus = (lift_delta > 0.005).astype(np.float32) * np.tanh(lift_delta / 0.1) * 0.5 * grasped
        success_bonus = (lift_delta >= success_threshold).astype(np.float32) * 1.0 * grasped
        reward = 0.30 * approach_r + 0.20 * grasp_r + height_bonus + success_bonus
    else:  # dense
        reward = 0.30 * approach_r + 0.30 * grasp_r + 0.40 * height_r

    return reward


# Registry: task name → relabeling function
RELABEL_FUNCTIONS = {
    "cup": relabel_cup,
    "pnp": relabel_pnp,
    "lift": relabel_lift,
}


def relabel_offline_reward(
    data: dict, task: str, reward_type: str
) -> Optional[np.ndarray]:
    """Dispatch to the appropriate task-specific relabeling function.

    Returns numpy reward array, or None if the task has no relabeler
    or required features are missing.
    """
    fn = RELABEL_FUNCTIONS.get(task)
    if fn is None:
        return None
    return fn(data, reward_type)
