"""
Compute reward from saved raw features (offline data).

The offline collector saves pre-step features at time t. Since the env computes
reward AFTER stepping (from the resulting state s_{t+1}), we reconstruct reward[t]
using features[t+1]. For the final transition (done=True), we use features[t]
itself since the episode ended in that state.

This module provides the SAME reward formula as mujoco_vec_env_stack.py
_compute_reward(), ensuring online/offline parity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


# ── Default config (matches mujoco_vec_env_stack.py dense reward) ──

DEFAULT_CONFIG = {
    "stages": {
        "approach": {
            "weight": 0.15,
            "type": "tanh_decay",
            "feature": "tcp_white_dist",
            "scale": 0.10,
            "condition": None,
        },
        "grasp": {
            "weight": 0.10,
            "type": "flat",
            "condition": "grasped",
        },
        "proximity": {
            "weight": 0.40,
            "type": "tanh_decay",
            "feature": "white_to_green_top_3d",
            "scale": 0.08,
            "condition": "grasped",
        },
        "success": {
            "weight": 0.35,
            "type": "flat",
            "condition": "success",  # white_z > green_z AND contact
        },
    },
    "clip_min": 0.0,
    "clip_max": 1.0,
}


def load_reward_config(config_path: str | Path | None = None) -> dict:
    """Load reward config from YAML file, or return default."""
    if config_path is None:
        return DEFAULT_CONFIG.copy()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Reward config not found: {path}")
    if yaml is None:
        raise ImportError("pyyaml required to load YAML config: pip install pyyaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Merge with defaults for missing keys
    merged = DEFAULT_CONFIG.copy()
    merged.update(cfg)
    return merged


def _eval_condition(condition: str | None, features: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate a condition string into a boolean mask.
    
    Supported conditions:
      - None → all True
      - "grasped" → features["grasped"]
      - "success" → (white_z > green_z) & white_green_contact
    """
    n = len(next(iter(features.values())))
    if condition is None:
        return np.ones(n, dtype=bool)
    if condition == "grasped":
        return features["grasped"].astype(bool)
    if condition == "success":
        return (features["white_z"] > features["green_z"]) & features["white_green_contact"].astype(bool)
    raise ValueError(f"Unknown condition: {condition}")


def _compute_stage(stage_cfg: dict, features: dict[str, np.ndarray]) -> np.ndarray:
    """Compute a single reward stage (vectorized over timesteps)."""
    n = len(next(iter(features.values())))
    stage_type = stage_cfg["type"]
    weight = stage_cfg["weight"]
    condition = stage_cfg.get("condition", None)
    
    mask = _eval_condition(condition, features)
    
    if stage_type == "flat":
        values = np.where(mask, weight, 0.0)
    elif stage_type == "tanh_decay":
        feature_key = stage_cfg["feature"]
        scale = stage_cfg["scale"]
        dist = features[feature_key].astype(np.float64)
        raw = (1.0 - np.tanh(dist / scale)) * weight
        values = np.where(mask, raw, 0.0)
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")
    
    return values.astype(np.float32)


def compute_reward_from_features(
    features: dict[str, np.ndarray],
    config: dict | None = None,
    use_next_step: bool = True,
) -> np.ndarray:
    """Compute dense reward from raw features.
    
    Args:
        features: dict of numpy arrays, each shape (T,).
            Required keys depend on config stages.
        config: reward config dict. If None, uses DEFAULT_CONFIG.
        use_next_step: If True, reward[t] is computed from features[t+1]
            (matching env behavior where reward is post-step).
            If False, reward[t] uses features[t] directly.
    
    Returns:
        reward: numpy array of shape (T,), clipped to [clip_min, clip_max].
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    n = len(next(iter(features.values())))
    
    if use_next_step and n > 1:
        # Shift features by 1: reward[t] = f(features[t+1])
        # Features are saved PRE-step, but reward is computed POST-step.
        shifted = {}
        done = features.get("done", np.zeros(n, dtype=bool))
        terminated = features.get("terminated", done)
        for k, v in features.items():
            if k in ("done", "terminated", "env_id", "step_idx"):
                shifted[k] = v
                continue
            # Shift left by 1, last element stays
            s = np.empty_like(v)
            s[:-1] = v[1:]
            s[-1] = v[-1]
            # At done boundaries, use current step (no next available)
            if done.any():
                done_mask = done.astype(bool)
                s[done_mask] = v[done_mask]
            shifted[k] = s
        # For terminal success steps: force success condition True
        # (features are pre-step so contact may not show, but episode terminated = success)
        if terminated.any():
            term_mask = terminated.astype(bool)
            # If the episode was successful (terminated=True means success in this env),
            # override contact/position to ensure success reward fires
            shifted["white_green_contact"] = shifted["white_green_contact"].copy()
            shifted["white_green_contact"][term_mask] = True
        feat_for_reward = shifted
    else:
        feat_for_reward = features
    
    # Validate required keys
    stages = config.get("stages", DEFAULT_CONFIG["stages"])
    
    # Compute total reward
    total = np.zeros(n, dtype=np.float32)
    for stage_name, stage_cfg in stages.items():
        total += _compute_stage(stage_cfg, feat_for_reward)
    
    # Clip
    clip_min = config.get("clip_min", 0.0)
    clip_max = config.get("clip_max", 1.0)
    total = np.clip(total, clip_min, clip_max)
    
    return total
