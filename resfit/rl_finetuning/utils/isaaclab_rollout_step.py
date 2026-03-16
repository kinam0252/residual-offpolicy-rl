from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _to_chw_uint8_84(frame_hwc: np.ndarray) -> np.ndarray:
    frame = torch.from_numpy(frame_hwc)
    if frame.dtype != torch.uint8:
        if frame.max() <= 1.0:
            frame = (frame.clamp(0, 1) * 255).to(torch.uint8)
        else:
            frame = frame.clamp(0, 255).to(torch.uint8)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    frame = frame.permute(2, 0, 1).contiguous().float().unsqueeze(0)
    frame = F.interpolate(frame, size=(84, 84), mode="bilinear", align_corners=False)
    return frame.squeeze(0).to(torch.uint8).cpu().numpy()


def _obs_front_to_chw_uint8(obs: dict[str, Any], env_idx: int) -> np.ndarray | None:
    front = obs.get("observation.images.front") if isinstance(obs, dict) else None
    if front is None:
        return None

    if isinstance(front, torch.Tensor):
        arr = front[env_idx].detach().cpu().numpy()
    else:
        arr = np.asarray(front[env_idx])

    if arr.ndim != 3:
        return None

    if arr.shape[0] == 3:
        chw = arr
    elif arr.shape[-1] in (3, 4):
        chw = _to_chw_uint8_84(arr)
    else:
        return None

    if chw.dtype != np.uint8:
        if np.max(chw) <= 1.0:
            chw = np.clip(chw, 0.0, 1.0) * 255.0
        chw = np.clip(chw, 0.0, 255.0).astype(np.uint8)

    return chw


def rollout_step_with_export_path(
    *,
    env,
    obs: dict[str, Any],
    residual_action: torch.Tensor,
    env_idx: int = 0,
) -> dict[str, Any]:
    front_chw = _obs_front_to_chw_uint8(obs, env_idx)
    if front_chw is None:
        raise RuntimeError("rollout_step_with_export_path failed to capture observation.images.front frame")
    front_hwc = np.transpose(front_chw, (1, 2, 0)).astype(np.uint8)
    source = "obs"

    next_obs, reward, terminated, truncated, info = env.step(residual_action)
    combined = info.get("scaled_action", residual_action)

    return {
        "next_obs": next_obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
        "combined_action": combined,
        "front_image_chw": front_chw,
        "front_image_hwc": front_hwc,
        "front_image_source": source,
    }
