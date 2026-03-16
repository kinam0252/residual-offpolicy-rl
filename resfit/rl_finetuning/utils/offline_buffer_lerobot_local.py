from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer

from resfit.rl_finetuning.utils.offline_buffer_csv import _read_video_frames


_VIDEO_MAP = {
    "right_image": "observation.images.front",
    "image": "observation.images.back",
    "wrist_image": "observation.images.wrist",
}

# ── Canonical reward formula signature ──
# Used to verify offline data reward matches online reward.
REWARD_FORMULA_SIGNATURE = {
    "type": "shaped_proximity_4stage",
    "distance_std": 0.1,
    "distance_weight": 1.0,
    "grasp_finger_threshold": 0.05,
    "grasp_gripper_threshold": 0.03,
    "grasp_weight": 2.0,
    "height_minimal": 0.005,
    "height_std": 0.1,
    "height_weight": 100.0,
}


def get_reward_formula_signature(success_threshold: float = 0.005) -> dict:
    """Return the canonical reward formula dict with the given success threshold."""
    sig = dict(REWARD_FORMULA_SIGNATURE)
    sig["success_threshold"] = success_threshold
    sig["success_weight"] = 100.0
    return sig


def validate_offline_reward_formula(data_dir: str | Path, success_threshold: float = 0.005) -> None:
    """Check that the offline dataset's reward formula matches the online one.

    Reads the first parquet file and verifies that shaped reward columns exist.
    If a reward_config_and_stats.json is present, also checks parameters.
    Raises ValueError on mismatch.
    """
    root = Path(data_dir)
    data_chunk = root / "data" / "chunk-000"
    parquet_files = sorted(data_chunk.glob("episode_*.parquet")) if data_chunk.exists() else []
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_chunk}")

    # 1. Check that shaped reward columns exist in parquet
    df = pd.read_parquet(parquet_files[0])
    required_cols = {"reward_total", "reward_distance", "reward_grasp", "reward_height", "reward_success"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Offline dataset is missing shaped reward columns: {missing}.\n"
            f"Found columns: {sorted(df.columns)}\n"
            f"The offline data may have been generated with sparse reward (0/1). "
            f"Please regenerate with the updated replay_all_worker.py that saves all reward components."
        )

    # 2. If metadata JSON exists, cross-check reward config
    meta_path = root / "reward_config_and_stats.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if "reward_config" in meta:
            rc = meta["reward_config"]
            online_sig = get_reward_formula_signature(success_threshold)
            mismatches = []
            for k, v in online_sig.items():
                if k in rc and abs(float(rc[k]) - float(v)) > 1e-6:
                    mismatches.append(f"  {k}: offline={rc[k]} vs online={v}")
            if mismatches:
                raise ValueError(
                    f"Reward formula mismatch between offline data and online config:\n"
                    + "\n".join(mismatches)
                )
            print(f"[offline-reward-check] Reward formula validated against {meta_path}")

    # 3. Sanity: for dense reward, check not all zero. Sparse reward (0/1) can have mostly zeros.
    reward_col = df["reward_total"].astype(float)
    if reward_col.max() < 1e-6:
        print(f"[offline-reward-check] WARNING: reward_total is all zero in {parquet_files[0].name} (may be sparse with no success)")
    else:
        print(f"[offline-reward-check] Reward range: [{reward_col.min():.2f}, {reward_col.max():.2f}]")

    print(f"[offline-reward-check] Shaped reward columns verified in {len(parquet_files)} episodes")


def _to_state34(state_vec: np.ndarray, contact_force: float = 0.0) -> np.ndarray:
    """Match online env's state dim: EEF pos(3) + quat(4) + gripper(2) + contact_force(1) = 10D."""
    out = np.zeros(10, dtype=np.float32)
    n = min(len(state_vec), 9)  # first 9D from original state
    out[:n] = state_vec[:n]
    out[9] = contact_force  # dim 9 = contact force magnitude
    return out


def populate_offline_buffer_from_lerobot_local(
    data_dir: str | Path,
    rb: ReplayBuffer,
    image_keys: list[str] | None = None,
    image_size: tuple[int, int] = (84, 84),
    max_episodes: int | None = None,
) -> int:
    """Populate replay buffer from local LeRobot-style dataset folder.

    Expected structure:
      data/chunk-000/episode_XXXXXX.parquet
      videos/chunk-000/{image,right_image,wrist_image}/episode_XXXXXX.mp4
    """
    root = Path(data_dir)
    data_chunk = root / "data" / "chunk-000"
    video_chunk = root / "videos" / "chunk-000"

    if not data_chunk.exists():
        raise FileNotFoundError(f"LeRobot data chunk not found: {data_chunk}")

    if image_keys is None:
        image_keys = list(_VIDEO_MAP.values())

    episode_files = sorted(data_chunk.glob("episode_*.parquet"))
    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]

    total_transitions = 0

    for ep_idx, parquet_path in enumerate(episode_files):
        episode_name = parquet_path.stem
        df = pd.read_parquet(parquet_path)
        if len(df) < 2:
            continue

        if "state" not in df.columns or "actions" not in df.columns:
            print(f"[offline-lerobot] Skip {episode_name}: missing state/actions columns")
            continue

        states = np.stack(df["state"].to_numpy()).astype(np.float32)
        actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
        T = min(len(states), len(actions))
        if T < 2:
            continue

        # Detect shaped reward columns
        has_shaped_reward = "reward_total" in df.columns
        if has_shaped_reward:
            rewards_arr = df["reward_total"].to_numpy().astype(np.float32)
        else:
            rewards_arr = None

        view_frames: dict[str, torch.Tensor] = {}
        for view_dir, obs_key in _VIDEO_MAP.items():
            video_path = video_chunk / view_dir / f"{episode_name}.mp4"
            if video_path.exists():
                view_frames[obs_key] = _read_video_frames(video_path, max_frames=T, target_size=image_size)
            else:
                view_frames[obs_key] = torch.zeros((T, 3, image_size[0], image_size[1]), dtype=torch.uint8)

        T = min(T, *(len(v) for v in view_frames.values()))
        if T < 2:
            continue

        states = states[:T]
        actions = actions[:T]

        # Extract contact force if available in parquet
        has_contact_force = "contact_force" in df.columns
        contact_forces = df["contact_force"].to_numpy().astype(np.float32) if has_contact_force else np.zeros(T, dtype=np.float32)

        for t in range(T - 1):
            state_t = _to_state34(states[t], contact_force=contact_forces[t])
            state_next = _to_state34(states[t + 1], contact_force=contact_forces[min(t + 1, T - 1)])
            action_t = actions[t][:7].astype(np.float32)
            next_action = actions[t + 1][:7].astype(np.float32)

            curr_obs = {
                "observation.state": torch.tensor(state_t, dtype=torch.float32),
                "observation.base_action": torch.tensor(action_t, dtype=torch.float32),
            }
            next_obs = {
                "observation.state": torch.tensor(state_next, dtype=torch.float32),
                "observation.base_action": torch.tensor(next_action, dtype=torch.float32),
            }

            for obs_key in image_keys:
                curr_obs[obs_key] = view_frames[obs_key][t]
                next_obs[obs_key] = view_frames[obs_key][t + 1]

            # Use shaped reward from parquet if available, else fall back to sparse
            if has_shaped_reward:
                step_reward = float(rewards_arr[t])  # Use reward as-is (sparse: 0/1)
            else:
                step_reward = 1.0 if t == (T - 2) else 0.0

            td = TensorDict(
                {
                    "obs": TensorDict(curr_obs, batch_size=[]),
                    "action": torch.tensor(action_t, dtype=torch.float32),
                    "next": TensorDict(
                        {
                            "obs": TensorDict(next_obs, batch_size=[]),
                            "done": torch.tensor(t == (T - 2), dtype=torch.bool),
                            "reward": torch.tensor(step_reward, dtype=torch.float32),
                        },
                        batch_size=[],
                    ),
                    "_priority": torch.tensor(10.0, dtype=torch.float32),
                },
                batch_size=[],
            ).unsqueeze(0)

            rb.add(td)
            total_transitions += 1

        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            print(f"[offline-lerobot] {ep_idx + 1}/{len(episode_files)} episodes, {total_transitions} transitions")

    print(f"[offline-lerobot] Done: {total_transitions} transitions from {len(episode_files)} episodes")
    return total_transitions
