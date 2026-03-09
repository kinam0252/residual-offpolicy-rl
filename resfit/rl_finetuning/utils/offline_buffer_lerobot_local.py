from __future__ import annotations

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


def _to_state34(state_vec: np.ndarray) -> np.ndarray:
    out = np.zeros(34, dtype=np.float32)
    n = min(len(state_vec), 34)
    out[:n] = state_vec[:n]
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

        for t in range(T - 1):
            state_t = _to_state34(states[t])
            state_next = _to_state34(states[t + 1])
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

            td = TensorDict(
                {
                    "obs": TensorDict(curr_obs, batch_size=[]),
                    "action": torch.tensor(action_t, dtype=torch.float32),
                    "next": TensorDict(
                        {
                            "obs": TensorDict(next_obs, batch_size=[]),
                            "done": torch.tensor(t == (T - 2), dtype=torch.bool),
                            "reward": torch.tensor(1.0 if t == (T - 2) else 0.0, dtype=torch.float32),
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
