"""
Populate the offline replay buffer from CSV episode data.

Each episode folder contains:
  - ``isaac_deltaEEF_gripper.csv``  — 7D delta actions (dx,dy,dz,dyaw,dpitch,droll,gripper)
  - ``franka_joint_states.csv``     — joint positions/velocities (115 rows)
  - ``gripper_joint_states.csv``    — gripper state
  - ``object_pos.csv``              — initial cube pose (1 row)
  - ``sim_images_multi_view/``      — left_view.mp4, right_view.mp4, wrist_view.mp4

Video frames (133) > CSV rows (115).  We use only the first N frames where
N = number of CSV data rows, so actions and images are aligned 1:1.

Each transition stored in the replay buffer:
  obs = {
      "observation.state":       (state_dim,) float32    — joint pos/vel + gripper + ee pose
      "observation.base_action": (7,)         float32    — delta action (treated as base policy output)
      "observation.images.front": (3, 84, 84) uint8
      "observation.images.back":  (3, 84, 84) uint8
      "observation.images.wrist": (3, 84, 84) uint8
  }
  action = (7,) float32   — the GT delta action (= base + 0 residual)
  next.obs, next.done, next.reward
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import ReplayBuffer

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio  # type: ignore[no-redef]


# ── View mapping: video filename -> resfit obs key ──
_VIEW_MAP = {
    "right_view.mp4": "observation.images.front",
    "left_view.mp4": "observation.images.back",
    "wrist_view.mp4": "observation.images.wrist",
}


def _read_video_frames(video_path: Path, max_frames: int, target_size: tuple[int, int] = (84, 84)):
    """Read up to ``max_frames`` from an mp4 and return (N, 3, H, W) uint8 tensor."""
    import av

    frames = []
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        if len(frames) >= max_frames:
            break
        img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
        if t.shape[2] != target_size[0] or t.shape[3] != target_size[1]:
            t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
        frames.append(t.to(torch.uint8).squeeze(0))  # (3, H, W)
    container.close()
    return torch.stack(frames) if frames else torch.zeros((0, 3, target_size[0], target_size[1]), dtype=torch.uint8)


def _build_state(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    gripper_pos: float,
    ee_xyzrpy: np.ndarray,
) -> np.ndarray:
    """Build the 34D state vector matching ``IsaacLabVecEnvWrapper``'s layout.

    Layout (34D):
      [0:9]   joint_pos_scaled (7 arm + 2 finger, normalised to ~[-1,1])
      [9:18]  joint_vel_scaled
      [18:25] cogact_reference (zeros for offline — no live GR00T)
      [25:28] contact_obs (zeros — no physics)
      [28:34] ee_xyzrpy (6D)
    """
    # Approximate scaling: raw joint pos mapped with typical Franka limits
    # For offline buffer this doesn't need to be exact; the RL agent will
    # see similar distributions during online collection.
    arm_pos = joint_pos[:7]
    finger_pos = np.array([gripper_pos * 0.04, gripper_pos * 0.04], dtype=np.float32)
    full_pos = np.concatenate([arm_pos, finger_pos])  # (9,)

    arm_vel = joint_vel[:7]
    finger_vel = np.zeros(2, dtype=np.float32)
    full_vel = np.concatenate([arm_vel, finger_vel]) * 0.1  # dof_velocity_scale

    cogact_ref = np.zeros(7, dtype=np.float32)
    contact = np.zeros(3, dtype=np.float32)

    return np.concatenate([full_pos, full_vel, cogact_ref, contact, ee_xyzrpy]).astype(np.float32)


def populate_offline_buffer_from_csv(
    data_dir: str | Path,
    rb: ReplayBuffer,
    split_file: str | Path | None = None,
    split: str = "training",
    image_keys: list[str] | None = None,
    image_size: tuple[int, int] = (84, 84),
    max_episodes: int | None = None,
    device: str = "cpu",
) -> int:
    """Read CSV episodes and fill ``rb`` with transitions.

    Parameters
    ----------
    data_dir : path
        Root folder containing episode sub-folders (e.g. ``pickMushroom/``).
    rb : ReplayBuffer
        Target replay buffer.
    split_file : path, optional
        ``train_test_split.txt`` — if given, only episodes matching ``split`` are used.
    split : str
        ``"training"`` or ``"testing"``.
    image_keys : list[str]
        Resfit obs keys for images. Default: front/back/wrist.
    image_size : tuple
        Target (H, W) for downscaling.
    max_episodes : int, optional
        Cap on number of episodes.
    device : str
        Torch device for tensors stored in buffer.

    Returns
    -------
    int — total transitions added.
    """
    data_dir = Path(data_dir)
    if image_keys is None:
        image_keys = list(_VIEW_MAP.values())

    # ── Resolve episode list ──
    if split_file is not None:
        split_path = Path(split_file)
        ep_names = []
        with open(split_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1].lower() == split:
                    ep_names.append(parts[0])
    else:
        ep_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())

    if max_episodes is not None:
        ep_names = ep_names[:max_episodes]

    total_transitions = 0

    for ep_idx, ep_name in enumerate(ep_names):
        ep_dir = data_dir / ep_name
        delta_csv = ep_dir / "isaac_deltaEEF_gripper.csv"
        joint_csv = ep_dir / "franka_joint_states.csv"
        gripper_csv = ep_dir / "gripper_joint_states.csv"
        abs_eef_csv = ep_dir / "isaac_absEEF_gripper_convert314.csv"
        img_dir = ep_dir / "sim_images_multi_view"

        if not delta_csv.exists() or not joint_csv.exists():
            print(f"[offline] Skipping {ep_name}: missing CSV")
            continue

        # ── Load CSVs ──
        delta_df = pd.read_csv(delta_csv)
        joint_df = pd.read_csv(joint_csv)
        gripper_df = pd.read_csv(gripper_csv)

        # N = min of all CSV row counts (use first N frames from video)
        N = min(len(delta_df), len(joint_df), len(gripper_df))

        # Joint positions: columns 10:17 (position_fr3_joint1..7)
        # Skip first row (often duplicate/header artifact in some episodes)
        js_raw = joint_df.values[1:, 10:17].astype(np.float32)  # (N-1, 7)
        js_vel = joint_df.values[1:, 17:24].astype(np.float32)  # velocities
        gs_raw = gripper_df.values[1:, 4:5].astype(np.float32)  # gripper pos
        delta_raw = delta_df.values[:N].astype(np.float32)       # (N, 7)

        # Align: use min length after skip
        T = min(len(js_raw), len(delta_raw))
        js_raw = js_raw[:T]
        js_vel = js_vel[:T]
        gs_raw = gs_raw[:T]
        delta_raw = delta_raw[:T]

        # Absolute EE pose for state (if available)
        if abs_eef_csv.exists():
            abs_eef = pd.read_csv(abs_eef_csv).values.astype(np.float32)
        else:
            abs_eef = np.zeros((T, 7), dtype=np.float32)
        # Clip to common length (abs_eef may be shorter than other CSVs)
        T = min(T, len(abs_eef))
        js_raw = js_raw[:T]
        js_vel = js_vel[:T]
        gs_raw = gs_raw[:T]
        delta_raw = delta_raw[:T]
        abs_eef = abs_eef[:T]

        # ── Load video frames (first T frames only) ──
        view_frames = {}
        for vid_name, obs_key in _VIEW_MAP.items():
            vid_path = img_dir / vid_name
            if vid_path.exists():
                view_frames[obs_key] = _read_video_frames(vid_path, max_frames=T, target_size=image_size)
            else:
                view_frames[obs_key] = torch.zeros((T, 3, image_size[0], image_size[1]), dtype=torch.uint8)

        # ── Build transitions (t, t+1) ──
        for t in range(T - 1):
            gripper_frac = float(np.clip(gs_raw[t, 0], 0.0, 1.0))
            ee_xyzrpy = abs_eef[t, :6] if abs_eef.shape[1] >= 6 else np.zeros(6, dtype=np.float32)

            state_t = _build_state(js_raw[t], js_vel[t], gripper_frac, ee_xyzrpy)

            gripper_frac_next = float(np.clip(gs_raw[t + 1, 0], 0.0, 1.0))
            ee_xyzrpy_next = abs_eef[t + 1, :6] if abs_eef.shape[1] >= 6 else np.zeros(6, dtype=np.float32)
            state_next = _build_state(js_raw[t + 1], js_vel[t + 1], gripper_frac_next, ee_xyzrpy_next)

            # Action = GT delta (same as what GR00T would output as base action)
            action_t = delta_raw[t]  # (7,): dx,dy,dz,dyaw,dpitch,droll,gripper

            # For residual RL: base_action = GT delta, residual = 0
            # So the stored action IS the combined action (base + 0)
            is_last = (t == T - 2)
            reward = 1.0 if is_last else 0.0

            curr_obs = {
                "observation.state": torch.tensor(state_t),
                "observation.base_action": torch.tensor(action_t),
            }
            next_obs = {
                "observation.state": torch.tensor(state_next),
                "observation.base_action": torch.tensor(
                    delta_raw[t + 1] if t + 1 < T else delta_raw[t]
                ),
            }

            for obs_key in image_keys:
                curr_obs[obs_key] = view_frames[obs_key][t]        # (3, H, W) uint8
                next_obs[obs_key] = view_frames[obs_key][t + 1]

            td = TensorDict(
                {
                    "obs": TensorDict(curr_obs, batch_size=[]),
                    "action": torch.tensor(action_t, dtype=torch.float32),
                    "next": TensorDict(
                        {
                            "obs": TensorDict(next_obs, batch_size=[]),
                            "done": torch.tensor(is_last, dtype=torch.bool),
                            "reward": torch.tensor(reward, dtype=torch.float32),
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
            print(f"[offline] {ep_idx + 1}/{len(ep_names)} episodes, {total_transitions} transitions")

    print(f"[offline] Done: {total_transitions} transitions from {len(ep_names)} episodes")
    return total_transitions
