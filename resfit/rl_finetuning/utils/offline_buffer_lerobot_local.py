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
    lowdim_keys: list[str] | None = None,
    object_state_mode: str = "raw",
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

    # Load depth normalization config if depth keys are requested
    _depth_norm = None
    _depth_data_dir = root  # depth_front/ and depth_wrist/ next to data/
    if any(k.startswith("observation.depth") for k in image_keys):
        _depth_norm_path = root / ".." / ".." / ".." / "residual-offpolicy-rl" / "depth_min_max" / "depth_normalization.json"
        # Try common locations
        for _dnp in [
            root / "depth_normalization.json",
            Path("/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/depth_min_max/depth_normalization.json"),
        ]:
            if _dnp.exists():
                import json as _json_mod
                _depth_norm = _json_mod.loads(_dnp.read_text())
                print(f"[offline-buffer] Depth normalization loaded from {_dnp}")
                break
        if _depth_norm is None:
            _depth_norm = {"front": {"min": 0.3, "max": 1.8}, "wrist": {"min": 0.03, "max": 0.4}}
            print(f"[offline-buffer] Using default depth normalization")
        _depth_debug_logged = False

    episode_files = sorted(data_chunk.glob("episode_*.parquet"))
    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]

    # Asymmetric mode validation: check first episode has both depth AND cube_pos
    _first_df = pd.read_parquet(episode_files[0]) if episode_files else None
    _want_depth = any(k.startswith("observation.depth") for k in image_keys) if image_keys else False
    _want_obj = lowdim_keys is not None and "observation.object_state" in lowdim_keys
    if _want_depth and _want_obj and _first_df is not None:
        _has_cube = "cube_pos" in _first_df.columns and "cube_quat_wxyz" in _first_df.columns
        print(f"[offline-buffer] ASYMMETRIC check: depth_keys={[k for k in image_keys if 'depth' in k]} "
              f"obj_state_wanted={_want_obj} parquet_has_cube_pos={_has_cube}")
        if not _has_cube:
            print(f"[offline-buffer] WARNING: depth+state requested but parquet missing cube_pos/cube_quat_wxyz! "
                  f"Columns: {sorted(_first_df.columns.tolist())}")

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
        if image_keys:  # skip video loading if no image keys (state-only mode)
            for view_dir, obs_key in _VIDEO_MAP.items():
                video_path = video_chunk / view_dir / f"{episode_name}.mp4"
                if video_path.exists():
                    view_frames[obs_key] = _read_video_frames(video_path, max_frames=T, target_size=image_size)
                else:
                    view_frames[obs_key] = torch.zeros((T, 3, image_size[0], image_size[1]), dtype=torch.uint8)

        if view_frames:
            T = min(T, *(len(v) for v in view_frames.values()))
        if T < 2:
            continue

        states = states[:T]
        actions = actions[:T]

        # Extract contact force if available in parquet
        has_contact_force = "contact_force" in df.columns
        contact_forces = df["contact_force"].to_numpy().astype(np.float32) if has_contact_force else np.zeros(T, dtype=np.float32)

        # Extract VLM latent if available in parquet (2048D)
        has_vlm_latent = "vlm_latent" in df.columns
        if has_vlm_latent:
            vlm_latents = np.stack(df["vlm_latent"].to_numpy()).astype(np.float32)  # (T, 2048)
        else:
            vlm_latents = None  # will use zeros

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

            # VLM latent (2048D raw — projected in agent)
            _want_vlm = lowdim_keys is None or "observation.vlm_latent" in lowdim_keys
            if _want_vlm:
                if has_vlm_latent:
                    curr_obs["observation.vlm_latent"] = torch.tensor(vlm_latents[t], dtype=torch.float32)
                    next_obs["observation.vlm_latent"] = torch.tensor(vlm_latents[min(t+1, T-1)], dtype=torch.float32)
                else:
                    curr_obs["observation.vlm_latent"] = torch.zeros(2048, dtype=torch.float32)
                    next_obs["observation.vlm_latent"] = torch.zeros(2048, dtype=torch.float32)

            # Object state: mode-dependent from parquet (cube_pos + cube_quat_wxyz)
            _want_obj_state = lowdim_keys is not None and "observation.object_state" in lowdim_keys
            if _want_obj_state:
                _has_cube = "cube_pos" in df.columns and "cube_quat_wxyz" in df.columns
                if _has_cube:
                    _ck = f"_obj_raw_{id(df)}"
                    if not hasattr(populate_offline_buffer_from_lerobot_local, '_obj_cache_id') or \
                       populate_offline_buffer_from_lerobot_local._obj_cache_id != _ck:
                        populate_offline_buffer_from_lerobot_local._obj_cp = np.stack(df["cube_pos"].to_numpy()).astype(np.float32)
                        populate_offline_buffer_from_lerobot_local._obj_cq = np.stack(df["cube_quat_wxyz"].to_numpy()).astype(np.float32)
                        populate_offline_buffer_from_lerobot_local._obj_cache_id = _ck
                    _cp = populate_offline_buffer_from_lerobot_local._obj_cp
                    _cq = populate_offline_buffer_from_lerobot_local._obj_cq
                    _tn = min(t + 1, T - 1)

                    if object_state_mode == "raw":
                        os_t = np.concatenate([_cp[t], _cq[t]])
                        os_n = np.concatenate([_cp[_tn], _cq[_tn]])
                    elif object_state_mode == "relative":
                        rel_t = _cp[t] - state_t[:3]
                        rel_n = _cp[_tn] - state_next[:3]
                        ct_t = np.array([1.0 if contact_forces[t] > 0.1 else 0.0], dtype=np.float32)
                        ct_n = np.array([1.0 if contact_forces[_tn] > 0.1 else 0.0], dtype=np.float32)
                        os_t = np.concatenate([rel_t, ct_t])
                        os_n = np.concatenate([rel_n, ct_n])
                    elif object_state_mode == "full":
                        rel_t = _cp[t] - state_t[:3]
                        rel_n = _cp[_tn] - state_next[:3]
                        ct_t = np.array([1.0 if contact_forces[t] > 0.1 else 0.0], dtype=np.float32)
                        ct_n = np.array([1.0 if contact_forces[_tn] > 0.1 else 0.0], dtype=np.float32)
                        os_t = np.concatenate([rel_t, _cq[t], ct_t])
                        os_n = np.concatenate([rel_n, _cq[_tn], ct_n])
                    else:
                        os_t = np.concatenate([_cp[t], _cq[t]])
                        os_n = np.concatenate([_cp[_tn], _cq[_tn]])

                    curr_obs["observation.object_state"] = torch.tensor(os_t, dtype=torch.float32)
                    next_obs["observation.object_state"] = torch.tensor(os_n, dtype=torch.float32)
                    if ep_idx == 0 and t == 0:
                        print(f"[offline-lerobot] Object state mode={object_state_mode}: "
                              f"dim={len(os_t)} sample={os_t}")
                else:
                    _dim = {"raw": 7, "relative": 4, "full": 8}.get(object_state_mode, 7)
                    curr_obs["observation.object_state"] = torch.zeros(_dim, dtype=torch.float32)
                    next_obs["observation.object_state"] = torch.zeros(_dim, dtype=torch.float32)
                    if ep_idx == 0 and t == 0:
                        print(f"[offline-lerobot] WARNING: no cube_pos/cube_quat in parquet")

            for obs_key in image_keys:
                if obs_key.startswith("observation.depth."):
                    # Load depth from .npz file if available
                    cam_name = obs_key.split(".")[-1]  # "front" or "wrist"
                    depth_npz_path = root / f"depth_{cam_name}" / f"{episode_name}.npz"
                    if depth_npz_path.exists() and f"_depth_{cam_name}" not in view_frames:
                        _raw = np.load(str(depth_npz_path))["frames"]  # (T, 84, 84) float32 meters
                        # Normalize using fixed min/max → [0, 1]
                        if _depth_norm and cam_name in _depth_norm:
                            _dmin = _depth_norm[cam_name]["min"]
                            _dmax = _depth_norm[cam_name]["max"]
                        else:
                            _dmin, _dmax = 0.1, 2.0
                        _raw = np.clip(_raw, _dmin, _dmax)
                        _raw = (_raw - _dmin) / max(_dmax - _dmin, 1e-6)
                        _raw = np.nan_to_num(_raw, nan=0.0, posinf=1.0, neginf=0.0)
                        # Store as (T, 1, 84, 84) float32
                        view_frames[f"_depth_{cam_name}"] = torch.tensor(
                            _raw[:, None, :, :], dtype=torch.float32)
                    # Get frame from cache
                    _dcache_key = f"_depth_{cam_name}"
                    if _dcache_key in view_frames and t < view_frames[_dcache_key].shape[0]:
                        curr_obs[obs_key] = view_frames[_dcache_key][t]  # (1, 84, 84)
                        next_obs[obs_key] = view_frames[_dcache_key][min(t+1, T-1)]
                        # Debug log first load
                        if not _depth_debug_logged:
                            _v = view_frames[_dcache_key][t]
                            print(f"[offline-depth] Loaded {obs_key}: shape={_v.shape} "
                                  f"dtype={_v.dtype} range=[{_v.min().item():.4f}, {_v.max().item():.4f}] "
                                  f"from {depth_npz_path}")
                            _depth_debug_logged = True
                    else:
                        curr_obs[obs_key] = torch.zeros(1, 84, 84, dtype=torch.float32)
                        next_obs[obs_key] = torch.zeros(1, 84, 84, dtype=torch.float32)
                elif obs_key in view_frames:
                    curr_obs[obs_key] = view_frames[obs_key][t]
                    next_obs[obs_key] = view_frames[obs_key][t + 1]
                else:
                    raise KeyError(f"Offline data missing image key '{obs_key}'. "
                                   f"Available: {list(view_frames.keys())}")

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
