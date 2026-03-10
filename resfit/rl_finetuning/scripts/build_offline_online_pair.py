from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd


def _frame_stats(name: str, frames: np.ndarray) -> tuple[int, int, float, int]:
    n = int(frames.shape[0])
    maxv = int(frames.max()) if n > 0 else 0
    meanv = float(frames.mean()) if n > 0 else 0.0
    zero = int((frames.reshape(n, -1).max(axis=1) == 0).sum()) if n > 0 else 0
    print(
        f"[{name}] frames={n} shape={frames.shape} dtype={frames.dtype} max={maxv} mean={meanv:.3f} zero_frames={zero}",
        flush=True,
    )
    return n, maxv, meanv, zero


def _to_state34(state_vec: np.ndarray) -> np.ndarray:
    out = np.zeros(34, dtype=np.float32)
    n = min(len(state_vec), 34)
    out[:n] = state_vec[:n]
    return out


def _read_video_frames_chw(video_path: Path, max_frames: int) -> np.ndarray:
    reader = imageio.get_reader(video_path.as_posix())
    frames: list[np.ndarray] = []
    try:
        for idx, frame in enumerate(reader):
            if idx >= max_frames:
                break
            rgb = np.asarray(frame, dtype=np.uint8)
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            if rgb.shape[0] != 84 or rgb.shape[1] != 84:
                # keep dependency-light nearest resize
                y_idx = np.linspace(0, rgb.shape[0] - 1, 84).astype(np.int32)
                x_idx = np.linspace(0, rgb.shape[1] - 1, 84).astype(np.int32)
                rgb = rgb[np.ix_(y_idx, x_idx)]
            frames.append(np.transpose(rgb, (2, 0, 1)))
    finally:
        reader.close()

    if not frames:
        return np.zeros((0, 3, 84, 84), dtype=np.uint8)
    return np.asarray(frames, dtype=np.uint8)


def _load_offline_from_lerobot(data_dir: Path, episode_index: int) -> dict[str, np.ndarray]:
    data_chunk = data_dir / "data" / "chunk-000"
    right_vid_dir = data_dir / "videos" / "chunk-000" / "right_image"
    if not data_chunk.exists():
        raise FileNotFoundError(f"LeRobot data chunk not found: {data_chunk}")

    episode_files = sorted(data_chunk.glob("episode_*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No parquet episodes found under: {data_chunk}")
    if episode_index < 0 or episode_index >= len(episode_files):
        raise IndexError(f"episode_index={episode_index} out of range [0, {len(episode_files)-1}]")

    parquet_path = episode_files[episode_index]
    episode_name = parquet_path.stem
    df = pd.read_parquet(parquet_path)

    if "state" not in df.columns or "actions" not in df.columns:
        raise KeyError(f"Episode {episode_name} missing required columns: state/actions")

    states = np.stack(df["state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
    T = min(len(states), len(actions))
    if T < 2:
        raise RuntimeError(f"Episode {episode_name} too short: {T}")

    video_path = right_vid_dir / f"{episode_name}.mp4"
    if video_path.exists():
        front = _read_video_frames_chw(video_path, max_frames=T)
    else:
        front = np.zeros((T, 3, 84, 84), dtype=np.uint8)

    T = min(T, len(front))
    if T < 2:
        raise RuntimeError(f"Episode {episode_name} has insufficient aligned frames: {T}")

    states = states[:T]
    actions = actions[:T]
    front = front[:T]

    n = T - 1
    reward = np.zeros((n,), dtype=np.float32)
    reward[-1] = 1.0
    done = np.zeros((n,), dtype=np.bool_)
    done[-1] = True

    return {
        "timestep": np.arange(n, dtype=np.int32),
        "state": np.asarray([_to_state34(s) for s in states[:n]], dtype=np.float32),
        "base_action": np.asarray([a[:7] for a in actions[:n]], dtype=np.float32),
        "action": np.asarray([a[:7] for a in actions[:n]], dtype=np.float32),
        "reward": reward,
        "done": done,
        "front_image": np.asarray(front[:n], dtype=np.uint8),
    }


def _build_state_csv(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    gripper_pos: float,
    ee_xyzrpy: np.ndarray,
) -> np.ndarray:
    arm_pos = joint_pos[:7]
    finger_pos = np.array([gripper_pos * 0.04, gripper_pos * 0.04], dtype=np.float32)
    full_pos = np.concatenate([arm_pos, finger_pos])

    arm_vel = joint_vel[:7]
    finger_vel = np.zeros(2, dtype=np.float32)
    full_vel = np.concatenate([arm_vel, finger_vel]) * 0.1

    cogact_ref = np.zeros(7, dtype=np.float32)
    contact = np.zeros(3, dtype=np.float32)
    return np.concatenate([full_pos, full_vel, cogact_ref, contact, ee_xyzrpy]).astype(np.float32)


def _load_offline_from_csv_episode(episode_dir: Path) -> dict[str, np.ndarray]:
    delta_csv = episode_dir / "isaac_deltaEEF_gripper.csv"
    joint_csv = episode_dir / "franka_joint_states.csv"
    gripper_csv = episode_dir / "gripper_joint_states.csv"
    abs_eef_csv = episode_dir / "isaac_absEEF_gripper_convert314.csv"
    video_path = episode_dir / "sim_images_multi_view" / "right_view.mp4"

    if not (delta_csv.exists() and joint_csv.exists() and gripper_csv.exists()):
        raise FileNotFoundError(f"CSV episode files not found under: {episode_dir}")

    delta_df = pd.read_csv(delta_csv)
    joint_df = pd.read_csv(joint_csv)
    gripper_df = pd.read_csv(gripper_csv)

    js_raw = joint_df.values[1:, 10:17].astype(np.float32)
    js_vel = joint_df.values[1:, 17:24].astype(np.float32)
    gs_raw = gripper_df.values[1:, 4:5].astype(np.float32)
    delta_raw = delta_df.values.astype(np.float32)

    t = min(len(js_raw), len(js_vel), len(gs_raw), len(delta_raw))
    if abs_eef_csv.exists():
        abs_eef = pd.read_csv(abs_eef_csv).values.astype(np.float32)
        t = min(t, len(abs_eef))
    else:
        abs_eef = np.zeros((t, 7), dtype=np.float32)

    js_raw = js_raw[:t]
    js_vel = js_vel[:t]
    gs_raw = gs_raw[:t]
    delta_raw = delta_raw[:t]
    abs_eef = abs_eef[:t]

    if video_path.exists():
        front = _read_video_frames_chw(video_path, max_frames=t)
    else:
        front = np.zeros((t, 3, 84, 84), dtype=np.uint8)

    t = min(t, len(front))
    if t < 2:
        raise RuntimeError(f"CSV episode too short after alignment: {t}")

    n = t - 1
    states = []
    for i in range(n):
        gripper_frac = float(np.clip(gs_raw[i, 0], 0.0, 1.0))
        ee_xyzrpy = abs_eef[i, :6] if abs_eef.shape[1] >= 6 else np.zeros(6, dtype=np.float32)
        states.append(_build_state_csv(js_raw[i], js_vel[i], gripper_frac, ee_xyzrpy))

    reward = np.zeros((n,), dtype=np.float32)
    reward[-1] = 1.0
    done = np.zeros((n,), dtype=np.bool_)
    done[-1] = True

    return {
        "timestep": np.arange(n, dtype=np.int32),
        "state": np.asarray(states, dtype=np.float32),
        "base_action": np.asarray(delta_raw[:n, :7], dtype=np.float32),
        "action": np.asarray(delta_raw[:n, :7], dtype=np.float32),
        "reward": reward,
        "done": done,
        "front_image": np.asarray(front[:n], dtype=np.uint8),
    }


def _canonicalize_offline_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "state" in data.files and "front_image" in data.files:
        return {
            "timestep": data["timestep"].astype(np.int32),
            "state": data["state"].astype(np.float32),
            "base_action": data["base_action"].astype(np.float32),
            "action": data["action"].astype(np.float32),
            "reward": data["reward"].astype(np.float32),
            "done": data["done"].astype(np.bool_),
            "front_image": data["front_image"].astype(np.uint8),
        }

    required = {
        "observation_state",
        "observation_base_action",
        "action",
        "reward",
        "done",
        "observation_images_front",
    }
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"offline npz missing keys: {sorted(missing)}")

    n = int(data["action"].shape[0])
    return {
        "timestep": np.arange(n, dtype=np.int32),
        "state": data["observation_state"].astype(np.float32),
        "base_action": data["observation_base_action"].astype(np.float32),
        "action": data["action"].astype(np.float32),
        "reward": data["reward"].astype(np.float32),
        "done": data["done"].astype(np.bool_),
        "front_image": data["observation_images_front"].astype(np.uint8),
    }


def _canonicalize_online_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = {"state", "base_action", "action", "reward", "done", "front_image"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"online npz missing keys: {sorted(missing)}")

    n = int(data["action"].shape[0])
    timestep = data["timestep"].astype(np.int32) if "timestep" in data.files else np.arange(n, dtype=np.int32)
    return {
        "timestep": timestep,
        "state": data["state"].astype(np.float32),
        "base_action": data["base_action"].astype(np.float32),
        "action": data["action"].astype(np.float32),
        "reward": data["reward"].astype(np.float32),
        "done": data["done"].astype(np.bool_),
        "front_image": data["front_image"].astype(np.uint8),
    }


def _override_online_front_with_video(online: dict[str, np.ndarray], right_view_mp4: Path) -> dict[str, np.ndarray]:
    if not right_view_mp4.exists():
        raise FileNotFoundError(f"online_right_view_mp4 not found: {right_view_mp4}")
    frames = _read_video_frames_chw(right_view_mp4, max_frames=int(online["action"].shape[0]))
    if frames.shape[0] == 0:
        raise RuntimeError(f"No frames decoded from online_right_view_mp4: {right_view_mp4}")

    n = int(online["action"].shape[0])
    if frames.shape[0] < n:
        last = frames[-1:]
        pad = np.repeat(last, n - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    elif frames.shape[0] > n:
        frames = frames[:n]

    out = dict(online)
    out["front_image"] = frames.astype(np.uint8)
    return out


def _validate_canonical(name: str, sample: dict[str, np.ndarray]) -> None:
    n = sample["action"].shape[0]
    for key in ["timestep", "state", "base_action", "action", "reward", "done", "front_image"]:
        if sample[key].shape[0] != n:
            raise ValueError(f"{name}: key {key} first dim mismatch: {sample[key].shape[0]} vs {n}")
    if sample["front_image"].ndim != 4 or sample["front_image"].shape[1:] != (3, 84, 84):
        raise ValueError(f"{name}: front_image must be (N,3,84,84), got {sample['front_image'].shape}")


def _save_side_by_side_video(
    offline_front: np.ndarray,
    online_front: np.ndarray,
    out_path: Path,
    fps: int,
) -> None:
    t = max(len(offline_front), len(online_front))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(out_path.as_posix(), fps=fps, codec="libx264", macro_block_size=1) as writer:
        for i in range(t):
            off_idx = min(i, len(offline_front) - 1)
            on_idx = min(i, len(online_front) - 1)
            off = np.transpose(offline_front[off_idx], (1, 2, 0))
            on = np.transpose(online_front[on_idx], (1, 2, 0))
            if off.shape[:2] != (96, 96):
                y_idx = np.linspace(0, off.shape[0] - 1, 96).astype(np.int32)
                x_idx = np.linspace(0, off.shape[1] - 1, 96).astype(np.int32)
                off = off[np.ix_(y_idx, x_idx)]
            if on.shape[:2] != (96, 96):
                y_idx = np.linspace(0, on.shape[0] - 1, 96).astype(np.int32)
                x_idx = np.linspace(0, on.shape[1] - 1, 96).astype(np.int32)
                on = on[np.ix_(y_idx, x_idx)]
            frame = np.concatenate([off, on], axis=1)
            writer.append_data(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline+online canonical buffers, merge, and visualize together")
    parser.add_argument(
        "--offline_npz",
        type=str,
        default="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/offline_buffer_exports/offline_buffer_sample_1.npz",
    )
    parser.add_argument(
        "--offline_lerobot_dir",
        type=str,
        default="/home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_libero_replay_poseinit",
    )
    parser.add_argument("--offline_episode_index", type=int, default=0)
    parser.add_argument(
        "--online_npz",
        type=str,
        default="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/online_buffer_exports/online_episode_000.npz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/paired_buffer_exports",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--force_rebuild_offline", action="store_true")
    parser.add_argument("--allow_black_online", action="store_true")
    parser.add_argument(
        "--online_right_view_mp4",
        type=str,
        default=None,
        help="Optional override video from GR00T iface env (right_view.mp4). If set, online front_image is replaced from this video.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir={out_dir}", flush=True)

    offline_path = Path(args.offline_npz)
    print(f"offline_npz={offline_path}", flush=True)
    print(f"online_npz={Path(args.online_npz)}", flush=True)
    if args.force_rebuild_offline or (not offline_path.exists()):
        offline_root = Path(args.offline_lerobot_dir)
        if (offline_root / "data" / "chunk-000").exists():
            print(f"offline_source=lerobot_local root={offline_root}")
            offline_sample = _load_offline_from_lerobot(offline_root, args.offline_episode_index)
        else:
            print(f"offline_source=csv_episode root={offline_root}")
            offline_sample = _load_offline_from_csv_episode(offline_root)
        np.savez_compressed(offline_path, **offline_sample)
        print(f"saved_offline_npz={offline_path}")

    offline = _canonicalize_offline_npz(offline_path)
    online = _canonicalize_online_npz(Path(args.online_npz))
    if args.online_right_view_mp4:
        right_path = Path(args.online_right_view_mp4)
        print(f"online_front_override={right_path}", flush=True)
        online = _override_online_front_with_video(online, right_path)

    _validate_canonical("offline", offline)
    _validate_canonical("online", online)

    _, _, _, off_zero = _frame_stats("offline.front_image", offline["front_image"])
    _, _, _, on_zero = _frame_stats("online.front_image", online["front_image"])
    if on_zero == int(online["front_image"].shape[0]) and not args.allow_black_online:
        raise RuntimeError(
            "Online front_image is fully black (all frames zero). "
            "Regenerate online NPZ with camera/render capture, or pass --allow_black_online."
        )

    n_off = offline["action"].shape[0]
    n_on = online["action"].shape[0]

    offline_canonical_path = out_dir / "offline_canonical.npz"
    online_canonical_path = out_dir / "online_canonical.npz"
    np.savez_compressed(offline_canonical_path, **offline)
    np.savez_compressed(online_canonical_path, **online)

    merged = {
        "timestep": np.arange(n_off + n_on, dtype=np.int32),
        "state": np.concatenate([offline["state"], online["state"]], axis=0),
        "base_action": np.concatenate([offline["base_action"], online["base_action"]], axis=0),
        "action": np.concatenate([offline["action"], online["action"]], axis=0),
        "reward": np.concatenate([offline["reward"], online["reward"]], axis=0),
        "done": np.concatenate([offline["done"], online["done"]], axis=0),
        "front_image": np.concatenate([offline["front_image"], online["front_image"]], axis=0),
        "source_id": np.concatenate(
            [np.zeros(n_off, dtype=np.int32), np.ones(n_on, dtype=np.int32)],
            axis=0,
        ),
        "offline_length": np.asarray(n_off, dtype=np.int32),
        "online_length": np.asarray(n_on, dtype=np.int32),
    }

    merged_path = out_dir / "offline_then_online_merged.npz"
    np.savez_compressed(merged_path, **merged)

    compare_video = out_dir / "offline_vs_online_side_by_side.mp4"
    _save_side_by_side_video(offline["front_image"], online["front_image"], compare_video, fps=args.fps)

    print(f"offline_steps={n_off}")
    print(f"online_steps={n_on}")
    print(f"saved_offline_canonical={offline_canonical_path}")
    print(f"saved_online_canonical={online_canonical_path}")
    print(f"saved_merged_npz={merged_path}")
    print(f"saved_compare_video={compare_video}")


if __name__ == "__main__":
    main()
