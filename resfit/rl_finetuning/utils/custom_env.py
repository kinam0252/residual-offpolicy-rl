"""
Custom environment snapshot system.

Saves/loads deterministic env configurations so warmup, training, and eval
all use the exact same cube positions, perturbations, and initial frames.

Snapshot structure:
  custom_env/
    {hash16}/
      env_config.json      # csv_path, perturbation table, success_threshold, etc.
      first_frames/        # first-frame images per env (sanity check)
        env0_center.png
        env1_right_5cm.png
        ...
      cube_poses.pt        # (N, 7) tensor of cube root poses after settle
      robot_joint_init.pt  # (N, J) tensor of initial joint positions
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def compute_env_hash(
    csv_base_dir: str,
    success_threshold: float,
    cube_perturb_table_path: str | None,
    cube_perturb_range: float,
    num_envs: int,
    max_episode_steps: int,
) -> str:
    """Compute a deterministic hash of env configuration."""
    key = {
        "csv_base_dir": str(csv_base_dir),
        "success_threshold": float(success_threshold),
        "cube_perturb_table_path": str(cube_perturb_table_path or ""),
        "cube_perturb_range": float(cube_perturb_range),
        "num_envs": int(num_envs),
        "max_episode_steps": int(max_episode_steps),
    }
    # Include perturbation table content if it exists
    if cube_perturb_table_path and Path(cube_perturb_table_path).exists():
        key["perturb_table_content"] = Path(cube_perturb_table_path).read_text()
    key_str = json.dumps(key, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def get_snapshot_dir(custom_env_root: Path, env_hash: str) -> Path:
    return custom_env_root / env_hash


def snapshot_exists(custom_env_root: Path, env_hash: str) -> bool:
    d = get_snapshot_dir(custom_env_root, env_hash)
    return (d / "env_config.json").exists() and (d / "cube_poses.pt").exists()


def save_env_snapshot(
    custom_env_root: Path,
    env_hash: str,
    env,  # IfaceEnvWrapper
    config: dict,
    perturb_names: list[str] | None = None,
):
    """Save env snapshot after reset+settle: cube poses, robot joints, first frames."""
    snap_dir = get_snapshot_dir(custom_env_root, env_hash)
    snap_dir.mkdir(parents=True, exist_ok=True)

    N = env.num_envs

    # Save config
    config["hash"] = env_hash
    (snap_dir / "env_config.json").write_text(json.dumps(config, indent=2))

    # Save cube poses (N, 7): pos(3) + quat(4)
    cube_poses = env.cube.data.root_state_w[:N, :7].detach().cpu()
    torch.save(cube_poses, str(snap_dir / "cube_poses.pt"))

    # Save robot initial joint positions
    robot_q = env.robot.data.joint_pos[:N].detach().cpu()
    torch.save(robot_q, str(snap_dir / "robot_joint_init.pt"))

    # Save initial_cube_z
    torch.save(env._initial_cube_z[:N].detach().cpu(), str(snap_dir / "initial_cube_z.pt"))

    # Save first-frame images
    frames_dir = snap_dir / "first_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(env, "get_frame"):
        names = perturb_names or [f"env{i}" for i in range(N)]
        for eid in range(N):
            fr = env.get_frame(eid, camera="front", size=(128, 160))
            # Save as PNG
            try:
                from PIL import Image
                img = Image.fromarray(fr)
                img.save(str(frames_dir / f"env{eid}_{names[eid]}.png"))
            except ImportError:
                np.save(str(frames_dir / f"env{eid}_{names[eid]}.npy"), fr)

    print(f"[custom_env] Saved snapshot hash={env_hash}: {N} envs, cube_poses, frames → {snap_dir}")
    return snap_dir


def load_env_snapshot(
    custom_env_root: Path,
    env_hash: str,
    env,  # IfaceEnvWrapper
    settle_steps: int = 50,
) -> dict:
    """Load cube poses and robot joints from snapshot into env.
    Runs settle_steps of sim to let physics stabilize after pose override.
    Returns the saved config dict."""
    snap_dir = get_snapshot_dir(custom_env_root, env_hash)

    config = json.loads((snap_dir / "env_config.json").read_text())
    N = env.num_envs

    # Load and apply cube poses
    cube_poses = torch.load(str(snap_dir / "cube_poses.pt"), map_location=env.device)
    n_saved = min(cube_poses.shape[0], N)
    env.cube.write_root_pose_to_sim(
        cube_poses[:n_saved],
        env_ids=torch.arange(n_saved, device=env.device, dtype=torch.long),
    )
    env.cube.write_root_velocity_to_sim(
        torch.zeros(n_saved, 6, device=env.device),
        env_ids=torch.arange(n_saved, device=env.device, dtype=torch.long),
    )

    # Settle physics so cube lands properly after pose override
    sim_dt = env.sim_dt
    for _ in range(settle_steps):
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)
        # Update cameras so sanity frames are correct
        for cam in (env.camera_front, env.camera_back, env.camera_wrist):
            cam.update(dt=sim_dt)

    # Override initial_cube_z with snapshot values (not the post-settle values)
    if (snap_dir / "initial_cube_z.pt").exists():
        saved_z = torch.load(str(snap_dir / "initial_cube_z.pt"), map_location=env.device)
        env._initial_cube_z[:n_saved] = saved_z[:n_saved]
        env._max_cube_z[:n_saved] = saved_z[:n_saved].clone()

    print(f"[custom_env] Loaded snapshot hash={env_hash}: {n_saved} env cube poses + {settle_steps} settle steps from {snap_dir}")
    return config


def capture_sanity_frames(env, output_dir: Path, tag: str, perturb_names: list[str] | None = None):
    """Capture first frame of each env and save as sanity check images.
    Includes actual cube XY offset in filename for random perturbation verification."""
    import numpy as np
    N = env.num_envs
    frames_dir = output_dir / f"sanity_frames_{tag}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Build names with actual cube positions if available
    names = []
    for eid in range(N):
        base_name = perturb_names[eid] if perturb_names and eid < len(perturb_names) else f"env{eid}"
        # Get actual cube XY relative to env origin
        try:
            cube_pos = env.cube.data.root_state_w[eid, :3].detach().cpu().numpy()
            env_origin = env.scene.env_origins[eid].detach().cpu().numpy()
            rel_pos = cube_pos - env_origin
            base_name += f"_x{rel_pos[0]*100:+.0f}cm_y{rel_pos[1]*100:+.0f}cm"
        except Exception:
            pass
        names.append(base_name)
    
    if hasattr(env, "get_frame"):
        for eid in range(N):
            fr = env.get_frame(eid, camera="front", size=(128, 160))
            try:
                from PIL import Image
                img = Image.fromarray(fr)
                img.save(str(frames_dir / f"env{eid}_{names[eid]}.png"))
            except ImportError:
                np.save(str(frames_dir / f"env{eid}_{names[eid]}.npy"), fr)
    print(f"[custom_env] Sanity frames ({tag}): {N} images → {frames_dir}")
    return frames_dir
