#!/usr/bin/env python3
"""
Depth visualization test: use the existing IfaceEnvWrapper (which correctly
sets up the robot + scene), but patch cameras to also output depth.
Runs 1000 steps with base policy, saves RGB+Depth side-by-side video.
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--output_dir", type=str, default="/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/depth_test")
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import imageio
import time
from pathlib import Path

import isaaclab.sim as sim_utils


def depth_to_colormap(depth_tensor, env_id=0, max_depth=2.0, adaptive=True):
    """Convert depth tensor to colorized numpy image (H,W,3) uint8.
    If adaptive=True, uses per-frame min-max normalization for max contrast."""
    d = depth_tensor[env_id].cpu().numpy().squeeze()
    d = np.nan_to_num(d, nan=max_depth, posinf=max_depth, neginf=0)
    d = np.clip(d, 0, max_depth)
    
    if adaptive:
        dmin = d.min()
        dmax = d.max()
        if dmax - dmin < 1e-4:
            d_norm = np.zeros_like(d)
        else:
            d_norm = 1.0 - (d - dmin) / (dmax - dmin)  # close=1, far=0
    else:
        d_norm = 1.0 - (d / max_depth)
    
    # Turbo-like colormap for better perceptual contrast
    # close (d_norm=1) → red/yellow, far (d_norm=0) → blue/dark
    r = np.clip(np.where(d_norm > 0.5, 1.0, d_norm * 2), 0, 1)
    g = np.clip(np.where(d_norm > 0.5, 2 - d_norm * 2, d_norm * 2), 0, 1)
    b = np.clip(np.where(d_norm < 0.5, 1.0 - d_norm * 2, 0), 0, 1)
    img = np.stack([r, g, b], axis=-1)
    return (img * 255).astype(np.uint8)


def depth_to_colormap_fixed(depth_np, d_min, d_max):
    """Convert raw depth numpy array to colorized (H,W,3) uint8 using fixed min/max."""
    d = np.nan_to_num(depth_np, nan=d_max, posinf=d_max, neginf=0)
    d = np.clip(d, d_min, d_max)
    d_norm = 1.0 - (d - d_min) / max(d_max - d_min, 1e-6)  # close=1, far=0
    r = np.clip(np.where(d_norm > 0.5, 1.0, d_norm * 2), 0, 1)
    g = np.clip(np.where(d_norm > 0.5, 2 - d_norm * 2, d_norm * 2), 0, 1)
    b = np.clip(np.where(d_norm < 0.5, 1.0 - d_norm * 2, 0), 0, 1)
    img = np.stack([r, g, b], axis=-1)
    return (img * 255).astype(np.uint8)


def rgb_to_numpy(rgb_tensor, env_id=0):
    rgb = rgb_tensor[env_id]
    if rgb.is_cuda:
        rgb = rgb.cpu()
    arr = rgb.numpy()
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.float32:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr


def main():
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Patch _make_scene_cfg to add depth to cameras BEFORE env creation ──
    import resfit.rl_finetuning.wrappers.iface_env_wrapper as wrapper_mod
    _orig_make_scene_cfg = wrapper_mod._make_scene_cfg

    def _patched_make_scene_cfg(*a, **kw):
        cfg = _orig_make_scene_cfg(*a, **kw)
        cfg.camera_front.data_types = ["rgb", "depth"]
        cfg.camera_back.data_types = ["rgb", "depth"]
        cfg.camera_wrist.data_types = ["rgb", "depth"]
        print("[PATCH] Cameras patched to output RGB + Depth")
        return cfg

    wrapper_mod._make_scene_cfg = _patched_make_scene_cfg

    # ── Create env using existing wrapper ──
    from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    _sim = sim_utils.SimulationContext(sim_cfg)

    env = IfaceEnvWrapper(
        sim=_sim,
        csv_dir=args.csv_base_dir,
        groot_model_path=args.groot_model_path,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=args.groot_policy_device,
        policy_strict=True,
        task_description="Pick up the mushroom.",
        language_override="pick up mushroom",
        max_episode_steps=args.max_steps,
        num_envs=args.num_envs,
        success_threshold=0.03,
    )

    print(f"Env created: {args.num_envs} envs")
    print(f"Front camera data_types: {env.camera_front.cfg.data_types}")
    print(f"Wrist camera data_types: {env.camera_wrist.cfg.data_types}")
    print(f"Front camera output keys: {list(env.camera_front.data.output.keys())}")

    # ── Reset and run ──
    env.reset()

    # Pass 1: collect global depth stats
    print(f"Pass 1: Collecting depth stats over {args.max_steps} steps...")
    t0 = time.time()

    front_depths_min, front_depths_max = [], []
    wrist_depths_min, wrist_depths_max = [], []
    back_depths_min, back_depths_max = [], []
    raw_frames = []  # store (front_rgb, front_depth_raw, wrist_rgb, wrist_depth_raw)

    for step in range(args.max_steps):
        obs, reward, terminated, truncated, info = env.step(residual_action=None)

        if step % 5 == 0:
            try:
                fd = env.camera_front.data.output["depth"][0].cpu().numpy().squeeze()
                fd = np.nan_to_num(fd, nan=10.0, posinf=10.0, neginf=0)
                front_depths_min.append(fd[fd > 0.01].min() if (fd > 0.01).any() else 0.1)
                front_depths_max.append(fd[fd < 9.0].max() if (fd < 9.0).any() else 2.0)

                wd = env.camera_wrist.data.output["depth"][0].cpu().numpy().squeeze()
                wd = np.nan_to_num(wd, nan=5.0, posinf=5.0, neginf=0)
                wrist_depths_min.append(wd[wd > 0.01].min() if (wd > 0.01).any() else 0.03)
                wrist_depths_max.append(wd[wd < 4.0].max() if (wd < 4.0).any() else 0.5)

                bd = env.camera_back.data.output["depth"][0].cpu().numpy().squeeze()
                bd = np.nan_to_num(bd, nan=10.0, posinf=10.0, neginf=0)
                back_depths_min.append(bd[bd > 0.01].min() if (bd > 0.01).any() else 0.1)
                back_depths_max.append(bd[bd < 9.0].max() if (bd < 9.0).any() else 2.0)

                raw_frames.append((
                    rgb_to_numpy(env.camera_front.data.output["rgb"], 0),
                    fd.copy(),
                    rgb_to_numpy(env.camera_wrist.data.output["rgb"], 0),
                    wd.copy(),
                ))
            except Exception as e:
                if step < 20:
                    print(f"[WARN] step {step}: {e}")

        if step % 200 == 0:
            elapsed = time.time() - t0
            print(f"  step {step}/{args.max_steps} ({elapsed:.1f}s)")

    # Compute global min/max (use 2nd percentile / 98th percentile for robustness)
    front_min = float(np.percentile(front_depths_min, 2)) if front_depths_min else 0.3
    front_max = float(np.percentile(front_depths_max, 98)) if front_depths_max else 1.8
    wrist_min = float(np.percentile(wrist_depths_min, 2)) if wrist_depths_min else 0.03
    wrist_max = float(np.percentile(wrist_depths_max, 98)) if wrist_depths_max else 0.4
    back_min = float(np.percentile(back_depths_min, 2)) if back_depths_min else 0.3
    back_max = float(np.percentile(back_depths_max, 98)) if back_depths_max else 1.8

    print(f"\nGlobal depth ranges:")
    print(f"  Front: [{front_min:.4f}, {front_max:.4f}]m")
    print(f"  Wrist: [{wrist_min:.4f}, {wrist_max:.4f}]m")
    print(f"  Back:  [{back_min:.4f}, {back_max:.4f}]m")

    # Save depth min/max to JSON
    import json
    depth_config_dir = Path("/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/depth_min_max")
    depth_config_dir.mkdir(parents=True, exist_ok=True)
    depth_config = {
        "front": {"min": front_min, "max": front_max},
        "wrist": {"min": wrist_min, "max": wrist_max},
        "back": {"min": back_min, "max": back_max},
        "notes": "Computed from 1000-step rollout with base policy, 2nd/98th percentile",
    }
    config_path = depth_config_dir / "depth_normalization.json"
    with open(config_path, "w") as f:
        json.dump(depth_config, f, indent=2)
    print(f"Saved depth config: {config_path}")

    # Pass 2: render with fixed normalization
    print(f"\nPass 2: Rendering {len(raw_frames)} frames with fixed normalization...")
    frames_front = []
    frames_wrist = []

    for front_rgb, fd_raw, wrist_rgb, wd_raw in raw_frames:
        front_depth_vis = depth_to_colormap_fixed(fd_raw, front_min, front_max)
        h = min(front_rgb.shape[0], front_depth_vis.shape[0])
        w = min(front_rgb.shape[1], front_depth_vis.shape[1])
        frames_front.append(np.concatenate([front_rgb[:h, :w], front_depth_vis[:h, :w]], axis=1))

        wrist_depth_vis = depth_to_colormap_fixed(wd_raw, wrist_min, wrist_max)
        h = min(wrist_rgb.shape[0], wrist_depth_vis.shape[0])
        w = min(wrist_rgb.shape[1], wrist_depth_vis.shape[1])
        frames_wrist.append(np.concatenate([wrist_rgb[:h, :w], wrist_depth_vis[:h, :w]], axis=1))

    elapsed = time.time() - t0
    print(f"Done: {args.max_steps} steps in {elapsed:.1f}s, {len(frames_front)} frames")

    if frames_front:
        front_path = output_dir / "front_rgb_depth.mp4"
        w = imageio.get_writer(str(front_path), fps=20)
        for f in frames_front:
            w.append_data(f)
        w.close()
        print(f"Saved: {front_path}")

    if frames_wrist:
        wrist_path = output_dir / "wrist_rgb_depth.mp4"
        w = imageio.get_writer(str(wrist_path), fps=20)
        for f in frames_wrist:
            w.append_data(f)
        w.close()
        print(f"Saved: {wrist_path}")

    if frames_front:
        mid = len(frames_front) // 2
        imageio.imwrite(str(output_dir / "sample_front.png"), frames_front[mid])
        imageio.imwrite(str(output_dir / "sample_wrist.png"), frames_wrist[mid])
        print("Saved sample PNGs")

    env.close()
    print("=== DEPTH TEST DONE ===")
    os._exit(0)


if __name__ == "__main__":
    main()
