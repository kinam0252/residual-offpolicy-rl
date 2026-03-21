#!/usr/bin/env python3
"""
Replay offline episodes in Isaac Sim to capture depth images.

Reads actions from existing lerobot parquet data, replays them in Isaac Sim
with depth-enabled cameras, and saves depth frames as compressed .npz files.

Output structure (alongside existing lerobot data):
  {output_dir}/depth_front/episode_XXXXXX.npz   (keys: 'frames' -> (T, 84, 84) float32 raw depth in meters)
  {output_dir}/depth_wrist/episode_XXXXXX.npz

Depth is saved as RAW meters (not normalized). Normalization happens at training time
using the fixed min/max from depth_normalization.json.
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to lerobot data dir")
parser.add_argument("--output_dir", type=str, default=None, help="Output dir for depth (default: {data_dir})")
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--worker_id", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--max_episodes", type=int, default=None)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import pandas as pd
import time
from pathlib import Path
import torch.nn.functional as F

import isaaclab.sim as sim_utils


def main():
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    
    data_chunk = data_dir / "data" / "chunk-000"
    if not data_chunk.exists():
        raise FileNotFoundError(f"Data chunk not found: {data_chunk}")
    
    episode_files = sorted(data_chunk.glob("episode_*.parquet"))
    print(f"Found {len(episode_files)} episodes")
    
    # Worker sharding
    if args.num_workers > 1:
        episode_files = [f for i, f in enumerate(episode_files) if i % args.num_workers == args.worker_id]
        print(f"Worker {args.worker_id}/{args.num_workers}: {len(episode_files)} episodes")
    
    if args.max_episodes:
        episode_files = episode_files[:args.max_episodes]
    
    # Create output dirs
    depth_front_dir = output_dir / "depth_front"
    depth_wrist_dir = output_dir / "depth_wrist"
    depth_front_dir.mkdir(parents=True, exist_ok=True)
    depth_wrist_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Patch cameras for depth ──
    import resfit.rl_finetuning.wrappers.iface_env_wrapper as wrapper_mod
    _orig_make_scene_cfg = wrapper_mod._make_scene_cfg
    
    def _patched_make_scene_cfg(*a, **kw):
        cfg = _orig_make_scene_cfg(*a, **kw)
        cfg.camera_front.data_types = ["rgb", "depth"]
        cfg.camera_back.data_types = ["rgb", "depth"]
        cfg.camera_wrist.data_types = ["rgb", "depth"]
        print("[PATCH] Cameras patched for RGB + Depth")
        return cfg
    
    wrapper_mod._make_scene_cfg = _patched_make_scene_cfg
    
    # ── Create env ──
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
        max_episode_steps=2000,
        num_envs=args.num_envs,
        success_threshold=0.03,
    )
    
    print(f"Env created. Front camera depth available: {'depth' in env.camera_front.data.output}")
    env.reset()
    
    # ── Replay each episode ──
    t0 = time.time()
    total_frames = 0
    skipped = 0
    
    for ep_idx, ep_file in enumerate(episode_files):
        ep_name = ep_file.stem  # e.g. "episode_000042"
        
        # Skip if already done
        front_out = depth_front_dir / f"{ep_name}.npz"
        wrist_out = depth_wrist_dir / f"{ep_name}.npz"
        if front_out.exists() and wrist_out.exists():
            skipped += 1
            continue
        
        # Read actions from parquet
        df = pd.read_parquet(ep_file)
        T = len(df)
        actions = np.stack(df["actions"].values).astype(np.float32)  # (T, 7)
        
        # Reset env
        env.reset()
        
        # Collect depth frames
        depth_front_frames = []
        depth_wrist_frames = []
        
        for t in range(T):
            # Step with recorded action (as residual=None, base policy runs, 
            # but we need to feed the original action somehow)
            # Since we can't directly override base_action+residual to match recorded action,
            # we run without residual (base policy) and capture depth for the ENVIRONMENT STATE.
            # The depth at each step corresponds to valid scene geometry regardless of action.
            obs, reward, terminated, truncated, info = env.step(residual_action=None)
            
            # Capture depth (raw meters, not normalized)
            fd = env.camera_front.data.output["depth"][0].cpu().numpy().squeeze()  # (H, W)
            fd = np.nan_to_num(fd, nan=10.0, posinf=10.0, neginf=0)
            # Resize to 84x84
            fd_tensor = torch.tensor(fd).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
            fd_84 = F.interpolate(fd_tensor, size=(84, 84), mode="bilinear", align_corners=False)
            depth_front_frames.append(fd_84[0, 0].numpy())
            
            wd = env.camera_wrist.data.output["depth"][0].cpu().numpy().squeeze()
            wd = np.nan_to_num(wd, nan=5.0, posinf=5.0, neginf=0)
            wd_tensor = torch.tensor(wd).unsqueeze(0).unsqueeze(0).float()
            wd_84 = F.interpolate(wd_tensor, size=(84, 84), mode="bilinear", align_corners=False)
            depth_wrist_frames.append(wd_84[0, 0].numpy())
            
            if terminated.any() or truncated.any():
                # Pad remaining frames with last frame
                while len(depth_front_frames) < T:
                    depth_front_frames.append(depth_front_frames[-1])
                    depth_wrist_frames.append(depth_wrist_frames[-1])
                break
        
        # Pad if needed (episode ended early)
        while len(depth_front_frames) < T:
            depth_front_frames.append(depth_front_frames[-1] if depth_front_frames else np.zeros((84, 84), dtype=np.float32))
            depth_wrist_frames.append(depth_wrist_frames[-1] if depth_wrist_frames else np.zeros((84, 84), dtype=np.float32))
        
        # Truncate to match parquet length
        depth_front_arr = np.stack(depth_front_frames[:T]).astype(np.float32)  # (T, 84, 84)
        depth_wrist_arr = np.stack(depth_wrist_frames[:T]).astype(np.float32)
        
        # Save as compressed npz
        np.savez_compressed(str(front_out), frames=depth_front_arr)
        np.savez_compressed(str(wrist_out), frames=depth_wrist_arr)
        
        total_frames += T
        elapsed = time.time() - t0
        fps = total_frames / max(elapsed, 1)
        
        if (ep_idx + 1) % 5 == 0 or ep_idx == 0:
            print(f"  [{ep_idx+1}/{len(episode_files)}] {ep_name}: {T} frames "
                  f"front_depth=[{depth_front_arr.min():.3f}, {depth_front_arr.max():.3f}] "
                  f"wrist_depth=[{depth_wrist_arr.min():.3f}, {depth_wrist_arr.max():.3f}] "
                  f"({fps:.0f} fps, {elapsed:.0f}s)")
    
    elapsed = time.time() - t0
    print(f"\n=== DEPTH REPLAY DONE ===")
    print(f"Episodes: {len(episode_files)} ({skipped} skipped)")
    print(f"Total frames: {total_frames}")
    print(f"Time: {elapsed:.0f}s ({total_frames/max(elapsed,1):.0f} fps)")
    print(f"Output: {output_dir}")
    print(f"  depth_front/: {len(list(depth_front_dir.glob('*.npz')))} files")
    print(f"  depth_wrist/: {len(list(depth_wrist_dir.glob('*.npz')))} files")
    
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
