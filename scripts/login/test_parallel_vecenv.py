#!/usr/bin/env python3
"""Test parallel SubprocVecEnv vs sequential in MuJoCoVecEnvStack.

Usage:
  python scripts/test_parallel_vecenv.py --num_envs 4 --steps 10
"""
import os
import sys
import time
import argparse

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Setup LD_LIBRARY_PATH for EGL
ld = os.environ.get("LD_LIBRARY_PATH", "")
extra = "/home/nas_main/junhahyung/miniconda3/lib"
if extra not in ld:
    os.environ["LD_LIBRARY_PATH"] = f"{extra}:{ld}"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "resfit", "rl_finetuning", "wrappers"))

import numpy as np
import torch


def run_test(num_envs: int, num_steps: int, parallel: bool, num_workers: int = 4):
    from mujoco_vec_env_stack import MuJoCoVecEnvStack

    print(f"\n{'='*60}")
    print(f"Testing: parallel={parallel}, num_envs={num_envs}, steps={num_steps}, workers={num_workers}")
    print(f"{'='*60}")

    t0 = time.time()
    env = MuJoCoVecEnvStack(
        num_envs=num_envs,
        max_episode_steps=100,
        reward_type="sparse",
        device="cpu",
        parallel_envs=parallel,
        num_workers=num_workers,
    )
    init_time = time.time() - t0
    print(f"Init time: {init_time:.2f}s")

    # Reset
    t0 = time.time()
    obs, _ = env.reset()
    reset_time = time.time() - t0
    print(f"Reset time: {reset_time:.2f}s")
    print(f"  State shape: {obs['observation.state'].shape}")
    print(f"  Object state shape: {obs['observation.object_state'].shape}")

    # Run steps with random actions
    step_times = []
    all_rewards = []
    for s in range(num_steps):
        # Random absolute action: home-ish pose + small noise
        actions = np.zeros((num_envs, 8), dtype=np.float32)
        actions[:, 0] = 0.35 + np.random.uniform(-0.05, 0.05, num_envs)  # x
        actions[:, 1] = 0.0 + np.random.uniform(-0.05, 0.05, num_envs)   # y
        actions[:, 2] = 0.35 + np.random.uniform(-0.05, 0.05, num_envs)  # z
        actions[:, 3:7] = [0, 0.707, 0.707, 0]  # downward quat
        actions[:, 7] = 0.04  # open gripper
        actions_t = torch.as_tensor(actions)

        t0 = time.time()
        obs, rewards, terminated, truncated, info = env.step(actions_t, render_mode="none")
        dt = time.time() - t0
        step_times.append(dt)
        all_rewards.append(rewards.numpy())

    avg_step = np.mean(step_times)
    print(f"\nStep timing ({num_steps} steps):")
    print(f"  Average: {avg_step*1000:.1f}ms")
    print(f"  Min: {min(step_times)*1000:.1f}ms")
    print(f"  Max: {max(step_times)*1000:.1f}ms")
    print(f"  Total: {sum(step_times):.2f}s")

    # Test get_groot_obs (triggers qpos sync in parallel mode)
    t0 = time.time()
    groot_obs = env.get_groot_obs(0)
    dt_groot = time.time() - t0
    print(f"\nget_groot_obs time: {dt_groot*1000:.1f}ms")
    print(f"  cam_base shape: {groot_obs['video']['cam_base'].shape}")
    print(f"  eef_pos: {groot_obs['state']['proprio.eef_pos'][0,0]}")

    # Final state check
    final_state = obs["observation.state"][0].numpy()
    print(f"\nFinal state[0]: pos={final_state[:3]}, quat={final_state[3:7]}")

    env.close()
    return avg_step, obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Run sequential
    seq_time, seq_obs = run_test(args.num_envs, args.steps, parallel=False)

    # Run parallel
    par_time, par_obs = run_test(args.num_envs, args.steps, parallel=True, num_workers=args.workers)

    # Compare
    print(f"\n{'='*60}")
    print(f"COMPARISON (num_envs={args.num_envs}, steps={args.steps})")
    print(f"{'='*60}")
    print(f"Sequential: {seq_time*1000:.1f}ms/step")
    print(f"Parallel:   {par_time*1000:.1f}ms/step")
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"Speedup:    {speedup:.2f}x")


if __name__ == "__main__":
    main()
