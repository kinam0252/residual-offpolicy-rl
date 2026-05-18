"""
Base-only drawer eval — lightweight, runs on login node.

Tests GR00T base policy (residual=0) on each drawer independently.
Keeps RAM low: 5 envs at a time, no parallel workers.

Usage:
    cd ~/Repos/Intern/residual-offpolicy-rl
    python3 scripts/eval_drawer_base.py
"""
from __future__ import annotations

import importlib
import os
import sys
import types as _types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DS_BUILD_OPS", "0")

# Deepspeed mock
if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed")
    _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__ = []
    sys.modules["deepspeed"] = _ds

import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified


def eval_drawer(
    drawer_id: int,
    num_envs: int,
    num_episodes: int,
    groot_ckpt: str,
    max_episode_steps: int,
    device: str,
):
    """Evaluate base policy on a single drawer type."""
    print(f"\n{'='*60}")
    print(f"  Evaluating D{drawer_id} | {num_envs} envs × {num_episodes} episodes")
    print(f"{'='*60}")

    # Create env — all envs use same drawer, no parallel workers
    active_drawers = [drawer_id] * num_envs
    vec_env = MuJoCoVecEnvDrawer(
        num_envs=num_envs,
        active_drawers=active_drawers,
        max_episode_steps=max_episode_steps,
        reward_type="delta",
        device=device,
        rl_img_size=84,
        parallel_envs=False,  # no subprocess workers → less RAM
    )

    # Create wrapper with GR00T (base policy)
    wrapper = MuJoCoResidualWrapperUnified(
        vec_env=vec_env,
        groot_checkpoint=groot_ckpt,
        task_description="Close the drawer.",
        chunk_sync=True,
        async_prefetch=False,
        policy_device=device,
    )

    successes = 0
    total_episodes = 0
    per_env_results = [[] for _ in range(num_envs)]

    for ep in range(num_episodes):
        obs, _ = wrapper.reset()
        env_done = [False] * num_envs
        env_success = [False] * num_envs

        for step in range(max_episode_steps):
            # Base action only — feed zero residual
            with torch.no_grad():
                obs, reward, terminated, truncated, info = wrapper.step(
                    torch.zeros(num_envs, 7, device=wrapper.device)
                )
            done = terminated | truncated

            for i in range(num_envs):
                if not env_done[i] and done[i]:
                    env_done[i] = True
                    if terminated[i]:
                        env_success[i] = True

            if all(env_done):
                break

        for i in range(num_envs):
            s = env_success[i]
            per_env_results[i].append(s)
            if s:
                successes += 1
            total_episodes += 1

        ep_sr = sum(env_success) / num_envs
        print(f"  Episode {ep}: {sum(env_success)}/{num_envs} = {ep_sr*100:.0f}%")

    # Summary
    overall_sr = successes / max(1, total_episodes)
    print(f"\n  D{drawer_id} Overall: {successes}/{total_episodes} = {overall_sr*100:.1f}%")
    for i in range(num_envs):
        env_sr = sum(per_env_results[i]) / max(1, len(per_env_results[i]))
        print(f"    env{i}: {sum(per_env_results[i])}/{len(per_env_results[i])} = {env_sr*100:.0f}%")

    # Cleanup to free RAM
    del wrapper, vec_env
    torch.cuda.empty_cache()

    return overall_sr, successes, total_episodes


def main():
    parser = argparse.ArgumentParser(description="Base-only drawer eval")
    parser.add_argument("--drawers", type=int, nargs="+", default=[2, 3, 4],
                        help="Drawer IDs to test (default: 2 3 4)")
    parser.add_argument("--num_envs", type=int, default=5,
                        help="Envs per drawer (default: 5)")
    parser.add_argument("--num_episodes", type=int, default=3,
                        help="Episodes per eval (default: 3)")
    parser.add_argument("--groot_checkpoint", type=str,
                        default="~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000")
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    groot_ckpt = str(Path(args.groot_checkpoint).expanduser())
    print(f"GR00T checkpoint: {groot_ckpt}")
    print(f"Drawers: {args.drawers}")
    print(f"Config: {args.num_envs} envs × {args.num_episodes} episodes")

    results = {}
    for d in args.drawers:
        sr, succ, total = eval_drawer(
            drawer_id=d,
            num_envs=args.num_envs,
            num_episodes=args.num_episodes,
            groot_ckpt=groot_ckpt,
            max_episode_steps=args.max_episode_steps,
            device=args.device,
        )
        results[f"D{d}"] = {"sr": sr, "success": succ, "total": total}

    # Final summary
    print(f"\n{'='*60}")
    print("  SUMMARY: Base Policy Drawer Eval")
    print(f"{'='*60}")
    total_succ = sum(r["success"] for r in results.values())
    total_ep = sum(r["total"] for r in results.values())
    for name, r in results.items():
        print(f"  {name}: {r['success']}/{r['total']} = {r['sr']*100:.1f}%")
    print(f"  Overall: {total_succ}/{total_ep} = {total_succ/max(1,total_ep)*100:.1f}%")


if __name__ == "__main__":
    main()
