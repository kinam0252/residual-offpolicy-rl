#!/usr/bin/env python3
"""Evaluate GR00T drawer checkpoints: SR per drawer per checkpoint.
Work-stealing with file locks for multi-worker parallelism.

Usage:
    python scripts/eval_drawer_ckpt_sr.py --output_dir outputs/ckpt_sr_drawer
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ['MUJOCO_GL'] = 'egl'

import importlib, types as _types
if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed"); _ds.__version__="0.0.0"
    _ds.__spec__=importlib.machinery.ModuleSpec("deepspeed",None); _ds.__path__=[]; _ds.__file__=__file__
    _dz = _types.ModuleType("deepspeed.zero"); _dz.__spec__=importlib.machinery.ModuleSpec("deepspeed.zero",None)
    _dz.Init=lambda *a,**kw:(lambda f:f); _ds.zero=_dz
    sys.modules["deepspeed"]=_ds; sys.modules["deepspeed.zero"]=_dz
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except: pass

import numpy as np
import torch
from pathlib import Path
from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer

CKPT_BASE = os.path.expanduser("~/DATA/INTERN/training/groot_drawer_sim_33ep")
CKPT_STEPS = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000]
DRAWERS = [2, 3, 4]
NUM_EPISODES = 20
MAX_STEPS = 300
CONTACT_THRESHOLD = 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/ckpt_sr_drawer")
    p.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    return p.parse_args()


def try_lock(lock_path):
    """Attempt to create a directory lock. Returns True if acquired."""
    try:
        os.makedirs(lock_path)
        return True
    except FileExistsError:
        return False


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build work items: (ckpt_step, drawer_id)
    work_items = [(step, d) for step in CKPT_STEPS for d in DRAWERS]
    np.random.shuffle(work_items)

    wrapper = None
    current_ckpt = None

    for ckpt_step, drawer_id in work_items:
        result_path = output_dir / f"ckpt_{ckpt_step:06d}_d{drawer_id}.json"
        lock_path = output_dir / f"ckpt_{ckpt_step:06d}_d{drawer_id}.lock"

        # Skip if done
        if result_path.exists():
            continue
        # Try to claim
        if not try_lock(str(lock_path)):
            continue

        ckpt_path = os.path.join(CKPT_BASE, f"checkpoint-{ckpt_step}")
        if not os.path.exists(ckpt_path):
            os.rmdir(str(lock_path))
            continue

        print(f"[EVAL] ckpt={ckpt_step}, D{drawer_id}, {args.num_episodes} episodes...")

        # Create env
        vec_env = MuJoCoVecEnvDrawer(
            num_envs=1,
            max_episode_steps=args.max_steps,
            reward_type="dense",
            active_drawers=[drawer_id],
            contact_threshold=CONTACT_THRESHOLD,
        )

        # Load/reuse GR00T
        if wrapper is None or current_ckpt != ckpt_step:
            del wrapper
            wrapper = MuJoCoResidualWrapperDrawer(
                vec_env=vec_env,
                groot_checkpoint=ckpt_path,
                policy_device="cuda:0",
            )
            current_ckpt = ckpt_step
        else:
            wrapper.vec_env = vec_env
            wrapper.num_envs = vec_env.num_envs
            wrapper._cached_chunks = [None] * vec_env.num_envs
            wrapper._chunk_idx = [wrapper.open_loop_horizon] * vec_env.num_envs
            wrapper._held_base_action = np.zeros((vec_env.num_envs, 8), dtype=np.float32)
            wrapper._ema_pos = [None] * vec_env.num_envs
            wrapper._ema_quat = [None] * vec_env.num_envs

        env = wrapper
        successes = 0
        total_rewards = []
        episode_steps = []

        for ep in range(args.num_episodes):
            obs, info = env.reset()
            ep_reward = 0.0

            for step in range(args.max_steps):
                action = np.zeros((1, 7))
                obs, reward, terminated, truncated, info = env.step(action)
                r = reward[0].item() if hasattr(reward[0], 'item') else float(reward[0])
                ep_reward += r
                if terminated[0] or truncated[0]:
                    break

            success = bool(terminated[0])
            successes += int(success)
            total_rewards.append(ep_reward)
            episode_steps.append(step + 1)

        sr = successes / args.num_episodes
        result = {
            "checkpoint": ckpt_step,
            "drawer": drawer_id,
            "num_episodes": args.num_episodes,
            "successes": successes,
            "success_rate": sr,
            "avg_reward": float(np.mean(total_rewards)),
            "avg_steps": float(np.mean(episode_steps)),
            "contact_threshold": CONTACT_THRESHOLD,
        }

        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Release lock
        try:
            os.rmdir(str(lock_path))
        except:
            pass

        print(f"  => SR: {sr*100:.0f}% ({successes}/{args.num_episodes}), "
              f"avg_reward: {np.mean(total_rewards):.1f}, avg_steps: {np.mean(episode_steps):.0f}")

    print("\nAll work items complete!")


if __name__ == "__main__":
    main()
