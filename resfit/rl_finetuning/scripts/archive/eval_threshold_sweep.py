#!/usr/bin/env python3
"""Threshold sweep: evaluate base policy SR at different contact_threshold values.

For each threshold, runs all 33 episodes and records per-drawer + total SR.
Model is loaded once; only the env's contact_threshold is changed between sweeps.

Usage:
    python eval_threshold_sweep.py --threshold 1.0
    python eval_threshold_sweep.py --threshold inf   # baseline (no gate)
"""
import sys, os, json, time, argparse, importlib.machinery
from pathlib import Path

import types as _types

# Mock deepspeed properly (need __spec__ for transformers)
if "deepspeed" not in _types.__builtins__:
    pass
_ds = _types.ModuleType("deepspeed"); _ds.__version__ = "0.0.0"
_ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
_ds.__path__ = []; _ds.__file__ = __file__
_dz = _types.ModuleType("deepspeed.zero")
_dz.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
_dz.Init = lambda *a, **kw: (lambda f: f); _ds.zero = _dz
_dc = _types.ModuleType("deepspeed.comm")
_dc.__spec__ = importlib.machinery.ModuleSpec("deepspeed.comm", None)
sys.modules["deepspeed"] = _ds
sys.modules["deepspeed.zero"] = _dz
sys.modules["deepspeed.comm"] = _dc

# Patch huggingface_hub for local paths
try:
    import huggingface_hub.utils._validators as _hf_val
    _hf_val.validate_repo_id = lambda x: None
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, load_episode_mapping, FLOOR_TO_DRAWER,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import (
    MuJoCoResidualWrapperDrawer,
)

torch.backends.cuda.enable_cudnn_sdp(False)

MAX_STEPS = 500


def run_one(wrapper, mujoco_env, drawer_idx, max_steps):
    """Run one episode, return success and steps."""
    mujoco_env._active_drawers[0] = drawer_idx
    obs, _ = wrapper.reset()
    success = False
    for step in range(max_steps):
        residual = torch.zeros((1, 7), dtype=torch.float32)
        obs, reward, terminated, truncated, info = wrapper.step(residual)
        if terminated[0] or truncated[0]:
            # wrapper auto-resets on done, so check terminated (=success) before break
            success = bool(terminated[0])
            break
    closed_frac = float(info.get("closed_fraction", 0.0)) if isinstance(info, dict) else 0.0
    return {"success": success, "steps": step + 1, "closed_fraction": closed_frac}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, required=True,
                   help="Contact threshold (use 999 for inf)")
    p.add_argument("--ckpt_step", type=int, default=100000)
    p.add_argument("--output_dir", default="outputs/threshold_sweep")
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    threshold = float("inf") if args.threshold >= 999 else args.threshold

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = "inf" if threshold == float("inf") else f"{threshold:.1f}"
    result_path = out_dir / f"threshold_{tag}.json"
    if result_path.exists():
        print(f"Already done: {result_path}")
        return

    ckpt_path = os.path.expanduser(
        f"~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-{args.ckpt_step}"
    )

    eps = load_episode_mapping()
    print(f"=== Threshold Sweep: threshold={tag} ===")
    print(f"  ckpt: {args.ckpt_step}, episodes: {len(eps)}")

    mujoco_env = MuJoCoVecEnvDrawer(
        num_envs=1,
        reward_type="sparse",
        max_episode_steps=args.max_steps,
        contact_threshold=threshold,
    )
    wrapper = MuJoCoResidualWrapperDrawer(
        vec_env=mujoco_env,
        groot_checkpoint=ckpt_path,
        policy_device=args.device,
    )
    print("Ready.\n")

    t_start = time.time()
    drawer_stats = {d: {"total": 0, "success": 0} for d in [2, 3, 4]}

    for ei, ep_info in enumerate(eps):
        floor = ep_info["floor"]
        drawer_idx = FLOOR_TO_DRAWER.get(floor)
        if drawer_idx is None:
            continue

        print(f"  ep{ei} d{drawer_idx} ...", end=" ", flush=True)
        result = run_one(wrapper, mujoco_env, drawer_idx, args.max_steps)
        tag_str = "SUCC" if result["success"] else "FAIL"
        print(f"{tag_str} steps={result['steps']} closed={result['closed_fraction']*100:.0f}%")

        ds = drawer_stats[drawer_idx]
        ds["total"] += 1
        if result["success"]:
            ds["success"] += 1

    elapsed = time.time() - t_start

    # Summary
    total_succ = sum(ds["success"] for ds in drawer_stats.values())
    total_ep = sum(ds["total"] for ds in drawer_stats.values())
    total_sr = total_succ / total_ep if total_ep > 0 else 0.0

    summary = {
        "threshold": tag,
        "total_sr": total_sr,
        "total_success": total_succ,
        "total_episodes": total_ep,
        "elapsed_s": round(elapsed, 1),
        "per_drawer": {},
    }
    print(f"\n{'='*50}")
    print(f"Threshold={tag}  Total SR={total_succ}/{total_ep}={total_sr*100:.0f}%")
    for d in [2, 3, 4]:
        ds = drawer_stats[d]
        sr = ds["success"] / ds["total"] if ds["total"] > 0 else 0.0
        summary["per_drawer"][str(d)] = {
            "success": ds["success"], "total": ds["total"], "sr": sr
        }
        print(f"  D{d}: {ds['success']}/{ds['total']}={sr*100:.0f}%")

    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
