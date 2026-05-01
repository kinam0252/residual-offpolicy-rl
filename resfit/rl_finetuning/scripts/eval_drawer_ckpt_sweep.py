"""Evaluate GR00T close-drawer checkpoints at fixed cabinet pose.

For each checkpoint, run 10 episodes per drawer (2, 3, 4) with zero
residual (base policy only).  Uses work-stealing with directory locks
so multiple SLURM workers can share the workload.

Work items: 11 ckpts × 3 drawers × 10 trials = 330

Layout:
    outputs/ckpt_sweep_drawer/
      ckpt_100000/
        d2_t03.json       # drawer 2, trial 3 result
        d2_t03.lock/      # dir-lock while running
      ...

Usage:
    python eval_drawer_ckpt_sweep.py [--output_dir outputs/ckpt_sweep_drawer]
"""
import sys, os, json, time, argparse, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

import torch
torch.backends.cuda.enable_cudnn_sdp(False)
import importlib, types as _types
if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed"); _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__ = []; _ds.__file__ = __file__
    _dz = _types.ModuleType("deepspeed.zero")
    _dz.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _dz.Init = lambda *a, **kw: (lambda f: f); _ds.zero = _dz
    sys.modules["deepspeed"] = _ds; sys.modules["deepspeed.zero"] = _dz
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except Exception:
    pass

import numpy as np
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, DRAWER_SLIDE, ACTIVE_DRAWERS,
    WALL_THICK, DRAWER_SLOT_HEIGHT,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import (
    MuJoCoResidualWrapperDrawer,
)

# ── Config ──
CKPT_BASE = os.path.expanduser("~/DATA/INTERN/training/groot_drawer_sim_33ep")
CKPT_STEPS = [25000, 50000, 75000, 100000, 125000, 150000,
              175000, 200000, 225000, 250000, 275000]

DRAWERS = ACTIVE_DRAWERS  # [2, 3, 4]
NUM_TRIALS = 10
MAX_STEPS = 500


def try_lock(lock_path: Path) -> bool:
    try:
        lock_path.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        return False


def run_one(wrapper, mujoco_env, drawer_idx, max_steps):
    """Run one episode for a specific drawer. Returns result dict."""
    mujoco_env._active_drawers[0] = drawer_idx
    env_data = mujoco_env._envs[0]
    env_data["active_drawer"] = drawer_idx

    slot_h = DRAWER_SLOT_HEIGHT
    dz = WALL_THICK + slot_h / 2 + drawer_idx * (slot_h + WALL_THICK)
    env_data["drawer_z_center"] = dz
    env_data["drawer_z_min"] = dz - slot_h / 2 - 0.02
    env_data["drawer_z_max"] = dz + slot_h / 2 + 0.02

    obs, _ = wrapper.reset()

    t0 = time.time()
    success = False
    slides = []
    for step in range(max_steps):
        residual = torch.zeros((1, 7), dtype=torch.float32)
        next_obs, reward, terminated, truncated, info = wrapper.step(residual)
        obs = next_obs

        slide = float(mujoco_env._drawer_qpos[0])
        slides.append(slide)

        if terminated[0] or truncated[0]:
            if terminated[0]:
                success = True
            break

    elapsed = time.time() - t0
    n_steps = step + 1

    min_slide = min(slides) if slides else DRAWER_SLIDE
    closed_frac = 1.0 - (min_slide / DRAWER_SLIDE) if DRAWER_SLIDE > 0 else 0.0

    return {
        "success": success,
        "steps": n_steps,
        "time_s": round(elapsed, 1),
        "drawer": drawer_idx,
        "min_slide": round(min_slide, 4),
        "closed_fraction": round(closed_frac, 4),
        "final_slide": round(slides[-1], 4) if slides else DRAWER_SLIDE,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/ckpt_sweep_drawer")
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    return p.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    total = len(CKPT_STEPS) * len(DRAWERS) * NUM_TRIALS
    print(f"Checkpoint sweep: {len(CKPT_STEPS)} ckpts × {len(DRAWERS)} drawers × {NUM_TRIALS} trials = {total} items")

    # Build work items: (ckpt_step, drawer, trial)
    all_work = [
        (ckpt, d, t)
        for ckpt in CKPT_STEPS
        for d in DRAWERS
        for t in range(NUM_TRIALS)
    ]

    mujoco_env = None
    wrapper = None
    current_ckpt = None
    completed = 0

    while True:
        claimed = False
        for ckpt_step, drawer_idx, trial in all_work:
            ckpt_tag = f"ckpt_{ckpt_step:06d}"
            ckpt_dir = out_base / ckpt_tag
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            result_path = ckpt_dir / f"d{drawer_idx}_t{trial:02d}.json"
            lock_path = ckpt_dir / f"d{drawer_idx}_t{trial:02d}.lock"

            if result_path.exists():
                continue
            if not try_lock(lock_path):
                continue

            claimed = True
            ckpt_path = os.path.join(CKPT_BASE, f"checkpoint-{ckpt_step}")

            # Load new checkpoint if needed
            if current_ckpt != ckpt_step:
                print(f"\n{'='*60}")
                print(f"Loading checkpoint: {ckpt_step}")
                print(f"{'='*60}")
                if wrapper is not None:
                    del wrapper
                    torch.cuda.empty_cache()
                if mujoco_env is None:
                    mujoco_env = MuJoCoVecEnvDrawer(
                        num_envs=1,
                        reward_type="sparse",
                        max_episode_steps=args.max_steps,
                    )
                wrapper = MuJoCoResidualWrapperDrawer(
                    vec_env=mujoco_env,
                    groot_checkpoint=ckpt_path,
                    policy_device="cuda:0",
                )
                current_ckpt = ckpt_step
                print("Ready.\n")

            print(f"  [{ckpt_tag}] d{drawer_idx} trial={trial} ...", end=" ", flush=True)

            try:
                result = run_one(wrapper, mujoco_env, drawer_idx, args.max_steps)
                result["checkpoint"] = ckpt_step
                result["trial"] = trial

                tmp = ckpt_dir / f"d{drawer_idx}_t{trial:02d}.tmp.json"
                with open(tmp, "w") as f:
                    json.dump(result, f, indent=2)
                tmp.rename(result_path)

                tag = "SUCC" if result["success"] else "FAIL"
                print(f"{tag} steps={result['steps']} closed={result['closed_fraction']*100:.0f}%")
                completed += 1
            except Exception as e:
                import traceback
                print(f"ERROR: {e}")
                traceback.print_exc()
            finally:
                shutil.rmtree(lock_path, ignore_errors=True)

            break

        if not claimed:
            break

    print(f"\nWorker done. Completed: {completed}")
    print_summary(out_base)


def print_summary(out_base: Path):
    """Print SR table: checkpoints × drawers."""
    results = []
    for p in out_base.rglob("d*_t*.json"):
        if p.name.endswith(".tmp.json"):
            continue
        with open(p) as f:
            results.append(json.load(f))

    if not results:
        print("No results found.")
        return

    total = len(results)
    total_succ = sum(1 for r in results if r["success"])
    print(f"\n{'='*80}")
    print(f"Total: {total_succ}/{total} = {total_succ/total*100:.1f}% SR")
    print(f"{'='*80}")

    # Header
    hdr = f"{'Checkpoint':>12} | {'Overall':>8}"
    for d in DRAWERS:
        hdr += f" | {'D'+str(d):>8}"
    hdr += f" | {'Avg Closed%':>11}"
    print(hdr)
    print("-" * len(hdr))

    for ckpt_step in CKPT_STEPS:
        ckpt_rs = [r for r in results if r["checkpoint"] == ckpt_step]
        if not ckpt_rs:
            print(f"{ckpt_step:>12} |     --")
            continue

        n = len(ckpt_rs)
        sr = sum(1 for r in ckpt_rs if r["success"]) / n * 100
        avg_closed = np.mean([r["closed_fraction"] for r in ckpt_rs]) * 100

        line = f"{ckpt_step:>12} | {sr:6.1f}%"
        for d in DRAWERS:
            dr = [r for r in ckpt_rs if r["drawer"] == d]
            if dr:
                dsr = sum(1 for r in dr if r["success"]) / len(dr) * 100
                line += f" |  {dsr:5.1f}%"
            else:
                line += f" |     --"
        line += f" |     {avg_closed:5.1f}%"
        print(line)

    print(f"{'='*80}")

    # Per-drawer summary across all ckpts
    print(f"\nPer-drawer overall:")
    for d in DRAWERS:
        dr = [r for r in results if r["drawer"] == d]
        if dr:
            sr = sum(1 for r in dr if r["success"]) / len(dr) * 100
            print(f"  Drawer {d}: {sr:.1f}% ({sum(1 for r in dr if r['success'])}/{len(dr)})")


if __name__ == "__main__":
    main()
