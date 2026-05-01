"""Evaluate GR00T close-drawer base policy across a grid of cabinet poses.

Sweeps cabinet position (dx, dy) and rotation (dθ) to find where the
base policy starts to fail.  Uses work-stealing with directory locks so
multiple SLURM workers can run in parallel.

Pose grid:
    dx ∈ [-0.04, -0.02, 0, +0.02, +0.04]   (world X offset, m)
    dy ∈ [-0.04, -0.02, 0, +0.02, +0.04]   (world Y offset, m)
    dθ ∈ [-15, -10, -5, 0, +5, +10, +15]    (euler-z offset, deg)
    → 5 × 5 × 7 = 175 poses
    × 3 drawers (2, 3, 4) = 525 work items

Work-stealing layout:
    outputs/pose_sweep_drawer/
      dx+0.02_dy-0.04_dth+05/
        d2.json           # completed result
        d2.lock/          # dir-lock while running
        d3.json
      ...

Usage:
    python eval_drawer_pose_sweep.py [--output_dir outputs/pose_sweep_drawer]
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
import mujoco
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, DRAWER_HOME_QPOS, DRAWER_SLIDE, FLOOR_TO_DRAWER,
    ACTIVE_DRAWERS, get_tcp_pose, get_drawer_pos,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import (
    MuJoCoResidualWrapperDrawer,
)

# ── Sweep parameters ──
DX_VALUES = [-0.04, -0.02, 0.0, 0.02, 0.04]
DY_VALUES = [-0.04, -0.02, 0.0, 0.02, 0.04]
DTHETA_VALUES = [-15, -10, -5, 0, 5, 10, 15]

BASE_POS = np.array([0.5177, 0.2974, 0.005])
BASE_EULER = np.array([0.0, 0.0, -450.0])

DRAWERS = ACTIVE_DRAWERS  # [2, 3, 4]

# Checkpoint
CKPT_STEP = 100000
CKPT_PATH = os.path.expanduser(
    f"~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-{CKPT_STEP}"
)

MAX_STEPS = 500


# ── Pose grid helpers ──

def make_pose_grid():
    """Return list of {dx, dy, dtheta, pose_id, cabinet_pos, cabinet_euler}."""
    grid = []
    for dx in DX_VALUES:
        for dy in DY_VALUES:
            for dth in DTHETA_VALUES:
                pose_id = f"dx{dx:+.2f}_dy{dy:+.2f}_dth{dth:+03d}"
                pos = BASE_POS + np.array([dx, dy, 0.0])
                euler = BASE_EULER + np.array([0.0, 0.0, dth])
                grid.append({
                    "dx": dx, "dy": dy, "dtheta": dth,
                    "pose_id": pose_id,
                    "cabinet_pos": pos.tolist(),
                    "cabinet_euler": euler.tolist(),
                })
    return grid


def pose_tag(dx, dy, dth):
    return f"dx{dx:+.2f}_dy{dy:+.2f}_dth{dth:+03d}"


# ── Lock helpers ──

def try_lock(lock_path: Path) -> bool:
    try:
        lock_path.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        return False


# ── Eval one episode ──

def run_one(wrapper, mujoco_env, drawer_idx, max_steps):
    """Run one episode for a specific drawer. Returns result dict."""
    mujoco_env._active_drawers[0] = drawer_idx
    env_data = mujoco_env._envs[0]
    env_data["active_drawer"] = drawer_idx

    # Recompute drawer z-range
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
        WALL_THICK, DRAWER_SLOT_HEIGHT,
    )
    slot_h = DRAWER_SLOT_HEIGHT
    dz = WALL_THICK + slot_h / 2 + drawer_idx * (slot_h + WALL_THICK)
    env_data["drawer_z_center"] = dz
    env_data["drawer_z_min"] = dz - slot_h / 2 - 0.02
    env_data["drawer_z_max"] = dz + slot_h / 2 + 0.02

    obs, _ = wrapper.reset()

    # Rollout with zero residual (base policy only)
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
    n_steps = step + 1 if 'step' in dir() else 0

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


# ── Main ──

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/pose_sweep_drawer")
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    return p.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    grid = make_pose_grid()
    total = len(grid) * len(DRAWERS)
    print(f"Pose sweep: {len(grid)} poses × {len(DRAWERS)} drawers = {total} work items")
    print(f"Checkpoint: {CKPT_PATH}")
    print()

    # Build work items: (pose_dict, drawer_idx)
    all_work = [(p, d) for p in grid for d in DRAWERS]

    mujoco_env = None
    wrapper = None
    current_pose_id = None
    completed = 0

    while True:
        claimed = False
        for pose, drawer_idx in all_work:
            pose_id = pose["pose_id"]
            pose_dir = out_base / pose_id
            pose_dir.mkdir(parents=True, exist_ok=True)

            result_path = pose_dir / f"d{drawer_idx}.json"
            lock_path = pose_dir / f"d{drawer_idx}.lock"

            if result_path.exists():
                continue
            if not try_lock(lock_path):
                continue

            # Claimed this work item
            claimed = True

            # Recreate env if pose changed
            if current_pose_id != pose_id:
                if wrapper is not None:
                    del wrapper
                    torch.cuda.empty_cache()
                if mujoco_env is not None:
                    del mujoco_env

                cab_pos = np.array(pose["cabinet_pos"])
                cab_euler = np.array(pose["cabinet_euler"])

                mujoco_env = MuJoCoVecEnvDrawer(
                    num_envs=1,
                    active_drawers=[drawer_idx],
                    cabinet_pos=cab_pos,
                    cabinet_euler=cab_euler,
                    reward_type="sparse",
                    max_episode_steps=args.max_steps,
                )
                wrapper = MuJoCoResidualWrapperDrawer(
                    vec_env=mujoco_env,
                    groot_checkpoint=CKPT_PATH,
                    policy_device="cuda:0",
                )
                current_pose_id = pose_id
                print(f"\n  New pose: {pose_id}")

            print(f"  [{pose_id}] drawer={drawer_idx} ...", end=" ", flush=True)

            try:
                result = run_one(wrapper, mujoco_env, drawer_idx, args.max_steps)
                result["pose_id"] = pose_id
                result["dx"] = pose["dx"]
                result["dy"] = pose["dy"]
                result["dtheta"] = pose["dtheta"]
                result["checkpoint"] = CKPT_STEP

                tmp = pose_dir / f"d{drawer_idx}.tmp.json"
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

            break  # re-scan for next unclaimed

        if not claimed:
            break

    print(f"\nWorker done. Completed: {completed}")
    print_summary(out_base)


def print_summary(out_base: Path):
    """Print summary table grouped by dθ, showing dx×dy SR grid."""
    results = []
    for p in out_base.rglob("d*.json"):
        if p.name.endswith(".tmp.json"):
            continue
        with open(p) as f:
            results.append(json.load(f))

    if not results:
        print("No results found.")
        return

    total = len(results)
    total_succ = sum(1 for r in results if r["success"])
    print(f"\n{'='*70}")
    print(f"Total: {total_succ}/{total} = {total_succ/total*100:.1f}% SR")
    print(f"{'='*70}")

    # Group by dtheta
    from collections import defaultdict
    by_dtheta = defaultdict(list)
    for r in results:
        by_dtheta[r["dtheta"]].append(r)

    for dth in sorted(by_dtheta.keys()):
        rs = by_dtheta[dth]
        print(f"\n  dθ = {dth:+d}°  ({sum(1 for r in rs if r['success'])}/{len(rs)} SR)")

        # dx × dy grid
        sr_grid = {}
        for r in rs:
            key = (r["dx"], r["dy"])
            if key not in sr_grid:
                sr_grid[key] = {"n": 0, "succ": 0}
            sr_grid[key]["n"] += 1
            if r["success"]:
                sr_grid[key]["succ"] += 1

        # Print header
        header = "dx\\dy"
        print(f"  {header:>8}", end="")
        for dy in DY_VALUES:
            print(f"  {dy:+.02f}", end="")
        print()
        print(f"  {'':>8}", end="")
        for _ in DY_VALUES:
            print(f"  -----", end="")
        print()

        for dx in DX_VALUES:
            print(f"  {dx:+.02f}  ", end="")
            for dy in DY_VALUES:
                key = (dx, dy)
                if key in sr_grid:
                    s = sr_grid[key]
                    pct = s["succ"] / s["n"] * 100
                    print(f"  {pct:4.0f}%", end="")
                else:
                    print(f"    --", end="")
            print()

    # Per-drawer breakdown
    print(f"\n  Per-drawer SR:")
    for d in DRAWERS:
        dr = [r for r in results if r["drawer"] == d]
        if dr:
            sr = sum(1 for r in dr if r["success"]) / len(dr) * 100
            print(f"    Drawer {d}: {sr:.1f}% ({sum(1 for r in dr if r['success'])}/{len(dr)})")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
