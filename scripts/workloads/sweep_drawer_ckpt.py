"""
Sweep GR00T drawer base policy checkpoints — measure SR per (ckpt, drawer).

Work-stealing design: multiple SLURM jobs share one output directory.
Each job claims (ckpt, drawer) combos via atomic lockfiles.

Usage:
    python scripts/workloads/sweep_drawer_ckpt.py \
        --ckpt_dir checkpoints/<TASK>/checkpoint \
        --output_dir outputs/sweep_drawer_ckpt \
        --episodes 3
"""
import sys, os, json, time, argparse, signal, importlib, types as _types
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DS_BUILD_OPS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if "deepspeed" not in sys.modules:
    m = _types.ModuleType("deepspeed"); m.__version__ = "0.0.0"
    m.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None); m.__path__ = []; m.__file__ = "fake"
    z = _types.ModuleType("deepspeed.zero"); z.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    z.Init = lambda *a, **kw: (lambda f: f); m.zero = z
    sys.modules["deepspeed"] = m; sys.modules["deepspeed.zero"] = z

try:
    import huggingface_hub.utils._validators as v; o = v.validate_repo_id
    def _p(r):
        if r and (r.startswith("/") or r.startswith(".")): return
        return o(r)
    v.validate_repo_id = _p
except: pass

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer, DRAWER_SLIDE
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified

_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    print(f"\n[SIGNAL] {signal.Signals(sig).name} — finishing current task then exit...", flush=True)
    _shutdown = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ── Lock-based work claiming ──

def claim_task(out_dir: Path, tasks: list[tuple[str, int]], job_id: str) -> tuple[str, int] | None:
    """Try to claim one (ckpt_name, drawer_id) task. Returns None if all done/claimed."""
    now = time.time()
    for ckpt_name, drawer_id in tasks:
        tag = f"{ckpt_name}_D{drawer_id}"
        done_path = out_dir / f"{tag}.done"
        lock_path = out_dir / f"{tag}.lock"
        if done_path.exists():
            continue
        if lock_path.exists():
            try:
                age = now - lock_path.stat().st_mtime
                if age < 3600:  # 1hr timeout
                    continue
                lock_path.unlink(missing_ok=True)
            except OSError:
                continue
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{job_id}\n".encode())
            os.close(fd)
            return (ckpt_name, drawer_id)
        except FileExistsError:
            continue
    return None


def mark_done(out_dir: Path, ckpt_name: str, drawer_id: int, result: dict):
    tag = f"{ckpt_name}_D{drawer_id}"
    result_path = out_dir / f"{tag}.json"
    result_path.write_text(json.dumps(result, indent=2))
    done_path = out_dir / f"{tag}.done"
    done_path.write_text(f"{os.environ.get('SLURM_JOB_ID', os.getpid())}\n")
    lock_path = out_dir / f"{tag}.lock"
    lock_path.unlink(missing_ok=True)


def release_locks(out_dir: Path, job_id: str):
    for lf in out_dir.glob("*.lock"):
        try:
            if lf.read_text().strip() == job_id:
                lf.unlink(missing_ok=True)
        except:
            pass


# ── Eval function ──

def eval_drawer(ckpt_path: str, drawer_id: int, episodes: int,
                max_steps: int, device: str, physics_drawer: bool) -> dict:
    """Run base-only eval, return {sr, successes, episodes, dq_finals}."""
    vec_env = MuJoCoVecEnvDrawer(
        num_envs=1,
        active_drawers=[drawer_id],
        max_episode_steps=max_steps,
        reward_type="delta",
        device=device,
        rl_img_size=84,
        parallel_envs=False,
        physics_drawer=physics_drawer,
    )
    wrapper = MuJoCoResidualWrapperUnified(
        vec_env=vec_env,
        groot_checkpoint=ckpt_path,
        task_description="Close the drawer.",
        chunk_sync=True,
        async_prefetch=False,
        policy_device=device,
    )
    zero_act = torch.zeros(1, 7, device=device)

    successes = 0
    dq_finals = []

    for ep in range(episodes):
        obs, info = wrapper.reset()
        success = False
        for step in range(max_steps):
            obs, reward, terminated, truncated, _ = wrapper.step(zero_act)
            if terminated[0]:
                success = True
                break
            if truncated[0]:
                break
        dq = float(vec_env._drawer_qpos[0])
        dq_finals.append(dq)
        if success:
            successes += 1
        print(f"    ep{ep}: {'OK' if success else 'FAIL'} dq={dq:.4f}", flush=True)

    del wrapper, vec_env
    torch.cuda.empty_cache()

    return {
        "sr": successes / episodes,
        "successes": successes,
        "episodes": episodes,
        "dq_finals": dq_finals,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep drawer base policy checkpoints")
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoints/<TASK>/checkpoint")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/sweep_drawer_ckpt")
    parser.add_argument("--drawers", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--physics_drawer", action="store_true", default=True)
    parser.add_argument("--no_physics_drawer", dest="physics_drawer", action="store_false")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover checkpoints
    ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    ckpt_names = [c.name for c in ckpts]
    print(f"Found {len(ckpts)} checkpoints: {[c.name for c in ckpts]}")
    print(f"Drawers: {args.drawers}, Episodes: {args.episodes}")
    print(f"Physics drawer: {args.physics_drawer}")

    # Build task list
    tasks = [(cn, d) for cn in ckpt_names for d in args.drawers]
    print(f"Total tasks: {len(tasks)}")

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    completed = 0

    try:
        while not _shutdown:
            task = claim_task(out_dir, tasks, job_id)
            if task is None:
                print("All tasks claimed/done. Exiting.")
                break

            ckpt_name, drawer_id = task
            ckpt_path = str(ckpt_dir / ckpt_name)
            print(f"\n[{job_id}] Running {ckpt_name} D{drawer_id} ...", flush=True)

            result = eval_drawer(
                ckpt_path=ckpt_path,
                drawer_id=drawer_id,
                episodes=args.episodes,
                max_steps=args.max_steps,
                device=args.device,
                physics_drawer=args.physics_drawer,
            )
            result["checkpoint"] = ckpt_name
            result["drawer"] = drawer_id
            result["physics_drawer"] = args.physics_drawer

            mark_done(out_dir, ckpt_name, drawer_id, result)
            print(f"  => {ckpt_name} D{drawer_id}: SR={result['sr']*100:.0f}%", flush=True)
            completed += 1

    finally:
        release_locks(out_dir, job_id)
        print(f"\n[{job_id}] Completed {completed} tasks.")

    # Print summary if all done
    print_summary(out_dir, ckpt_names, args.drawers)


def print_summary(out_dir: Path, ckpt_names: list[str], drawers: list[int]):
    """Print summary table from collected results."""
    results = {}
    for jf in out_dir.glob("*.json"):
        r = json.loads(jf.read_text())
        results[(r["checkpoint"], r["drawer"])] = r

    if not results:
        return

    # Header
    header = f"{'Checkpoint':<20}" + "".join(f"  D{d:>2}" for d in drawers) + "  Avg"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for cn in ckpt_names:
        row = f"{cn:<20}"
        srs = []
        for d in drawers:
            r = results.get((cn, d))
            if r:
                sr = r["sr"] * 100
                row += f"  {sr:>3.0f}%"
                srs.append(sr)
            else:
                row += f"    -"
        if srs:
            row += f"  {np.mean(srs):>4.0f}%"
        print(row)

    print(f"{'='*len(header)}")


if __name__ == "__main__":
    main()
