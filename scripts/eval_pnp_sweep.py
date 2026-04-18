"""
Sweep eval: iterate over all GR00T PnP checkpoints, evaluate each on easy positions.
Uses file-based locking so multiple SLURM jobs can run independently without conflict.

Usage:
    python scripts/eval_pnp_sweep.py [--max-evals N] [--output-dir DIR]
"""
import sys, os, json, time, importlib, types as _types, argparse, signal
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DS_BUILD_OPS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Deepspeed mock
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

_repo = str(Path(__file__).resolve().parents[1] / "residual-offpolicy-rl")
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import numpy as np
import torch
import psutil
import subprocess
torch.backends.cuda.enable_cudnn_sdp(False)

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

FFMPEG = os.path.expanduser(
    "~/.venvs/groot/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)

TRAINING_BASE = os.path.expanduser("~/DATA/INTERN/training")
POSITIONS_FILE = os.path.join(_repo, "configs/pnp_sim33ep_positions.json")
SCENE_XML = os.path.expanduser(
    "~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
)

# Eval config
NUM_ENVS = 5
NUM_EPISODES = 2  # 2 batches x 5 envs = 10 episodes per checkpoint
MAX_STEPS = 500
SEED = 42

# Graceful shutdown
_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    print(f"\n[SIGNAL] Received {signal.Signals(sig).name}, will finish current eval then exit...")
    _shutdown = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def discover_checkpoints():
    """Scan all GR00T PnP checkpoints and return sorted list."""
    checkpoints = []
    epoch_dirs = sorted(Path(TRAINING_BASE).glob("groot_pnp_sim_*ep"))
    for ep_dir in epoch_dirs:
        ep_name = ep_dir.name  # e.g. groot_pnp_sim_10ep
        ep_num = ep_name.replace("groot_pnp_sim_", "").replace("ep", "")
        for ckpt_dir in sorted(ep_dir.glob("checkpoint-*")):
            step_num = int(ckpt_dir.name.replace("checkpoint-", ""))
            checkpoints.append({
                "epoch": ep_name,
                "ep_num": int(ep_num),
                "step": step_num,
                "path": str(ckpt_dir),
                "key": f"{ep_name}/checkpoint-{step_num}",
            })
    # Sort by epoch num, then step
    checkpoints.sort(key=lambda c: (c["ep_num"], c["step"]))
    return checkpoints


def try_lock(lock_path):
    """Try to acquire lock via mkdir (atomic on NFS). Returns lock_path if acquired, None otherwise."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    try:
        os.mkdir(lock_path)  # atomic on NFS — fails if already exists
        # Write info file inside lock dir
        with open(os.path.join(lock_path, "info"), "w") as f:
            f.write(f"pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID','N/A')} time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return lock_path
    except OSError:
        # Check for stale lock (older than 30 min)
        try:
            info_file = os.path.join(lock_path, "info")
            if os.path.exists(info_file) and time.time() - os.path.getmtime(info_file) > 1800:
                import shutil
                shutil.rmtree(lock_path, ignore_errors=True)
                os.mkdir(lock_path)
                with open(os.path.join(lock_path, "info"), "w") as f:
                    f.write(f"pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID','N/A')} time={time.strftime('%Y-%m-%d %H:%M:%S')} (stale reclaim)\n")
                return lock_path
        except OSError:
            pass
        return None


def release_lock(fh_or_path, lock_path=None):
    """Release mkdir-based lock."""
    path = lock_path if lock_path else fh_or_path
    try:
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    except:
        pass


def get_mem_usage():
    proc = psutil.Process(os.getpid())
    ram_gb = proc.memory_info().rss / (1024**3)
    try:
        gpu_mb = torch.cuda.memory_allocated() / (1024**2)
    except:
        gpu_mb = 0
    return ram_gb, gpu_mb


def eval_checkpoint(ckpt_path, positions, output_dir):
    """Evaluate a single checkpoint. Returns result dict."""
    num_envs = len(positions)
    cube_positions = [p["cube_pos"] for p in positions]
    bowl_positions = [p["bowl_pos"] for p in positions]
    cube_yaws = [p.get("cube_yaw_deg", 0.0) for p in positions]

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ram, gpu = get_mem_usage()
    print(f"  [MEM baseline] RAM={ram:.1f}GB GPU={gpu:.0f}MB")

    # Create env
    t0 = time.time()
    vec_env = MuJoCoVecEnvPnP(
        num_envs=num_envs,
        cube_positions=cube_positions,
        bowl_positions=bowl_positions,
        max_episode_steps=MAX_STEPS,
        reward_type="dense",
        scene_xml=SCENE_XML,
    )

    # Create wrapper (loads GR00T)
    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=ckpt_path,
        task_description="Pick up the red cube and place it onto the plate.",
        ema_alpha=0.0,
    )
    ram, gpu = get_mem_usage()
    print(f"  [MEM loaded] RAM={ram:.1f}GB GPU={gpu:.0f}MB ({time.time()-t0:.1f}s)")

    from scipy.spatial.transform import Rotation
    import mujoco

    all_results = []
    total_success = 0
    total_episodes = 0
    last_ep_frames = None

    for ep in range(NUM_EPISODES):
        obs = wrapper.reset()

        # Override positions
        for ei in range(num_envs):
            env_data = vec_env._envs[ei]
            data, model = env_data["data"], env_data["model"]
            cqa = env_data["cube_qposadr"]
            if cqa is not None:
                data.qpos[cqa:cqa+3] = cube_positions[ei]
                q_xyzw = Rotation.from_euler("z", np.radians(cube_yaws[ei])).as_quat()
                data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            bowl_body_id = env_data["bowl_body_id"]
            if bowl_body_id >= 0:
                model.body_pos[bowl_body_id] = bowl_positions[ei]
            env_data["cube_pos_init"] = np.array(cube_positions[ei])
            env_data["bowl_pos_init"] = np.array(bowl_positions[ei])
            mujoco.mj_forward(model, data)

        vec_env._step_counts = np.zeros(num_envs, dtype=int)
        vec_env._episode_rewards = np.zeros(num_envs, dtype=np.float64)

        # Rebuild obs
        obs = wrapper.reset()
        for ei in range(num_envs):
            env_data = vec_env._envs[ei]
            data, model = env_data["data"], env_data["model"]
            cqa = env_data["cube_qposadr"]
            if cqa is not None:
                data.qpos[cqa:cqa+3] = cube_positions[ei]
                q_xyzw = Rotation.from_euler("z", np.radians(cube_yaws[ei])).as_quat()
                data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            bowl_body_id = env_data["bowl_body_id"]
            if bowl_body_id >= 0:
                model.body_pos[bowl_body_id] = bowl_positions[ei]
            env_data["cube_pos_init"] = np.array(cube_positions[ei])
            env_data["bowl_pos_init"] = np.array(bowl_positions[ei])
            mujoco.mj_forward(model, data)

        vec_env._step_counts = np.zeros(num_envs, dtype=int)
        vec_env._episode_rewards = np.zeros(num_envs, dtype=np.float64)

        # Step loop
        env_done = [False] * num_envs
        env_success = [False] * num_envs
        env_min_dist = [999.0] * num_envs
        ep_frames = [[] for _ in range(num_envs)]

        for step in range(MAX_STEPS):
            action = np.zeros((num_envs, wrapper.action_dim))
            obs, rew, term, trunc, info = wrapper.step(action)

            if step % 3 == 0:
                for ei in range(num_envs):
                    if not env_done[ei]:
                        frame = vec_env.get_frame(ei, camera='back', size=(240, 320))
                        ep_frames[ei].append(frame)

            done = term | trunc if hasattr(term, '__or__') else np.array(term) | np.array(trunc)

            for ei in range(num_envs):
                if env_done[ei]:
                    continue
                cqa = vec_env._envs[ei]["cube_qposadr"]
                if cqa is not None:
                    cp = vec_env._envs[ei]["data"].qpos[cqa:cqa+3].copy()
                    d = np.linalg.norm(cp[:2] - np.array(bowl_positions[ei][:2]))
                    env_min_dist[ei] = min(env_min_dist[ei], d)
                    if d < 0.095 and cp[2] < 0.08:
                        env_success[ei] = True

                d_val = done[ei] if hasattr(done, '__getitem__') else done
                if d_val:
                    env_done[ei] = True
                    t_val = term[ei] if hasattr(term, '__getitem__') else term
                    if t_val:
                        env_success[ei] = True

            if all(env_done):
                break

        n_succ = sum(env_success)
        total_success += n_succ
        total_episodes += num_envs
        print(f"    Batch {ep+1}: {n_succ}/{num_envs} success")
        for ei in range(num_envs):
            tag = "OK" if env_success[ei] else "FAIL"
            all_results.append({
                "batch": ep, "env": ei, "episode": positions[ei]["episode"],
                "success": env_success[ei], "min_dist": env_min_dist[ei],
            })

        if ep == NUM_EPISODES - 1:
            last_ep_frames = ep_frames

    sr = total_success / total_episodes if total_episodes > 0 else 0
    print(f"    SR: {total_success}/{total_episodes} = {sr:.0%}")

    # Save video
    video_path = os.path.join(output_dir, "eval_grid.mp4")
    if imageio is not None and last_ep_frames is not None:
        max_len = max(len(f) for f in last_ep_frames)
        grid_frames = []
        for t in range(max_len):
            row = []
            for ei in range(num_envs):
                if t < len(last_ep_frames[ei]):
                    row.append(last_ep_frames[ei][t])
                elif last_ep_frames[ei]:
                    row.append(last_ep_frames[ei][-1])
            if row:
                grid_frames.append(np.concatenate(row, axis=1))

        tmp_path = video_path + ".tmp.mp4"
        imageio.mimwrite(tmp_path, grid_frames, fps=15, macro_block_size=1)
        subprocess.run([
            FFMPEG, "-y", "-i", tmp_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-tag:v", "avc1", "-crf", "18",
            video_path
        ], capture_output=True)
        os.remove(tmp_path)
        print(f"    Video: {video_path}")

    # Cleanup GPU memory
    del wrapper, vec_env
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return {
        "checkpoint": ckpt_path,
        "success_rate": sr,
        "total_success": total_success,
        "total_episodes": total_episodes,
        "max_steps": MAX_STEPS,
        "num_envs": NUM_ENVS,
        "num_episode_batches": NUM_EPISODES,
        "episodes": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep eval over GR00T PnP checkpoints")
    parser.add_argument("--output-dir", default=os.path.expanduser("~/Repos/Intern/outputs/pnp_eval_sweep"))
    parser.add_argument("--max-evals", type=int, default=0, help="Max checkpoints to eval (0=all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load positions (easy = first 5)
    with open(POSITIONS_FILE) as f:
        all_pos = json.load(f)
    positions = all_pos[:5]

    print(f"{'='*70}")
    print(f"PnP Eval Sweep — Easy (5 positions), {MAX_STEPS} steps, {NUM_EPISODES} batches")
    print(f"Output: {args.output_dir}")
    print(f"PID: {os.getpid()}, SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"{'='*70}")

    evals_done = 0

    while True:
        if _shutdown:
            print("[SWEEP] Shutdown requested, exiting loop.")
            break

        # Discover and filter
        checkpoints = discover_checkpoints()
        pending = []
        for ckpt in checkpoints:
            result_path = os.path.join(args.output_dir, ckpt["key"], "result.json")
            lock_path = os.path.join(args.output_dir, ckpt["key"], ".lock")
            if os.path.exists(result_path):
                continue  # already evaluated
            pending.append(ckpt)

        if not pending:
            print("[SWEEP] All checkpoints evaluated! Exiting.")
            break

        print(f"\n[SCAN] {len(pending)}/{len(checkpoints)} checkpoints remaining")

        # Try to lock one
        acquired = None
        lock_fh = None
        for ckpt in pending:
            lock_path = os.path.join(args.output_dir, ckpt["key"], ".lock")
            fh = try_lock(lock_path)
            if fh is not None:
                acquired = ckpt
                lock_fh = fh
                break

        if acquired is None:
            print("[SWEEP] All pending checkpoints are locked by other jobs. Waiting 30s...")
            time.sleep(30)
            continue

        ckpt = acquired
        ckpt_output = os.path.join(args.output_dir, ckpt["key"])
        os.makedirs(ckpt_output, exist_ok=True)
        lock_path = os.path.join(ckpt_output, ".lock")

        print(f"\n{'='*70}")
        print(f"[EVAL] {ckpt['key']}")
        print(f"  Path: {ckpt['path']}")
        print(f"{'='*70}")

        try:
            t_start = time.time()
            result = eval_checkpoint(ckpt["path"], positions, ckpt_output)
            elapsed = time.time() - t_start
            result["elapsed_seconds"] = elapsed
            result["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "N/A")
            result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # Save result
            result_path = os.path.join(ckpt_output, "result.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Result saved: {result_path} ({elapsed:.0f}s)")

        except Exception as e:
            print(f"  [ERROR] {ckpt['key']}: {e}")
            import traceback; traceback.print_exc()
        finally:
            release_lock(lock_fh, lock_path)

        evals_done += 1
        if args.max_evals > 0 and evals_done >= args.max_evals:
            print(f"[SWEEP] Reached max-evals={args.max_evals}, exiting.")
            break

        ram, gpu = get_mem_usage()
        print(f"[MEM between evals] RAM={ram:.1f}GB GPU={gpu:.0f}MB")

    # Print summary
    print(f"\n{'='*70}")
    print("SWEEP SUMMARY")
    print(f"{'='*70}")
    checkpoints = discover_checkpoints()
    for ckpt in checkpoints:
        result_path = os.path.join(args.output_dir, ckpt["key"], "result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            print(f"  {ckpt['key']:50s} SR={r['success_rate']:.0%} ({r['total_success']}/{r['total_episodes']})")
        else:
            print(f"  {ckpt['key']:50s} [pending]")


if __name__ == "__main__":
    main()
