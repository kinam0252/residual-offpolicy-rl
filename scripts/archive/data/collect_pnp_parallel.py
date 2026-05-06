"""
Parallel offline data collection with dense_v3 reward.

Work-stealing design: multiple SLURM jobs share one output directory.
Each job claims uncollected positions via atomic lockfiles, collects
episodes in 15-env batches, then claims more until all positions are done.

Usage:
    python scripts/collect_pnp_parallel.py \
        --positions_file configs/pnp_66ep_positions.json \
        --groot_checkpoint ~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000 \
        --output_dir outputs/offline_data/pnp_66ep_dense_v3 \
        --episodes_per_pos 10 --batch_size 15 --reward_type dense_v3
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

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation

torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

SCENE_XML = os.path.expanduser(
    "~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
)

_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    print(f"\n[SIGNAL] {signal.Signals(sig).name} — finishing current batch then exit...", flush=True)
    _shutdown = True


# ── Position claiming (filesystem-based) ──────────────────────────

def claim_positions(out_dir: Path, all_positions: list, batch_size: int,
                    job_id: str) -> list[int]:
    """Atomically claim up to batch_size uncollected positions.

    A position is 'done' if pos_{idx:03d}.done exists.
    A position is 'claimed' if pos_{idx:03d}.lock exists.
    Stale locks (>2 hours) are reclaimed.
    """
    claimed = []
    now = time.time()
    for idx in range(len(all_positions)):
        if len(claimed) >= batch_size:
            break
        done_path = out_dir / f"pos_{idx:03d}.done"
        lock_path = out_dir / f"pos_{idx:03d}.lock"
        if done_path.exists():
            continue
        # Check for stale lock (>2h)
        if lock_path.exists():
            try:
                age = now - lock_path.stat().st_mtime
                if age < 7200:
                    continue
                # Stale — remove and reclaim
                lock_path.unlink(missing_ok=True)
            except OSError:
                continue
        # Atomic claim
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{job_id}\n".encode())
            os.close(fd)
            claimed.append(idx)
        except FileExistsError:
            continue
    return claimed


def mark_done(out_dir: Path, idx: int):
    """Mark a position as fully collected."""
    done_path = out_dir / f"pos_{idx:03d}.done"
    done_path.write_text(f"{os.environ.get('SLURM_JOB_ID', os.getpid())}\n")
    lock_path = out_dir / f"pos_{idx:03d}.lock"
    lock_path.unlink(missing_ok=True)


def release_locks(out_dir: Path, job_id: str):
    """Release any locks owned by this job (cleanup on exit)."""
    for path in out_dir.glob("pos_*.lock"):
        try:
            content = path.read_text().strip()
            if content == job_id:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def count_progress(out_dir: Path, total: int) -> tuple[int, int]:
    """Return (done, claimed) counts."""
    done = sum(1 for i in range(total) if (out_dir / f"pos_{i:03d}.done").exists())
    claimed = sum(1 for i in range(total) if (out_dir / f"pos_{i:03d}.lock").exists())
    return done, claimed


# ── Env helpers ───────────────────────────────────────────────────

def override_positions(vec_env, positions):
    for ei in range(vec_env.num_envs):
        env_data = vec_env._envs[ei]
        data, model = env_data["data"], env_data["model"]
        cqa = env_data["cube_qposadr"]
        pos = positions[ei]
        if cqa is not None:
            data.qpos[cqa:cqa + 3] = pos["cube_pos"]
            q_xyzw = Rotation.from_euler("z", np.radians(pos.get("cube_yaw_deg", 0.0))).as_quat()
            data.qpos[cqa + 3:cqa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        bowl_body_id = env_data["bowl_body_id"]
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = pos["bowl_pos"]
        env_data["cube_pos_init"] = np.array(pos["cube_pos"])
        env_data["bowl_pos_init"] = np.array(pos["bowl_pos"])
        mujoco.mj_forward(model, data)
    vec_env._step_counts = np.zeros(vec_env.num_envs, dtype=int)
    vec_env._episode_rewards = np.zeros(vec_env.num_envs, dtype=np.float64)


def override_single_env(vec_env, ei, pos):
    env_data = vec_env._envs[ei]
    data, model = env_data["data"], env_data["model"]
    cqa = env_data["cube_qposadr"]
    if cqa is not None:
        data.qpos[cqa:cqa + 3] = pos["cube_pos"]
        q_xyzw = Rotation.from_euler("z", np.radians(pos.get("cube_yaw_deg", 0.0))).as_quat()
        data.qpos[cqa + 3:cqa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    bowl_body_id = env_data["bowl_body_id"]
    if bowl_body_id >= 0:
        model.body_pos[bowl_body_id] = pos["bowl_pos"]
    env_data["cube_pos_init"] = np.array(pos["cube_pos"])
    env_data["bowl_pos_init"] = np.array(pos["bowl_pos"])
    mujoco.mj_forward(model, data)
    vec_env._step_counts[ei] = 0
    vec_env._episode_rewards[ei] = 0.0
    vec_env._initial_cube_z[ei] = data.qpos[cqa + 2] if cqa is not None else 0.0


def _to_np(t):
    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)


# ── Collect one batch of positions ────────────────────────────────

def collect_batch(wrapper, vec_env, batch_positions, pos_indices,
                  episodes_per_pos, max_steps, out_dir):
    """Collect episodes for a batch of positions. Returns (n_done, n_success, n_transitions)."""
    num_envs = vec_env.num_envs

    # Reset and override positions
    obs, _ = wrapper.reset()
    override_positions(vec_env, batch_positions)
    obs, _ = wrapper.reset()
    override_positions(vec_env, batch_positions)

    env_ep_count = [0] * num_envs
    env_active = [True] * num_envs
    env_transitions = [[] for _ in range(num_envs)]
    step_in_ep = np.zeros(num_envs, dtype=int)

    total_success = 0
    total_done = 0
    total_trans = 0

    while any(env_active) and not _shutdown:
        residual = torch.zeros((num_envs, wrapper.action_dim), dtype=torch.float32)

        prev_obs = {}
        for ei in range(num_envs):
            if env_active[ei]:
                prev_obs[ei] = {
                    k: v[ei].clone() if isinstance(v, torch.Tensor) else v
                    for k, v in obs.items()
                }

        next_obs, reward, terminated, truncated, info = wrapper.step(residual)
        done = terminated | truncated

        for ei in range(num_envs):
            if not env_active[ei]:
                continue

            po = prev_obs[ei]
            r_env = float(reward[ei]) if hasattr(reward, "__getitem__") else float(reward)

            env_transitions[ei].append({
                "obs_state": _to_np(po.get("observation.state", torch.zeros(10))),
                "obs_base_action": _to_np(po.get("observation.base_action", torch.zeros(7))),
                "obs_object_state": _to_np(po.get("observation.object_state", torch.zeros(10))),
                "obs_depth_front": _to_np(po.get("observation.depth.front", torch.zeros(1, 84, 84))),
                "obs_depth_wrist": _to_np(po.get("observation.depth.wrist", torch.zeros(1, 84, 84))),
                "action": np.zeros(7, dtype=np.float32),
                "reward": np.float32(r_env),
                "done": bool(done[ei]),
                "terminated": bool(terminated[ei]),
                "env_id": int(pos_indices[ei]),
                "step": int(step_in_ep[ei]),
            })
            step_in_ep[ei] += 1

            d_val = done[ei].item() if hasattr(done[ei], 'item') else bool(done[ei])
            if d_val or step_in_ep[ei] >= max_steps:
                success = bool(terminated[ei].item() if hasattr(terminated[ei], 'item') else terminated[ei])
                ep_data = env_transitions[ei]
                n_steps = len(ep_data)

                # Save npz: pos{pos_idx}_ep{ep_count}.npz
                pi = pos_indices[ei]
                ep_num = env_ep_count[ei]
                tmp_path = out_dir / f"pos{pi:03d}_ep{ep_num:03d}.tmp.npz"
                final_path = out_dir / f"pos{pi:03d}_ep{ep_num:03d}.npz"

                np.savez_compressed(
                    tmp_path,
                    obs_state=np.array([t["obs_state"] for t in ep_data]),
                    obs_base_action=np.array([t["obs_base_action"] for t in ep_data]),
                    obs_object_state=np.array([t["obs_object_state"] for t in ep_data]),
                    obs_depth_front=np.array([t["obs_depth_front"] for t in ep_data]),
                    obs_depth_wrist=np.array([t["obs_depth_wrist"] for t in ep_data]),
                    action=np.array([t["action"] for t in ep_data]),
                    reward=np.array([t["reward"] for t in ep_data]),
                    done=np.array([t["done"] for t in ep_data]),
                    terminated=np.array([t["terminated"] for t in ep_data]),
                    env_id=np.array([t["env_id"] for t in ep_data]),
                    step_idx=np.array([t["step"] for t in ep_data]),
                    success=np.array([success]),
                )
                tmp_path.rename(final_path)

                if success:
                    total_success += 1
                total_done += 1
                total_trans += n_steps

                env_ep_count[ei] += 1
                env_transitions[ei] = []
                step_in_ep[ei] = 0

                if env_ep_count[ei] >= episodes_per_pos:
                    env_active[ei] = False
                else:
                    override_single_env(vec_env, ei, batch_positions[ei])

        obs = next_obs

    return total_done, total_success, total_trans


# ── Main ──────────────────────────────────────────────────────────

def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    p = argparse.ArgumentParser(description="Parallel offline data collection")
    p.add_argument("--positions_file", type=str, required=True)
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--reward_type", type=str, default="dense_v3",
                   choices=["dense", "dense_clipped", "dense_v2", "dense_v3", "sparse"])
    p.add_argument("--episodes_per_pos", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=15,
                   help="Max positions per batch (= num MuJoCo envs)")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ema_alpha", type=float, default=0.0)
    args = p.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.positions_file) as f:
        all_positions = json.load(f)
    total_pos = len(all_positions)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = out_dir / f"config_{job_id}.json"
    config_path.write_text(json.dumps(vars(args), indent=2))

    print(f"=== Parallel Offline Collection ===", flush=True)
    print(f"  Job ID: {job_id}", flush=True)
    print(f"  Positions: {total_pos}", flush=True)
    print(f"  Episodes/pos: {args.episodes_per_pos}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  Reward type: {args.reward_type}", flush=True)
    print(f"  Max steps: {args.max_episode_steps}", flush=True)
    done_count, claimed_count = count_progress(out_dir, total_pos)
    print(f"  Already done: {done_count}/{total_pos}, claimed: {claimed_count}", flush=True)
    print(flush=True)

    t_start = time.time()
    grand_done = 0
    grand_success = 0
    grand_trans = 0
    wrapper = None
    vec_env = None
    round_num = 0

    try:
        while not _shutdown:
            # Claim next batch of positions
            claimed_indices = claim_positions(out_dir, all_positions, args.batch_size, job_id)
            if not claimed_indices:
                print(f"No more positions to claim. Done!", flush=True)
                break

            round_num += 1
            batch_positions = [all_positions[i] for i in claimed_indices]
            n_batch = len(batch_positions)

            done_count, _ = count_progress(out_dir, total_pos)
            print(f"── Round {round_num}: claimed {n_batch} positions "
                  f"{claimed_indices} ({done_count}/{total_pos} done) ──", flush=True)

            # Create/recreate envs if batch size changed
            if vec_env is None or vec_env.num_envs != n_batch:
                if wrapper is not None:
                    try: wrapper.close()
                    except: pass
                if vec_env is not None:
                    vec_env.close()

                cube_positions = [p["cube_pos"] for p in batch_positions]
                bowl_positions = [p["bowl_pos"] for p in batch_positions]

                vec_env = MuJoCoVecEnvPnP(
                    num_envs=n_batch,
                    cube_positions=cube_positions,
                    bowl_positions=bowl_positions,
                    max_episode_steps=args.max_episode_steps,
                    reward_type=args.reward_type,
                    scene_xml=SCENE_XML,
                )

                wrapper = MuJoCoResidualWrapperPnP(
                    vec_env=vec_env,
                    groot_checkpoint=args.groot_checkpoint,
                    task_description="Pick up the red cube and place it onto the plate.",
                    ema_alpha=args.ema_alpha,
                )
            else:
                # Same batch size — reuse env, just update positions
                cube_positions = [p["cube_pos"] for p in batch_positions]
                bowl_positions = [p["bowl_pos"] for p in batch_positions]
                # Need to update the env's internal positions
                obs, _ = wrapper.reset()
                override_positions(vec_env, batch_positions)

            # Collect
            t_batch = time.time()
            n_done, n_success, n_trans = collect_batch(
                wrapper, vec_env, batch_positions, claimed_indices,
                args.episodes_per_pos, args.max_episode_steps, out_dir,
            )
            dt = time.time() - t_batch

            grand_done += n_done
            grand_success += n_success
            grand_trans += n_trans

            sr = n_success / max(n_done, 1) * 100
            print(f"  Round {round_num} done: {n_done} eps, SR={sr:.1f}%, "
                  f"{n_trans} trans, {dt:.0f}s", flush=True)

            # Mark positions as done
            for idx in claimed_indices:
                mark_done(out_dir, idx)

    finally:
        # Cleanup locks on exit
        release_locks(out_dir, job_id)

    # Final summary
    elapsed = time.time() - t_start
    sr = grand_success / max(grand_done, 1) * 100
    done_count, _ = count_progress(out_dir, total_pos)

    print(f"\n{'='*60}", flush=True)
    print(f"JOB {job_id} FINISHED", flush=True)
    print(f"  Collected: {grand_done} episodes, {grand_trans} transitions", flush=True)
    print(f"  SR: {sr:.1f}% ({grand_success}/{grand_done})", flush=True)
    print(f"  Time: {elapsed:.0f}s", flush=True)
    print(f"  Global progress: {done_count}/{total_pos} positions done", flush=True)
    print(f"{'='*60}", flush=True)

    # Per-job result
    result = {
        "job_id": job_id,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "episodes": grand_done,
        "transitions": grand_trans,
        "success_rate": sr,
        "n_success": grand_success,
        "rounds": round_num,
        "time_seconds": elapsed,
        "reward_type": args.reward_type,
        "positions_file": args.positions_file,
        "episodes_per_pos": args.episodes_per_pos,
    }
    with open(out_dir / f"result_{job_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    if wrapper is not None:
        try: wrapper.close()
        except: pass
    if vec_env is not None:
        vec_env.close()


if __name__ == "__main__":
    main()
