"""
Parallel offline data collection for Cup with granular state features.

Work-stealing design: multiple SLURM jobs share one output directory.
Each job claims uncollected positions via atomic lockfiles, collects
episodes in batches, then claims more until all positions are done.

Usage:
    python scripts/workloads/collect_cup_offline.py \
        --positions_file configs/cup_positions.json \
        --groot_checkpoint ~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000 \
        --output_dir outputs/offline_cup_batch \
        --num_positions 27 --episodes_per_pos 10 --batch_size 9
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

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import numpy as np
import torch
import mujoco

torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup, get_tcp_pose
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified

_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    print(f"\n[SIGNAL] {signal.Signals(sig).name} — finishing current batch then exit...", flush=True)
    _shutdown = True


# ── Position claiming (filesystem-based) ──────────────────────────

def claim_positions(out_dir: Path, total: int, batch_size: int,
                    job_id: str) -> list[int]:
    claimed = []
    now = time.time()
    for idx in range(total):
        if len(claimed) >= batch_size:
            break
        done_path = out_dir / f"pos_{idx:03d}.done"
        lock_path = out_dir / f"pos_{idx:03d}.lock"
        if done_path.exists():
            continue
        if lock_path.exists():
            try:
                age = now - lock_path.stat().st_mtime
                if age < 7200:
                    continue
                lock_path.unlink(missing_ok=True)
            except OSError:
                continue
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{job_id}\n".encode())
            os.close(fd)
            claimed.append(idx)
        except FileExistsError:
            continue
    return claimed


def mark_done(out_dir: Path, idx: int):
    done_path = out_dir / f"pos_{idx:03d}.done"
    done_path.write_text(f"{os.environ.get('SLURM_JOB_ID', os.getpid())}\n")
    lock_path = out_dir / f"pos_{idx:03d}.lock"
    lock_path.unlink(missing_ok=True)


def release_locks(out_dir: Path, job_id: str):
    for path in out_dir.glob("pos_*.lock"):
        try:
            content = path.read_text().strip()
            if content == job_id:
                path.unlink(missing_ok=True)
        except: pass


def count_progress(out_dir: Path, total: int) -> tuple[int, int]:
    done = sum(1 for i in range(total) if (out_dir / f"pos_{i:03d}.done").exists())
    claimed = sum(1 for i in range(total) if (out_dir / f"pos_{i:03d}.lock").exists())
    return done, claimed


# ── Feature extraction ────────────────────────────────────────────

def _to_np(t):
    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)


def _extract_features(vec_env, ei):
    """Extract granular reward-relevant features from Cup env state."""
    env = vec_env._envs[ei]
    model, data, ids = env["model"], env["data"], env["ids"]
    cup_qpa = env["cup_qposadr"]
    cup_dof = env["cup_dofadr"]

    # TCP pose
    tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])

    # Cup state
    if cup_qpa is not None:
        cup_pos = data.qpos[cup_qpa:cup_qpa + 3].copy()
        cup_quat_wxyz = data.qpos[cup_qpa + 3:cup_qpa + 7].copy()
    else:
        cup_pos = np.zeros(3)
        cup_quat_wxyz = np.array([1, 0, 0, 0])

    # Uprightness: 1 - 2*(x^2 + y^2)
    w, x, y, z = cup_quat_wxyz
    uprightness = float(1.0 - 2.0 * (x * x + y * y))

    tcp_cup_dist = float(np.linalg.norm(tcp_pos - cup_pos))
    cup_z = float(cup_pos[2])

    # Cup velocity
    if cup_dof is not None:
        cup_vel = float(np.linalg.norm(data.qvel[cup_dof:cup_dof + 6]))
    else:
        cup_vel = 0.0

    # Grasp state
    grasped = env["grasp_state"]["grasped"]

    # Gripper width
    finger_id = ids["finger_ids"][0]
    if finger_id >= 0:
        grip_width = max(0.0, float(data.qpos[model.jnt_qposadr[finger_id]]))
    else:
        grip_width = 0.04

    return {
        "tcp_pos": tcp_pos.astype(np.float32),
        "cup_pos": cup_pos.astype(np.float32),
        "cup_quat_wxyz": cup_quat_wxyz.astype(np.float32),
        "uprightness": np.float32(uprightness),
        "tcp_cup_dist": np.float32(tcp_cup_dist),
        "cup_z": np.float32(cup_z),
        "cup_vel": np.float32(cup_vel),
        "grasped": np.float32(float(grasped)),
        "grip_width": np.float32(grip_width),
    }


# ── Collect one batch of positions ────────────────────────────────

def collect_batch(wrapper, vec_env, pos_indices, episodes_per_pos,
                  max_steps, out_dir):
    num_envs = vec_env.num_envs

    obs, _ = wrapper.reset()

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
        prev_features = {}
        for ei in range(num_envs):
            if env_active[ei]:
                prev_obs[ei] = {
                    k: v[ei].clone() if isinstance(v, torch.Tensor) else v
                    for k, v in obs.items()
                }
                prev_features[ei] = _extract_features(vec_env, ei)

        next_obs, reward, terminated, truncated, info = wrapper.step(residual)
        done = terminated | truncated

        for ei in range(num_envs):
            if not env_active[ei]:
                continue

            po = prev_obs[ei]
            feat = prev_features[ei]
            r_env = float(reward[ei]) if hasattr(reward, "__getitem__") else float(reward)

            transition = {
                "obs_state": _to_np(po.get("observation.state", torch.zeros(8))),
                "obs_base_action": _to_np(po.get("observation.base_action", torch.zeros(7))),
                "obs_object_state": _to_np(po.get("observation.object_state", torch.zeros(8))),
                "action": np.zeros(7, dtype=np.float32),
                "reward": np.float32(r_env),
                "done": bool(done[ei]),
                "terminated": bool(terminated[ei]),
                "env_id": int(pos_indices[ei]),
                "step": int(step_in_ep[ei]),
            }
            for k, v in feat.items():
                transition[k] = v

            env_transitions[ei].append(transition)
            step_in_ep[ei] += 1

            d_val = done[ei].item() if hasattr(done[ei], 'item') else bool(done[ei])
            if d_val or step_in_ep[ei] >= max_steps:
                success = bool(terminated[ei].item() if hasattr(terminated[ei], 'item') else terminated[ei])
                ep_data = env_transitions[ei]

                pi = pos_indices[ei]
                ep_num = env_ep_count[ei]
                tmp_path = out_dir / f"pos{pi:03d}_ep{ep_num:03d}.tmp.npz"
                final_path = out_dir / f"pos{pi:03d}_ep{ep_num:03d}.npz"

                save_dict = {
                    "obs_state": np.array([t["obs_state"] for t in ep_data]),
                    "obs_base_action": np.array([t["obs_base_action"] for t in ep_data]),
                    "obs_object_state": np.array([t["obs_object_state"] for t in ep_data]),
                    "action": np.array([t["action"] for t in ep_data]),
                    "reward": np.array([t["reward"] for t in ep_data]),
                    "done": np.array([t["done"] for t in ep_data]),
                    "terminated": np.array([t["terminated"] for t in ep_data]),
                    "env_id": np.array([t["env_id"] for t in ep_data]),
                    "step_idx": np.array([t["step"] for t in ep_data]),
                    "success": np.array([success]),
                    # Granular features for reward relabeling
                    "tcp_pos": np.array([t["tcp_pos"] for t in ep_data]),
                    "cup_pos": np.array([t["cup_pos"] for t in ep_data]),
                    "cup_quat_wxyz": np.array([t["cup_quat_wxyz"] for t in ep_data]),
                    "uprightness": np.array([t["uprightness"] for t in ep_data]),
                    "tcp_cup_dist": np.array([t["tcp_cup_dist"] for t in ep_data]),
                    "cup_z": np.array([t["cup_z"] for t in ep_data]),
                    "cup_vel": np.array([t["cup_vel"] for t in ep_data]),
                    "grasped": np.array([t["grasped"] for t in ep_data]),
                    "grip_width": np.array([t["grip_width"] for t in ep_data]),
                }

                np.savez_compressed(tmp_path, **save_dict)
                tmp_path.rename(final_path)

                if success:
                    total_success += 1
                total_done += 1
                total_trans += len(ep_data)

                env_ep_count[ei] += 1
                env_transitions[ei] = []
                step_in_ep[ei] = 0

                if env_ep_count[ei] >= episodes_per_pos:
                    env_active[ei] = False
                else:
                    # Reset this env for next episode (same position)
                    vec_env.reset_envs([ei])

        obs = next_obs

    return total_done, total_success, total_trans


# ── Main ──────────────────────────────────────────────────────────

def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    p = argparse.ArgumentParser(description="Cup offline data collection with state features")
    p.add_argument("--positions_file", type=str, required=True)
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_positions", type=int, default=None,
                   help="Only use first N positions from file (default: all)")
    p.add_argument("--reward_type", type=str, default="dense",
                   choices=["dense", "dense_bonus", "dense_v2", "dense_v3", "sparse"])
    p.add_argument("--episodes_per_pos", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=9,
                   help="Max positions per batch (= num MuJoCo envs)")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load positions to get total count
    with open(args.positions_file) as f:
        all_positions = json.load(f)

    total_pos = args.num_positions or len(all_positions)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / f"config_{job_id}.json"
    config_path.write_text(json.dumps(vars(args), indent=2))

    print(f"=== Cup Offline Collection (with state features) ===", flush=True)
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
            claimed_indices = claim_positions(out_dir, total_pos, args.batch_size, job_id)
            if not claimed_indices:
                print(f"No more positions to claim. Done!", flush=True)
                break

            round_num += 1
            n_batch = len(claimed_indices)

            done_count, _ = count_progress(out_dir, total_pos)
            print(f"── Round {round_num}: claimed {n_batch} positions "
                  f"{claimed_indices} ({done_count}/{total_pos} done) ──", flush=True)

            # Cup VecEnv uses episode_ids to assign positions
            # Recreate env each batch since episode_ids change
            if wrapper is not None:
                try: wrapper.close()
                except: pass
            if vec_env is not None:
                vec_env.close()

            vec_env = MuJoCoVecEnvCup(
                num_envs=n_batch,
                episode_ids=claimed_indices,
                cup_positions_path=args.positions_file,
                max_episode_steps=args.max_episode_steps,
                reward_type=args.reward_type,
                parallel_envs=False,  # single-process for collection
            )

            wrapper = MuJoCoResidualWrapperUnified(
                vec_env=vec_env,
                groot_checkpoint=args.groot_checkpoint,
                task_description="Pick up the cup lying on its side and stand it upright.",
                grip_min=0.0,
                grip_max=0.04,
                use_gripper_latch=True,
                grip_close_latch_thresh=0.015,
                grip_open_latch_thresh=0.035,
                grip_latch_open_steps=32,
                ema_alpha=0.0,
            )

            t_batch = time.time()
            n_done, n_success, n_trans = collect_batch(
                wrapper, vec_env, claimed_indices,
                args.episodes_per_pos, args.max_episode_steps, out_dir,
            )
            dt = time.time() - t_batch

            grand_done += n_done
            grand_success += n_success
            grand_trans += n_trans

            sr = n_success / max(n_done, 1) * 100
            print(f"  Round {round_num} done: {n_done} eps, SR={sr:.1f}%, "
                  f"{n_trans} trans, {dt:.0f}s", flush=True)

            for idx in claimed_indices:
                mark_done(out_dir, idx)

    finally:
        release_locks(out_dir, job_id)

    elapsed = time.time() - t_start
    sr = grand_success / max(grand_done, 1) * 100
    done_count, _ = count_progress(out_dir, total_pos)

    print(f"\n{'='*60}", flush=True)
    print(f"JOB {job_id} FINISHED", flush=True)
    print(f"  Collected: {grand_done} episodes, {grand_trans} transitions", flush=True)
    print(f"  SR: {sr:.1f}% ({grand_success}/{grand_done})", flush=True)
    print(f"  Time: {elapsed:.0f}s", flush=True)
    print(f"  Global progress: {done_count}/{total_pos} positions done", flush=True)
    print(f"  Features: tcp_pos, cup_pos, cup_quat_wxyz, uprightness,", flush=True)
    print(f"            tcp_cup_dist, cup_z, cup_vel, grasped, grip_width", flush=True)
    print(f"{'='*60}", flush=True)

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
        "num_positions": total_pos,
        "episodes_per_pos": args.episodes_per_pos,
        "features": ["tcp_pos", "cup_pos", "cup_quat_wxyz", "uprightness",
                      "tcp_cup_dist", "cup_z", "cup_vel", "grasped", "grip_width"],
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
