"""
Parallel offline data collection for PnP with granular state features.

Work-stealing design: multiple SLURM jobs share one output directory.
Each job claims uncollected positions via atomic lockfiles, collects
episodes in batches, then claims more until all positions are done.

Saves granular reward-relevant features (like stack/cup collectors) to
enable reward relabeling during training.

Usage:
    python scripts/workloads/collect_pnp_offline.py \
        --positions_file configs/pnp_66ep_positions.json \
        --groot_checkpoint ~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000 \
        --output_dir outputs/offline_data/pnp_30pos_states \
        --num_positions 30 --episodes_per_pos 10 --batch_size 15
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
from scipy.spatial.transform import Rotation

torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.archive.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

SCENE_XML = os.path.expanduser(
    "~/Repos/Intern/Mujoco_Franka/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
)

_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    print(f"\n[SIGNAL] {signal.Signals(sig).name} — finishing current batch then exit...", flush=True)
    _shutdown = True


# ── Position claiming (filesystem-based) ──────────────────────────

def claim_positions(out_dir: Path, all_positions: list, batch_size: int,
                    job_id: str) -> list[int]:
    claimed = []
    now = time.time()
    for idx in range(len(all_positions)):
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


def _extract_features(vec_env, ei):
    """Extract granular reward-relevant features from env state."""
    env = vec_env._envs[ei]
    model, data, ids = env["model"], env["data"], env["ids"]
    qa = env["cube_qposadr"]
    bowl_body_id = env["bowl_body_id"]

    # TCP pose (inline, no external import needed)
    TCP_OFFSET = np.array([0.0, 0.0, 0.103])
    hand_pos = data.xpos[ids["hand_id"]].copy()
    hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
    tcp_pos = hand_pos + hand_mat @ TCP_OFFSET
    tcp_R = hand_mat.copy()

    cube_pos = data.qpos[qa:qa + 3].copy() if qa is not None else np.zeros(3)
    cube_quat_wxyz = data.qpos[qa + 3:qa + 7].copy() if qa is not None else np.array([1, 0, 0, 0])

    if bowl_body_id >= 0:
        bowl_pos = model.body_pos[bowl_body_id].copy()
    else:
        bowl_pos = np.array([0.42, 0.03, 0.0])

    tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
    cube_bowl_xy = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))
    cube_z = float(cube_pos[2])
    lift_delta = cube_z - float(vec_env._initial_cube_z[ei])

    # Alignment (same as dense_v3)
    cube_R = Rotation.from_quat([cube_quat_wxyz[1], cube_quat_wxyz[2],
                                  cube_quat_wxyz[3], cube_quat_wxyz[0]]).as_matrix()
    grip_dir = tcp_R[:2, 1]
    cube_dir = cube_R[:2, 1]
    grip_dir_n = grip_dir / (np.linalg.norm(grip_dir) + 1e-8)
    cube_dir_n = cube_dir / (np.linalg.norm(cube_dir) + 1e-8)
    cos_align = abs(float(np.dot(grip_dir_n, cube_dir_n)))

    # Gripper width
    grip_vals = []
    for fid in ids["finger_ids"]:
        if fid >= 0:
            grip_vals.append(data.qpos[model.jnt_qposadr[fid]])
        else:
            grip_vals.append(0.04)
    gripper_width = float(sum(grip_vals) / 0.04) if grip_vals else 1.0

    grasped = env["grasp_state"]["grasped"]
    cube_above_bowl = float(cube_bowl_xy < 0.095 and lift_delta > 0.01)

    return {
        "tcp_pos": tcp_pos.astype(np.float32),
        "cube_pos": cube_pos.astype(np.float32),
        "bowl_pos": bowl_pos.astype(np.float32),
        "cube_quat_wxyz": cube_quat_wxyz.astype(np.float32),
        "tcp_cube_dist": np.float32(tcp_cube_dist),
        "cube_bowl_xy_dist": np.float32(cube_bowl_xy),
        "cube_z": np.float32(cube_z),
        "lift_delta": np.float32(lift_delta),
        "alignment": np.float32(cos_align),
        "gripper_width": np.float32(gripper_width),
        "grasped": np.float32(float(grasped)),
        "cube_above_bowl": np.float32(cube_above_bowl),
    }


# ── Collect one batch of positions ────────────────────────────────

def collect_batch(wrapper, vec_env, batch_positions, pos_indices,
                  episodes_per_pos, max_steps, out_dir):
    num_envs = vec_env.num_envs

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
                "obs_state": _to_np(po.get("observation.state", torch.zeros(10))),
                "obs_base_action": _to_np(po.get("observation.base_action", torch.zeros(7))),
                "obs_object_state": _to_np(po.get("observation.object_state", torch.zeros(10))),
                "action": np.zeros(7, dtype=np.float32),
                "reward": np.float32(r_env),
                "done": bool(done[ei]),
                "terminated": bool(terminated[ei]),
                "env_id": int(pos_indices[ei]),
                "step": int(step_in_ep[ei]),
            }
            # Add granular features
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
                    # Core obs
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
                    "cube_pos": np.array([t["cube_pos"] for t in ep_data]),
                    "bowl_pos": np.array([t["bowl_pos"] for t in ep_data]),
                    "cube_quat_wxyz": np.array([t["cube_quat_wxyz"] for t in ep_data]),
                    "tcp_cube_dist": np.array([t["tcp_cube_dist"] for t in ep_data]),
                    "cube_bowl_xy_dist": np.array([t["cube_bowl_xy_dist"] for t in ep_data]),
                    "cube_z": np.array([t["cube_z"] for t in ep_data]),
                    "lift_delta": np.array([t["lift_delta"] for t in ep_data]),
                    "alignment": np.array([t["alignment"] for t in ep_data]),
                    "gripper_width": np.array([t["gripper_width"] for t in ep_data]),
                    "grasped": np.array([t["grasped"] for t in ep_data]),
                    "cube_above_bowl": np.array([t["cube_above_bowl"] for t in ep_data]),
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
                    override_single_env(vec_env, ei, batch_positions[ei])

        obs = next_obs

    return total_done, total_success, total_trans


# ── Main ──────────────────────────────────────────────────────────

def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    p = argparse.ArgumentParser(description="PnP offline data collection with state features")
    p.add_argument("--positions_file", type=str, required=True)
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_positions", type=int, default=None,
                   help="Only use first N positions from file (default: all)")
    p.add_argument("--reward_type", type=str, default="dense_v3",
                   choices=["dense", "dense_clipped", "dense_v2", "dense_v3", "dense_v4", "sparse"])
    p.add_argument("--episodes_per_pos", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=15,
                   help="Max positions per batch (= num MuJoCo envs)")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.positions_file) as f:
        all_positions = json.load(f)

    if args.num_positions is not None:
        all_positions = all_positions[:args.num_positions]
    total_pos = len(all_positions)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / f"config_{job_id}.json"
    config_path.write_text(json.dumps(vars(args), indent=2))

    print(f"=== PnP Offline Collection (with state features) ===", flush=True)
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
                    ema_alpha=0.0,
                )
            else:
                obs, _ = wrapper.reset()
                override_positions(vec_env, batch_positions)

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
    print(f"  Features: tcp_pos, cube_pos, bowl_pos, cube_quat, tcp_cube_dist,", flush=True)
    print(f"            cube_bowl_xy_dist, cube_z, lift_delta, alignment,", flush=True)
    print(f"            gripper_width, grasped, cube_above_bowl", flush=True)
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
        "features": ["tcp_pos", "cube_pos", "bowl_pos", "cube_quat_wxyz",
                      "tcp_cube_dist", "cube_bowl_xy_dist", "cube_z", "lift_delta",
                      "alignment", "gripper_width", "grasped", "cube_above_bowl"],
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
