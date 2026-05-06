"""Collect offline PnP data with reward components — work-stealing pattern.

Each worker independently picks unclaimed episodes via file locking.
Saves per-episode npz with raw reward components for flexible recomputation.

Supports difficulty levels (cumulative):
  easy   — 5 fixed base positions, no perturbation
  normal — 50% easy + 50% small perturb (±3cm pos, ±15° yaw)
  hard   — 33% easy + 33% normal-level perturb + 33% large perturb (±6cm pos, ±45° yaw)

Usage:
    python scripts/collect_pnp_offline_v3.py \
        --groot_checkpoint ~/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-200000 \
        --positions_file configs/pnp_common5_positions.json \
        --difficulty easy \
        --num_episodes_per_pos 100 \
        --output_dir outputs/offline_data/pnp_easy_v3
"""
import sys, os, json, time, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ['MUJOCO_GL'] = 'egl'

import torch
torch.backends.cuda.enable_cudnn_sdp(False)

# Deepspeed mock
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
import mujoco
from scipy.spatial.transform import Rotation
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP
from resfit.rl_finetuning.wrappers.mujoco_vec_env import get_tcp_pose

REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])
BOWL_HEIGHT = 0.03

# Difficulty perturbation configs (cumulative)
DIFFICULTY_PERTURB = {
    "easy": None,  # no perturbation
    "normal": [
        None,                                         # 50% easy (no perturb)
        {"dxy": 0.03, "dyaw": 15},                   # 50% small perturb
    ],
    "hard": [
        None,                                         # 33% easy (no perturb)
        {"dxy": 0.03, "dyaw": 15},                   # 33% normal-level perturb
        {"dxy": 0.06, "dyaw": 45},                   # 33% large perturb
    ],
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--positions_file", type=str, required=True)
    p.add_argument("--difficulty", type=str, default="easy", choices=["easy", "normal", "hard"])
    p.add_argument("--num_episodes_per_pos", type=int, default=100)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--output_dir", type=str, default="outputs/offline_data/pnp_easy_v3")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ema_alpha", type=float, default=0.9)
    p.add_argument("--task_description", type=str, default="Pick up the red cube and place it onto the plate.")
    return p.parse_args()


def extract_reward_components(vec_env, env_idx=0):
    """Extract raw physical quantities for reward recomputation."""
    e = vec_env._envs[env_idx]
    model, data, ids = e["model"], e["data"], e["ids"]
    cqa = e["cube_qposadr"]
    bowl_body_id = e["bowl_body_id"]

    tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
    cube_pos = data.qpos[cqa:cqa+3].copy() if cqa is not None else np.zeros(3)
    cube_z = float(cube_pos[2])
    init_z = float(vec_env._initial_cube_z[env_idx])
    lift_delta = cube_z - init_z

    if bowl_body_id >= 0:
        bowl_pos = model.body_pos[bowl_body_id].copy()
    else:
        bowl_pos = np.array([0.42, 0.03, 0.0])

    tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
    cube_bowl_xy = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))
    grasped = bool(e["grasp_state"]["grasped"])
    success = (cube_bowl_xy < 0.095 and cube_z < BOWL_HEIGHT + 0.03 and not grasped)

    fid = ids["finger_ids"][0]
    grip_width = float(data.qpos[model.jnt_qposadr[fid]] / 0.04) if fid >= 0 else 1.0

    return {
        "tcp_pos": tcp_pos.astype(np.float32),
        "cube_pos": cube_pos.astype(np.float32),
        "bowl_pos": bowl_pos.astype(np.float32),
        "tcp_cube_dist": np.float32(tcp_cube_dist),
        "cube_bowl_xy": np.float32(cube_bowl_xy),
        "lift_delta": np.float32(lift_delta),
        "grasped": grasped,
        "grip_width": np.float32(grip_width),
        "success": success,
    }


def compute_reward_uniform4(comp):
    """Uniform 4-phase reward (each phase 0.25, total 0~1.0).
    
    Phase 1 - Approach:  (1 - tanh(tcp_cube_dist / 0.1)) * 0.25
    Phase 2 - Grasp:     +0.25 if grasped
    Phase 3 - Transport: (1 - tanh(cube_bowl_xy / 0.1)) * 0.25
    Phase 4 - Success:   +0.25 if placed + released
    """
    if not comp["grasped"]:
        approach = (1.0 - np.tanh(comp["tcp_cube_dist"] / 0.1)) * 0.25
        return float(np.clip(approach, 0.0, 0.25))
    else:
        approach = 0.25
        grasp = 0.25
        transport = (1.0 - np.tanh(comp["cube_bowl_xy"] / 0.1)) * 0.25
        success = 0.25 if comp["success"] else 0.0
        total = approach + grasp + transport + success
        return float(np.clip(total, 0.0, 1.0))


def apply_perturbation(cube_pos, yaw_deg, perturb_cfg, rng):
    """Apply random perturbation to cube position and yaw.
    
    Returns perturbed (cube_pos, yaw_deg) and the perturbation applied.
    """
    if perturb_cfg is None:
        return cube_pos.copy(), yaw_deg, {"dxy": [0.0, 0.0], "dyaw": 0.0, "level": "easy"}
    
    dxy_range = perturb_cfg["dxy"]
    dyaw_range = perturb_cfg["dyaw"]
    
    dx = rng.uniform(-dxy_range, dxy_range)
    dy = rng.uniform(-dxy_range, dxy_range)
    dyaw = rng.uniform(-dyaw_range, dyaw_range)
    
    new_pos = cube_pos.copy()
    new_pos[0] += dx
    new_pos[1] += dy
    new_yaw = yaw_deg + dyaw
    
    level = f"perturb_dxy{dxy_range}_dyaw{dyaw_range}"
    return new_pos, new_yaw, {"dxy": [dx, dy], "dyaw": dyaw, "level": level}


def setup_env_state(vec_env, wrapper, cube_pos, bowl_pos, cube_yaw_deg):
    """Reset wrapper first, then override env state (critical ordering)."""
    reset_out = wrapper.reset()
    # wrapper.reset() may return (obs, info) tuple or just obs
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out

    env_data = vec_env._envs[0]
    data = env_data["data"]
    model = env_data["model"]
    cqa = env_data["cube_qposadr"]
    bowl_body_id = env_data["bowl_body_id"]

    for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
    for fid in env_data["ids"]["finger_ids"]:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = 0.04

    if cqa is not None:
        data.qpos[cqa:cqa+3] = cube_pos
        q_xyzw = Rotation.from_euler("z", np.radians(cube_yaw_deg)).as_quat()
        data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    if bowl_body_id >= 0:
        model.body_pos[bowl_body_id] = bowl_pos

    env_data["cube_pos_init"] = np.array(cube_pos, dtype=np.float64)
    env_data["bowl_pos_init"] = np.array(bowl_pos, dtype=np.float64)
    env_data["grasp_active"] = False
    env_data["grasp_offset"] = None
    env_data["grasp_quat_offset"] = None

    mujoco.mj_forward(model, data)
    vec_env._step_counts = np.zeros(vec_env.num_envs, dtype=int)
    vec_env._episode_rewards = np.zeros(vec_env.num_envs, dtype=np.float64)

    return obs


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load positions (deduplicate)
    positions = json.load(open(args.positions_file))
    seen = set()
    unique_positions = []
    for p in positions:
        key = p["episode"]
        if key not in seen:
            seen.add(key)
            unique_positions.append(p)
    positions = unique_positions
    num_pos = len(positions)

    # Build task list: (pos_idx, ep_idx) pairs
    tasks = []
    for pos_idx in range(num_pos):
        for ep_idx in range(args.num_episodes_per_pos):
            tasks.append((pos_idx, ep_idx))

    total_tasks = len(tasks)
    perturb_levels = DIFFICULTY_PERTURB[args.difficulty]
    rng = np.random.default_rng()  # per-worker random state
    print(f"=== PnP Offline Collection v3 (work-stealing) ===")
    print(f"  Worker PID: {os.getpid()}")
    print(f"  Model: {args.groot_checkpoint}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Positions: {num_pos} unique, {args.num_episodes_per_pos} ep/pos")
    print(f"  Total tasks: {total_tasks}")
    print(f"  EMA: {args.ema_alpha}, Max steps: {args.max_episode_steps}")
    print(f"  Output: {args.output_dir}")

    # Create env (pass first position like eval does)
    first_pos = positions[0]
    print("\nCreating environment...")
    env_kwargs = dict(
        num_envs=1,
        cube_positions=[first_pos["cube_pos"]],
        bowl_positions=[first_pos["bowl_pos"]],
        max_episode_steps=args.max_episode_steps,
        reward_type="dense",
    )
    if args.scene_xml:
        env_kwargs["scene_xml"] = args.scene_xml
    vec_env = MuJoCoVecEnvPnP(**env_kwargs)

    print("Loading GR00T policy...")
    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=args.groot_checkpoint,
        task_description=args.task_description,
        ema_alpha=args.ema_alpha,
    )
    print("Ready.\n")

    LOCK_TIMEOUT = 3600
    completed = 0
    total_success = 0
    t_start = time.time()

    while True:
        found_work = False

        for pos_idx, ep_idx in tasks:
            pos = positions[pos_idx]
            ep_name = f"pos{pos_idx}_ep{ep_idx:03d}"
            ep_dir = out_dir / ep_name
            npz_path = ep_dir / "episode.npz"
            lock_dir = ep_dir / ".lock"

            # Already done?
            if npz_path.exists():
                continue

            ep_dir.mkdir(parents=True, exist_ok=True)

            # Try to claim via mkdir lock
            try:
                lock_dir.mkdir()
            except FileExistsError:
                # Check stale lock
                lock_info = lock_dir / "info"
                if lock_info.exists():
                    try:
                        lock_time = int(lock_info.read_text().split("@")[2])
                        if time.time() - lock_time > LOCK_TIMEOUT:
                            import shutil
                            shutil.rmtree(lock_dir, ignore_errors=True)
                        else:
                            continue
                    except:
                        continue
                else:
                    continue
                # Retry claim
                try:
                    lock_dir.mkdir()
                except FileExistsError:
                    continue

            # Write lock info
            (lock_dir / "info").write_text(f"{os.getpid()}@{os.uname().nodename}@{int(time.time())}")

            print(f"\n[Worker {os.getpid()}] === {ep_name} (pos ep{pos['episode']}) ===")

            cube_pos_base = np.array(pos["cube_pos"], dtype=np.float64)
            bowl_pos = np.array(pos["bowl_pos"], dtype=np.float64)
            yaw_deg_base = pos.get("cube_yaw_deg", 0.0)

            # Apply difficulty perturbation
            if perturb_levels is None:
                # easy: no perturbation
                cube_pos, yaw_deg = cube_pos_base.copy(), yaw_deg_base
                perturb_info = {"dxy": [0.0, 0.0], "dyaw": 0.0, "level": "easy"}
            else:
                # Pick a random perturbation level
                level_cfg = perturb_levels[rng.integers(len(perturb_levels))]
                cube_pos, yaw_deg, perturb_info = apply_perturbation(
                    cube_pos_base, yaw_deg_base, level_cfg, rng)

            obs = setup_env_state(vec_env, wrapper, cube_pos, bowl_pos, yaw_deg)

            ep_transitions = []
            ep_reward = 0.0

            for step in range(args.max_episode_steps):
                residual = torch.zeros((1, 7), dtype=torch.float32)
                prev_obs = {k: v[0].clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}

                next_obs, reward, terminated, truncated, info = wrapper.step(residual)
                r_env = float(reward[0]) if hasattr(reward, "__getitem__") else float(reward)

                # Extract reward components after step
                comp = extract_reward_components(vec_env, 0)
                r_uniform4 = compute_reward_uniform4(comp)

                ep_reward += r_uniform4

                transition = {
                    "obs_state": prev_obs.get("observation.state", torch.zeros(10)).cpu().numpy(),
                    "obs_base_action": prev_obs.get("observation.base_action", torch.zeros(10)).cpu().numpy(),
                    "action": residual[0].numpy(),
                    "reward_uniform4": np.float32(r_uniform4),
                    "reward_env": np.float32(r_env),
                    "done": bool(terminated[0] or truncated[0]),
                    "terminated": bool(terminated[0]),
                    "step": step,
                    # Raw reward components
                    "tcp_cube_dist": comp["tcp_cube_dist"],
                    "cube_bowl_xy": comp["cube_bowl_xy"],
                    "lift_delta": comp["lift_delta"],
                    "grasped": comp["grasped"],
                    "grip_width": comp["grip_width"],
                    "success": comp["success"],
                    "tcp_pos": comp["tcp_pos"],
                    "cube_pos": comp["cube_pos"],
                }
                ep_transitions.append(transition)
                obs = next_obs

                if terminated[0] or truncated[0]:
                    break

            success = bool(terminated[0])
            if success:
                total_success += 1
            completed += 1

            # Save episode npz
            np.savez_compressed(
                npz_path,
                obs_state=np.array([t["obs_state"] for t in ep_transitions]),
                obs_base_action=np.array([t["obs_base_action"] for t in ep_transitions]),
                action=np.array([t["action"] for t in ep_transitions]),
                reward_uniform4=np.array([t["reward_uniform4"] for t in ep_transitions]),
                reward_env=np.array([t["reward_env"] for t in ep_transitions]),
                done=np.array([t["done"] for t in ep_transitions]),
                terminated=np.array([t["terminated"] for t in ep_transitions]),
                step_idx=np.array([t["step"] for t in ep_transitions]),
                success=np.array([success]),
                # Reward components
                tcp_cube_dist=np.array([t["tcp_cube_dist"] for t in ep_transitions]),
                cube_bowl_xy=np.array([t["cube_bowl_xy"] for t in ep_transitions]),
                lift_delta=np.array([t["lift_delta"] for t in ep_transitions]),
                grasped=np.array([t["grasped"] for t in ep_transitions]),
                grip_width=np.array([t["grip_width"] for t in ep_transitions]),
                is_success=np.array([t["success"] for t in ep_transitions]),
                tcp_pos=np.array([t["tcp_pos"] for t in ep_transitions]),
                cube_pos_traj=np.array([t["cube_pos"] for t in ep_transitions]),
                # Metadata
                pos_idx=np.array([pos_idx]),
                episode_id=np.array([pos["episode"]]),
                cube_pos_init=cube_pos,
                cube_pos_base=cube_pos_base,
                bowl_pos_init=bowl_pos,
                cube_yaw_deg=np.array([yaw_deg]),
                cube_yaw_deg_base=np.array([yaw_deg_base]),
                difficulty=np.array([args.difficulty], dtype='U'),
                perturb_level=np.array([perturb_info["level"]], dtype='U'),
                perturb_dxy=np.array(perturb_info["dxy"], dtype=np.float32),
                perturb_dyaw=np.array([perturb_info["dyaw"]], dtype=np.float32),
            )

            # Release lock
            import shutil
            shutil.rmtree(lock_dir, ignore_errors=True)

            tag = "SUCC" if success else "FAIL"
            sr = total_success / completed * 100
            elapsed = time.time() - t_start
            plvl = perturb_info["level"][:8]
            print(f"  {tag} [{plvl}] steps={step+1:3d} r={ep_reward:.2f} "
                  f"SR={sr:.1f}% ({total_success}/{completed}) t={elapsed:.0f}s", flush=True)

            found_work = True
            break  # restart task scan from beginning

        if not found_work:
            print(f"\n[Worker {os.getpid()}] No more work. Completed {completed} episodes.")
            break

    elapsed = time.time() - t_start
    sr = total_success / completed * 100 if completed > 0 else 0
    print(f"\n=== DONE ===")
    print(f"  Completed: {completed}, Success: {total_success} ({sr:.1f}%)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
