"""Collect offline data with dense reward using GR00T base policy for Stack Cube.

Runs GR00T (zero residual) across Stack Cube positions,
records (obs, action, reward, next_obs, done) transitions, and saves as
per-episode .npz files compatible with the TD3 training script.

Usage:
    python collect_offline_data_stack.py \
        --groot_checkpoint ~/DATA/.../groot_stack_sim_66ep/checkpoint-100000 \
        --episode_positions_file configs/stack_cube_positions.json \
        --num_episodes_per_env 300 \
        --output_dir outputs/offline_data/stack_easy
"""
import sys, os, json, time, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
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
import torch
import mujoco
from scipy.spatial.transform import Rotation
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import (
    MuJoCoVecEnvStack, CUBE_HALF, get_tcp_pose,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack

REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])


def extract_features(mujoco_env, env_idx=0):
    """Extract raw reward features from current env state."""
    env = mujoco_env._envs[env_idx]
    model, data, ids = env["model"], env["data"], env["ids"]
    qa = env["white_qposadr"]
    white_pos = data.qpos[qa:qa + 3].copy() if qa else np.zeros(3)
    tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
    grasped = env["grasp_state"]["grasped"]
    green_body_id = env["green_body_id"]
    green_pos = model.body_pos[green_body_id].copy() if green_body_id >= 0 else np.zeros(3)
    green_top = green_pos.copy()
    green_top[2] += CUBE_HALF * 2

    tcp_white_dist = float(np.linalg.norm(tcp_pos - white_pos))
    white_to_green_top_3d = float(np.linalg.norm(white_pos - green_top))
    white_green_xy_dist = float(np.linalg.norm(white_pos[:2] - green_pos[:2]))

    white_gid = env["white_geom_id"]
    green_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_cube_geom")
    white_green_contact = False
    for ci in range(data.ncon):
        c = data.contact[ci]
        if {c.geom1, c.geom2} == {white_gid, green_gid}:
            white_green_contact = True
            break

    return {
        "tcp_white_dist": np.float32(tcp_white_dist),
        "white_to_green_top_3d": np.float32(white_to_green_top_3d),
        "white_green_xy_dist": np.float32(white_green_xy_dist),
        "white_z": np.float32(white_pos[2]),
        "green_z": np.float32(green_pos[2]),
        "grasped": np.bool_(grasped),
        "white_green_contact": np.bool_(white_green_contact),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--episode_positions_file", type=str, required=True,
                   help="JSON file with stack_cube positions (configs/stack_cube_positions.json)")
    p.add_argument("--num_episodes_per_env", type=int, default=300)
    p.add_argument("--max_episode_steps", type=int, default=1000)
    p.add_argument("--output_dir", type=str, default="outputs/offline_data/stack_75k")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--task_description", type=str,
                   default="Pick up the white cube and stack it on the green cube.")
    p.add_argument("--random_cube_range", type=str, default=None,
                   help='JSON perturbation, e.g. \'{"dx":[-6,6],"dy":[-6,6],"yaw":[-30,30]}\'')
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument("--calib_path", type=str, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip already-saved episodes in output_dir")
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect calibrated wrist
    use_calibrated = False
    if args.groot_checkpoint and ("66ep" in args.groot_checkpoint or "100ep" in args.groot_checkpoint):
        use_calibrated = True
        if not args.calib_path:
            base = os.path.dirname(os.path.abspath(__file__))
            default_calib = os.path.join(base, '..', '..', '..', '..',
                'Mujoco_Franka', 'config', 'camera_info_66ep.yaml')
            if os.path.exists(default_calib):
                args.calib_path = default_calib
        print(f"[auto-detect] Using calibrated wrist camera for 66ep/100ep")

    # ── Load positions ──
    all_positions = json.load(open(args.episode_positions_file))
    seen_eps = set()
    unique_positions = []
    for p in all_positions:
        if p["episode"] not in seen_eps:
            seen_eps.add(p["episode"])
            unique_positions.append(p)

    # Perturbation
    _rcr_parsed = None
    rng = np.random.default_rng()
    if args.random_cube_range:
        _raw_rcr = json.loads(args.random_cube_range)
        def _cvt_one(r):
            if r is None: return None
            return {
                "dx": (r["dx"][0]/100.0, r["dx"][1]/100.0),
                "dy": (r["dy"][0]/100.0, r["dy"][1]/100.0),
                "yaw": (float(r.get("yaw", [0,0])[0]), float(r.get("yaw", [0,0])[1])),
            }
        if isinstance(_raw_rcr, list):
            _rcr_parsed = [_cvt_one(item) for item in _raw_rcr]
        else:
            _rcr_parsed = [_cvt_one(_raw_rcr)]
        print(f"  Random cube range: {args.random_cube_range}")

    num_positions = len(unique_positions)
    total_episodes_target = num_positions * args.num_episodes_per_env
    print(f"=== Stack Cube Offline Data Collection ===")
    print(f"  Positions: {num_positions} unique")
    print(f"  Episodes/position: {args.num_episodes_per_env}")
    print(f"  Total episodes: {total_episodes_target}")
    print(f"  Max steps/episode: {args.max_episode_steps}")
    print(f"  Reward type: dense (stage-based: approach/grasp/lift/align/stack)")
    print()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Create env ──
    first_ep = unique_positions[0]
    env_kwargs = dict(
        num_envs=1,
        white_cube_positions=[first_ep["white_cube_pos"]],
        green_cube_positions=[first_ep["green_cube_pos"]],
        max_episode_steps=args.max_episode_steps,
        reward_type="dense",
    )
    if args.calib_path:
        env_kwargs['calib_path'] = args.calib_path

    print("Creating Stack Cube environment...")
    mujoco_env = MuJoCoVecEnvStack(**env_kwargs)

    print("Loading GR00T policy...")
    wrapper = MuJoCoResidualWrapperStack(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag='NEW_EMBODIMENT',
        policy_device=args.device,
        task_description=args.task_description,
        ema_alpha=args.ema_alpha,
    )
    print("Ready.\n")

    total_success = 0
    total_episodes = 0
    ep_steps_list = []
    t_start = time.time()

    for pos_idx, pos_info in enumerate(unique_positions):
        pos_success = 0

        for ep in range(args.num_episodes_per_env):
            ep_global = pos_idx * args.num_episodes_per_env + ep

            # Work-stealing: skip if already done
            ep_save_path = out_dir / f"ep_{ep_global:04d}.npz"
            lock_dir = out_dir / f"ep_{ep_global:04d}.lock"
            if ep_save_path.exists():
                continue
            try:
                lock_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue

            # ── Reset + override env state ──
            obs, _ = wrapper.reset()

            env_data = mujoco_env._envs[0]
            data = env_data["data"]
            model = env_data["model"]
            wqa = env_data["white_qposadr"]
            green_body_id = env_data["green_body_id"]

            # Robot joints → real teleop initial pose
            for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
                data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
            for fid in env_data["ids"]["finger_ids"]:
                if fid >= 0:
                    data.qpos[model.jnt_qposadr[fid]] = 0.04

            # White cube position + yaw
            base_white_pos = np.array(pos_info["white_cube_pos"], dtype=np.float64)
            actual_white_pos = base_white_pos.copy()
            white_yaw_deg = pos_info.get("white_yaw_deg", 0.0)

            # Green cube position
            green_pos = np.array(pos_info["green_cube_pos"], dtype=np.float64)
            green_yaw_deg = pos_info.get("green_yaw_deg", 0.0)

            # Apply difficulty perturbation to white cube
            if _rcr_parsed is not None:
                level = _rcr_parsed[rng.integers(len(_rcr_parsed))]
                if level is not None:
                    actual_white_pos[0] += rng.uniform(*level["dx"])
                    actual_white_pos[1] += rng.uniform(*level["dy"])
                    white_yaw_deg += rng.uniform(*level["yaw"])

            # Set white cube
            if wqa is not None:
                data.qpos[wqa:wqa+3] = actual_white_pos
                q_xyzw = Rotation.from_euler("z", np.radians(white_yaw_deg)).as_quat()
                data.qpos[wqa+3:wqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

            # Set green cube
            if green_body_id >= 0:
                model.body_pos[green_body_id] = green_pos

            # Reset grasp state
            env_data["grasp_state"] = {"grasped": False, "contact_count": 0}
            if "weld_eq_idx" in env_data and env_data["weld_eq_idx"] is not None:
                model.eq_active[env_data["weld_eq_idx"]] = 0

            mujoco.mj_forward(model, data)
            mujoco_env._step_counts = np.zeros(mujoco_env.num_envs, dtype=int)
            mujoco_env._initial_white_z[0] = data.qpos[wqa + 2] if wqa is not None else 0.0

            ep_transitions = []

            for step in range(args.max_episode_steps):
                residual = torch.zeros((1, 7), dtype=torch.float32)
                prev_obs = {k: v[0].clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}

                # Extract features BEFORE step (current state)
                feat = extract_features(mujoco_env, 0)

                next_obs, reward, terminated, truncated, info = wrapper.step(residual)

                def _to_np(t):
                    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)

                depth_front = _to_np(prev_obs.get("observation.depth.front", torch.zeros(1, 84, 84)))
                depth_wrist = _to_np(prev_obs.get("observation.depth.wrist", torch.zeros(1, 84, 84)))

                transition = {
                    "obs_state": _to_np(prev_obs.get("observation.state", torch.zeros(10))),
                    "obs_base_action": _to_np(prev_obs.get("observation.base_action", torch.zeros(7))),
                    "obs_object_state": _to_np(prev_obs.get("observation.object_state", torch.zeros(10))),
                    "obs_depth_front": depth_front,
                    "obs_depth_wrist": depth_wrist,
                    "action": residual[0].numpy(),
                    # Raw features (reward computed later from config)
                    "tcp_white_dist": feat["tcp_white_dist"],
                    "white_to_green_top_3d": feat["white_to_green_top_3d"],
                    "white_green_xy_dist": feat["white_green_xy_dist"],
                    "white_z": feat["white_z"],
                    "green_z": feat["green_z"],
                    "grasped": feat["grasped"],
                    "white_green_contact": feat["white_green_contact"],
                    "done": bool(terminated[0] or truncated[0]),
                    "terminated": bool(terminated[0]),
                    "env_id": pos_idx,
                    "step": step,
                }
                ep_transitions.append(transition)

                obs = next_obs
                done = terminated | truncated
                if done[0]:
                    break

            success = bool(terminated[0]) if isinstance(terminated, torch.Tensor) else bool(terminated[0])
            if success:
                pos_success += 1
                total_success += 1
            total_episodes += 1
            ep_steps_list.append(step + 1)

            # Save per-episode data atomically
            ep_tmp_path = out_dir / f"ep_{ep_global:04d}.tmp.npz"
            np.savez_compressed(
                ep_tmp_path,
                obs_state=np.array([t["obs_state"] for t in ep_transitions]),
                obs_base_action=np.array([t["obs_base_action"] for t in ep_transitions]),
                obs_object_state=np.array([t["obs_object_state"] for t in ep_transitions]),
                obs_depth_front=np.array([t["obs_depth_front"] for t in ep_transitions]),
                obs_depth_wrist=np.array([t["obs_depth_wrist"] for t in ep_transitions]),
                action=np.array([t["action"] for t in ep_transitions]),
                # Raw features for deferred reward computation
                tcp_white_dist=np.array([t["tcp_white_dist"] for t in ep_transitions]),
                white_to_green_top_3d=np.array([t["white_to_green_top_3d"] for t in ep_transitions]),
                white_green_xy_dist=np.array([t["white_green_xy_dist"] for t in ep_transitions]),
                white_z=np.array([t["white_z"] for t in ep_transitions]),
                green_z=np.array([t["green_z"] for t in ep_transitions]),
                grasped=np.array([t["grasped"] for t in ep_transitions]),
                white_green_contact=np.array([t["white_green_contact"] for t in ep_transitions]),
                done=np.array([t["done"] for t in ep_transitions]),
                terminated=np.array([t["terminated"] for t in ep_transitions]),
                env_id=np.array([t["env_id"] for t in ep_transitions]),
                step_idx=np.array([t["step"] for t in ep_transitions]),
                success=np.array([success]),
            )
            # Atomic rename + cleanup lock
            ep_tmp_path.rename(ep_save_path)
            import shutil
            shutil.rmtree(lock_dir, ignore_errors=True)

            tag = "SUCC" if success else "FAIL"
            sr = total_success / total_episodes * 100
            elapsed = time.time() - t_start
            print(f"  ep={total_episodes:3d}/{total_episodes_target} {tag} "
                  f"steps={step+1:3d} "
                  f"SR={sr:5.1f}% ({total_success}/{total_episodes}) "
                  f"t={elapsed:.0f}s", flush=True)

            if total_episodes % 10 == 0:
                avg_steps = np.mean(ep_steps_list[-10:])
                print(f"  --- 10ep summary: overall_SR={sr:.1f}% "
                      f"avg_steps={avg_steps:.0f} ---", flush=True)

        if total_episodes > 0:
            print(f"  [pos{pos_idx}] Done: {pos_success}/{args.num_episodes_per_env} success")

    # Final summary
    elapsed = time.time() - t_start
    sr = total_success / max(1, total_episodes) * 100
    print(f"\n{'='*60}")
    print(f"FINAL: {total_episodes} episodes")
    print(f"SR={sr:.1f}% ({total_success}/{total_episodes})")
    print(f"Avg steps: {np.mean(ep_steps_list):.0f}")
    print(f"Time: {elapsed:.0f}s ({elapsed/max(1,total_episodes):.1f}s/ep)")

    meta_path = out_dir / "result.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime('%Y%m%d_%H%M%S'),
            "total_episodes": total_episodes,
            "success_rate": sr,
            "n_success": total_success,
            "avg_steps": float(np.mean(ep_steps_list)) if ep_steps_list else 0,
            "episodes_per_env": args.num_episodes_per_env,
            "max_episode_steps": args.max_episode_steps,
            "checkpoint": args.groot_checkpoint,
            "random_cube_range": args.random_cube_range,
            "time_seconds": elapsed,
        }, f, indent=2)
    print(f"Result: {meta_path}")

    wrapper.close()
    print("Done.")


if __name__ == "__main__":
    main()
