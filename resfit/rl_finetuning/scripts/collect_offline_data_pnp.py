"""Collect offline data with dense_clipped reward using GR00T base policy.

Runs GR00T (zero residual) across perturbation table positions,
records (obs, action, reward, next_obs, done) transitions with
pre-computed dense_clipped reward components, and saves as
TensorDict-compatible format.

Usage:
    python collect_offline_data.py \
        --groot_checkpoint ~/DATA/.../checkpoint-30000 \
        --random_cube_range '{"dx":[-5,5],"dy":[-5,5]}' \
        --num_episodes_per_env 320 \
        --output_dir outputs/offline_data
"""
import sys, os, json, time, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

# Disable cuDNN SDPA backend to avoid cuDNN Frontend error on B200
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
# Deepspeed mock (same as training script)
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

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP as MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP as MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import get_tcp_pose

REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--perturb_table", type=str, default=None)
    p.add_argument("--num_episodes_per_env", type=int, default=10)
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--output_dir", type=str, default="outputs/offline_data")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--debug", action="store_true", help="Print detailed reward debug info")
    p.add_argument("--cube_yaw", type=float, default=0.0, help="Cube yaw rotation in degrees")
    p.add_argument("--cube_pos", type=float, nargs=3, default=[0.42, -0.03, 0.02])
    p.add_argument("--bowl_pos", type=float, nargs=3, default=[0.42, 0.03, 0.0])
    p.add_argument("--episode_positions_file", type=str, default=None)
    p.add_argument("--task_description", type=str, default="Pick up the red cube and place it onto the plate.")
    p.add_argument("--env_ids", type=int, nargs="+", default=None,
                   help="Only collect for these env_ids (default: all)")
    p.add_argument("--random_cube_range", type=str, default=None, help="JSON random cube range")
    p.add_argument("--ema_alpha", type=float, default=0.0, help="EMA smoothing for GR00T base action (0=off)")
    p.add_argument("--use_calibrated_wrist", action="store_true",
                   help="Use calibrated wrist camera (for 66ep/100ep checkpoints)")
    p.add_argument("--calib_path", type=str, default=None,
                   help="Path to camera calibration YAML (for 66ep/100ep)")
    p.add_argument("--resume", action="store_true",
                   help="Resume collection: skip already-saved episodes in output_dir")
    return p.parse_args()


def compute_dense_clipped_reward_debug(env, env_idx, success_threshold=0.095):
    """Compute dense_clipped reward with full component breakdown for debugging."""
    e = env._envs[env_idx]
    model, data, ids = e["model"], e["data"], e["ids"]
    qa = e["cube_qposadr"]
    
    if qa is None:
        return 0.0, {}
    
    # TCP position
    tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
    
    # Cube position
    cube_pos = data.qpos[qa:qa+3].copy()
    cube_z = cube_pos[2]
    init_z = env._initial_cube_z[env_idx]
    lift_delta = cube_z - init_z
    
    # Finger-cube distance
    finger_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
    
    # Grasp state
    grasped = float(e["grasp_state"]["grasped"])
    
    # Finger positions
    fid = ids["finger_ids"][0]
    grip_width = data.qpos[model.jnt_qposadr[fid]] / 0.04 if fid >= 0 else 1.0
    
    # Reward components (v1 dense: high success/height bonus)
    distance_reward = (1.0 - np.tanh(finger_cube_dist / 0.1)) * 1.0
    contact_reward = grasped * 2.0
    height_reward = float(lift_delta > 0.005) * np.tanh(lift_delta / 0.1) * 100.0 * grasped
    success_reward = float(lift_delta >= success_threshold) * 100.0 * grasped
    
    total = float(distance_reward + contact_reward + height_reward + success_reward)
    
    debug = {
        "tcp_pos": tcp_pos.tolist(),
        "cube_pos": cube_pos.tolist(),
        "finger_cube_dist": finger_cube_dist,
        "grip_width": grip_width,
        "grasped": bool(grasped),
        "lift_cm": lift_delta * 100,
        "reward_distance": distance_reward,
        "reward_contact": contact_reward,
        "reward_height": height_reward,
        "reward_success": success_reward,
        "reward_total": total,
    }
    return total, debug


def count_existing_episodes(out_dir):
    """Count already-saved episode .npz files for resume support."""
    pattern = os.path.join(out_dir, "ep_*.npz")
    return len(glob.glob(pattern))


def main():
    args = parse_args()
    
    # Auto-detect calibrated wrist for 66ep/100ep checkpoints
    if not args.use_calibrated_wrist and args.groot_checkpoint:
        if "66ep" in args.groot_checkpoint or "100ep" in args.groot_checkpoint:
            args.use_calibrated_wrist = True
            if not args.calib_path:
                # Default calib path
                base = os.path.dirname(os.path.abspath(__file__))
                default_calib = os.path.join(base, '..', '..', '..', '..', 
                    'Mujoco_Franka', 'config', 'camera_info_66ep.yaml')
                if os.path.exists(default_calib):
                    args.calib_path = default_calib
            print(f"[auto-detect] Using calibrated wrist camera for 66ep/100ep")
    
    # ── Load positions from JSON (like eval script) ──
    all_positions = json.load(open(args.episode_positions_file))
    # Deduplicate
    seen_eps = set()
    unique_positions = []
    for p in all_positions:
        if p["episode"] not in seen_eps:
            seen_eps.add(p["episode"])
            unique_positions.append(p)
    
    # Parse random cube range for difficulty perturbation
    _rcr_parsed = None
    rng = np.random.default_rng()
    if args.random_cube_range:
        import json as _json2
        _raw_rcr = _json2.loads(args.random_cube_range)
        def _cvt_one(r):
            if r is None: return None
            return {"dx": (r["dx"][0]/100.0, r["dx"][1]/100.0), "dy": (r["dy"][0]/100.0, r["dy"][1]/100.0), "yaw": (float(r.get("yaw",[0,0])[0]), float(r.get("yaw",[0,0])[1]))}
        if isinstance(_raw_rcr, list):
            _rcr_parsed = [_cvt_one(item) for item in _raw_rcr]
            print(f"  Random cube range (list, {len(_rcr_parsed)} levels): {args.random_cube_range}")
        else:
            _rcr_parsed = [_cvt_one(_raw_rcr)]
            print(f"  Random cube range: {_rcr_parsed}")

    num_positions = len(unique_positions)
    total_episodes_target = num_positions * args.num_episodes_per_env
    print(f"=== Offline Data Collection (eval-style) ===")
    print(f"  Positions: {num_positions} unique")
    print(f"  Episodes/position: {args.num_episodes_per_env}")
    print(f"  Total episodes: {total_episodes_target}")
    print(f"  Max steps/episode: {args.max_episode_steps}")
    print(f"  Difficulty range: {args.random_cube_range or 'easy (no perturb)'}")
    print()

    # Resume support (now handled by work-stealing: skip existing .npz files)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_ep = unique_positions[0]
    print("Creating environment...")
    env_kwargs = dict(
        num_envs=1,
        cube_positions=[first_ep["cube_pos"]],
        bowl_positions=[first_ep["bowl_pos"]],
        scene_xml=args.scene_xml,
        max_episode_steps=args.max_episode_steps,
        reward_type="dense_clipped",
        success_threshold=0.095,
    )
    if args.use_calibrated_wrist:
        env_kwargs['use_calibrated_wrist'] = True
        if args.calib_path:
            env_kwargs['calib_path'] = args.calib_path
    
    mujoco_env = MuJoCoVecEnv(**env_kwargs)
    
    print("Loading GR00T policy...")
    wrapper = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag='NEW_EMBODIMENT',
        policy_device=args.device,
        task_description=args.task_description,
        ema_alpha=args.ema_alpha,
    )
    print("Ready.\n")
    
    all_transitions = []
    total_success = 0
    total_episodes = 0
    ep_rewards = []
    ep_steps_list = []
    t_start = time.time()
    
    for pos_idx, pos_info in enumerate(unique_positions):
        pos_success = 0
        
        for ep in range(args.num_episodes_per_env):
            ep_global = pos_idx * args.num_episodes_per_env + ep
            
            # Work-stealing: skip if already done or claimed by another worker
            ep_save_path = out_dir / f"ep_{ep_global:04d}.npz"
            lock_dir = out_dir / f"ep_{ep_global:04d}.lock"
            if ep_save_path.exists():
                continue
            try:
                lock_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue  # another worker claimed it
            
            # ── Reset + override env state (eval-style) ──
            obs, _ = wrapper.reset()
            
            env_data = mujoco_env._envs[0]
            data = env_data["data"]
            model = env_data["model"]
            cqa = env_data["cube_qposadr"]
            bowl_body_id = env_data["bowl_body_id"]
            
            # Robot joints → real teleop initial pose
            for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
                data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
            for fid in env_data["ids"]["finger_ids"]:
                if fid >= 0:
                    data.qpos[model.jnt_qposadr[fid]] = 0.04
            
            # Cube position + yaw from JSON
            base_cube_pos = np.array(pos_info["cube_pos"], dtype=np.float64)
            bowl_pos = np.array(pos_info["bowl_pos"], dtype=np.float64)
            actual_cube_pos = base_cube_pos.copy()
            yaw_deg = pos_info.get("cube_yaw_deg", 0.0)
            
            # Apply difficulty perturbation
            if _rcr_parsed is not None:
                level = _rcr_parsed[rng.integers(len(_rcr_parsed))]
                if level is not None:
                    actual_cube_pos[0] += rng.uniform(*level["dx"])
                    actual_cube_pos[1] += rng.uniform(*level["dy"])
                    yaw_deg += rng.uniform(*level["yaw"])
            
            if cqa is not None:
                data.qpos[cqa:cqa+3] = actual_cube_pos
                q_xyzw = Rotation.from_euler("z", np.radians(yaw_deg)).as_quat()
                data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            
            if bowl_body_id >= 0:
                model.body_pos[bowl_body_id] = bowl_pos
            
            env_data["cube_pos_init"] = actual_cube_pos.copy()
            env_data["bowl_pos_init"] = bowl_pos.copy()
            env_data["grasp_active"] = False
            env_data["grasp_offset"] = None
            env_data["grasp_quat_offset"] = None
            
            mujoco.mj_forward(model, data)
            mujoco_env._step_counts = np.zeros(mujoco_env.num_envs, dtype=int)
            mujoco_env._episode_rewards = np.zeros(mujoco_env.num_envs, dtype=np.float64)
            mujoco_env._initial_cube_z[0] = data.qpos[cqa + 2] if cqa is not None else 0.0
            
            ep_transitions = []
            ep_reward = 0.0
            
            for step in range(args.max_episode_steps):
                residual = torch.zeros((1, 7), dtype=torch.float32)
                prev_obs = {k: v[0].clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
                
                next_obs, reward, terminated, truncated, info = wrapper.step(residual)
                
                # Use env dense_clipped reward directly (correct PnP reward)
                r_env = float(reward[0]) if hasattr(reward, "__getitem__") else float(reward)

                ep_reward += r_env
                
                def _to_np(t):
                    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
                
                depth_front = _to_np(prev_obs.get("observation.depth.front", torch.zeros(1, 84, 84)))
                depth_wrist = _to_np(prev_obs.get("observation.depth.wrist", torch.zeros(1, 84, 84)))

                transition = {
                    "obs_state": _to_np(prev_obs.get("observation.state", torch.zeros(10))),
                    "obs_base_action": _to_np(prev_obs.get("observation.base_action", torch.zeros(10))),
                    "obs_object_state": _to_np(prev_obs.get("observation.object_state", torch.zeros(10))),
                    "obs_depth_front": depth_front,
                    "obs_depth_wrist": depth_wrist,
                    "action": residual[0].numpy(),
                    "reward": np.float32(r_env),
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
            ep_rewards.append(ep_reward)
            ep_steps_list.append(step + 1)
            
            # Save per-episode data atomically (write to tmp, rename)
            ep_tmp_path = out_dir / f"ep_{ep_global:04d}.tmp.npz"
            np.savez_compressed(
                ep_tmp_path,
                obs_state=np.array([t["obs_state"] for t in ep_transitions]),
                obs_base_action=np.array([t["obs_base_action"] for t in ep_transitions]),
                obs_object_state=np.array([t["obs_object_state"] for t in ep_transitions]),
                obs_depth_front=np.array([t["obs_depth_front"] for t in ep_transitions]),
                obs_depth_wrist=np.array([t["obs_depth_wrist"] for t in ep_transitions]),
                action=np.array([t["action"] for t in ep_transitions]),
                reward=np.array([t["reward"] for t in ep_transitions]),
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
            
            all_transitions.extend(ep_transitions)
            
            # Per-episode log
            tag = "SUCC" if success else "FAIL"
            sr = total_success / total_episodes * 100
            elapsed = time.time() - t_start
            print(f"  ep={total_episodes:3d}/{total_episodes_target} {tag} "
                  f"steps={step+1:3d} reward={ep_reward:7.1f} "
                  f"SR={sr:5.1f}% ({total_success}/{total_episodes}) "
                  f"t={elapsed:.0f}s", flush=True)
            
            # Every 10 episodes: summary
            if total_episodes % 10 == 0:
                recent_sr = sum(1 for r in ep_rewards[-10:] if r > 50) / min(10, len(ep_rewards[-10:])) * 100
                avg_reward = np.mean(ep_rewards[-10:])
                avg_steps = np.mean(ep_steps_list[-10:])
                print(f"  --- 10ep summary: recent_SR={recent_sr:.0f}% overall_SR={sr:.1f}% "
                      f"avg_reward={avg_reward:.1f} avg_steps={avg_steps:.0f} ---", flush=True)
        
        if total_episodes > 0:
            print(f"  [pos{pos_idx}] Done: {pos_success}/{args.num_episodes_per_env} success")
    
    # Final summary
    elapsed = time.time() - t_start
    sr = total_success / max(1, total_episodes) * 100
    print(f"\n{'='*60}")
    print(f"FINAL: {total_episodes} episodes, {len(all_transitions)} transitions")
    print(f"SR={sr:.1f}% ({total_success}/{total_episodes})")
    print(f"Avg reward: {np.mean(ep_rewards):.1f} | Avg steps: {np.mean(ep_steps_list):.0f}")
    print(f"Time: {elapsed:.0f}s ({elapsed/max(1,total_episodes):.1f}s/ep)")
    
    # Save metadata/result
    meta_path = out_dir / "result.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime('%Y%m%d_%H%M%S'),
            "total_episodes": total_episodes,
            "total_transitions": len(all_transitions),
            "success_rate": sr,
            "n_success": total_success,
            "avg_reward": float(np.mean(ep_rewards)) if ep_rewards else 0,
            "avg_steps": float(np.mean(ep_steps_list)) if ep_steps_list else 0,
            "episodes_per_env": args.num_episodes_per_env,
            "max_episode_steps": args.max_episode_steps,
            "perturb_table": args.perturb_table,
            "checkpoint": args.groot_checkpoint,
            "random_cube_range": args.random_cube_range,
            "use_calibrated_wrist": args.use_calibrated_wrist,
            "resumed_from": 0,
            "time_seconds": elapsed,
        }, f, indent=2)
    print(f"Result: {meta_path}")
    
    wrapper.close()
    print("Done.")


if __name__ == "__main__":
    main()
