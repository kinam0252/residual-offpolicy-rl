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
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import get_tcp_pose


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


def compute_dense_clipped_reward_debug(env, env_idx, success_threshold=0.03):
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
    
    # Load perturbation table or use pure random mode
    if args.random_cube_range and not args.perturb_table:
        base_pos = [0.45, -0.05, 0.02]
        n_positions = 1
        envs_config = [{"env_id": 0, "name": "random", "dx": 0.0, "dy": 0.0}]
        cube_positions = [base_pos]
        total_episodes_target = args.num_episodes_per_env
        print(f"=== Offline Data Collection (Pure Random) ===")
        print(f"Total episodes: {total_episodes_target}")
        print(f"Max steps/episode: {args.max_episode_steps}")
        print()
    else:
        with open(args.perturb_table) as f:
            perturb = json.load(f)
        base_pos = perturb["base_pos"]
        envs_config = perturb["envs"]
        if args.env_ids is not None:
            envs_config = [ec for ec in envs_config if ec["env_id"] in args.env_ids]
        n_positions = len(envs_config)
        total_episodes_target = n_positions * args.num_episodes_per_env
        print(f"=== Offline Data Collection ===")
        print(f"Positions: {n_positions}")
        print(f"Episodes/position: {args.num_episodes_per_env}")
        print(f"Total episodes: {total_episodes_target}")
        print(f"Max steps/episode: {args.max_episode_steps}")
        print()
        cube_positions = []
        for ec in envs_config:
            pos = [base_pos[0] + ec["dx"], base_pos[1] + ec["dy"], base_pos[2]]
            cube_positions.append(pos)
            print(f"  env {ec['env_id']:2d} ({ec['name']:>16s}): cube=({pos[0]:.3f}, {pos[1]:.3f})")
    
    # Resume support
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_episodes = 0
    if args.resume:
        skip_episodes = count_existing_episodes(str(out_dir))
        if skip_episodes > 0:
            print(f"[resume] Found {skip_episodes} existing episodes, skipping them")
        if skip_episodes >= total_episodes_target:
            print(f"[resume] Already have {skip_episodes}/{total_episodes_target} episodes. Done!")
            return
    
    # Parse random cube range
    _rcr_parsed = None
    if args.random_cube_range:
        import json as _json2
        _rcr = _json2.loads(args.random_cube_range)
        _rcr_parsed = {"dx": (_rcr["dx"][0]/100.0, _rcr["dx"][1]/100.0), "dy": (_rcr["dy"][0]/100.0, _rcr["dy"][1]/100.0), "yaw": (float(_rcr.get("yaw",[0,0])[0]), float(_rcr.get("yaw",[0,0])[1]))}
        print(f"  Random cube range: {_rcr_parsed}")

    print("\nCreating environment...")
    env_kwargs = dict(
        num_envs=1,
        cube_positions=[cube_positions[0]],
        scene_xml=args.scene_xml,
        device='cpu',
        max_episode_steps=args.max_episode_steps,
        reward_type="dense",
        success_threshold=0.03,
        cube_yaw_deg=args.cube_yaw,
        random_cube_range=_rcr_parsed,
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
        task_description='lift the cube',
        ema_alpha=args.ema_alpha,
    )
    print("Ready.\n")
    
    all_transitions = []
    total_success = 0
    total_episodes = 0
    ep_rewards = []
    ep_steps_list = []
    t_start = time.time()
    
    global_ep_idx = skip_episodes  # for file naming
    
    for pos_idx, ec in enumerate(envs_config):
        pos_name = ec["name"]
        pos = cube_positions[pos_idx]
        mujoco_env._envs[0]["cube_pos_init"] = np.array(pos, dtype=np.float64)
        
        pos_success = 0
        
        for ep in range(args.num_episodes_per_env):
            # Skip already-collected episodes when resuming
            ep_global = pos_idx * args.num_episodes_per_env + ep
            if ep_global < skip_episodes:
                continue
            
            obs, _ = wrapper.reset()
            ep_transitions = []
            ep_reward = 0.0
            
            for step in range(args.max_episode_steps):
                residual = torch.zeros((1, 7), dtype=torch.float32)
                prev_obs = {k: v[0].clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
                
                next_obs, reward, terminated, truncated, info = wrapper.step(residual)
                
                r_total, r_debug = compute_dense_clipped_reward_debug(
                    mujoco_env, 0, success_threshold=0.03)
                
                if args.debug and ep == 0 and (step % 20 == 0 or step >= 100):
                    print(f"  [{pos_name}] s={step:3d} dist={r_debug['finger_cube_dist']:.4f} "
                          f"grip={r_debug['grip_width']:.3f} grasped={r_debug['grasped']} "
                          f"lift={r_debug['lift_cm']:.2f}cm "
                          f"r_total={r_debug['reward_total']:.4f}")
                
                ep_reward += r_total
                
                depth_front = prev_obs.get("observation.depth.front", torch.zeros(1, 84, 84)).numpy()
                depth_wrist = prev_obs.get("observation.depth.wrist", torch.zeros(1, 84, 84)).numpy()

                transition = {
                    "obs_state": prev_obs.get("observation.state", torch.zeros(10)).numpy(),
                    "obs_base_action": prev_obs.get("observation.base_action", torch.zeros(7)).numpy(),
                    "obs_object_state": prev_obs.get("observation.object_state", torch.zeros(7)).numpy(),
                    "obs_depth_front": depth_front,
                    "obs_depth_wrist": depth_wrist,
                    "action": residual[0].numpy(),
                    "reward": np.float32(r_total),
                    "reward_distance": np.float32(r_debug.get("reward_distance", 0)),
                    "reward_contact": np.float32(r_debug.get("reward_contact", 0)),
                    "reward_height": np.float32(r_debug.get("reward_height", 0)),
                    "reward_success": np.float32(r_debug.get("reward_success", 0)),
                    "done": bool(terminated[0] or truncated[0]),
                    "terminated": bool(terminated[0]),
                    "env_id": ec["env_id"],
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
            
            # Save per-episode data incrementally
            ep_save_path = out_dir / f"ep_{global_ep_idx:04d}.npz"
            np.savez_compressed(
                ep_save_path,
                obs_state=np.array([t["obs_state"] for t in ep_transitions]),
                obs_base_action=np.array([t["obs_base_action"] for t in ep_transitions]),
                obs_object_state=np.array([t["obs_object_state"] for t in ep_transitions]),
                obs_depth_front=np.array([t["obs_depth_front"] for t in ep_transitions]),
                obs_depth_wrist=np.array([t["obs_depth_wrist"] for t in ep_transitions]),
                action=np.array([t["action"] for t in ep_transitions]),
                reward=np.array([t["reward"] for t in ep_transitions]),
                reward_distance=np.array([t["reward_distance"] for t in ep_transitions]),
                reward_contact=np.array([t["reward_contact"] for t in ep_transitions]),
                reward_height=np.array([t["reward_height"] for t in ep_transitions]),
                reward_success=np.array([t["reward_success"] for t in ep_transitions]),
                done=np.array([t["done"] for t in ep_transitions]),
                terminated=np.array([t["terminated"] for t in ep_transitions]),
                env_id=np.array([t["env_id"] for t in ep_transitions]),
                step_idx=np.array([t["step"] for t in ep_transitions]),
                success=np.array([success]),
            )
            global_ep_idx += 1
            
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
            print(f"  [{pos_name}] Done: {pos_success}/{args.num_episodes_per_env} success")
    
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
            "total_episodes": total_episodes + skip_episodes,
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
            "resumed_from": skip_episodes,
            "time_seconds": elapsed,
        }, f, indent=2)
    print(f"Result: {meta_path}")
    
    wrapper.close()
    print("Done.")


if __name__ == "__main__":
    main()
