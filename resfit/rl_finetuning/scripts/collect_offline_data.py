"""Collect offline data with dense_clipped reward using GR00T base policy.

Runs GR00T (zero residual) across perturbation table positions,
records (obs, action, reward, next_obs, done) transitions with
pre-computed dense_clipped reward components, and saves as
TensorDict-compatible format.

Usage:
    python collect_offline_data.py \
        --groot_checkpoint ~/DATA/.../checkpoint-30000 \
        --perturb_table configs/mujoco_cube_perturb_table.json \
        --num_episodes_per_env 10 \
        --output_dir outputs/offline_data
"""
import sys, os, json, time, argparse
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
    p.add_argument("--perturb_table", type=str, default="configs/mujoco_cube_perturb_table.json")
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


def main():
    args = parse_args()
    
    # Load perturbation table
    with open(args.perturb_table) as f:
        perturb = json.load(f)
    
    base_pos = perturb["base_pos"]
    envs_config = perturb["envs"]
    
    # Filter by env_ids if specified
    if args.env_ids is not None:
        envs_config = [ec for ec in envs_config if ec["env_id"] in args.env_ids]
    
    n_positions = len(envs_config)
    
    print(f"=== Offline Data Collection ===")
    print(f"Positions: {n_positions}")
    print(f"Episodes/position: {args.num_episodes_per_env}")
    print(f"Total episodes: {n_positions * args.num_episodes_per_env}")
    print(f"Max steps/episode: {args.max_episode_steps}")
    print()
    
    # Build cube positions
    cube_positions = []
    for ec in envs_config:
        pos = [base_pos[0] + ec["dx"], base_pos[1] + ec["dy"], base_pos[2]]
        cube_positions.append(pos)
        print(f"  env {ec['env_id']:2d} ({ec['name']:>16s}): cube=({pos[0]:.3f}, {pos[1]:.3f})")
    
    # Create env (1 at a time for sequential collection)
    # Parse random cube range
    _rcr_parsed = None
    if args.random_cube_range:
        import json as _json2
        _rcr = _json2.loads(args.random_cube_range)
        _rcr_parsed = {"dx": (_rcr["dx"][0]/100.0, _rcr["dx"][1]/100.0), "dy": (_rcr["dy"][0]/100.0, _rcr["dy"][1]/100.0), "yaw": (float(_rcr.get("yaw",[0,0])[0]), float(_rcr.get("yaw",[0,0])[1]))}
        print(f"  Random cube range: {_rcr_parsed}")

    print("\nCreating environment...")
    mujoco_env = MuJoCoVecEnv(
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
    
    # Output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_transitions = []
    total_success = 0
    total_episodes = 0
    
    for pos_idx, ec in enumerate(envs_config):
        pos_name = ec["name"]
        pos = cube_positions[pos_idx]
        mujoco_env._envs[0]["cube_pos_init"] = np.array(pos, dtype=np.float64)
        
        pos_success = 0
        
        for ep in range(args.num_episodes_per_env):
            obs, _ = wrapper.reset()
            ep_transitions = []
            ep_reward = 0.0
            
            for step in range(args.max_episode_steps):
                # Zero residual action
                residual = torch.zeros((1, 7), dtype=torch.float32)
                
                # Get current obs for storing
                prev_obs = {k: v[0].clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
                
                next_obs, reward, terminated, truncated, info = wrapper.step(residual)
                
                # Compute detailed reward
                r_total, r_debug = compute_dense_clipped_reward_debug(
                    mujoco_env, 0, success_threshold=0.03)
                
                # Debug print for first few steps of first episode
                if args.debug and ep == 0 and (step % 20 == 0 or step >= 100):
                    print(f"  [{pos_name}] s={step:3d} dist={r_debug['finger_cube_dist']:.4f} "
                          f"grip={r_debug['grip_width']:.3f} grasped={r_debug['grasped']} "
                          f"lift={r_debug['lift_cm']:.2f}cm "
                          f"r_dist={r_debug['reward_distance']:.4f} "
                          f"r_cont={r_debug['reward_contact']:.4f} "
                          f"r_hgt={r_debug['reward_height']:.4f} "
                          f"r_succ={r_debug['reward_success']:.4f} "
                          f"total={r_debug['reward_total']:.4f}")
                
                ep_reward += r_total
                
                # Store transition
                # Depth images (1, 84, 84) float32
                depth_front = prev_obs.get("observation.depth.front", torch.zeros(1, 84, 84)).numpy()
                depth_wrist = prev_obs.get("observation.depth.wrist", torch.zeros(1, 84, 84)).numpy()

                transition = {
                    "obs_state": prev_obs.get("observation.state", torch.zeros(10)).numpy(),
                    "obs_base_action": prev_obs.get("observation.base_action", torch.zeros(7)).numpy(),
                    "obs_object_state": prev_obs.get("observation.object_state", torch.zeros(7)).numpy(),
                    "obs_depth_front": depth_front,
                    "obs_depth_wrist": depth_wrist,
                    "action": residual[0].numpy(),  # 7D zero residual
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
            
            all_transitions.extend(ep_transitions)
            
            tag = "SUCC" if success else "FAIL"
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  [{pos_name}] ep={ep+1}/{args.num_episodes_per_env} {tag} "
                      f"steps={step+1} reward={ep_reward:.1f} "
                      f"pos_sr={pos_success}/{ep+1}")
        
        print(f"  [{pos_name}] Done: {pos_success}/{args.num_episodes_per_env} success")
    
    # Save
    print(f"\n{'='*60}")
    print(f"Total: {total_episodes} episodes, {len(all_transitions)} transitions")
    print(f"Success rate: {total_success}/{total_episodes} = {total_success/max(1,total_episodes):.0%}")
    
    # Save as numpy arrays
    save_path = out_dir / f"offline_data_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    
    # Convert to arrays
    np.savez_compressed(
        save_path,
        obs_state=np.array([t["obs_state"] for t in all_transitions]),
        obs_base_action=np.array([t["obs_base_action"] for t in all_transitions]),
        obs_object_state=np.array([t["obs_object_state"] for t in all_transitions]),
        obs_depth_front=np.array([t["obs_depth_front"] for t in all_transitions]),
        obs_depth_wrist=np.array([t["obs_depth_wrist"] for t in all_transitions]),
        action=np.array([t["action"] for t in all_transitions]),
        reward=np.array([t["reward"] for t in all_transitions]),
        reward_distance=np.array([t["reward_distance"] for t in all_transitions]),
        reward_contact=np.array([t["reward_contact"] for t in all_transitions]),
        reward_height=np.array([t["reward_height"] for t in all_transitions]),
        reward_success=np.array([t["reward_success"] for t in all_transitions]),
        done=np.array([t["done"] for t in all_transitions]),
        terminated=np.array([t["terminated"] for t in all_transitions]),
        env_id=np.array([t["env_id"] for t in all_transitions]),
        step=np.array([t["step"] for t in all_transitions]),
    )
    print(f"Saved: {save_path} ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Save metadata
    meta_path = out_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime('%Y%m%d_%H%M%S'),
            "total_episodes": total_episodes,
            "total_transitions": len(all_transitions),
            "success_rate": total_success / max(1, total_episodes),
            "n_success": total_success,
            "episodes_per_env": args.num_episodes_per_env,
            "max_episode_steps": args.max_episode_steps,
            "perturb_table": args.perturb_table,
            "checkpoint": args.groot_checkpoint,
            "reward_type": "dense",
            "success_threshold": 0.03,
            "ema_alpha": args.ema_alpha,
        }, f, indent=2)
    print(f"Metadata: {meta_path}")
    
    wrapper.close()
    print("Done.")


if __name__ == "__main__":
    main()
