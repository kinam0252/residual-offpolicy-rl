"""Collect offline data using GR00T base policy for Stand Cup.

Runs GR00T (zero residual) across cup positions,
records (obs, action, reward, next_obs, done) transitions, and saves as
per-episode .npz files compatible with the TD3 training script.

Usage:
    python collect_offline_data_cup.py \
        --groot_checkpoint ~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-200000 \
        --num_episodes_per_env 100 \
        --output_dir outputs/offline_cup_27ep
"""
import sys, os, json, time, argparse
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

from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import (
    MuJoCoVecEnvCup, CUP_HEIGHT, get_tcp_pose,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_cup import MuJoCoResidualWrapperCup


def extract_features(mujoco_env, env_idx=0):
    """Extract cup-specific reward features from current env state."""
    env = mujoco_env._envs[env_idx]
    model, data, ids = env["model"], env["data"], env["ids"]

    # TCP pose
    tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
    tcp_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat().astype(np.float32)

    # Cup state via env dict
    cup_jnt_id = env.get("cup_jnt_id", -1)
    cup_qpa = env.get("cup_qposadr")
    cup_dof = env.get("cup_dofadr")
    cup_pos = np.zeros(3, dtype=np.float32)
    cup_quat_wxyz = np.array([1, 0, 0, 0], dtype=np.float32)
    uprightness = 0.0
    cup_vel = 0.0

    if cup_qpa is not None:
        cup_pos = data.qpos[cup_qpa:cup_qpa + 3].copy().astype(np.float32)
        cup_quat_wxyz = data.qpos[cup_qpa + 3:cup_qpa + 7].copy().astype(np.float32)
    
    # Uprightness via vec_env method
    uprightness = mujoco_env._get_uprightness(env_idx)
    
    if cup_dof is not None:
        cup_vel = float(np.linalg.norm(data.qvel[cup_dof:cup_dof + 6]))

    # Grasp state (now contact-based)
    grasped = env.get("grasp_state", {}).get("grasped", False)

    # TCP to cup distance
    tcp_cup_dist = float(np.linalg.norm(tcp_pos - cup_pos))

    # Gripper width
    finger_ids = ids.get("finger_ids", [])
    grip_width = 0.04
    for fid in finger_ids:
        if fid >= 0:
            grip_width = float(data.qpos[model.jnt_qposadr[fid]])
            break

    return {
        "tcp_pos": tcp_pos.astype(np.float32),
        "tcp_quat_xyzw": tcp_quat_xyzw.astype(np.float32),
        "cup_pos": cup_pos,
        "cup_quat_wxyz": cup_quat_wxyz,
        "uprightness": np.float32(uprightness),
        "cup_vel": np.float32(cup_vel),
        "cup_z": np.float32(cup_pos[2]),
        "tcp_cup_dist": np.float32(tcp_cup_dist),
        "grasped": np.bool_(grasped),
        "grip_width": np.float32(grip_width),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--cup_positions_path", type=str, default=None,
                   help="cup_positions.json path (default: auto-detect)")
    p.add_argument("--episode_ids", type=int, nargs="+", default=None,
                   help="Specific episode IDs to collect (default: all 27)")
    p.add_argument("--num_episodes_per_env", type=int, default=100,
                   help="Rollouts per cup position")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--output_dir", type=str, default="outputs/offline_cup_27ep")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--calib_path", type=str, default=None)
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument("--resume", action="store_true",
                   help="Skip already-saved episodes")
    return p.parse_args()


def main():
    args = parse_args()

    # Determine episodes
    if args.episode_ids is not None:
        episode_ids = args.episode_ids
    else:
        episode_ids = list(range(27))  # 27 cup episodes

    num_envs = len(episode_ids)
    total_episodes_target = num_envs * args.num_episodes_per_env
    print(f"=== Stand Cup Offline Data Collection (BATCH) ===")
    print(f"  Cup positions: {num_envs} episodes (parallel envs)")
    print(f"  Rollouts/position: {args.num_episodes_per_env}")
    print(f"  Total episodes: {total_episodes_target}")
    print(f"  Max steps/episode: {args.max_episode_steps}")
    print(f"  GR00T: {args.groot_checkpoint}")
    print()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Create env with ALL positions at once ──
    env_kwargs = dict(
        num_envs=num_envs,
        episode_ids=episode_ids,
        max_episode_steps=args.max_episode_steps,
        reward_type="dense",
        device=args.device,
    )
    if args.cup_positions_path:
        env_kwargs["cup_positions_path"] = args.cup_positions_path
    if args.calib_path:
        env_kwargs["calib_path"] = args.calib_path

    print("Creating Cup environment (all positions batch)...")
    mujoco_env = MuJoCoVecEnvCup(**env_kwargs)

    print("Loading GR00T policy...")
    wrapper = MuJoCoResidualWrapperCup(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag='NEW_EMBODIMENT',
        policy_device=args.device,
        task_description="Pick up the cup lying on its side and stand it upright",
        ema_alpha=args.ema_alpha,
    )
    print("Ready.\n")

    total_success = 0
    total_collected = 0
    t_start = time.time()

    # Per-env transition buffers
    env_transitions = [[] for _ in range(num_envs)]
    env_rollout_idx = [0] * num_envs  # current rollout index per env
    env_done_count = [0] * num_envs   # completed rollouts per env
    envs_active = list(range(num_envs))  # envs still collecting

    # Check resume: skip already-saved episodes
    if args.resume:
        for i, ep_id in enumerate(episode_ids):
            while env_rollout_idx[i] < args.num_episodes_per_env:
                ep_global = i * args.num_episodes_per_env + env_rollout_idx[i]
                ep_path = out_dir / f"ep{ep_global:04d}.npz"
                if ep_path.exists():
                    env_rollout_idx[i] += 1
                    env_done_count[i] += 1
                    total_collected += 1
                else:
                    break
        envs_active = [i for i in range(num_envs) if env_rollout_idx[i] < args.num_episodes_per_env]
        if total_collected > 0:
            print(f"  Resume: skipping {total_collected} already-saved episodes")
        if not envs_active:
            print("All episodes already collected!")
            return

    obs, _ = wrapper.reset()

    while envs_active:
        # Step all envs with zero residual (pure GR00T)
        residual = torch.zeros((num_envs, 7), dtype=torch.float32)

        # Extract features BEFORE step for all envs
        feats = [extract_features(mujoco_env, i) for i in range(num_envs)]

        next_obs, reward, terminated, truncated, info = wrapper.step(residual)
        done = terminated | truncated

        # Store transitions for active envs
        for i in envs_active:
            feat = feats[i]
            transition = {
                "obs_state": obs["observation.state"][i].cpu().numpy(),
                "obs_base_action": obs.get("observation.base_action", torch.zeros(num_envs, 7))[i].cpu().numpy(),
                "obs_object_state": obs["observation.object_state"][i].cpu().numpy(),
                "action": residual[i].numpy(),
                "tcp_pos": feat["tcp_pos"],
                "cup_pos": feat["cup_pos"],
                "cup_quat_wxyz": feat["cup_quat_wxyz"],
                "uprightness": feat["uprightness"],
                "cup_vel": feat["cup_vel"],
                "cup_z": feat["cup_z"],
                "tcp_cup_dist": feat["tcp_cup_dist"],
                "grasped": feat["grasped"],
                "grip_width": feat["grip_width"],
                "reward": float(reward[i].item()),
                "done": bool(done[i].item()),
                "terminated": bool(terminated[i].item()),
                "step": len(env_transitions[i]),
            }
            env_transitions[i].append(transition)

        obs = next_obs

        # Handle episode completions
        done_envs = [i for i in envs_active if done[i].item()]
        if done_envs:
            reset_ids = []
            for i in done_envs:
                ep_id = episode_ids[i]
                rollout = env_rollout_idx[i]
                ep_global = i * args.num_episodes_per_env + rollout
                success = bool(terminated[i].item())
                n_steps = len(env_transitions[i])

                # Save episode
                ep_save_path = out_dir / f"ep{ep_global:04d}.npz"
                trans = env_transitions[i]
                np.savez_compressed(
                    ep_save_path,
                    obs_state=np.array([t["obs_state"] for t in trans], dtype=np.float32),
                    obs_base_action=np.array([t["obs_base_action"] for t in trans], dtype=np.float32),
                    obs_object_state=np.array([t["obs_object_state"] for t in trans], dtype=np.float32),
                    action=np.array([t["action"] for t in trans], dtype=np.float32),
                    reward=np.array([t["reward"] for t in trans], dtype=np.float32),
                    tcp_pos=np.array([t["tcp_pos"] for t in trans], dtype=np.float32),
                    cup_pos=np.array([t["cup_pos"] for t in trans], dtype=np.float32),
                    cup_quat_wxyz=np.array([t["cup_quat_wxyz"] for t in trans], dtype=np.float32),
                    uprightness=np.array([t["uprightness"] for t in trans], dtype=np.float32),
                    cup_vel=np.array([t["cup_vel"] for t in trans], dtype=np.float32),
                    cup_z=np.array([t["cup_z"] for t in trans], dtype=np.float32),
                    tcp_cup_dist=np.array([t["tcp_cup_dist"] for t in trans], dtype=np.float32),
                    grasped=np.array([t["grasped"] for t in trans]),
                    grip_width=np.array([t["grip_width"] for t in trans], dtype=np.float32),
                    done=np.array([t["done"] for t in trans]),
                    terminated=np.array([t["terminated"] for t in trans]),
                    step_idx=np.array([t["step"] for t in trans], dtype=np.int64),
                    num_steps=np.array([n_steps], dtype=np.int64),
                    success=np.array([success]),
                    episode_id=np.array([ep_id], dtype=np.int64),
                )

                if success:
                    total_success += 1
                total_collected += 1
                env_done_count[i] += 1

                tag = "SUCC" if success else "FAIL"
                sr = total_success / total_collected * 100
                elapsed = time.time() - t_start
                feat_last = trans[-1]
                print(f"  ep={total_collected:3d}/{total_episodes_target} "
                      f"[pos{ep_id:02d} roll{rollout:02d}] {tag} "
                      f"steps={n_steps:3d} upright={feat_last['uprightness']:.3f} "
                      f"grasped={feat_last['grasped']} "
                      f"SR={sr:5.1f}% ({total_success}/{total_collected}) "
                      f"t={elapsed:.0f}s", flush=True)

                # Advance to next rollout
                env_transitions[i] = []
                env_rollout_idx[i] += 1

                if env_rollout_idx[i] < args.num_episodes_per_env:
                    reset_ids.append(i)
                # else: this env is done

            # Reset completed envs that still have rollouts remaining
            if reset_ids:
                mujoco_env.reset_envs(reset_ids)
                # Re-get obs after reset (wrapper handles GR00T obs internally)

            # Update active envs
            envs_active = [i for i in range(num_envs) if env_rollout_idx[i] < args.num_episodes_per_env]

    # Final summary
    elapsed = time.time() - t_start
    sr = total_success / max(1, total_collected) * 100
    print(f"\n{'='*60}")
    print(f"FINAL: {total_collected} episodes collected")
    print(f"SR={sr:.1f}% ({total_success}/{total_collected})")
    print(f"Time: {elapsed:.0f}s ({elapsed/max(1,total_collected):.1f}s/ep)")

    meta_path = out_dir / "result.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime('%Y%m%d_%H%M%S'),
            "total_episodes": total_collected,
            "success_rate": sr,
            "n_success": total_success,
            "avg_steps": args.max_episode_steps,
            "episodes_per_env": args.num_episodes_per_env,
            "max_episode_steps": args.max_episode_steps,
            "checkpoint": args.groot_checkpoint,
            "num_envs_batch": num_envs,
        }, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    mujoco_env.close()


if __name__ == "__main__":
    main()
