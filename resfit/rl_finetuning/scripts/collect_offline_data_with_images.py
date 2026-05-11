"""Collect offline data WITH RGB images using GR00T base policy (unified wrapper).

Runs GR00T (zero residual) across perturbation positions,
records (obs_state, obs_base_action, obs_object_state, RGB images, reward, done)
and saves as per-episode .npz files compatible with _load_offline_data().

The key addition vs the state-only collector: saves RGB images from all
cameras defined in the task's CAMERA_MAP (e.g. back + wrist).

Usage:
    python collect_offline_data_with_images.py \
        --task pnp \
        --groot_checkpoint ~/DATA/.../checkpoint-100000 \
        --episode_positions_file configs/pnp_train_30.json \
        --num_episodes_per_env 1 \
        --output_dir outputs/offline_data/pnp_img_test \
        --rl_img_size 84
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

from resfit.rl_finetuning.configs.task_configs import get_task_config
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["pnp", "stack", "lift", "drawer", "cup"])
    p.add_argument("--groot_checkpoint", type=str, default=None)
    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--num_episodes_per_env", type=int, default=1)
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--output_dir", type=str, default="outputs/offline_data/img_test")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--rl_img_size", type=int, default=84)
    p.add_argument("--episode_positions_file", type=str, required=True)
    p.add_argument("--task_description", type=str, default=None)
    p.add_argument("--random_cube_range", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    task_cfg = get_task_config(args.task)

    # Default GR00T checkpoint from task config
    if args.groot_checkpoint is None:
        args.groot_checkpoint = os.path.expanduser(task_cfg.groot_checkpoint_hint)
    if args.task_description is None:
        args.task_description = task_cfg.task_description

    # Image keys from task config
    image_keys = task_cfg.rl_image_keys
    print(f"[collect] Task: {args.task}")
    print(f"[collect] Image keys: {image_keys}")
    print(f"[collect] Image size: {args.rl_img_size}")

    # Load positions
    all_positions = json.load(open(args.episode_positions_file))
    # Deduplicate by episode key (if present) or use all positions
    if all_positions and "episode" in all_positions[0]:
        seen_eps = set()
        unique_positions = []
        for p in all_positions:
            if p["episode"] not in seen_eps:
                seen_eps.add(p["episode"])
                unique_positions.append(p)
    else:
        unique_positions = all_positions

    # Random cube perturbation
    _rcr_parsed = None
    rng = np.random.default_rng()
    if args.random_cube_range:
        _raw_rcr = json.loads(args.random_cube_range)
        def _cvt_one(r):
            if r is None: return None
            return {"dx": (r["dx"][0]/100.0, r["dx"][1]/100.0),
                    "dy": (r["dy"][0]/100.0, r["dy"][1]/100.0),
                    "yaw": (float(r.get("yaw",[0,0])[0]), float(r.get("yaw",[0,0])[1]))}
        if isinstance(_raw_rcr, list):
            _rcr_parsed = [_cvt_one(item) for item in _raw_rcr]
        else:
            _rcr_parsed = [_cvt_one(_raw_rcr)]
        print(f"[collect] Random cube range: {args.random_cube_range}")

    num_positions = len(unique_positions)
    total_episodes_target = num_positions * args.num_episodes_per_env
    print(f"[collect] Positions: {num_positions}, Episodes/pos: {args.num_episodes_per_env}")
    print(f"[collect] Total episodes: {total_episodes_target}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create VecEnv (task-specific)
    first_ep = unique_positions[0]
    mod = importlib.import_module(task_cfg.vec_env_module)
    VecEnvClass = getattr(mod, task_cfg.vec_env_class)

    env_kwargs = dict(num_envs=1, max_episode_steps=args.max_episode_steps,
                      rl_img_size=args.rl_img_size)

    if args.task == "pnp":
        env_kwargs.update(cube_positions=[first_ep["cube_pos"]],
                          bowl_positions=[first_ep["bowl_pos"]],
                          reward_type="dense_clipped", success_threshold=0.095)
        if args.scene_xml:
            env_kwargs["scene_xml"] = args.scene_xml
    elif args.task == "stack":
        env_kwargs.update(white_cube_positions=[first_ep.get("white_cube_pos", [0.42, 0.0, 0.02])],
                          green_cube_positions=[first_ep.get("green_cube_pos", [0.42, 0.05, 0.02])])
    elif args.task == "lift":
        env_kwargs.update(cube_positions=[first_ep.get("cube_pos", [0.42, 0.0, 0.02])])

    print("[collect] Creating environment...")
    mujoco_env = VecEnvClass(**env_kwargs)

    # Camera keys mapping for unified wrapper
    camera_keys = None
    if hasattr(mujoco_env, 'CAMERA_MAP'):
        camera_keys = mujoco_env.CAMERA_MAP

    print("[collect] Loading GR00T policy...")
    wrapper = MuJoCoResidualWrapperUnified(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag='NEW_EMBODIMENT',
        policy_device=args.device,
        task_description=args.task_description,
        camera_keys=camera_keys,
        use_images=True,  # key: enable RGB rendering
        async_prefetch=False,  # EGL thread safety
    )
    print("[collect] Ready.\n")

    REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])

    total_success = 0
    total_episodes = 0
    ep_rewards = []
    ep_steps_list = []
    total_img_bytes = 0
    total_state_bytes = 0
    t_start = time.time()

    for pos_idx, pos_info in enumerate(unique_positions):
        for ep in range(args.num_episodes_per_env):
            ep_global = pos_idx * args.num_episodes_per_env + ep

            ep_save_path = out_dir / f"ep_{ep_global:04d}.npz"
            if ep_save_path.exists() and args.resume:
                continue

            # Reset
            obs, _ = wrapper.reset()

            # Override env state (PnP-specific position setup)
            if args.task == "pnp":
                env_data = mujoco_env._envs[0]
                data = env_data["data"]
                model = env_data["model"]
                cqa = env_data["cube_qposadr"]
                bowl_body_id = env_data["bowl_body_id"]

                for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
                    data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
                for fid in env_data["ids"]["finger_ids"]:
                    if fid >= 0:
                        data.qpos[model.jnt_qposadr[fid]] = 0.04

                base_cube_pos = np.array(pos_info["cube_pos"], dtype=np.float64)
                bowl_pos = np.array(pos_info["bowl_pos"], dtype=np.float64)
                actual_cube_pos = base_cube_pos.copy()
                yaw_deg = pos_info.get("cube_yaw_deg", 0.0)

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
                prev_obs = {k: v[0].clone() if isinstance(v, torch.Tensor) else v
                            for k, v in obs.items()}

                next_obs, reward, terminated, truncated, info = wrapper.step(residual)

                r_env = float(reward[0]) if hasattr(reward, "__getitem__") else float(reward)
                ep_reward += r_env

                def _to_np(t):
                    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)

                transition = {
                    "obs_state": _to_np(prev_obs.get("observation.state", torch.zeros(10))),
                    "obs_base_action": _to_np(prev_obs.get("observation.base_action", torch.zeros(7))),
                    "action": np.zeros(7, dtype=np.float32),  # zero residual
                    "reward": np.float32(r_env),
                    "done": bool(terminated[0] or truncated[0]),
                    "terminated": bool(terminated[0]),
                    "env_id": pos_idx,
                    "step": step,
                }

                # Object state
                obj_key = "observation.object_state"
                if obj_key in prev_obs:
                    transition["obs_object_state"] = _to_np(prev_obs[obj_key])

                # RGB images - save with npz-compatible key names
                for img_key in image_keys:
                    npz_key = img_key.replace(".", "_")
                    if img_key in prev_obs:
                        img_data = _to_np(prev_obs[img_key])
                        transition[npz_key] = img_data.astype(np.uint8)
                    else:
                        transition[npz_key] = np.zeros(
                            (3, args.rl_img_size, args.rl_img_size), dtype=np.uint8)

                ep_transitions.append(transition)

                obs = next_obs
                done = terminated | truncated
                if done[0]:
                    break

            success = bool(terminated[0])
            if success:
                total_success += 1
            total_episodes += 1
            ep_rewards.append(ep_reward)
            ep_steps_list.append(step + 1)

            # Build save dict
            save_dict = {
                "obs_state": np.array([t["obs_state"] for t in ep_transitions]),
                "obs_base_action": np.array([t["obs_base_action"] for t in ep_transitions]),
                "action": np.array([t["action"] for t in ep_transitions]),
                "reward": np.array([t["reward"] for t in ep_transitions]),
                "done": np.array([t["done"] for t in ep_transitions]),
                "terminated": np.array([t["terminated"] for t in ep_transitions]),
                "env_id": np.array([t["env_id"] for t in ep_transitions]),
                "step_idx": np.array([t["step"] for t in ep_transitions]),
                "success": np.array([success]),
            }
            if "obs_object_state" in ep_transitions[0]:
                save_dict["obs_object_state"] = np.array(
                    [t["obs_object_state"] for t in ep_transitions])

            # Add image arrays
            for img_key in image_keys:
                npz_key = img_key.replace(".", "_")
                arr = np.array([t[npz_key] for t in ep_transitions])
                save_dict[npz_key] = arr
                total_img_bytes += arr.nbytes

            # Track state bytes
            total_state_bytes += save_dict["obs_state"].nbytes
            total_state_bytes += save_dict["obs_base_action"].nbytes

            # Save
            ep_tmp_path = out_dir / f"ep_{ep_global:04d}.tmp.npz"
            np.savez_compressed(ep_tmp_path, **save_dict)
            ep_tmp_path.rename(ep_save_path)

            file_size_mb = ep_save_path.stat().st_size / (1024 * 1024)
            n_trans = len(ep_transitions)

            tag = "SUCC" if success else "FAIL"
            sr = total_success / total_episodes * 100
            elapsed = time.time() - t_start
            print(f"  ep={total_episodes:3d}/{total_episodes_target} {tag} "
                  f"steps={step+1:3d} reward={ep_reward:7.1f} "
                  f"SR={sr:5.1f}% file={file_size_mb:.2f}MB ({n_trans} trans) "
                  f"t={elapsed:.0f}s", flush=True)

    # Final summary
    elapsed = time.time() - t_start
    sr = total_success / max(1, total_episodes) * 100
    total_trans = sum(ep_steps_list)

    # Disk usage
    total_disk_bytes = sum(f.stat().st_size for f in out_dir.glob("ep_*.npz"))
    total_disk_mb = total_disk_bytes / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"FINAL: {total_episodes} episodes, {total_trans} transitions")
    print(f"SR={sr:.1f}% ({total_success}/{total_episodes})")
    print(f"Avg reward: {np.mean(ep_rewards):.1f} | Avg steps: {np.mean(ep_steps_list):.0f}")
    print(f"Time: {elapsed:.0f}s ({elapsed/max(1,total_episodes):.1f}s/ep)")
    print()
    print(f"=== Storage Analysis ===")
    print(f"  Raw image bytes: {total_img_bytes / (1024**2):.1f} MB")
    print(f"  Raw state bytes: {total_state_bytes / (1024**2):.3f} MB")
    print(f"  Disk (npz compressed): {total_disk_mb:.1f} MB")
    print(f"  Per transition (raw): {total_img_bytes / max(1, total_trans) / 1024:.1f} KB (images)")
    print(f"  Per transition (disk): {total_disk_bytes / max(1, total_trans) / 1024:.1f} KB")
    print(f"  Compression ratio: {total_img_bytes / max(1, total_disk_bytes):.2f}x")

    # Save metadata
    meta = {
        "timestamp": time.strftime('%Y%m%d_%H%M%S'),
        "task": args.task,
        "total_episodes": total_episodes,
        "total_transitions": total_trans,
        "success_rate": sr,
        "image_keys": image_keys,
        "rl_img_size": args.rl_img_size,
        "raw_image_bytes": total_img_bytes,
        "disk_bytes": total_disk_bytes,
        "per_transition_raw_kb": total_img_bytes / max(1, total_trans) / 1024,
        "per_transition_disk_kb": total_disk_bytes / max(1, total_trans) / 1024,
        "time_seconds": elapsed,
    }
    with open(out_dir / "result.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nResult: {out_dir / 'result.json'}")

    wrapper.close()
    print("Done.")


if __name__ == "__main__":
    main()
