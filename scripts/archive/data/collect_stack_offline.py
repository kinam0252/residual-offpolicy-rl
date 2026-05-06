"""Collect offline Stack Cube data using GR00T base policy (residual=0).

Runs base policy rollouts and saves per-episode .npz files with transitions
and reward features for offline RL and ActionScaler calibration.

Usage:
    python scripts/collect_stack_offline.py \
        --groot_checkpoint ~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000 \
        --num_episodes 66 \
        --output_dir outputs/offline_stack_66ep
"""
import sys, os, time, argparse, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ['MUJOCO_GL'] = 'egl'

import torch
torch.backends.cuda.enable_cudnn_sdp(False)

# Deepspeed mock
import importlib, types as _types
if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed"); _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__ = []; _ds.__file__ = __file__
    _dz = _types.ModuleType("deepspeed.zero")
    _dz.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _dz.Init = lambda *a, **kw: (lambda f: f)
    _ds.zero = _dz
    sys.modules["deepspeed"] = _ds
    sys.modules["deepspeed.zero"] = _dz
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except:
    pass

import numpy as np
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import (
    MuJoCoVecEnvStack, CUBE_HALF, get_tcp_pose,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--num_episodes", type=int, default=66)
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--output_dir", type=str, default="outputs/offline_stack_66ep")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ema_alpha", type=float, default=0.9)
    p.add_argument("--num_envs", type=int, default=1,
                   help="Parallel envs (1 for serial collection)")
    p.add_argument("--white_pos", type=float, nargs=3, default=[0.45, 0.22, 0.02])
    p.add_argument("--green_pos", type=float, nargs=3, default=[0.45, -0.06, 0.02])
    return p.parse_args()


def extract_features(vec_env, env_idx=0):
    """Extract reward-relevant features from env state."""
    env = vec_env._envs[env_idx]
    model, data, ids = env["model"], env["data"], env["ids"]
    qa = env["white_qposadr"]

    white_pos = data.qpos[qa:qa + 3].copy()
    tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
    green_pos = vec_env._get_green_pos(env_idx)
    grasped = env["grasp_state"]["grasped"]

    tcp_white_dist = float(np.linalg.norm(tcp_pos - white_pos))

    green_top = green_pos.copy()
    green_top[2] += CUBE_HALF * 2
    white_to_green_top_3d = float(np.linalg.norm(white_pos - green_top))
    white_green_xy_dist = float(np.linalg.norm(white_pos[:2] - green_pos[:2]))

    # Contact between white and green
    import mujoco
    white_gid = env["white_geom_id"]
    green_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_cube_geom")
    white_green_contact = False
    for ci in range(data.ncon):
        c = data.contact[ci]
        if (c.geom1 == white_gid and c.geom2 == green_gid) or \
           (c.geom1 == green_gid and c.geom2 == white_gid):
            white_green_contact = True
            break

    return {
        "tcp_white_dist": np.float32(tcp_white_dist),
        "white_to_green_top_3d": np.float32(white_to_green_top_3d),
        "white_green_xy_dist": np.float32(white_green_xy_dist),
        "white_z": np.float32(white_pos[2]),
        "green_z": np.float32(green_pos[2]),
        "grasped": bool(grasped),
        "white_green_contact": bool(white_green_contact),
        "white_pos": white_pos.astype(np.float32),
        "green_pos": green_pos.astype(np.float32),
        "tcp_pos": tcp_pos.astype(np.float32),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Stack Cube Offline Collection ===")
    print(f"  Model: {args.groot_checkpoint}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_episode_steps}")
    print(f"  Output: {args.output_dir}")
    print(f"  White pos: {args.white_pos}, Green pos: {args.green_pos}")

    # Create env + wrapper
    vec_env = MuJoCoVecEnvStack(
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        reward_type="dense",
        white_cube_positions=[args.white_pos],
        green_cube_positions=[args.green_pos],
    )
    wrapper = MuJoCoResidualWrapperStack(
        vec_env=vec_env,
        groot_checkpoint=args.groot_checkpoint,
        ema_alpha=args.ema_alpha,
        policy_device=args.device,
    )

    total_success = 0
    total_transitions = 0
    t_start = time.time()

    for ep_idx in range(args.num_episodes):
        ep_name = f"ep{ep_idx:03d}"
        npz_path = out_dir / f"{ep_name}.npz"
        lock_dir = out_dir / f"{ep_name}.lock"

        # Skip if already done
        if npz_path.exists():
            print(f"  [ep {ep_idx}] already exists, skipping")
            continue

        # mkdir lock for multi-worker safety
        try:
            lock_dir.mkdir(parents=True)
        except FileExistsError:
            lock_info = lock_dir / "info"
            if lock_info.exists():
                try:
                    lock_time = int(lock_info.read_text().split("@")[2])
                    if time.time() - lock_time > 3600:
                        shutil.rmtree(lock_dir, ignore_errors=True)
                        lock_dir.mkdir(parents=True)
                    else:
                        continue
                except:
                    continue
            else:
                continue

        (lock_dir / "info").write_text(f"{os.getpid()}@{os.uname().nodename}@{int(time.time())}")

        obs, info = wrapper.reset()
        ep_transitions = []
        success = False

        for step in range(args.max_episode_steps):
            residual = torch.zeros((args.num_envs, 7), dtype=torch.float32)
            prev_obs = {
                k: v[0].clone() if isinstance(v, torch.Tensor) else v
                for k, v in obs.items()
            }

            next_obs, reward, terminated, truncated, info = wrapper.step(residual)
            r_env = float(reward[0]) if hasattr(reward, "__getitem__") else float(reward)

            features = extract_features(vec_env, 0)

            transition = {
                "obs_state": prev_obs.get(
                    "observation.state", torch.zeros(10)
                ).cpu().numpy().astype(np.float32),
                "obs_base_action": prev_obs.get(
                    "observation.base_action", torch.zeros(7)
                ).cpu().numpy().astype(np.float32),
                "action": residual[0].numpy().astype(np.float32),
                "reward": np.float32(r_env),
                "done": bool(terminated[0] or truncated[0]),
                "terminated": bool(terminated[0]),
                "step": step,
                # Reward features
                "tcp_white_dist": features["tcp_white_dist"],
                "white_to_green_top_3d": features["white_to_green_top_3d"],
                "white_green_xy_dist": features["white_green_xy_dist"],
                "white_z": features["white_z"],
                "green_z": features["green_z"],
                "grasped": features["grasped"],
                "white_green_contact": features["white_green_contact"],
                "white_pos": features["white_pos"],
                "green_pos": features["green_pos"],
                "tcp_pos": features["tcp_pos"],
            }
            ep_transitions.append(transition)
            obs = next_obs

            if terminated[0]:
                success = True
                break
            if truncated[0]:
                break

        if success:
            total_success += 1
        total_transitions += len(ep_transitions)

        # Save episode npz
        np.savez_compressed(
            npz_path,
            obs_state=np.array([t["obs_state"] for t in ep_transitions]),
            obs_base_action=np.array([t["obs_base_action"] for t in ep_transitions]),
            action=np.array([t["action"] for t in ep_transitions]),
            reward=np.array([t["reward"] for t in ep_transitions]),
            done=np.array([t["done"] for t in ep_transitions]),
            terminated=np.array([t["terminated"] for t in ep_transitions]),
            step_idx=np.array([t["step"] for t in ep_transitions]),
            success=np.array([success]),
            num_steps=np.array([len(ep_transitions)]),
            # Reward features for offline recomputation
            tcp_white_dist=np.array([t["tcp_white_dist"] for t in ep_transitions]),
            white_to_green_top_3d=np.array([t["white_to_green_top_3d"] for t in ep_transitions]),
            white_green_xy_dist=np.array([t["white_green_xy_dist"] for t in ep_transitions]),
            white_z=np.array([t["white_z"] for t in ep_transitions]),
            green_z=np.array([t["green_z"] for t in ep_transitions]),
            grasped=np.array([t["grasped"] for t in ep_transitions]),
            white_green_contact=np.array([t["white_green_contact"] for t in ep_transitions]),
            white_pos=np.array([t["white_pos"] for t in ep_transitions]),
            green_pos=np.array([t["green_pos"] for t in ep_transitions]),
            tcp_pos=np.array([t["tcp_pos"] for t in ep_transitions]),
        )

        # Release lock
        shutil.rmtree(lock_dir, ignore_errors=True)

        tag = "SUCC" if success else "FAIL"
        sr = total_success / (ep_idx + 1) * 100
        elapsed = time.time() - t_start
        print(f"  [{tag}] ep={ep_idx:3d} steps={step+1:3d} r_final={r_env:.3f} "
              f"SR={sr:.1f}% ({total_success}/{ep_idx+1}) "
              f"trans={total_transitions} t={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== Done ===")
    print(f"  Episodes: {args.num_episodes}, Success: {total_success} "
          f"({total_success/max(args.num_episodes,1)*100:.1f}%)")
    print(f"  Total transitions: {total_transitions}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(args.num_episodes,1):.1f}s/ep)")
    print(f"  Output: {args.output_dir}")

    try:
        vec_env.close()
    except:
        pass


if __name__ == "__main__":
    main()
