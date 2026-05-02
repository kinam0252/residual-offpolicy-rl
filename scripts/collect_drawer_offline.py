"""Collect offline drawer data with contact_z_gate — work-stealing pattern.

Each worker independently picks unclaimed episodes via mkdir locking.
Saves per-episode npz with transitions for offline RL.

Usage:
    python scripts/collect_drawer_offline.py \
        --groot_checkpoint ~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000 \
        --active_drawers 2 3 4 \
        --num_episodes 100 \
        --output_dir outputs/offline_drawer_zgate_100k
"""
import sys, os, json, time, argparse, shutil
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
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, TCP_OFFSET, DRAWER_SLOT_HEIGHT, DRAWER_GAP, DRAWER_SLIDE,
    DEMO_CONTACT_MEAN, DEMO_CONTACT_Z_BOUNDS,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--active_drawers", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--num_episodes", type=int, default=30,
                   help="Number of episodes per drawer")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--output_dir", type=str, default="outputs/offline_drawer_zgate_100k")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ema_alpha", type=float, default=0.9)
    p.add_argument("--contact_z_gate", action="store_true", default=True)
    p.add_argument("--no_contact_z_gate", dest="contact_z_gate", action="store_false")
    return p.parse_args()


def get_contact_info(vec_env, env_idx=0):
    """Get current TCP contact position and reward components."""
    env = vec_env._envs[env_idx]
    data, ids = env["data"], env["ids"]
    active_drawer = env["active_drawer"]
    face_hy = vec_env._face_hy
    face_hz = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP
    dz = env["drawer_z_min"] + face_hz + DRAWER_GAP

    hand_pos = data.xpos[ids["hand_id"]].copy()
    hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3).copy()
    tcp = hand_pos + hand_mat @ TCP_OFFSET
    cab_pos_w = data.xpos[env["cab_body_id"]].copy()
    cab_mat = data.xmat[env["cab_body_id"]].reshape(3, 3).copy()
    tcp_local = cab_mat.T @ (tcp - cab_pos_w)

    y_norm = tcp_local[1] / face_hy if face_hy > 0 else 0.0
    z_norm = (tcp_local[2] - dz) / face_hz if face_hz > 0 else 0.0

    # Check if within z gate bounds
    zb = DEMO_CONTACT_Z_BOUNDS.get(active_drawer, {})
    in_z_bounds = (zb.get("z_min", -np.inf) <= z_norm <= zb.get("z_max", np.inf))

    # Distance to demo mean contact position
    dm = DEMO_CONTACT_MEAN.get(active_drawer, {"y_norm": 0.0, "z_norm": 0.0})
    dist_to_demo_mean = float(np.sqrt((y_norm - dm["y_norm"])**2 + (z_norm - dm["z_norm"])**2))

    return {
        "y_norm": np.float32(y_norm),
        "z_norm": np.float32(z_norm),
        "tcp_local_x": np.float32(tcp_local[0]),
        "tcp_local_y": np.float32(tcp_local[1]),
        "tcp_local_z": np.float32(tcp_local[2]),
        "in_z_bounds": bool(in_z_bounds),
        "dist_to_demo_mean": np.float32(dist_to_demo_mean),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build task list: (drawer_id, ep_idx)
    tasks = []
    for d in args.active_drawers:
        for ep in range(args.num_episodes):
            tasks.append((d, ep))

    total_tasks = len(tasks)
    print(f"=== Drawer Offline Collection (contact_z_gate={args.contact_z_gate}) ===")
    print(f"  Worker PID: {os.getpid()}")
    print(f"  Model: {args.groot_checkpoint}")
    print(f"  Drawers: {args.active_drawers}, Episodes/drawer: {args.num_episodes}")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Max steps: {args.max_episode_steps}")
    print(f"  Output: {args.output_dir}")

    # We'll create env per-drawer (cache to avoid recreation)
    wrappers = {}  # drawer_id -> wrapper
    vec_envs = {}  # drawer_id -> vec_env

    def get_wrapper(drawer_id):
        if drawer_id not in wrappers:
            print(f"\n  Creating env for D{drawer_id}...")
            vec_env = MuJoCoVecEnvDrawer(
                num_envs=1,
                max_episode_steps=args.max_episode_steps,
                reward_type="dense",
                active_drawers=[drawer_id],
                contact_threshold=float("inf"),
                contact_z_gate=args.contact_z_gate,
            )
            wrapper = MuJoCoResidualWrapperDrawer(
                vec_env=vec_env,
                groot_checkpoint=args.groot_checkpoint,
                ema_alpha=args.ema_alpha,
                policy_device=args.device,
            )
            wrappers[drawer_id] = wrapper
            vec_envs[drawer_id] = vec_env
        return wrappers[drawer_id], vec_envs[drawer_id]

    LOCK_TIMEOUT = 3600
    completed = 0
    total_success = 0
    t_start = time.time()

    while True:
        found_work = False

        for drawer_id, ep_idx in tasks:
            ep_name = f"d{drawer_id}_ep{ep_idx:03d}"
            npz_path = out_dir / f"{ep_name}.npz"
            lock_dir = out_dir / f"{ep_name}.lock"

            # Already done?
            if npz_path.exists():
                continue

            # Try to claim via mkdir lock
            try:
                lock_dir.mkdir(parents=True)
            except FileExistsError:
                # Check stale lock
                lock_info = lock_dir / "info"
                if lock_info.exists():
                    try:
                        lock_time = int(lock_info.read_text().split("@")[2])
                        if time.time() - lock_time > LOCK_TIMEOUT:
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

            print(f"\n[Worker {os.getpid()}] === {ep_name} ===")

            wrapper, vec_env = get_wrapper(drawer_id)
            obs, info = wrapper.reset()

            ep_transitions = []
            prev_drawer_qpos = float(vec_env._drawer_qpos[0])
            success = False

            for step in range(args.max_episode_steps):
                residual = torch.zeros((1, 7), dtype=torch.float32)
                prev_obs = {k: v[0].clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}

                next_obs, reward, terminated, truncated, info = wrapper.step(residual)
                r_env = float(reward[0]) if hasattr(reward, "__getitem__") else float(reward)

                cur_drawer_qpos = float(vec_env._drawer_qpos[0])
                closed_frac = 1.0 - (cur_drawer_qpos / DRAWER_SLIDE)
                contact = get_contact_info(vec_env, 0)

                transition = {
                    "obs_state": prev_obs.get("observation.state", torch.zeros(10)).cpu().numpy().astype(np.float32),
                    "obs_base_action": prev_obs.get("observation.base_action", torch.zeros(10)).cpu().numpy().astype(np.float32),
                    "action": residual[0].numpy().astype(np.float32),
                    "reward": np.float32(r_env),
                    "done": bool(terminated[0] or truncated[0]),
                    "terminated": bool(terminated[0]),
                    "step": step,
                    # Reward components (for offline recomputation)
                    "drawer_qpos": np.float32(cur_drawer_qpos),
                    "closed_frac": np.float32(closed_frac),
                    "z_norm": contact["z_norm"],
                    "y_norm": contact["y_norm"],
                    "tcp_local_x": contact["tcp_local_x"],
                    "tcp_local_y": contact["tcp_local_y"],
                    "tcp_local_z": contact["tcp_local_z"],
                    "in_z_bounds": contact["in_z_bounds"],
                    "dist_to_demo_mean": contact["dist_to_demo_mean"],
                    "drawer_moved": cur_drawer_qpos < prev_drawer_qpos - 1e-6,
                }
                ep_transitions.append(transition)
                obs = next_obs
                prev_drawer_qpos = cur_drawer_qpos

                if terminated[0]:
                    # terminated means _is_success was True (before auto-reset)
                    success = True
                    break
                if truncated[0]:
                    break

            if success:
                total_success += 1
            completed += 1

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
                # Reward components
                drawer_qpos=np.array([t["drawer_qpos"] for t in ep_transitions]),
                closed_frac=np.array([t["closed_frac"] for t in ep_transitions]),
                z_norm=np.array([t["z_norm"] for t in ep_transitions]),
                y_norm=np.array([t["y_norm"] for t in ep_transitions]),
                tcp_local_x=np.array([t["tcp_local_x"] for t in ep_transitions]),
                tcp_local_y=np.array([t["tcp_local_y"] for t in ep_transitions]),
                tcp_local_z=np.array([t["tcp_local_z"] for t in ep_transitions]),
                in_z_bounds=np.array([t["in_z_bounds"] for t in ep_transitions]),
                dist_to_demo_mean=np.array([t["dist_to_demo_mean"] for t in ep_transitions]),
                drawer_moved=np.array([t["drawer_moved"] for t in ep_transitions]),
                # Metadata
                success=np.array([success]),
                drawer_id=np.array([drawer_id]),
                num_steps=np.array([len(ep_transitions)]),
                contact_z_gate=np.array([args.contact_z_gate]),
            )

            # Release lock
            shutil.rmtree(lock_dir, ignore_errors=True)

            tag = "SUCC" if success else "FAIL"
            sr = total_success / completed * 100
            elapsed = time.time() - t_start
            final_closed = float(ep_transitions[-1]["closed_frac"])
            print(f"  {tag} steps={step+1:3d} closed={final_closed:.2f} "
                  f"SR={sr:.1f}% ({total_success}/{completed}) t={elapsed:.0f}s", flush=True)

            found_work = True
            break  # restart task scan

        if not found_work:
            print(f"\n[Worker {os.getpid()}] No more work. Completed {completed} episodes.")
            print(f"  Total SR: {total_success}/{completed} = {total_success/max(completed,1)*100:.1f}%")
            break

    # Cleanup
    for w in wrappers.values():
        try:
            w.vec_env.close()
        except:
            pass


if __name__ == "__main__":
    main()
