#!/usr/bin/env python3
"""Rollout GR00T on drawer and record contact positions (y_norm, z_norm).
Work-stealing for multi-worker parallelism.

Usage:
    python scripts/eval_drawer_contact.py --output_dir outputs/contact_analysis_100k
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ['MUJOCO_GL'] = 'egl'

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
from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, DEMO_CONTACT_MEAN,
    DRAWER_SLOT_HEIGHT, DRAWER_GAP, TCP_OFFSET,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer

CKPT_BASE = os.path.expanduser("~/DATA/INTERN/training/groot_drawer_sim_33ep")
DRAWERS = [2, 3, 4]
NUM_EPISODES = 20
MAX_STEPS = 300


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/contact_analysis_100k")
    p.add_argument("--checkpoint", type=int, default=100000)
    p.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    return p.parse_args()


def try_lock(lock_path):
    try:
        os.makedirs(lock_path)
        return True
    except FileExistsError:
        return False


def compute_contact_coords(vec_env, env_idx=0):
    """Compute y_norm, z_norm of TCP relative to active drawer face.
    Returns (y_norm, z_norm, is_contact) where is_contact means TCP is in front of face.
    """
    env = vec_env._envs[env_idx]
    model, data, ids = env["model"], env["data"], env["ids"]
    
    # TCP position (same as _apply_contact)
    hand_pos = data.xpos[ids["hand_id"]]
    hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
    tcp = hand_pos + hand_mat @ TCP_OFFSET
    
    # Cabinet frame
    cab_pos_w = data.xpos[env["cab_body_id"]]
    cab_mat = data.xmat[env["cab_body_id"]].reshape(3, 3)
    
    # TCP in cabinet local frame
    tcp_local = cab_mat.T @ (tcp - cab_pos_w)
    
    # Check if TCP is aligned with drawer face
    y_ok = abs(tcp_local[1]) < vec_env._face_hy + 0.03
    z_ok = (tcp_local[2] > env["drawer_z_min"]) and (tcp_local[2] < env["drawer_z_max"])
    
    # Compute normalized contact position
    face_hz = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP
    dz = env["drawer_z_min"] + face_hz + DRAWER_GAP
    y_norm = tcp_local[1] / vec_env._face_hy if vec_env._face_hy > 0 else 0.0
    z_norm = (tcp_local[2] - dz) / face_hz if face_hz > 0 else 0.0
    
    # Is TCP in contact zone (in front of face)?
    in_contact = y_ok and z_ok and (tcp_local[0] >= vec_env._face_x_closed)
    
    return float(y_norm), float(z_norm), bool(in_contact), float(tcp_local[0])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = os.path.join(CKPT_BASE, f"checkpoint-{args.checkpoint}")
    
    # Build work items: (drawer_id, episode_idx)
    work_items = [(d, ep) for d in DRAWERS for ep in range(args.num_episodes)]
    np.random.shuffle(work_items)
    
    wrapper = None
    
    for drawer_id, ep_idx in work_items:
        result_path = output_dir / f"d{drawer_id}_ep{ep_idx:02d}.npz"
        lock_path = output_dir / f"d{drawer_id}_ep{ep_idx:02d}.lock"
        
        if result_path.exists():
            continue
        if not try_lock(str(lock_path)):
            continue
        
        print(f"[CONTACT] D{drawer_id} ep{ep_idx}...", end=" ", flush=True)
        
        # Create env (no contact_threshold - we record all positions)
        vec_env = MuJoCoVecEnvDrawer(
            num_envs=1,
            max_episode_steps=args.max_steps,
            reward_type="dense",
            active_drawers=[drawer_id],
            contact_threshold=float("inf"),  # no gating, just record
        )
        
        if wrapper is None:
            wrapper = MuJoCoResidualWrapperDrawer(
                vec_env=vec_env,
                groot_checkpoint=ckpt_path,
                policy_device="cuda:0",
            )
        else:
            wrapper.vec_env = vec_env
            wrapper.num_envs = vec_env.num_envs
            wrapper._cached_chunks = [None] * vec_env.num_envs
            wrapper._chunk_idx = [wrapper.open_loop_horizon] * vec_env.num_envs
            wrapper._held_base_action = np.zeros((vec_env.num_envs, 8), dtype=np.float32)
            wrapper._ema_pos = [None] * vec_env.num_envs
            wrapper._ema_quat = [None] * vec_env.num_envs
        
        env = wrapper
        obs, info = env.reset()
        
        # Record arrays
        steps_arr = []
        y_norms = []
        z_norms = []
        in_contacts = []
        tcp_xs = []
        drawer_qpos_arr = []
        rewards_arr = []
        
        for step in range(args.max_steps):
            # Record contact info BEFORE step
            y_n, z_n, contact, tcp_x = compute_contact_coords(vec_env, 0)
            steps_arr.append(step)
            y_norms.append(y_n)
            z_norms.append(z_n)
            in_contacts.append(contact)
            tcp_xs.append(tcp_x)
            drawer_qpos_arr.append(float(vec_env._drawer_qpos[0]))
            
            # Step with zero residual
            action = np.zeros((1, 7))
            obs, reward, terminated, truncated, info = env.step(action)
            r = reward[0].item() if hasattr(reward[0], 'item') else float(reward[0])
            rewards_arr.append(r)
            
            if terminated[0] or truncated[0]:
                break
        
        success = bool(terminated[0])
        
        # Save
        np.savez_compressed(str(result_path),
            steps=np.array(steps_arr),
            y_norm=np.array(y_norms),
            z_norm=np.array(z_norms),
            in_contact=np.array(in_contacts),
            tcp_x=np.array(tcp_xs),
            drawer_qpos=np.array(drawer_qpos_arr),
            rewards=np.array(rewards_arr),
            success=success,
            drawer=drawer_id,
            episode=ep_idx,
            checkpoint=args.checkpoint,
            demo_y_norm=DEMO_CONTACT_MEAN[drawer_id]["y_norm"],
            demo_z_norm=DEMO_CONTACT_MEAN[drawer_id]["z_norm"],
        )
        
        try:
            os.rmdir(str(lock_path))
        except:
            pass
        
        print(f"{'OK' if success else 'FAIL'} ({step+1} steps)")
    
    print("\nAll work items done!")


if __name__ == "__main__":
    main()
