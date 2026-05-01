#!/usr/bin/env python3
"""Debug kinematic drawer physics — print TCP local coords at home position."""
import os, sys
os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import numpy as np
import mujoco

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, DRAWER_SLIDE, FLOOR_TO_DRAWER,
    WALL_THICK, DRAWER_SLOT_HEIGHT, CABINET_LONG, CABINET_SHORT, DRAWER_GAP,
    get_tcp_pose,
)

env = MuJoCoVecEnvDrawer(num_envs=1, max_episode_steps=500)
env.reset()

env_data = env._envs[0]
model, data, ids = env_data["model"], env_data["data"], env_data["ids"]

# TCP at home position
from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import TCP_OFFSET
hand_pos = data.xpos[ids["hand_id"]]
hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
tcp = hand_pos + hand_mat @ TCP_OFFSET
tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])

print(f"=== Home Position Debug ===")
print(f"TCP world: {tcp}")
print(f"TCP (get_tcp_pose): {tcp_pos}")
print()

# Cabinet transform
cab_body_id = env_data["cab_body_id"]
cab_pos_w = data.xpos[cab_body_id]
cab_mat = data.xmat[cab_body_id].reshape(3, 3)
print(f"Cabinet world pos: {cab_pos_w}")
print(f"Cabinet world mat:\n{cab_mat}")

# TCP in cabinet local frame
tcp_local = cab_mat.T @ (tcp - cab_pos_w)
print(f"\nTCP in cabinet local: {tcp_local}")
print(f"  x={tcp_local[0]:.4f} (slide dir: 0=back, +x=front)")
print(f"  y={tcp_local[1]:.4f}")
print(f"  z={tcp_local[2]:.4f}")

# Drawer geometry
d_hx = (CABINET_LONG - 2 * WALL_THICK) / 2 - DRAWER_GAP
face_hx = WALL_THICK / 2
face_x_closed = d_hx + face_hx
face_hy = CABINET_SHORT / 2 - DRAWER_GAP
print(f"\nDrawer geometry:")
print(f"  face_x_closed = {face_x_closed:.4f}")
print(f"  face_hy = {face_hy:.4f}")
print(f"  face_x_open = {face_x_closed + DRAWER_SLIDE:.4f}")

# Check each active drawer
for drawer_idx in [2, 3, 4]:
    slot_h = DRAWER_SLOT_HEIGHT
    dz = WALL_THICK + slot_h / 2 + drawer_idx * (slot_h + WALL_THICK)
    z_min = dz - slot_h / 2 - 0.02
    z_max = dz + slot_h / 2 + 0.02
    
    y_ok = abs(tcp_local[1]) < face_hy + 0.03
    z_ok = (tcp_local[2] > z_min) and (tcp_local[2] < z_max)
    face_x_now = face_x_closed + DRAWER_SLIDE  # drawer fully open
    x_check = tcp_local[0] < face_x_now + 0.02
    
    would_trigger = y_ok and z_ok and x_check
    new_qpos = tcp_local[0] - face_x_closed if would_trigger else None
    if new_qpos is not None:
        new_qpos = max(0.0, min(DRAWER_SLIDE, new_qpos))
    
    print(f"\n  Drawer {drawer_idx}: dz={dz:.4f} z_range=[{z_min:.4f}, {z_max:.4f}]")
    print(f"    y_ok={y_ok} (|{tcp_local[1]:.4f}| < {face_hy + 0.03:.4f})")
    print(f"    z_ok={z_ok} ({z_min:.4f} < {tcp_local[2]:.4f} < {z_max:.4f})")
    print(f"    x_check={x_check} ({tcp_local[0]:.4f} < {face_x_now + 0.02:.4f})")
    print(f"    WOULD_TRIGGER={would_trigger}")
    if new_qpos is not None:
        print(f"    new_qpos={new_qpos:.4f} (was {DRAWER_SLIDE:.4f})")
