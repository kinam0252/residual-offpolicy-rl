"""Resume: recalculate only pnp_hard ep_0200~ep_0299 (100 files)."""
import numpy as np
from pathlib import Path

BASE = Path('/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/offline_data')
GRIPPER_CLOSE_THRESHOLD = 0.65
BOWL_HEIGHT = 0.005
SUCCESS_XY_THRESH = 0.095

def is_success(cube_pos, bowl_pos):
    xy_dist = np.linalg.norm(cube_pos[:2] - bowl_pos[:2])
    return xy_dist < SUCCESS_XY_THRESH and cube_pos[2] < BOWL_HEIGHT + 0.03

def recalc_episode(npz_path):
    data = dict(np.load(str(npz_path), allow_pickle=True))
    n = len(data['obs_state'])
    new_rewards = np.zeros(n, dtype=np.float32)
    grasped = False
    for i in range(n):
        tcp_pos = data['obs_state'][i, :3]
        cube_pos = data['obs_object_state'][i, :3]
        bowl_pos = data['obs_object_state'][i, 7:10]
        grip_raw = data['obs_state'][i, 7]
        contact = data['obs_state'][i, 9]
        gripper_width = np.clip(grip_raw / 0.04, 0.0, 1.0)
        if not grasped:
            if gripper_width < GRIPPER_CLOSE_THRESHOLD and contact > 0.5:
                if np.linalg.norm(tcp_pos - cube_pos) < 0.20:
                    grasped = True
        else:
            if gripper_width >= GRIPPER_CLOSE_THRESHOLD:
                grasped = False
        tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
        cube_bowl_xy = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))
        if not grasped:
            total = (1.0 - np.tanh(tcp_cube_dist / 0.1)) * 0.1
        else:
            total = 0.1 + 0.2 + (1.0 - np.tanh(cube_bowl_xy / 0.1)) * 0.5
            if is_success(cube_pos, bowl_pos):
                total += 1.0
        new_rewards[i] = float(np.clip(total, 0.0, 1.0))
        if data['done'][i]:
            grasped = False
    data['reward'] = new_rewards
    np.savez_compressed(str(npz_path), **data)
    return n, new_rewards.mean()

d = BASE / 'pnp_hard'
files = sorted(d.glob('*.npz'))
remaining = [f for f in files if int(f.stem.split('_')[1]) >= 200]
print(f"pnp_hard remaining: {len(remaining)} files")
total_n = 0
for i, f in enumerate(remaining):
    n, rm = recalc_episode(f)
    total_n += n
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(remaining)} done")
print(f"Done! {total_n} transitions recalculated")
