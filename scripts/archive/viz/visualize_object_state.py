"""Visualize object_state correctness: render images + overlay state info
for multiple cube positions to verify state matches visual."""
import sys, os

# Set EGL before any mujoco import
os.environ["MUJOCO_GL"] = "egl"
# Add local EGL libraries
for p in [os.path.expanduser("~/.local/lib/gl"), os.path.expanduser("~/.local/lib")]:
    if os.path.isdir(p):
        os.environ["LD_LIBRARY_PATH"] = p + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        break

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "resfit", "rl_finetuning", "wrappers"))
# Mujoco_Franka utils (contains setup_egl, make_model_with_cube, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Mujoco_Franka", "src"))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

# 6 diverse cube positions
CUBE_POSITIONS = [
    [0.43, -0.08, 0.02],   # left-back
    [0.43,  0.03, 0.02],   # left-front
    [0.48, -0.05, 0.02],   # center
    [0.48,  0.05, 0.02],   # center-front
    [0.52, -0.02, 0.02],   # right
    [0.52,  0.06, 0.02],   # right-front
]

OUT_DIR = "outputs/state_visualization"
os.makedirs(OUT_DIR, exist_ok=True)

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(len(CUBE_POSITIONS), 4, figure=fig, wspace=0.3, hspace=0.4)

for i, cube_pos in enumerate(CUBE_POSITIONS):
    print(f"\n--- Cube pos {i}: {cube_pos} ---")
    env = MuJoCoVecEnv(
        num_envs=1,
        cube_positions=[cube_pos],
        device="cpu",
        max_episode_steps=500,
        rl_img_size=84,
    )
    obs, info = env.reset()
    
    # Run a few steps with zero action (8D: pos3+quat4+grip1)
    for step in range(30):
        action = torch.zeros(1, 8)
        action[0, 3] = 1.0  # quat_w=1 (identity rotation)
        obs, reward, terminated, truncated, info = env.step(action)
    
    # Get state values
    state = obs["observation.state"][0].numpy()
    obj_state = obs["observation.object_state"][0].numpy()
    
    eef_pos = state[:3]
    eef_quat = state[3:7]
    grip = state[7:9]
    contact = state[9]
    
    cube_pos_obs = obj_state[:3]
    cube_quat = obj_state[3:7]
    
    print(f"  obs.object_state: pos={cube_pos_obs}, quat={cube_quat}")
    print(f"  obs.state (eef):  pos={eef_pos}, quat={eef_quat}")
    print(f"  Expected cube pos: {cube_pos}")
    print(f"  Cube pos error:    {np.abs(np.array(cube_pos) - cube_pos_obs)}")
    
    # Get camera images
    full_obs = env._build_obs_dict(render_mode="full")
    
    img_back = full_obs["observation.images.back"][0].numpy()     # (3,H,W)
    img_wrist = full_obs["observation.images.wrist"][0].numpy()   # (3,H,W)
    img_front = full_obs["observation.images.front"][0].numpy()   # (3,H,W)
    
    # Transpose to HWC for matplotlib
    img_back = np.transpose(img_back, (1, 2, 0))
    img_wrist = np.transpose(img_wrist, (1, 2, 0))
    img_front = np.transpose(img_front, (1, 2, 0))
    
    # Plot: [back_cam | wrist_cam | front_cam | state_info]
    ax_back = fig.add_subplot(gs[i, 0])
    ax_back.imshow(img_back)
    ax_back.set_title(f"Back cam", fontsize=10)
    ax_back.axis('off')
    
    ax_wrist = fig.add_subplot(gs[i, 1])
    ax_wrist.imshow(img_wrist)
    ax_wrist.set_title(f"Wrist cam", fontsize=10)
    ax_wrist.axis('off')
    
    ax_front = fig.add_subplot(gs[i, 2])
    ax_front.imshow(img_front)
    ax_front.set_title(f"Front cam", fontsize=10)
    ax_front.axis('off')
    
    # State info text
    ax_text = fig.add_subplot(gs[i, 3])
    ax_text.axis('off')
    
    state_text = (
        f"Cube #{i}: init_pos={cube_pos}\n"
        f"─────────────────────────\n"
        f"obs.object_state:\n"
        f"  pos: [{cube_pos_obs[0]:.4f}, {cube_pos_obs[1]:.4f}, {cube_pos_obs[2]:.4f}]\n"
        f"  quat(wxyz): [{cube_quat[0]:.4f}, {cube_quat[1]:.4f}, {cube_quat[2]:.4f}, {cube_quat[3]:.4f}]\n"
        f"─────────────────────────\n"
        f"obs.state (eef):\n"
        f"  pos: [{eef_pos[0]:.4f}, {eef_pos[1]:.4f}, {eef_pos[2]:.4f}]\n"
        f"  quat: [{eef_quat[0]:.4f}, {eef_quat[1]:.4f}, {eef_quat[2]:.4f}, {eef_quat[3]:.4f}]\n"
        f"  grip: [{grip[0]:.4f}, {grip[1]:.4f}], contact: {contact:.1f}\n"
        f"─────────────────────────\n"
        f"pos error (init-obs): {np.linalg.norm(np.array(cube_pos)-cube_pos_obs):.6f}m"
    )
    ax_text.text(0.05, 0.95, state_text, transform=ax_text.transAxes,
                 fontsize=8, fontfamily='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    env.close()

fig.suptitle("Object State Verification: Camera Images + State Values\n"
             "(6 cube positions, after 30 zero-action steps)", fontsize=14, fontweight='bold')
out_path = os.path.join(OUT_DIR, "object_state_verification.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n=== Saved to {out_path} ===")

# Also make a per-episode visualization: run 50 steps, plot state trajectory
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
for i, cube_pos in enumerate(CUBE_POSITIONS):
    ax = axes2[i // 3][i % 3]
    env = MuJoCoVecEnv(
        num_envs=1, cube_positions=[cube_pos],
        device="cpu", max_episode_steps=500, rl_img_size=84,
    )
    obs, _ = env.reset()
    
    traj_cube = []
    traj_eef = []
    for step in range(100):
        # Small random action (8D: pos3+quat4+grip1)
        action = torch.zeros(1, 8)
        action[0, :3] = torch.randn(3) * 0.02  # small pos noise
        action[0, 3] = 1.0  # identity quat
        action[0, 7] = 0.04  # open gripper
        obs, reward, terminated, truncated, info = env.step(action)
        traj_cube.append(obs["observation.object_state"][0].numpy()[:3].copy())
        traj_eef.append(obs["observation.state"][0].numpy()[:3].copy())
    
    traj_cube = np.array(traj_cube)
    traj_eef = np.array(traj_eef)
    
    ax.plot(traj_cube[:,0], traj_cube[:,1], 'ro-', markersize=2, label='cube xy', alpha=0.7)
    ax.plot(traj_eef[:,0], traj_eef[:,1], 'b.-', markersize=2, label='eef xy', alpha=0.7)
    ax.plot(traj_cube[0,0], traj_cube[0,1], 'rs', markersize=10, label=f'cube start')
    ax.plot(traj_eef[0,0], traj_eef[0,1], 'bs', markersize=10, label=f'eef start')
    ax.set_title(f"Cube init: [{cube_pos[0]:.2f}, {cube_pos[1]:.2f}]", fontsize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    env.close()

fig2.suptitle("XY Trajectories: Cube vs EEF (100 steps, small random actions)", fontsize=13, fontweight='bold')
out_path2 = os.path.join(OUT_DIR, "xy_trajectories.png")
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"=== Saved to {out_path2} ===")
