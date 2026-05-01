#!/usr/bin/env python3
"""Render 5 envs with idx=2 joint pose, eef+90 yaw, back camera."""
import sys, os, json
sys.path.insert(0, "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl")
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from PIL import Image, ImageDraw

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP

# idx=2 initial joint position from real teleop data (sorted by joint name)
REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])

positions = json.load(open("configs/pnp_common5_positions.json"))
scene_xml = "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"

vec_env = MuJoCoVecEnvPnP(
    num_envs=1,
    max_episode_steps=500,
    reward_type="dense",
    scene_xml=scene_xml,
)

row_frames = []
for i in range(5):
    ep = positions[i]
    cube_pos = np.array(ep["cube_pos"], dtype=np.float64)
    bowl_pos = np.array(ep["bowl_pos"], dtype=np.float64)
    base_yaw = ep.get("cube_yaw", 0.0)
    yaw_deg = base_yaw + 90.0  # eef+90

    env_data = vec_env._envs[0]
    data = env_data["data"]
    model = env_data["model"]
    cqa = env_data["cube_qposadr"]
    bowl_body_id = env_data["bowl_body_id"]

    # Set robot to real teleop initial joint position (idx=2)
    for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
    for fid in env_data["ids"]["finger_ids"]:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = 0.04

    if cqa is not None:
        data.qpos[cqa:cqa + 3] = cube_pos
        q_xyzw = Rotation.from_euler("z", np.radians(yaw_deg)).as_quat()
        data.qpos[cqa + 3:cqa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    if bowl_body_id >= 0:
        model.body_pos[bowl_body_id] = bowl_pos

    mujoco.mj_forward(model, data)
    frame = vec_env.get_frame(0, camera="back", size=(240, 320))
    row_frames.append(frame)
    print(f"Env {i}: cube_yaw={yaw_deg:.1f}")

row = np.concatenate(row_frames, axis=1)
img = Image.fromarray(row)
draw = ImageDraw.Draw(img)
draw.text((5, 2), "real_joint + eef+90 | back cam", fill=(255, 255, 0))
out = "outputs/joint_pose_check.png"
img.save(out)
print(f"Saved {out} shape={row.shape}")
