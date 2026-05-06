"""Visualize Easy/Normal/Hard difficulty levels as a stacked video."""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
import numpy as np
sys.path.insert(0, '/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src')

import mujoco
from scipy.spatial.transform import Rotation
from utils import (
    _wrist_cam_xml, _bind_wrist_cam, load_calib, DEFAULT_SCENE_XML,
    INTRINSICS, CAM_H, CAM_W, get_model_ids, solve_ik, HOME_QPOS,
    get_tcp_pose
)
import json
import imageio
import cv2

# Fixed robot pose from ep0 frame0
TCP_POS = np.array([0.4094890058040619, 0.09699899703264236, 0.19647499918937683])
TCP_QUAT_XYZW = np.array([0.9989979863166809, -0.01721999980509281,
                           0.03540699928998947, -0.021263999864459038])
GRIPPER_WIDTH = 0.9958999752998352

# Dataset positions
with open('/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/configs/pnp_sim33ep_positions.json') as f:
    all_positions = json.load(f)

# Base positions (mean of dataset)
CUBE_MEAN = np.array([0.42, -0.03, 0.02])
BOWL_MEAN = np.array([0.42, 0.03, 0.0])

def generate_positions(difficulty, n_samples=12, seed=42):
    """Generate cube/bowl positions for a difficulty level."""
    rng = np.random.RandomState(seed)
    positions = []
    
    if difficulty == 'easy':
        # Use actual dataset positions (no noise)
        indices = rng.choice(len(all_positions), n_samples, replace=False)
        for idx in indices:
            ep = all_positions[idx]
            positions.append({
                'cube_pos': ep['cube_pos'],
                'bowl_pos': ep['bowl_pos'],
                'cube_yaw': 0.0,
            })
    
    elif difficulty == 'normal':
        # Dataset positions + moderate noise
        indices = rng.choice(len(all_positions), n_samples, replace=False)
        for idx in indices:
            ep = all_positions[idx]
            cube_noise = rng.uniform(-0.06, 0.06, 2)
            bowl_noise = rng.uniform(-0.06, 0.06, 2)
            cube_pos = [np.clip(ep['cube_pos'][0] + cube_noise[0], 0.32, 0.55),
                       np.clip(ep['cube_pos'][1] + cube_noise[1], -0.22, 0.22), 0.02]
            bowl_pos = [np.clip(ep['bowl_pos'][0] + bowl_noise[0], 0.30, 0.53),
                       np.clip(ep['bowl_pos'][1] + bowl_noise[1], -0.24, 0.22), 0.0]
            yaw = rng.uniform(-30, 30)
            positions.append({
                'cube_pos': cube_pos,
                'bowl_pos': bowl_pos,
                'cube_yaw': yaw,
            })
    
    elif difficulty == 'hard':
        for _ in range(n_samples):
            # Random positions across extended workspace
            cx = rng.uniform(0.28, 0.58)
            cy = rng.uniform(-0.28, 0.28)
            bx = rng.uniform(0.26, 0.56)
            by = rng.uniform(-0.30, 0.28)
            # Ensure min distance 30cm
            while np.sqrt((cx-bx)**2 + (cy-by)**2) < 0.30:
                bx = rng.uniform(0.26, 0.56)
                by = rng.uniform(-0.30, 0.28)
            yaw = rng.uniform(-60, 60)
            positions.append({
                'cube_pos': [cx, cy, 0.02],
                'bowl_pos': [bx, by, 0.0],
                'cube_yaw': yaw,
            })
    
    return positions


def build_and_render(cube_pos, bowl_pos, cube_yaw_deg=0.0):
    """Build model, set robot pose, render cam_base frame."""
    T_base_cam = load_calib()
    R_cam = T_base_cam[:3, :3]
    t_cam = T_base_cam[:3, 3]
    R_mj = R_cam @ np.diag([1.0, -1.0, -1.0])
    quat_xyzw = Rotation.from_matrix(R_mj).as_quat()
    quat_wxyz_cam = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    fovy = 2 * np.degrees(np.arctan2(CAM_H / 2.0, INTRINSICS['fy']))
    scene_xml = DEFAULT_SCENE_XML
    wrist_xml = _wrist_cam_xml()

    # Cube quaternion
    yaw = np.radians(cube_yaw_deg)
    q_xyzw = Rotation.from_euler('z', yaw).as_quat()
    cube_quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    cpos = ' '.join(f'{v:.6f}' for v in cube_pos)
    cquat = ' '.join(f'{v:.6f}' for v in cube_quat_wxyz)
    csz = '0.0400 0.0200 0.0200'
    bpos = ' '.join(f'{v:.6f}' for v in bowl_pos)

    xml = f"""
    <mujoco model="fr3 pnp difficulty viz">
      <include file="fr3_with_hand.xml"/>
      <statistic center="0.3 0 0.4" extent="1.0"/>
      <visual>
        <headlight diffuse="0.4 0.4 0.4" ambient="0.45 0.43 0.40" specular="0.15 0.15 0.15"/>
        <rgba haze="0.15 0.20 0.25 1"/>
        <global offwidth="{CAM_W}" offheight="{CAM_H}"/>
        <quality shadowsize="4096"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.35 0.35 0.38" rgb2="0.18 0.18 0.20"
                 width="512" height="3072"/>
        <texture type="2d" name="labfloor" builtin="flat"
                 rgb1="0.35 0.35 0.35" width="1" height="1"/>
        <material name="labfloor" texture="labfloor" texuniform="true" reflectance="0.05"/>
        <material name="dark_table" rgba="0.12 0.14 0.18 1" specular="0.3" shininess="0.1" reflectance="0.08"/>
        <material name="red_cube" rgba="0.85 0.15 0.10 1" specular="0.15" shininess="0.05" reflectance="0.03"/>
        <material name="white_bowl" rgba="0.92 0.90 0.88 1" specular="0.2" shininess="0.1" reflectance="0.05"/>
      </asset>
      <worldbody>
        <light pos="0.3 0.0 1.8" dir="0 0 -1" directional="true"
               diffuse="0.45 0.45 0.45" ambient="0.15 0.14 0.13" specular="0.1 0.1 0.1"
               castshadow="true"/>
        <light pos="0.8 0.5 1.2" dir="-0.4 -0.3 -1" directional="false"
               diffuse="0.2 0.2 0.2" specular="0.05 0.05 0.05"/>
        <geom name="labfloor" size="3 3 0.01" type="plane" material="labfloor"
              pos="0.3 0 -0.02"/>
        <body name="table" pos="0.3 0 -0.01">
          <geom name="table_top" type="box" size="0.55 0.45 0.01"
                material="dark_table" contype="1" conaffinity="1"/>
        </body>
        <camera name="cam_base"
                pos="{t_cam[0]:.6f} {t_cam[1]:.6f} {t_cam[2]:.6f}"
                quat="{quat_wxyz_cam[0]:.6f} {quat_wxyz_cam[1]:.6f} {quat_wxyz_cam[2]:.6f} {quat_wxyz_cam[3]:.6f}"
                fovy="{fovy:.4f}"/>
        {wrist_xml}
        <body name="cube" pos="{cpos}" quat="{cquat}">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="{csz}" material="red_cube"
                mass="0.1"/>
        </body>
        <body name="bowl" pos="{bpos}">
          <geom name="bowl_base" type="cylinder" size="0.0950 0.005"
                material="white_bowl" mass="0.3" pos="0 0 0.005"/>
        </body>
      </worldbody>
    </mujoco>
    """

    orig_cwd = os.getcwd()
    os.chdir(os.path.dirname(scene_xml))
    model = mujoco.MjModel.from_xml_string(xml)
    os.chdir(orig_cwd)

    _bind_wrist_cam(model)
    data = mujoco.MjData(model)
    ids = get_model_ids(model)

    # Set HOME
    for i, jid in enumerate(ids['jnt_ids']):
        data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
    mujoco.mj_forward(model, data)

    # IK to ep0 TCP pose
    current_tcp_pos, current_tcp_R = get_tcp_pose(model, data, ids['hand_id'])
    current_quat_xyzw = Rotation.from_matrix(current_tcp_R).as_quat()
    target_q = TCP_QUAT_XYZW.copy()
    if np.dot(current_quat_xyzw, target_q) < 0:
        target_q = -target_q
    solve_ik(model, data, ids['hand_id'], ids['jnt_ids'],
             TCP_POS, target_q, max_iter=500, ns_gain=1.0)

    # Set gripper
    finger_pos = np.clip(GRIPPER_WIDTH, 0.0, 1.0) * 0.04
    for fid in ids['finger_ids']:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = finger_pos
    mujoco.mj_forward(model, data)

    return model, data, ids


def render_frame(model, data, W=320, H=180):
    """Render cam_base view."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'cam_base')
    ctx = mujoco.GLContext(W, H)
    ctx.make_current()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    viewport = mujoco.MjrRect(0, 0, W, H)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    opt = mujoco.MjvOption()
    opt.geomgroup[4] = 1
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    frame = np.empty((H, W, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(frame, None, viewport, context)
    frame = np.flipud(frame).copy()
    ctx.free()
    return frame


# Main
N = 12
W, H = 320, 180
OUT_DIR = '/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/debug_pnp'
os.makedirs(OUT_DIR, exist_ok=True)

all_frames = []
for diff in ['easy', 'normal', 'hard']:
    positions = generate_positions(diff, n_samples=N)
    row_frames = []
    for i, p in enumerate(positions):
        print(f"  {diff} sample {i}/{N}")
        model, data, ids = build_and_render(p['cube_pos'], p['bowl_pos'], p.get('cube_yaw', 0))
        frame = render_frame(model, data, W, H)
        # Add label
        cv2.putText(frame, f"{diff.upper()} #{i}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        row_frames.append(frame)
    all_frames.append(row_frames)

# Create video: each frame shows one sample from each difficulty (stacked vertically)
video_frames = []
for i in range(N):
    stacked = np.vstack([all_frames[0][i], all_frames[1][i], all_frames[2][i]])
    video_frames.append(stacked)

out_path = f'{OUT_DIR}/pnp_difficulty_levels.mp4'
imageio.mimsave(out_path, video_frames, fps=2, macro_block_size=1)
print(f"\nSaved {len(video_frames)} frames to {out_path}")
print(f"Frame size: {video_frames[0].shape}")
