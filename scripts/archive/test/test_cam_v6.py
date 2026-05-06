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

T_base_cam = load_calib()
R_cam = T_base_cam[:3, :3]
t_cam = T_base_cam[:3, 3]
R_mj = R_cam @ np.diag([1.0, -1.0, -1.0])
quat_xyzw = Rotation.from_matrix(R_mj).as_quat()
quat_wxyz_cam = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
fovy = 2 * np.degrees(np.arctan2(CAM_H / 2.0, INTRINSICS['fy']))
scene_xml = DEFAULT_SCENE_XML
cam_name = 'cam_base'
wrist_xml = _wrist_cam_xml()

# ep10 positions
pos_file = '/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/configs/pnp_sim33ep_positions.json'
with open(pos_file) as f:
    all_pos = json.load(f)
ep = all_pos['10'] if '10' in all_pos else all_pos[10]
cube_pos = ep['cube_pos']
bowl_pos = ep['bowl_pos']
cube_quat_wxyz_obj = ep.get('cube_quat_wxyz', [1, 0, 0, 0])
cube_size = (0.04, 0.02, 0.02)
bowl_radius = 0.095

cpos = ' '.join(f'{v:.6f}' for v in cube_pos)
cquat = ' '.join(f'{v:.6f}' for v in cube_quat_wxyz_obj)
csz = ' '.join(f'{v:.4f}' for v in cube_size)
bpos = ' '.join(f'{v:.6f}' for v in bowl_pos)

xml = f"""
<mujoco model="fr3 pnp cam test v6">
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
    <camera name="{cam_name}"
            pos="{t_cam[0]:.6f} {t_cam[1]:.6f} {t_cam[2]:.6f}"
            quat="{quat_wxyz_cam[0]:.6f} {quat_wxyz_cam[1]:.6f} {quat_wxyz_cam[2]:.6f} {quat_wxyz_cam[3]:.6f}"
            fovy="{fovy:.4f}"/>
    {wrist_xml}
    <body name="cube" pos="{cpos}" quat="{cquat}">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="{csz}" material="red_cube"
            mass="0.1" friction="1.0 0.005 0.0001"/>
    </body>
    <body name="bowl" pos="{bpos}">
      <geom name="bowl_base" type="cylinder" size="{bowl_radius:.4f} 0.005"
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

# Set HOME first
for i, jid in enumerate(ids['jnt_ids']):
    data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
mujoco.mj_forward(model, data)

# Position TCP directly above cube (5cm above) - to verify wrist cam sees cube
tcp_above_cube = np.array([cube_pos[0], cube_pos[1], 0.12])  # 12cm above table
# Gripper pointing down (same as home orientation)
current_tcp_pos, current_tcp_R = get_tcp_pose(model, data, ids['hand_id'])
target_quat_xyzw = Rotation.from_matrix(current_tcp_R).as_quat()

pe, re = solve_ik(model, data, ids['hand_id'], ids['jnt_ids'],
                  tcp_above_cube, target_quat_xyzw, max_iter=500, ns_gain=1.0)
print(f'IK above cube: pos_err={pe:.6f}m, rot_err={re:.6f}rad')

# Open gripper
for fid in ids['finger_ids']:
    if fid >= 0:
        data.qpos[model.jnt_qposadr[fid]] = 0.04
mujoco.mj_forward(model, data)

final_tcp, _ = get_tcp_pose(model, data, ids['hand_id'])
print(f'Target TCP: {tcp_above_cube}')
print(f'Final  TCP: {final_tcp}')
print(f'Cube pos:   {cube_pos}')

# Render setup
W, H = 640, 360  # Match sim data resolution
ctx = mujoco.GLContext(W, H)
ctx.make_current()
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
viewport = mujoco.MjrRect(0, 0, W, H)

import imageio
out_dir = '/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/scripts'

# === Wrist cam (hide hand geoms to avoid self-occlusion) ===
hide_bodies = []
for bname in ['hand', 'fr3_link6', 'fr3_link7']:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid >= 0:
        hide_bodies.append(bid)
for gi in range(model.ngeom):
    if model.geom_bodyid[gi] in hide_bodies and model.geom_group[gi] == 2:
        model.geom_group[gi] = 4

opt_wrist = mujoco.MjvOption()
opt_wrist.geomgroup[3] = 0
opt_wrist.geomgroup[4] = 0

cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'cam_wrist')
cam_w = mujoco.MjvCamera()
cam_w.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam_w.fixedcamid = cam_wrist_id
mujoco.mjv_updateScene(model, data, opt_wrist, None, cam_w, mujoco.mjtCatBit.mjCAT_ALL, scene)
mujoco.mjr_render(viewport, scene, context)
frame = np.empty((H, W, 3), dtype=np.uint8)
mujoco.mjr_readPixels(frame, None, viewport, context)
frame = np.flipud(frame)
imageio.imwrite(f'{out_dir}/pnp_wrist_v6.png', frame)
print(f'Saved pnp_wrist_v6.png ({W}x{H})')

# === Base cam (show all geoms) ===
cam_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
cam_b = mujoco.MjvCamera()
cam_b.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam_b.fixedcamid = cam_base_id
opt_base = mujoco.MjvOption()
# Restore hand geom visibility for base cam render
for gi in range(model.ngeom):
    if model.geom_bodyid[gi] in hide_bodies and model.geom_group[gi] == 4:
        model.geom_group[gi] = 2
opt_base.geomgroup[4] = 1
mujoco.mjv_updateScene(model, data, opt_base, None, cam_b, mujoco.mjtCatBit.mjCAT_ALL, scene)
mujoco.mjr_render(viewport, scene, context)
frame2 = np.empty((H, W, 3), dtype=np.uint8)
mujoco.mjr_readPixels(frame2, None, viewport, context)
frame2 = np.flipud(frame2)
imageio.imwrite(f'{out_dir}/pnp_base_v6.png', frame2)
print(f'Saved pnp_base_v6.png ({W}x{H})')

ctx.free()
