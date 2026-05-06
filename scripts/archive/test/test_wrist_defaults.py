import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), 'Mujoco_Franka', 'src'))
sys.path.insert(0, '/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src')
# os.environ['MUJOCO_GL'] = 'egl'  # let LD_PRELOAD handle it

import mujoco
from scipy.spatial.transform import Rotation
from utils import (
    _wrist_cam_xml, _bind_wrist_cam, load_calib, DEFAULT_SCENE_XML,
    INTRINSICS, WRIST_CAM_POS_IN_HAND, WRIST_CAM_QUAT_XYZW_IN_HAND,
    WRIST_CAM_INTRINSICS
)

# NOTE: NOT calling load_calib_wrist, so globals stay at hardcoded defaults
print(f'Wrist pos (defaults): {WRIST_CAM_POS_IN_HAND}')
print(f'Wrist quat_xyzw (defaults): {WRIST_CAM_QUAT_XYZW_IN_HAND}')

# Generate wrist XML
wrist_xml = _wrist_cam_xml()
print(f'Wrist XML: {wrist_xml}')

# Build a simple model
T_base_cam = load_calib()
R_cam = T_base_cam[:3, :3]
t_cam = T_base_cam[:3, 3]
R_mj = R_cam @ np.diag([1.0, -1.0, -1.0])
quat_xyzw = Rotation.from_matrix(R_mj).as_quat()
quat_wxyz_cam = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
fovy = 2 * np.degrees(np.arctan2(360 / 2.0, INTRINSICS['fy']))
scene_xml = DEFAULT_SCENE_XML

cube_pos = [0.45, 0.0, 0.02]
bowl_pos = [0.40, 0.12, 0.005]

xml = f"""
<mujoco model="fr3 pnp test">
  <include file="{scene_xml}"/>
  <visual>
    <global offwidth="640" offheight="360"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
  </visual>
  <worldbody>
    <light pos="0.5 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="true"/>
    <light pos="0.5 0.5 1.2" dir="-0.2 -0.3 -1" diffuse="0.5 0.5 0.5"/>
    <geom name="labfloor" type="plane" size="3 3 0.1" rgba="0.5 0.5 0.5 1" pos="0 0 0"/>
    <geom name="dark_table" type="box" size="0.5 0.7 0.01" pos="0.45 0 0" rgba="0.2 0.2 0.2 1"/>
    <camera name="cam_base" pos="{t_cam[0]:.6f} {t_cam[1]:.6f} {t_cam[2]:.6f}"
            quat="{quat_wxyz_cam[0]:.6f} {quat_wxyz_cam[1]:.6f} {quat_wxyz_cam[2]:.6f} {quat_wxyz_cam[3]:.6f}"
            fovy="{fovy:.4f}"/>
    {wrist_xml}
    <body name="cube" pos="{cube_pos[0]} {cube_pos[1]} {cube_pos[2]}">
      <freejoint/>
      <geom type="box" size="0.04 0.02 0.02" rgba="1 0 0 1" mass="0.1"/>
    </body>
    <body name="bowl" pos="{bowl_pos[0]} {bowl_pos[1]} {bowl_pos[2]}">
      <geom type="cylinder" size="0.095 0.005" rgba="1 1 1 1" mass="0.3"/>
    </body>
  </worldbody>
</mujoco>
"""

orig_cwd = os.getcwd()
os.chdir(os.path.dirname(scene_xml))
model = mujoco.MjModel.from_xml_string(xml)
os.chdir(orig_cwd)

# Bind wrist to hand
_bind_wrist_cam(model)

cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'cam_wrist')
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
print(f'cam_wrist_id={cam_wrist_id}, cam_bodyid={model.cam_bodyid[cam_wrist_id]}, hand_id={hand_id}')
print(f'cam_pos: {model.cam_pos[cam_wrist_id]}')
print(f'cam_quat: {model.cam_quat[cam_wrist_id]}')

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Hide hand geoms for wrist
opt = mujoco.MjvOption()
hide_bodies = []
for bname in ['hand', 'fr3_link6', 'fr3_link7']:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid >= 0:
        hide_bodies.append(bid)
for gi in range(model.ngeom):
    if model.geom_bodyid[gi] in hide_bodies and model.geom_group[gi] == 2:
        model.geom_group[gi] = 4
opt.geomgroup[3] = 0
opt.geomgroup[4] = 0

renderer = mujoco.Renderer(model, height=360, width=640)
renderer.update_scene(data, camera=cam_wrist_id, scene_option=opt)
frame = renderer.render()
print(f'Wrist frame: shape={frame.shape}, min={frame.min()}, max={frame.max()}')

import imageio
imageio.imwrite('/tmp/pnp_wrist_defaults.png', frame)
print('Saved /tmp/pnp_wrist_defaults.png')
