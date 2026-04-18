import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
import numpy as np
sys.path.insert(0, '/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src')

import mujoco
from scipy.spatial.transform import Rotation
from utils import load_calib, DEFAULT_SCENE_XML, get_model_ids, solve_ik, HOME_QPOS, get_tcp_pose

scene_xml = DEFAULT_SCENE_XML
xml = '<mujoco><include file="fr3_with_hand.xml"/></mujoco>'
os.chdir(os.path.dirname(scene_xml))
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
ids = get_model_ids(model)

# Reset to HOME
for i, jid in enumerate(ids['jnt_ids']):
    data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
mujoco.mj_forward(model, data)

tcp_home, R_home = get_tcp_pose(model, data, ids['hand_id'])
home_quat_xyzw = Rotation.from_matrix(R_home).as_quat()
print(f'Home TCP: {tcp_home}')
print(f'Home quat (xyzw): {home_quat_xyzw}')

# ep10 target
target_pos = np.array([0.409489, 0.096999, 0.196475])
qw, qx, qy, qz = 0.998998, -0.017220, 0.035407, -0.021264
target_quat = np.array([qx, qy, qz, qw])

print(f'Target pos: {target_pos}')
print(f'Target quat (xyzw): {target_quat}')

# Check: is this quat close to home quat?
R_target = Rotation.from_quat(target_quat).as_matrix()
R_home_mat = Rotation.from_quat(home_quat_xyzw).as_matrix()
angle_diff = np.linalg.norm(Rotation.from_matrix(R_target @ R_home_mat.T).as_rotvec())
print(f'Rotation diff from home: {np.degrees(angle_diff):.1f} deg')

# Test: IK with home orientation (only change position)
for i, jid in enumerate(ids['jnt_ids']):
    data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
mujoco.mj_forward(model, data)

pe, re = solve_ik(model, data, ids['hand_id'], ids['jnt_ids'],
                  target_pos, home_quat_xyzw, max_iter=500, ns_gain=1.0)
tcp_f, _ = get_tcp_pose(model, data, ids['hand_id'])
print(f'\nIK (pos only, home rot): pos_err={pe:.6f}, rot_err={re:.6f}')
print(f'Final TCP: {tcp_f}')

# Test: IK with target orientation
for i, jid in enumerate(ids['jnt_ids']):
    data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
mujoco.mj_forward(model, data)

pe2, re2 = solve_ik(model, data, ids['hand_id'], ids['jnt_ids'],
                     target_pos, target_quat, max_iter=500, ns_gain=1.0)
tcp_f2, _ = get_tcp_pose(model, data, ids['hand_id'])
print(f'\nIK (target rot): pos_err={pe2:.6f}, rot_err={re2:.6f}')
print(f'Final TCP: {tcp_f2}')
