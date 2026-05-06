import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
import numpy as np
sys.path.insert(0, '/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src')

import mujoco
from scipy.spatial.transform import Rotation
from utils import load_calib, DEFAULT_SCENE_XML, get_model_ids, solve_ik, HOME_QPOS, _wrist_cam_xml, get_tcp_pose, TCP_OFFSET

scene_xml = DEFAULT_SCENE_XML
wrist_xml = _wrist_cam_xml()

xml = f"""<mujoco>
  <include file="fr3_with_hand.xml"/>
  <worldbody>
    {wrist_xml}
  </worldbody>
</mujoco>"""

os.chdir(os.path.dirname(scene_xml))
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
ids = get_model_ids(model)

# Set HOME
for i, jid in enumerate(ids['jnt_ids']):
    data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
mujoco.mj_forward(model, data)

tcp_pos_home, tcp_R_home = get_tcp_pose(model, data, ids['hand_id'])
print(f'Home hand_pos: {data.xpos[ids["hand_id"]]}')
print(f'Home TCP (with offset): {tcp_pos_home}')
print(f'TCP_OFFSET: {TCP_OFFSET}')

# EP10 state: [x, y, z, qw, qx, qy, qz, gripper_width]
# These are TCP positions (pos + offset applied)
tcp_pos = np.array([0.4094890058040619, 0.09699899703264236, 0.19647499918937683])
qw, qx, qy, qz = 0.9989979863166809, -0.01721999980509281, 0.03540699928998947, -0.021263999864459038
tcp_quat_xyzw = np.array([qx, qy, qz, qw])

print(f'\nTarget TCP: {tcp_pos}')
print(f'Target quat (xyzw): {tcp_quat_xyzw}')

pe, re = solve_ik(model, data, ids['hand_id'], ids['jnt_ids'],
                  tcp_pos, tcp_quat_xyzw, max_iter=500, q_ref=HOME_QPOS, ns_gain=1.0)
print(f'\nIK result: pos_err={pe:.6f}m, rot_err={re:.6f}rad')

tcp_pos_final, _ = get_tcp_pose(model, data, ids['hand_id'])
print(f'Final TCP: {tcp_pos_final}')
print(f'Final hand: {data.xpos[ids["hand_id"]]}')

# Print joint angles
for i, jid in enumerate(ids['jnt_ids']):
    print(f'  joint {i}: {data.qpos[model.jnt_qposadr[jid]]:.6f}')
