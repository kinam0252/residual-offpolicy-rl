import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
import numpy as np
sys.path.insert(0, '/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src')

import mujoco
from scipy.spatial.transform import Rotation
from utils import load_calib, DEFAULT_SCENE_XML, get_model_ids, solve_ik, HOME_QPOS, get_tcp_pose, TCP_OFFSET

scene_xml = DEFAULT_SCENE_XML

xml = """<mujoco>
  <include file="fr3_with_hand.xml"/>
</mujoco>"""

os.chdir(os.path.dirname(scene_xml))
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
ids = get_model_ids(model)

# Set HOME
for i, jid in enumerate(ids['jnt_ids']):
    data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
mujoco.mj_forward(model, data)

tcp_home, _ = get_tcp_pose(model, data, ids['hand_id'])
print(f'Home TCP: {tcp_home}')

# Test 1: target = home TCP (should converge immediately)
tcp_quat_xyzw = Rotation.from_matrix(data.xmat[ids['hand_id']].reshape(3,3)).as_quat()
pe, re = solve_ik(model, data, ids['hand_id'], ids['jnt_ids'],
                  tcp_home, tcp_quat_xyzw, max_iter=10)
print(f'Test1 (self): pos_err={pe:.6f}, rot_err={re:.6f}')

# Test 2: small offset
target2 = tcp_home + np.array([0.01, -0.05, 0.0])
pe2, re2 = solve_ik(model, data, ids['hand_id'], ids['jnt_ids'],
                     target2, tcp_quat_xyzw, max_iter=200, ns_gain=1.0)
tcp2, _ = get_tcp_pose(model, data, ids['hand_id'])
print(f'Test2 (small): pos_err={pe2:.6f}, rot_err={re2:.6f}, tcp={tcp2}')

# Test 3: ep10 target
tcp_target = np.array([0.4094890058040619, 0.09699899703264236, 0.19647499918937683])
qw, qx, qy, qz = 0.9989979863166809, -0.01721999980509281, 0.03540699928998947, -0.021263999864459038
target_quat = np.array([qx, qy, qz, qw])

# Reset to HOME first
for i, jid in enumerate(ids['jnt_ids']):
    data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
mujoco.mj_forward(model, data)

pe3, re3 = solve_ik(model, data, ids['hand_id'], ids['jnt_ids'],
                     tcp_target, target_quat, max_iter=500, ns_gain=1.0)
tcp3, _ = get_tcp_pose(model, data, ids['hand_id'])
print(f'Test3 (ep10): pos_err={pe3:.6f}, rot_err={re3:.6f}, tcp={tcp3}')

# Print all joint values
for i, jid in enumerate(ids['jnt_ids']):
    print(f'  j{i}: {data.qpos[model.jnt_qposadr[jid]]:.4f}')
