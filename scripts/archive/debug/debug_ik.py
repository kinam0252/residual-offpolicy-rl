import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
import numpy as np
sys.path.insert(0, '/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src')

import mujoco
from scipy.spatial.transform import Rotation
from utils import load_calib, DEFAULT_SCENE_XML, get_model_ids, solve_ik, HOME_QPOS, _wrist_cam_xml, _bind_wrist_cam, INTRINSICS, CAM_H, CAM_W

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
print('hand_id:', ids['hand_id'])
print('jnt_ids:', ids['jnt_ids'])
print('n_jnts:', len(ids['jnt_ids']))
print('HOME_QPOS:', HOME_QPOS)

# Set HOME and forward
for i, jid in enumerate(ids['jnt_ids']):
    data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
mujoco.mj_forward(model, data)
print('Home TCP pos:', data.xpos[ids['hand_id']])

# IK to ep10 pose
tcp_pos = np.array([0.4094890058040619, 0.09699899703264236, 0.19647499918937683])
tcp_quat_wxyz = np.array([0.9989979863166809, -0.01721999980509281, 0.03540699928998947, -0.021263999864459038])
tcp_quat_xyzw = np.array([tcp_quat_wxyz[1], tcp_quat_wxyz[2], tcp_quat_wxyz[3], tcp_quat_wxyz[0]])

pe, re = solve_ik(model, data, ids['hand_id'], ids['jnt_ids'], tcp_pos, tcp_quat_xyzw, max_iter=500)
print(f'IK: pos_err={pe:.6f}m, rot_err={re:.6f}rad')
print('Final TCP pos:', data.xpos[ids['hand_id']])
