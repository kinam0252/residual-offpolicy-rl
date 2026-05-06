#!/bin/bash
# Debug: check if cube yaw persists through wrapper.reset() -> override -> step
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
/home/nas_main/kinamkim/.venvs/groot/bin/python << 'PYEOF'
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP

positions = json.load(open("configs/pnp_common5_positions.json"))
scene_xml = "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"

first_ep = positions[0]
vec_env = MuJoCoVecEnvPnP(
    num_envs=1,
    cube_positions=[first_ep["cube_pos"]],
    bowl_positions=[first_ep["bowl_pos"]],
    max_episode_steps=500,
    reward_type="dense",
    scene_xml=scene_xml,
)

env_data = vec_env._envs[0]
data = env_data["data"]
model = env_data["model"]
cqa = env_data["cube_qposadr"]

def print_cube_quat(label):
    q_wxyz = data.qpos[cqa+3:cqa+7].copy()
    r = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    yaw = r.as_euler("zyx")[0]
    print(f"  [{label}] cube quat(wxyz)={q_wxyz}, yaw={np.degrees(yaw):.1f} deg")

print("=== Before any reset ===")
print_cube_quat("initial")

print("\n=== After vec_env.reset() ===")
vec_env.reset()
print_cube_quat("after reset")

print("\n=== Override cube yaw to 108.95 deg ===")
yaw_deg = 108.95
q_xyzw = Rotation.from_euler("z", np.radians(yaw_deg)).as_quat()
data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
print_cube_quat("after qpos set")

print("\n=== After mj_forward ===")
mujoco.mj_forward(model, data)
print_cube_quat("after mj_forward")

print("\n=== After mj_step (simulating one physics step) ===")
mujoco.mj_step(model, data)
print_cube_quat("after mj_step")

# Now test: does vec_env.step() overwrite it?
print("\n=== After vec_env._build_obs_dict() ===")
vec_env._build_obs_dict()
print_cube_quat("after build_obs")

# Check step function
print("\n=== Reset again, override, then do vec_env.step() ===")
vec_env.reset()
data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
mujoco.mj_forward(model, data)
print_cube_quat("before step")

import torch
act = torch.zeros(1, 7)
vec_env.step(act)
print_cube_quat("after vec_env.step()")

PYEOF
