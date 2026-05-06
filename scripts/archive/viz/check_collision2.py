"""Check if gripper-cube collision actually works by teleporting gripper onto cube."""
import sys
sys.path.insert(0, "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl")
import numpy as np
import mujoco
from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP

env = MuJoCoVecEnvPnP(
    num_envs=1,
    cube_positions=[[0.39, 0.16, 0.02]],
    bowl_positions=[[0.42, 0.03, 0.0]],
    max_episode_steps=300,
    reward_type="dense",
)
obs = env.reset()
model = env._envs[0]["model"]
data = env._envs[0]["data"]

# Find finger geom IDs by checking collision geoms in hand area
print("=== Finger-area collision geoms ===")
finger_geom_ids = env._envs[0].get("finger_geom_ids", [])
print(f"finger_geom_ids from env: {finger_geom_ids}")
for gid in finger_geom_ids:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
    print(f"  {gid}: {name}, contype={model.geom_contype[gid]}, conaffinity={model.geom_conaffinity[gid]}")

# Check cube geom
cube_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
print(f"\ncube_geom id={cube_gid}: contype={model.geom_contype[cube_gid]}, conaffinity={model.geom_conaffinity[cube_gid]}")

# Now use IK-like approach: set mocap/target to cube pos and simulate
# Actually easier: just run mj_step many times with gravity and see if cube falls through table
cube_qposadr = env._envs[0]["cube_qposadr"]
print(f"\n=== Gravity test: cube should stay on table ===")
for i in range(5):
    mujoco.mj_step(model, data, nstep=100)
    cube_pos = data.qpos[cube_qposadr:cube_qposadr+3].copy()
    print(f"  after {(i+1)*100} steps: cube_z={cube_pos[2]:.6f}")

# Reset and test: place cube at higher z and drop it
print(f"\n=== Drop test: cube from z=0.2 should land on table ===")
data.qpos[cube_qposadr:cube_qposadr+3] = [0.39, 0.16, 0.2]
data.qvel[:] = 0
mujoco.mj_forward(model, data)
for i in range(10):
    mujoco.mj_step(model, data, nstep=100)
    cube_pos = data.qpos[cube_qposadr:cube_qposadr+3].copy()
    ncon = data.ncon
    contacts = []
    for ci in range(ncon):
        c = data.contact[ci]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
        if "cube" in g1 or "cube" in g2:
            contacts.append(f"{g1}<->{g2}")
    print(f"  step {(i+1)*100}: cube_z={cube_pos[2]:.6f}, cube_contacts={contacts}")

print("\nDone!")
