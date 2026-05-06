import sys
sys.path.insert(0, "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl")
import numpy as np
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
import mujoco

print("=== ALL GEOMS ===")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
    print(f"  {name}: contype={model.geom_contype[i]}, conaffinity={model.geom_conaffinity[i]}, group={model.geom_group[i]}")

print(f"\n=== CONTACTS (ncon={data.ncon}) ===")
for ci in range(data.ncon):
    c = data.contact[ci]
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
    print(f"  {g1} <-> {g2}, dist={c.dist:.4f}")

# Step with action pushing down
print("\n=== Stepping EEF down toward cube ===")
for step in range(100):
    action = np.zeros(7)
    action[2] = -1.0  # z down
    obs, rew, done, info = env.step(action)
    eef = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")]
    cube = data.qpos[env._envs[0]["cube_qposadr"]:env._envs[0]["cube_qposadr"]+3]
    if step % 10 == 0:
        print(f"  step {step}: eef={eef}, cube={cube}, ncon={data.ncon}")
