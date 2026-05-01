#!/bin/bash
set -e
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
EVAL=scripts/eval_pnp_base_sr.py

# The problem: wrapper.reset() calls vec_env.reset() which calls _reset_single_env()
# and overwrites our cube position/yaw. We need to set cube/joint AFTER wrapper.reset().
# Fix: move the manual env setup AFTER wrapper.reset(), then re-forward and rebuild obs.

/home/nas_main/kinamkim/.venvs/groot/bin/python << 'PYEOF'
import re

with open("scripts/eval_pnp_base_sr.py") as f:
    code = f.read()

# Find and replace the section from "# Manually reset environment" to "obs = wrapper.reset()"
# We need to: 1) call wrapper.reset() first, 2) then set positions, 3) then rebuild obs

old_block = '''        # Manually reset environment with specific positions
        env_data = vec_env._envs[0]
        data = env_data["data"]
        model = env_data["model"]
        cqa = env_data["cube_qposadr"]
        bowl_body_id = env_data["bowl_body_id"]

        # Reset robot joints to home
        for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
        for fid in env_data["ids"]["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04

        # Set cube position + yaw
        if cqa is not None:
            data.qpos[cqa:cqa + 3] = actual_cube_pos
            q_xyzw = Rotation.from_euler("z", np.radians(yaw_deg)).as_quat()
            data.qpos[cqa + 3:cqa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        # Set bowl position
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = bowl_pos

        env_data["cube_pos_init"] = actual_cube_pos.copy()
        env_data["bowl_pos_init"] = bowl_pos.copy()
        env_data["grasp_active"] = False
        env_data["grasp_offset"] = None
        env_data["grasp_quat_offset"] = None

        mujoco.mj_forward(model, data)
        vec_env._step_counts = np.zeros(vec_env.num_envs, dtype=int)
        vec_env._episode_rewards = np.zeros(vec_env.num_envs, dtype=np.float64)

        # Reset wrapper (resets GR00T action buffer etc.)
        obs = wrapper.reset()'''

new_block = '''        # Reset wrapper first (this also resets vec_env internally)
        obs = wrapper.reset()

        # Now override positions AFTER reset (reset would overwrite them)
        env_data = vec_env._envs[0]
        data = env_data["data"]
        model = env_data["model"]
        cqa = env_data["cube_qposadr"]
        bowl_body_id = env_data["bowl_body_id"]

        # Set robot joints to real teleop initial pose
        for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
        for fid in env_data["ids"]["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04

        # Set cube position + yaw
        if cqa is not None:
            data.qpos[cqa:cqa + 3] = actual_cube_pos
            q_xyzw = Rotation.from_euler("z", np.radians(yaw_deg)).as_quat()
            data.qpos[cqa + 3:cqa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        # Set bowl position
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = bowl_pos

        env_data["cube_pos_init"] = actual_cube_pos.copy()
        env_data["bowl_pos_init"] = bowl_pos.copy()
        env_data["grasp_active"] = False
        env_data["grasp_offset"] = None
        env_data["grasp_quat_offset"] = None

        mujoco.mj_forward(model, data)
        vec_env._step_counts = np.zeros(vec_env.num_envs, dtype=int)
        vec_env._episode_rewards = np.zeros(vec_env.num_envs, dtype=np.float64)'''

if old_block in code:
    code = code.replace(old_block, new_block)
    with open("scripts/eval_pnp_base_sr.py", "w") as f:
        f.write(code)
    print("Patched successfully: wrapper.reset() now called BEFORE manual env setup")
else:
    print("ERROR: Could not find the block to replace!")
    # Debug: print lines around the area
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'Manually reset' in line or 'wrapper.reset' in line:
            print(f"  Line {i+1}: {line}")
PYEOF
