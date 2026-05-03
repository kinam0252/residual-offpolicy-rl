"""Compare: what grip width does the finger actuator reach with ctrl=0.0
when the cup is in contact, with and without 30deg correction?"""
import os, sys, numpy as np
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl"))
import mujoco, torch
torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_cup import MuJoCoResidualWrapperCup
from scipy.spatial.transform import Rotation

GROOT_CKPT = os.path.expanduser("~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000")

env = MuJoCoVecEnvCup(num_envs=1, episode_ids=[0], max_episode_steps=300,
    reward_type='dense', device='cuda:0',
    cup_positions_path='configs/cup_positions.json', parallel_envs=False)
wrapper = MuJoCoResidualWrapperCup(
    vec_env=env, groot_checkpoint=GROOT_CKPT,
    embodiment_tag='NEW_EMBODIMENT', policy_device='cuda:0',
    task_description="Pick up the cup lying on its side and stand it upright",
    ema_alpha=0.3)

obs, info = wrapper.reset()
e = env._envs[0]
model, data = e['model'], e['data']
zero_res = torch.zeros((1, wrapper.action_dim), device='cuda:0')

# Run until latch triggers
latch_step = -1
for step in range(250):
    obs, reward, terminated, truncated, info = wrapper.step(zero_res)
    fid = e['ids']['finger_ids'][0]
    grip = data.qpos[model.jnt_qposadr[fid]]
    faid = e['finger_actuator_ids'][0]
    ctrl = data.ctrl[faid]
    
    # Count cup contacts
    cup_gids = set(e['cup_col_gids'])
    cup_con = sum(1 for ci in range(data.ncon) 
                  if data.contact[ci].geom1 in cup_gids or data.contact[ci].geom2 in cup_gids)
    
    if step >= 190:
        print(f"s={step} ctrl={ctrl:.4f} grip={grip:.4f} cup_con={cup_con}")
    
    if ctrl < 0.001 and cup_con > 3 and latch_step < 0:
        latch_step = step
        print(f"\n=== LATCH at step {step}, grip={grip:.4f} ===")
        
        # Now snapshot and try: (A) no correction, (B) 30deg correction
        qpos_snap = data.qpos.copy()
        qvel_snap = data.qvel.copy()
        ctrl_snap = data.ctrl.copy()
        
        # --- Case A: No correction, just keep stepping with ctrl=0 ---
        print("\n--- Case A: No correction ---")
        data.ctrl[faid] = 0.0
        if len(e['finger_actuator_ids']) > 1:
            data.ctrl[e['finger_actuator_ids'][1]] = 0.0
        for sub_step in range(20):
            for _ in range(e['n_substeps']):
                mujoco.mj_step(model, data)
            g = data.qpos[model.jnt_qposadr[fid]]
            nc = sum(1 for ci in range(data.ncon) 
                     if data.contact[ci].geom1 in cup_gids or data.contact[ci].geom2 in cup_gids)
            if sub_step % 5 == 0:
                print(f"  substep={sub_step} grip={g:.5f} cup_con={nc}")
        grip_no_corr = data.qpos[model.jnt_qposadr[fid]]
        
        # --- Case B: 30deg correction then keep stepping with ctrl=0 ---
        data.qpos[:] = qpos_snap
        data.qvel[:] = qvel_snap
        data.ctrl[:] = ctrl_snap
        mujoco.mj_forward(model, data)
        
        # Apply 30deg upright correction to cup
        cup_qpa = e['cup_qposadr']
        cur_quat_wxyz = data.qpos[cup_qpa+3:cup_qpa+7].copy()
        cur_R = Rotation.from_quat([cur_quat_wxyz[1], cur_quat_wxyz[2],
                                    cur_quat_wxyz[3], cur_quat_wxyz[0]])
        cup_z_world = cur_R.as_matrix()[:, 2]
        rot_axis = np.cross(cup_z_world, [0, 0, 1])
        if np.linalg.norm(rot_axis) > 1e-6:
            rot_axis /= np.linalg.norm(rot_axis)
            correction_R = Rotation.from_rotvec(rot_axis * np.radians(30.0))
            new_R = correction_R * cur_R
            new_q = new_R.as_quat()  # xyzw
            data.qpos[cup_qpa+3] = new_q[3]   # w
            data.qpos[cup_qpa+4:cup_qpa+7] = new_q[:3]  # xyz
        mujoco.mj_forward(model, data)
        
        print("\n--- Case B: 30deg correction applied ---")
        data.ctrl[faid] = 0.0
        if len(e['finger_actuator_ids']) > 1:
            data.ctrl[e['finger_actuator_ids'][1]] = 0.0
        for sub_step in range(20):
            for _ in range(e['n_substeps']):
                mujoco.mj_step(model, data)
            g = data.qpos[model.jnt_qposadr[fid]]
            nc = sum(1 for ci in range(data.ncon) 
                     if data.contact[ci].geom1 in cup_gids or data.contact[ci].geom2 in cup_gids)
            if sub_step % 5 == 0:
                print(f"  substep={sub_step} grip={g:.5f} cup_con={nc}")
        grip_with_corr = data.qpos[model.jnt_qposadr[fid]]
        
        print(f"\n=== RESULT: no_correction grip={grip_no_corr:.5f}, with_30deg grip={grip_with_corr:.5f} ===")
        break

if latch_step < 0:
    print("Latch never triggered!")
