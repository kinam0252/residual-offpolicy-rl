"""
Key question: when we command ctrl=0.0 to gripper with cup in contact,
why does grip_qpos stop at ~0.028 instead of going lower?

Test: After GR00T episode reaches grasp, 
- Log the FORCE on finger actuators
- Check if arm actuators are interfering
- Check finger qvel (is something resisting?)
"""
import os, sys, numpy as np
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl"))
import mujoco, torch
torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_cup import MuJoCoResidualWrapperCup

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

# Finger actuator IDs
faids = e['finger_actuator_ids']
fid = e['ids']['finger_ids'][0]
fid2 = e['ids']['finger_ids'][1] if len(e['ids']['finger_ids']) > 1 else -1

print("Finger actuator IDs:", faids)
print("Finger joint IDs:", e['ids']['finger_ids'])
print()

# Check actuator properties
for faid in faids:
    print(f"Actuator {faid}:")
    print(f"  gainprm[0] = {model.actuator_gainprm[faid, 0]}")
    print(f"  biasprm = {model.actuator_biasprm[faid, :3]}")
    print(f"  forcelimited = {model.actuator_forcelimited[faid]}")
    print(f"  forcerange = {model.actuator_forcerange[faid]}")
    print(f"  trntype = {model.actuator_trntype[faid]}")
    jnt_id = model.actuator_trnid[faid, 0]
    print(f"  drives joint id = {jnt_id}, name = {model.joint(jnt_id).name}")

# Run until grasp phase
print("\n--- Running episode ---")
for step in range(250):
    obs, reward, terminated, truncated, info = wrapper.step(zero_res)
    grip = data.qpos[model.jnt_qposadr[fid]]
    ctrl0 = data.ctrl[faids[0]]
    
    if step >= 195 and step <= 220:
        # Check actuator force
        actuator_force_0 = data.actuator_force[faids[0]]
        actuator_force_1 = data.actuator_force[faids[1]] if len(faids) > 1 else 0
        
        # Finger qvel
        fqvel = data.qvel[model.jnt_dofadr[fid]]
        
        # Contact forces on fingers
        finger_set = e['finger_geom_id_set']
        cup_gids = set(e['cup_col_gids'])
        max_contact_force = 0
        n_finger_cup = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if (g1 in cup_gids or g2 in cup_gids) and (g1 in finger_set or g2 in finger_set):
                n_finger_cup += 1
                # Get contact force
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, force)
                max_contact_force = max(max_contact_force, np.linalg.norm(force[:3]))
        
        # Grip of both fingers
        grip2 = data.qpos[model.jnt_qposadr[fid2]] if fid2 >= 0 else -1
        
        print(f"s={step:3d} ctrl={ctrl0:.4f} qpos=[{grip:.5f},{grip2:.5f}] "
              f"qvel={fqvel:+.5f} act_force=[{actuator_force_0:.1f},{actuator_force_1:.1f}] "
              f"finger_cup_con={n_finger_cup} max_cforce={max_contact_force:.1f}")

