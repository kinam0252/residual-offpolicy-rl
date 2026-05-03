import sys, os, numpy as np, torch
torch.backends.cuda.enable_cudnn_sdp(False)
os.chdir('/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl')
sys.path.insert(0, '.')
from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_cup import MuJoCoResidualWrapperCup

env = MuJoCoVecEnvCup(num_envs=1, episode_ids=[0], max_episode_steps=250,
    reward_type='dense', device='cuda:0',
    cup_positions_path='configs/cup_positions.json', parallel_envs=False)
wrapper = MuJoCoResidualWrapperCup(
    vec_env=env, groot_checkpoint=os.path.expanduser(
        '~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000'),
    embodiment_tag='NEW_EMBODIMENT', policy_device='cuda:0',
    task_description="Pick up the cup lying on its side and stand it upright",
    ema_alpha=0.3)

obs, info = wrapper.reset()
e = env._envs[0]
model, data = e['model'], e['data']
zero_res = torch.zeros((1, wrapper.action_dim), device='cuda:0')

for step in range(250):
    obs, reward, terminated, truncated, info = wrapper.step(zero_res)
    fid = e['ids']['finger_ids'][0]
    grip = data.qpos[model.jnt_qposadr[fid]]
    ctrl = data.ctrl[e['finger_actuator_ids'][0]]
    gs = e['grasp_state']
    cup_qpa = e['cup_qposadr']
    cup_z = data.qpos[cup_qpa+2]
    cup_gids = set(e['cup_col_gids'])
    cup_con = sum(1 for ci in range(data.ncon)
                 if data.contact[ci].geom1 in cup_gids or data.contact[ci].geom2 in cup_gids)
    tcp_z = data.xpos[e['ids']['hand_id']][2]
    
    if step >= 180:
        print(f"s={step:3d} ctrl={ctrl:.4f} grip={grip:.4f} cup_con={cup_con:2d} "
              f"grasped={str(gs['grasped']):5s} cup_z={cup_z:.4f} tcp_z={tcp_z:.4f}")
