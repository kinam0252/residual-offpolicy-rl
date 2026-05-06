"""Check TCP position at the moment of grasp in RL env.
Also check cup geometry to understand where the rim vs body is."""
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

# Cup geometry info
cup_qpa = e['cup_qposadr']
print("=== CUP GEOMETRY ===")
for gid in e['cup_col_gids']:
    geom = model.geom(gid)
    print(f"  geom '{geom.name}': type={geom.type}, size={geom.size[:3]}, pos={geom.pos}")

# Finger geom info
print("\n=== FINGER GEOMETRY ===")
for gid in e['finger_geom_ids']:
    geom = model.geom(gid)
    print(f"  geom '{geom.name}': type={geom.type}, size={geom.size[:3]}, pos={geom.pos}")

# Hand (TCP) site
hand_id = e['ids']['hand_id']
print(f"\nHand body id: {hand_id}, name: {model.body(hand_id).name}")

# Run to grasp
print("\n=== TCP & CUP POSITION AT GRASP ===")
for step in range(250):
    obs, reward, terminated, truncated, info = wrapper.step(zero_res)
    fid = e['ids']['finger_ids'][0]
    grip = data.qpos[model.jnt_qposadr[fid]]
    ctrl = data.ctrl[e['finger_actuator_ids'][0]]
    
    cup_pos = data.qpos[cup_qpa:cup_qpa+3].copy()
    tcp_pos = data.xpos[hand_id].copy()
    
    # Also get fingertip positions (geom positions in world frame)
    finger_gids = e['finger_geom_ids']
    ftip_pos = [data.geom_xpos[gid].copy() for gid in finger_gids]
    
    if 185 <= step <= 215:
        cup_gids = set(e['cup_col_gids'])
        cup_con = sum(1 for ci in range(data.ncon) 
                     if data.contact[ci].geom1 in cup_gids or data.contact[ci].geom2 in cup_gids)
        
        # Which cup geoms are in contact with fingers?
        finger_set = e['finger_geom_id_set']
        contact_pairs = []
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if (g1 in cup_gids and g2 in finger_set):
                contact_pairs.append((model.geom(g1).name, model.geom(g2).name))
            elif (g2 in cup_gids and g1 in finger_set):
                contact_pairs.append((model.geom(g2).name, model.geom(g1).name))
        
        print(f"s={step:3d} ctrl={ctrl:.4f} grip={grip:.4f} "
              f"tcp=[{tcp_pos[0]:.4f},{tcp_pos[1]:.4f},{tcp_pos[2]:.4f}] "
              f"cup=[{cup_pos[0]:.4f},{cup_pos[1]:.4f},{cup_pos[2]:.4f}] "
              f"cup_con={cup_con}")
        if contact_pairs and step % 5 == 0:
            unique_pairs = list(set(contact_pairs))
            print(f"       contact pairs: {unique_pairs[:6]}")

