"""Debug grasp detection - check contacts, gripper width, grasp state frame by frame.
Uses ResidualWrapper with zero residual (= base GR00T policy only)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["MUJOCO_GL"] = "egl"
import json
import numpy as np
import mujoco
from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import (
    MuJoCoVecEnvPnP, GRIPPER_CLOSE_THRESHOLD, GRASP_CONTACT_THRESHOLD
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP
from scipy.spatial.transform import Rotation

GROOT_CKPT = "/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000"
SCENE_XML = os.path.expanduser(
    "~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
)
POS_FILE = "configs/pnp_eval_medium_hard.json"

with open(POS_FILE) as f:
    all_pos = json.load(f)
# Use first 3 hard positions (idx 10-12 = hard in medium_hard)
hard_positions = [p for p in all_pos if p.get("distance", 0) >= 0.23][:3]
if not hard_positions:
    hard_positions = all_pos[10:13]

num_envs = len(hard_positions)
print(f"Using {num_envs} hard positions")

cube_positions = [p["cube_pos"] for p in hard_positions]
bowl_positions = [p["bowl_pos"] for p in hard_positions]
cube_yaws = [p["cube_yaw_deg"] for p in hard_positions]

vec_env = MuJoCoVecEnvPnP(
    num_envs=num_envs,
    cube_positions=cube_positions,
    bowl_positions=bowl_positions,
    max_episode_steps=500,
    reward_type="dense_v3",
    scene_xml=SCENE_XML,
)

wrapper = MuJoCoResidualWrapperPnP(
    vec_env=vec_env,
    groot_checkpoint=GROOT_CKPT,
    task_description="Pick up the red cube and place it onto the plate.",
    ema_alpha=0.0,
)

obs, _ = wrapper.reset()

# Override positions
for ei in range(num_envs):
    env_data = vec_env._envs[ei]
    data, model = env_data["data"], env_data["model"]
    cqa = env_data["cube_qposadr"]
    if cqa is not None:
        data.qpos[cqa:cqa+3] = cube_positions[ei]
        q_xyzw = Rotation.from_euler("z", np.radians(cube_yaws[ei])).as_quat()
        data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    bowl_body_id = env_data["bowl_body_id"]
    if bowl_body_id >= 0:
        model.body_pos[bowl_body_id] = bowl_positions[ei]
    env_data["cube_pos_init"] = np.array(cube_positions[ei])
    env_data["bowl_pos_init"] = np.array(bowl_positions[ei])
    mujoco.mj_forward(model, data)

vec_env._step_counts = np.zeros(num_envs, dtype=int)
vec_env._episode_rewards = np.zeros(num_envs, dtype=np.float64)

obs, _ = wrapper.reset()
for ei in range(num_envs):
    env_data = vec_env._envs[ei]
    data, model = env_data["data"], env_data["model"]
    cqa = env_data["cube_qposadr"]
    if cqa is not None:
        data.qpos[cqa:cqa+3] = cube_positions[ei]
        q_xyzw = Rotation.from_euler("z", np.radians(cube_yaws[ei])).as_quat()
        data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    bowl_body_id = env_data["bowl_body_id"]
    if bowl_body_id >= 0:
        model.body_pos[bowl_body_id] = bowl_positions[ei]
    env_data["cube_pos_init"] = np.array(cube_positions[ei])
    env_data["bowl_pos_init"] = np.array(bowl_positions[ei])
    mujoco.mj_forward(model, data)

vec_env._step_counts = np.zeros(num_envs, dtype=int)
vec_env._episode_rewards = np.zeros(num_envs, dtype=np.float64)

# Print collision info
env0 = vec_env._envs[0]
model = env0["model"]
cube_gid = env0["cube_geom_id"]
finger_gids = env0["finger_geom_ids"]
print(f"cube_geom_id={cube_gid}, finger_geom_ids={finger_gids}")
for gid in finger_gids + [cube_gid]:
    ct = model.geom_contype[gid]
    ca = model.geom_conaffinity[gid]
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
    print(f"  geom{gid} ({bname}): contype={ct} conaffinity={ca}")

# Step loop
env_done = [False] * num_envs
MAX_STEPS = 500

for step in range(MAX_STEPS):
    action = np.zeros((num_envs, wrapper.action_dim))
    obs, rew, term, trunc, info = wrapper.step(action)

    for ei in range(num_envs):
        if env_done[ei]:
            continue
        env_data = vec_env._envs[ei]
        data = env_data["data"]
        model = env_data["model"]
        
        # Gripper width
        fids = env_data["ids"]["finger_ids"]
        if fids:
            finger_qpos = data.qpos[model.jnt_qposadr[fids[0]]]
            gripper_width = finger_qpos / 0.04
        else:
            gripper_width = -1
        
        # Contacts
        contacts = vec_env._get_contacts(ei)
        left_c, right_c, both_c = contacts
        
        # Grasp state
        grasped = env_data["grasp_state"]["grasped"]
        contact_count = env_data["grasp_state"]["contact_count"]
        
        # TCP-cube distance
        from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import get_tcp_pose
        tcp_pos, _ = get_tcp_pose(model, data, env_data["ids"]["hand_id"])
        cube_body = model.geom_bodyid[env_data["cube_geom_id"]]
        cube_pos = data.xpos[cube_body].copy()
        tcp_cube_dist = np.linalg.norm(tcp_pos - cube_pos)
        
        # Raw cube contacts
        cube_contacts = []
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if env_data["cube_geom_id"] in (g1, g2):
                n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"g{g1}"
                n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"g{g2}"
                cube_contacts.append(f"{n1}↔{n2}")
        
        closing = gripper_width < GRIPPER_CLOSE_THRESHOLD
        
        # Print: every 25 steps, on contact, on grasp, when close
        should_print = (step % 25 == 0 or grasped or both_c > 0 or 
                       left_c > 0 or right_c > 0 or tcp_cube_dist < 0.05)
        if should_print:
            print(f"env{ei} step={step:3d}  dist={tcp_cube_dist:.3f}  "
                  f"gw={gripper_width:.3f}  closing={closing}  "
                  f"L={left_c:.0f} R={right_c:.0f} B={both_c:.0f}  "
                  f"cnt={contact_count}  grasped={grasped}  "
                  f"contacts={cube_contacts}")
        
        done = (term[ei] if hasattr(term, '__getitem__') else term) or \
               (trunc[ei] if hasattr(trunc, '__getitem__') else trunc)
        if done:
            env_done[ei] = True
            success = info.get("is_success", [False]*num_envs)[ei] if isinstance(info, dict) else False
            print(f"*** env{ei} DONE at step {step}, success={success} ***")

    if all(env_done):
        break

for ei in range(num_envs):
    env_data = vec_env._envs[ei]
    print(f"\nenv{ei} final: grasped={env_data['grasp_state']['grasped']}, "
          f"contact_count={env_data['grasp_state']['contact_count']}")
