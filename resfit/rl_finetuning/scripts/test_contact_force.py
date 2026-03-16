"""
Test contact force reading by replaying CSV joint poses.
Run: docker exec isaaclab_resfit bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/test_contact_force.py --headless"
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, sys, torch, numpy as np, pandas as pd

# Imports after AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

CSV_DIR = "/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081744_174"

# Scene config (1 env) with contact sensors enabled
from isaaclab.terrains import TerrainImporterCfg

scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5)
scene_cfg.terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5, restitution=0.0),
)
scene_cfg.robot = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"panda_joint1": 1.157, "panda_joint2": -1.066, "panda_joint3": -0.155,
                    "panda_joint4": -2.239, "panda_joint5": -1.841, "panda_joint6": 1.003,
                    "panda_joint7": 0.469, "panda_finger_joint.*": 0.035},
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(joint_names_expr=["panda_joint[1-4]"], effort_limit_sim=87.0, stiffness=8000.0, damping=400.0),
        "panda_forearm": ImplicitActuatorCfg(joint_names_expr=["panda_joint[5-7]"], effort_limit_sim=12.0, stiffness=8000.0, damping=400.0),
        "panda_hand": ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"], effort_limit_sim=500.0, stiffness=2e4, damping=1500.0),
    },
)
scene_cfg.cube = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.05, 0.12, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=2.0, disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.15, 0.15)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.06)),
)

# Create sim
sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.3])

scene = InteractiveScene(scene_cfg)
sim.reset()
scene.reset()

robot = scene["robot"]
cube = scene["cube"]

# Get joint IDs
arm_names = sorted([n for n in robot.data.joint_names if "panda_joint" in n and "finger" not in n])
hand_names = sorted([n for n in robot.data.joint_names if "panda_finger_joint" in n])
arm_ids = [robot.data.joint_names.index(n) for n in arm_names]
hand_ids = [robot.data.joint_names.index(n) for n in hand_names]
all_ids = arm_ids + hand_ids

# Finger body indices
left_finger_idx = robot.find_bodies("panda_leftfinger")[0][0]
right_finger_idx = robot.find_bodies("panda_rightfinger")[0][0]
print(f"Finger body indices: left={left_finger_idx}, right={right_finger_idx}")

# Load CSV
js = pd.read_csv(f"{CSV_DIR}/franka_joint_states.csv", header=0).values.astype(np.float32)
gs = pd.read_csv(f"{CSV_DIR}/gripper_joint_states.csv", header=0).values.astype(np.float32)
js = js[1:, 10:17]  # joint positions
gs = gs[1:, 4:5]    # gripper width
print(f"CSV: {js.shape[0]} timesteps")

# Initialize contact view
contact_view = None
contact_initialized = False

def init_contact_view():
    global contact_view, contact_initialized
    try:
        from omni.physics.tensors import create_simulation_view
    except ImportError:
        from omni.physx.tensors import create_simulation_view
    
    physics_sim_view = create_simulation_view("torch")
    physics_sim_view.set_subspace_roots("/")
    pattern = "/World/envs/env_*/Robot/panda_leftfinger"
    contact_view = physics_sim_view.create_rigid_contact_view(pattern)
    contact_initialized = True
    print(f"Contact view initialized: {pattern}")
    # Check prim paths
    if hasattr(contact_view, 'prim_paths'):
        for i, p in enumerate(contact_view.prim_paths):
            print(f"  [{i}] {p}")

# Also try reading from robot's physx view directly
def get_contact_from_robot_view():
    """Alternative: read contact forces from robot.root_physx_view"""
    try:
        forces = robot.root_physx_view.get_link_incoming_joint_force()
        return forces
    except Exception as e:
        return None

# Replay CSV poses
print("\n=== Replaying CSV joint poses ===")
max_steps = min(500, js.shape[0])
max_contact_force_seen = 0.0

for step in range(max_steps):
    arm_q = js[step, :7]
    grip = float(np.clip(gs[step, 0], 0.0, 1.0)) * 0.04
    q = np.concatenate([arm_q, [grip] * len(hand_ids)]).astype(np.float32)
    q_t = torch.tensor(q, device="cuda:0").unsqueeze(0)
    
    robot.set_joint_position_target(q_t, joint_ids=all_ids)
    scene.write_data_to_sim()
    sim.step()
    scene.update(0.01)
    
    # Initialize contact view after first sim step
    if not contact_initialized and step >= 5:
        init_contact_view()
    
    # Read contact forces
    contact_fx = 0.0
    contact_fy = 0.0
    contact_fz = 0.0
    contact_mag = 0.0
    
    if contact_view is not None and contact_initialized:
        try:
            forces = contact_view.get_net_contact_forces(dt=0.01)
            n_bodies = forces.shape[0]
            forces_reshaped = forces.view(1, n_bodies, 3)  # (1 env, n_bodies, 3)
            left_force = forces_reshaped[0, 0, :]
            contact_fx = float(left_force[0].item())
            contact_fy = float(left_force[1].item())
            contact_fz = float(left_force[2].item())
            contact_mag = float(torch.norm(left_force).item())
            if contact_mag > max_contact_force_seen:
                max_contact_force_seen = contact_mag
        except Exception as e:
            if step % 100 == 0:
                print(f"  Contact read error at step {step}: {e}")
    
    # Finger positions
    left_fp = robot.data.body_pos_w[0, left_finger_idx]
    right_fp = robot.data.body_pos_w[0, right_finger_idx]
    cube_pos = cube.data.root_state_w[0, :3]
    finger_avg = (left_fp + right_fp) / 2
    finger_cube_dist = float(torch.norm(finger_avg - cube_pos).item())
    cube_h = float(cube_pos[2].item())
    finger_joint = float(robot.data.joint_pos[0, hand_ids[0]].item())
    
    if step % 20 == 0:
        print(f"  step={step:4d} | contact: fx={contact_fx:8.3f} fy={contact_fy:8.3f} fz={contact_fz:8.3f} mag={contact_mag:8.3f} "
              f"| finger_dist={finger_cube_dist:.4f} cube_z={cube_h:.4f} finger_j={finger_joint:.4f}")

print(f"\n=== SUMMARY ===")
print(f"Max contact force magnitude seen: {max_contact_force_seen:.4f}")
print(f"Contact force ever non-zero: {max_contact_force_seen > 0.01}")

# Force exit
import os
os._exit(0)
