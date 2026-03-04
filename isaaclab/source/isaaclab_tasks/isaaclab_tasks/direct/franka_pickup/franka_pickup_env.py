# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import collections
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import omni.usd
import omni.timeline
import json
import cv2
import pickle
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms, euler_xyz_from_quat
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors.camera import Camera, CameraCfg

try:
    from gr00t.policy.server_client import PolicyClient
except ImportError:
    PolicyClient = None

"""
python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Franka-Pickup-Direct-v0 --num_envs=20 --max_iterations=1000000000 --enable_cameras
"""

@configclass
class FrankaPickupEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 16.67 # ~1000 steps at env dt=1/60
    decimation = 2
    action_horizon_reset_steps = 1000  # Force reset when step count exceeds this horizon
    # Action space breakdown:
    # - actions[0:3]: delta position (dx, dy, dz) relative to API reference pose (scaled by delta_pos_scale)
    # - actions[3:6]: delta orientation (roll, pitch, yaw) relative to API reference pose (scaled by delta_rpy_scale)
    # - actions[6]: gripper control (single dimension for gripper width, mapped to [min_gripper_width, max_gripper_width])
    action_space = 7
    # Observation space calculation:
    # - dof_pos_scaled: 9 (all joints)
    # - joint_vel: 9 (all joints)
    ## - cube_pos: 3
    ## - cube_rot: 4
    ## - to_target: 3
    # - gr00t_reference: 7 (GR00T reference pose for this timestep, one of N in window)
    # - contact_force_binary: 3 (binary 3D force for left finger: 1 if |force| > 0.1, else 0)
    # - endeffector_xyzrpy: 6 (XYZ position + RPY orientation)
    # - latent_value: 4096 (kept for observation shape compatibility; filled with zeros)
    # Total: 9 + 9 + 7 + 3 + 6 + 4096 = 4130
    observation_space = 4130
    state_space = 0

    # Camera settings
    use_camera = False                  # Enable/disable camera (requires --enable_cameras flag when running)
    save_camera_images = False          # Enable/disable camera image saving
    camera_save_interval = 10           # Save images every N steps
    camera_save_dir = "camera_output"   # Directory to save images

    # cube parameters
    cube_half_size = 0.02  # Half size of the cube in meters
    
    # exploration parameters
    exploration_noise = 0.2  # Noise level for actions to encourage exploration
    gripper_exploration_noise = 0.4  # Additional noise level specifically for gripper actions (higher = more exploration)
    joint_reset_noise = 0.0  # Keep deterministic to match GR00T reference scene
    
    # Action normalization: scale raw policy outputs before tanh to prevent saturation
    # Larger values = less saturation (more range in normalized actions)
    # Smaller values = more saturation (actions closer to -1 or 1)
    # Example: if raw actions are ~600, scale=100 gives tanh(6)≈0.9999, scale=500 gives tanh(1.2)≈0.83
    action_tanh_scale = 100.0  # Scale factor: tanh(action / scale) gives better range


    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=True,  # Enable contact sensors for contact force observations
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=15.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=32,  # More position iterations for contact resolution
                solver_velocity_iteration_count=6  # More velocity iterations for contact resolution
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=8000.0,
                damping=400.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=8000.0,
                damping=400.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=500.0,  # Moderate force for better grip without crushing
                stiffness=2e4,  # Balanced stiffness for responsive grip
                damping=900.0,  # Good damping for stability
            ),
        },
    )

    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.12, cube_half_size*2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=2.0,
                disable_gravity=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                enable_gyroscopic_forces=True,
                retain_accelerations=False,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, 0.0, 0.06),
            rot=(0.707, 0.0, 0.0, 0.707),  # Quaternion for 90° rotation around X-axis
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0)
        ),
    )

    camera_front = CameraCfg(
        prim_path="/World/envs/env_.*/DebugCamera_front",
        update_period=0,  # Update every frame
        height=480,
        width=640,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(0.4, -0.7, 0.8),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0)
        ),
    )

    camera_back = CameraCfg(
        prim_path="/World/envs/env_.*/DebugCamera_back",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(0.4, 0.7, 0.8),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
    )

    camera_wrist = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(0.06, 0.0, 0.0),
            rot=(0.69636424, -0.1227878, -0.1227878, 0.69636424),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=0.5,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
    )
    
    # GR00T policy server endpoints
    gr00t_host: str = "127.0.0.1"
    gr00t_port: int = 5555
    task_description: str = "Pick up the white object."  # Task description for GR00T
    
    # API call settings
    api_call_interval: int = 16  # Call API every N steps (16 = matches API's 16 action windows, one window per step)
    use_api_for_pose: bool = True  # Use API instead of CSV for end-effector poses
    post_reset_settle_steps: int = 30  # Hold reset pose for N sim-steps before sending first inference
    target_action_fps: float = 20.0
    rot_clamp_rad: float = 0.35
    # Robustness / performance
    api_max_envs_per_step: int = 1      # hard cap on number of envs that can call API per sim step
    api_fail_cooldown_steps: int = 60   # wait this many sim steps before retrying after a failure
    api_disable_after_consecutive_failures: int = 10  # set to 0 to never auto-disable

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    dof_velocity_scale = 0.1
    
    # Delta pose adjustment parameters (for RL to adjust reference poses from API or CSV)
    delta_pos_scale = 0.12 # Scale factor for delta position adjustments (meters)
    delta_rpy_scale = 0.2   # Scale factor for delta RPY adjustments (radians)
    # When using API: actions[0:3] adjust API reference position, actions[3:6] adjust API reference orientation
    # When using CSV: actions[0:3] adjust CSV reference position, actions[3:6] adjust CSV reference orientation
    # actions[6]: gripper control (always controlled by RL)

    # reward scales - using tanh-kernel with std for smoother rewards
    # Distance reward: tanh-kernel for gripper-cube distance
    distance_reward_std = 0.1        # Standard deviation for distance reward (smaller = more sensitive)
    distance_reward_weight = 1.0     # Weight for distance reward  # Height threshold for success (3cm)
    
    # Contact reward: encourage stable grasping with both fingers
    contact_reward_weight = 1.0     # Weight for contact reward
    contact_threshold = 0.1         # Minimum contact force to be considered in contact (N)
    # Gate contact reward: only count contact if fingertip is below the cube center height + margin.
    # Set margin=0.0 to simply require finger_z < cube_center_z.
    finger_length = 0.06
    contact_above_cube_margin = cube_half_size + finger_length - 0.01
    contact_ground_threshold = 0.01  # Minimum finger height above ground to receive contact reward (meters)
    
    # Gripper constraints
    min_gripper_width = 0.03  # Minimum gripper width in meters (3cm)
    max_gripper_width = 0.12 # Maximum gripper width in meters (12cm) - normalized action range

    # Height reward: tanh-kernel for cube height
    height_reward_std = 0.1         # Standard deviation for height reward
    height_reward_weight = 100.0     # Weight for height reward
    minimal_height = 0.005             # Minimum height 
    
    # Success bonus: binary reward if lifted above threshold
    success_reward_scale = 100.0      # Bonus reward for successful lift
    height_threshold = 0.03       
    
    # Rolling detection: terminate if cube is rolling on the ground
    rolling_lin_vel_threshold = 0.6  # Horizontal linear velocity threshold (m/s) to detect rolling
    rolling_ang_vel_threshold = 2.0  # Angular velocity threshold (rad/s) to detect rolling
    rolling_min_displacement = 0.05  # Minimum horizontal displacement from initial position to consider rolling (5cm)

    use_wandb = True  # Enable wandb logging for reward tracking

    # success tracking parameters
    success_tracking_window = 1000  # Number of recent trials to track for success rate

    # CSV file paths for initial poses (optional - if None, uses default randomization)
    csv_base_dir: str | None = None  # Disabled by default to keep deterministic GR00T-aligned initialization
    joint_states_csv: str = 'franka_joint_states.csv'  # Filename for joint states CSV
    gripper_states_csv: str = 'gripper_joint_states.csv'  # Filename for gripper states CSV
    object_pos_csv: str = 'object_pos.csv'  # Filename for object position CSV
    csv_init_row_index: int = 0  # Always initialize from this CSV row (after preprocessing)
    enforce_csv_reset_init: bool = True  # If True, fail fast when CSV init data is unavailable

    # Debug compare mode (for matching behavior with standalone GR00T script)
    debug_compare_mode: bool = False
    debug_compare_interval: int = 120
    debug_compare_dump: bool = False
    debug_compare_dump_dir: str = "logs/skrl/debug"
    reference_match_mode: bool = False
    


class FrankaPickupEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaPickupEnvCfg

    VIDEO_KEYS = ("wrist_view", "left_view", "right_view")
    STATE_JOINT_KEY = "proprio.joint_pos"
    STATE_GRIPPER_KEY = "proprio.gripper_pos"
    LANGUAGE_KEY = "annotation.human.action.task_description"

    def __init__(self, cfg: FrankaPickupEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        
        # Initialize wandb if available

        if self.cfg.use_wandb:
            try:
                # Dynamic import of wandb to avoid linter errors
                wandb_module = __import__('wandb', fromlist=['init', 'run'])
                if hasattr(wandb_module, 'run') and wandb_module.run is None:
                    # Use getattr to access init method to avoid linter errors
                    getattr(wandb_module, 'init')(
                        project="Isaaclab_Franka_Pickup", 
                        name="franka_pickup_direct_velocity_add_height_bonus_tune",
                    )
            except ImportError:
                print("wandb not installed, logging disabled")
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")

        self._debug_compare_last_ts = time.time()
        self._debug_compare_file_path = None
        self._debug_compare_header_printed = False
        self._camera_setup_log_printed = False
        if self.cfg.debug_compare_mode and self.cfg.debug_compare_dump:
            os.makedirs(self.cfg.debug_compare_dump_dir, exist_ok=True)
            self._debug_compare_file_path = os.path.join(
                self.cfg.debug_compare_dump_dir,
                f"franka_gr00t_compare_{int(time.time())}.jsonl",
            )
            print(f"[GR00T-DEBUG] JSONL dump path: {self._debug_compare_file_path}")

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.steps_per_action = max(1, int(round((1.0 / float(self.cfg.target_action_fps)) / float(self.dt))))
        if self.cfg.use_api_for_pose:
            print(
                f"[INFO] GR00T control timing: dt={self.dt:.6f}s, target_action_fps={self.cfg.target_action_fps}, "
                f"steps_per_action={self.steps_per_action}"
            )
            if self.cfg.reference_match_mode:
                print("[INFO] reference_match_mode enabled: CSV init + API action semantics (no exploration/RL delta overlay)")

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)


        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        # Gripper width normalization: actions will be mapped to [min_gripper_width, max_gripper_width]
        print(f"[INFO] Gripper width normalization: [{self.cfg.min_gripper_width*100:.1f}cm, {self.cfg.max_gripper_width*100:.1f}cm]")

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.reset_joint_pos_targets = self._robot.data.default_joint_pos.clone().to(device=self.device)

        stage = get_current_stage()
        PRIM ="/World/envs/env_0/Robot" 

        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath(PRIM+"/panda_hand")),
            self.device,
        )
        left_finger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath(PRIM+"/panda_leftfinger")),
            self.device,
        )
        right_finger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath(PRIM+"/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (left_finger_pose[0:3] + right_finger_pose[0:3]) / 2.0
        finger_pose[3:7] = (left_finger_pose[3:7] + right_finger_pose[3:7]) / 2.0
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        # Find finger body indices for contact force observations
        left_finger_body_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.finger_body_indices = [left_finger_body_idx, right_finger_body_idx]
        
        # Initialize contact view for robot finger bodies (will be created lazily on first use)
        # This requires activate_contact_sensors=True in robot config
        self._contact_physx_view = None
        self._contact_view_initialized = False
        
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Initialize finger position (average of left and right fingers) for distance reward
        self.finger_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # Store individual finger positions (world frame) for contact gating
        self.left_finger_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_finger_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Store 3D force vectors for observations and rewards
        self.left_finger_force = torch.zeros((self.num_envs, 3), device=self.device)
        # self.right_finger_force = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Initialize cube position and rotation for all environments
        self.cube_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_rot = torch.zeros((self.num_envs, 4), device=self.device)
        
        # Initialize cube velocity for rolling detection
        self.cube_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

        # Track initial cube positions
        self.cube_initial_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Initialize success tracking
        self.lift_successes = collections.deque(maxlen=self.cfg.success_tracking_window)
        # self.total_episodes = 0
        self.num_lift_successes = 0
        self.current_episode_has_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Initialize CSV data for initial poses
        self.csv_data_list = []  # List of dicts, each containing joint_states, gripper_states, object_pos for one directory
        self.csv_dir_paths = []  # List of directory paths for each CSV dataset
        self.csv_row_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Track current CSV row index for each environment
        self.csv_step_counters = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Track steps since last CSV row update
        self.latent_dim = 4096  # Latent dimension (will be set after loading latent data)
        self._load_csv_data()
        self._api_gripper_action = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        
        # Initialize Differential IK Controller for end-effector control (using API)
        self.robot_entity_cfg = None
        self.diff_ik_controller = None
        self.arm_joint_ids = None
        if self.cfg.use_api_for_pose:
            # Set up robot entity configuration for IK
            robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
            robot_entity_cfg.resolve(self.scene)
            self.robot_entity_cfg = robot_entity_cfg
            
            # Get arm joint IDs directly from joint names (matching reference script)
            arm_joint_names = [name for name in self._robot.data.joint_names if "panda_joint" in name and "finger" not in name]
            arm_joint_ids_list = [self._robot.data.joint_names.index(name) for name in sorted(arm_joint_names)]
            # Convert to tensor for proper indexing
            self.arm_joint_ids = torch.tensor(arm_joint_ids_list, device=self.device, dtype=torch.long)
            
            # Create Differential IK Controller
            diff_ik_cfg = DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,  # Use absolute poses in world frame
                ik_method="dls"
            )
            self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
            self.diff_ik_controller.reset()
            if self.cfg.use_api_for_pose:
                print(f"[INFO] Initialized Differential IK Controller for API-based end-effector control")
            else:
                print(f"[INFO] Initialized Differential IK Controller for end-effector control from CSV")
            print(f"[INFO] Arm joint IDs: {self.arm_joint_ids.cpu().tolist()}")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cube = RigidObject(self.cfg.cube)
        
        self.scene.articulations["robot"] = self._robot
        
        # Initialize camera only if enabled in config or using API
        self._camera_enabled = False
        self._camera_front = None
        self._camera_back = None
        self._camera_wrist = None
        if self.cfg.use_camera or self.cfg.use_api_for_pose:
            try:
                self._camera_front = Camera(self.cfg.camera_front)
                self._camera_back = Camera(self.cfg.camera_back)
                self._camera_wrist = Camera(self.cfg.camera_wrist)
                self.scene.sensors["camera_front"] = self._camera_front
                self.scene.sensors["camera_back"] = self._camera_back
                self.scene.sensors["camera_wrist"] = self._camera_wrist
                self._camera_enabled = True
            except RuntimeError as e:
                # Camera initialization failed (likely because --enable_cameras flag not set)
                print(f"Warning: Camera initialization failed: {e}")
                print("Continuing without camera support.")
            except Exception as e:
                print(f"Warning: Error during camera setup: {e}")

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # Add floor style to align with GR00T inference scene setup
        floor_cfg = sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        floor_cfg.func(
            "/World/Environment/floor",
            floor_cfg,
            translation=(0.0, 0.0, -0.01),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        if bool(getattr(self.cfg, "reference_match_mode", False)):
            stage = get_current_stage()
            for prim_path in ("/World/ground", "/World/defaultGroundPlane", "/World/GroundPlane"):
                prim = stage.GetPrimAtPath(prim_path)
                if prim and prim.IsValid():
                    try:
                        UsdGeom.Imageable(prim).MakeInvisible()
                        print(f"[INFO] reference_match: hid default ground prim '{prim_path}' to avoid floor overlap")
                    except Exception as e:
                        print(f"[WARN] reference_match: failed to hide '{prim_path}': {e}")

        # Add lights to align with GR00T inference scene setup
        # Keep default dome from adding extra brightness.
        default_dome_off_cfg = sim_utils.DomeLightCfg(intensity=0.0, color=(1.0, 1.0, 1.0))
        default_dome_off_cfg.func("/World/defaultDomeLight", default_dome_off_cfg)

        dome_light_cfg = sim_utils.DomeLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0))
        dome_light_cfg.func("/World/Lights/DomeLight", dome_light_cfg)
        sun_light_cfg = sim_utils.DistantLightCfg(intensity=500.0, color=(1.0, 0.98, 0.95), angle=0.53)
        sun_light_cfg.func("/World/Lights/SunLight", sun_light_cfg)
        try:
            key_light_cfg = sim_utils.RectLightCfg(
                intensity=3000.0,
                color=(1.0, 0.98, 0.95),
                width=1.2,
                height=0.8,
            )
            key_light_cfg.func("/World/Lights/KeyLight", key_light_cfg)
        except AttributeError:
            pass
        
        # Note: We'll set up camera positions in the first step to ensure camera is fully initialized
        self._camera_setup_done = False

        # Defer contact view init until physics scene becomes active.
        self._contact_init_retry_interval = 20
        self._contact_init_next_step = 0
        self._contact_init_wait_log_emitted = False
        
        # Initialize API call tracking (per-environment)
        self.api_step_counter = 0
        self.current_action_windows = [None] * self.num_envs  # Store current action windows from API (one per env)
        self.current_window_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Current action window index per env
        self.steps_in_current_window = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Steps in current window per env
        self.num_windows = [None] * self.num_envs  # Number of windows per env
        self.steps_per_window = [1] * self.num_envs  # Steps per window per env
        self.current_mapped_latent = [None] * self.num_envs  # Store current mapped latent vector from API (one per env)
        self.steps_since_last_inference = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Track steps since last API call per env
        self.post_reset_settle_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # API robustness state
        self.policy_client = None
        if self.cfg.use_api_for_pose:
            if PolicyClient is None:
                raise ImportError("gr00t not found. Add Isaac-GR00T to PYTHONPATH or install dependencies in this env.")
            self.policy_client = PolicyClient(host=self.cfg.gr00t_host, port=self.cfg.gr00t_port, strict=False)
            if not self.policy_client.ping():
                raise RuntimeError(f"GR00T server ping failed at {self.cfg.gr00t_host}:{self.cfg.gr00t_port}")
            print(f"[INFO] GR00T client connected to {self.cfg.gr00t_host}:{self.cfg.gr00t_port}")

            try:
                mod_cfg = self.policy_client.get_modality_config()
                if "action" in mod_cfg and hasattr(mod_cfg["action"], "delta_indices"):
                    action_horizon = len(mod_cfg["action"].delta_indices)
                    self.cfg.api_call_interval = max(1, int(action_horizon) * int(self.steps_per_action))
                    print(
                        f"[INFO] GR00T action_horizon={action_horizon}, "
                        f"api_call_interval={self.cfg.api_call_interval} "
                        f"({self.steps_per_action} steps/token)"
                    )
            except Exception as e:
                print(f"[WARN] Failed to read GR00T modality config: {e}")

        self._api_rr_cursor = 0  # round-robin cursor for which envs we call next
        self.api_consecutive_failures = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.api_cooldown_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.api_disabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Thread pool for async API calls (max 4 workers to avoid overwhelming API)
        self._api_executor = ThreadPoolExecutor(max_workers=min(4, self.cfg.api_max_envs_per_step))
        self._api_lock = Lock()  # Lock for thread-safe updates to shared state
        
        # Ensemble support: store recent API responses when inference_interval < num_windows
        self.api_response_buffer = [[] for _ in range(self.num_envs)]  # Buffer to store recent API responses for ensembling per env
        self.max_buffer_size = 3  # Maximum number of API responses to store
        
        # Create per-environment directories for camera images if saving is enabled
        base_dir = os.path.abspath(self.cfg.camera_save_dir)
        self.api_camera_save_dirs = []  # List of directories, one per environment
        self.api_camera_save_counters = [0] * self.num_envs  # Counter per environment
        
        if self.cfg.save_camera_images:
            os.makedirs(base_dir, exist_ok=True)
            # Create per-environment subdirectories
            for env_id in range(self.num_envs):
                env_camera_dir = os.path.join(base_dir, f"env_{env_id:03d}")
                env_camera_dir = os.path.abspath(env_camera_dir)
                os.makedirs(env_camera_dir, exist_ok=True)
                self.api_camera_save_dirs.append(env_camera_dir)
            print(f"[INFO] Camera image saving enabled: images will be saved to per-environment directories in {base_dir}")
        else:
            # Still create directories but don't save (for compatibility)
            for env_id in range(self.num_envs):
                self.api_camera_save_dirs.append(None)
            print(f"[INFO] Camera image saving disabled (save_camera_images=False)")
        
        print(
            f"[DEBUG] use_api_for_pose={self.cfg.use_api_for_pose}, _camera_enabled={self._camera_enabled}, "
            f"front={self._camera_front is not None}, back={self._camera_back is not None}, wrist={self._camera_wrist is not None}"
        )

    def _emit_compare_debug(self, env_id: int, current_action: np.ndarray | None = None, adjusted_pos_w: np.ndarray | None = None, adjusted_rpy: np.ndarray | None = None):
        if not self.cfg.debug_compare_mode:
            return
        if env_id < 0 or env_id >= self.num_envs:
            return

        if (self.api_step_counter % max(1, int(self.cfg.debug_compare_interval))) != 0:
            return

        now = time.time()
        dt_wall = now - self._debug_compare_last_ts
        self._debug_compare_last_ts = now

        has_windows = self.current_action_windows[env_id] is not None
        num_windows = int(self.num_windows[env_id]) if self.num_windows[env_id] is not None else -1
        current_window_idx = int(self.current_window_idx[env_id].item())
        steps_in_window = int(self.steps_in_current_window[env_id].item())
        steps_per_window = int(self.steps_per_window[env_id]) if self.steps_per_window[env_id] is not None else -1
        steps_since_infer = int(self.steps_since_last_inference[env_id].item())
        cooldown = int(self.api_cooldown_remaining[env_id].item()) if hasattr(self, "api_cooldown_remaining") else -1
        disabled = bool(self.api_disabled[env_id].item()) if hasattr(self, "api_disabled") else False

        effective_control_hz = 0.0
        if self.dt > 0 and steps_per_window > 0:
            effective_control_hz = 1.0 / (self.dt * float(steps_per_window))

        ee_pose = self._robot.data.body_state_w[env_id, self.robot_entity_cfg.body_ids[0], :7]
        ee_pos = ee_pose[0:3].detach().cpu().numpy()
        ee_quat = ee_pose[3:7].detach().cpu().numpy()
        ee_rpy = Rotation.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]).as_euler("xyz")

        raw_action = self.raw_actions[env_id].detach().cpu().numpy()
        norm_action = self.actions[env_id].detach().cpu().numpy()

        msg = {
            "step": int(self.api_step_counter),
            "env_id": int(env_id),
            "sim_dt_effective": float(self.dt),
            "api_call_interval": int(self.cfg.api_call_interval),
            "has_windows": bool(has_windows),
            "num_windows": int(num_windows),
            "window_idx": int(current_window_idx),
            "steps_in_window": int(steps_in_window),
            "steps_per_window": int(steps_per_window),
            "steps_since_infer": int(steps_since_infer),
            "api_cooldown": int(cooldown),
            "api_disabled": bool(disabled),
            "effective_control_hz": float(effective_control_hz),
            "ee_pos_w": ee_pos.tolist(),
            "ee_rpy": ee_rpy.tolist(),
            "raw_action": raw_action.tolist(),
            "norm_action": norm_action.tolist(),
            "current_action": current_action.tolist() if current_action is not None else None,
            "adjusted_pos_w": adjusted_pos_w.tolist() if adjusted_pos_w is not None else None,
            "adjusted_rpy": adjusted_rpy.tolist() if adjusted_rpy is not None else None,
            "wall_dt": float(dt_wall),
        }

        if not self._debug_compare_header_printed:
            self._debug_compare_header_printed = True
            print(
                "[GR00T-DEBUG] Compare mode enabled: "
                f"sim_dt={self.dt:.6f}s, api_interval={self.cfg.api_call_interval}, "
                f"log_interval={self.cfg.debug_compare_interval} steps"
            )

        print(
            "[GR00T-DEBUG] "
            f"step={msg['step']} env={env_id} has_windows={msg['has_windows']} "
            f"idx={msg['window_idx']}/{msg['num_windows']} spw={msg['steps_per_window']} "
            f"since_infer={msg['steps_since_infer']} ctrl_hz={msg['effective_control_hz']:.2f}"
        )

        if self._debug_compare_file_path is not None:
            try:
                with open(self._debug_compare_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            except Exception as exc:
                print(f"[GR00T-DEBUG] failed to write JSONL: {exc}")

    def _make_api_call(self, env_id, wrist_np, left_np, right_np, joint_pos_np, gripper_frac):
        """Make GR00T policy call for a single environment (runs in thread pool)."""
        try:
            video_dict = {
                "wrist_view": wrist_np[None, None, ...].astype(np.uint8),
                "ego_view": wrist_np[None, None, ...].astype(np.uint8),
                "left_view": left_np[None, None, ...].astype(np.uint8),
                "right_view": right_np[None, None, ...].astype(np.uint8),
            }

            state_dict = {
                self.STATE_JOINT_KEY: joint_pos_np[None, None, :].astype(np.float32),
                self.STATE_GRIPPER_KEY: np.array([[[float(gripper_frac)]]], dtype=np.float32),
            }
            language_dict = {self.LANGUAGE_KEY: [[str(self.cfg.task_description)]]}
            obs = {"video": video_dict, "state": state_dict, "language": language_dict}

            result = self.policy_client.get_action(obs)
            return (env_id, True, result, None)
        except Exception as e:
            return (env_id, False, None, str(e))

    def _process_api_response(self, env_id, result):
        """Process API response and update state (thread-safe)."""
        with self._api_lock:
            pred_action = result[0] if isinstance(result, (tuple, list)) and len(result) == 2 else result
            action_windows = None
            if isinstance(pred_action, dict):
                if "action" in pred_action:
                    arr = np.array(pred_action["action"], dtype=np.float32)
                    action_windows = arr[0] if arr.ndim == 3 else arr
                elif "action.ee_delta" in pred_action and "action.gripper_pos" in pred_action:
                    ee = np.array(pred_action["action.ee_delta"], dtype=np.float32)
                    gr = np.array(pred_action["action.gripper_pos"], dtype=np.float32)
                    ee = ee[0] if ee.ndim == 3 else ee
                    gr = gr[0] if gr.ndim == 3 else gr
                    action_windows = np.concatenate([ee, gr], axis=-1)
            else:
                arr = np.array(pred_action, dtype=np.float32)
                action_windows = arr[0] if arr.ndim == 3 else arr

            if action_windows is not None:
                if isinstance(action_windows, (list, np.ndarray)):
                    action_array = np.array(action_windows, dtype=np.float32)
                    if len(action_array.shape) == 2 and action_array.shape[1] == 7:
                        detected_num_windows = action_array.shape[0]
                        
                        # Validate for NaN/Inf
                        if np.any(~np.isfinite(action_array)):
                            # NOTE: _process_api_response is called from the main thread (after future.result()).
                            # Direct tensor indexing here is safe and much faster than CPU<->GPU copies.
                            self.api_consecutive_failures[env_id] += 1
                            self.api_cooldown_remaining[env_id] = int(self.cfg.api_fail_cooldown_steps)
                            return
                        
                        # Ensembling
                        self.api_response_buffer[env_id].append(action_array.copy())
                        if len(self.api_response_buffer[env_id]) > self.max_buffer_size:
                            self.api_response_buffer[env_id].pop(0)
                        
                        if self.cfg.api_call_interval < detected_num_windows and len(self.api_response_buffer[env_id]) >= 2:
                            buffer = self.api_response_buffer[env_id]
                            weights = np.linspace(0.3, 1.0, len(buffer))
                            weights = weights / weights.sum()
                            ensembled_windows = np.zeros_like(action_array)
                            all_windows = np.stack(buffer, axis=0)
                            ensembled_windows[:, [0,1,2,6]] = np.average(all_windows[:, :, [0,1,2,6]], axis=0, weights=weights)
                            
                            for i in range(detected_num_windows):
                                rpy_list = all_windows[:, i, 3:6]
                                quat_list = np.array([self._euler_to_quaternion(rpy) for rpy in rpy_list])
                                weighted_quat = np.average(quat_list, axis=0, weights=weights)
                                weighted_quat = weighted_quat / np.linalg.norm(weighted_quat)
                                interp_rot = Rotation.from_quat([weighted_quat[1], weighted_quat[2], weighted_quat[3], weighted_quat[0]])
                                ensembled_windows[i, 3:6] = interp_rot.as_euler('xyz')
                            
                            if np.any(~np.isfinite(ensembled_windows)):
                                action_array = self.api_response_buffer[env_id][-1]
                            else:
                                action_array = ensembled_windows
                        elif self.cfg.api_call_interval >= detected_num_windows:
                            self.api_response_buffer[env_id].clear()
                        
                        self.current_action_windows[env_id] = action_array
                        self.num_windows[env_id] = detected_num_windows

                        desired_interval = max(1, int(detected_num_windows) * int(self.steps_per_action))
                        if int(self.cfg.api_call_interval) != desired_interval:
                            self.cfg.api_call_interval = desired_interval
                            if env_id == 0:
                                print(
                                    f"[INFO] Adjusted api_call_interval to {self.cfg.api_call_interval} "
                                    f"(num_windows={detected_num_windows}, steps_per_action={self.steps_per_action})"
                                )

                        # Reset window tracking for this env
                        self.current_window_idx[env_id] = 0
                        self.steps_in_current_window[env_id] = 0
                        self.steps_per_window[env_id] = max(1, int(self.steps_per_action))

                        # Mark inference as fresh / clear failures
                        self.steps_since_last_inference[env_id] = 0
                        self.api_consecutive_failures[env_id] = 0
                    else:
                        if env_id == 0:
                            print(f"[WARNING] Action windows shape mismatch for env {env_id}: expected (N, 7), got {action_array.shape}")

    def _setup_cameras(self):
        """Set up camera positions for each environment (relative to each environment's origin)."""
        if not self._camera_enabled or self._camera_front is None or self._camera_back is None:
            return
            
        try:
            camera_front_local = torch.tensor([0.4, -0.7, 0.8], device=self.device)
            camera_back_local = torch.tensor([0.4, 0.7, 0.8], device=self.device)
            target_pos_local = torch.tensor([0.25, 0.0, -0.05], device=self.device)  # Local target
            
            # Convert to world coordinates by adding environment origins
            env_origins = self.scene.env_origins
            camera_front_world = camera_front_local.unsqueeze(0).repeat(self.num_envs, 1) + env_origins
            camera_back_world = camera_back_local.unsqueeze(0).repeat(self.num_envs, 1) + env_origins
            target_pos_world = target_pos_local.unsqueeze(0).repeat(self.num_envs, 1) + env_origins

            self._camera_front.set_world_poses_from_view(camera_front_world, target_pos_world)
            self._camera_back.set_world_poses_from_view(camera_back_world, target_pos_world)
            if not self._camera_setup_log_printed:
                if self.cfg.use_api_for_pose:
                    print(f"Set up {self.num_envs} cameras for GR00T-based control")
                else:
                    print(f"Set up {self.num_envs} cameras with GR00T view positions")
                self._camera_setup_log_printed = True
        except Exception as e:
            print(f"Warning: Could not set camera poses: {e}")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions: normalize, call API, apply RL deltas, compute IK."""
        self.raw_actions = actions.clone()
        
        # Normalize actions
        scaled_actions = actions / self.cfg.action_tanh_scale
        self.actions = torch.tanh(scaled_actions)
        
        # Add noise to actions for exploration (only during training)
        if hasattr(self, 'is_training') and self.is_training and (not self.cfg.reference_match_mode):
            # Generate random noise on scaled actions
            noise = torch.randn_like(scaled_actions) * self.cfg.exploration_noise
            
            # Apply noise and re-normalize with tanh to keep within range [-1, 1]
            self.actions = torch.tanh(scaled_actions + noise)
            
            # Add extra exploration noise specifically for gripper action (action[6])
            gripper_noise = torch.randn_like(self.actions[:, 6:7]) * self.cfg.gripper_exploration_noise
            # Add noise to gripper action (in normalized space [-1, 1])
            self.actions[:, 6:7] = torch.clamp(
                self.actions[:, 6:7] + gripper_noise, 
                -1.0, 
                1.0
            )

        # API-based pose control (if enabled)
        if self.cfg.use_api_for_pose:
            # Debug: check why API control might not be active
            if not self._camera_enabled or self._camera_front is None or self._camera_back is None or self._camera_wrist is None:
                if self.api_step_counter == 0:
                    print(
                        f"[WARNING] API control requested but camera not ready: use_api_for_pose={self.cfg.use_api_for_pose}, "
                        f"_camera_enabled={self._camera_enabled}"
                    )
                    print(f"[WARNING] Make sure to run with --enable_cameras flag")
        
        if self.cfg.use_api_for_pose and self._camera_enabled and self._camera_front is not None and self._camera_back is not None and self._camera_wrist is not None:
            # Update camera (position is already set in _setup_cameras)
            try:
                for cam in (self._camera_front, self._camera_back, self._camera_wrist):
                    if not hasattr(cam, 'is_initialized') or not cam.is_initialized:
                        if hasattr(cam, '_initialize_impl'):
                            try:
                                cam._initialize_impl()
                            except Exception:
                                pass
                    if hasattr(cam, 'update'):
                        cam.update(dt=self.dt)

                # Keep front/back camera gaze fixed like GR00T eval script
                self._setup_cameras()
                
                # Call API for each environment individually based on interval
                if (
                    hasattr(self._camera_front, 'data') and hasattr(self._camera_front.data, 'output')
                    and hasattr(self._camera_back, 'data') and hasattr(self._camera_back.data, 'output')
                    and hasattr(self._camera_wrist, 'data') and hasattr(self._camera_wrist.data, 'output')
                ):
                    camera_front_data = self._camera_front.data.output
                    camera_back_data = self._camera_back.data.output
                    camera_wrist_data = self._camera_wrist.data.output
                    if "rgb" in camera_front_data and "rgb" in camera_back_data and "rgb" in camera_wrist_data:
                        if self.api_step_counter == 0:
                            print(
                                f"[DEBUG] Camera data available, front={camera_front_data['rgb'].shape}, "
                                f"back={camera_back_data['rgb'].shape}, wrist={camera_wrist_data['rgb'].shape}"
                            )
                        rgb_front = camera_front_data["rgb"]
                        rgb_back = camera_back_data["rgb"]
                        rgb_wrist = camera_wrist_data["rgb"]

                        # Save right_view frames (camera_front) when enabled
                        if self.cfg.save_camera_images:
                            for env_id in range(self.num_envs):
                                if self.api_camera_save_dirs[env_id] is not None:
                                    try:
                                        rgb_np = rgb_front[env_id].detach().cpu().numpy()
                                        if rgb_np.dtype != np.uint8:
                                            if rgb_np.max() <= 1.0:
                                                rgb_np = (np.clip(rgb_np, 0.0, 1.0) * 255).astype(np.uint8)
                                            else:
                                                rgb_np = np.clip(rgb_np, 0.0, 255.0).astype(np.uint8)
                                        if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
                                            rgb_np = rgb_np[..., :3]
                                        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                                        save_image_path = os.path.join(
                                            self.api_camera_save_dirs[env_id],
                                            f"frame_{self.api_camera_save_counters[env_id]:05d}.png",
                                        )
                                        if cv2.imwrite(save_image_path, bgr_np):
                                            self.api_camera_save_counters[env_id] += 1
                                    except Exception:
                                        pass
                        
                        # Vectorized check for environments that need API calls
                        # Use tensor operations instead of Python loop
                        settle_ready = self.post_reset_settle_remaining == 0
                        steps_ready = self.steps_since_last_inference >= self.cfg.api_call_interval
                        cooldown_ready = self.api_cooldown_remaining == 0
                        not_disabled = ~self.api_disabled if hasattr(self, 'api_disabled') else torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                        envs_ready_mask = settle_ready & steps_ready & cooldown_ready & not_disabled
                        envs_to_call = torch.where(envs_ready_mask)[0].cpu().numpy()[:self.cfg.api_max_envs_per_step].tolist()
                        
                        # Submit async API calls
                        futures = []
                        for env_id in envs_to_call:
                            wrist_np = rgb_wrist[env_id].cpu().numpy()
                            left_np = rgb_back[env_id].cpu().numpy()
                            right_np = rgb_front[env_id].cpu().numpy()
                            joint_pos_np = self._robot.data.joint_pos[env_id, self.arm_joint_ids].detach().cpu().numpy().astype(np.float32)
                            hand_joint_ids = [idx for idx, name in enumerate(self._robot.data.joint_names) if "panda_finger_joint" in name]
                            gripper_joint = float(self._robot.data.joint_pos[env_id, hand_joint_ids[0]].detach().cpu().item()) if hand_joint_ids else 0.0
                            gripper_frac = float(np.clip(gripper_joint / 0.04, 0.0, 1.0))
                            future = self._api_executor.submit(
                                self._make_api_call,
                                env_id,
                                wrist_np,
                                left_np,
                                right_np,
                                joint_pos_np,
                                gripper_frac,
                            )
                            futures.append(future)
                        
                        # Process completed API calls (non-blocking, process as they complete)
                        for future in as_completed(futures):
                            env_id, success, result, error = future.result()
                            if success and result is not None:
                                self._process_api_response(env_id, result)
                            else:
                                # Handle failure (thread-safe tensor updates)
                                with self._api_lock:
                                    # Thread-safe tensor updates: update on CPU first, then sync to GPU
                                    failures_cpu = self.api_consecutive_failures.cpu().clone()
                                    failures_cpu[env_id] += 1
                                    self.api_consecutive_failures.copy_(failures_cpu.to(self.device))
                                    
                                    cooldown_cpu = self.api_cooldown_remaining.cpu().clone()
                                    cooldown_cpu[env_id] = int(self.cfg.api_fail_cooldown_steps)
                                    self.api_cooldown_remaining.copy_(cooldown_cpu.to(self.device))
                                    
                                    if self.cfg.api_disable_after_consecutive_failures > 0:
                                        failures_val = failures_cpu[env_id].item()
                                        if failures_val >= int(self.cfg.api_disable_after_consecutive_failures):
                                            disabled_cpu = self.api_disabled.cpu().clone()
                                            disabled_cpu[env_id] = True
                                            self.api_disabled.copy_(disabled_cpu.to(self.device))
                                    
                                    if env_id == 0 and error:
                                        print(f"[WARNING] API call failed for env {env_id}: {error}")
                    else:
                        if self.api_step_counter == 0:
                            print(
                                "[WARNING] Missing RGB in one or more camera outputs: "
                                f"front={list(camera_front_data.keys())}, back={list(camera_back_data.keys())}, wrist={list(camera_wrist_data.keys())}"
                            )
                else:
                    if self.api_step_counter == 0:
                        print("[WARNING] Camera data not available for one or more GR00T cameras")
                
                # Decrement cooldown/settle counters and check for API calls
                self.api_cooldown_remaining = torch.clamp(self.api_cooldown_remaining - 1, min=0)
                self.post_reset_settle_remaining = torch.clamp(self.post_reset_settle_remaining - 1, min=0)
                
                # Increment steps for all environments (vectorized) - happens every step
                self.steps_since_last_inference += 1
            except Exception as e:
                print(f"[WARNING] API call error: {e}")
                import traceback
                traceback.print_exc()
        
        # Periodically re-enable API for environments that were disabled (recovery mechanism)
        # Re-enable after 5x the cooldown period to give API time to recover
        if self.api_step_counter % 100 == 0:  # Check every 100 steps
            reenable_cooldown = self.cfg.api_fail_cooldown_steps * 5
            for env_id in range(self.num_envs):
                if hasattr(self, 'api_disabled') and self.api_disabled[env_id]:
                    # Check if enough time has passed since disable
                    if self.steps_since_last_inference[env_id] >= reenable_cooldown:
                        self.api_disabled[env_id] = False
                        self.api_consecutive_failures[env_id] = 0  # Reset failure count
                        self.api_cooldown_remaining[env_id] = 0
                        if env_id == 0:
                            print(f"[INFO] Re-enabling API for env {env_id} after cooldown period")
        
        self.api_step_counter += 1
        
        # Use action windows from API if available (per-environment)
        if self.cfg.use_api_for_pose and self.diff_ik_controller is not None:
            # Vectorized check for valid environments with action windows
            has_windows = torch.tensor([w is not None for w in self.current_action_windows], device=self.device, dtype=torch.bool)
            has_num_windows = torch.tensor([n is not None for n in self.num_windows], device=self.device, dtype=torch.bool)
            valid_mask = has_windows & has_num_windows
            
            # Process valid environments (cache env_origins to avoid repeated CPU transfer)
            target_positions = []
            target_quaternions = []
            valid_env_ids = []
            env_origins_cached = self.scene.env_origins.cpu().numpy()
            actions_cached = self.actions.cpu().numpy()  # Cache actions to reduce CPU-GPU transfers
            
            # Process only valid environments
            valid_env_list = torch.where(valid_mask)[0].cpu().numpy().tolist()
            for env_id in valid_env_list:
                if self.current_action_windows[env_id] is not None and self.num_windows[env_id] is not None:
                    window_idx = self.current_window_idx[env_id].item()
                    if window_idx < self.num_windows[env_id]:
                        current_action = self.current_action_windows[env_id][window_idx]
                        
                        # Validate current action for NaN/Inf - if invalid, skip this env
                        if np.any(~np.isfinite(current_action)):
                            if not hasattr(self, '_action_window_nan_warning_count'):
                                self._action_window_nan_warning_count = 0
                            self._action_window_nan_warning_count += 1
                            if self._action_window_nan_warning_count <= 5:
                                print(f"[WARNING] Action window {window_idx} for env {env_id} contains NaN/Inf. Skipping this environment.")
                            # Clear invalid action windows to trigger fallback
                            self.current_action_windows[env_id] = None
                            self.num_windows[env_id] = None
                            continue
                        
                        interpolated_action = current_action
                        
                        #  Validate interpolated action for NaN/Inf
                        if np.any(~np.isfinite(interpolated_action)):
                            if not hasattr(self, '_interpolated_nan_warning_count'):
                                self._interpolated_nan_warning_count = 0
                            self._interpolated_nan_warning_count += 1
                            if self._interpolated_nan_warning_count <= 5:
                                print(f"[WARNING] Interpolated action for env {env_id} contains NaN/Inf. Skipping this environment.")
                            # Clear invalid action windows to trigger fallback
                            self.current_action_windows[env_id] = None
                            self.num_windows[env_id] = None
                            continue
                        
                        # Script-equivalent semantics:
                        # action[0:3] = EE delta position, action[3:6] = EE delta rotation (rotvec)
                        # Apply relative to current EE pose in world frame.
                        api_dp = interpolated_action[:3].astype(np.float32)
                        api_drot = interpolated_action[3:6].astype(np.float32)
                        if self.cfg.reference_match_mode:
                            rl_dp = np.zeros(3, dtype=np.float32)
                            rl_drot = np.zeros(3, dtype=np.float32)
                        else:
                            rl_dp = actions_cached[env_id, 0:3] * self.cfg.delta_pos_scale
                            rl_drot = actions_cached[env_id, 3:6] * self.cfg.delta_rpy_scale
                        total_dp = api_dp + rl_dp
                        total_drot = api_drot + rl_drot

                        drot_norm = float(np.linalg.norm(total_drot))
                        if drot_norm > float(self.cfg.rot_clamp_rad) and drot_norm > 1e-6:
                            total_drot = total_drot * (float(self.cfg.rot_clamp_rad) / drot_norm)

                        ee_pose_w_single = self._robot.data.body_state_w[env_id, self.robot_entity_cfg.body_ids[0], :7]
                        ee_pos_w_single = ee_pose_w_single[0:3].detach().cpu().numpy().astype(np.float32)
                        ee_quat_w_single = ee_pose_w_single[3:7].detach().cpu().numpy().astype(np.float32)

                        adjusted_pos_w = ee_pos_w_single + total_dp

                        dq_xyzw = Rotation.from_rotvec(total_drot).as_quat()
                        dq_wxyz = np.array([dq_xyzw[3], dq_xyzw[0], dq_xyzw[1], dq_xyzw[2]], dtype=np.float32)

                        w1, x1, y1, z1 = dq_wxyz
                        w2, x2, y2, z2 = ee_quat_w_single
                        adjusted_quat_w = np.array([
                            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                        ], dtype=np.float32)
                        adjusted_quat_w = adjusted_quat_w / max(1e-6, float(np.linalg.norm(adjusted_quat_w)))
                        adjusted_rpy = Rotation.from_quat(
                            [adjusted_quat_w[1], adjusted_quat_w[2], adjusted_quat_w[3], adjusted_quat_w[0]]
                        ).as_euler('xyz').astype(np.float32)

                        # Validate adjusted pose before IK
                        if np.any(~np.isfinite(adjusted_pos_w)) or np.any(~np.isfinite(adjusted_quat_w)):
                            if not hasattr(self, '_adjusted_pose_nan_warning_count'):
                                self._adjusted_pose_nan_warning_count = 0
                            self._adjusted_pose_nan_warning_count += 1
                            if self._adjusted_pose_nan_warning_count <= 5:
                                print(f"[WARNING] Adjusted pose for env {env_id} contains NaN/Inf. Skipping this environment.")
                            # Clear invalid action windows to trigger fallback
                            self.current_action_windows[env_id] = None
                            self.num_windows[env_id] = None
                            continue
                        
                        target_positions.append(adjusted_pos_w)
                        target_quaternions.append(adjusted_quat_w)
                        valid_env_ids.append(env_id)
                        self._api_gripper_action[env_id] = float(np.clip(interpolated_action[6], 0.0, 1.0))

                        if env_id == 0:
                            self._emit_compare_debug(
                                env_id=env_id,
                                current_action=interpolated_action,
                                adjusted_pos_w=adjusted_pos_w,
                                adjusted_rpy=adjusted_rpy,
                            )
            
            # Track environments without action windows for diagnostics and fallback (vectorized)
            # IMPORTANT: Only use fallback for environments that have actually had a chance to get API data
            no_windows_mask = ~valid_mask
            disabled_mask = self.api_disabled if hasattr(self, 'api_disabled') else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            waited_long_mask = self.steps_since_last_inference >= (self.cfg.api_call_interval * 3)
            no_cooldown_mask = self.api_cooldown_remaining == 0
            
            # Do not move without fresh GR00T windows; only keep diagnostic list for disabled envs.
            fallback_mask = disabled_mask
            envs_without_windows = torch.where(fallback_mask)[0].cpu().numpy().tolist()
            
            # Log warning if many environments don't have action windows
            if len(envs_without_windows) > 0 and self.api_step_counter % 100 == 0:
                if not hasattr(self, '_no_windows_warning_count'):
                    self._no_windows_warning_count = 0
                self._no_windows_warning_count += 1
                if self._no_windows_warning_count <= 10:
                    print(f"[WARNING] {len(envs_without_windows)}/{self.num_envs} environments have no action windows. "
                          f"Using fallback: current pose + RL delta. "
                          f"Sample envs: {envs_without_windows[:min(5, len(envs_without_windows))]}")
            
            # If no windows are available, do not apply fallback motion.
            # Keep arm at current pose until fresh inference arrives.
            if len(envs_without_windows) > 0:
                for env_id in envs_without_windows:
                    pass
            
            # Apply IK for all valid environments at once
            if len(valid_env_ids) > 0:
                valid_env_tensor = torch.tensor(valid_env_ids, device=self.device, dtype=torch.long)
                target_pos_w = torch.tensor(np.array(target_positions), device=self.device, dtype=torch.float32)
                target_quat_w = torch.tensor(np.array(target_quaternions), device=self.device, dtype=torch.float32)
                target_pose_w = torch.cat([target_pos_w, target_quat_w], dim=-1)
                
                # Create full-size command tensor (IK controller expects all envs)
                full_command = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
                full_command[valid_env_tensor] = target_pose_w
                self.diff_ik_controller.set_command(full_command)
                
                # Compute IK with error handling for singular jacobian cases
                ee_pose_w = self._robot.data.body_state_w[valid_env_tensor, self.robot_entity_cfg.body_ids[0], :7]
                ee_pos_w = ee_pose_w[:, 0:3]
                ee_quat_w = ee_pose_w[:, 3:7]
                
                ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
                jacobian = self._robot.root_physx_view.get_jacobians()[valid_env_tensor, ee_jacobi_idx, :, :]
                jacobian = jacobian[:, :, self.arm_joint_ids]
                joint_pos = self._robot.data.joint_pos[valid_env_tensor][:, self.arm_joint_ids]
                
                # Extract desired poses for valid environments only (to match shapes with current poses)
                # This prevents shape mismatch errors when calling compute()
                ee_pos_des_valid = self.diff_ik_controller.ee_pos_des[valid_env_tensor]
                ee_quat_des_valid = self.diff_ik_controller.ee_quat_des[valid_env_tensor]
                
                try:
                    # Use the IK controller's internal method to compute pose error and delta joint positions
                    from isaaclab.utils.math import compute_pose_error
                    position_error, axis_angle_error = compute_pose_error(
                        ee_pos_w, ee_quat_w, ee_pos_des_valid, ee_quat_des_valid, rot_error_type="axis_angle"
                    )
                    pose_error = torch.cat((position_error, axis_angle_error), dim=1)
                    
                    # Compute delta joint positions using the IK controller's method
                    delta_joint_pos = self.diff_ik_controller._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)
                    joint_pos_arm = joint_pos + delta_joint_pos
                    
                    # Immediate validation after IK computation
                    if not torch.isfinite(joint_pos_arm).all():
                        # IK produced invalid values, use current positions
                        invalid_count = (~torch.isfinite(joint_pos_arm)).sum().item()
                        if not hasattr(self, '_ik_invalid_output_count'):
                            self._ik_invalid_output_count = 0
                        self._ik_invalid_output_count += 1
                        if self._ik_invalid_output_count <= 5:
                            print(f"[WARNING] IK solver produced invalid (NaN/Inf) joint positions ({invalid_count} values). "
                                  f"Using current joint positions instead.")
                        # Replace invalid values with current positions
                        invalid_mask = ~torch.isfinite(joint_pos_arm).any(dim=-1)
                        joint_pos_arm[invalid_mask] = joint_pos[invalid_mask]
                        
                except (ValueError, RuntimeError, torch._C._LinAlgError) as e:
                    # Handle shape mismatch, singular jacobian, or other errors
                    if not hasattr(self, '_ik_error_count'):
                        self._ik_error_count = 0
                    self._ik_error_count += 1
                    if self._ik_error_count <= 5:
                        print(f"[WARNING] IK solver error: {e}. Using current joint positions instead.")
                    # Fallback to current joint positions on error
                    joint_pos_arm = joint_pos.clone()
                
                # Apply velocity limiting
                prev_joint_pos_arm = self._robot.data.joint_pos_target[valid_env_tensor][:, self.arm_joint_ids]
                joint_pos_arm_delta = joint_pos_arm - prev_joint_pos_arm
                max_joint_step = 10.0
                joint_pos_arm_delta_clipped = torch.clamp(joint_pos_arm_delta, -max_joint_step, max_joint_step)
                joint_pos_arm = prev_joint_pos_arm + joint_pos_arm_delta_clipped
                joint_pos_arm = torch.clamp(joint_pos_arm, self.robot_dof_lower_limits[self.arm_joint_ids], self.robot_dof_upper_limits[self.arm_joint_ids])
                
                # Set joint targets
                targets_subset = self.robot_dof_targets[valid_env_tensor].clone()
                targets_subset[:, self.arm_joint_ids] = joint_pos_arm
                self.robot_dof_targets[valid_env_tensor] = targets_subset
                
                # Advance action windows for valid environments (vectorized where possible)
                if len(valid_env_ids) > 0:
                    valid_env_tensor_for_advance = torch.tensor(valid_env_ids, device=self.device, dtype=torch.long)
                    self.steps_in_current_window[valid_env_tensor_for_advance] += 1
                    
                    # Vectorized check for window advancement
                    for env_id in valid_env_ids:
                        env_id_item = env_id.item() if isinstance(env_id, torch.Tensor) else env_id
                        if self.num_windows[env_id_item] is not None and \
                           self.steps_in_current_window[env_id_item] >= self.steps_per_window[env_id_item]:
                            self.steps_in_current_window[env_id_item] = 0
                            if self.current_window_idx[env_id_item] < self.num_windows[env_id_item] - 1:
                                self.current_window_idx[env_id_item] += 1
                            else:
                                # Force API call if stuck on last window
                                if self.steps_since_last_inference[env_id_item] >= self.cfg.api_call_interval * 2:
                                    self.steps_since_last_inference[env_id_item] = self.cfg.api_call_interval
                                    if env_id_item == 0 and self.api_step_counter % 50 == 0:
                                        print(f"[INFO] Forcing API call for env {env_id_item} - stuck on last window for too long")
            else:
                self.robot_dof_targets[:, self.arm_joint_ids] = self._robot.data.joint_pos[:, self.arm_joint_ids]
            
            # Gripper control: RL direct (actions[6])
            env_ids_all = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if self.cfg.reference_match_mode:
                target_joint_pos = self._api_gripper_action[env_ids_all].unsqueeze(-1) * 0.04
            else:
                gripper_action = self.actions[env_ids_all, 6:7]
                target_width = 0.5 * (gripper_action + 1.0) * (self.cfg.max_gripper_width - self.cfg.min_gripper_width) + self.cfg.min_gripper_width
                target_joint_pos = target_width / 2.0
            num_finger_joints = self._robot.num_joints - 7  # Number of gripper joints (typically 2)
            target_joint_pos_expanded = target_joint_pos.expand(-1, num_finger_joints)
            # Gripper joints are at indices 7 onwards (after the 7 arm joints)
            self.robot_dof_targets[env_ids_all, 7:] = target_joint_pos_expanded

            hold_joint_mask = (self.post_reset_settle_remaining > 0) | (~valid_mask)
            if torch.any(hold_joint_mask):
                hold_env_ids = torch.where(hold_joint_mask)[0]
                self.robot_dof_targets[hold_env_ids] = self.reset_joint_pos_targets[hold_env_ids]
            
            return  # Skip CSV logic when using API
        
        # Save camera images periodically (even when API is disabled, for debugging)
        if self._camera_enabled and self._camera_front is not None and self.cfg.save_camera_images:
            # Save images every 30 steps (about once per second at 30Hz) when API is disabled
            if not self.cfg.use_api_for_pose and self.api_step_counter % 30 == 0:
                try:
                    # Update camera (position is already set in _setup_cameras)
                    if hasattr(self._camera_front, 'update'):
                        self._camera_front.update(dt=self.dt)
                    
                    # Capture and save images for all environments
                    if hasattr(self._camera_front, 'data') and hasattr(self._camera_front.data, 'output'):
                        camera_data = self._camera_front.data.output
                        if "rgb" in camera_data:
                            rgb_image = camera_data["rgb"]  # Shape: (num_envs, H, W, 3)
                            
                            # Save images for each environment
                            for env_id in range(self.num_envs):
                                if self.api_camera_save_dirs[env_id] is not None:
                                    rgb_np = rgb_image[env_id].cpu().numpy()
                                    
                                    # Convert to uint8 if needed
                                    if rgb_np.dtype != np.uint8:
                                        if rgb_np.max() <= 1.0:
                                            rgb_np = (rgb_np * 255).astype(np.uint8)
                                    
                                    # Convert RGB to BGR for OpenCV
                                    bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                                    
                                    # Save image to per-environment directory
                                    save_image_path = os.path.join(
                                        self.api_camera_save_dirs[env_id], 
                                        f"frame_{self.api_camera_save_counters[env_id]:05d}.png"
                                    )
                                    success = cv2.imwrite(save_image_path, bgr_np)
                                    if success:
                                        self.api_camera_save_counters[env_id] += 1
                                        if self.api_camera_save_counters[env_id] % 10 == 0 or self.api_camera_save_counters[env_id] == 1:
                                            print(f"[INFO] Saved camera image for env {env_id} (count: {self.api_camera_save_counters[env_id]}) to {save_image_path}")
                except Exception as e:
                    if self.api_step_counter == 0:
                        print(f"[WARNING] Failed to save camera images: {e}")
            
            # Increment step counter for image saving when API is disabled
            if not self.cfg.use_api_for_pose:
                self.api_step_counter += 1
        
        # When CSV data is loaded, control end-effector using absolute poses from CSV
        if len(self.csv_data_list) > 0 and self.diff_ik_controller is not None:
            has_absolute_poses = any(csv_data.get('absolute_poses') is not None for csv_data in self.csv_data_list)
            
            if has_absolute_poses:
                # Playback rate: 20 Hz means each CSV row should be held for multiple steps
                playback_rate_hz = 20.0
                steps_per_csv_row = int((1.0 / playback_rate_hz) / self.dt)
                env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                
                # Advance CSV row indices at playback rate
                for env_id in env_ids:
                    csv_idx = env_id.item() % len(self.csv_data_list)
                    csv_data = self.csv_data_list[csv_idx]
                    if csv_data.get('absolute_poses') is not None:
                        self.csv_step_counters[env_id] += 1
                        if self.csv_step_counters[env_id] >= steps_per_csv_row:
                            self.csv_step_counters[env_id] = 0
                            max_rows = len(csv_data['absolute_poses'])
                            if self.csv_row_indices[env_id] < max_rows - 1:
                                self.csv_row_indices[env_id] += 1
                            # Note: If we've reached max_rows - 1, we stay at the last row
                            # The episode will terminate in _get_dones() when csv_row_indices >= max_rows - 1
                
                # Prepare target poses: CSV poses are in local coordinates, convert to world
                # Then apply RL delta adjustments from actions
                env_origins = self.scene.env_origins[env_ids]
                target_positions = []
                target_quaternions = []
                
                # Extract RL delta position adjustments for all environments (actions[0:3] are delta dx, dy, dz)
                # Delta actions are in range [-1, 1], scale by delta_pos_scale
                delta_pos_actions = self.actions[env_ids, 0:3] * self.cfg.delta_pos_scale  # Shape: (num_envs, 3)
                
                for i, env_id in enumerate(env_ids):
                    csv_idx = env_id.item() % len(self.csv_data_list)
                    csv_data = self.csv_data_list[csv_idx]
                    row_idx = min(self.csv_row_indices[env_id].item(), len(csv_data['absolute_poses']) - 1)
                    
                    pose_row = csv_data['absolute_poses'][row_idx]
                    pos_local = pose_row[:3]
                    rpy = pose_row[3:6]
                    
                    # Convert to world coordinates
                    pos_w = pos_local + env_origins[i].cpu().numpy()
                    # Apply RL delta orientation adjustment (actions[3:6] are d_roll, d_pitch, d_yaw)
                    delta_rpy_action = self.actions[env_id, 3:6].cpu().numpy()
                    delta_rpy_adjustment = delta_rpy_action * self.cfg.delta_rpy_scale
                    rpy_adjusted = rpy + delta_rpy_adjustment
                    quat_w = self._euler_to_quaternion(rpy_adjusted)
                    
                    # Apply RL delta position adjustment
                    delta_pos_adjustment = delta_pos_actions[i].cpu().numpy()
                    pos_w_adjusted = pos_w + delta_pos_adjustment
                    
                    target_positions.append(pos_w_adjusted)
                    target_quaternions.append(quat_w)
                
                # Set IK command with adjusted poses
                target_pos_w = torch.tensor(np.array(target_positions), device=self.device, dtype=torch.float32)
                target_quat_w = torch.tensor(np.array(target_quaternions), device=self.device, dtype=torch.float32)
                target_pose_w = torch.cat([target_pos_w, target_quat_w], dim=-1)
                self.diff_ik_controller.set_command(target_pose_w)
                
                # Compute IK
                ee_pose_w = self._robot.data.body_state_w[env_ids, self.robot_entity_cfg.body_ids[0], :7]
                ee_pos_w = ee_pose_w[:, 0:3]
                ee_quat_w = ee_pose_w[:, 3:7]
                
                ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
                jacobian = self._robot.root_physx_view.get_jacobians()[env_ids, ee_jacobi_idx, :, :]
                jacobian = jacobian[:, :, self.arm_joint_ids]
                joint_pos = self._robot.data.joint_pos[env_ids][:, self.arm_joint_ids]
                
                joint_pos_arm = self.diff_ik_controller.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos)
                
                # Apply velocity limiting and joint limits
                prev_joint_pos_arm = self._robot.data.joint_pos_target[env_ids][:, self.arm_joint_ids]
                joint_pos_arm_delta = torch.clamp(joint_pos_arm - prev_joint_pos_arm, -0.05, 0.05)
                joint_pos_arm = prev_joint_pos_arm + joint_pos_arm_delta
                joint_pos_arm = torch.clamp(joint_pos_arm, self.robot_dof_lower_limits[self.arm_joint_ids], self.robot_dof_upper_limits[self.arm_joint_ids])
                
                # Set joint targets
                targets_subset = self.robot_dof_targets[env_ids].clone()
                targets_subset[:, self.arm_joint_ids] = joint_pos_arm
                self.robot_dof_targets[env_ids] = targets_subset
            else:
                # No absolute poses available, keep arm joints at initial CSV positions
                if not hasattr(self, '_csv_no_poses_warning_printed'):
                    print("[WARNING] CSV data loaded but no absolute poses found - keeping arm at initial CSV positions")
                    self._csv_no_poses_warning_printed = True
                self.robot_dof_targets[:, :7] = self._robot.data.joint_pos[:, :7]
        else:
            # Process actions for all environments (fallback when CSV not available)
            if not hasattr(self, '_csv_fallback_warning_printed'):
                print(f"[WARNING] CSV control not active - using action-based control. "
                      f"csv_data_list={len(self.csv_data_list)}, "
                      f"diff_ik_controller={self.diff_ik_controller is not None}, "
                      f"robot_entity_cfg={self.robot_entity_cfg is not None}, "
                      f"arm_joint_ids={self.arm_joint_ids is not None}")
                self._csv_fallback_warning_printed = True
            ### Arm
            # Create controller
            diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
            diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
            diff_ik_controller.reset()
            diff_ik_controller.set_command(self.actions[:, 0:7])

            robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
            robot_entity_cfg.resolve(self.scene)
            ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
            jacobian = self._robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = self._robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = self._robot.data.root_state_w[:, 0:7]
            joint_pos = self._robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # Compute joint positions and apply limits
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            joint_pos_des = torch.clamp(joint_pos_des, self.robot_dof_lower_limits[:7], self.robot_dof_upper_limits[:7])
            self.robot_dof_targets[:, :7] = joint_pos_des

        ### Fingers (always controlled, regardless of CSV data)
        # Franka has 2 finger joints (indices 7 and 8), but we use only action[6] to control both symmetrically
        # Normalize gripper action from [-1, 1] to width range [min_gripper_width, max_gripper_width]
        num_finger_joints = self._robot.num_joints - 7
        gripper_action = self.actions[:, 6:7]  # Use only action[6] (index 6, the 7th element)
        
        # Map action from [-1, 1] to width range [min_gripper_width, max_gripper_width]
        # Linear mapping: action=-1 -> min_width, action=1 -> max_width
        target_width = 0.5 * (gripper_action + 1.0) * (self.cfg.max_gripper_width - self.cfg.min_gripper_width) + self.cfg.min_gripper_width
        
        # Convert width to joint position
        # For Franka: joint_pos=0.0 is closed, joint_pos=0.04 is fully open
        # Total width = 2 * joint_position, so joint_position = width / 2
        target_joint_pos = target_width / 2.0
        
        # Apply to all finger joints (they move symmetrically)
        self.robot_dof_targets[:, 7:] = target_joint_pos.repeat(1, num_finger_joints)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate horizontal distance from initial position (only x and y, not z)
        cube_horizontal_displacement = torch.norm(
            self.cube_pos[:, :2] - self.cube_initial_pos[:, :2], 
            p=2, 
            dim=-1
        )
        
        # Terminate if the cube moves more than 0.15m horizontally from initial position
        out_of_bounds = cube_horizontal_displacement > 0.15
        
        # Also terminate if the cube is lifted too high
        too_high = self.cube_pos[:, 2] - self.cfg.cube_half_size > 0.3
        
        # Terminate if cube is rolling on the ground
        # Check if cube is near ground level (low height)
        cube_height = self.cube_pos[:, 2] - self.cfg.cube_half_size
        
        # Check if cube has moved significantly from initial position (to avoid false positives from initial touch)
        cube_horizontal_displacement_from_start = torch.norm(
            self.cube_pos[:, :2] - self.cube_initial_pos[:, :2], 
            p=2, 
            dim=-1
        )
        has_moved_significantly = cube_horizontal_displacement_from_start > self.cfg.rolling_min_displacement
        
        # Check if cube has significant horizontal velocity (rolling)
        cube_horizontal_vel = torch.norm(self.cube_lin_vel[:, :2], p=2, dim=-1)
        has_horizontal_velocity = cube_horizontal_vel > self.cfg.rolling_lin_vel_threshold
        
        # Check if cube has significant angular velocity (spinning/rolling)
        cube_angular_vel_magnitude = torch.norm(self.cube_ang_vel, p=2, dim=-1)
        has_angular_velocity = cube_angular_vel_magnitude > self.cfg.rolling_ang_vel_threshold
        
        # Cube is rolling if:
        # 1. It's near ground level
        # 2. It has moved significantly from initial position (to avoid false positives from initial touch)
        # 3. AND (has significant horizontal velocity OR significant angular velocity)
        is_rolling = has_moved_significantly & (has_horizontal_velocity | has_angular_velocity)
        
        # Terminate if CSV playback has completed (for each environment)
        csv_playback_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(self.csv_data_list) > 0:
            for env_id in range(self.num_envs):
                csv_idx = env_id % len(self.csv_data_list)
                csv_data = self.csv_data_list[csv_idx]
                if csv_data.get('absolute_poses') is not None:
                    max_rows = len(csv_data['absolute_poses'])
                    # Check if we've reached the last row of CSV data
                    if self.csv_row_indices[env_id] >= max_rows - 1:
                        csv_playback_complete[env_id] = True
        
        terminated = out_of_bounds | too_high | is_rolling | csv_playback_complete
        if self.cfg.reference_match_mode:
            terminated = torch.zeros_like(terminated, dtype=torch.bool)

        horizon_limit = int(getattr(self.cfg, "action_horizon_reset_steps", 1000))
        if horizon_limit > 0:
            # Timeout either at env max length or every fixed horizon steps.
            # Using modulo ensures periodic reset even if max_episode_length differs from horizon_limit.
            time_limit_trunc = self.episode_length_buf >= self.max_episode_length
            periodic_trunc = (self.episode_length_buf > 0) & (self.episode_length_buf % horizon_limit == 0)
            truncated = time_limit_trunc | periodic_trunc
        else:
            truncated = self.episode_length_buf >= self.max_episode_length
        
        # Track which environments are done for episode tracking
        done_env_ids = torch.where(terminated | truncated)[0]
        
        # Log rolling terminations (for debugging/monitoring)
        if torch.any(is_rolling):
            rolling_env_ids = torch.where(is_rolling)[0]
            if len(rolling_env_ids) > 0 and not hasattr(self, '_rolling_log_counter'):
                self._rolling_log_counter = 0
            if hasattr(self, '_rolling_log_counter'):
                self._rolling_log_counter += len(rolling_env_ids)
                if self._rolling_log_counter % 50 == 0:  # Log every 50 rolling terminations
                    print(f"[INFO] Rolling terminations detected: {len(rolling_env_ids)} envs. "
                          f"Total rolling terminations: {self._rolling_log_counter}")
        
        # Update success tracking for completed episodes
        for env_id in done_env_ids:
            if self.current_episode_has_success[env_id]:
                self.lift_successes.append(1)
                self.num_lift_successes += 1
            else:
                self.lift_successes.append(0)
            # self.total_episodes += 1
            
            self.print_success_rate()
        
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        rewards = self._compute_rewards(
            self.cube_pos,
            self.num_envs,
        )

        # NaN/Inf sentinel: if anything goes non-finite, force reset of those envs to avoid poisoning training.
        # This can happen if IK diverges (e.g., unreachable targets) or the simulator state becomes unstable.
        bad_reward = ~torch.isfinite(rewards)
        bad_state = (
            (~torch.isfinite(self.cube_pos).all(dim=-1))
            | (~torch.isfinite(self.cube_lin_vel).all(dim=-1))
            | (~torch.isfinite(self.cube_ang_vel).all(dim=-1))
            | (~torch.isfinite(self.robot_dof_targets).all(dim=-1))
        )
        bad = bad_reward | bad_state
        if torch.any(bad):
            bad_ids = torch.where(bad)[0]
            rewards[bad_ids] = 0.0
            # Ensure DirectRLEnv.step() resets these envs this same step
            if hasattr(self, "reset_terminated"):
                self.reset_terminated[bad_ids] = True
            if hasattr(self, "reset_buf"):
                self.reset_buf[bad_ids] = True
            # Rate-limited diagnostic
            if not hasattr(self, "_nan_reward_log_count"):
                self._nan_reward_log_count = 0
            self._nan_reward_log_count += 1
            if self._nan_reward_log_count <= 5:
                print(
                    f"[WARNING] Non-finite reward/state detected -> resetting {len(bad_ids)} envs. "
                    f"Example env_id={bad_ids[0].item()}, reward={rewards[bad_ids[0]].item()}"
                )
            # Optional: surface to logger
            self.extras.setdefault("log", {})
            self.extras["log"]["debug/nonfinite_envs"] = int(len(bad_ids))

        return rewards


    def _reset_idx(self, env_ids: torch.Tensor | None):
        num_envs_to_reset = len(env_ids) if env_ids is not None else self.num_envs
        
        # Get environment IDs (convert None to all environments)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if self.cfg.enforce_csv_reset_init and len(self.csv_data_list) == 0:
            raise RuntimeError(
                "CSV reset initialization is enforced but no CSV datasets are loaded. "
                "Set a valid csv_base_dir containing franka_joint_states.csv, gripper_joint_states.csv, object_pos.csv."
            )
        
        # Initialize robot joint positions from CSV
        if len(self.csv_data_list) > 0:
            # Cycle through CSV datasets for each environment
            joint_pos_list = []
            for i, env_id in enumerate(env_ids):
                # Cycle through available CSV datasets
                csv_idx = env_id.item() % len(self.csv_data_list)
                csv_data = self.csv_data_list[csv_idx]
                init_row = max(int(self.cfg.csv_init_row_index), 0)
                
                if csv_data['joint_states'] is not None and len(csv_data['joint_states']) > init_row:
                    arm_joint_angles = csv_data['joint_states'][init_row, :7]  # 7 arm joints

                    if csv_data.get('gripper_states') is not None and len(csv_data['gripper_states']) > init_row:
                        gripper_width = float(np.clip(float(csv_data['gripper_states'][init_row, 0]), 0.0, 1.0))
                    elif self.cfg.enforce_csv_reset_init:
                        raise RuntimeError(
                            f"CSV gripper state missing/short for env_id={env_id.item()} in {csv_data.get('dir_path', '<unknown>')}"
                        )
                    else:
                        gripper_width = 0.9
                    max_gripper_open = 0.04
                    finger_joint_pos = gripper_width * max_gripper_open
                    
                    # Combine arm and gripper joint positions
                    all_joint_positions = np.concatenate([arm_joint_angles, [finger_joint_pos] * (self._robot.num_joints - 7)])
                    joint_pos_list.append(all_joint_positions)
                else:
                    if self.cfg.enforce_csv_reset_init:
                        raise RuntimeError(
                            f"CSV joint states missing/short for env_id={env_id.item()} in {csv_data.get('dir_path', '<unknown>')}"
                        )
                    default_joint_pos = self._robot.data.default_joint_pos[env_id].cpu().numpy()
                    joint_pos_list.append(default_joint_pos)
            
            # Convert to torch tensor
            joint_pos = torch.tensor(np.array(joint_pos_list), device=self.device, dtype=torch.float32)
            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        else:
            # Default behavior aligned to GR00T script (deterministic reset)
            joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        # Call parent reset first (resets scene including robot to default)
        super()._reset_idx(env_ids)
        
        # Immediately reset robot to CSV positions to override default/home position
        # Do this immediately after scene.reset to minimize visibility of default position
        joint_vel = torch.zeros_like(joint_pos)
        
        # Reset robot root state first (if needed)
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        
        # Set joint positions and targets immediately
        # Note: Always use franka_joint_states.csv for initial position (never compute from isaac_absEEF_gripper.csv)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos
        self.reset_joint_pos_targets[env_ids] = joint_pos
        
        # Force update to ensure positions are applied before any rendering
        self.scene.write_data_to_sim()

        # Get environment origins for the selected envs
        env_origins = self.scene.env_origins[env_ids]

        # Initialize cube position from CSV
        if len(self.csv_data_list) > 0:
            # Cycle through CSV datasets for each environment
            cube_pos_list = []
            cube_rot_list = []
            
            for i, env_id in enumerate(env_ids):
                # Cycle through available CSV datasets
                csv_idx = env_id.item() % len(self.csv_data_list)
                csv_data = self.csv_data_list[csv_idx]
                
                if csv_data['object_pos'] is not None:
                    # Use object position from CSV (x, y, z, yaw)
                    obj_pos_local = csv_data['object_pos'][:3]  # x, y, z (local position within env)
                    obj_yaw = csv_data['object_pos'][3]  # yaw
                    
                    # Get default rotation from cube's default state (matches config: 90° around X-axis)
                    default_rot = self._cube.data.default_root_state[0, 3:7].cpu().numpy()  # (w, x, y, z)
                    
                    # Convert yaw to quaternion and combine with base rotation
                    obj_quat = self._yaw_to_quaternion(obj_yaw)  # Yaw rotation around Z-axis
                    
                    # Combine base rotation with yaw rotation using scipy (same as reference script)
                    base_rot_scipy = np.array([default_rot[1], default_rot[2], default_rot[3], default_rot[0]])  # (x,y,z,w) for scipy
                    yaw_quat_scipy = np.array([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]])  # (x,y,z,w)
                    
                    base_r = Rotation.from_quat(base_rot_scipy)
                    yaw_r = Rotation.from_quat(yaw_quat_scipy)
                    combined_r = yaw_r * base_r  # Apply base first, then yaw (yaw in world frame)
                    combined_quat_scipy = combined_r.as_quat()  # Returns (x,y,z,w)
                    combined_quat = np.array([combined_quat_scipy[3], combined_quat_scipy[0], combined_quat_scipy[1], combined_quat_scipy[2]])  # Convert to (w,x,y,z)
                    
                    cube_pos_list.append(obj_pos_local)
                    cube_rot_list.append(combined_quat)
                else:
                    if self.cfg.enforce_csv_reset_init:
                        raise RuntimeError(
                            f"CSV object_pos missing for env_id={env_id.item()} in {csv_data.get('dir_path', '<unknown>')}"
                        )
                    cube_pos_list.append(np.array([0.2, 0.0, 0.06], dtype=np.float32))
                    cube_rot_list.append(np.array([0.707, 0.0, 0.0, 0.707], dtype=np.float32))
            
            # Convert to world coordinates
            cube_pos_local = torch.tensor(np.array(cube_pos_list), device=self.device, dtype=torch.float32)
            cube_pos = env_origins + cube_pos_local  # shape: (len(env_ids), 3)
            cube_rot = torch.tensor(np.array(cube_rot_list), device=self.device, dtype=torch.float32)
        else:
            # Default behavior aligned to GR00T script (deterministic)
            cube_offset = torch.tensor([0.2, 0.0, 0.06], device=self.device)
            cube_pos = env_origins + cube_offset
            cube_rot = torch.tensor([0.707, 0.0, 0.0, 0.707], device=self.device).repeat(num_envs_to_reset, 1)
        
        cube_vel = torch.zeros((num_envs_to_reset, 6), device=self.device)

        # Write to sim
        self._cube.write_root_pose_to_sim(torch.cat((cube_pos, cube_rot), dim=-1), env_ids=env_ids)
        self._cube.write_root_velocity_to_sim(cube_vel, env_ids=env_ids)


        if env_ids is None:
            self.cube_initial_pos = cube_pos.clone()
        else:
            self.cube_initial_pos[env_ids] = cube_pos.clone()

        # Position the robot's end-effector above the cube using IK
        # Removed - robot is positioned from CSV joint states when CSV data is available

        # Refresh intermediate values for the specified environments
        self._compute_intermediate_values(env_ids)
        
        # Set training flag for exploration
        self.is_training = True

        # Reset success tracking for these environments
        if env_ids is not None:
            self.current_episode_has_success[env_ids] = False
        else:
            self.current_episode_has_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Reset CSV row indices for reset environments
        if len(self.csv_data_list) > 0:
            init_row = max(int(getattr(self.cfg, "csv_init_row_index", 0)), 0)
            if env_ids is not None:
                self.csv_row_indices[env_ids] = init_row
                self.csv_step_counters[env_ids] = 0  # Reset step counters
                # Reset IK target tracking to ensure command is set on first step
                if hasattr(self, '_last_ik_target_indices'):
                    self._last_ik_target_indices[env_ids] = -1
            else:
                self.csv_row_indices[:] = init_row
                self.csv_step_counters[:] = 0
                # Reset IK target tracking to ensure command is set on first step
                if hasattr(self, '_last_ik_target_indices'):
                    self._last_ik_target_indices[:] = -1
        
        # Reset API call tracking for reset environments
        if env_ids is not None:
            self.steps_since_last_inference[env_ids] = int(self.cfg.api_call_interval)
            self.post_reset_settle_remaining[env_ids] = int(getattr(self.cfg, "post_reset_settle_steps", 0))
            self.api_consecutive_failures[env_ids] = 0
            self.api_cooldown_remaining[env_ids] = 0
            self.api_disabled[env_ids] = False
            # Reset action window tracking
            for env_id in env_ids:
                env_id_item = env_id.item()
                self.current_action_windows[env_id_item] = None
                self.current_window_idx[env_id_item] = 0
                self.steps_in_current_window[env_id_item] = 0
                self.num_windows[env_id_item] = None
                self.steps_per_window[env_id_item] = 1
                self.api_response_buffer[env_id_item].clear()
        else:
            self.steps_since_last_inference[:] = int(self.cfg.api_call_interval)
            self.post_reset_settle_remaining[:] = int(getattr(self.cfg, "post_reset_settle_steps", 0))
            self.api_consecutive_failures[:] = 0
            self.api_cooldown_remaining[:] = 0
            self.api_disabled[:] = False
            # Reset action window tracking for all environments
            for env_id in range(self.num_envs):
                self.current_action_windows[env_id] = None
                self.current_window_idx[env_id] = 0
                self.steps_in_current_window[env_id] = 0
                self.num_windows[env_id] = None
                self.steps_per_window[env_id] = 1
                self.api_response_buffer[env_id].clear()

        # Update camera positions after reset
        if env_ids is None or len(env_ids) > 0:
            # Only update cameras for the reset environments
            self._update_camera_poses_for_envs(env_ids)

    def _get_observations(self) -> dict:
        # Get robot state observations
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.cube_pos - self.robot_grasp_pos 

        # CogACT reference for this timestep (the one that will be used when policy's action is applied)
        # Note: window index is advanced in _pre_physics_step, so we read the current index here
        cogact_ref_list = []
        if self.cfg.use_api_for_pose:
            for env_id in range(self.num_envs):
                # Before first API call: return zeros (normal case)
                if self.current_action_windows[env_id] is None or self.num_windows[env_id] is None:
                    cogact_ref_list.append(np.zeros(7, dtype=np.float32))
                    continue
                
                # After API call: validate and extract reference
                idx = self.current_window_idx[env_id].item()
                nw = self.num_windows[env_id]
                if idx >= nw:
                    raise IndexError(
                        f"CogACT window index out of range for env_id={env_id}: "
                        f"current_window_idx={idx}, num_windows={nw}. idx must be < num_windows."
                    )
                cogact_ref_list.append(self.current_action_windows[env_id][idx].astype(np.float32))
        else:
            cogact_ref_list = [np.zeros(7, dtype=np.float32) for _ in range(self.num_envs)]
        
        cogact_reference = torch.tensor(np.array(cogact_ref_list), device=self.device, dtype=torch.float32)

        # Ensure actions tensor exists and has proper shape (for internal use; not fed to policy obs)
        if not hasattr(self, 'actions') or self.actions.shape[0] != self.num_envs:
            self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Get contact forces from left and right fingers
        # Use contact PhysX view (requires activate_contact_sensors=True in robot config)
        # Contact view is initialized in _setup_scene for better performance
        
        # Get contact forces using the initialized view
        try:
            if self._contact_physx_view is not None and self._contact_view_initialized:
                sim_dt = self.sim.get_physics_dt()
                # Get net contact forces for finger bodies (should be 2 bodies per env: left and right fingers)
                contact_forces_w = self._contact_physx_view.get_net_contact_forces(dt=sim_dt)
                # Reshape to (num_envs, num_bodies, 3)
                num_bodies_in_view = contact_forces_w.shape[0] // self.num_envs
                contact_forces_w = contact_forces_w.view(self.num_envs, num_bodies_in_view, 3)
                
                if num_bodies_in_view == 2:
                    # Use stored body order mapping to correctly assign left vs right finger forces
                    if hasattr(self, '_finger_body_order') and self._finger_body_order is not None:
                        left_idx, right_idx = self._finger_body_order[0], self._finger_body_order[1]
                        left_finger_force = contact_forces_w[:, left_idx, :]
                        # right_finger_force = contact_forces_w[:, right_idx, :]
                    else:
                        # Fallback: assume order [0, 1]
                        print(f"[WARNING] No finger body order mapping found in _get_observations, using default [0, 1]")
                        left_finger_force = contact_forces_w[:, 0, :]
                        # right_finger_force = contact_forces_w[:, 1, :]
                else:
                    # Fallback: use zeros if body count doesn't match
                    print(f"[WARNING] Contact view has {num_bodies_in_view} bodies per env, expected 2. Using zero forces.")
                    left_finger_force = torch.zeros((self.num_envs, 3), device=self.device)
                    # right_finger_force = torch.zeros((self.num_envs, 3), device=self.device)
            else:
                # Fallback: use zeros if contact view not initialized
                left_finger_force = torch.zeros((self.num_envs, 3), device=self.device)
                # right_finger_force = torch.zeros((self.num_envs, 3), device=self.device)
        except (AttributeError, RuntimeError, Exception) as e:
            # Fallback: use zeros if contact sensors not enabled or not available
            print(f"[WARNING] Error retrieving contact forces: {e}. Using zero forces.")
            left_finger_force = torch.zeros((self.num_envs, 3), device=self.device)
            # right_finger_force = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Convert 3D force vectors to binary per dimension: 1 if |force| > 0.1, else 0
        # For left finger only, check each axis (x, y, z) independently
        contact_threshold = 0.1  # Newtons
        left_finger_force_binary = ((torch.abs(left_finger_force) > contact_threshold)).float()  # Shape: (num_envs, 3)
        
        # Use only left finger binary forces: [left_x, left_y, left_z]
        contact_obs = left_finger_force_binary  # Shape: (num_envs, 3)

        # Get endeffector XYZRPY (position + orientation)
        # Convert quaternion to RPY (roll, pitch, yaw)
        roll, pitch, yaw = euler_xyz_from_quat(self.robot_grasp_rot)  # Each returns (num_envs,)
        # Stack RPY into (num_envs, 3)
        rpy = torch.stack([roll, pitch, yaw], dim=-1)  # Shape: (num_envs, 3)
        # Concatenate position (XYZ) and orientation (RPY)
        endeffector_xyzrpy = torch.cat([self.robot_grasp_pos, rpy], dim=-1)  # Shape: (num_envs, 6)

        # Get latent values from API mapping (if using API) or zeros otherwise (per-environment)
        latent_obs = None
        if self.cfg.use_api_for_pose:
            # Collect latents for each environment
            latent_list = []
            for env_id in range(self.num_envs):
                if self.current_mapped_latent[env_id] is not None:
                    latent_list.append(self.current_mapped_latent[env_id])
                    # Update latent_dim if needed (use first non-None latent)
                    if self.latent_dim == 0:
                        self.latent_dim = len(self.current_mapped_latent[env_id])
                else:
                    # Use zeros if no latent available for this env
                    if self.latent_dim > 0:
                        latent_list.append(np.zeros(self.latent_dim, dtype=np.float32))
                    else:
                        # Default: use 4096 dimensions
                        self.latent_dim = 4096
                        latent_list.append(np.zeros(self.latent_dim, dtype=np.float32))
            
            # Convert to tensor
            latent_obs = torch.tensor(np.array(latent_list), device=self.device, dtype=torch.float32)
        elif self.latent_dim > 0:
            # Use zeros if no mapped latent available but dimension is set
            latent_obs = torch.zeros((self.num_envs, self.latent_dim), device=self.device, dtype=torch.float32)
        else:
            # Default: use 4096 dimensions (typical latent size)
            self.latent_dim = 4096
            latent_obs = torch.zeros((self.num_envs, self.latent_dim), device=self.device, dtype=torch.float32)

        # Robot-state based observations
        obs_components = [
            dof_pos_scaled,
            self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
            # self.cube_pos,
            # self.cube_rot,
            # to_target,
            cogact_reference,
            contact_obs,
            endeffector_xyzrpy,
        ]
        
        # Add latent values if available (only when latent_dim > 0)
        if latent_obs is not None and self.latent_dim > 0:
            obs_components.append(latent_obs)
        
        policy_obs = torch.cat(obs_components, dim=-1)
        
        
        obs_dict = {"policy": torch.clamp(policy_obs, -5.0, 5.0)}
        
        # Get camera observations if available
        if self._camera_enabled and hasattr(self, '_camera_front') and self._camera_front is not None:
            try:
                self._camera_front.update(dt=self.sim.get_physics_dt())
                if hasattr(self._camera_front, 'data') and hasattr(self._camera_front.data, 'output'):
                    if "rgb" in self._camera_front.data.output:
                        rgb_data = self._camera_front.data.output["rgb"].to(dtype=torch.float32)
                        if rgb_data.max() > 1.0:
                            rgb_data = rgb_data / 255.0
                        obs_dict["rgb"] = rgb_data
                    if "distance_to_image_plane" in self._camera_front.data.output:
                        depth_data = self._camera_front.data.output["distance_to_image_plane"]
                        max_depth = 10.0
                        normalized_depth = torch.clamp(depth_data, 0.0, max_depth) / max_depth
                        obs_dict["depth"] = normalized_depth
            except Exception as e:
                # Silently ignore camera errors if camera is not properly initialized
                pass
        
        return obs_dict

    # auxiliary methods

    def _initialize_contact_view(self):
        """Initialize contact view for finger bodies (called once in _setup_scene)."""
        # Initialize flag if it doesn't exist (for first call)
        if not hasattr(self, '_contact_view_initialized'):
            self._contact_view_initialized = False
        
        if self._contact_view_initialized:
            return
        
        try:
            import omni.physics.tensors.impl.api as physx
            from pxr import PhysxSchema, UsdPhysics
            import isaaclab.sim as sim_utils
            # Create physics simulation view (Isaac Lab uses "torch" backend)
            physics_sim_view = physx.create_simulation_view("torch")
            physics_sim_view.set_subspace_roots("/")
            
            # Find finger body names from robot's body_names
            finger_body_names = [name for name in self._robot.body_names if "leftfinger" in name or "rightfinger" in name]
            
            if len(finger_body_names) >= 2:
                robot_prim_path = self.cfg.robot.prim_path
                finger_pattern_glob = f"{robot_prim_path.replace('.*', '*')}/*finger*"
                
                body_physx_view = physics_sim_view.create_rigid_body_view(finger_pattern_glob)
                if body_physx_view is not None and body_physx_view.count > 0:
                    num_bodies = body_physx_view.count // self.num_envs
                    
                    # Verify contact sensors
                    finger_prims = []
                    if hasattr(body_physx_view, 'prim_paths') and len(body_physx_view.prim_paths) > 0:
                        first_env_paths = body_physx_view.prim_paths[:num_bodies]
                        stage = get_current_stage()
                        for path in first_env_paths:
                            prim = stage.GetPrimAtPath(path)
                            if prim.IsValid():
                                finger_prims.append(prim)
                    
                    contact_sensors_enabled = [prim.HasAPI(PhysxSchema.PhysxContactReportAPI) for prim in finger_prims]
                    
                    if not all(contact_sensors_enabled) if contact_sensors_enabled else False:
                        print(f"[WARNING] Some finger bodies do not have contact sensors enabled!")
                    elif len(contact_sensors_enabled) > 0:
                        print(f"[INFO] All {len(contact_sensors_enabled)} finger bodies have contact sensors enabled!")
                    
                    # Create contact view
                    self._contact_physx_view = physics_sim_view.create_rigid_contact_view(finger_pattern_glob)
                    if self._contact_physx_view is not None:
                        self._finger_body_order = [None, None]
                        
                        # Get order from contact view
                        try:
                            if hasattr(self._contact_physx_view, 'prim_paths'):
                                contact_prim_paths = self._contact_physx_view.prim_paths
                                if contact_prim_paths is not None and len(contact_prim_paths) > 0:
                                    num_bodies_contact = len(contact_prim_paths) // self.num_envs
                                    first_env_contact_paths = contact_prim_paths[:num_bodies_contact]
                                    for i, path in enumerate(first_env_contact_paths):
                                        body_name = path.split("/")[-1]
                                        if "leftfinger" in body_name.lower():
                                            self._finger_body_order[0] = i
                                        elif "rightfinger" in body_name.lower():
                                            self._finger_body_order[1] = i
                        except Exception:
                            pass
                        
                        # Fallback to body view
                        if (self._finger_body_order[0] is None or self._finger_body_order[1] is None) and \
                           hasattr(body_physx_view, 'prim_paths') and len(body_physx_view.prim_paths) > 0:
                            first_env_paths = body_physx_view.prim_paths[:num_bodies]
                            for i, path in enumerate(first_env_paths):
                                body_name = path.split("/")[-1]
                                if "leftfinger" in body_name.lower():
                                    self._finger_body_order[0] = i
                                elif "rightfinger" in body_name.lower():
                                    self._finger_body_order[1] = i
                        
                        if self._finger_body_order[0] is None or self._finger_body_order[1] is None:
                            print(f"[WARNING] Could not determine finger body order, assuming [0, 1]")
                            self._finger_body_order = [0, 1]
                        
                        print(f"[INFO] Contact view initialized with {num_bodies} bodies per environment")
                        print(f"[INFO] Finger body order: left={self._finger_body_order[0]}, right={self._finger_body_order[1]}")
                        self._contact_view_initialized = True
                    else:
                        print("[WARNING] Failed to create contact view. Contact observations will be zeros.")
                else:
                    print(f"[WARNING] Failed to create body view. Contact observations will be zeros.")
            else:
                print(f"[WARNING] Could not find finger body names. Contact observations will be zeros.")
        except Exception as e:
            err_msg = str(e)
            if "no active physics scene found" in err_msg.lower():
                return
            print(f"[WARNING] Error initializing contact view: {e}. Contact observations will be zeros.")

    def _compute_contact_forces(self, env_ids: torch.Tensor | None = None):
        """Compute contact forces for left and right fingers."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        
        # Lazily initialize/retry contact view after physics is active.
        if not self._contact_view_initialized:
            step_count = int(getattr(self, "common_step_counter", 0))
            if step_count >= int(getattr(self, "_contact_init_next_step", 0)):
                self._initialize_contact_view()
                if not self._contact_view_initialized:
                    self._contact_init_next_step = step_count + int(getattr(self, "_contact_init_retry_interval", 20))
            if not self._contact_view_initialized and not self._contact_init_wait_log_emitted:
                print("[INFO] Contact view not ready yet; using zero contact forces temporarily.")
                self._contact_init_wait_log_emitted = True
        
        # Get contact forces
        try:
            if self._contact_physx_view is not None and self._contact_view_initialized:
                sim_dt = self.sim.get_physics_dt()
                contact_forces_w = self._contact_physx_view.get_net_contact_forces(dt=sim_dt)
                num_bodies_in_view = contact_forces_w.shape[0] // self.num_envs
                contact_forces_w = contact_forces_w.view(self.num_envs, num_bodies_in_view, 3)
                
                if num_bodies_in_view == 2:
                    # Use stored body order mapping to correctly assign left vs right finger forces
                    if hasattr(self, '_finger_body_order') and self._finger_body_order is not None:
                        left_idx, right_idx = self._finger_body_order[0], self._finger_body_order[1]
                        left_finger_force = contact_forces_w[:, left_idx, :]
                        # right_finger_force = contact_forces_w[:, right_idx, :]
                    else:
                        # Fallback: assume order [0, 1]
                        print(f"[WARNING] No finger body order mapping found in _compute_contact_forces, using default [0, 1]")
                        left_finger_force = contact_forces_w[:, 0, :]
                        # right_finger_force = contact_forces_w[:, 1, :]
                    
                    # Store 3D force vectors
                    self.left_finger_force[env_ids] = left_finger_force[env_ids]
                    # self.right_finger_force[env_ids] = right_finger_force[env_ids]
                else:
                    self.left_finger_force[env_ids] = 0.0
                    # self.right_finger_force[env_ids] = 0.0
            else:
                self.left_finger_force[env_ids] = 0.0
                # self.right_finger_force[env_ids] = 0.0
        except Exception:
            self.left_finger_force[env_ids] = 0.0
            # self.right_finger_force[env_ids] = 0.0

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        self._cube.reset()
        self._cube.write_data_to_sim()
        sim_dt = self.sim.get_physics_dt()
        self._cube.update(sim_dt)
        cube_pos = self._cube.data.root_pos_w[env_ids]
        cube_rot = self._cube.data.root_quat_w[env_ids]
        cube_lin_vel = self._cube.data.root_lin_vel_w[env_ids]
        cube_ang_vel = self._cube.data.root_ang_vel_w[env_ids]
        
        # Update only the specified environments
        self.cube_pos[env_ids] = cube_pos
        self.cube_rot[env_ids] = cube_rot
        self.cube_lin_vel[env_ids] = cube_lin_vel
        self.cube_ang_vel[env_ids] = cube_ang_vel
        
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
        )
        
        # Compute finger position (average of left and right fingers) for distance reward
        left_finger_pos = self._robot.data.body_pos_w[env_ids, self.finger_body_indices[0]]
        right_finger_pos = self._robot.data.body_pos_w[env_ids, self.finger_body_indices[1]]
        self.finger_pos[env_ids] = (left_finger_pos + right_finger_pos) / 2.0
        self.left_finger_pos_w[env_ids] = left_finger_pos
        self.right_finger_pos_w[env_ids] = right_finger_pos
        
        # Compute contact forces for reward
        self._compute_contact_forces(env_ids)

    def _compute_rewards(
        self,
        cube_pos,
        num_envs,
    ):
        # Distance between gripper fingers (average of left and right) and cube center
        finger_cube_distance = torch.norm(self.finger_pos - cube_pos - self.cfg.finger_length, p=2, dim=-1) 
        
        # ========== STAGE 1: Distance reward (always active) ==========
        # Distance reward: tanh-kernel (reward for being close to cube)
        # Formula: (1 - tanh(distance / std)) * weight
        # Returns 1.0 when distance=0, decreases smoothly as distance increases
        distance_reward = (1.0 - torch.tanh(finger_cube_distance / self.cfg.distance_reward_std)) * self.cfg.distance_reward_weight
        # print("finger_cube_distance, distance_reward:", finger_cube_distance, distance_reward)

        # Calculate cube height from ground
        cube_height = cube_pos[:, 2] - self.cfg.cube_half_size
        
        # ========== STAGE 2: Contact reward (only if left finger is below cube height + margin AND above ground) ==========
        # Contact reward: use binary x-axis force only (1 if |force_x| > 0.1, else 0)
        contact_threshold = 0.1  # Newtons
        left_finger_force_x_binary = (torch.abs(self.left_finger_force[:, 0]) > contact_threshold).float()
        # print("Namiko self.left_finger_force, ", self.left_finger_force)
        
        left_contact_reward = left_finger_force_x_binary * self.cfg.contact_reward_weight

        # Gate contact reward: only count contact if left finger is:
        # 1. Below the cube height + margin (to ensure finger is at grasping position)
        # 2. Above ground (to avoid rewarding contact when finger is on the ground)
        cube_z = cube_pos[:, 2]
        left_below_cube = (self.left_finger_pos_w[:, 2] < (cube_z + self.cfg.contact_above_cube_margin)).float()
        left_above_ground = (self.left_finger_pos_w[:, 2] > self.cfg.contact_ground_threshold).float()
        
        # Apply both gating conditions
        left_contact_gate = left_below_cube * left_above_ground
        left_contact_reward = left_contact_reward * left_contact_gate
        
        # Contact reward is based on left finger only
        contact_reward = left_contact_reward

        # ========== STAGE 3: Height reward (only if contact is ON) ==========
        # Height reward: tanh-kernel (reward for lifting cube higher)
        # Only reward if cube is lifted above minimal_height
        # Use tanh to create smooth reward that increases with height
        # This gives higher reward as cube gets higher, with smooth saturation
        height_reward = (cube_height > self.cfg.minimal_height).float() * torch.tanh(cube_height / self.cfg.height_reward_std) * self.cfg.height_reward_weight
        # Gate height reward: only reward lifting if left finger is in contact

        height_reward = height_reward * left_finger_force_x_binary * left_contact_gate
        # print("Namiko", height_reward, contact_reward)

        # Success bonus: binary reward if lifted above threshold
        cube_is_lifted = cube_height >= self.cfg.height_threshold
        success_reward = cube_is_lifted.float() * self.cfg.success_reward_scale * left_finger_force_x_binary * left_contact_gate
        
        # Track success for statistics
        self.current_episode_has_success |= cube_is_lifted
        
        rewards = height_reward + success_reward + distance_reward + contact_reward
        # print("Namiko rewards, ", rewards,  "height_reward, ", height_reward, "success_reward, ", success_reward, "distance_reward, ", distance_reward, "contact_reward, ", contact_reward)
        
        # Logging
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 50 == 0:
            num_lifted = torch.sum(cube_is_lifted).item()
            print(f"Agents lifted: {num_lifted}/{num_envs}, Avg height: {cube_height.mean().item():.3f}m, Max Reward: {rewards.max().item():.2f}")
            
            # Prepare logging dictionary with organized reward breakdown
            log_dict = {
                # Total rewards
                "rewards/total": rewards.mean().item(),
                "rewards/success": success_reward.mean().item(),
                
                # Individual reward components (for easy tracking in wandb)
                "rewards/distance": distance_reward.mean().item(),
                "rewards/contact": contact_reward.mean().item(),
                "rewards/height": height_reward.mean().item(),
                
                # State metrics
                "state/finger_cube_distance": finger_cube_distance.mean().item(),
                "state/cube_height": cube_height.mean().item(),
                "state/num_lifted": num_lifted,
                "state/left_finger_contact_binary": left_finger_force_x_binary.mean().item(),
                
                # Reward parameters (for reference)
                "config/distance_reward_std": self.cfg.distance_reward_std,
                "config/height_reward_std": self.cfg.height_reward_std,
            }
            
            
            self.extras["log"] = log_dict
            
            # Log to wandb if enabled
            if self.cfg.use_wandb:
                try:
                    wandb_module = __import__('wandb', fromlist=['log'])
                    if hasattr(wandb_module, 'run') and wandb_module.run is not None:
                        wandb_module.log(self.extras["log"])
                except Exception:
                    pass
            
        return rewards

    def _euler_to_quaternion(self, euler_angles):
        """Convert Euler angles (XYZ) to quaternion (w,x,y,z)."""
        r = Rotation.from_euler('xyz', euler_angles)
        quat = r.as_quat()  # Returns (x,y,z,w)
        # Reorder to (w,x,y,z)
        return np.array([quat[3], quat[0], quat[1], quat[2]])

    def _yaw_to_quaternion(self, yaw):
        """Convert yaw angle (rotation around Z-axis) to quaternion (w,x,y,z)."""
        r = Rotation.from_euler('z', yaw, degrees=False)
        quat = r.as_quat()  # Returns (x,y,z,w)
        # Reorder to (w,x,y,z)
        return np.array([quat[3], quat[0], quat[1], quat[2]])

    def _calculate_adaptive_episode_length(self):
        """Calculate episode length based on maximum CSV length."""
        if len(self.csv_data_list) == 0:
            return
        
        # Find maximum number of rows across all CSV datasets
        max_csv_rows = 0
        for csv_data in self.csv_data_list:
            if csv_data.get('absolute_poses') is not None:
                max_csv_rows = max(max_csv_rows, len(csv_data['absolute_poses']))
        
        if max_csv_rows == 0:
            return
        
        # Playback rate: 20 Hz means each CSV row takes 0.05 seconds
        playback_rate_hz = 20.0
        playback_dt = 1.0 / playback_rate_hz
        
        # Calculate required time: number of rows * time per row + small buffer for safety
        # Small buffer to account for timing differences, but episodes will terminate when CSV completes
        buffer_time = 0.5  # 0.5 seconds buffer (reduced from 2.0 since we now terminate on CSV completion)
        required_time = max_csv_rows * playback_dt + buffer_time
        
        # Update episode length (ensure minimum of original value)
        original_episode_length = self.cfg.episode_length_s
        self.cfg.episode_length_s = max(required_time, original_episode_length)
        
        print(f"[INFO] Adaptive episode length: {self.cfg.episode_length_s:.2f}s "
              f"(based on max CSV rows: {max_csv_rows}, original: {original_episode_length:.2f}s)")
    
    def _load_csv_data(self):
        """Load CSV data for initial poses from all pickMushroom_* directories."""
        if self.cfg.csv_base_dir is None:
            print("[INFO] csv_base_dir not configured, using default initialization")
            return
        
        if not os.path.exists(self.cfg.csv_base_dir):
            print(f"[WARNING] CSV base directory does not exist: {self.cfg.csv_base_dir}, using default initialization")
            return
        
        # If base dir itself is an episode directory, use it directly.
        base_joint_csv = os.path.join(self.cfg.csv_base_dir, self.cfg.joint_states_csv)
        if os.path.exists(base_joint_csv):
            subdirs = ["."]
        else:
            # Prefer pickMushroom_* naming, then fall back to any subdir containing required joint CSV.
            preferred_subdirs = [
                d for d in os.listdir(self.cfg.csv_base_dir)
                if os.path.isdir(os.path.join(self.cfg.csv_base_dir, d)) and d.startswith('pickMushroom_')
            ]
            if preferred_subdirs:
                subdirs = sorted(preferred_subdirs)
            else:
                subdirs = []
                for d in sorted(os.listdir(self.cfg.csv_base_dir)):
                    d_path = os.path.join(self.cfg.csv_base_dir, d)
                    if not os.path.isdir(d_path) or d.startswith('.'):
                        continue
                    if os.path.exists(os.path.join(d_path, self.cfg.joint_states_csv)):
                        subdirs.append(d)

        if not subdirs:
            print(f"[WARNING] No valid CSV episode directories found in {self.cfg.csv_base_dir}, using default initialization")
            return
        
        print(f"[INFO] Found {len(subdirs)} CSV episode directories to load:")
        for subdir in subdirs:
            print(f"  - {subdir}")
        
        # Load CSV data from each directory
        for subdir in subdirs:
            csv_dir = self.cfg.csv_base_dir if subdir == "." else os.path.join(self.cfg.csv_base_dir, subdir)
            
            # Determine CSV file paths
            joint_csv_path = os.path.join(csv_dir, self.cfg.joint_states_csv)
            gripper_csv_path = os.path.join(csv_dir, self.cfg.gripper_states_csv)
            object_csv_path = os.path.join(csv_dir, self.cfg.object_pos_csv)
            
            csv_data = {
                'joint_states': None,
                'gripper_states': None,
                'object_pos': None,
                'dir_path': csv_dir,
                'dir_name': os.path.basename(csv_dir)
            }
            
            # Load joint states CSV
            if os.path.exists(joint_csv_path):
                try:
                    joint_states_data = pd.read_csv(joint_csv_path, header=0).values.astype(np.float32)
                    # Extract columns 10-16 (7 arm joints) starting from row 1 (skip header)
                    if joint_states_data.shape[0] > 1 and joint_states_data.shape[1] > 16:
                        csv_data['joint_states'] = joint_states_data[1:, 10:17]  # Skip header, get columns 10-16
                    else:
                        print(f"[WARNING] Joint states CSV has insufficient data: {joint_csv_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to load joint states CSV {joint_csv_path}: {e}")
            else:
                print(f"[WARNING] Joint states CSV not found: {joint_csv_path}")
            
            # Load gripper states CSV (used for reset-time initialization to match my_test)
            if os.path.exists(gripper_csv_path):
                try:
                    gripper_states_data = pd.read_csv(gripper_csv_path, header=0).values.astype(np.float32)
                    # Extract column 4 (index 4) starting from row 1 (same convention as my_test)
                    if gripper_states_data.shape[0] > 1 and gripper_states_data.shape[1] > 4:
                        csv_data['gripper_states'] = gripper_states_data[1:, 4:5]
                    else:
                        print(f"[WARNING] Gripper states CSV has insufficient data: {gripper_csv_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to load gripper states CSV {gripper_csv_path}: {e}")
            else:
                print(f"[WARNING] Gripper states CSV not found: {gripper_csv_path}")
            
            # Load object position CSV
            if os.path.exists(object_csv_path):
                try:
                    object_pos_data = pd.read_csv(object_csv_path, header=0).values.astype(np.float32)
                    # Extract columns 3-7 (indices 3,4,5,6) = x, y, z, yaw from first row
                    if object_pos_data.shape[0] > 0 and object_pos_data.shape[1] > 6:
                        csv_data['object_pos'] = object_pos_data[0, 3:7].copy()  # x, y, z, yaw
                    else:
                        print(f"[WARNING] Object position CSV has insufficient data: {object_csv_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to load object position CSV {object_csv_path}: {e}")
            else:
                print(f"[WARNING] Object position CSV not found: {object_csv_path}")
            
            # Only add to list if we have at least joint states (required)
            if csv_data['joint_states'] is not None:
                self.csv_data_list.append(csv_data)
                self.csv_dir_paths.append(csv_dir)
                print(f"[INFO] Loaded CSV data from {subdir}: joint_states shape={csv_data['joint_states'].shape}, "
                      f"gripper_states={'loaded' if csv_data['gripper_states'] is not None else 'missing'}, "
                      f"object_pos={'loaded' if csv_data['object_pos'] is not None else 'missing'}")
            else:
                print(f"[WARNING] Skipping {subdir} due to missing joint states CSV")
        
        if len(self.csv_data_list) == 0:
            print(f"[WARNING] No valid CSV data loaded, using default initialization")
        else:
            print(f"[INFO] Successfully loaded {len(self.csv_data_list)} CSV datasets")
            # Calculate adaptive episode length based on maximum CSV length
            self._calculate_adaptive_episode_length()

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos

    def print_success_rate(self):
        """Print the success rate of lifting cubes among recent trials."""
        if len(self.lift_successes) > 0:
            recent_success_rate = sum(self.lift_successes) / len(self.lift_successes)
            # overall_success_rate = self.num_lift_successes / self.total_episodes
            
            print(f"\n=== CUBE LIFTING SUCCESS STATISTICS ===")
            print(f"Recent success rate (last {len(self.lift_successes)} trials): {recent_success_rate:.2%}")
            # print(f"Overall success rate ({self.total_episodes} trials): {overall_success_rate:.2%}")
            print(f"Successful lifts in tracking window: {sum(self.lift_successes)} / {len(self.lift_successes)}")
            # print(f"Total successful lifts: {self.num_lift_successes} / {self.total_episodes}")
            print(f"=========================================\n")
            
    def _update_camera_poses_for_envs(self, env_ids: torch.Tensor | None):
        """Update camera poses for specific environments after reset."""
        if not self._camera_enabled or self._camera_front is None or self._camera_back is None:
            return
        # Simply call setup cameras which sets all cameras
        self._setup_cameras()
        
    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply actions, simulate, and return observations, rewards, dones, and info."""
        # Set up cameras after the first step (if enabled)
        if self._camera_enabled and (not hasattr(self, '_camera_setup_done') or not self._camera_setup_done):
            self._setup_cameras()
            self._camera_setup_done = True
        
        # Step the environment
        return super().step(actions)