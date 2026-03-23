"""
Iface-based environment wrapper that replicates the EXACT online iface script loop.

This wrapper creates the same scene, cameras, IK controller, and action window logic
as franka_cogact_motion_generation_abs_window_gr00t_training_only_iface.py,
but exposes a gymnasium-like step/reset interface for RL training.

Key guarantee: when residual_action=0, the robot behaves identically to the
online iface script (same physics, same GR00T inference, same IK, same smoothing).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


def _bootstrap_typing_extensions() -> None:
    try:
        workspace_dir = Path(__file__).resolve().parents[3] / "workspace"
        deps_dir = Path(
            os.environ.get("ISO_DEPS_DIR", str(workspace_dir / ".pydeps_groot_iso"))
        ).expanduser().resolve()
        te_file = deps_dir / "typing_extensions.py"
        if not te_file.exists():
            return
        deps_dir_str = str(deps_dir)
        if deps_dir_str in sys.path:
            sys.path.remove(deps_dir_str)
        sys.path.insert(0, deps_dir_str)
        spec = importlib.util.spec_from_file_location("typing_extensions", str(te_file))
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "NoExtraItems"):
            sys.modules["typing_extensions"] = module
    except Exception:
        return


_bootstrap_typing_extensions()

import pandas as pd
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

try:
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
except ImportError:
    Gr00tPolicy = None
    EmbodimentTag = None


# ── Constants (same as online iface script) ──
VIDEO_KEYS = ["wrist_view", "left_view", "right_view"]
STATE_JOINT_KEY = "proprio.joint_pos"
STATE_GRIPPER_KEY = "proprio.gripper_pos"
LANGUAGE_KEY = "annotation.human.action.task_description"


# ── Scene config (copied verbatim from online iface script) ──
@configclass
class _TableTopSceneCfg(InteractiveSceneCfg):
    floor = AssetBaseCfg(
        prim_path="/World/Environment/floor",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Lights/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0)),
    )
    sun_light = AssetBaseCfg(
        prim_path="/World/Lights/SunLight",
        spawn=sim_utils.DistantLightCfg(intensity=500.0, color=(1.0, 0.98, 0.95), angle=0.53),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0), rot=(0.892, 0.239, 0.369, 0.0)),
    )
    try:
        key_light = AssetBaseCfg(
            prim_path="/World/Lights/KeyLight",
            spawn=sim_utils.RectLightCfg(intensity=3000.0, color=(1.0, 0.98, 0.95), width=1.2, height=0.8),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.2, -1.0, 2.0), rot=(0.92, 0.23, 0.30, 0.0)),
        )
    except AttributeError:
        pass
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.12, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=2.0, disable_gravity=False,
                solver_position_iteration_count=12, solver_velocity_iteration_count=1,
                enable_gyroscopic_forces=True, retain_accelerations=False,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.06), rot=(0.707, 0.0, 0.0, 0.707)),
    )
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=True,  # Enable for contact-force reward
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157, "panda_joint2": -1.066, "panda_joint3": -0.155,
                "panda_joint4": -2.239, "panda_joint5": -1.841, "panda_joint6": 1.003,
                "panda_joint7": 0.469, "panda_finger_joint.*": 0.035,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(joint_names_expr=["panda_joint[1-4]"], effort_limit_sim=87.0, stiffness=8000.0, damping=400.0),
            "panda_forearm": ImplicitActuatorCfg(joint_names_expr=["panda_joint[5-7]"], effort_limit_sim=12.0, stiffness=8000.0, damping=400.0),
            "panda_hand": ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"], effort_limit_sim=500.0, stiffness=2e4, damping=1500.0),
        },
    )

    # Contact sensor for finger force detection (grasp reward gating)
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,  # every physics step
        history_length=0,
        track_air_time=False,
    )

def _make_scene_cfg(num_envs: int = 1, env_spacing: float = 2.0) -> _TableTopSceneCfg:
    cfg = _TableTopSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    q0 = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32)
    theta = np.deg2rad(-20.0)
    q_pitch = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0], dtype=np.float32)
    base_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]], dtype=np.float32)
    pitch_xyzw = np.array([q_pitch[1], q_pitch[2], q_pitch[3], q_pitch[0]], dtype=np.float32)
    q_new_xyzw = (Rotation.from_quat(base_xyzw) * Rotation.from_quat(pitch_xyzw)).as_quat()
    q_new_wxyz = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]], dtype=np.float32)

    cfg.camera_front = CameraCfg(
        prim_path="/World/envs/env_.*/DebugCamera_front", width=640, height=480,
        offset=CameraCfg.OffsetCfg(pos=(0.4, -0.7, 0.8), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, focus_distance=1.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)),
        data_types=["rgb"],
    )
    cfg.camera_back = CameraCfg(
        prim_path="/World/envs/env_.*/DebugCamera_back", width=640, height=480,
        offset=CameraCfg.OffsetCfg(pos=(0.4, 0.7, 0.8), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, focus_distance=1.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)),
        data_types=["rgb"],
    )
    cfg.camera_wrist = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera", width=640, height=480,
        offset=CameraCfg.OffsetCfg(pos=(0.06, 0.0, 0.0), rot=(float(q_new_wxyz[0]), float(q_new_wxyz[1]), float(q_new_wxyz[2]), float(q_new_wxyz[3])), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, focus_distance=0.5, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)),
        data_types=["rgb"],
    )
    return cfg


# ── Helpers (same as online iface script) ──
def _rgb_to_uint8(rgb_tensor, env_id: int = 0) -> np.ndarray:
    rgb = rgb_tensor[env_id]
    if rgb.is_cuda:
        rgb = rgb.cpu()
    arr = rgb.numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _hwc_to_chw84(frame_hwc: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(frame_hwc)
    if t.dtype != torch.uint8:
        t = t.clamp(0, 255).to(torch.uint8)
    if t.shape[-1] == 4:
        t = t[..., :3]
    t = t.permute(2, 0, 1).contiguous().float().unsqueeze(0)
    t = F.interpolate(t, size=(84, 84), mode="bilinear", align_corners=False)
    return t.squeeze(0).to(torch.uint8).cpu().numpy()


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    q_xyzw = Rotation.from_euler("z", yaw, degrees=False).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def _resolve_embodiment_tag(tag_str: str):
    for member in EmbodimentTag:
        if str(member.value) == str(tag_str):
            return member
    raise ValueError(f"Unknown embodiment_tag='{tag_str}'")


class IfaceEnvWrapper:
    """Wraps IsaacLab sim + GR00T into a per-step env using EXACTLY the online iface loop.

    Supports num_envs >= 1 for batched parallel rollouts with batched GR00T inference.
    This creates its own scene (not FrankaPickupEnv) to guarantee identical physics.
    """

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        csv_dir: str,
        groot_model_path: str,
        embodiment_tag: str = "new_embodiment",
        policy_device: str = "cuda:0",
        policy_strict: bool = True,
        task_description: str = "Pick up the white object.",
        language_override: str | None = None,
        target_action_fps: float = 20.0,
        rot_clamp_rad: float = 0.35,
        camera_warmup_steps: int = 10,
        post_reset_settle_steps: int = 200,
        max_episode_steps: int = 1000,
        image_size: tuple[int, int] = (84, 84),
        num_envs: int = 1,
        success_threshold: float = 0.005,
        cube_perturb_range: float = 0.0,
        cube_perturb_table_path: str | None = None,
        random_cube_perturb: bool = False,
        random_cube_xy_range: float = 0.05,   # ±5cm (reduced from ±10cm)
        random_cube_yaw_range_deg: float = 10.0,  # ±10° (reduced from ±15°)
        reward_type: str = "sparse",  # "sparse", "dense", "dense_clipped"
        use_depth: bool = False,  # Enable depth observations for residual actor
        depth_norm_path: str | None = None,  # Path to depth_normalization.json
        use_state: bool = False,  # Enable object state (cube 6D pose) observations
    ):
        self.sim = sim
        self.csv_dir = Path(csv_dir).expanduser().resolve()
        self.reward_type = reward_type
        self.use_depth = use_depth
        self.use_state = use_state
        self.language_override = language_override
        self.random_cube_perturb = random_cube_perturb
        self.random_cube_xy_range = random_cube_xy_range
        self.random_cube_yaw_range_deg = random_cube_yaw_range_deg
        self.task_description = task_description
        self.target_action_fps = target_action_fps
        self.rot_clamp_rad = rot_clamp_rad
        self.camera_warmup_steps = camera_warmup_steps
        self.post_reset_settle_steps = post_reset_settle_steps
        self.max_episode_steps = max_episode_steps
        self.image_size = image_size
        self.device = sim.device
        self.num_envs = num_envs
        self.success_threshold = success_threshold
        self.cube_perturb_range = cube_perturb_range

        # Load fixed cube perturbation table (if provided)
        # Priority: JSON table > unit table * range > no perturbation
        self._cube_offsets = torch.zeros(num_envs, 2, device=sim.device)
        if cube_perturb_table_path:
            import json as _json
            with open(cube_perturb_table_path) as f:
                table = _json.load(f)
            for entry in table["envs"]:
                eid = entry["env_id"]
                if eid < num_envs:
                    self._cube_offsets[eid, 0] = entry["dx"]
                    self._cube_offsets[eid, 1] = entry["dy"]
            print(f"[IfaceEnvWrapper] Loaded cube perturbation table ({len(table['envs'])} entries) from {cube_perturb_table_path}")
            for eid in range(min(num_envs, len(table["envs"]))):
                e = table["envs"][eid]
                print(f"  env {eid}: {e['name']} dx={e['dx']:+.3f} dy={e['dy']:+.3f}")
        elif cube_perturb_range > 0 and num_envs > 1:
            _PERTURB_TABLE_UNIT = [
                (0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7), (1.0, 0.5),
            ]
            for eid in range(num_envs):
                dx, dy = _PERTURB_TABLE_UNIT[eid % len(_PERTURB_TABLE_UNIT)]
                self._cube_offsets[eid, 0] = dx * cube_perturb_range
                self._cube_offsets[eid, 1] = dy * cube_perturb_range
            print(f"[IfaceEnvWrapper] Using unit perturbation table * {cube_perturb_range}m")
        if self.random_cube_perturb:
            print(f"[IfaceEnvWrapper] Random cube perturbation ENABLED: XY=±{self.random_cube_xy_range*100:.0f}cm, Yaw=±{self.random_cube_yaw_range_deg:.0f}°")
        N = num_envs

        # ── Build scene ──
        scene_cfg = _make_scene_cfg(num_envs=N, env_spacing=2.0)
        # Patch cameras to also output depth if use_depth mode
        if self.use_depth:
            scene_cfg.camera_front.data_types = ["rgb", "depth"]
            scene_cfg.camera_back.data_types = ["rgb", "depth"]
            scene_cfg.camera_wrist.data_types = ["rgb", "depth"]
            print("[IfaceEnvWrapper] Depth mode ON: cameras patched to output RGB + Depth")
        self.scene = InteractiveScene(scene_cfg)
        sim.reset()

        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        self.camera_front = self.scene["camera_front"]
        self.camera_back = self.scene["camera_back"]
        self.camera_wrist = self.scene["camera_wrist"]
        self.sim_dt = sim.get_physics_dt()
        self.steps_per_action = max(1, int(round((1.0 / target_action_fps) / self.sim_dt)))

        # ── Load GR00T ──
        if Gr00tPolicy is None:
            raise ImportError("gr00t not found")
        resolved_tag = _resolve_embodiment_tag(embodiment_tag)
        self.groot_policy = Gr00tPolicy(
            embodiment_tag=resolved_tag, model_path=groot_model_path,
            device=policy_device, strict=policy_strict,
        )
        mod_cfg = self.groot_policy.get_modality_config()
        self.video_modality_keys = list(getattr(mod_cfg["video"], "modality_keys", VIDEO_KEYS))
        self.state_modality_keys = list(getattr(mod_cfg["state"], "modality_keys", [STATE_JOINT_KEY, STATE_GRIPPER_KEY]))
        lang_keys = list(getattr(mod_cfg["language"], "modality_keys", [LANGUAGE_KEY]))
        self.language_key = lang_keys[0] if lang_keys else LANGUAGE_KEY
        self.action_horizon = len(mod_cfg["action"].delta_indices)
        self.effective_horizon = self.action_horizon
        self.inference_interval = self.steps_per_action * self.effective_horizon
        print(f"[IfaceEnvWrapper] num_envs={N}, steps_per_action={self.steps_per_action}, "
              f"action_horizon={self.action_horizon}, inference_interval={self.inference_interval}")

        # ── Per-env IK controllers (batched IK has shape issues, use per-env) ──
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_cfg.resolve(self.scene)
        self.robot_entity_cfg = robot_entity_cfg
        self.diff_iks: list[DifferentialIKController] = []
        for _ in range(N):
            ik = DifferentialIKController(
                DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
                num_envs=1, device=self.device,
            )
            ik.reset()
            self.diff_iks.append(ik)
        # Backward compat alias for code that uses self.diff_ik
        if N == 1:
            self.diff_ik = self.diff_iks[0]

        # Joint IDs
        self.arm_joint_names = sorted([n for n in self.robot.data.joint_names if "panda_joint" in n and "finger" not in n])
        self.arm_joint_ids = [self.robot.data.joint_names.index(n) for n in self.arm_joint_names]
        self.hand_joint_names = sorted([n for n in self.robot.data.joint_names if "panda_finger_joint" in n])
        self.hand_joint_ids = [self.robot.data.joint_names.index(n) for n in self.hand_joint_names]
        self.all_joint_ids = self.arm_joint_ids + self.hand_joint_ids

        # Friction boost (same as online)
        robot_materials = self.robot.root_physx_view.get_material_properties().to("cpu")
        robot_materials[..., 0] = 5e3
        robot_materials[..., 1] = 5e3
        self.robot.root_physx_view.set_material_properties(robot_materials, torch.arange(N, device="cpu", dtype=torch.long))
        print(f"[IfaceEnvWrapper] Friction boost 5000 applied to {N} envs")

        # ── Finger body indices for contact reward ──
        self.left_finger_body_idx = self.robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_idx = self.robot.find_bodies("panda_rightfinger")[0][0]
        self.finger_body_indices = [self.left_finger_body_idx, self.right_finger_body_idx]

        # ── PhysX contact view (initialized lazily after first sim step) ──
        self._contact_physx_view = None
        self._contact_view_initialized = False
        self._contact_init_attempts = 0
        self.left_finger_force = torch.zeros((N, 3), device=self.device)
        self.left_finger_pos_w = torch.zeros((N, 3), device=self.device)
        self.right_finger_pos_w = torch.zeros((N, 3), device=self.device)
        self.finger_pos = torch.zeros((N, 3), device=self.device)
        self._ever_contacted = torch.zeros(N, device=self.device, dtype=torch.bool)  # cumulative contact flag
        self._contact_history = torch.zeros(N, 3, device=self.device, dtype=torch.bool)  # last 3 steps contact state

        # ── Gym spaces (match original IsaacLabVecEnvWrapper: 34D state) ──
        # state = [dof_pos_scaled(9), dof_vel_scaled(9), cogact_reference(7), contact_obs(3), ee_xyzrpy(6)]
        self._state_dim = 10  # eef_pos(3) + eef_quat(4) + gripper_qpos(2) + contact_force(1)
        self._vlm_latent_dim = 2048  # GR00T VLM backbone hidden dim (raw, before projection)
        self.action_dim = 7
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(N, 7), dtype=np.float32)
        obs_spaces = {
            "observation.state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N, self._state_dim), dtype=np.float32),
            "observation.base_action": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N, 7), dtype=np.float32),
            "observation.vlm_latent": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N, self._vlm_latent_dim), dtype=np.float32),
            "observation.images.front": gym.spaces.Box(low=0, high=255, shape=(N, 3, 84, 84), dtype=np.uint8),
            "observation.images.back": gym.spaces.Box(low=0, high=255, shape=(N, 3, 84, 84), dtype=np.uint8),
            "observation.images.wrist": gym.spaces.Box(low=0, high=255, shape=(N, 3, 84, 84), dtype=np.uint8),
        }
        # Depth observations (1-channel, float32 [0,1] normalized)
        if self.use_depth:
            obs_spaces["observation.depth.front"] = gym.spaces.Box(low=0, high=1, shape=(N, 1, 84, 84), dtype=np.float32)
            obs_spaces["observation.depth.wrist"] = gym.spaces.Box(low=0, high=1, shape=(N, 1, 84, 84), dtype=np.float32)
            # Load depth normalization config
            self._depth_norm = {"front": {"min": 0.3, "max": 1.8}, "wrist": {"min": 0.03, "max": 0.4}}
            if depth_norm_path and Path(depth_norm_path).exists():
                import json as _json
                self._depth_norm = _json.loads(Path(depth_norm_path).read_text())
                print(f"[IfaceEnvWrapper] Depth normalization loaded from {depth_norm_path}:")
                for k, v in self._depth_norm.items():
                    if isinstance(v, dict):
                        print(f"  {k}: min={v.get('min')}, max={v.get('max')}")
            else:
                print(f"[IfaceEnvWrapper] Using default depth normalization (no file at {depth_norm_path}):")
                for k, v in self._depth_norm.items():
                    if isinstance(v, dict):
                        print(f"  {k}: min={v.get('min')}, max={v.get('max')}")
            print(f"[IfaceEnvWrapper] Depth obs spaces added: observation.depth.front, observation.depth.wrist")
        # Object state (cube 6D pose): pos(3) + quat_wxyz(4) = 7D
        self._object_state_dim = 7
        if self.use_state:
            obs_spaces["observation.object_state"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(N, self._object_state_dim), dtype=np.float32)
            print(f"[IfaceEnvWrapper] Object state mode ON: observation.object_state ({self._object_state_dim}D)")
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # ── VLM latent cache (updated on GR00T inference, cached between) ──
        self._cached_vlm_latent = torch.zeros(N, self._vlm_latent_dim, device=self.device, dtype=torch.float32)

        # ── Active env control (for train=1 / eval=N) ──
        self._active_env_ids: list[int] = list(range(N))

        # ── Per-env internal state ──
        self._step_count = torch.zeros(N, device=self.device, dtype=torch.long)
        self._steps_since_infer = [self.inference_interval] * N  # trigger inference on first step
        self._action_windows: list[np.ndarray | None] = [None] * N
        self._current_window_idx = [0] * N
        self._current_action_7 = [np.zeros(7, dtype=np.float32) for _ in range(N)]
        self._last_grip_open = [1.0] * N
        self._last_combined_naction = [np.zeros(7, dtype=np.float32) for _ in range(N)]
        self._smoothed_dp: list[torch.Tensor | None] = [None] * N
        self._smoothed_drot: list[torch.Tensor | None] = [None] * N
        self._joint_pos_des: torch.Tensor | None = None  # set in reset
        self._initial_cube_z = torch.zeros(N, device=self.device)
        self._max_cube_z = torch.zeros(N, device=self.device)
        self._env_ids_gpu = torch.arange(N, device=self.device, dtype=torch.long)
        self._env_origins = self.scene.env_origins[self._env_ids_gpu]

    # ─────────────────────────────────────────────────────────────────
    # reset
    # ─────────────────────────────────────────────────────────────────
    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Full reset: CSV init + camera warmup + settle (same as online) for all envs."""
        N = self.num_envs
        sim_dt = self.sim_dt
        csv_dir = self.csv_dir

        # Camera warmup
        for _ in range(self.camera_warmup_steps):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(sim_dt)
            for cam in (self.camera_front, self.camera_back, self.camera_wrist):
                cam.update(dt=sim_dt)

        # Camera poses (per-env, offset by env origins)
        eye_front = torch.tensor([[0.4, -0.7, 0.8]], device=self.device).repeat(N, 1) + self._env_origins
        eye_back = torch.tensor([[0.4, 0.7, 0.8]], device=self.device).repeat(N, 1) + self._env_origins
        target_pos = torch.tensor([[0.25, 0.0, -0.05]], device=self.device).repeat(N, 1) + self._env_origins
        self.camera_front.set_world_poses_from_view(eye_front, target_pos)
        self.camera_back.set_world_poses_from_view(eye_back, target_pos)
        self._eye_front = eye_front
        self._eye_back = eye_back
        self._target_pos = target_pos

        # CSV init (same for all envs)
        js = pd.read_csv(csv_dir / "franka_joint_states.csv", header=0).values.astype(np.float32)
        gs = pd.read_csv(csv_dir / "gripper_joint_states.csv", header=0).values.astype(np.float32)
        js = js[1:, 10:17]
        gs = gs[1:, 4:5]
        arm_init = js[1, :7]
        grip_width = float(np.clip(float(gs[1, 0]), 0.0, 1.0))
        finger_pos = grip_width * 0.04
        q_init = np.concatenate([arm_init, [finger_pos] * len(self.hand_joint_ids)]).astype(np.float32)
        q_init_t = torch.tensor(q_init, device=self.device).unsqueeze(0).repeat(N, 1)
        qd_init_t = torch.zeros_like(q_init_t)
        self.robot.write_joint_state_to_sim(q_init_t, qd_init_t, joint_ids=self.all_joint_ids)
        self.robot.set_joint_position_target(q_init_t, joint_ids=self.all_joint_ids)

        # Cube init
        object_pos_csv = csv_dir / "object_pos.csv"
        if object_pos_csv.exists():
            od = pd.read_csv(object_pos_csv, header=0).values.astype(np.float32)
            obj = od[0, 3:7].copy()
        else:
            obj = np.array([0.2, 0.0, 0.06, 0.0], dtype=np.float32)
        obj_pos_w = self._env_origins + torch.tensor(obj[:3], device=self.device).unsqueeze(0)
        # Apply fixed per-env cube XY offsets
        obj_pos_w[:, 0] += self._cube_offsets[:, 0]
        obj_pos_w[:, 1] += self._cube_offsets[:, 1]
        # Random perturbation on reset() too (warmup gets diverse positions)
        if self.random_cube_perturb:
            for eid in range(N):
                rand_dx = (torch.rand(1, device=self.device).item() * 2 - 1) * self.random_cube_xy_range
                rand_dy = (torch.rand(1, device=self.device).item() * 2 - 1) * self.random_cube_xy_range
                obj_pos_w[eid, 0] += rand_dx
                obj_pos_w[eid, 1] += rand_dy
            print(f"[RandomPerturb] reset(): applied random XY to {N} envs")
        default_rot = self.cube.data.default_root_state[0, 3:7].cpu().numpy()
        base_yaw = float(obj[3])
        obj_quat_list = []
        for eid in range(N):
            yaw = base_yaw
            if self.random_cube_perturb:
                yaw += (np.random.rand() * 2 - 1) * np.deg2rad(self.random_cube_yaw_range_deg)
            yaw_quat = _yaw_to_quat_wxyz(yaw)
            base_r = Rotation.from_quat([default_rot[1], default_rot[2], default_rot[3], default_rot[0]])
            yaw_r = Rotation.from_quat([yaw_quat[1], yaw_quat[2], yaw_quat[3], yaw_quat[0]])
            cq = (yaw_r * base_r).as_quat()
            obj_quat_list.append(torch.tensor([cq[3], cq[0], cq[1], cq[2]], device=self.device, dtype=torch.float32))
        obj_quat_w = torch.stack(obj_quat_list)
        self.cube.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=self._env_ids_gpu)

        # Settle
        for _ in range(self.post_reset_settle_steps):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(sim_dt)
            self.robot.set_joint_position_target(q_init_t, joint_ids=self.all_joint_ids)
            for cam in (self.camera_front, self.camera_back, self.camera_wrist):
                cam.update(dt=sim_dt)

        self._initial_cube_z = self.cube.data.root_state_w[:N, 2].clone()
        self._max_cube_z = self._initial_cube_z.clone()

        # Reset per-env internal state
        self._step_count[:] = 0
        self._ever_contacted[:] = False
        self._contact_history[:] = False
        self._joint_pos_des = q_init_t.clone()
        for eid in range(N):
            self._steps_since_infer[eid] = self.inference_interval  # trigger inference on first step
            self._action_windows[eid] = None
            self._current_window_idx[eid] = 0
            self._current_action_7[eid] = np.zeros(7, dtype=np.float32)
            self._last_grip_open[eid] = 1.0
            self._last_combined_naction[eid] = np.zeros(7, dtype=np.float32)
            self._smoothed_dp[eid] = None
            self._smoothed_drot[eid] = None
            self.diff_iks[eid].reset()

        obs = self._build_obs()
        return obs, {}

    # ─────────────────────────────────────────────────────────────────
    # Active env control
    # ─────────────────────────────────────────────────────────────────
    def set_active_env_ids(self, ids: list[int]):
        """Set which envs are active (GR00T inference + IK control).

        During training: set to [0] for single-env training speed.
        During eval: set to list(range(N)) for batched evaluation.
        """
        self._active_env_ids = list(ids)

    def reset_envs(self, env_ids: list[int]):
        """Reset specific envs only (robot + cube poses from CSV). Does NOT reset other envs.

        This is used to reset eval envs without disturbing the training env.
        Only resets joint state, cube pose, and per-env internal state for the given IDs.
        Runs settle steps with sim.step() which advances ALL envs, but only the specified
        envs get their state overwritten.
        """
        csv_dir = self.csv_dir
        sim_dt = self.sim_dt
        eids = env_ids
        N_reset = len(eids)
        eid_tensor = torch.tensor(eids, device=self.device, dtype=torch.long)

        # CSV init
        js = pd.read_csv(csv_dir / "franka_joint_states.csv", header=0).values.astype(np.float32)
        gs = pd.read_csv(csv_dir / "gripper_joint_states.csv", header=0).values.astype(np.float32)
        js = js[1:, 10:17]
        gs = gs[1:, 4:5]
        arm_init = js[1, :7]
        grip_width = float(np.clip(float(gs[1, 0]), 0.0, 1.0))
        finger_pos = grip_width * 0.04
        q_init = np.concatenate([arm_init, [finger_pos] * len(self.hand_joint_ids)]).astype(np.float32)
        q_init_t = torch.tensor(q_init, device=self.device).unsqueeze(0).repeat(N_reset, 1)
        qd_init_t = torch.zeros_like(q_init_t)

        # Write joint state only for these envs
        full_q = self.robot.data.joint_pos.clone()
        full_qd = self.robot.data.joint_vel.clone()
        for i, eid in enumerate(eids):
            for j, jid in enumerate(self.all_joint_ids):
                full_q[eid, jid] = q_init_t[i, j]
                full_qd[eid, jid] = 0.0
        self.robot.write_joint_state_to_sim(full_q, full_qd, joint_ids=self.all_joint_ids)

        # Set joint targets for these envs
        full_target = self.robot.data.joint_pos_target.clone()
        for i, eid in enumerate(eids):
            for j, jid in enumerate(self.all_joint_ids):
                full_target[eid, jid] = q_init_t[i, j]
        self.robot.set_joint_position_target(full_target, joint_ids=self.all_joint_ids)

        # Cube init for these envs
        object_pos_csv = csv_dir / "object_pos.csv"
        if object_pos_csv.exists():
            od = pd.read_csv(object_pos_csv, header=0).values.astype(np.float32)
            obj = od[0, 3:7].copy()
        else:
            obj = np.array([0.2, 0.0, 0.06, 0.0], dtype=np.float32)
        origins = self.scene.env_origins[eid_tensor]
        obj_pos_w = origins + torch.tensor(obj[:3], device=self.device).unsqueeze(0)
        # Apply fixed per-env cube XY offsets
        for i, eid in enumerate(eids):
            obj_pos_w[i, 0] += self._cube_offsets[eid, 0]
            obj_pos_w[i, 1] += self._cube_offsets[eid, 1]
            # Random perturbation: add random XY offset + random yaw on each reset
            if self.random_cube_perturb:
                rand_dx = (torch.rand(1, device=self.device).item() * 2 - 1) * self.random_cube_xy_range
                rand_dy = (torch.rand(1, device=self.device).item() * 2 - 1) * self.random_cube_xy_range
                obj_pos_w[i, 0] += rand_dx
                obj_pos_w[i, 1] += rand_dy
                if not hasattr(self, '_random_perturb_log_count'):
                    self._random_perturb_log_count = 0
                if self._random_perturb_log_count < 5:
                    print(f"[RandomPerturb] env={eid} dx={rand_dx:+.3f} dy={rand_dy:+.3f}")
                    self._random_perturb_log_count += 1
        default_rot = self.cube.data.default_root_state[0, 3:7].cpu().numpy()
        # Per-env yaw: base yaw from CSV + optional random yaw
        base_yaw = float(obj[3])
        obj_quat_list = []
        for i, eid in enumerate(eids):
            yaw = base_yaw
            if self.random_cube_perturb:
                yaw += (np.random.rand() * 2 - 1) * np.deg2rad(self.random_cube_yaw_range_deg)
            yaw_quat = _yaw_to_quat_wxyz(yaw)
            base_r = Rotation.from_quat([default_rot[1], default_rot[2], default_rot[3], default_rot[0]])
            yaw_r = Rotation.from_quat([yaw_quat[1], yaw_quat[2], yaw_quat[3], yaw_quat[0]])
            cq = (yaw_r * base_r).as_quat()
            obj_quat_list.append(torch.tensor([cq[3], cq[0], cq[1], cq[2]], device=self.device, dtype=torch.float32))
        obj_quat_w = torch.stack(obj_quat_list)
        self.cube.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=eid_tensor)

        # Settle (short — fewer steps than full reset since sim is already warm)
        settle_steps = min(self.post_reset_settle_steps, 100)
        for _ in range(settle_steps):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(sim_dt)
            self.robot.set_joint_position_target(full_target, joint_ids=self.all_joint_ids)
            for cam in (self.camera_front, self.camera_back, self.camera_wrist):
                cam.update(dt=sim_dt)

        # Reset per-env internal state for these envs only
        for eid in eids:
            self._initial_cube_z[eid] = self.cube.data.root_state_w[eid, 2].clone()
            self._max_cube_z[eid] = self._initial_cube_z[eid].clone()
            self._steps_since_infer[eid] = self.inference_interval  # trigger inference
            self._action_windows[eid] = None
            self._current_window_idx[eid] = 0
            self._current_action_7[eid] = np.zeros(7, dtype=np.float32)
            self._last_grip_open[eid] = 1.0
            self._last_combined_naction[eid] = np.zeros(7, dtype=np.float32)
            self._smoothed_dp[eid] = None
            self._smoothed_drot[eid] = None
            self.diff_iks[eid].reset()
            # Reset joint pos des for this env
            self._joint_pos_des[eid] = q_init_t[0].clone()

        for eid in eids:
            self._step_count[eid] = 0  # reset step counter per-env
            self._ever_contacted[eid] = False  # reset contact flag per-env
            self._contact_history[eid] = False

    # ─────────────────────────────────────────────────────────────────
    # step
    # ─────────────────────────────────────────────────────────────────
    def step(self, residual_action: torch.Tensor | None = None):
        """One sim step for all envs. Runs batched GR00T inference when needed, applies action + optional residual."""
        N = self.num_envs
        sim_dt = self.sim_dt
        # Use min step count of active envs for control tick timing
        count = int(self._step_count[self._active_env_ids[0]].item()) if self._active_env_ids else 0

        # Update cameras
        for cam in (self.camera_front, self.camera_back, self.camera_wrist):
            cam.update(dt=sim_dt)
        try:
            self.camera_front.set_world_poses_from_view(self._eye_front, self._target_pos)
            self.camera_back.set_world_poses_from_view(self._eye_back, self._target_pos)
        except Exception:
            pass

        # GR00T inference for active envs that need it (batched)
        active = self._active_env_ids

        # SNAPSHOT: Save pre-inference base_action for each env.
        # This is the value _build_obs() returned to the actor, and must be
        # used in the control tick below to guarantee exact match.
        pre_step_base_action = {eid: self._current_action_7[eid].copy() for eid in active}

        need_infer = [eid for eid in active if self._steps_since_infer[eid] >= self.inference_interval]
        if need_infer:
            self._run_groot_inference(need_infer)
            for eid in need_infer:
                self._steps_since_infer[eid] = 0
        for eid in active:
            self._steps_since_infer[eid] += 1

        # Control tick logic (active envs only)
        # Uses pre_step_base_action (the value that _build_obs() returned),
        # NOT self._current_action_7 (which may have been updated by GR00T inference above).
        control_tick = (count % self.steps_per_action) == 0
        for eid in active:
            if self._action_windows[eid] is None:
                continue

            # ALWAYS update _last_combined_naction (for info["scaled_action"])
            # This ensures action stored in replay buffer matches obs.base_action + residual
            interp_for_store = pre_step_base_action[eid]
            ba_for_store = np.concatenate([
                interp_for_store[:3] / 0.02,
                interp_for_store[3:6] / 0.2,
                [interp_for_store[6] * 2.0 - 1.0]
            ]).astype(np.float32)
            if residual_action is not None:
                res_for_store = residual_action[eid].detach().cpu().numpy().astype(np.float32)
                self._last_combined_naction[eid] = np.clip(ba_for_store + res_for_store, -1.0, 1.0)
            else:
                self._last_combined_naction[eid] = ba_for_store.copy()

            if control_tick:
                # Use the SNAPSHOTTED base action for IK execution
                interp = pre_step_base_action[eid]
                dp_raw = interp[:3].astype(np.float32)
                drot_raw = interp[3:6].astype(np.float32)
                grip_raw = float(interp[6])

                # Compute combined in normalized space (reuse ba_for_store)
                combined_naction = self._last_combined_naction[eid]  # already set above

                # Unscale back to raw action space
                dp = combined_naction[:3] * 0.02
                drot = combined_naction[3:6] * 0.2
                grip = (combined_naction[6] + 1.0) / 2.0  # back to [0, 1]

                rot_norm = float(np.linalg.norm(drot))
                if rot_norm > self.rot_clamp_rad and rot_norm > 1e-6:
                    drot = drot * (self.rot_clamp_rad / rot_norm)
                grip = float(np.clip(grip, 0.0, 1.0))

                # EMA smoothing (α=0.35)
                if self._smoothed_dp[eid] is None:
                    self._smoothed_dp[eid] = torch.tensor(dp, device=self.device).unsqueeze(0)
                    self._smoothed_drot[eid] = torch.tensor(drot, device=self.device).unsqueeze(0)
                smooth_a = 0.35
                dp_t = torch.tensor(dp, device=self.device).unsqueeze(0)
                drot_t = torch.tensor(drot, device=self.device).unsqueeze(0)
                self._smoothed_dp[eid] = (1.0 - smooth_a) * self._smoothed_dp[eid] + smooth_a * dp_t
                self._smoothed_drot[eid] = (1.0 - smooth_a) * self._smoothed_drot[eid] + smooth_a * drot_t

                # IK target
                ee_pose_w = self.robot.data.body_state_w[eid:eid+1, self.robot_entity_cfg.body_ids[0], :7]
                ee_pos_w = ee_pose_w[:, :3]
                ee_quat_w = ee_pose_w[:, 3:7]
                target_pos_w = ee_pos_w + self._smoothed_dp[eid]
                dq_xyzw = Rotation.from_rotvec(self._smoothed_drot[eid][0].detach().cpu().numpy()).as_quat()
                dq_wxyz = np.array([dq_xyzw[3], dq_xyzw[0], dq_xyzw[1], dq_xyzw[2]], dtype=np.float32)
                dq = torch.tensor(dq_wxyz, device=self.device).unsqueeze(0)
                target_quat_w = _quat_mul_wxyz(dq, ee_quat_w)
                target_quat_w = target_quat_w / target_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                target_pose_w = torch.cat([target_pos_w, target_quat_w], dim=-1)
                self.diff_iks[eid].set_command(target_pose_w)
                self._last_grip_open[eid] = float(np.clip(grip, 0.0, 1.0))

                # ADVANCE: prepare _current_action_7 for the NEXT _build_obs() call.
                # After executing the current base_action, move window forward.
                wi = self._current_window_idx[eid]
                if wi < len(self._action_windows[eid]) - 1:
                    self._current_window_idx[eid] += 1
                self._current_action_7[eid] = self._action_windows[eid][self._current_window_idx[eid]].astype(np.float32)

            # IK compute every sim step (per-env)
            ee_pose_w = self.robot.data.body_state_w[eid:eid+1, self.robot_entity_cfg.body_ids[0], :7]
            jacobian = self.robot.root_physx_view.get_jacobians()[eid:eid+1, self.robot_entity_cfg.body_ids[0] - 1, :, :]
            jacobian = jacobian[:, :, self.arm_joint_ids]
            joint_pos = self.robot.data.joint_pos[eid:eid+1][:, self.arm_joint_ids]
            joint_pos_arm = self.diff_iks[eid].compute(ee_pose_w[:, :3], ee_pose_w[:, 3:7], jacobian, joint_pos)

            prev = self.robot.data.joint_pos_target[eid:eid+1][:, self.arm_joint_ids]
            delta = torch.clamp(joint_pos_arm - prev, -0.05, 0.05)
            joint_pos_arm = prev + delta

            finger = float(self._last_grip_open[eid]) * 0.04
            joint_pos_gripper = torch.full((1, len(self.hand_joint_ids)), finger, device=self.device)
            self._joint_pos_des[eid] = torch.cat([joint_pos_arm[0], joint_pos_gripper[0]])

        # Apply targets + sim step (all envs at once)
        self.robot.set_joint_position_target(self._joint_pos_des, joint_ids=self.all_joint_ids)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(sim_dt)

        # Per-env shaped reward (distance + contact + height + success)
        N = self.num_envs
        cube_pos = self.cube.data.root_state_w[:N, :3]
        cube_z = cube_pos[:, 2]
        cube_half_size = 0.02
        finger_length = 0.06

        # Update finger positions
        self.left_finger_pos_w = self.robot.data.body_pos_w[:N, self.left_finger_body_idx]
        self.right_finger_pos_w = self.robot.data.body_pos_w[:N, self.right_finger_body_idx]
        self.finger_pos = (self.left_finger_pos_w + self.right_finger_pos_w) / 2.0

        # Compute contact forces
        self._compute_contact_forces()

        # ===== Reward computation (configurable) =====
        cube_height = cube_z - self._initial_cube_z

        # Contact force check: gripper must be exerting force on something
        finger_contact_force = self.left_finger_force[:N].norm(dim=-1)  # (N,)
        has_contact = (finger_contact_force > 0.1).float()  # threshold 0.1N

        # ── Smoothed contact state (3-step window) ──
        # HELD = contact was ON for at least 1 of the last 3 steps (including current)
        # LOST = contact was OFF for all of the last 3 steps
        # This smooths over brief 1-2 frame contact sensor gaps.
        self._contact_history = torch.cat([
            self._contact_history[:, 1:],  # shift left (drop oldest)
            (has_contact > 0.5).unsqueeze(1),  # append current
        ], dim=1)  # (N, 3)
        contact_smoothed = self._contact_history.any(dim=1)  # True if any of last 3 had contact

        # Track ever-contacted and lifting for diagnostics
        self._ever_contacted = self._ever_contacted | (has_contact > 0.5)

        if self.reward_type == "sparse":
            # Binary: 1.0 if success, 0.0 otherwise (original ResFit)
            rewards = (cube_height >= self.success_threshold).float()

        elif self.reward_type == "dense":
            # 4-stage shaped reward (distance + contact + height + success)
            # grasp_gate = has_contact (simplified: contact force is the most reliable signal)
            finger_cube_dist = torch.norm(self.finger_pos - cube_pos, dim=-1)
            distance_reward = (1.0 - torch.tanh(finger_cube_dist / 0.1)) * 1.0
            left_finger_cube_dist = torch.norm(self.left_finger_pos_w - cube_pos, dim=-1)
            right_finger_cube_dist = torch.norm(self.right_finger_pos_w - cube_pos, dim=-1)
            both_close = ((left_finger_cube_dist < 0.05) & (right_finger_cube_dist < 0.05)).float()
            finger_joint_pos = self.robot.data.joint_pos[:N][:, self.hand_joint_ids[0]]
            gripper_closing = (finger_joint_pos < 0.04).float()
            grasp_gate = has_contact  # contact force alone gates reward
            contact_reward = grasp_gate * 2.0
            height_reward = (cube_height > 0.005).float() * torch.tanh(cube_height / 0.1) * 100.0 * grasp_gate
            success_reward = (cube_height >= self.success_threshold).float() * 100.0 * grasp_gate
            rewards = distance_reward + contact_reward + height_reward + success_reward

        elif self.reward_type == "dense_clipped":
            # Dense shaped reward, normalized to [0, 1] range via clipping
            # grasp_gate = has_contact (simplified: contact force is the most reliable signal)
            finger_cube_dist = torch.norm(self.finger_pos - cube_pos, dim=-1)
            distance_reward = (1.0 - torch.tanh(finger_cube_dist / 0.1)) * 0.1
            left_finger_cube_dist = torch.norm(self.left_finger_pos_w - cube_pos, dim=-1)
            right_finger_cube_dist = torch.norm(self.right_finger_pos_w - cube_pos, dim=-1)
            both_close = ((left_finger_cube_dist < 0.05) & (right_finger_cube_dist < 0.05)).float()
            finger_joint_pos = self.robot.data.joint_pos[:N][:, self.hand_joint_ids[0]]
            gripper_closing = (finger_joint_pos < 0.04).float()
            grasp_gate = has_contact  # contact force alone gates reward
            contact_reward = grasp_gate * 0.2
            height_reward = (cube_height > 0.005).float() * torch.tanh(cube_height / 0.1) * 0.5 * grasp_gate
            success_reward = (cube_height >= self.success_threshold).float() * 1.0 * grasp_gate
            rewards = torch.clamp(distance_reward + contact_reward + height_reward + success_reward, 0.0, 1.0)

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        self._step_count += 1

        obs = self._build_obs()
        # Success: cube lifted to threshold AND smoothed contact is ON
        # contact_smoothed uses 3-step window: HELD if any contact in last 3 steps
        sustained_contact = contact_smoothed  # for debug overlay
        success_flag = (cube_height >= self.success_threshold) & contact_smoothed
        time_limit = (self._step_count >= self.max_episode_steps)
        terminated = success_flag | time_limit
        truncated = torch.zeros(N, device=self.device, dtype=torch.bool)
        info = {"scaled_action": torch.tensor(
            np.stack([self._last_combined_naction[eid] for eid in range(N)]),
            device=self.device, dtype=torch.float32,
        )}

        # Store per-step debug info for visualization
        finger_cube_dist = torch.norm(self.finger_pos - cube_pos, dim=-1)
        left_finger_cube_dist = torch.norm(self.left_finger_pos_w - cube_pos, dim=-1)
        right_finger_cube_dist = torch.norm(self.right_finger_pos_w - cube_pos, dim=-1)
        self._last_step_debug = {
            "contact_force": finger_contact_force.detach().cpu(),        # (N,)
            "has_contact": has_contact.detach().cpu(),                   # (N,)
            "cube_height": cube_height.detach().cpu(),                   # (N,)
            "reward": rewards.detach().cpu(),                            # (N,)
            "finger_cube_dist": finger_cube_dist.detach().cpu(),         # (N,)
            "success": success_flag.detach().cpu().float(),              # (N,)
            "sustained_contact": sustained_contact.detach().cpu().float(),  # (N,) smoothed 3-step
            "contact_lost_during_lift": (~contact_smoothed).detach().cpu().float(),  # (N,) inverse of smoothed
            "step": self._step_count,
        }
        # Store per-component rewards for debug visualization (dense / dense_clipped)
        if self.reward_type in ("dense", "dense_clipped"):
            self._last_step_debug["reward_distance"] = distance_reward.detach().cpu()
            self._last_step_debug["reward_contact"] = contact_reward.detach().cpu()
            self._last_step_debug["reward_height"] = height_reward.detach().cpu()
            self._last_step_debug["reward_success"] = success_reward.detach().cpu()
            self._last_step_debug["grasp_gate"] = grasp_gate.detach().cpu()
            # Sub-components of grasp_gate for diagnosis
            self._last_step_debug["both_close"] = both_close.detach().cpu()
            self._last_step_debug["gripper_closing"] = gripper_closing.detach().cpu()
            self._last_step_debug["left_finger_cube_dist"] = left_finger_cube_dist.detach().cpu()
            self._last_step_debug["right_finger_cube_dist"] = right_finger_cube_dist.detach().cpu()
            self._last_step_debug["finger_joint_pos"] = finger_joint_pos.detach().cpu()

        return obs, rewards, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────
    # Contact force computation (PhysX tensor API, same as Honda_IsaacLab)
    # ─────────────────────────────────────────────────────────────────
    def _initialize_contact_view(self):
        """Initialize PhysX contact view for finger bodies (called once after sim is live)."""
        try:
            from omni.physics.tensors import create_simulation_view
            import omni.physics.tensors.impl.api as physx
        except ImportError:
            try:
                import omni.physx.tensors.impl.api as physx
                from omni.physx.tensors import create_simulation_view
            except ImportError:
                print("[IfaceEnvWrapper] WARNING: Cannot import physx tensors API — contact reward disabled")
                self._contact_view_initialized = True  # mark as done to stop retrying
                return

        try:
            physics_sim_view = create_simulation_view("torch")
            physics_sim_view.set_subspace_roots("/")

            robot_prim = self.scene.cfg.robot.prim_path
            finger_pattern = f"{robot_prim.replace('.*', '*')}/*finger*"
            self._contact_physx_view = physics_sim_view.create_rigid_contact_view(finger_pattern)
            self._contact_view_initialized = True

            # Determine left/right ordering from prim paths
            prim_paths = [str(p) for p in self._contact_physx_view.prim_paths] if hasattr(self._contact_physx_view, 'prim_paths') else []
            self._finger_body_order = [0, 1]  # default: left=0, right=1
            for i, p in enumerate(prim_paths):
                if "leftfinger" in p.lower():
                    self._finger_body_order[0] = i % 2
                elif "rightfinger" in p.lower():
                    self._finger_body_order[1] = i % 2

            print(f"[IfaceEnvWrapper] Contact view initialized: {finger_pattern}")
        except Exception as e:
            self._contact_init_attempts += 1
            if self._contact_init_attempts >= 5:
                print(f"[IfaceEnvWrapper] WARNING: Contact view init failed after 5 attempts: {e}")
                self._contact_view_initialized = True  # stop retrying

    def _compute_contact_forces(self):
        """Read contact forces for finger bodies using Isaac Lab's ContactSensor."""
        N = self.num_envs
        try:
            contact_sensor = self.scene["contact_forces"]
            # net_forces_w: (N, num_bodies, 3)
            forces = contact_sensor.data.net_forces_w[:N]
            # Find finger body indices in the contact sensor
            if not hasattr(self, '_finger_sensor_ids'):
                body_names = contact_sensor.body_names
                self._finger_sensor_ids = []
                for i, name in enumerate(body_names):
                    if "finger" in name.lower():
                        self._finger_sensor_ids.append(i)
                if not self._finger_sensor_ids:
                    # Fallback: use last 2 bodies (likely fingers)
                    self._finger_sensor_ids = list(range(max(0, len(body_names)-2), len(body_names)))
                print(f"[IfaceEnvWrapper] Contact sensor finger body IDs: {self._finger_sensor_ids} from {body_names}")
            
            # Sum forces from all finger bodies
            if self._finger_sensor_ids:
                finger_forces = forces[:, self._finger_sensor_ids, :]  # (N, n_fingers, 3)
                self.left_finger_force[:] = finger_forces.sum(dim=1)  # (N, 3)
        except Exception as e:
            if not hasattr(self, '_contact_warn_printed'):
                print(f"[IfaceEnvWrapper] WARNING: Contact sensor read failed: {e}")
                self._contact_warn_printed = True

    # ─────────────────────────────────────────────────────────────────
    # Batched GR00T inference
    # ─────────────────────────────────────────────────────────────────
    def _run_groot_inference(self, env_ids_to_infer: list[int] | None = None):
        """Build obs and call GR00T with batched input for specified envs."""
        if env_ids_to_infer is None:
            env_ids_to_infer = list(range(self.num_envs))
        B = len(env_ids_to_infer)

        # Build batched video obs
        vid_batch: dict[str, np.ndarray] = {}
        for vk in self.video_modality_keys:
            frames = []
            for eid in env_ids_to_infer:
                lk = vk.lower()
                if "wrist" in lk or "ego" in lk:
                    f = _rgb_to_uint8(self.camera_wrist.data.output["rgb"], eid)
                elif "left" in lk:
                    f = _rgb_to_uint8(self.camera_back.data.output["rgb"], eid)
                else:
                    f = _rgb_to_uint8(self.camera_front.data.output["rgb"], eid)
                frames.append(f)
            vid_batch[vk] = np.stack(frames)[:, None, ...].astype(np.uint8)  # (B,1,H,W,3)

        # Build batched state obs
        state_batch: dict[str, np.ndarray] = {}
        for sk in self.state_modality_keys:
            vals = []
            for eid in env_ids_to_infer:
                if "joint" in sk.lower():
                    v = self.robot.data.joint_pos[eid, self.arm_joint_ids].detach().cpu().numpy().astype(np.float32)
                elif "gripper" in sk.lower():
                    gj = float(self.robot.data.joint_pos[eid, self.hand_joint_ids[0]].item()) if self.hand_joint_ids else 0.0
                    v = np.array([float(np.clip(gj / 0.04, 0.0, 1.0))], dtype=np.float32)
                else:
                    v = np.zeros(1, dtype=np.float32)
                vals.append(v)
            state_batch[sk] = np.stack(vals)[:, None, :].astype(np.float32)  # (B,1,D)

        # Language batch
        lang_str = self.language_override if self.language_override else self.task_description
        lang_batch = {self.language_key: [[str(lang_str)]] * B}

        obs = {"video": vid_batch, "state": state_batch, "language": lang_batch}
        pred_action, groot_info = self.groot_policy.get_action(obs)

        # Cache VLM backbone features (mean-pooled over seq_len → [B, 2048])
        if "backbone_features" in groot_info:
            bf = groot_info["backbone_features"]  # [B, seq_len, 2048] float32
            if "backbone_attention_mask" in groot_info:
                mask = groot_info["backbone_attention_mask"].unsqueeze(-1).float()  # [B, seq_len, 1]
                vlm_pooled = (bf * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, 2048]
            else:
                vlm_pooled = bf.mean(dim=1)  # [B, 2048]

            # ── VLM latent validation ──
            assert vlm_pooled.shape == (B, self._vlm_latent_dim), (
                f"VLM pooled shape {vlm_pooled.shape} != expected ({B}, {self._vlm_latent_dim})"
            )
            assert not torch.isnan(vlm_pooled).any(), "VLM pooled contains NaN!"
            vlm_nonzero = (vlm_pooled.abs() > 1e-8).any(dim=-1).all().item()
            if not vlm_nonzero:
                print(f"[IfaceEnvWrapper] WARNING: VLM latent is all zeros for some envs!")
            if not hasattr(self, '_vlm_first_log_done'):
                print(f"[IfaceEnvWrapper] VLM latent extracted: shape={vlm_pooled.shape} "
                      f"norm={vlm_pooled.norm(dim=-1).mean().item():.2f} "
                      f"range=[{vlm_pooled.min().item():.3f}, {vlm_pooled.max().item():.3f}]")
                self._vlm_first_log_done = True

            for i, eid in enumerate(env_ids_to_infer):
                self._cached_vlm_latent[eid] = vlm_pooled[i].detach().to(self.device)

        # Parse batched action chunks
        chunks = self._parse_batched_action(pred_action, B)  # (B, H, 7)
        for i, eid in enumerate(env_ids_to_infer):
            self._action_windows[eid] = chunks[i][:self.action_horizon]
            self._current_window_idx[eid] = 0
            # Set _current_action_7 to window[0] immediately.
            # The current step's control_tick uses pre_step_base_action (snapshotted),
            # so this update only affects the NEXT _build_obs() call — which is correct.
            self._current_action_7[eid] = chunks[i][0].astype(np.float32)

    # ─────────────────────────────────────────────────────────────────
    # Observation building
    # ─────────────────────────────────────────────────────────────────
    def _build_obs(self) -> dict[str, torch.Tensor]:
        """Build resfit-compatible batched obs dict from current sim state.

        State matches original ResFit DexMimicGen: 9D
        [eef_pos(3), eef_quat_xyzw(4), gripper_qpos(2)]
        """
        N = self.num_envs

        # 1. EEF position (3D) — panda_hand body world position
        hand_body_idx = self.robot.find_bodies("panda_hand")[0][0]
        ee_pos = self.robot.data.body_pos_w[:N, hand_body_idx].detach().cpu().numpy()  # (N, 3)

        # 2. EEF quaternion (4D) — xyzw convention (matching robosuite/ResFit)
        ee_quat_wxyz = self.robot.data.body_quat_w[:N, hand_body_idx].detach().cpu().numpy()  # (N, 4) wxyz from Isaac
        # Convert wxyz → xyzw (robosuite convention)
        ee_quat_xyzw = np.stack([ee_quat_wxyz[:, 1], ee_quat_wxyz[:, 2], ee_quat_wxyz[:, 3], ee_quat_wxyz[:, 0]], axis=-1)  # (N, 4)

        # 3. Gripper qpos (2D) — both finger joint positions
        all_joint_pos = self.robot.data.joint_pos[:N]
        gripper_qpos = all_joint_pos[:, self.hand_joint_ids].detach().cpu().numpy().astype(np.float32)  # (N, 2)

        # Contact force magnitude (scalar per env)
        contact_force_mag = self.left_finger_force[:N].norm(dim=-1, keepdim=True).detach().cpu().numpy().astype(np.float32)  # (N, 1)

        # Concatenate: 3 + 4 + 2 + 1 = 10
        states = np.concatenate([ee_pos, ee_quat_xyzw, gripper_qpos, contact_force_mag], axis=-1).astype(np.float32)  # (N, 10)

        # Base action in normalized [-1,1] space (matching action normalization in step())
        base_actions_raw = np.stack([self._current_action_7[eid] for eid in range(N)]).astype(np.float32)  # (N, 7)
        base_actions_norm = np.concatenate([
            base_actions_raw[:, :3] / 0.02,
            base_actions_raw[:, 3:6] / 0.2,
            base_actions_raw[:, 6:7] * 2.0 - 1.0,
        ], axis=-1).astype(np.float32)  # (N, 7) in ~[-1,1]

        # Images
        front_frames, back_frames, wrist_frames = [], [], []
        for eid in range(N):
            front_frames.append(_hwc_to_chw84(_rgb_to_uint8(self.camera_front.data.output["rgb"], eid)))
            back_frames.append(_hwc_to_chw84(_rgb_to_uint8(self.camera_back.data.output["rgb"], eid)))
            wrist_frames.append(_hwc_to_chw84(_rgb_to_uint8(self.camera_wrist.data.output["rgb"], eid)))

        obs = {
            "observation.state": torch.tensor(states, device=self.device, dtype=torch.float32),
            "observation.base_action": torch.tensor(base_actions_norm, device=self.device, dtype=torch.float32),
            "observation.vlm_latent": self._cached_vlm_latent[:N].clone(),  # (N, 2048) cached from GR00T
            "observation.images.front": torch.tensor(np.stack(front_frames), device=self.device, dtype=torch.uint8),
            "observation.images.back": torch.tensor(np.stack(back_frames), device=self.device, dtype=torch.uint8),
            "observation.images.wrist": torch.tensor(np.stack(wrist_frames), device=self.device, dtype=torch.uint8),
        }

        # Depth observations (normalized [0,1] float32)
        if self.use_depth:
            import torch.nn.functional as F
            for cam_name, cam_obj in [("front", self.camera_front), ("wrist", self.camera_wrist)]:
                depth_raw = cam_obj.data.output["depth"][:N]  # (N, H, W, 1) or (N, H, W)
                if depth_raw.dim() == 3:
                    depth_raw = depth_raw.unsqueeze(-1)  # (N, H, W, 1)
                # NCHW for interpolate
                depth_nchw = depth_raw.permute(0, 3, 1, 2).float()  # (N, 1, H, W)
                depth_nchw = F.interpolate(depth_nchw, size=(84, 84), mode="bilinear", align_corners=False)
                # Fixed normalization
                d_min = self._depth_norm[cam_name]["min"]
                d_max = self._depth_norm[cam_name]["max"]
                depth_nchw = torch.clamp(depth_nchw, d_min, d_max)
                depth_nchw = (depth_nchw - d_min) / max(d_max - d_min, 1e-6)
                # Handle NaN/Inf
                depth_nchw = torch.nan_to_num(depth_nchw, nan=0.0, posinf=1.0, neginf=0.0)
                obs[f"observation.depth.{cam_name}"] = depth_nchw  # (N, 1, 84, 84) float32 [0,1]

                # Depth debug logging (first 5 calls only)
                if not hasattr(self, '_depth_debug_count'):
                    self._depth_debug_count = 0
                if self._depth_debug_count < 5:
                    print(f"[DEPTH DEBUG] {cam_name}: raw_shape={depth_raw.shape} "
                          f"raw_range=[{depth_raw.min().item():.4f}, {depth_raw.max().item():.4f}] "
                          f"norm_range=[{depth_nchw.min().item():.4f}, {depth_nchw.max().item():.4f}] "
                          f"d_min={d_min} d_max={d_max} "
                          f"output_shape={depth_nchw.shape} dtype={depth_nchw.dtype}")
            if self._depth_debug_count < 5:
                self._depth_debug_count += 1

        # Object state: cube 6D pose → pos(3) + quat_wxyz(4) = 7D
        if self.use_state:
            cube_state = self.cube.data.root_state_w[:N]  # (N, 13)
            cube_pos = cube_state[:, :3]   # (N, 3)
            cube_quat_wxyz = cube_state[:, 3:7]  # (N, 4)
            obs["observation.object_state"] = torch.cat(
                [cube_pos, cube_quat_wxyz], dim=-1
            ).to(dtype=torch.float32)  # (N, 7)
            # Debug: first 3 calls
            if not hasattr(self, '_obj_state_debug_count'):
                self._obj_state_debug_count = 0
            if self._obj_state_debug_count < 3:
                v = obs["observation.object_state"][0]
                print(f"[OBJ_STATE] pos=({v[0]:.4f},{v[1]:.4f},{v[2]:.4f}) "
                      f"quat=({v[3]:.4f},{v[4]:.4f},{v[5]:.4f},{v[6]:.4f})")
                self._obj_state_debug_count += 1

        # ── Strict output validation ──
        assert obs["observation.state"].shape == (N, self._state_dim), (
            f"_build_obs: state shape {obs['observation.state'].shape} != expected ({N}, {self._state_dim})"
        )
        assert obs["observation.vlm_latent"].shape == (N, self._vlm_latent_dim), (
            f"_build_obs: vlm_latent shape {obs['observation.vlm_latent'].shape} != expected ({N}, {self._vlm_latent_dim})"
        )
        assert obs["observation.base_action"].shape == (N, 7), (
            f"_build_obs: base_action shape {obs['observation.base_action'].shape} != expected ({N}, 7)"
        )
        for cam_key in ["observation.images.front", "observation.images.back", "observation.images.wrist"]:
            assert obs[cam_key].shape[0] == N and obs[cam_key].shape[1] == 3, (
                f"_build_obs: {cam_key} shape {obs[cam_key].shape} invalid"
            )

        return obs

    # ─────────────────────────────────────────────────────────────────
    # Efficient frame capture for video saving
    # ─────────────────────────────────────────────────────────────────
    def get_frame(self, env_id: int = 0, camera: str = "front", size: tuple[int, int] = (128, 160)) -> np.ndarray:
        """Get a downscaled frame from specified camera for video saving. Returns (H, W, 3) uint8."""
        cam_map = {"front": self.camera_front, "back": self.camera_back, "wrist": self.camera_wrist}
        cam = cam_map.get(camera, self.camera_front)
        f = _rgb_to_uint8(cam.data.output["rgb"], env_id)
        ft = torch.from_numpy(f).permute(2, 0, 1).float().unsqueeze(0)
        ft = F.interpolate(ft, size=size, mode="bilinear", align_corners=False)
        return ft.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

    def get_debug_frame(self, env_id: int = 0, size: tuple[int, int] = (256, 400)) -> np.ndarray:
        """Get a frame with reward component overlays for debugging.

        Returns (H, W, 3) uint8 frame with text showing:
        - Each reward component value
        - Contact force, cube height, finger-cube distance
        - Success status and sustained contact flag
        """
        import cv2

        frame = self.get_frame(env_id, camera="front", size=size)
        debug = getattr(self, "_last_step_debug", None)
        if debug is None:
            return frame

        frame = frame.copy()
        eid = env_id
        h, w = frame.shape[:2]

        # Gather values
        step = int(debug["step"][eid].item()) if hasattr(debug["step"], '__getitem__') else int(debug["step"])
        cf = debug["contact_force"][eid].item()
        hc = debug["has_contact"][eid].item()
        ch = debug["cube_height"][eid].item()
        fd = debug["finger_cube_dist"][eid].item()
        rw = debug["reward"][eid].item()
        succ = debug["success"][eid].item()
        sustained = debug.get("sustained_contact", None)
        cl = debug.get("contact_lost_during_lift", None)

        # Reward components (dense_clipped only)
        rd = debug.get("reward_distance", None)
        rc = debug.get("reward_contact", None)
        rh = debug.get("reward_height", None)
        rs = debug.get("reward_success", None)
        gg = debug.get("grasp_gate", None)

        # Semi-transparent black panel at top
        panel_h = 185 if rd is not None else 95
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.38
        c_white = (255, 255, 255)
        c_green = (0, 255, 0)
        c_red = (0, 0, 255)
        c_yellow = (0, 255, 255)
        c_cyan = (255, 255, 0)
        c_gray = (160, 160, 160)

        y = 14
        dy = 13  # line spacing

        # Line 1: step + total reward + success
        succ_str = "SUCCESS" if succ > 0.5 else ""
        cv2.putText(frame, f"step={step:>4d}  R={rw:.4f}", (4, y), font, fs, c_white, 1)
        if succ_str:
            cv2.putText(frame, succ_str, (w - 65, y), font, fs, c_green, 1)
        y += dy

        # Line 2: cube height + finger dist
        cv2.putText(frame, f"h={ch*100:.1f}cm  dist={fd*100:.1f}cm", (4, y), font, fs, c_white, 1)
        y += dy

        # Line 3: contact force + flags
        hc_str = "ON" if hc > 0.5 else "OFF"
        hc_color = c_green if hc > 0.5 else c_red
        cv2.putText(frame, f"cf={cf:.1f}N contact=", (4, y), font, fs, c_white, 1)
        cv2.putText(frame, hc_str, (125, y), font, fs, hc_color, 1)
        if cl is not None:
            cl_val = cl[eid].item()
            if cl_val > 0.5:
                cv2.putText(frame, "LOST!", (160, y), font, fs, c_red, 1)
            elif sustained is not None and sustained[eid].item() > 0.5 and ch >= 0.005:
                cv2.putText(frame, "HELD", (160, y), font, fs, c_green, 1)
        y += dy

        # Line 4: grasp gate + sub-components
        if gg is not None:
            gg_val = gg[eid].item()
            bc = debug.get("both_close", None)
            gc = debug.get("gripper_closing", None)
            lfd = debug.get("left_finger_cube_dist", None)
            rfd = debug.get("right_finger_cube_dist", None)
            fjp = debug.get("finger_joint_pos", None)

            gg_color = c_green if gg_val > 0.5 else c_red
            cv2.putText(frame, f"gate={gg_val:.0f}", (4, y), font, fs, gg_color, 1)

            # Show which sub-components pass/fail
            parts = []
            if bc is not None:
                bc_v = bc[eid].item()
                parts.append(("close", bc_v > 0.5))
            if gc is not None:
                gc_v = gc[eid].item()
                parts.append(("grip", gc_v > 0.5))
            parts.append(("contact", hc > 0.5))

            x_off = 65
            for pname, pval in parts:
                pc = c_green if pval else c_red
                cv2.putText(frame, f"{pname}={'Y' if pval else 'N'}", (x_off, y), font, 0.32, pc, 1)
                x_off += 62
            y += dy

            # Line 5: finger distances detail
            if lfd is not None and rfd is not None:
                ld = lfd[eid].item() * 100  # cm
                rd_v = rfd[eid].item() * 100
                fj = fjp[eid].item() if fjp is not None else -1
                cv2.putText(frame, f"Lfing={ld:.1f}cm Rfing={rd_v:.1f}cm grip_q={fj:.3f}", (4, y), font, 0.32, c_gray, 1)
                y += dy

        # Lines: reward components (dense_clipped)
        if rd is not None:
            components = [
                ("R_dist", rd[eid].item(), 0.1, c_cyan),
                ("R_contact", rc[eid].item(), 0.2, c_yellow),
                ("R_height", rh[eid].item(), 0.5, c_green),
                ("R_success", rs[eid].item(), 1.0, c_green),
            ]
            for name, val, maxv, color in components:
                bar_x = 120
                bar_w = int((val / max(maxv, 1e-6)) * (w - bar_x - 10))
                bar_w = max(0, min(bar_w, w - bar_x - 10))
                cv2.putText(frame, f"{name:>10s}={val:.4f}", (4, y), font, fs, color, 1)
                cv2.rectangle(frame, (bar_x, y - 8), (bar_x + bar_w, y), color, -1)
                y += dy

        # Bottom bar: reward as color (green=high, red=low)
        bar_h = 6
        rw_clamp = max(0.0, min(1.0, rw))
        bar_color = (0, int(rw_clamp * 255), int((1 - rw_clamp) * 255))
        cv2.rectangle(frame, (0, h - bar_h), (w, h), bar_color, -1)

        # ── Cube 6D pose arrows (if use_state mode) ──
        if self.use_state:
            self._draw_pose_arrows(frame, eid)

        return frame

    def _draw_pose_arrows(self, frame: np.ndarray, env_id: int) -> None:
        """Draw 3D orientation axes of cube and EE at their projected positions."""
        import cv2
        from scipy.spatial.transform import Rotation

        N = self.num_envs
        h, w = frame.shape[:2]

        # Camera setup (front camera pinhole)
        W_nat, H_nat = 640, 480
        focal = 22.0
        aperture = 20.955
        fx = focal * W_nat / aperture
        fy = fx
        cx_n, cy_n = W_nat / 2, H_nat / 2

        # Camera extrinsic: look-at (same as replay viz)
        env_origin = self.scene.env_origins[env_id].detach().cpu().numpy()
        eye = env_origin + np.array([0.4, -0.7, 0.8])
        target = env_origin + np.array([0.25, 0.0, -0.05])
        world_up = np.array([0.0, 0.0, 1.0])
        fwd = target - eye; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, world_up); right /= np.linalg.norm(right)
        cam_up = np.cross(right, fwd)
        R_cv = np.stack([right, -cam_up, fwd], axis=0)
        t_cv = -R_cv @ eye

        def _proj(pt):
            p = R_cv @ pt + t_cv
            if p[2] <= 0.01:
                return None
            u = int(fx * p[0] / p[2] + cx_n) * w // W_nat
            v = int(fy * p[1] / p[2] + cy_n) * h // H_nat
            return (u, v)

        def _draw_axes(origin_w, quat_wxyz, colors_label, axis_len=0.05):
            o = _proj(origin_w)
            if o is None or o[0] < -50 or o[0] > w+50 or o[1] < -50 or o[1] > h+50:
                return
            xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            R_obj = Rotation.from_quat(xyzw).as_matrix()
            colors = [(0,0,255),(0,255,0),(255,0,0)]  # X=R, Y=G, Z=B
            labels = ["X","Y","Z"]
            for i in range(3):
                tip = _proj(origin_w + R_obj[:, i] * axis_len)
                if tip:
                    cv2.arrowedLine(frame, o, tip, colors[i], 2, tipLength=0.25)
                    cv2.putText(frame, labels[i], (tip[0]+2, tip[1]-2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[i], 1)
            cv2.circle(frame, o, 3, colors_label, -1)

        # Cube
        cube_pos = self.cube.data.root_state_w[env_id, :3].detach().cpu().numpy()
        cube_quat = self.cube.data.root_state_w[env_id, 3:7].detach().cpu().numpy()
        _draw_axes(cube_pos, cube_quat, (0, 255, 0))  # green dot

        # EE
        hand_idx = self.robot.find_bodies("panda_hand")[0][0]
        ee_pos = self.robot.data.body_pos_w[env_id, hand_idx].detach().cpu().numpy()
        ee_quat = self.robot.data.body_quat_w[env_id, hand_idx].detach().cpu().numpy()
        _draw_axes(ee_pos, ee_quat, (0, 255, 255))  # cyan dot

    def get_all_frames_batch(self, camera: str = "front", size: tuple[int, int] = (256, 320)) -> list[np.ndarray]:
        """GPU-batched: resize all env frames at once, return list of (H,W,3) uint8 numpy."""
        cam_map = {"front": self.camera_front, "back": self.camera_back, "wrist": self.camera_wrist}
        cam = cam_map.get(camera, self.camera_front)
        N = self.num_envs
        # cam.data.output["rgb"] is (N, H, W, 4) on GPU
        rgb = cam.data.output["rgb"][:N]  # (N, H, W, 4)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]  # drop alpha → (N, H, W, 3)
        # Convert to float and NCHW for interpolate
        if rgb.dtype == torch.uint8:
            batch = rgb.permute(0, 3, 1, 2).float()  # (N, 3, H, W)
        else:
            # float [0,1] → [0,255]
            batch = (rgb.clamp(0, 1) * 255.0).permute(0, 3, 1, 2)  # (N, 3, H, W)
        # Batch resize on GPU
        batch = F.interpolate(batch, size=size, mode="bilinear", align_corners=False)
        # To uint8 numpy — single GPU→CPU transfer
        batch_np = batch.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()  # (N, H, W, 3)
        return [batch_np[i] for i in range(N)]

    def get_all_frames(self, camera: str = "front", size: tuple[int, int] = (128, 160)) -> list[np.ndarray]:
        """Get downscaled frames for all envs. Returns list of (H, W, 3) uint8."""
        return self.get_all_frames_batch(camera, size)

    # ─────────────────────────────────────────────────────────────────
    # Action parsing
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_action(pred_action) -> np.ndarray | None:
        if isinstance(pred_action, dict):
            if "action" in pred_action:
                arr = np.array(pred_action["action"], dtype=np.float32)
                return arr[0] if arr.ndim == 3 else arr
            if "action.ee_delta" in pred_action and "action.gripper_pos" in pred_action:
                ee = np.array(pred_action["action.ee_delta"], dtype=np.float32)
                gr = np.array(pred_action["action.gripper_pos"], dtype=np.float32)
                ee = ee[0] if ee.ndim == 3 else ee
                gr = gr[0] if gr.ndim == 3 else gr
                return np.concatenate([ee, gr], axis=-1)
            return None
        arr = np.array(pred_action, dtype=np.float32)
        return arr[0] if arr.ndim == 3 else arr

    @staticmethod
    def _parse_batched_action(pred_action, batch_size: int) -> np.ndarray:
        """Parse batched GR00T output into (B, H, 7) action chunks."""
        if isinstance(pred_action, dict):
            if "action" in pred_action:
                arr = np.array(pred_action["action"], dtype=np.float32)
                return arr if arr.ndim == 3 else arr[None, ...]
            if "action.ee_delta" in pred_action and "action.gripper_pos" in pred_action:
                ee = np.array(pred_action["action.ee_delta"], dtype=np.float32)
                gr = np.array(pred_action["action.gripper_pos"], dtype=np.float32)
                if ee.ndim == 2:
                    ee = ee[None, ...]
                    gr = gr[None, ...]
                return np.concatenate([ee, gr], axis=-1)
            return np.zeros((batch_size, 16, 7), dtype=np.float32)
        arr = np.array(pred_action, dtype=np.float32)
        return arr if arr.ndim == 3 else arr[None, ...]

    def close(self):
        pass

