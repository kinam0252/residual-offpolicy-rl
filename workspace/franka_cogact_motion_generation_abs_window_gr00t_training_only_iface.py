# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
GR00T-based motion generation with 3 cameras and local model inference (no policy server).

Key fixes integrated:
- View mapping aligned to dataset generation pipeline:
    ego_view  <- wrist camera
    left_view <- back/side camera
    right_view<- front camera
  (dataset gen used front->right_view, back->left_view, wrist->wrist_view; we map ego<-wrist, left<-back, right<-front) [1](https://microsoftapc-my.sharepoint.com/personal/t-kinamkim_microsoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/franka_motion_from_pose_csv.py)
- Camera poses for front/back are explicitly set via set_world_poses_from_view() to match dataset capture. [1](https://microsoftapc-my.sharepoint.com/personal/t-kinamkim_microsoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/franka_motion_from_pose_csv.py)
- Language can be forced to GT-id style (e.g., "0") to match LeRobot parquet input. [3](https://wiki.arcoslab.org/en/tutorials/panda)[2](https://microsoftapc-my.sharepoint.com/personal/t-kinamkim_microsoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/franka_cogact_motion_generation_abs_window_gr00t.py)
- action.ee_delta is applied as delta (dp + rotvec) and integrated onto current EE pose (quat multiply), not treated as absolute RPY. [2](https://microsoftapc-my.sharepoint.com/personal/t-kinamkim_microsoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/franka_cogact_motion_generation_abs_window_gr00t.py)
- inference_interval is derived from TARGET_ACTION_FPS and action_horizon (from modality config), removing conflicting hard-coded 200-step interval. [2](https://microsoftapc-my.sharepoint.com/personal/t-kinamkim_microsoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/franka_cogact_motion_generation_abs_window_gr00t.py)

Run:
  ./isaaclab.sh -p scripts/my_test/franka_cogact_motion_generation_abs_window_gr00t.py --enable_cameras --headless \
    --gr00t_host 127.0.0.1 --gr00t_port 5555 \
    --csv_dir /home/t-kinamkim/Repos/VLA_RL/.Data/tmp_dataset/pickMushroom_20251209_081744_174 \
    --language_override 0
"""

"""Launch Isaac Sim Simulator first."""
import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any


def _bootstrap_typing_extensions_override() -> None:
    """Force-load workspace typing_extensions before Isaac prebundle version.

    Isaac's pip_prebundle may provide an older typing_extensions without
    `NoExtraItems`, which breaks `typeguard` imported by `tyro`.
    """
    workspace_dir = Path(__file__).resolve().parent
    deps_dir = Path(
        os.environ.get("ISO_DEPS_DIR", str(workspace_dir / ".pydeps_groot_iso"))
    ).expanduser().resolve()
    te_file = deps_dir / "typing_extensions.py"
    if not te_file.exists():
        return

    deps_dir_str = str(deps_dir)
    if deps_dir_str not in sys.path:
        sys.path.insert(0, deps_dir_str)
    else:
        sys.path.remove(deps_dir_str)
        sys.path.insert(0, deps_dir_str)

    spec = importlib.util.spec_from_file_location("typing_extensions", str(te_file))
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "NoExtraItems"):
        sys.modules["typing_extensions"] = module


import os

_bootstrap_typing_extensions_override()

from isaaclab.app import AppLauncher

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser(description="GR00T motion generation with 3 cameras and differential IK.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--model_path",
    type=str,
    default="/home/kinam/Desktop/Repos/VLA_RL/Isaac-GR00T/outputs/checkpoint-100000",
    help="Path to local GR00T checkpoint directory (loaded in-process, no server).",
)
parser.add_argument(
    "--embodiment_tag",
    type=str,
    default="new_embodiment",
    help="Embodiment tag string for Gr00tPolicy (e.g., new_embodiment, libero_panda).",
)
parser.add_argument(
    "--policy_device",
    type=str,
    default="cuda",
    help="Device for local GR00T policy inference (e.g., cuda, cuda:0, cpu).",
)
parser.add_argument(
    "--policy_strict",
    action="store_true",
    help="Enable strict policy input/output validation.",
)
parser.add_argument("--task_description", type=str, default="Pick up the white object.", help="Task description.")
parser.add_argument(
    "--csv_dir",
    type=str,
    default=None,
    help="Directory containing CSVs (franka_joint_states.csv, gripper_joint_states.csv, object_pos.csv). "
         "If not set, all subfolders under --data_dir are used.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="/home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom",
    help="Root folder containing episode CSV subfolders (default: /home/t-kinamkim/Repos/Data/pickMushroom).",
)
parser.add_argument(
    "--split_file",
    type=str,
    default=None,
    help="Path to split file with '<episode_name> <training|testing>' rows. "
         "If not set, uses <data_dir>/train_test_split.txt.",
)
parser.add_argument(
    "--lift_success_threshold",
    type=float,
    default=0.005,
    help="Success threshold in meters for (max_cube_z - initial_cube_z). Default: 0.005m",
)
parser.add_argument(
    "--success_log_csv",
    type=str,
    default=None,
    help="Optional output CSV path for per-episode success logs. "
         "Default: outputs_gr00t/training_lift_success_log.csv",
)
parser.add_argument(
    "--max_attempts_per_episode",
    type=int,
    default=5,
    help="How many times to run each episode. If any trial succeeds, the episode is marked successful.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, save per-episode debug videos (left/right/wrist) under outputs_gr00t/debug/<episode_name>/",
)
parser.add_argument(
    "--compare_debug_dump",
    action="store_true",
    help="Dump runtime comparison metrics JSONL under outputs_gr00t/debug_compare.",
)
parser.add_argument(
    "--compare_debug_interval",
    type=int,
    default=60,
    help="Comparison debug logging interval in sim steps.",
)
parser.add_argument(
    "--heartbeat_log_interval",
    type=int,
    default=100,
    help="Emit runtime heartbeat logs every N sim steps (0 disables).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=1000,
    help="Maximum sim steps per episode before forced stop.",
)
parser.add_argument(
    "--rl_bootstrap",
    action="store_true",
    help="Step-1 RL migration switch: keep current non-RL behavior but write outputs to RL bootstrap folder.",
)
parser.add_argument(
    "--rl_episode_limit",
    type=int,
    default=0,
    help="Step-2 RL migration: when > 0 and --rl_bootstrap is set, run only first N training episodes.",
)
parser.add_argument(
    "--rl_disable_retries",
    action="store_true",
    help="Step-2 RL migration: run only one attempt per episode (disable multi-try success search).",
)

# IMPORTANT: to match GT language in LeRobot, override with "0"
parser.add_argument(
    "--language_override",
    type=str,
    default=None,
    help="If set (e.g. '0'), use this string as language instead of --task_description. "
         "Useful to match LeRobot parquet language (often '0').",
)

# Control-rate knobs
parser.add_argument("--target_action_fps", type=float, default=20.0, help="Target policy action FPS (default 20Hz).")
parser.add_argument("--rot_clamp_rad", type=float, default=0.35, help="Clamp |rotvec| for ee_delta rotation (rad).")

parser.add_argument(
    "--exec_horizon",
    type=int,
    default=16,
    help="(Deprecated) Execute only the first N actions from the predicted action chunk. "
         "If unset, the full model horizon is used.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# cameras needed
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------
# imports after AppLauncher
# -----------------------------
import os
import json
import time
import faulthandler
import traceback
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
import omni.timeline
import imageio

try:
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
except ImportError:
    raise ImportError(
        "gr00t local policy API not found. Add Isaac-GR00T to PYTHONPATH or install dependencies in this env."
    )

from resfit.rl_finetuning.policies.base_policy_interface import BaseChunkPolicy

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

faulthandler.enable(all_threads=True)


# -----------------------------
# constants
# -----------------------------
MODEL_PATH = getattr(args_cli, "model_path", None)
EMBODIMENT_TAG = getattr(args_cli, "embodiment_tag", "new_embodiment")
POLICY_DEVICE = getattr(args_cli, "policy_device", "cuda")
POLICY_STRICT = bool(getattr(args_cli, "policy_strict", False))

CSV_BASE_DIR = getattr(args_cli, "csv_dir", None)
DATA_DIR = getattr(args_cli, "data_dir", "/home/t-kinamkim/Repos/Data/pickMushroom")
SPLIT_FILE = getattr(args_cli, "split_file", None)
TASK_DESCRIPTION = getattr(args_cli, "task_description", "Pick up the white object.")
LANGUAGE_OVERRIDE = getattr(args_cli, "language_override", None)
LIFT_SUCCESS_THRESHOLD = float(getattr(args_cli, "lift_success_threshold", 0.005))
SUCCESS_LOG_CSV = getattr(args_cli, "success_log_csv", None)
MAX_ATTEMPTS_PER_EPISODE = int(getattr(args_cli, "max_attempts_per_episode", 5))

TARGET_ACTION_FPS = float(getattr(args_cli, "target_action_fps", 20.0))
ROT_CLAMP_RAD = float(getattr(args_cli, "rot_clamp_rad", 0.35))
EXEC_HORIZON = int(getattr(args_cli, "exec_horizon", 16))
DEBUG_MODE = bool(getattr(args_cli, "debug", False))
COMPARE_DEBUG_DUMP = bool(getattr(args_cli, "compare_debug_dump", False))
COMPARE_DEBUG_INTERVAL = int(getattr(args_cli, "compare_debug_interval", 60))
HEARTBEAT_LOG_INTERVAL = int(getattr(args_cli, "heartbeat_log_interval", 100))
MAX_STEPS = int(getattr(args_cli, "max_steps", 1000))
RL_BOOTSTRAP = bool(getattr(args_cli, "rl_bootstrap", False))
RL_EPISODE_LIMIT = int(getattr(args_cli, "rl_episode_limit", 0))
RL_DISABLE_RETRIES = bool(getattr(args_cli, "rl_disable_retries", False))

# Policy modality keys (same names used by GR00T + LeRobot)
VIDEO_KEYS = ["wrist_view", "left_view", "right_view"]
STATE_JOINT_KEY = "proprio.joint_pos"
STATE_GRIPPER_KEY = "proprio.gripper_pos"
LANGUAGE_KEY = "annotation.human.action.task_description"  # policy expects this key [2](https://microsoftapc-my.sharepoint.com/personal/t-kinamkim_microsoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/franka_cogact_motion_generation_abs_window_gr00t.py)


# -----------------------------
# scene config
# -----------------------------
@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Table-top scene with custom floor, lighting, and cube material."""

    # Large flat floor instead of default ground plane
    floor = AssetBaseCfg(
        prim_path="/World/Environment/floor",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )

    # Softer dome light as environment light
    dome_light = AssetBaseCfg(
        prim_path="/World/Lights/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=600.0,
            color=(1.0, 1.0, 1.0),
        ),
    )

    # Directional sun-like light for shading and shadows
    sun_light = AssetBaseCfg(
        prim_path="/World/Lights/SunLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=500.0,
            color=(1.0, 0.98, 0.95),
            angle=0.53,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 3.0),
            rot=(0.892, 0.239, 0.369, 0.0),
        ),
    )

    # Optional key light if RectLightCfg is available
    try:
        key_light = AssetBaseCfg(
            prim_path="/World/Lights/KeyLight",
            spawn=sim_utils.RectLightCfg(
                intensity=3000.0,
                color=(1.0, 0.98, 0.95),
                width=1.2,
                height=0.8,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(1.2, -1.0, 2.0),
                rot=(0.92, 0.23, 0.30, 0.0),
            ),
        )
    except AttributeError:
        # Older Isaac Lab versions may not have RectLightCfg
        pass

    # Cube with more realistic friction and visible color
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.12, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=2.0,
                disable_gravity=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                enable_gyroscopic_forces=True,
                retain_accelerations=False,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.75, 0.15, 0.15)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, 0.0, 0.06),
            rot=(0.707, 0.0, 0.0, 0.707),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
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
                effort_limit_sim=500.0,
                stiffness=2e4,
                damping=1500.0,
            ),
        },
    )


def _make_gr00t_scene_cfg(num_envs: int, env_spacing: float):
    """
    Scene config with 3 cameras for policy input.
    IMPORTANT mapping (aligned to dataset generation):
      right_view <- camera_front
      left_view  <- camera_back
      wrist_view <- camera_wrist
    We will map video_dict keys accordingly later. [1](https://microsoftapc-my.sharepoint.com/personal/t-kinamkim_microsoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/franka_motion_from_pose_csv.py)
    """
    cfg = TableTopSceneCfg(num_envs=num_envs, env_spacing=env_spacing)

    # Front fixed camera (this will become right_view in mapping)
    cfg.camera_front = CameraCfg(
        prim_path="/World/envs/env_.*/DebugCamera_front",
        width=640,
        height=480,
        offset=CameraCfg.OffsetCfg(pos=(0.4, -0.7, 0.8), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        data_types=["rgb"],
    )

    # Back/side fixed camera (this will become left_view in mapping)
    cfg.camera_back = CameraCfg(
        prim_path="/World/envs/env_.*/DebugCamera_back",
        width=640,
        height=480,
        offset=CameraCfg.OffsetCfg(pos=(0.4, 0.7, 0.8), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        data_types=["rgb"],
    )

    # Wrist camera (this will become wrist_view in mapping)
    # Keep base orientation but pitch down by ~20 degrees to see more of the table.
    q0 = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32)  # (w,x,y,z)
    theta = np.deg2rad(-20.0)
    q_pitch = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0], dtype=np.float32)
    # Compose rotations using scipy (convert to (x,y,z,w) for Rotation)
    base_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]], dtype=np.float32)
    pitch_xyzw = np.array([q_pitch[1], q_pitch[2], q_pitch[3], q_pitch[0]], dtype=np.float32)
    q_new_xyzw = (Rotation.from_quat(base_xyzw) * Rotation.from_quat(pitch_xyzw)).as_quat()
    q_new_wxyz = np.array(
        [q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]], dtype=np.float32
    )

    cfg.camera_wrist = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera",
        width=640,
        height=480,
        offset=CameraCfg.OffsetCfg(
            pos=(0.06, 0.0, 0.0),
            rot=(float(q_new_wxyz[0]), float(q_new_wxyz[1]), float(q_new_wxyz[2]), float(q_new_wxyz[3])),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=0.5,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        data_types=["rgb"],
    )

    return cfg


# -----------------------------
# helpers
# -----------------------------
def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    q_xyzw = Rotation.from_euler("z", yaw, degrees=False).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def _rgb_to_uint8(rgb_tensor):
    """Convert Isaac Lab RGB tensor (1,H,W,3 or similar) to (H,W,3) uint8."""
    rgb = rgb_tensor[0]
    if rgb.is_cuda:
        rgb = rgb.cpu()
    arr = rgb.numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    # drop alpha if exists
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply (wxyz): returns q = q1 * q2."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _resolve_embodiment_tag(tag_str: str) -> EmbodimentTag:
    tag_str = str(tag_str).strip()
    for tag in EmbodimentTag:
        if tag.value == tag_str:
            return tag
    valid = ", ".join(sorted(t.value for t in EmbodimentTag))
    raise ValueError(f"Unknown embodiment_tag='{tag_str}'. Valid values: {valid}")


class LocalGr00tChunkPolicyAdapter:
    def __init__(self, policy: Gr00tPolicy):
        self._policy = policy

    def get_modality_config(self) -> Any:
        return self._policy.get_modality_config()

    def infer_action_chunk(self, obs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        pred_action, info = self._policy.get_action(obs)
        if info is None:
            info = {}
        return pred_action, info


def _pick_video_for_key(key: str, wrist: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lk = key.lower()
    if "wrist" in lk or "ego" in lk:
        return wrist
    if "left" in lk:
        return left
    if "right" in lk or "front" in lk:
        return right
    return wrist


def _pick_state_for_key(key: str, joint_pos_np: np.ndarray, gripper_frac: float) -> np.ndarray:
    lk = key.lower()
    if "joint" in lk:
        return joint_pos_np.astype(np.float32)
    if "gripper" in lk:
        return np.array([float(gripper_frac)], dtype=np.float32)
    return np.zeros((1,), dtype=np.float32)


def _read_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return float("nan")


def _log_heartbeat(step: int, tag: str = "") -> None:
    rss_mb = _read_rss_mb()
    msg = f"[HB] step={step} rss_mb={rss_mb:.1f}"
    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            used_mb = (total_b - free_b) / (1024.0 ** 2)
            total_mb = total_b / (1024.0 ** 2)
            msg += f" cuda_used_mb={used_mb:.1f} cuda_total_mb={total_mb:.1f}"
        except Exception as e:
            msg += f" cuda_mem_query_failed={e}"
    if tag:
        msg += f" tag={tag}"
    print(msg, flush=True)


# -----------------------------
# main simulator loop
# -----------------------------
def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    csv_dir: Path,
    video_path: Path,
    policy: BaseChunkPolicy,
    debug_dir: Path | None = None,
):
    """Runs the simulation loop with GR00T inference (3 cameras + state + language) for one episode.

    - csv_dir: folder containing franka_joint_states.csv, gripper_joint_states.csv, object_pos.csv
    - video_path: where to save the right-view (front camera) mp4
    - debug_dir: if not None, saves separate left/right/wrist view videos under this folder (debug mode).
    """
    robot = scene["robot"]
    cube = scene["cube"]

    # Ensure cube has backing _data and no external wrench (avoids AttributeError in write_data_to_sim)
    if not hasattr(cube, "_data") and hasattr(cube, "data"):
        try:
            cube._data = cube.data
        except Exception:
            pass
    if not hasattr(cube, "has_external_wrench"):
        cube.has_external_wrench = False
    camera_front = scene["camera_front"]
    camera_back = scene["camera_back"]
    camera_wrist = scene["camera_wrist"]

    sim_dt = sim.get_physics_dt()
    steps_per_action = max(1, int(round((1.0 / TARGET_ACTION_FPS) / sim_dt)))
    action_horizon = 16  # will override after get_modality_config

    # ---- init cameras (set_world_poses_from_view to match dataset capture) [1](https://microsoftapc-my.sharepoint.com/personal/t-kinamkim_microsoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/franka_motion_from_pose_csv.py)
    timeline = omni.timeline.get_timeline_interface()
    for _ in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        for cam in (camera_front, camera_back, camera_wrist):
            cam.update(dt=sim_dt)

    env_ids_gpu = torch.arange(scene.num_envs, device=sim.device, dtype=torch.long)
    env_origins = scene.env_origins[env_ids_gpu]

    # Always set poses (headless often still works; don't gate on timeline)
    eye_front = torch.tensor([[0.4, -0.7, 0.8]], device=sim.device, dtype=torch.float32).repeat(scene.num_envs, 1) + env_origins
    eye_back  = torch.tensor([[0.4,  0.7, 0.8]], device=sim.device, dtype=torch.float32).repeat(scene.num_envs, 1) + env_origins
    target_pos = torch.tensor([[0.25, 0.0, -0.05]], device=sim.device, dtype=torch.float32).repeat(scene.num_envs, 1) + env_origins
    try:
        camera_front.set_world_poses_from_view(eye_front, target_pos)
        camera_back.set_world_poses_from_view(eye_back, target_pos)
        print("[INFO] set_world_poses_from_view applied for front/back cameras.")
    except Exception as e:
        print(f"[WARN] Failed to set camera poses from view: {e}")

    # ---- prepare right-view video writer (camera_front -> right_view)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    right_fps = TARGET_ACTION_FPS if TARGET_ACTION_FPS > 0 else 20.0
    right_writer = imageio.get_writer(str(video_path), fps=float(right_fps))

    # Optional debug writers for left/right/wrist views
    debug_left_writer = debug_right_writer = debug_wrist_writer = None
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_fps = right_fps
        debug_left_writer = imageio.get_writer(str(debug_dir / "left_view.mp4"), fps=float(debug_fps))
        debug_right_writer = imageio.get_writer(str(debug_dir / "right_view.mp4"), fps=float(debug_fps))
        debug_wrist_writer = imageio.get_writer(str(debug_dir / "wrist_view.mp4"), fps=float(debug_fps))

    # ---- local GR00T policy modality config
    mod_cfg = policy.get_modality_config()
    video_modality_keys = list(getattr(mod_cfg["video"], "modality_keys", []))
    state_modality_keys = list(getattr(mod_cfg["state"], "modality_keys", []))
    language_modality_keys = list(getattr(mod_cfg["language"], "modality_keys", []))
    language_key = language_modality_keys[0] if len(language_modality_keys) > 0 else LANGUAGE_KEY

    print(f"[INFO] local GR00T modality.video keys: {video_modality_keys}")
    print(f"[INFO] local GR00T modality.state keys: {state_modality_keys}")
    print(f"[INFO] local GR00T modality.language key: {language_key}")

    try:
        action_horizon = len(mod_cfg["action"].delta_indices)
    except Exception:
        action_horizon = 16
    # Use full model horizon by default; allow overriding via EXEC_HORIZON only if it is smaller.
    effective_horizon = min(action_horizon, EXEC_HORIZON)
    inference_interval = steps_per_action * effective_horizon
    print(
        f"[INFO] steps_per_action={steps_per_action}, action_horizon_pred={action_horizon}, "
        f"exec_horizon_effective={effective_horizon}, inference_interval={inference_interval} (sim steps)"
    )

    # ---- friction boost
    robot_materials = robot.root_physx_view.get_material_properties().to("cpu")
    robot_materials[..., 0] = 5e3
    robot_materials[..., 1] = 5e3
    env_ids_cpu = torch.arange(scene.num_envs, device="cpu", dtype=torch.long)
    robot.root_physx_view.set_material_properties(robot_materials, env_ids_cpu)

    # ---- IK controller
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    diff_ik.reset()

    # joint IDs
    arm_joint_names = sorted([n for n in robot.data.joint_names if ("panda_joint" in n and "finger" not in n)])
    arm_joint_ids = [robot.data.joint_names.index(n) for n in arm_joint_names]
    hand_joint_names = sorted([n for n in robot.data.joint_names if "panda_finger_joint" in n])
    hand_joint_ids = [robot.data.joint_names.index(n) for n in hand_joint_names]
    all_joint_ids = arm_joint_ids + hand_joint_ids

    # ---- init robot/object from CSV (same pattern you used) [2](https://microsoftapc-my.sharepoint.com/personal/t-kinamkim_microsoft_com/Documents/Microsoft%20Copilot%20Chat%20Files/franka_cogact_motion_generation_abs_window_gr00t.py)
    joint_states_csv = csv_dir / "franka_joint_states.csv"
    gripper_states_csv = csv_dir / "gripper_joint_states.csv"
    object_pos_csv = csv_dir / "object_pos.csv"

    js = pd.read_csv(joint_states_csv, header=0).values.astype(np.float32)
    gs = pd.read_csv(gripper_states_csv, header=0).values.astype(np.float32)
    js = js[1:, 10:17]  # (N,7)
    gs = gs[1:, 4:5]    # (N,1)

    arm_init = js[1, :7]
    grip_width = float(np.clip(float(gs[1, 0]), 0.0, 1.0))
    finger_pos = grip_width * 0.04

    q_init = np.concatenate([arm_init, [finger_pos] * len(hand_joint_ids)]).astype(np.float32)
    q_init_t = torch.tensor(q_init, device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
    qd_init_t = torch.zeros_like(q_init_t)
    robot.write_joint_state_to_sim(q_init_t, qd_init_t, joint_ids=all_joint_ids)
    robot.set_joint_position_target(q_init_t, joint_ids=all_joint_ids)

    # Initialize cube pose from object_pos.csv, matching franka_motion_from_pose_csv.py convention:
    # base rotation (90° around X from config) composed with yaw (around Z).
    if object_pos_csv.exists():
        od = pd.read_csv(object_pos_csv, header=0).values.astype(np.float32)
        obj = od[0, 3:7].copy()  # [x, y, z, yaw]
    else:
        obj = np.array([0.2, 0.0, 0.06, 0.0], dtype=np.float32)

    # world positions
    obj_pos_local = torch.tensor(obj[:3], device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
    obj_pos_w = env_origins + obj_pos_local

    # base rotation from default_root_state (wxyz)
    env_ids = torch.arange(scene.num_envs, device=sim.device, dtype=torch.long)
    default_root_state = cube.data.default_root_state[env_ids].clone()
    base_rot = default_root_state[0, 3:7].cpu().numpy()  # (w,x,y,z)

    # yaw quaternion (wxyz)
    yaw = float(obj[3])
    yaw_quat = yaw_to_quat_wxyz(yaw)

    # Compose yaw (world Z) * base_rot (90deg X), using scipy Rotation in (x,y,z,w)
    base_rot_scipy = np.array([base_rot[1], base_rot[2], base_rot[3], base_rot[0]])  # (x,y,z,w)
    yaw_quat_scipy = np.array([yaw_quat[1], yaw_quat[2], yaw_quat[3], yaw_quat[0]])  # (x,y,z,w)
    base_r = Rotation.from_quat(base_rot_scipy)
    yaw_r = Rotation.from_quat(yaw_quat_scipy)
    combined_r = yaw_r * base_r
    combined_quat_xyzw = combined_r.as_quat()  # (x,y,z,w)
    combined_quat_wxyz = np.array(
        [combined_quat_xyzw[3], combined_quat_xyzw[0], combined_quat_xyzw[1], combined_quat_xyzw[2]], dtype=np.float32
    )

    obj_quat_w = torch.tensor(combined_quat_wxyz, device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(
        scene.num_envs, 1
    )
    cube_pose = torch.cat([obj_pos_w, obj_quat_w], dim=-1)
    cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids_gpu)

    # settle
    for _ in range(200):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        robot.set_joint_position_target(q_init_t, joint_ids=all_joint_ids)
        camera_front.update(dt=sim_dt)
        camera_back.update(dt=sim_dt)
        camera_wrist.update(dt=sim_dt)

    try:
        initial_cube_z = float(cube.data.root_state_w[0, 2].detach().cpu().item())
    except Exception:
        initial_cube_z = float(obj_pos_w[0, 2].detach().cpu().item())
    max_cube_z = initial_cube_z

    # ---- action window state
    action_windows = None  # (H,7)
    num_windows = None
    steps_per_window = None
    current_window_idx = 0
    steps_in_window = 0
    steps_since_infer = 0

    # smoothing buffers
    if hasattr(robot, "_smoothed_dp"):
        delattr(robot, "_smoothed_dp")
    if hasattr(robot, "_smoothed_drot"):
        delattr(robot, "_smoothed_drot")

    # main loop
    count = 0
    joint_pos_des = q_init_t.clone()
    debug_compare_file = None
    debug_compare_last_ts = time.time()
    if COMPARE_DEBUG_DUMP:
        compare_dir = Path(__file__).resolve().parents[2] / "outputs_gr00t" / "debug_compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        debug_compare_file = compare_dir / f"{csv_dir.name}_mytest_compare.jsonl"
        if debug_compare_file.exists():
            debug_compare_file.unlink()
        print(f"[INFO] compare debug dump file: {debug_compare_file}")

    try:
        while simulation_app.is_running():
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.step(render=False)
                continue

            if HEARTBEAT_LOG_INTERVAL > 0 and (count % max(1, HEARTBEAT_LOG_INTERVAL) == 0):
                _log_heartbeat(count, tag="loop")

            # update cameras & keep front/back facing target
            for cam in (camera_front, camera_back, camera_wrist):
                cam.update(dt=sim_dt)
            try:
                camera_front.set_world_poses_from_view(eye_front, target_pos)
                camera_back.set_world_poses_from_view(eye_back, target_pos)
            except Exception:
                pass

            # append current right_view (front camera) frame to video (+ optional debug streams)
            if (count % steps_per_action) == 0:

                try:
                    right_frame = _rgb_to_uint8(camera_front.data.output["rgb"])
                    right_writer.append_data(right_frame)
                    if debug_right_writer is not None:
                        debug_right_writer.append_data(right_frame)
                    if debug_left_writer is not None:
                        left_frame_dbg = _rgb_to_uint8(camera_back.data.output["rgb"])
                        debug_left_writer.append_data(left_frame_dbg)
                    if debug_wrist_writer is not None:
                        wrist_frame_dbg = _rgb_to_uint8(camera_wrist.data.output["rgb"])
                        debug_wrist_writer.append_data(wrist_frame_dbg)
                except Exception as e:
                    print(f"[WARN] failed to append debug/right_view frame: {e}")

            # periodically run inference
            if steps_since_infer >= inference_interval:
                # build video dict by modality key (mapped from available cameras)
                video_dict = {}
                try:
                    wrist = _rgb_to_uint8(camera_wrist.data.output["rgb"])
                    left = _rgb_to_uint8(camera_back.data.output["rgb"])
                    right = _rgb_to_uint8(camera_front.data.output["rgb"])
                    for vkey in video_modality_keys:
                        frame = _pick_video_for_key(vkey, wrist=wrist, left=left, right=right)
                        video_dict[vkey] = frame[None, None, ...].astype(np.uint8)
                except Exception as e:
                    print(f"[WARN] failed to read cameras: {e}")
                    video_dict = None

                if video_dict is not None:
                    joint_pos_np = robot.data.joint_pos[0, arm_joint_ids].detach().cpu().numpy().astype(np.float32)
                    gripper_joint = float(robot.data.joint_pos[0, hand_joint_ids[0]].detach().cpu().item()) if hand_joint_ids else 0.0
                    gripper_frac = float(np.clip(gripper_joint / 0.04, 0.0, 1.0))

                    state_dict = {}
                    for skey in state_modality_keys:
                        sval = _pick_state_for_key(skey, joint_pos_np=joint_pos_np, gripper_frac=gripper_frac)
                        state_dict[skey] = sval[None, None, :].astype(np.float32)

                    lang_str = LANGUAGE_OVERRIDE if (LANGUAGE_OVERRIDE is not None) else TASK_DESCRIPTION
                    language_dict = {language_key: [[str(lang_str)]]}

                    obs = {"video": video_dict, "state": state_dict, "language": language_dict}

                    try:
                        pred_action, _info = policy.infer_action_chunk(obs)

                        # normalize to (H,7)
                        if isinstance(pred_action, dict):
                            if "action" in pred_action:
                                arr = np.array(pred_action["action"], dtype=np.float32)
                                chunk = arr[0] if arr.ndim == 3 else arr
                            elif "action.ee_delta" in pred_action and "action.gripper_pos" in pred_action:
                                ee = np.array(pred_action["action.ee_delta"], dtype=np.float32)
                                gr = np.array(pred_action["action.gripper_pos"], dtype=np.float32)
                                ee = ee[0] if ee.ndim == 3 else ee
                                gr = gr[0] if gr.ndim == 3 else gr
                                chunk = np.concatenate([ee, gr], axis=-1)
                            elif "ee_delta" in pred_action and "gripper_pos" in pred_action:
                                ee = np.array(pred_action["ee_delta"], dtype=np.float32)
                                gr = np.array(pred_action["gripper_pos"], dtype=np.float32)
                                ee = ee[0] if ee.ndim == 3 else ee
                                gr = gr[0] if gr.ndim == 3 else gr
                                chunk = np.concatenate([ee, gr], axis=-1)
                            else:
                                chunk = None
                        else:
                            arr = np.array(pred_action, dtype=np.float32)
                            chunk = arr[0] if arr.ndim == 3 else arr

                        if chunk is not None and chunk.ndim == 2 and chunk.shape[1] == 7:
                            action_windows_full = chunk
                            # Use full horizon predicted by the model (clipped only by actual chunk length).
                            num_tokens = min(action_horizon, chunk.shape[0])
                            action_windows = chunk[:num_tokens]
                            num_windows = action_windows.shape[0]
                            steps_per_window = steps_per_action  # one action token per control tick
                            current_window_idx = 0
                            steps_in_window = 0
                            print(f"[INFO] got action_windows: H={num_windows}, steps_per_window={steps_per_window}")
                        else:
                            print(f"[WARN] invalid action shape: {getattr(chunk,'shape',None)}")
                    except Exception as e:
                        print(f"[WARN] local GR00T inference failed: {e}")

                steps_since_infer = 0
            else:
                steps_since_infer += 1

            control_tick = (count % steps_per_action) == 0  # update target only at control tick

                        # apply current action window (patched: update token+target only at control tick)
            # NOTE:
            # - GR00T returns a chunk (H_pred=16), we execute only action_windows (H=EXEC_HORIZON=8)  [1](blob:https://www.microsoft365.com/68498443-4572-4d6f-9c10-40fcacef96ea)
            # - steps_per_action=5 (20Hz) so one token should be applied once per 5 sim steps, not every sim step  [1](blob:https://www.microsoft365.com/68498443-4572-4d6f-9c10-40fcacef96ea)

            if action_windows is not None and num_windows is not None:
                # initialize last gripper (hold across sim steps)
                if not hasattr(robot, "_last_grip_open"):
                    robot._last_grip_open = 1.0
                
                # Only advance token & update IK target at control tick (20Hz)
                if control_tick:
                    # pick current token (no intra-token interpolation; target is held for steps_per_action sim steps)
                    interp = action_windows[current_window_idx]  # (7,)
                
                    # advance token index (one per control tick)
                    if current_window_idx < num_windows - 1:
                        current_window_idx += 1
                    # else: stay at last token; next inference will refresh windows anyway
                
                    dp = interp[:3].astype(np.float32)
                    drot = interp[3:6].astype(np.float32)  # rotvec
                    rot_norm = float(np.linalg.norm(drot))
                    if rot_norm > ROT_CLAMP_RAD and rot_norm > 1e-6:
                        drot = drot * (ROT_CLAMP_RAD / rot_norm)
                    grip = float(interp[6])
                
                    # read current EE pose ONCE at control tick, then compute target_pose_w and hold it
                    env_ids = torch.arange(scene.num_envs, device=sim.device, dtype=torch.long)
                    ee_pose_w = robot.data.body_state_w[env_ids, robot_entity_cfg.body_ids[0], :7]
                    ee_pos_w = ee_pose_w[:, 0:3]
                    ee_quat_w = ee_pose_w[:, 3:7]
                
                    # smooth deltas (only updated at control tick)
                    if not hasattr(robot, "_smoothed_dp"):
                        robot._smoothed_dp = torch.tensor(dp, device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
                        robot._smoothed_drot = torch.tensor(drot, device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
                    smooth_a = 0.35
                    dp_t = torch.tensor(dp, device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
                    drot_t = torch.tensor(drot, device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
                    robot._smoothed_dp = (1.0 - smooth_a) * robot._smoothed_dp + smooth_a * dp_t
                    robot._smoothed_drot = (1.0 - smooth_a) * robot._smoothed_drot + smooth_a * drot_t
                
                    target_pos_w = ee_pos_w + robot._smoothed_dp
                    dq_xyzw = Rotation.from_rotvec(robot._smoothed_drot[0].detach().cpu().numpy()).as_quat()
                    dq_wxyz = np.array([dq_xyzw[3], dq_xyzw[0], dq_xyzw[1], dq_xyzw[2]], dtype=np.float32)
                    dq = torch.tensor(dq_wxyz, device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
                    target_quat_w = quat_mul_wxyz(dq, ee_quat_w)
                    target_quat_w = target_quat_w / torch.linalg.norm(target_quat_w, dim=-1, keepdim=True).clamp(min=1e-6)
                
                    target_pose_w = torch.cat([target_pos_w, target_quat_w], dim=-1)
                    diff_ik.set_command(target_pose_w)  # ✅ only here
                
                    robot._last_grip_open = float(np.clip(grip, 0.0, 1.0))
                
                # IK compute EVERY sim step, but toward the held target (diff_ik's internal command)
                env_ids = torch.arange(scene.num_envs, device=sim.device, dtype=torch.long)
                ee_pose_w = robot.data.body_state_w[env_ids, robot_entity_cfg.body_ids[0], :7]
                ee_pos_w = ee_pose_w[:, 0:3]
                ee_quat_w = ee_pose_w[:, 3:7]
                ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
                jacobian = robot.root_physx_view.get_jacobians()[env_ids, ee_jacobi_idx, :, :]
                jacobian = jacobian[:, :, arm_joint_ids]
                joint_pos = robot.data.joint_pos[env_ids, arm_joint_ids]
                joint_pos_arm = diff_ik.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos)
                
                prev = robot.data.joint_pos_target[env_ids, arm_joint_ids]
                max_step = 0.05
                delta = torch.clamp(joint_pos_arm - prev, -max_step, max_step)
                joint_pos_arm = prev + delta
                
                finger_joint = float(robot._last_grip_open) * 0.04
                joint_pos_gripper = torch.full((scene.num_envs, len(hand_joint_ids)), finger_joint, device=sim.device, dtype=torch.float32)
                joint_pos_des = torch.cat([joint_pos_arm, joint_pos_gripper], dim=-1)

            if debug_compare_file is not None and (count % max(1, COMPARE_DEBUG_INTERVAL) == 0):
                has_windows = action_windows is not None and num_windows is not None
                ee_pose_dbg = robot.data.body_state_w[0, robot_entity_cfg.body_ids[0], :7]
                ee_pos_dbg = ee_pose_dbg[0:3].detach().cpu().numpy().astype(np.float32)
                ee_quat_dbg = ee_pose_dbg[3:7].detach().cpu().numpy().astype(np.float32)
                ee_rpy_dbg = Rotation.from_quat([ee_quat_dbg[1], ee_quat_dbg[2], ee_quat_dbg[3], ee_quat_dbg[0]]).as_euler("xyz").astype(np.float32)
                now_ts = time.time()
                wall_dt = now_ts - debug_compare_last_ts
                debug_compare_last_ts = now_ts
                effective_control_hz = 1.0 / (sim_dt * float(steps_per_action)) if steps_per_action > 0 else 0.0
                payload = {
                    "step": int(count),
                    "env_id": 0,
                    "sim_dt_effective": float(sim_dt),
                    "api_call_interval": int(inference_interval),
                    "has_windows": bool(has_windows),
                    "num_windows": int(num_windows) if num_windows is not None else -1,
                    "window_idx": int(current_window_idx),
                    "steps_in_window": int(steps_in_window),
                    "steps_per_window": int(steps_per_action),
                    "steps_since_infer": int(steps_since_infer),
                    "api_cooldown": 0,
                    "api_disabled": False,
                    "effective_control_hz": float(effective_control_hz),
                    "ee_pos_w": ee_pos_dbg.tolist(),
                    "ee_rpy": ee_rpy_dbg.tolist(),
                    "wall_dt": float(wall_dt),
                    "source": "my_test_script",
                }
                with open(debug_compare_file, "a", encoding="utf-8") as dbg_f:
                    dbg_f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# apply targets
            robot.set_joint_position_target(joint_pos_des, joint_ids=all_joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            try:
                curr_cube_z = float(cube.data.root_state_w[0, 2].detach().cpu().item())
                if curr_cube_z > max_cube_z:
                    max_cube_z = curr_cube_z
            except Exception:
                pass

            count += 1
            if count >= MAX_STEPS:
                print(f"[INFO] reached max steps ({MAX_STEPS}), stopping.")
                break
    finally:
        try:
            right_writer.close()
            print(f"[INFO] Saved right-view video to: {video_path}")
        except Exception as e:
            print(f"[WARN] Failed to close right-view writer: {e}")

        # Close debug writers if any
        for name, w in [
            ("left_view", debug_left_writer),
            ("right_view_debug", debug_right_writer),
            ("wrist_view", debug_wrist_writer),
        ]:
            if w is None:
                continue
            try:
                w.close()
                print(f"[INFO] Saved debug {name} video under {debug_dir}")
            except Exception as e:
                print(f"[WARN] Failed to close debug writer {name}: {e}")

    lift_delta = float(max_cube_z - initial_cube_z)
    success = bool(lift_delta >= LIFT_SUCCESS_THRESHOLD)
    return {
        "episode": csv_dir.name,
        "initial_cube_z": float(initial_cube_z),
        "max_cube_z": float(max_cube_z),
        "lift_delta": float(lift_delta),
        "lift_success": success,
        "threshold": float(LIFT_SUCCESS_THRESHOLD),
        "video_path": str(video_path),
    }


def main():
    data_root = Path(DATA_DIR).expanduser().resolve()
    csv_arg = Path(CSV_BASE_DIR).expanduser().resolve() if CSV_BASE_DIR is not None else None
    split_path = Path(SPLIT_FILE).expanduser().resolve() if SPLIT_FILE is not None else (data_root / "train_test_split.txt")

    if csv_arg is not None:
        csv_dirs = [csv_arg]
    else:
        if not data_root.exists():
            raise FileNotFoundError(f"data_dir does not exist: {data_root}")
        if not split_path.exists():
            raise FileNotFoundError(f"split file does not exist: {split_path}")
        training_eps = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1].lower() == "training":
                    training_eps.append(parts[0])
        if not training_eps:
            raise RuntimeError(f"No training episodes found in split file: {split_path}")
        csv_dirs = [data_root / ep for ep in training_eps if (data_root / ep).is_dir()]
        if not csv_dirs:
            raise RuntimeError(
                f"No training episode directories found under data_dir={data_root} using split={split_path}"
            )
        if RL_BOOTSTRAP and RL_EPISODE_LIMIT > 0:
            csv_dirs = csv_dirs[:RL_EPISODE_LIMIT]

    out_root_name = "outputs_gr00t_rl_bootstrap" if RL_BOOTSTRAP else "outputs_gr00t"
    out_root = Path(__file__).resolve().parents[2] / out_root_name
    out_root.mkdir(parents=True, exist_ok=True)
    debug_root = out_root / "debug" if DEBUG_MODE else None
    success_log_csv = Path(SUCCESS_LOG_CSV).expanduser().resolve() if SUCCESS_LOG_CSV else (out_root / "training_lift_success_log.csv")

    print("[INFO] setup complete.")
    print(f"[INFO] local model_path={MODEL_PATH}")
    print(f"[INFO] embodiment_tag={EMBODIMENT_TAG}")
    print(f"[INFO] policy_device={POLICY_DEVICE}")
    print(f"[INFO] policy_strict={POLICY_STRICT}")
    print(f"[INFO] data_dir={data_root}")
    print(f"[INFO] split_file={split_path}")
    print(f"[INFO] training episodes={len(csv_dirs)}")
    if csv_arg is not None:
        print(f"[INFO] single csv_dir={csv_arg}")
    print(f"[INFO] language_override={LANGUAGE_OVERRIDE}  target_action_fps={TARGET_ACTION_FPS}  rot_clamp_rad={ROT_CLAMP_RAD}")
    print(f"[INFO] rl_bootstrap={RL_BOOTSTRAP}")
    print(f"[INFO] rl_episode_limit={RL_EPISODE_LIMIT}")
    print(f"[INFO] rl_disable_retries={RL_DISABLE_RETRIES}")
    print(f"[INFO] lift_success_threshold={LIFT_SUCCESS_THRESHOLD}")
    print(f"[INFO] max_attempts_per_episode={MAX_ATTEMPTS_PER_EPISODE}")
    print(f"[INFO] success_log_csv={success_log_csv}")
    print(f"[INFO] heartbeat_log_interval={HEARTBEAT_LOG_INTERVAL}")
    print(f"[INFO] max_steps={MAX_STEPS}")
    _log_heartbeat(0, tag="before_sim_init")

    # Create a single sim + scene and reuse them across episodes
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = _make_gr00t_scene_cfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    _log_heartbeat(0, tag="after_sim_reset")

    if MODEL_PATH is None:
        raise ValueError("--model_path is required for local inference mode")
    resolved_tag = _resolve_embodiment_tag(EMBODIMENT_TAG)
    policy_impl = Gr00tPolicy(
        embodiment_tag=resolved_tag,
        model_path=str(Path(MODEL_PATH).expanduser().resolve()),
        device=POLICY_DEVICE,
        strict=POLICY_STRICT,
    )
    policy = LocalGr00tChunkPolicyAdapter(policy_impl)
    print("[INFO] Local GR00T policy adapter loaded.")
    _log_heartbeat(0, tag="after_policy_load")

    success_rows = []
    for ep_dir in csv_dirs:
        trial_results = []
        any_success = False
        first_success_try = None

        attempts_this_episode = 1 if RL_DISABLE_RETRIES else max(1, MAX_ATTEMPTS_PER_EPISODE)
        for attempt in range(1, attempts_this_episode + 1):
            out_path = out_root / f"{ep_dir.name}.try{attempt:02d}.mp4"
            debug_dir = (debug_root / ep_dir.name / f"try{attempt:02d}") if debug_root is not None else None
            print(f"[INFO] Running episode={ep_dir.name} attempt={attempt}/{attempts_this_episode} -> video={out_path}")
            if debug_dir is not None:
                print(f"[INFO] Debug videos (left/right/wrist) will be saved under: {debug_dir}")

            result = run_simulator(sim, scene, ep_dir, out_path, policy, debug_dir)
            result["attempt"] = attempt
            trial_results.append(result)
            print(
                f"[RESULT][TRY] episode={result['episode']} attempt={attempt} success={result['lift_success']} "
                f"lift_delta={result['lift_delta']:.6f} max_z={result['max_cube_z']:.6f} init_z={result['initial_cube_z']:.6f}"
            )

            if result["lift_success"]:
                any_success = True
                first_success_try = attempt
                print(f"[INFO] episode={ep_dir.name} succeeded at attempt={attempt}; moving to next episode.")
                break

        best_trial = max(trial_results, key=lambda r: r["lift_delta"]) if trial_results else None
        episode_row = {
            "episode": ep_dir.name,
            "lift_success": bool(any_success),
            "attempts_run": len(trial_results),
            "max_attempts": int(attempts_this_episode),
            "first_success_try": int(first_success_try) if first_success_try is not None else -1,
            "best_lift_delta": float(best_trial["lift_delta"]) if best_trial is not None else float("nan"),
            "best_max_cube_z": float(best_trial["max_cube_z"]) if best_trial is not None else float("nan"),
            "best_initial_cube_z": float(best_trial["initial_cube_z"]) if best_trial is not None else float("nan"),
            "threshold": float(LIFT_SUCCESS_THRESHOLD),
            "trial_details_json": json.dumps(trial_results, ensure_ascii=False),
        }

        success_rows.append(episode_row)
        pd.DataFrame(success_rows).to_csv(success_log_csv, index=False)
        print(
            f"[RESULT][EPISODE] episode={episode_row['episode']} success={episode_row['lift_success']} "
            f"attempts_run={episode_row['attempts_run']}/{episode_row['max_attempts']} "
            f"best_lift_delta={episode_row['best_lift_delta']:.6f}"
        )
        print(f"[INFO] Updated success log: {success_log_csv}")

    num_success = sum(1 for r in success_rows if r["lift_success"])
    print(f"[SUMMARY] success {num_success}/{len(success_rows)}")
    print(f"[SUMMARY] saved success log: {success_log_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[FATAL] Unhandled exception in main()")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()