"""
MuJoCo vectorised environment wrapper for residual RL.

Exposes the **same interface** as ``IsaacLabVecEnvWrapper`` /
``IfaceEnvWrapper`` so that the existing TD3 training loop, QAgent,
and replay-buffer code work without modification.

Internally uses utilities from ``Mujoco_Franka/src/utils.py``
(IK solver, scene builder, rendering) and follows the inference
pattern of ``Mujoco_Franka/src/infer_gr00t_mujoco.py``.

Observation dict
----------------
* ``observation.state``  : (N, state_dim)  float32
* ``observation.images.front`` / ``back`` / ``wrist`` : (N, 3, 84, 84) uint8
* ``observation.raw_joint_pos`` : (N, 7) float32
* ``observation.raw_gripper_frac`` : (N, 1) float32

Action
------
8-D absolute EEF target: ``[pos_x, pos_y, pos_z, qx, qy, qz, qw, gripper_width]``
All values in world frame; gripper 0–1 (0 = closed, 1 = open).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# ── Import Mujoco_Franka utilities ──
_MUJOCO_FRANKA_SRC = str(Path(__file__).resolve().parents[4] / "Mujoco_Franka" / "src")
if _MUJOCO_FRANKA_SRC not in sys.path:
    sys.path.insert(0, _MUJOCO_FRANKA_SRC)

from utils import (  # noqa: E402
    CAM_H,
    CAM_W,
    HOME_QPOS,
    JOINT_LOWER,
    JOINT_UPPER,
    TCP_OFFSET,
    get_model_ids,
    get_tcp_pose,
    load_calib,
    load_calib_wrist,
    make_model_with_cube,
    set_robot_pose,
    setup_egl,
    solve_ik,
)

# ── Constants ──
RENDER_W = 640
RENDER_H = 360
RL_IMG_SIZE = 84
CUBE_HALF_SIZE = (0.06, 0.02, 0.02)  # 12×4×4 cm lying flat
GRIPPER_CLOSE_THRESHOLD = 0.5
LIFT_REWARD_THRESHOLD_M = 0.005  # 5 mm sparse reward threshold
LIFT_SUCCESS_THRESHOLD_M = 0.04  # 4 cm — episode terminates on success
FPS = 20


class MuJoCoVecEnv:
    """Vectorised MuJoCo environment for Franka FR3 cube-lift.

    Manages *N* independent ``(MjModel, MjData)`` pairs and persistent
    renderers.  Each environment has its own grasp-state tracker for
    kinematic cube attachment (same logic as ``infer_gr00t_mujoco.py``).
    """

    # Camera name mapping: matches IsaacLab wrapper convention
    CAMERA_MAP = {
        "front": "observation.images.front",
        "cam_base": "observation.images.back",
        "cam_wrist": "observation.images.wrist",
    }

    def __init__(
        self,
        num_envs: int = 1,
        cube_positions: list[list[float]] | np.ndarray | None = None,
        cube_yaw_deg: float = 0.0,
        cube_size: tuple[float, float, float] = CUBE_HALF_SIZE,
        scene_xml: str | None = None,
        calib_path: str | None = None,
        max_episode_steps: int = 300,
        success_threshold: float = LIFT_REWARD_THRESHOLD_M,
        success_lift_threshold: float = LIFT_SUCCESS_THRESHOLD_M,
        reward_type: str = "sparse",
        device: str = "cuda:0",
        rl_img_size: int = RL_IMG_SIZE,
    ):
        setup_egl()

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.success_lift_threshold = success_lift_threshold
        self.reward_type = reward_type
        self.device = device
        self.rl_img_size = rl_img_size
        self.cube_size = cube_size

        # Camera calibration
        self._T_base_cam = load_calib(calib_path)
        load_calib_wrist(calib_path)

        # Default cube positions if not provided
        if cube_positions is None:
            cube_positions = [[0.45, -0.05, 0.02]] * num_envs
        cube_positions = np.asarray(cube_positions, dtype=np.float64)
        if cube_positions.shape[0] < num_envs:
            cube_positions = np.tile(cube_positions, (num_envs, 1))[:num_envs]

        # Cube quaternion (wxyz for MuJoCo)
        yaw = np.radians(cube_yaw_deg)
        q_xyzw = Rotation.from_euler("z", yaw).as_quat()
        self._cube_quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        # ── Create N environments ──
        self._envs: list[dict[str, Any]] = []
        for i in range(num_envs):
            env = self._create_single_env(cube_positions[i], scene_xml)
            self._envs.append(env)

        # ── Build gymnasium spaces ──
        self._state_dim = 34  # matches IsaacLab: 9+9+7+3+6
        obs_spaces: dict[str, gym.spaces.Space] = {
            "observation.state": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(num_envs, self._state_dim), dtype=np.float32,
            ),
        }
        for _, resfit_key in self.CAMERA_MAP.items():
            obs_spaces[resfit_key] = gym.spaces.Box(
                low=0, high=255,
                shape=(num_envs, 3, rl_img_size, rl_img_size), dtype=np.uint8,
            )
            obs_spaces[f"{resfit_key}_hires"] = gym.spaces.Box(
                low=0, high=255,
                shape=(num_envs, 3, RENDER_H, RENDER_W), dtype=np.uint8,
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Action: 8-D absolute EEF target [pos3 + quat4 + grip1]
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_envs, 8), dtype=np.float32,
        )

        # Per-env state
        self._step_counts = np.zeros(num_envs, dtype=np.int64)
        self._initial_cube_z = np.zeros(num_envs, dtype=np.float64)
        self._last_actions = np.zeros((num_envs, 8), dtype=np.float32)

        # Joint scaling constants (for observation construction)
        self._joint_mid = (JOINT_UPPER + JOINT_LOWER) / 2.0
        self._joint_half_range = (JOINT_UPPER - JOINT_LOWER) / 2.0

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------

    def _create_single_env(self, cube_pos: np.ndarray, scene_xml: str | None) -> dict:
        """Create a single MuJoCo environment instance."""
        model = make_model_with_cube(
            self._T_base_cam,
            cube_pos=cube_pos,
            cube_quat_wxyz=self._cube_quat_wxyz,
            cube_size=self.cube_size,
            scene_xml=scene_xml,
        )
        data = mujoco.MjData(model)
        ids = get_model_ids(model)

        # Camera IDs
        cam_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_base")
        cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
        # Front camera (top-down-ish view)
        front_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")

        # Persistent renderers
        renderer_base = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        renderer_wrist = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        renderer_front = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

        # Cube joint address
        cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qposadr = model.jnt_qposadr[cube_jnt_id] if cube_jnt_id >= 0 else None

        # Cube geom id (for contact detection)
        cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        # Finger geom ids
        finger_geom_ids = []
        for name in ("finger_left_pad", "finger_right_pad",
                      "finger_left", "finger_right",
                      "left_finger_pad", "right_finger_pad"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                finger_geom_ids.append(gid)
        # Fallback: search by name pattern
        if not finger_geom_ids:
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if name and "finger" in name.lower():
                    finger_geom_ids.append(gid)

        # Set cube initial position
        if cube_qposadr is not None:
            data.qpos[cube_qposadr:cube_qposadr + 3] = cube_pos
            data.qpos[cube_qposadr + 3:cube_qposadr + 7] = self._cube_quat_wxyz

        # Reset robot to home pose
        for i, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04  # fully open
        mujoco.mj_forward(model, data)

        return {
            "model": model,
            "data": data,
            "ids": ids,
            "renderer_base": renderer_base,
            "renderer_wrist": renderer_wrist,
            "renderer_front": renderer_front,
            "cam_base_id": cam_base_id,
            "cam_wrist_id": cam_wrist_id,
            "front_cam_id": front_cam_id,
            "cube_qposadr": cube_qposadr,
            "cube_pos_init": cube_pos.copy(),
            "cube_geom_id": cube_geom_id,
            "finger_geom_ids": finger_geom_ids,
            "grasp_state": {"grasped": False, "T_cube_in_tcp": None},
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset all environments and return observation dict."""
        for i in range(self.num_envs):
            self._reset_single_env(i)

        obs_dict = self._build_obs_dict()
        return obs_dict, {}

    def reset_envs(self, env_ids: list[int]) -> None:
        """Reset specific environments (auto-reset on done)."""
        for eid in env_ids:
            self._reset_single_env(eid)

    def step(
        self, actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step all environments with absolute EEF targets.

        Parameters
        ----------
        actions : (num_envs, 8) — [pos3, quat_xyzw4, gripper_width1]
        """
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions)

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        for i in range(self.num_envs):
            action = actions_np[i]
            target_pos = action[:3]
            target_quat_xyzw = action[3:7]
            gripper_width = float(action[7])

            self._apply_action(i, target_pos, target_quat_xyzw, gripper_width)
            self._step_counts[i] += 1
            self._last_actions[i] = action

            # Reward
            rewards[i] = self._compute_reward(i)

            # Success termination: grasped + lift >= threshold
            if self._is_success(i):
                terminated[i] = True

            # Truncation (episode length limit)
            if self._step_counts[i] >= self.max_episode_steps:
                truncated[i] = True

        # Build observations (terminal state, before any auto-reset)
        obs_dict = self._build_obs_dict()

        info: dict[str, Any] = {}

        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        terminated_t = torch.as_tensor(terminated, device=self.device, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, device=self.device, dtype=torch.bool)

        return obs_dict, rewards_t, terminated_t, truncated_t, info

    # ------------------------------------------------------------------
    # GR00T observation (separate from RL obs)
    # ------------------------------------------------------------------

    def get_groot_obs(self, env_idx: int, task_str: str = "lift the cube") -> dict:
        """Build GR00T-format observation for a single environment.

        Returns the exact dict layout expected by ``Gr00tPolicy.get_action()``.
        """
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # Render cameras (640×360 RGB)
        env["renderer_base"].update_scene(data, camera=env["cam_base_id"])
        img_base = env["renderer_base"].render().copy()

        env["renderer_wrist"].update_scene(data, camera=env["cam_wrist_id"])
        img_wrist = env["renderer_wrist"].render().copy()

        # TCP state
        tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
        eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()

        # Gripper width (normalised 0–1)
        finger_id = ids["finger_ids"][0]
        if finger_id >= 0:
            finger_pos = data.qpos[model.jnt_qposadr[finger_id]]
            gripper_width = float(np.clip(finger_pos / 0.04, 0.0, 1.0))
        else:
            gripper_width = 1.0

        observation = {
            "video": {
                "cam_base": img_base[np.newaxis, np.newaxis].astype(np.uint8),
                "cam_wrist": img_wrist[np.newaxis, np.newaxis].astype(np.uint8),
            },
            "state": {
                "proprio.eef_pos": tcp_pos[np.newaxis, np.newaxis].astype(np.float32),
                "proprio.eef_quat": eef_quat_xyzw[np.newaxis, np.newaxis].astype(np.float32),
                "proprio.gripper_width": np.array([[[gripper_width]]], dtype=np.float32),
            },
            "language": {
                "annotation.human.action.task_description": [[task_str]],
            },
        }
        return observation

    # ------------------------------------------------------------------
    # Internal: action application (kinematic, same as infer_gr00t_mujoco.py)
    # ------------------------------------------------------------------

    def _apply_action(
        self,
        env_idx: int,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        gripper_width: float,
    ) -> None:
        """Apply a single action step via IK + kinematic cube attachment."""
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # Normalise quaternion
        qn = np.linalg.norm(target_quat_xyzw)
        if qn > 1e-6:
            target_quat_xyzw = target_quat_xyzw / qn

        # IK solve
        solve_ik(
            model, data, ids["hand_id"], ids["jnt_ids"],
            target_pos, target_quat_xyzw, max_iter=50,
        )

        # Set gripper
        finger_pos = np.clip(gripper_width, 0.0, 1.0) * 0.04
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = finger_pos

        # Kinematic cube attachment
        cube_qposadr = env["cube_qposadr"]
        grasp_state = env["grasp_state"]
        if cube_qposadr is not None:
            if not grasp_state["grasped"] and gripper_width < GRIPPER_CLOSE_THRESHOLD:
                # Gripper just closed → compute cube-in-TCP transform
                tcp_pos_now, tcp_R_now = get_tcp_pose(model, data, ids["hand_id"])
                T_tcp = np.eye(4)
                T_tcp[:3, :3] = tcp_R_now
                T_tcp[:3, 3] = tcp_pos_now

                cube_pos_now = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
                cube_qw = data.qpos[cube_qposadr + 3:cube_qposadr + 7].copy()
                cube_q_xyzw = [cube_qw[1], cube_qw[2], cube_qw[3], cube_qw[0]]
                T_cube = np.eye(4)
                T_cube[:3, :3] = Rotation.from_quat(cube_q_xyzw).as_matrix()
                T_cube[:3, 3] = cube_pos_now

                grasp_state["grasped"] = True
                grasp_state["T_cube_in_tcp"] = np.linalg.inv(T_tcp) @ T_cube

            if grasp_state["grasped"]:
                # Move cube with TCP
                tcp_pos_now, tcp_R_now = get_tcp_pose(model, data, ids["hand_id"])
                T_tcp = np.eye(4)
                T_tcp[:3, :3] = tcp_R_now
                T_tcp[:3, 3] = tcp_pos_now
                T_cube = T_tcp @ grasp_state["T_cube_in_tcp"]

                data.qpos[cube_qposadr:cube_qposadr + 3] = T_cube[:3, 3]
                q_c = Rotation.from_matrix(T_cube[:3, :3]).as_quat()  # xyzw
                data.qpos[cube_qposadr + 3:cube_qposadr + 7] = [
                    q_c[3], q_c[0], q_c[1], q_c[2],
                ]  # wxyz

        mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------
    # Internal: reset
    # ------------------------------------------------------------------

    def _reset_single_env(self, env_idx: int) -> None:
        """Reset a single environment to initial state."""
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # Reset robot to home pose
        for i, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04

        # Reset cube to initial position
        cube_qposadr = env["cube_qposadr"]
        if cube_qposadr is not None:
            data.qpos[cube_qposadr:cube_qposadr + 3] = env["cube_pos_init"]
            data.qpos[cube_qposadr + 3:cube_qposadr + 7] = self._cube_quat_wxyz

        # Reset velocities
        data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)

        # IK to hover above cube (same as infer_gr00t_mujoco.py)
        hover_pos = np.array([
            env["cube_pos_init"][0],
            env["cube_pos_init"][1],
            0.25,
        ])
        hover_quat = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_quat()  # point down
        solve_ik(
            model, data, ids["hand_id"], ids["jnt_ids"],
            hover_pos, hover_quat, max_iter=500, ns_gain=1.0,
        )
        mujoco.mj_forward(model, data)

        # Record initial cube z
        if cube_qposadr is not None:
            self._initial_cube_z[env_idx] = data.qpos[cube_qposadr + 2]
        else:
            self._initial_cube_z[env_idx] = 0.0

        # Reset state
        self._step_counts[env_idx] = 0
        self._last_actions[env_idx] = 0.0
        env["grasp_state"] = {"grasped": False, "T_cube_in_tcp": None}

    # ------------------------------------------------------------------
    # Internal: reward
    # ------------------------------------------------------------------

    def _compute_reward(self, env_idx: int) -> float:
        """Compute reward for a single environment."""
        env = self._envs[env_idx]
        cube_qposadr = env["cube_qposadr"]
        if cube_qposadr is None:
            return 0.0

        cube_z = env["data"].qpos[cube_qposadr + 2]
        lift_delta = cube_z - self._initial_cube_z[env_idx]

        if self.reward_type == "sparse":
            return 1.0 if lift_delta >= self.success_threshold else 0.0
        else:
            # Dense reward: height-based
            return float(np.clip(lift_delta * 100.0, 0.0, 1.0))

    def _is_success(self, env_idx: int) -> bool:
        """Check if cube is grasped and lifted >= success_lift_threshold."""
        env = self._envs[env_idx]
        cube_qposadr = env["cube_qposadr"]
        if cube_qposadr is None:
            return False

        if not env["grasp_state"]["grasped"]:
            return False

        cube_z = env["data"].qpos[cube_qposadr + 2]
        lift_delta = cube_z - self._initial_cube_z[env_idx]
        return lift_delta >= self.success_lift_threshold

    # ------------------------------------------------------------------
    # Internal: contact detection
    # ------------------------------------------------------------------

    def _get_contacts(self, env_idx: int) -> np.ndarray:
        """Detect finger-cube contacts. Returns [left, right, any] as float."""
        env = self._envs[env_idx]
        data = env["data"]
        cube_geom_id = env["cube_geom_id"]
        finger_geom_ids = env["finger_geom_ids"]

        if cube_geom_id < 0 or not finger_geom_ids:
            # Fallback: use gripper width as proxy
            finger_id = env["ids"]["finger_ids"][0]
            if finger_id >= 0:
                gw = data.qpos[env["model"].jnt_qposadr[finger_id]] / 0.04
                contact = 1.0 if gw < GRIPPER_CLOSE_THRESHOLD else 0.0
                return np.array([contact, contact, contact], dtype=np.float32)
            return np.zeros(3, dtype=np.float32)

        left_contact = 0.0
        right_contact = 0.0
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if cube_geom_id in (g1, g2):
                other = g2 if g1 == cube_geom_id else g1
                if other in finger_geom_ids:
                    idx = finger_geom_ids.index(other)
                    if idx % 2 == 0:
                        left_contact = 1.0
                    else:
                        right_contact = 1.0

        any_contact = max(left_contact, right_contact)
        return np.array([left_contact, right_contact, any_contact], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal: observation building
    # ------------------------------------------------------------------

    def _build_obs_dict(self) -> dict[str, torch.Tensor]:
        """Build RL observation dict for all environments."""
        states = []
        images: dict[str, list[np.ndarray]] = {k: [] for k in self.CAMERA_MAP.values()}
        images_hires: dict[str, list[np.ndarray]] = {f"{k}_hires": [] for k in self.CAMERA_MAP.values()}
        raw_joint_pos_list = []
        raw_gripper_frac_list = []

        for i in range(self.num_envs):
            env = self._envs[i]
            model, data, ids = env["model"], env["data"], env["ids"]

            # ── State (34D) ──
            state = self._build_state(i)
            states.append(state)

            # ── Raw joint pos / gripper ──
            joint_pos_7 = np.array([
                data.qpos[model.jnt_qposadr[jid]] for jid in ids["jnt_ids"]
            ], dtype=np.float32)
            raw_joint_pos_list.append(joint_pos_7)

            finger_id = ids["finger_ids"][0]
            if finger_id >= 0:
                gf = np.clip(data.qpos[model.jnt_qposadr[finger_id]] / 0.04, 0.0, 1.0)
            else:
                gf = 1.0
            raw_gripper_frac_list.append(np.array([gf], dtype=np.float32))

            # ── Camera images ──
            self._render_cameras(i, images, images_hires)

        out: dict[str, torch.Tensor] = {}

        # State
        out["observation.state"] = torch.as_tensor(
            np.stack(states), device=self.device, dtype=torch.float32,
        )

        # Raw joint pos / gripper
        out["observation.raw_joint_pos"] = torch.as_tensor(
            np.stack(raw_joint_pos_list), device=self.device, dtype=torch.float32,
        )
        out["observation.raw_gripper_frac"] = torch.as_tensor(
            np.stack(raw_gripper_frac_list), device=self.device, dtype=torch.float32,
        )

        # Images
        for key, frames in images.items():
            stacked = np.stack(frames)  # (N, 3, H, W)
            out[key] = torch.as_tensor(stacked, device=self.device, dtype=torch.uint8)
        for key, frames in images_hires.items():
            stacked = np.stack(frames)
            out[key] = torch.as_tensor(stacked, device=self.device, dtype=torch.uint8)

        return out

    def _build_state(self, env_idx: int) -> np.ndarray:
        """Build 34-D state vector matching IsaacLab convention.

        [dof_pos_scaled(9), dof_vel_scaled(9), cogact_ref(7), contact(3), ee_xyzrpy(6)]
        """
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # dof_pos_scaled: 7 arm joints + 2 finger joints → scaled to [-1, 1]
        arm_pos = np.array([
            data.qpos[model.jnt_qposadr[jid]] for jid in ids["jnt_ids"]
        ])
        arm_scaled = (arm_pos - self._joint_mid) / np.maximum(self._joint_half_range, 1e-8)

        # Finger joints (normalise 0..0.04 → -1..1)
        finger_pos = []
        for fid in ids["finger_ids"]:
            if fid >= 0:
                fp = data.qpos[model.jnt_qposadr[fid]]
                finger_pos.append((fp / 0.04) * 2.0 - 1.0)  # 0→-1, 0.04→1
            else:
                finger_pos.append(0.0)
        dof_pos_scaled = np.concatenate([arm_scaled, finger_pos]).astype(np.float32)  # (9,)

        # dof_vel_scaled: zeros for kinematic mode
        dof_vel_scaled = np.zeros(9, dtype=np.float32)

        # cogact_reference: last applied action (first 7 dims)
        cogact_ref = self._last_actions[env_idx, :7].copy()  # (7,)

        # contact: [left, right, any]
        contact = self._get_contacts(env_idx)  # (3,)

        # ee_xyzrpy: TCP position (3) + euler RPY (3)
        tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
        rpy = Rotation.from_matrix(tcp_R).as_euler("xyz")
        ee_xyzrpy = np.concatenate([tcp_pos, rpy]).astype(np.float32)  # (6,)

        state = np.concatenate([
            dof_pos_scaled,   # 9
            dof_vel_scaled,   # 9
            cogact_ref,       # 7
            contact,          # 3
            ee_xyzrpy,        # 6
        ])  # = 34
        return state.astype(np.float32)

    def _render_cameras(
        self,
        env_idx: int,
        images_out: dict[str, list[np.ndarray]],
        images_hires_out: dict[str, list[np.ndarray]],
    ) -> None:
        """Render all cameras for one environment and append to output lists."""
        env = self._envs[env_idx]
        data = env["data"]

        camera_config = [
            ("front", env["renderer_front"], env["front_cam_id"]),
            ("cam_base", env["renderer_base"], env["cam_base_id"]),
            ("cam_wrist", env["renderer_wrist"], env["cam_wrist_id"]),
        ]

        for mj_key, renderer, cam_id in camera_config:
            resfit_key = self.CAMERA_MAP[mj_key]
            if cam_id >= 0:
                renderer.update_scene(data, camera=cam_id)
                img_rgb = renderer.render().copy()  # (H, W, 3) RGB uint8
            else:
                img_rgb = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)

            # Hi-res: (3, H, W) CHW
            img_chw_hires = np.transpose(img_rgb, (2, 0, 1))  # (3, H, W)
            images_hires_out[f"{resfit_key}_hires"].append(img_chw_hires)

            # Downscaled to rl_img_size
            img_small = cv2.resize(
                img_rgb, (self.rl_img_size, self.rl_img_size),
                interpolation=cv2.INTER_LINEAR,
            )
            img_chw = np.transpose(img_small, (2, 0, 1))  # (3, rl_img_size, rl_img_size)
            images_out[resfit_key].append(img_chw)

    # ------------------------------------------------------------------
    # Cube state access
    # ------------------------------------------------------------------

    def get_cube_pos(self, env_idx: int) -> np.ndarray:
        """Get cube position (3,) in world frame."""
        env = self._envs[env_idx]
        qa = env["cube_qposadr"]
        if qa is not None:
            return env["data"].qpos[qa:qa + 3].copy()
        return np.zeros(3)

    def get_cube_quat_xyzw(self, env_idx: int) -> np.ndarray:
        """Get cube quaternion (4,) in xyzw convention."""
        env = self._envs[env_idx]
        qa = env["cube_qposadr"]
        if qa is not None:
            wxyz = env["data"].qpos[qa + 3:qa + 7].copy()
            return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        return np.array([0.0, 0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def fps(self) -> int:
        return FPS

    def close(self) -> None:
        """Release renderers."""
        for env in self._envs:
            for key in ("renderer_base", "renderer_wrist", "renderer_front"):
                renderer = env.get(key)
                if renderer is not None:
                    try:
                        renderer.close()
                    except Exception:
                        pass

    def render(self) -> np.ndarray:
        """Return (num_envs, H, W, 3) uint8 array for video recording."""
        frames = []
        for i in range(self.num_envs):
            env = self._envs[i]
            env["renderer_front"].update_scene(env["data"], camera=env["front_cam_id"])
            frame = env["renderer_front"].render().copy()
            frames.append(frame)
        return np.stack(frames)

    def get_frame(self, env_id: int, camera: str = "front", size: tuple[int, int] = (128, 160)) -> np.ndarray:
        """Get a single rendered frame for video/debug."""
        env = self._envs[env_id]
        cam_map = {
            "front": (env["renderer_front"], env["front_cam_id"]),
            "back": (env["renderer_base"], env["cam_base_id"]),
            "wrist": (env["renderer_wrist"], env["cam_wrist_id"]),
        }
        renderer, cam_id = cam_map.get(camera, cam_map["front"])
        renderer.update_scene(env["data"], camera=cam_id)
        frame = renderer.render().copy()
        if size is not None:
            frame = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        return frame
