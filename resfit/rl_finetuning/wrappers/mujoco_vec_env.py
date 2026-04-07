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
import time
from concurrent.futures import ThreadPoolExecutor
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

# Depth normalization defaults (2nd/98th percentile from IsaacLab rollout)
DEPTH_NORM_DEFAULTS = {
    "front": {"min": 0.554, "max": 1.495},
    "wrist": {"min": 0.056, "max": 0.711},
    "back":  {"min": 0.720, "max": 1.495},
}

# ── Wrist camera v2: direct MuJoCo quat (15° tilt toward gripper) ──
def _wrist_cam_mj_quat():
    tilt = np.radians(15)
    view_dir = np.array([np.sin(tilt), 0, np.cos(tilt)])
    up_dir = np.array([1, 0, 0])
    z = -view_dir / np.linalg.norm(view_dir)
    x = np.cross(up_dir, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz

WRIST_CAM_MJ_QUAT = _wrist_cam_mj_quat()
WRIST_CAM_MJ_POS = np.array([-0.08, 0.0, 0.0])


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
        random_cube_range: dict | None = None,
        hover_offset: dict | None = None,
        cube_size: tuple[float, float, float] = CUBE_HALF_SIZE,
        scene_xml: str | None = None,
        calib_path: str | None = None,
        max_episode_steps: int = 300,
        success_threshold: float = LIFT_REWARD_THRESHOLD_M,
        success_lift_threshold: float = LIFT_SUCCESS_THRESHOLD_M,
        reward_type: str = "sparse",
        device: str = "cuda:0",
        rl_img_size: int = RL_IMG_SIZE,
        groot_img_size: int = 256,
        depth_norm: dict[str, dict[str, float]] | None = None,
    ):
        setup_egl()

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.success_lift_threshold = success_lift_threshold
        self.reward_type = reward_type
        self.device = device
        self.rl_img_size = rl_img_size
        self.groot_img_size = groot_img_size
        self.cube_size = cube_size
        self._depth_norm = depth_norm or DEPTH_NORM_DEFAULTS

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

        # Random cube placement config (if set, overrides fixed positions on reset)
        self._random_cube_range = random_cube_range
        # Hover offset: {"xy": (dx, dy) or "random_xy": radius, "z": float or "random_z": (lo, hi)}
        self._hover_offset = hover_offset  # {"dx": (-0.08, 0.14), "dy": (-0.15, 0.15), "yaw": (-10, 10)}
        self._cube_base_pos = np.array([0.45, -0.05, 0.02])

        # ── Create N environments ──
        self._envs: list[dict[str, Any]] = []
        for i in range(num_envs):
            env = self._create_single_env(cube_positions[i], scene_xml)
            self._envs.append(env)

        # ── Build gymnasium spaces ──
        self._state_dim = 10  # matches IfaceEnvWrapper: eef_pos(3)+eef_quat(4)+grip(2)+contact(1)
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

        # Depth observation spaces
        for depth_key in ["observation.depth.front", "observation.depth.wrist"]:
            obs_spaces[depth_key] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(num_envs, 1, rl_img_size, rl_img_size), dtype=np.float32,
            )
        # Object state (cube pose: pos3 + quat_wxyz4 = 7D)
        obs_spaces["observation.object_state"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_envs, 7), dtype=np.float32,
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

        # ── Wrist cam v2: override quat/pos ──
        model.cam_quat[cam_wrist_id] = WRIST_CAM_MJ_QUAT
        model.cam_pos[cam_wrist_id] = WRIST_CAM_MJ_POS

        # Hide hand/link6/link7 visual geoms (group 2 → 4) for wrist cam
        hide_body_ids = []
        for bname in ("hand", "fr3_link6", "fr3_link7"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                hide_body_ids.append(bid)
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] in hide_body_ids and model.geom_group[gi] == 2:
                model.geom_group[gi] = 4

        # Scene options for cam_base (show hand, hide collision) and cam_wrist (hide hand)
        opt_base = mujoco.MjvOption()
        opt_base.geomgroup[3] = 0   # hide collision geoms
        opt_base.geomgroup[4] = 1   # show hand/link geoms
        opt_wrist = mujoco.MjvOption()
        opt_wrist.geomgroup[3] = 0  # hide collision geoms
        opt_wrist.geomgroup[4] = 0  # hide hand/link geoms

        # Persistent renderers — GR00T cameras at RENDER_W×RENDER_H (640×360), RL at rl_img_size
        _rs = self.rl_img_size
        renderer_base = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)   # GR00T cam_base
        renderer_wrist = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)  # GR00T cam_wrist
        renderer_front = mujoco.Renderer(model, height=_rs, width=_rs)  # RL front (already target size)

        # Depth renderers at rl_img_size (no resize needed)
        depth_renderer_front = mujoco.Renderer(model, height=_rs, width=_rs)
        depth_renderer_front.enable_depth_rendering()
        depth_renderer_wrist = mujoco.Renderer(model, height=_rs, width=_rs)
        depth_renderer_wrist.enable_depth_rendering()

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
            "depth_renderer_front": depth_renderer_front,
            "depth_renderer_wrist": depth_renderer_wrist,
            "cam_base_id": cam_base_id,
            "cam_wrist_id": cam_wrist_id,
            "opt_base": opt_base,
            "opt_wrist": opt_wrist,
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
        self, actions: torch.Tensor, render_mode: str = "full",
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

        _t_ik = time.time()
        def _step_single_env(i):
            action = actions_np[i]
            self._apply_action(i, action[:3], action[3:7], float(action[7]))
            self._step_counts[i] += 1
            self._last_actions[i] = action
            rewards[i] = self._compute_reward(i)
            if self._is_success(i):
                terminated[i] = True
            if self._step_counts[i] >= self.max_episode_steps:
                truncated[i] = True

        if self.num_envs > 1 and hasattr(self, "_thread_pool"):
            list(self._thread_pool.map(_step_single_env, range(self.num_envs)))
        else:
            for i in range(self.num_envs):
                _step_single_env(i)
        _dt_ik = time.time() - _t_ik

        # Build observations (terminal state, before any auto-reset)
        _t_obs = time.time()
        obs_dict = self._build_obs_dict(render_mode=render_mode)
        _dt_obs = time.time() - _t_obs

        if not hasattr(self, "_vec_timers"):
            self._vec_timers = {"ik_reward": [], "build_obs": []}
        self._vec_timers["ik_reward"].append(_dt_ik)
        self._vec_timers["build_obs"].append(_dt_obs)
        if len(self._vec_timers["ik_reward"]) % 500 == 0:
            ik = self._vec_timers["ik_reward"][-500:]
            ob = self._vec_timers["build_obs"][-500:]
            print(f"[vecenv-timing] ik+reward={sum(ik)/len(ik)*1000:.1f}ms "
                  f"build_obs(render)={sum(ob)/len(ob)*1000:.1f}ms", flush=True)

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

        # Render cameras (groot_img_size × groot_img_size RGB)
        env["renderer_base"].update_scene(data, camera=env["cam_base_id"],
                                           scene_option=env["opt_base"])
        img_base = env["renderer_base"].render().copy()

        env["renderer_wrist"].update_scene(data, camera=env["cam_wrist_id"],
                                            scene_option=env["opt_wrist"])
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

        # Reset cube to initial position (randomize if configured)
        cube_qposadr = env["cube_qposadr"]
        if cube_qposadr is not None:
            if self._random_cube_range is not None:
                rng = self._random_cube_range
                dx = np.random.uniform(rng["dx"][0], rng["dx"][1])
                dy = np.random.uniform(rng["dy"][0], rng["dy"][1])
                cube_pos = self._cube_base_pos.copy()
                cube_pos[0] += dx
                cube_pos[1] += dy
                env["cube_pos_init"] = cube_pos
                # Random yaw
                yaw_lo, yaw_hi = rng.get("yaw", (0, 0))
                yaw = np.random.uniform(yaw_lo, yaw_hi)
                q_xyzw = Rotation.from_euler("z", np.radians(yaw)).as_quat()
                cube_quat = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
                data.qpos[cube_qposadr + 3:cube_qposadr + 7] = cube_quat
            else:
                data.qpos[cube_qposadr + 3:cube_qposadr + 7] = self._cube_quat_wxyz
            data.qpos[cube_qposadr:cube_qposadr + 3] = env["cube_pos_init"]

        # Reset velocities
        data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)

        # IK to hover above cube (with optional offset)
        hover_x = env["cube_pos_init"][0]
        hover_y = env["cube_pos_init"][1]
        hover_z = 0.25
        if self._hover_offset is not None:
            ho = self._hover_offset
            if "xy" in ho:
                hover_x += ho["xy"][0]
                hover_y += ho["xy"][1]
            elif "random_xy" in ho:
                r = ho["random_xy"]
                hover_x += np.random.uniform(-r, r)
                hover_y += np.random.uniform(-r, r)
            if "z" in ho:
                hover_z = ho["z"]
            elif "random_z" in ho:
                hover_z = np.random.uniform(ho["random_z"][0], ho["random_z"][1])
        hover_pos = np.array([hover_x, hover_y, hover_z])
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
        model, data, ids = env["model"], env["data"], env["ids"]
        cube_qposadr = env["cube_qposadr"]
        if cube_qposadr is None:
            return 0.0

        cube_z = data.qpos[cube_qposadr + 2]
        lift_delta = cube_z - self._initial_cube_z[env_idx]

        if self.reward_type == "sparse":
            return 1.0 if lift_delta >= self.success_threshold else 0.0

        elif self.reward_type == "dense":
            # High-bonus dense reward (v1 style): success/height dominate distance
            tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
            cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3]
            finger_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))

            grasped = float(env["grasp_state"]["grasped"])

            distance_reward = (1.0 - np.tanh(finger_cube_dist / 0.1)) * 1.0
            contact_reward = grasped * 2.0
            height_reward = (
                float(lift_delta > 0.005)
                * np.tanh(lift_delta / 0.1)
                * 100.0
                * grasped
            )
            success_reward = float(lift_delta >= self.success_threshold) * 100.0 * grasped

            return float(distance_reward + contact_reward + height_reward + success_reward)

        elif self.reward_type == "dense_clipped":
            # 4-stage shaped reward matching IsaacLab v31a
            tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
            cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3]
            finger_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))

            grasped = float(env["grasp_state"]["grasped"])

            distance_reward = (1.0 - np.tanh(finger_cube_dist / 0.1)) * 0.1
            contact_reward = grasped * 0.2
            height_reward = (
                float(lift_delta > 0.005)
                * np.tanh(lift_delta / 0.1)
                * 0.5
                * grasped
            )
            success_reward = float(lift_delta >= self.success_threshold) * 1.0 * grasped

            return float(np.clip(
                distance_reward + contact_reward + height_reward + success_reward,
                0.0, 1.0,
            ))

        else:  # "dense"
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

    def _build_obs_dict(self, render_mode: str = "full") -> dict[str, torch.Tensor]:
        """Build RL observation dict for all environments.

        render_mode:
            "full" - render everything (RGB + depth)
            "rl_only" - skip RGB cameras, only render depth (for RL training steps)
            "none" - skip all rendering (state-only obs)
        """
        states = []
        images: dict[str, list[np.ndarray]] = {k: [] for k in self.CAMERA_MAP.values()}
        depth_images: dict[str, list[np.ndarray]] = {
            "observation.depth.front": [],
            "observation.depth.wrist": [],
        }
        object_states = []
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

            # ── Camera images (skip if rl_only or none) ──
            if render_mode == "full":
                self._render_cameras(i, images)
            else:
                # Fill with zeros for keys that expect images
                for key in images:
                    images[key].append(np.zeros((3, self.rl_img_size, self.rl_img_size), dtype=np.uint8))

            # ── Depth images (skip if none) ──
            if render_mode != "none":
                self._render_depth(i, depth_images)
            else:
                for key in depth_images:
                    depth_images[key].append(np.zeros((1, self.rl_img_size, self.rl_img_size), dtype=np.float32))

            # ── Object state (cube pose: pos3 + quat_wxyz4) ──
            object_states.append(self._get_object_state(i))

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

        # Depth images
        for key, frames in depth_images.items():
            stacked = np.stack(frames)  # (N, 1, H, W)
            out[key] = torch.as_tensor(stacked, device=self.device, dtype=torch.float32)

        # Object state
        out["observation.object_state"] = torch.as_tensor(
            np.stack(object_states), device=self.device, dtype=torch.float32,
        )

        return out

    def _build_state(self, env_idx: int) -> np.ndarray:
        """Build 10-D state vector matching IfaceEnvWrapper convention.

        [eef_pos(3), eef_quat_xyzw(4), gripper_qpos(2), contact_force(1)]
        """
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # EEF position (3) + orientation as quaternion xyzw (4)
        tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
        eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()  # xyzw

        # Gripper joint positions (2 fingers)
        grip = []
        for fid in ids["finger_ids"]:
            if fid >= 0:
                grip.append(data.qpos[model.jnt_qposadr[fid]])
            else:
                grip.append(0.04)

        # Contact force (scalar): 1.0 if grasped, else 0.0
        contact_force = 1.0 if env["grasp_state"]["grasped"] else 0.0

        state = np.concatenate([
            tcp_pos.astype(np.float32),           # 3
            eef_quat_xyzw.astype(np.float32),     # 4
            np.array(grip, dtype=np.float32),      # 2
            np.array([contact_force], dtype=np.float32),  # 1
        ])  # = 10
        return state

    def _render_cameras(
        self,
        env_idx: int,
        images_out: dict[str, list[np.ndarray]],
    ) -> None:
        """Render RGB cameras. front at rl_img_size, cam_base/wrist at groot_img_size."""
        env = self._envs[env_idx]
        data = env["data"]

        camera_config = [
            ("front", env["renderer_front"], env["front_cam_id"], None),
            ("cam_base", env["renderer_base"], env["cam_base_id"], env["opt_base"]),
            ("cam_wrist", env["renderer_wrist"], env["cam_wrist_id"], env["opt_wrist"]),
        ]

        for mj_key, renderer, cam_id, opt in camera_config:
            resfit_key = self.CAMERA_MAP[mj_key]
            if cam_id >= 0:
                kw = {"scene_option": opt} if opt else {}
                renderer.update_scene(data, camera=cam_id, **kw)
                img_rgb = renderer.render().copy()
            else:
                img_rgb = np.zeros((self.rl_img_size, self.rl_img_size, 3), dtype=np.uint8)

            # Resize to rl_img_size if renderer is larger (GR00T cams)
            h, w = img_rgb.shape[:2]
            if h != self.rl_img_size or w != self.rl_img_size:
                img_rgb = cv2.resize(
                    img_rgb, (self.rl_img_size, self.rl_img_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            img_chw = np.transpose(img_rgb, (2, 0, 1))
            images_out[resfit_key].append(img_chw)

    def _render_depth(
        self,
        env_idx: int,
        depth_out: dict[str, list[np.ndarray]],
    ) -> None:
        """Render depth for front and wrist cameras, normalize to [0,1]."""
        env = self._envs[env_idx]
        data = env["data"]

        depth_config = [
            ("front", env["depth_renderer_front"], env["front_cam_id"], None),
            ("wrist", env["depth_renderer_wrist"], env["cam_wrist_id"], env["opt_wrist"]),
        ]

        for cam_name, renderer, cam_id, opt in depth_config:
            key = f"observation.depth.{cam_name}"
            if cam_id >= 0:
                kw = {"scene_option": opt} if opt else {}
                renderer.update_scene(data, camera=cam_id, **kw)
                depth_raw = renderer.render().copy()  # (rl_img_size, rl_img_size) float32
            else:
                depth_raw = np.zeros((self.rl_img_size, self.rl_img_size), dtype=np.float32)

            # Normalize using per-camera min/max (no resize — renderer already at rl_img_size)
            norm = self._depth_norm.get(cam_name, {"min": 0.0, "max": 1.0})
            d_min, d_max = norm["min"], norm["max"]
            depth_raw = np.clip(depth_raw, d_min, d_max)
            depth_raw = (depth_raw - d_min) / max(d_max - d_min, 1e-6)

            # Handle NaN/Inf
            depth_raw = np.nan_to_num(depth_raw, nan=0.0, posinf=1.0, neginf=0.0)

            # (1, H, W) for channel dimension
            depth_out[key].append(depth_raw[np.newaxis].astype(np.float32))

    def _get_object_state(self, env_idx: int) -> np.ndarray:
        """Get cube pose as 7D vector: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]."""
        env = self._envs[env_idx]
        qa = env["cube_qposadr"]
        if qa is not None:
            pos = env["data"].qpos[qa:qa + 3].copy()
            quat_wxyz = env["data"].qpos[qa + 3:qa + 7].copy()
            return np.concatenate([pos, quat_wxyz]).astype(np.float32)
        return np.zeros(7, dtype=np.float32)

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
            for key in ("renderer_base", "renderer_wrist", "renderer_front",
                        "depth_renderer_front", "depth_renderer_wrist"):
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
            "front": (env["renderer_front"], env["front_cam_id"], None),
            "back": (env["renderer_base"], env["cam_base_id"], env["opt_base"]),
            "wrist": (env["renderer_wrist"], env["cam_wrist_id"], env["opt_wrist"]),
        }
        renderer, cam_id, opt = cam_map.get(camera, cam_map["front"])
        kw = {"scene_option": opt} if opt else {}
        renderer.update_scene(env["data"], camera=cam_id, **kw)
        frame = renderer.render().copy()
        if size is not None:
            frame = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        return frame
