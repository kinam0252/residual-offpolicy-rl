"""
MuJoCo vectorised environment for Franka FR3 Stack Cube task.

Independent environment — does NOT modify Lift or PnP code.
Uses make_model_with_two_cubes() from utils_stack.py.

Key differences from PnP env:
- Two cubes (white=pick, green=target) instead of cube+bowl
- Gripper in RAW METERS (0.02–0.04), NOT normalised 0–1
- FPS = 15 (matches GR00T training data)
- Success = white cube stacked on green cube
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# ── Import shared utilities ──
# Mujoco_Franka/src/utils.py is at Repos/Intern/Mujoco_Franka/src/
_MUJOCO_FRANKA_UTILS = str(Path(__file__).resolve().parents[4] / "Mujoco_Franka" / "src")
if _MUJOCO_FRANKA_UTILS not in sys.path:
    sys.path.insert(0, os.path.abspath(_MUJOCO_FRANKA_UTILS))

# Stack-specific physics (utils_stack, scene XML) are in MSRA/stack_cube/physics/
_STACK_PHYSICS = str(Path(__file__).resolve().parents[4] / "MSRA" / "stack_cube" / "physics")
if _STACK_PHYSICS not in sys.path:
    sys.path.insert(0, _STACK_PHYSICS)

# EGL must be set before mujoco import
os.environ.setdefault("MUJOCO_GL", "egl")
_gl_path = os.path.expanduser("~/.local/lib/gl")
if os.path.isdir(_gl_path):
    os.environ["LD_LIBRARY_PATH"] = _gl_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

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
    solve_ik,
    _wrist_cam_xml,
    _bind_wrist_cam,
    INTRINSICS,
)

from utils_stack import make_model_with_two_cubes  # noqa: E402

# ── Constants (matching GR00T training data) ──
RENDER_W = 640
RENDER_H = 360
RL_IMG_SIZE = 84
FPS = 15  # Stack training data is 15Hz (NOT 20Hz like PnP)
CUBE_HALF = 0.02  # 4cm cube

DEPTH_NORM_DEFAULTS = {
    "front": {"min": 0.554, "max": 1.495},
    "wrist": {"min": 0.056, "max": 0.711},
}

# Gripper: RAW METERS (not normalised!)
GRIPPER_MAX_WIDTH = 0.04   # fully open in meters
GRIPPER_MIN_WIDTH = 0.0    # fully closed
GRIPPER_CLOSE_THRESHOLD = 0.028  # from batch_replay_stack.py

# Physics
PHYSICS_SUBSTEPS = None  # Computed at runtime: round(1 / (FPS * model.opt.timestep))
GRASP_CONTACT_THRESHOLD = 1

# Stacking success criteria (from batch_replay_stack.py)
STACK_Z_TARGET = CUBE_HALF * 3  # 0.06 (green cube top + white cube half)
STACK_Z_ERR_THRESHOLD = 0.015   # 15mm
STACK_XY_ERR_THRESHOLD = 0.020  # 20mm
CUBE_SETTLED_VEL = 0.05  # m/s


class MuJoCoVecEnvStack:
    """Vectorised MuJoCo environment for Franka FR3 cube stacking.

    White cube (freejoint) must be picked and placed on top of green cube (static).
    Gripper uses raw meter values matching GR00T training data.
    """

    CAMERA_MAP = {
        "front": "observation.images.front",
        "cam_base": "observation.images.back",
        "cam_wrist": "observation.images.wrist",
    }

    def __init__(
        self,
        num_envs: int = 1,
        white_cube_positions: list[list[float]] | np.ndarray | None = None,
        green_cube_positions: list[list[float]] | np.ndarray | None = None,
        white_cube_quats_wxyz: list[list[float]] | None = None,
        green_cube_quats_wxyz: list[list[float]] | None = None,
        scene_xml: str | None = None,
        calib_path: str | None = None,
        max_episode_steps: int = 500,
        reward_type: str = "sparse",
        device: str = "cuda:0",
        rl_img_size: int = RL_IMG_SIZE,
        groot_img_size: int = 256,
        random_cube_range: dict | list | None = None,
        depth_norm: dict[str, dict[str, float]] | None = None,
    ):
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.device = device
        self.rl_img_size = rl_img_size
        self.groot_img_size = groot_img_size
        self._depth_norm = depth_norm or DEPTH_NORM_DEFAULTS

        # Random perturbation on reset
        # Format: [white_range, green_range] where each is
        #   {"dx": [lo, hi], "dy": [lo, hi], "yaw": [lo, hi]} (metres, degrees)
        #   or None (no perturbation for that cube)
        self._random_cube_range = random_cube_range

        # Camera calibration
        self._T_base_cam = load_calib(calib_path)

        # Default positions if not provided
        if white_cube_positions is None:
            white_cube_positions = [[0.45, 0.22, CUBE_HALF]] * num_envs
        if green_cube_positions is None:
            green_cube_positions = [[0.45, -0.06, CUBE_HALF]] * num_envs

        self._white_positions = np.asarray(white_cube_positions, dtype=np.float64)
        self._green_positions = np.asarray(green_cube_positions, dtype=np.float64)

        # Pad positions to num_envs
        if self._white_positions.shape[0] < num_envs:
            self._white_positions = np.tile(self._white_positions, (num_envs, 1))[:num_envs]
        if self._green_positions.shape[0] < num_envs:
            self._green_positions = np.tile(self._green_positions, (num_envs, 1))[:num_envs]

        # Quaternions (wxyz for MuJoCo)
        self._white_quats = white_cube_quats_wxyz or [[1, 0, 0, 0]] * num_envs
        self._green_quats = green_cube_quats_wxyz or [[1, 0, 0, 0]] * num_envs

        # Scene XML (use stack_cube's own franka model)
        self._scene_xml = scene_xml or str(
            Path(__file__).resolve().parents[4] / "MSRA" / "stack_cube" / "physics" / "franka_fr3" / "fr3_with_hand.xml"
        )

        # ── Create N environments ──
        self._envs: list[dict[str, Any]] = []
        for i in range(num_envs):
            env = self._init_single_env(
                self._white_positions[i],
                self._green_positions[i],
                self._white_quats[i] if i < len(self._white_quats) else [1, 0, 0, 0],
                self._green_quats[i] if i < len(self._green_quats) else [1, 0, 0, 0],
            )
            self._envs.append(env)

        # ── Gymnasium spaces ──
        # State: eef_pos(3) + eef_quat(4) + gripper_raw(2) + contact(1) = 10D
        self._state_dim = 10
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
        for depth_key in ["observation.depth.front", "observation.depth.wrist"]:
            obs_spaces[depth_key] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(num_envs, 1, rl_img_size, rl_img_size), dtype=np.float32,
            )
        # Object state: white_cube(7) + green_cube(3) = 10D
        obs_spaces["observation.object_state"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_envs, 10), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Action: 8D absolute [pos3, quat4, gripper_raw1]
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_envs, 8), dtype=np.float32,
        )

        # Per-env state
        self._step_counts = np.zeros(num_envs, dtype=np.int64)
        self._initial_white_z = np.zeros(num_envs, dtype=np.float64)
        self._last_actions = np.zeros((num_envs, 8), dtype=np.float32)

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------

    def _init_single_env(
        self,
        white_pos: np.ndarray,
        green_pos: np.ndarray,
        white_quat_wxyz: list[float],
        green_quat_wxyz: list[float],
    ) -> dict:
        """Create a single MuJoCo environment with two cubes."""
        model = make_model_with_two_cubes(
            self._T_base_cam,
            white_cube_pos=white_pos,
            green_cube_pos=green_pos,
            white_cube_quat_wxyz=white_quat_wxyz,
            green_cube_quat_wxyz=green_quat_wxyz,
            cube_half=CUBE_HALF,
            scene_xml=self._scene_xml,
        )

        data = mujoco.MjData(model)
        ids = get_model_ids(model)

        # Camera IDs
        cam_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_base")
        cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
        front_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")

        # Hide hand/link6/link7 for wrist cam (group 2 → 4)
        hide_body_ids = []
        for bname in ("hand", "fr3_link6", "fr3_link7"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                hide_body_ids.append(bid)
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] in hide_body_ids and model.geom_group[gi] == 2:
                model.geom_group[gi] = 4

        # Scene options
        opt_base = mujoco.MjvOption()
        opt_base.geomgroup[3] = 0
        opt_base.geomgroup[4] = 1
        opt_wrist = mujoco.MjvOption()
        opt_wrist.geomgroup[3] = 0
        opt_wrist.geomgroup[4] = 0

        # Persistent renderers
        _rs = self.rl_img_size
        renderer_base = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        renderer_wrist = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        renderer_front = mujoco.Renderer(model, height=_rs, width=_rs)

        # Depth renderers
        depth_renderer_front = mujoco.Renderer(model, height=_rs, width=_rs)
        depth_renderer_front.enable_depth_rendering()
        depth_renderer_wrist = mujoco.Renderer(model, height=_rs, width=_rs)
        depth_renderer_wrist.enable_depth_rendering()

        # White cube (freejoint)
        white_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "white_cube_joint")
        white_qposadr = model.jnt_qposadr[white_jnt_id] if white_jnt_id >= 0 else None
        white_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "white_cube_geom")

        # Green cube (static — no joint)
        green_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_cube")

        # Finger geom ids for contact detection
        finger_geom_ids = []
        for body_name in ("left_finger", "right_finger"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                continue
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid and model.geom_group[gid] == 3:
                    finger_geom_ids.append(gid)
        if not finger_geom_ids:
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if name and "finger" in name.lower():
                    finger_geom_ids.append(gid)
        print(f"[Stack] Found {len(finger_geom_ids)} finger collision geom(s)")

        # Actuator IDs
        arm_actuator_ids = []
        for jid in ids["jnt_ids"]:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            arm_actuator_ids.append(aid)

        # Set finger friction for stable grasping (tuned from PnP experience)
        for gid in finger_geom_ids:
            model.geom_friction[gid] = [5.0, 0.5, 0.1]
            model.geom_condim[gid] = 6

        # Increase cube friction for physics-based grasping
        # (batch_replay uses kinematic grasp so low friction was fine there)
        for cname in ("white_cube_geom", "green_cube_geom"):
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cname)
            if cid >= 0:
                model.geom_friction[cid] = [5.0, 0.5, 0.1]
                model.geom_condim[cid] = 6

        # Increase finger kp for grasp force (now using new finger_joint1/2 actuators)
        finger_actuator_ids = []
        for fname in ("finger_joint1", "finger_joint2"):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, fname)
            if aid >= 0:
                finger_actuator_ids.append(aid)

        # Disable the original 'gripper' actuator if present (ctrlrange [0,255])
        orig_grip_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        if orig_grip_aid >= 0:
            model.actuator_gainprm[orig_grip_aid, 0] = 0.0
            model.actuator_biasprm[orig_grip_aid, :] = 0.0

        # Extend finger joint range
        for fname in ("finger_joint1", "finger_joint2"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
            if jid >= 0:
                model.jnt_range[jid] = [-0.01, 0.04]

        # Set initial positions
        if white_qposadr is not None:
            data.qpos[white_qposadr:white_qposadr + 3] = white_pos
            data.qpos[white_qposadr + 3:white_qposadr + 7] = white_quat_wxyz

        # Reset robot to home
        for i, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04
        mujoco.mj_forward(model, data)

        # Compute physics substeps from model timestep (matches batch_replay_stack.py:212)
        n_substeps = int(round(1.0 / (FPS * model.opt.timestep)))

        # Weld constraint for sustained grasp
        weld_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "cube_grasp")
        if weld_eq_id >= 0:
            model.eq_active0[weld_eq_id] = 0
            data.eq_active[weld_eq_id] = 0

        return {
            "model": model,
            "data": data,
            "ids": ids,
            "n_substeps": n_substeps,
            "renderer_base": renderer_base,
            "renderer_wrist": renderer_wrist,
            "renderer_front": renderer_front,
            "depth_renderer_front": depth_renderer_front,
            "depth_renderer_wrist": depth_renderer_wrist,
            "cam_base_id": cam_base_id,
            "cam_wrist_id": cam_wrist_id,
            "front_cam_id": front_cam_id,
            "opt_base": opt_base,
            "opt_wrist": opt_wrist,
            "white_qposadr": white_qposadr,
            "white_pos_init": white_pos.copy(),
            "green_pos_init": green_pos.copy(),
            "white_quat_init": list(white_quat_wxyz),
            "green_quat_init": list(green_quat_wxyz),
            "white_geom_id": white_geom_id,
            "green_body_id": green_body_id,
            "finger_geom_ids": finger_geom_ids,
            "grasp_state": {"grasped": False, "contact_count": 0},
            "arm_actuator_ids": arm_actuator_ids,
            "finger_actuator_ids": finger_actuator_ids,
            "weld_eq_id": weld_eq_id,
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        for i in range(self.num_envs):
            self._reset_single_env(i)
        return self._build_obs_dict(), {}

    def reset_envs(self, env_ids: list[int]) -> None:
        for eid in env_ids:
            self._reset_single_env(eid)

    def step(
        self, actions: torch.Tensor, render_mode: str = "full",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step all environments with absolute EEF targets.

        actions: (num_envs, 8) — [pos3, quat_xyzw4, gripper_width_raw1]
        gripper_width is in RAW METERS (0.0–0.04).
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
            self._apply_action(i, action[:3], action[3:7], float(action[7]))
            self._step_counts[i] += 1
            self._last_actions[i] = action
            rewards[i] = self._compute_reward(i)
            if self._is_success(i):
                terminated[i] = True
            if self._step_counts[i] >= self.max_episode_steps:
                truncated[i] = True

        obs_dict = self._build_obs_dict(render_mode=render_mode)
        info: dict[str, Any] = {}

        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        terminated_t = torch.as_tensor(terminated, device=self.device, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, device=self.device, dtype=torch.bool)

        return obs_dict, rewards_t, terminated_t, truncated_t, info

    # ------------------------------------------------------------------
    # GR00T observation (raw meter gripper!)
    # ------------------------------------------------------------------

    def get_groot_obs(self, env_idx: int,
                      task_str: str = "pick up the white cube and stack it on the green cube") -> dict:
        """Build GR00T-format observation. Gripper is in RAW METERS."""
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # Render cameras
        env["renderer_base"].update_scene(data, camera=env["cam_base_id"],
                                           scene_option=env["opt_base"])
        img_base = env["renderer_base"].render().copy()

        env["renderer_wrist"].update_scene(data, camera=env["cam_wrist_id"],
                                            scene_option=env["opt_wrist"])
        img_wrist = env["renderer_wrist"].render().copy()

        # TCP state
        tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
        eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()

        # Gripper width in RAW METERS (NOT normalised!)
        finger_id = ids["finger_ids"][0]
        if finger_id >= 0:
            gripper_width_raw = float(data.qpos[model.jnt_qposadr[finger_id]])
            gripper_width_raw = max(0.0, gripper_width_raw)  # clamp negative
        else:
            gripper_width_raw = GRIPPER_MAX_WIDTH

        observation = {
            "video": {
                "cam_base": img_base[np.newaxis, np.newaxis].astype(np.uint8),
                "cam_wrist": img_wrist[np.newaxis, np.newaxis].astype(np.uint8),
            },
            "state": {
                "proprio.eef_pos": tcp_pos[np.newaxis, np.newaxis].astype(np.float32),
                "proprio.eef_quat": eef_quat_xyzw[np.newaxis, np.newaxis].astype(np.float32),
                "proprio.gripper_width": np.array([[[gripper_width_raw]]], dtype=np.float32),
            },
            "language": {
                "annotation.human.action.task_description": [[task_str]],
            },
        }
        return observation

    # ------------------------------------------------------------------
    # Internal: action application (physics-based)
    # ------------------------------------------------------------------

    def _apply_action(
        self,
        env_idx: int,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        gripper_width_raw: float,
    ) -> None:
        """Apply action. gripper_width_raw is in METERS (0.0–0.04)."""
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # Normalise quaternion
        qn = np.linalg.norm(target_quat_xyzw)
        if qn > 1e-6:
            target_quat_xyzw = target_quat_xyzw / qn

        # IK on scratch state
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()

        solve_ik(
            model, data, ids["hand_id"], ids["jnt_ids"],
            target_pos, target_quat_xyzw, max_iter=50,
        )

        target_joint_pos = np.array([
            data.qpos[model.jnt_qposadr[jid]] for jid in ids["jnt_ids"]
        ])

        # Restore
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_forward(model, data)

        # Arm actuators
        for aid, target_q in zip(env["arm_actuator_ids"], target_joint_pos):
            if aid >= 0:
                data.ctrl[aid] = target_q

        # Finger actuators: raw meter gripper → finger joint target
        # NO offset — matches batch_replay_stack.py (line 259: fp = clip(gw, 0.0, 0.04))
        finger_target = np.clip(gripper_width_raw, 0.0, GRIPPER_MAX_WIDTH)
        for aid in env["finger_actuator_ids"]:
            data.ctrl[aid] = finger_target

        # Physics simulation
        for _ in range(env["n_substeps"]):
            mujoco.mj_step(model, data)

        # Update grasp state
        self._update_grasp_state(env_idx, gripper_width_raw)

    def _update_grasp_state(self, env_idx: int, gripper_width_raw: float) -> None:
        """Update grasp state from physics contacts + weld constraint."""
        env = self._envs[env_idx]
        model, data = env["model"], env["data"]
        grasp_state = env["grasp_state"]
        contacts = self._get_contacts(env_idx)
        any_contact = contacts[2] > 0.5
        gripper_closing = gripper_width_raw < GRIPPER_CLOSE_THRESHOLD

        if gripper_closing and any_contact:
            grasp_state["contact_count"] = min(grasp_state["contact_count"] + 1, 10)
        else:
            grasp_state["contact_count"] = max(grasp_state["contact_count"] - 2, 0)

        was_grasped = grasp_state["grasped"]
        if was_grasped and gripper_closing:
            grasp_state["grasped"] = True
        else:
            grasp_state["grasped"] = (
                grasp_state["contact_count"] >= GRASP_CONTACT_THRESHOLD
                and gripper_closing
            )

        # Enable/disable weld constraint for sustained grasp
        weld_id = env.get("weld_eq_id", -1)
        if weld_id >= 0:
            if grasp_state["grasped"] and not was_grasped:
                # Just grasped: compute relative pose and enable weld
                hand_id = env["ids"]["hand_id"]
                white_body_id = model.geom_bodyid[env["white_geom_id"]]
                hand_pos = data.xpos[hand_id].copy()
                hand_mat = data.xmat[hand_id].reshape(3, 3).copy()
                cube_pos = data.xpos[white_body_id].copy()
                cube_mat = data.xmat[white_body_id].reshape(3, 3).copy()
                rel_pos = hand_mat.T @ (cube_pos - hand_pos)
                rel_mat = hand_mat.T @ cube_mat
                rel_quat = np.zeros(4)
                mujoco.mju_mat2Quat(rel_quat, rel_mat.flatten())
                model.eq_data[weld_id, 3:6] = rel_pos
                model.eq_data[weld_id, 6:10] = rel_quat
                model.eq_active0[weld_id] = 1
                data.eq_active[weld_id] = 1
            elif not grasp_state["grasped"] and was_grasped:
                model.eq_active0[weld_id] = 0
                data.eq_active[weld_id] = 0

    # ------------------------------------------------------------------
    # Internal: reset
    # ------------------------------------------------------------------

    def _reset_single_env(self, env_idx: int) -> None:
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # Robot to home
        for i, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04

        # Apply random perturbation if configured
        white_pos = env["white_pos_init"].copy()
        white_quat = env["white_quat_init"].copy()
        green_pos = env["green_pos_init"].copy()

        if self._random_cube_range is not None:
            rcr = self._random_cube_range
            # rcr can be [white_range, green_range] or a single dict (applied to both)
            if isinstance(rcr, list):
                white_range = rcr[0] if len(rcr) > 0 else None
                green_range = rcr[1] if len(rcr) > 1 else None
            else:
                white_range = rcr
                green_range = None

            if white_range is not None:
                dx = np.random.uniform(*white_range["dx"])
                dy = np.random.uniform(*white_range["dy"])
                white_pos[0] += dx
                white_pos[1] += dy
                yaw_lo, yaw_hi = white_range.get("yaw", (0, 0))
                if yaw_lo != 0 or yaw_hi != 0:
                    yaw = np.random.uniform(yaw_lo, yaw_hi)
                    q_xyzw = Rotation.from_euler("z", np.radians(yaw)).as_quat()
                    white_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

            if green_range is not None:
                dx = np.random.uniform(*green_range["dx"])
                dy = np.random.uniform(*green_range["dy"])
                green_pos[0] += dx
                green_pos[1] += dy

        # White cube position
        qa = env["white_qposadr"]
        if qa is not None:
            data.qpos[qa:qa + 3] = white_pos
            data.qpos[qa + 3:qa + 7] = white_quat

        # Green cube position (static body)
        green_body_id = env["green_body_id"]
        if green_body_id >= 0:
            model.body_pos[green_body_id] = green_pos

        data.qvel[:] = 0.0

        # Deactivate weld on reset
        weld_id = env.get("weld_eq_id", -1)
        if weld_id >= 0:
            model.eq_active0[weld_id] = 0
            data.eq_active[weld_id] = 0

        env["grasp_state"] = {"grasped": False, "contact_count": 0}
        mujoco.mj_forward(model, data)

        if qa is not None:
            self._initial_white_z[env_idx] = data.qpos[qa + 2]

        self._step_counts[env_idx] = 0
        self._last_actions[env_idx] = 0.0
        env["grasp_state"] = {"grasped": False, "contact_count": 0}

        for aid, jid in zip(env["arm_actuator_ids"], ids["jnt_ids"]):
            if aid >= 0:
                data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
        for aid in env["finger_actuator_ids"]:
            data.ctrl[aid] = 0.04

    # ------------------------------------------------------------------
    # Internal: reward
    # ------------------------------------------------------------------

    def _compute_reward(self, env_idx: int) -> float:
        if self.reward_type == "sparse":
            return 1.0 if self._is_success(env_idx) else 0.0

        # Dense stacking reward (0~1.0)
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]
        qa = env["white_qposadr"]
        if qa is None:
            return 0.0

        white_pos = data.qpos[qa:qa + 3].copy()
        tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
        grasped = env["grasp_state"]["grasped"]

        green_body_id = env["green_body_id"]
        green_pos = model.body_pos[green_body_id].copy() if green_body_id >= 0 else env["green_pos_init"]

        tcp_white_dist = float(np.linalg.norm(tcp_pos - white_pos))

        # Target: on top of green cube (green center + 2 * cube_half)
        green_top = green_pos.copy()
        green_top[2] += CUBE_HALF * 2
        white_to_target = float(np.linalg.norm(white_pos - green_top))

        # Stage 1: Approach white cube (max 0.15)
        approach = (1.0 - np.tanh(tcp_white_dist / 0.1)) * 0.15

        # Stage 2: Grasp (0.10 bonus)
        grasp = 0.10 if grasped else 0.0

        # Stage 3: Proximity — white cube → green top (max 0.40)
        proximity = 0.0
        if grasped:
            proximity = (1.0 - np.tanh(white_to_target / 0.08)) * 0.40

        # Stage 4: Stack success (0.35 bonus)
        success = 0.35 if self._is_success(env_idx) else 0.0

        total = approach + grasp + proximity + success
        return float(np.clip(total, 0.0, 1.0))

    def _is_success(self, env_idx: int) -> bool:
        """White cube stacked on green cube.

        Success = white cube is above green cube + physical contact between them.
        """
        env = self._envs[env_idx]
        model, data = env["model"], env["data"]
        qa = env["white_qposadr"]
        if qa is None:
            return False

        white_pos = data.qpos[qa:qa + 3].copy()
        green_body_id = env["green_body_id"]
        green_pos = model.body_pos[green_body_id].copy() if green_body_id >= 0 else env["green_pos_init"]

        # White cube must be above green cube
        if white_pos[2] <= green_pos[2]:
            return False

        # Physical contact between white and green cubes
        white_gid = env["white_geom_id"]
        green_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_cube_geom")
        for ci in range(data.ncon):
            c = data.contact[ci]
            if (c.geom1 == white_gid and c.geom2 == green_gid) or \
               (c.geom1 == green_gid and c.geom2 == white_gid):
                return True

        return False

    # ------------------------------------------------------------------
    # Internal: contact detection
    # ------------------------------------------------------------------

    def _get_contacts(self, env_idx: int) -> np.ndarray:
        """Detect finger-white_cube contacts. Returns [left, right, any]."""
        env = self._envs[env_idx]
        data = env["data"]
        white_geom_id = env["white_geom_id"]
        finger_geom_ids = env["finger_geom_ids"]

        if white_geom_id < 0 or not finger_geom_ids:
            return np.zeros(3, dtype=np.float32)

        left_c = 0.0
        right_c = 0.0
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if white_geom_id in (g1, g2):
                other = g2 if g1 == white_geom_id else g1
                if other in finger_geom_ids:
                    idx = finger_geom_ids.index(other)
                    if idx % 2 == 0:
                        left_c = 1.0
                    else:
                        right_c = 1.0

        return np.array([left_c, right_c, min(left_c, right_c)], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal: observation building
    # ------------------------------------------------------------------

    def _build_obs_dict(self, render_mode: str = "full") -> dict[str, torch.Tensor]:
        states = []
        images: dict[str, list[np.ndarray]] = {k: [] for k in self.CAMERA_MAP.values()}
        depth_images: dict[str, list[np.ndarray]] = {
            "observation.depth.front": [],
            "observation.depth.wrist": [],
        }
        object_states = []

        for i in range(self.num_envs):
            env = self._envs[i]
            model, data, ids = env["model"], env["data"], env["ids"]

            # State (10D)
            tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
            eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()
            grip = []
            for fid in ids["finger_ids"]:
                if fid >= 0:
                    grip.append(data.qpos[model.jnt_qposadr[fid]])
                else:
                    grip.append(GRIPPER_MAX_WIDTH)
            contact = 1.0 if env["grasp_state"]["grasped"] else 0.0
            state = np.concatenate([
                tcp_pos.astype(np.float32),
                eef_quat_xyzw.astype(np.float32),
                np.array(grip, dtype=np.float32),
                np.array([contact], dtype=np.float32),
            ])
            states.append(state)

            # Images
            if render_mode == "full":
                self._render_cameras(i, images)
                self._render_depth(i, depth_images)
            elif render_mode == "rl_only":
                for key in images:
                    images[key].append(np.zeros((3, self.rl_img_size, self.rl_img_size), dtype=np.uint8))
                self._render_depth(i, depth_images)
            else:
                for key in images:
                    images[key].append(np.zeros((3, self.rl_img_size, self.rl_img_size), dtype=np.uint8))
                for key in depth_images:
                    depth_images[key].append(np.zeros((1, self.rl_img_size, self.rl_img_size), dtype=np.float32))

            # Object state: white_cube(7) + green_cube_pos(3)
            qa = env["white_qposadr"]
            white_state = np.zeros(7, dtype=np.float32)
            if qa is not None:
                white_state[:3] = data.qpos[qa:qa + 3]
                white_state[3:7] = data.qpos[qa + 3:qa + 7]
            green_body_id = env["green_body_id"]
            green_pos = model.body_pos[green_body_id].copy().astype(np.float32) if green_body_id >= 0 else np.array(env["green_pos_init"], dtype=np.float32)
            object_states.append(np.concatenate([white_state, green_pos]))

        out: dict[str, torch.Tensor] = {}
        out["observation.state"] = torch.as_tensor(np.stack(states), device=self.device, dtype=torch.float32)
        for key, frames in images.items():
            out[key] = torch.as_tensor(np.stack(frames), device=self.device, dtype=torch.uint8)
        for key, frames in depth_images.items():
            out[key] = torch.as_tensor(np.stack(frames), device=self.device, dtype=torch.float32)
        out["observation.object_state"] = torch.as_tensor(np.stack(object_states), device=self.device, dtype=torch.float32)
        return out

    def _render_cameras(self, env_idx: int, images_out: dict[str, list[np.ndarray]]) -> None:
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
            h, w = img_rgb.shape[:2]
            if h != self.rl_img_size or w != self.rl_img_size:
                img_rgb = cv2.resize(img_rgb, (self.rl_img_size, self.rl_img_size), interpolation=cv2.INTER_LINEAR)
            images_out[resfit_key].append(np.transpose(img_rgb, (2, 0, 1)))

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
                depth_raw = renderer.render().copy()
            else:
                depth_raw = np.zeros((self.rl_img_size, self.rl_img_size), dtype=np.float32)

            norm = self._depth_norm.get(cam_name, {"min": 0.0, "max": 1.0})
            d_min, d_max = norm["min"], norm["max"]
            depth_raw = np.clip(depth_raw, d_min, d_max)
            depth_raw = (depth_raw - d_min) / max(d_max - d_min, 1e-6)
            depth_raw = np.nan_to_num(depth_raw, nan=0.0, posinf=1.0, neginf=0.0)

            depth_out[key].append(depth_raw[np.newaxis].astype(np.float32))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def fps(self) -> int:
        return FPS

    def get_frame(self, env_id: int, camera: str = "front",
                  size: tuple[int, int] = (360, 640)) -> np.ndarray:
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

    def get_white_cube_pos(self, env_idx: int) -> np.ndarray:
        env = self._envs[env_idx]
        qa = env["white_qposadr"]
        if qa is not None:
            return env["data"].qpos[qa:qa + 3].copy()
        return np.zeros(3)

    def get_green_cube_pos(self, env_idx: int) -> np.ndarray:
        env = self._envs[env_idx]
        gid = env["green_body_id"]
        if gid >= 0:
            return env["model"].body_pos[gid].copy()
        return np.array(env["green_pos_init"])

    def close(self) -> None:
        for env in self._envs:
            for key in ("renderer_base", "renderer_wrist", "renderer_front",
                        "depth_renderer_front", "depth_renderer_wrist"):
                renderer = env.get(key)
                if renderer is not None:
                    try:
                        renderer.close()
                    except Exception:
                        pass
