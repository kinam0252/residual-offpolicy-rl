"""
MuJoCo vectorised environment for Franka FR3 Stand Cup task.

Cup is a freejoint body with real physics (collision-based grasp,
gravity-based release). GR00T drives the robot with absolute EEF targets.

Key differences from Close Drawer env:
- Cup with freejoint (real physics, NOT kinematic)
- Gripper is GR00T-controlled (NOT forced closed)
- Per-episode cup position/orientation from cup_positions.json
- Success = cup upright (local z · world z > threshold)
- Camera calib: shared PnP calib (camera_info_66ep.yaml)
- TABLE_Z_OFFSET = 0.02
- Physics: high friction fingers + cup for stable grasp
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Shared Franka utilities ──
_MUJOCO_FRANKA_UTILS = str(Path(__file__).resolve().parents[4] / "Mujoco_Franka" / "src")
if _MUJOCO_FRANKA_UTILS not in sys.path:
    sys.path.insert(0, os.path.abspath(_MUJOCO_FRANKA_UTILS))

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

from utils import (  # noqa: E402  — Mujoco_Franka/src/utils.py
    CAM_H,
    CAM_W,
    HOME_QPOS,
    TCP_OFFSET,
    INTRINSICS,
    get_model_ids,
    get_tcp_pose,
    load_calib,
    load_calib_wrist,
    solve_ik,
    _wrist_cam_xml,
    _bind_wrist_cam,
)

# ── Cup model constants (from utils_stand_cup.py) ──
CUP_TOP_RADIUS = 0.0375
CUP_BOTTOM_RADIUS = 0.025
CUP_HEIGHT = 0.10
CUP_MASS = 0.15
N_SLICES = 50
N_COLLISION_SLICES = 16


def _cup_visual_geoms_xml(n_slices=N_SLICES):
    """Visual-only cup geoms (no collision)."""
    geoms = []
    for i in range(n_slices):
        frac = (i + 0.5) / n_slices
        r = CUP_BOTTOM_RADIUS + frac * (CUP_TOP_RADIUS - CUP_BOTTOM_RADIUS)
        half_h = CUP_HEIGHT / n_slices / 2
        z = -CUP_HEIGHT / 2 + frac * CUP_HEIGHT
        geoms.append(
            f'<geom name="cup_v{i}" type="cylinder" size="{r:.5f} {half_h:.5f}" '
            f'pos="0 0 {z:.5f}" rgba="0.85 0.15 0.15 1.0" '
            f'mass="0.0001" contype="0" conaffinity="0"/>'
        )
    return '\n          '.join(geoms)


def _cup_collision_geoms_xml(n_slices=N_COLLISION_SLICES):
    """Collision-only cup geoms (invisible, group 3)."""
    geoms = []
    slice_mass = CUP_MASS / n_slices
    for i in range(n_slices):
        frac = (i + 0.5) / n_slices
        r = CUP_BOTTOM_RADIUS + frac * (CUP_TOP_RADIUS - CUP_BOTTOM_RADIUS)
        half_h = CUP_HEIGHT / n_slices / 2
        z = -CUP_HEIGHT / 2 + frac * CUP_HEIGHT
        geoms.append(
            f'<geom name="cup_c{i}" type="cylinder" size="{r:.5f} {half_h:.5f}" '
            f'pos="0 0 {z:.5f}" rgba="0.85 0.15 0.15 0.0" '
            f'mass="{slice_mass:.5f}" group="3" '
            f'contype="1" conaffinity="1" friction="0.8 0.005 0.0001" '
            f'solref="-10000 -200" solimp="0.99 0.99 0.001"/>'
        )
    return '\n          '.join(geoms)


def make_model_with_cup(T_base_cam, cup_pos, cup_quat_wxyz=None,
                        scene_xml=None, cam_name='cam_base',
                        table_z_offset=0.0):
    """Build FR3 scene with a red truncated-cone cup (freejoint)."""
    scene_xml = scene_xml or _DEFAULT_SCENE_XML
    R_cam = T_base_cam[:3, :3]
    t_cam = T_base_cam[:3, 3]
    R_mj = R_cam @ np.diag([1.0, -1.0, -1.0])
    quat_xyzw = Rotation.from_matrix(R_mj).as_quat()
    quat_wxyz_cam = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    fovy = 2 * np.degrees(np.arctan2(CAM_H / 2.0, INTRINSICS['fy']))

    if cup_quat_wxyz is None:
        cup_quat_wxyz = [1, 0, 0, 0]

    cpos = ' '.join(f'{v:.6f}' for v in cup_pos)
    cquat = ' '.join(f'{v:.6f}' for v in cup_quat_wxyz)
    cup_visual = _cup_visual_geoms_xml()
    cup_collision = _cup_collision_geoms_xml()

    xml = f"""
    <mujoco model="fr3 scene with cup (stand_cup)">
      <include file="fr3_with_hand.xml"/>
      <option noslip_iterations="10"/>
      <statistic center="0.3 0 0.4" extent="1.0"/>
      <visual>
        <headlight diffuse="0.4 0.4 0.4" ambient="0.45 0.43 0.40" specular="0.15 0.15 0.15"/>
        <rgba haze="0.15 0.20 0.25 1"/>
        <global offwidth="{CAM_W}" offheight="{CAM_H}"/>
        <quality shadowsize="4096"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.35 0.35 0.38" rgb2="0.18 0.18 0.20"
                 width="512" height="3072"/>
        <texture type="2d" name="labfloor" builtin="flat"
                 rgb1="0.35 0.35 0.35" width="1" height="1"/>
        <material name="labfloor" texture="labfloor" texuniform="true" reflectance="0.05"/>
        <material name="dark_table" rgba="0.12 0.14 0.18 1" specular="0.3" shininess="0.1" reflectance="0.08"/>
      </asset>
      <worldbody>
        <light pos="0.3 0.0 1.8" dir="0 0 -1" directional="true"
               diffuse="0.45 0.45 0.45" ambient="0.15 0.14 0.13" specular="0.1 0.1 0.1"
               castshadow="true"/>
        <light pos="0.8 0.5 1.2" dir="-0.4 -0.3 -1" directional="false"
               diffuse="0.2 0.2 0.2" specular="0.05 0.05 0.05"/>
        <geom name="labfloor" size="3 3 0.01" type="plane" material="labfloor"
              pos="0.3 0 {-0.02 + table_z_offset:.4f}"/>
        <body name="table" pos="0.3 0 {-0.01 + table_z_offset:.4f}">
          <geom name="table_top" type="box" size="0.55 0.45 0.01"
                material="dark_table" contype="1" conaffinity="1"
                solref="-10000 -200" solimp="0.99 0.99 0.001" margin="0.02"/>
        </body>
        <camera name="{cam_name}"
                pos="{t_cam[0]:.6f} {t_cam[1]:.6f} {t_cam[2]:.6f}"
                quat="{quat_wxyz_cam[0]:.6f} {quat_wxyz_cam[1]:.6f} {quat_wxyz_cam[2]:.6f} {quat_wxyz_cam[3]:.6f}"
                fovy="{fovy:.4f}"/>
        <camera name="front" pos="1.0 0.0 0.8" xyaxes="0 1 0 -0.6 0 0.8"/>
        <camera name="side"  pos="0.0 0.8 0.6" xyaxes="-1 0 0 0 -0.6 0.8"/>
        <camera name="top"   pos="0.4 0.0 1.5" xyaxes="0 1 0 -1 0 0"/>
        {_wrist_cam_xml()}
        <body name="cup" pos="{cpos}" quat="{cquat}">
          <freejoint name="cup_joint"/>
          <inertial mass="0.15" pos="0 0 0.000"
                    diaginertia="0.0001 0.0001 0.00005"/>
          {cup_visual}
          {cup_collision}
        </body>
      </worldbody>
    </mujoco>
    """
    orig_dir = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(scene_xml)))
        model = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(orig_dir)
    _bind_wrist_cam(model)
    return model

# ── Constants ──
RENDER_W = 640
RENDER_H = 360
RL_IMG_SIZE = 84
FPS = 15
TABLE_Z_OFFSET = 0.02
GRIPPER_MAX_WIDTH = 0.04
GRIPPER_MIN_WIDTH = 0.0

# Physics constants (from batch_replay_stand.py)
CUP_FRICTION = (8.0, 0.001, 0.001)
FINGER_FRICTION = (8.0, 0.001, 0.001)
CUP_ROTATIONAL_DAMPING = 0.1
ARM_GAIN = 20000.0
ARM_DAMPING = 2000.0
FINGER_GAIN = 1500.0
FINGER_BIAS = -1500.0
N_SETTLE_INIT = 30  # cup settle frames before episode starts

# Grasp detection constants
GRIPPER_CLOSE_THRESHOLD = 0.028  # below this → gripper is "closing"
GRASP_CONTACT_THRESHOLD = 1     # contact_count frames needed to confirm grasp

# Real robot initial joints for stand cup
CUP_HOME_QPOS = np.array([-0.835542, -0.972155, 1.060006, -2.601308,
                            0.814271, 1.7855, 0.34296])

# Default paths
_CUP_POSITIONS = str(
    Path(__file__).resolve().parents[4] / "MSRA" / "stand_cup" / "cup_positions.json"
)
_DEFAULT_CALIB_CUP = str(
    Path(__file__).resolve().parents[4] / "Mujoco_Franka" / "config" / "camera_info_66ep.yaml"
)
_DEFAULT_SCENE_XML = str(
    Path(__file__).resolve().parents[4] / "MSRA" / "stand_cup" / "physics" / "franka_fr3" / "fr3_with_hand.xml"
)


def load_cup_positions(path=None):
    """Load per-episode cup positions. Returns dict: ep_key → {cup_position, cup_orientation_wxyz}."""
    path = path or _CUP_POSITIONS
    with open(path) as f:
        return json.load(f)


class MuJoCoVecEnvCup:
    """Vectorised MuJoCo environment for Franka FR3 stand cup.

    Each env has a cup placed at a specific position/orientation from
    cup_positions.json. The robot must pick up the cup lying on its side
    and stand it upright.

    Physics: full MuJoCo contact simulation. Cup has freejoint,
    grasp relies on finger friction.
    """

    CAMERA_MAP = {
        "cam_base": "observation.images.cam_base",
        "cam_wrist": "observation.images.cam_wrist",
    }

    def __init__(
        self,
        num_envs: int = 1,
        episode_ids: list[int] | None = None,
        cup_positions_path: str | None = None,
        calib_path: str | None = None,
        max_episode_steps: int = 500,
        reward_type: str = "sparse",
        device: str = "cuda:0",
        rl_img_size: int = RL_IMG_SIZE,
        groot_img_size: int = 256,
    ):
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.device = device
        self.rl_img_size = rl_img_size
        self.groot_img_size = groot_img_size

        # Load cup positions
        self._cup_data = load_cup_positions(cup_positions_path)
        self._episode_keys = sorted(self._cup_data.keys())  # episode_000, episode_001, ...
        print(f"[MuJoCoVecEnvCup] Loaded {len(self._episode_keys)} cup positions")

        # Episode assignment per env
        if episode_ids is None:
            episode_ids = list(range(num_envs))
        self._episode_ids = episode_ids

        # Camera calibration (shared PnP calib)
        calib_file = calib_path or _DEFAULT_CALIB_CUP
        self._T_base_cam = load_calib(calib_file)
        load_calib_wrist(calib_file)

        # Create N environments
        self._envs: list[dict[str, Any]] = []
        for i in range(num_envs):
            ep_id = episode_ids[i]
            env = self._init_single_env(ep_id)
            self._envs.append(env)

        # Gymnasium spaces
        self._state_dim = 8  # eef_pos(3) + eef_quat(4) + gripper_width(1)
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
        # Object state: cup_pos(3) + cup_quat_wxyz(4) + uprightness(1) = 8D
        obs_spaces["observation.object_state"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_envs, 8), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Action: 8D absolute [pos3, quat_xyzw4, gripper_raw1]
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_envs, 8), dtype=np.float32,
        )

        # Per-env state
        self._step_counts = np.zeros(num_envs, dtype=np.int64)
        self._last_actions = np.zeros((num_envs, 8), dtype=np.float32)

    def _get_cup_placement(self, episode_id: int):
        """Get cup position and quaternion for an episode."""
        ep_key = f"episode_{episode_id:03d}"
        if ep_key not in self._cup_data:
            raise ValueError(f"Episode {ep_key} not found in cup_positions.json "
                             f"(available: {self._episode_keys[:5]}...)")
        ep = self._cup_data[ep_key]
        cup_pos = np.array(ep["cup_position"], dtype=np.float64)
        cup_quat_wxyz = np.array(ep["cup_orientation_wxyz"], dtype=np.float64)
        return cup_pos, cup_quat_wxyz

    def _init_single_env(self, episode_id: int):
        """Create one MuJoCo environment with cup."""
        cup_pos, cup_quat_wxyz = self._get_cup_placement(episode_id)

        model = make_model_with_cup(
            self._T_base_cam,
            cup_pos=cup_pos,
            cup_quat_wxyz=cup_quat_wxyz,
            scene_xml=_DEFAULT_SCENE_XML,
            table_z_offset=TABLE_Z_OFFSET,
        )
        data = mujoco.MjData(model)
        ids = get_model_ids(model)

        # Camera IDs
        cam_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_base")
        cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")

        # Hide hand/link6/link7 for wrist cam
        hide_body_ids = []
        for bname in ("hand", "fr3_link6", "fr3_link7"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                hide_body_ids.append(bid)
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] in hide_body_ids and model.geom_group[gi] == 2:
                model.geom_group[gi] = 4

        # Scene render options
        opt_base = mujoco.MjvOption()
        opt_base.geomgroup[3] = 0
        opt_base.geomgroup[4] = 1
        opt_wrist = mujoco.MjvOption()
        opt_wrist.geomgroup[3] = 0
        opt_wrist.geomgroup[4] = 0

        # Renderers (RGB only)
        renderer_base = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        renderer_wrist = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

        # Arm actuator IDs
        arm_actuator_ids = []
        for jid in ids["jnt_ids"]:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            arm_actuator_ids.append(aid)

        # Gripper actuator — use the existing 'gripper' actuator which
        # controls both finger_joint1 and finger_joint2 via tendon.
        # (No separate finger_joint1/2 actuators in this XML.)
        gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        if gripper_actuator_id >= 0:
            # Reconfigure with high gain for strong grip
            model.actuator_gainprm[gripper_actuator_id, 0] = FINGER_GAIN
            model.actuator_biasprm[gripper_actuator_id, 0] = 0.0
            model.actuator_biasprm[gripper_actuator_id, 1] = -FINGER_GAIN
            model.actuator_biasprm[gripper_actuator_id, 2] = 0.0

        # Extend finger joint range
        for fname in ("finger_joint1", "finger_joint2"):
            jid_f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
            if jid_f >= 0:
                model.jnt_range[jid_f] = [-0.01, 0.04]

        # Configure arm actuator gains for physics-based tracking
        for aid in arm_actuator_ids:
            if aid >= 0:
                model.actuator_gainprm[aid, 0] = ARM_GAIN
                model.actuator_biasprm[aid, 0] = 0.0
                model.actuator_biasprm[aid, 1] = -ARM_GAIN
                model.actuator_biasprm[aid, 2] = -ARM_DAMPING

        # (Finger actuator gains already set above for gripper_actuator_id)

        # Set high friction on finger/pad/lip geoms
        for gi in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ""
            if "finger" in gname or "finger" in bname or "pad" in gname or "lip" in gname:
                model.geom_friction[gi] = FINGER_FRICTION

        # Cup joint info
        cup_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_joint")
        cup_qposadr = model.jnt_qposadr[cup_jnt_id] if cup_jnt_id >= 0 else None
        cup_dofadr = model.jnt_dofadr[cup_jnt_id] if cup_jnt_id >= 0 else None

        # Cup collision geom IDs
        cup_col_gids = []
        for gi in range(model.ngeom):
            bname_g = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ""
            if bname_g == "cup" and model.geom_contype[gi] > 0:
                cup_col_gids.append(gi)

        # Rotational damping for cup stability
        if cup_dofadr is not None:
            model.dof_damping[cup_dofadr + 3:cup_dofadr + 6] = CUP_ROTATIONAL_DAMPING

        # Set robot to home position
        for i, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = CUP_HOME_QPOS[i]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = GRIPPER_MAX_WIDTH  # gripper open

        mujoco.mj_forward(model, data)

        # Settle: let cup rest on table
        for aid in arm_actuator_ids:
            if aid >= 0:
                jid_a = model.actuator_trnid[aid, 0]
                data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid_a]]
        if gripper_actuator_id >= 0:
            data.ctrl[gripper_actuator_id] = GRIPPER_MAX_WIDTH
        data.qvel[:] = 0.0
        n_substeps = int(round(1.0 / (FPS * model.opt.timestep)))
        for _ in range(N_SETTLE_INIT):
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        # Apply high cup friction AFTER settle (matches batch_replay)
        for gi in cup_col_gids:
            model.geom_friction[gi] = CUP_FRICTION

        # Finger geom IDs for contact-based grasp detection
        finger_geom_ids = []
        for gi in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ""
            if "finger" in gname or "finger" in bname or "pad" in gname or "lip" in gname:
                finger_geom_ids.append(gi)

        return {
            "model": model,
            "data": data,
            "ids": ids,
            "n_substeps": n_substeps,
            "renderer_base": renderer_base,
            "renderer_wrist": renderer_wrist,
            "cam_base_id": cam_base_id,
            "cam_wrist_id": cam_wrist_id,
            "opt_base": opt_base,
            "opt_wrist": opt_wrist,
            "arm_actuator_ids": arm_actuator_ids,
            "gripper_actuator_id": gripper_actuator_id,
            "cup_jnt_id": cup_jnt_id,
            "cup_qposadr": cup_qposadr,
            "cup_dofadr": cup_dofadr,
            "cup_col_gids": cup_col_gids,
            "finger_geom_ids": finger_geom_ids,
            "grasp_state": {"grasped": False, "contact_count": 0},
            "episode_id": episode_id,
            "cup_pos_init": cup_pos.copy(),
            "cup_quat_wxyz_init": cup_quat_wxyz.copy(),
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        for i in range(self.num_envs):
            self._reset_single_env(i)
        obs_dict = self._build_obs_dict(render_mode="full")
        return obs_dict, {}

    def reset_envs(self, env_ids: list[int]) -> None:
        for i in env_ids:
            self._reset_single_env(i)

    def step(
        self, actions: torch.Tensor, render_mode: str = "full",
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step with absolute EEF targets: (num_envs, 8) [pos3, quat4, grip1]."""
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
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        terminated_t = torch.as_tensor(terminated, device=self.device, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, device=self.device, dtype=torch.bool)

        return obs_dict, rewards_t, terminated_t, truncated_t, {}

    # ------------------------------------------------------------------
    # GR00T observation
    # ------------------------------------------------------------------

    def get_groot_obs(self, env_idx: int,
                      task_str: str = "Pick up the cup lying on its side and stand it upright") -> dict:
        """Build GR00T-format observation. Gripper in RAW METERS."""
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

        # Gripper width in raw meters
        finger_id = ids["finger_ids"][0]
        if finger_id >= 0:
            gripper_width_raw = max(0.0, float(data.qpos[model.jnt_qposadr[finger_id]]))
        else:
            gripper_width_raw = GRIPPER_MAX_WIDTH

        return {
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

    # ------------------------------------------------------------------
    # Internal: action application
    # ------------------------------------------------------------------

    def _apply_action(self, env_idx, target_pos, target_quat_xyzw, gripper_width_raw):
        """Apply IK + step physics. Gripper controlled by GR00T output."""
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        qn = np.linalg.norm(target_quat_xyzw)
        if qn > 1e-6:
            target_quat_xyzw = target_quat_xyzw / qn

        # IK on scratch state
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()

        solve_ik(
            model, data, ids["hand_id"], ids["jnt_ids"],
            target_pos, target_quat_xyzw, max_iter=50,
            q_ref=CUP_HOME_QPOS,
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

        # Gripper actuator — GR00T controls gripper
        finger_target = np.clip(gripper_width_raw, GRIPPER_MIN_WIDTH, GRIPPER_MAX_WIDTH)
        grip_aid = env["gripper_actuator_id"]
        if grip_aid >= 0:
            data.ctrl[grip_aid] = finger_target

        # Physics simulation
        for _ in range(env["n_substeps"]):
            mujoco.mj_step(model, data)

        # Update grasp state based on contact
        self._update_grasp(env_idx)

    def _update_grasp(self, env_idx):
        """Contact-based grasp detection: both fingers touching cup + gripper closing."""
        env = self._envs[env_idx]
        model, data = env["model"], env["data"]
        gs = env["grasp_state"]
        cup_gids = set(env["cup_col_gids"])
        finger_gids = env["finger_geom_ids"]

        # Check bilateral finger-cup contact
        left_contact = False
        right_contact = False
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            # Check if one geom is cup and other is finger
            if g1 in cup_gids or g2 in cup_gids:
                other = g2 if g1 in cup_gids else g1
                if other in finger_gids:
                    idx = finger_gids.index(other)
                    if idx % 2 == 0:
                        left_contact = True
                    else:
                        right_contact = True

        bilateral_contact = left_contact and right_contact

        # Gripper width
        finger_id = env["ids"]["finger_ids"][0]
        gripper_width = float(data.qpos[model.jnt_qposadr[finger_id]]) if finger_id >= 0 else GRIPPER_MAX_WIDTH
        gripper_closing = gripper_width < GRIPPER_CLOSE_THRESHOLD

        # Update contact counter (hysteresis)
        if gripper_closing and bilateral_contact:
            gs["contact_count"] = min(gs["contact_count"] + 1, 10)
        else:
            gs["contact_count"] = max(gs["contact_count"] - 2, 0)

        # Grasp state with hysteresis (once grasped, stay grasped while gripper closed)
        was_grasped = gs["grasped"]
        if was_grasped and gripper_closing:
            gs["grasped"] = True
        else:
            gs["grasped"] = (gs["contact_count"] >= GRASP_CONTACT_THRESHOLD and gripper_closing)

    # ------------------------------------------------------------------
    # Internal: reset
    # ------------------------------------------------------------------

    def _reset_single_env(self, env_idx):
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]
        cup_qpa = env["cup_qposadr"]
        cup_dof = env["cup_dofadr"]

        # Robot to home
        for i, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = CUP_HOME_QPOS[i]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = GRIPPER_MAX_WIDTH  # gripper open

        # Reset cup to initial position/orientation
        if cup_qpa is not None:
            data.qpos[cup_qpa:cup_qpa + 3] = env["cup_pos_init"]
            data.qpos[cup_qpa + 3:cup_qpa + 7] = env["cup_quat_wxyz_init"]
        if cup_dof is not None:
            data.qvel[cup_dof:cup_dof + 6] = 0.0

        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        # Set ctrl to current pose
        for aid, jid in zip(env["arm_actuator_ids"], ids["jnt_ids"]):
            if aid >= 0:
                data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
        grip_aid = env["gripper_actuator_id"]
        if grip_aid >= 0:
            data.ctrl[grip_aid] = GRIPPER_MAX_WIDTH

        # Settle cup on table
        for _ in range(N_SETTLE_INIT):
            for _ in range(env["n_substeps"]):
                mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        # Re-apply high cup friction after settle
        for gi in env["cup_col_gids"]:
            model.geom_friction[gi] = CUP_FRICTION

        self._step_counts[env_idx] = 0
        self._last_actions[env_idx] = 0.0
        env["grasp_state"] = {"grasped": False, "contact_count": 0}

    # ------------------------------------------------------------------
    # Internal: reward and success
    # ------------------------------------------------------------------

    def _compute_reward(self, env_idx):
        if self.reward_type == "sparse":
            return 1.0 if self._is_success(env_idx) else 0.0

        # Staged dense reward:
        #   approach (0.20): tanh decay on tcp-cup distance
        #   grasp   (0.15): flat reward when grasped
        #   upright (0.65): uprightness (only when grasped)
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # TCP position
        tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
        # Cup position
        cup_qpa = env["cup_qposadr"]
        if cup_qpa is None:
            return 0.0
        cup_pos = data.qpos[cup_qpa:cup_qpa + 3]
        tcp_cup_dist = float(np.linalg.norm(tcp_pos - cup_pos))

        # Approach: 1 - tanh(dist / scale)
        approach_scale = 0.10
        approach_reward = 1.0 - float(np.tanh(tcp_cup_dist / approach_scale))

        # Grasp
        grasped = env["grasp_state"]["grasped"]
        grasp_reward = 1.0 if grasped else 0.0

        # Upright (only counts when grasped)
        uprightness = self._get_uprightness(env_idx)
        upright_reward = float(np.clip(uprightness, 0.0, 1.0)) if grasped else 0.0

        reward = 0.20 * approach_reward + 0.15 * grasp_reward + 0.65 * upright_reward
        return float(np.clip(reward, 0.0, 1.0))

    def _is_success(self, env_idx) -> bool:
        """Cup is upright and stable on table."""
        env = self._envs[env_idx]
        cup_qpa = env["cup_qposadr"]
        cup_dof = env["cup_dofadr"]
        if cup_qpa is None:
            return False

        model, data = env["model"], env["data"]
        uprightness = self._get_uprightness(env_idx)
        cup_z_pos = data.qpos[cup_qpa + 2]
        cup_vel = np.linalg.norm(data.qvel[cup_dof:cup_dof + 6]) if cup_dof is not None else 0.0

        return (uprightness > 0.85
                and cup_vel < 0.5
                and cup_z_pos > 0.0
                and cup_z_pos < CUP_HEIGHT * 1.5)

    def _get_uprightness(self, env_idx) -> float:
        """Cup local z dot world z. 1.0 = perfectly upright."""
        env = self._envs[env_idx]
        cup_qpa = env["cup_qposadr"]
        if cup_qpa is None:
            return 0.0
        data = env["data"]
        quat_wxyz = data.qpos[cup_qpa + 3:cup_qpa + 7]
        q_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        R = Rotation.from_quat(q_xyzw).as_matrix()
        return float(np.dot(R[:, 2], [0, 0, 1]))

    def get_cup_state(self, env_idx) -> dict:
        """Get cup position, orientation, uprightness for debugging."""
        env = self._envs[env_idx]
        cup_qpa = env["cup_qposadr"]
        cup_dof = env["cup_dofadr"]
        data = env["data"]
        if cup_qpa is None:
            return {"pos": np.zeros(3), "quat_wxyz": np.array([1, 0, 0, 0]),
                    "uprightness": 0.0, "vel": 0.0}
        return {
            "pos": data.qpos[cup_qpa:cup_qpa + 3].copy(),
            "quat_wxyz": data.qpos[cup_qpa + 3:cup_qpa + 7].copy(),
            "uprightness": self._get_uprightness(env_idx),
            "vel": float(np.linalg.norm(data.qvel[cup_dof:cup_dof + 6])) if cup_dof is not None else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal: observation building
    # ------------------------------------------------------------------

    def _build_obs_dict(self, render_mode: str = "full") -> dict[str, torch.Tensor]:
        states = []
        images: dict[str, list[np.ndarray]] = {k: [] for k in self.CAMERA_MAP.values()}
        object_states = []

        for i in range(self.num_envs):
            env = self._envs[i]
            model, data, ids = env["model"], env["data"], env["ids"]

            # State: 8D = eef_pos(3) + eef_quat(4) + gripper(1)
            tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
            eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()
            finger_id = ids["finger_ids"][0]
            grip = max(0.0, float(data.qpos[model.jnt_qposadr[finger_id]])) if finger_id >= 0 else GRIPPER_MAX_WIDTH
            state = np.concatenate([
                tcp_pos.astype(np.float32),
                eef_quat_xyzw.astype(np.float32),
                np.array([grip], dtype=np.float32),
            ])
            states.append(state)

            # Images (RGB only)
            if render_mode == "full":
                self._render_cameras(i, images)
            elif render_mode == "rl_only":
                self._render_cameras(i, images)
            else:
                for key in images:
                    images[key].append(np.zeros((3, self.rl_img_size, self.rl_img_size), dtype=np.uint8))

            # Object state: cup_pos(3) + cup_quat_wxyz(4) + uprightness(1)
            cup_state = self.get_cup_state(i)
            obj_s = np.concatenate([
                cup_state["pos"].astype(np.float32),
                cup_state["quat_wxyz"].astype(np.float32),
                np.array([cup_state["uprightness"]], dtype=np.float32),
            ])
            object_states.append(obj_s)

        obs = {
            "observation.state": torch.as_tensor(
                np.stack(states), device=self.device, dtype=torch.float32),
            "observation.object_state": torch.as_tensor(
                np.stack(object_states), device=self.device, dtype=torch.float32),
        }
        for key, img_list in images.items():
            obs[key] = torch.as_tensor(np.stack(img_list), device=self.device, dtype=torch.uint8)
        return obs

    def _render_cameras(self, env_idx, images_dict):
        env = self._envs[env_idx]
        data = env["data"]
        _rs = self.rl_img_size

        # cam_base
        env["renderer_base"].update_scene(data, camera=env["cam_base_id"],
                                           scene_option=env["opt_base"])
        frame_base = env["renderer_base"].render().copy()
        img_base = cv2.resize(frame_base, (_rs, _rs))
        images_dict["observation.images.cam_base"].append(
            np.transpose(img_base, (2, 0, 1)).astype(np.uint8)
        )

        # cam_wrist
        env["renderer_wrist"].update_scene(data, camera=env["cam_wrist_id"],
                                            scene_option=env["opt_wrist"])
        frame_wrist = env["renderer_wrist"].render().copy()
        img_wrist = cv2.resize(frame_wrist, (_rs, _rs))
        images_dict["observation.images.cam_wrist"].append(
            np.transpose(img_wrist, (2, 0, 1)).astype(np.uint8)
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        for env in self._envs:
            for key in ("renderer_base", "renderer_wrist"):
                if key in env and env[key] is not None:
                    env[key].close()
                    env[key] = None
