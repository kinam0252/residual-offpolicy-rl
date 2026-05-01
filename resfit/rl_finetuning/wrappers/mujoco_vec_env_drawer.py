"""
MuJoCo vectorised environment for Franka FR3 Close Drawer task.

Independent environment — uses cabinet XML generation from
MSRA/close_drawer/physics/utils_drawer.py logic, adapted to use the
same fr3_with_hand.xml base scene as the stack cube env.

Key differences from Stack Cube env:
- Cabinet with 5 prismatic-joint drawers (vs two cubes)
- Kinematic drawer physics: TCP-face contact detection → set drawer qpos
  (matches batch_replay_drawer.py; GR00T was trained on this physics)
- No grasping / weld constraint needed (push only)
- RGB only (NO depth rendering, unlike Stack Cube)
- Gripper in RAW METERS (0.0–0.04), same as Stack
- FPS = 15 (same as Stack)
- State: 8D = eef_pos(3) + eef_quat(4) + gripper_width(1)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Shared Franka utilities (same as stack env) ──
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
    solve_ik,
    _wrist_cam_xml,
    _bind_wrist_cam,
    INTRINSICS,
)

# ── Constants ──
RENDER_W = 640
RENDER_H = 360
RL_IMG_SIZE = 84
FPS = 15

# Drawer-specific home pose — actual robot joint positions from initial_joint_positions.json
# TCP at this pose: [0.258, -0.263, 0.335] (matches demo mean init ~2cm)
DRAWER_HOME_QPOS = np.array([-1.038979, -0.544473, 0.231092, -2.56324, 0.103947, 1.978203, -0.05349])

GRIPPER_MAX_WIDTH = 0.04
GRIPPER_MIN_WIDTH = 0.0

# Cabinet constants (from utils_drawer.py)
CABINET_LONG = 0.27
CABINET_SHORT = 0.35
CABINET_HEIGHT = 0.282
NUM_DRAWERS = 5
DRAWER_SLIDE = 0.13
WALL_THICK = 0.006
DRAWER_GAP = 0.001
DRAWER_SLOT_HEIGHT = (CABINET_HEIGHT - WALL_THICK * (NUM_DRAWERS + 1)) / NUM_DRAWERS

# Floor→drawer index mapping
FLOOR_TO_DRAWER = {3: 2, 4: 3, 5: 4}
ACTIVE_DRAWERS = [2, 3, 4]

# Demo mean contact positions (normalised Y/Z on drawer face)
# Measured from policy eval with ckpt-100k base policy
DEMO_CONTACT_MEAN = {
    2: {"y_norm": -0.015, "z_norm": 0.275},
    3: {"y_norm": -0.045, "z_norm": 0.268},
    4: {"y_norm": -0.069, "z_norm": -0.262},
}

# Physics substeps computed at runtime
PHYSICS_SUBSTEPS = None

# Paths
_DRAWER_PHYSICS = str(Path(__file__).resolve().parents[4] / "MSRA" / "close_drawer" / "physics")
_CABINET_PLACEMENT = os.path.join(_DRAWER_PHYSICS, "cabinet_placement.json")
_EPISODE_CLASSIFICATION = os.path.join(_DRAWER_PHYSICS, "episode_classification.json")
_EPISODE_MAPPING = os.path.join(_DRAWER_PHYSICS, "real_sim_episode_mapping.json")

# Default camera calibration for drawer task
_DEFAULT_CALIB_DRAWER = str(
    Path(__file__).resolve().parents[4] / "MSRA" / "close_drawer" / "physics" / "camera_info.yaml"
)


def _drawer_xml_snippet(drawer_idx, slot_h, wall_thick, gap,
                        cab_long, cab_short, slide):
    """Generate XML for one drawer body with prismatic joint."""
    z_off = wall_thick + slot_h / 2 + drawer_idx * (slot_h + wall_thick)
    d_hx = (cab_long - 2 * wall_thick) / 2 - gap
    d_hy = (cab_short - 2 * wall_thick) / 2 - gap
    d_hz = slot_h / 2 - gap
    bottom_thick = 0.003
    face_hx = wall_thick / 2
    face_hy = cab_short / 2 - gap
    face_hz = (slot_h + wall_thick) / 2 - gap

    name = f"drawer_{drawer_idx}"
    return """
        <body name="{name}" pos="0 0 {pz:.4f}">
          <joint name="{name}_slide" type="slide" axis="1 0 0"
                 range="0 {slide:.4f}" damping="2.0" frictionloss="0.5"/>
          <geom name="{name}_bottom" type="box"
                size="{d_hx:.4f} {d_hy:.4f} {bt:.4f}"
                pos="0 0 {bz:.4f}" material="wood_drawer" contype="1" conaffinity="1"/>
          <geom name="{name}_left" type="box"
                size="{d_hx:.4f} {wt:.4f} {d_hz:.4f}"
                pos="0 {ly:.4f} 0" material="wood_drawer" contype="1" conaffinity="1"/>
          <geom name="{name}_right" type="box"
                size="{d_hx:.4f} {wt:.4f} {d_hz:.4f}"
                pos="0 {ry:.4f} 0" material="wood_drawer" contype="1" conaffinity="1"/>
          <geom name="{name}_back" type="box"
                size="{wt:.4f} {d_hy:.4f} {d_hz:.4f}"
                pos="{bkx:.4f} 0 0" material="wood_drawer" contype="1" conaffinity="1"/>
          <geom name="{name}_face" type="box"
                size="{face_hx:.4f} {face_hy:.4f} {face_hz:.4f}"
                pos="{fx:.4f} 0 0" material="wood_face" contype="1" conaffinity="1"
                friction="0.8 0.005 0.0001"/>
        </body>""".format(
        name=name,
        pz=z_off,
        slide=slide,
        d_hx=d_hx, d_hy=d_hy, d_hz=d_hz,
        bt=bottom_thick, bz=-d_hz + bottom_thick,
        wt=wall_thick / 2,
        ly=d_hy - wall_thick / 2, ry=-(d_hy - wall_thick / 2),
        bkx=-(d_hx - wall_thick / 2),
        face_hx=face_hx, face_hy=face_hy, face_hz=face_hz,
        fx=d_hx + face_hx,
    )


def make_model_with_cabinet(T_base_cam, cabinet_pos, cabinet_euler=None,
                            scene_xml=None, cam_name="cam_base"):
    """Build FR3 scene with wooden 5-drawer cabinet.

    Uses fr3_with_hand.xml (from stack_cube physics) as base robot.
    Adapted from MSRA/close_drawer/physics/utils_drawer.py.
    """
    # Default scene: stack_cube's fr3_with_hand.xml (proven to work)
    scene_xml = scene_xml or str(
        Path(__file__).resolve().parents[4]
        / "MSRA" / "stack_cube" / "physics" / "franka_fr3" / "fr3_with_hand.xml"
    )

    cab_hx = CABINET_LONG / 2
    cab_hy = CABINET_SHORT / 2
    slot_h = DRAWER_SLOT_HEIGHT
    cp = cabinet_pos

    # Camera setup
    R_cam = T_base_cam[:3, :3]
    t_cam = T_base_cam[:3, 3]
    R_mj = R_cam @ np.diag([1.0, -1.0, -1.0])
    quat_xyzw = Rotation.from_matrix(R_mj).as_quat()
    quat_wxyz_cam = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    fovy = 2 * np.degrees(np.arctan2(CAM_H / 2.0, INTRINSICS["fy"]))

    # Cabinet shell geoms
    shell_geoms = ""
    # Bottom plate
    shell_geoms += """
          <geom name="cab_bottom" type="box" size="{hx:.4f} {hy:.4f} {wt:.4f}"
                pos="0 0 {z:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>""".format(
        hx=cab_hx, hy=cab_hy, wt=WALL_THICK / 2, z=WALL_THICK / 2)
    # Top plate
    shell_geoms += """
          <geom name="cab_top" type="box" size="{hx:.4f} {hy:.4f} {wt:.4f}"
                pos="0 0 {z:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>""".format(
        hx=cab_hx, hy=cab_hy, wt=WALL_THICK / 2, z=CABINET_HEIGHT - WALL_THICK / 2)
    # Left wall
    shell_geoms += """
          <geom name="cab_left" type="box" size="{hx:.4f} {wt:.4f} {hz:.4f}"
                pos="0 {y:.4f} {z:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>""".format(
        hx=cab_hx, wt=WALL_THICK / 2, hz=CABINET_HEIGHT / 2,
        y=cab_hy - WALL_THICK / 2, z=CABINET_HEIGHT / 2)
    # Right wall
    shell_geoms += """
          <geom name="cab_right" type="box" size="{hx:.4f} {wt:.4f} {hz:.4f}"
                pos="0 {y:.4f} {z:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>""".format(
        hx=cab_hx, wt=WALL_THICK / 2, hz=CABINET_HEIGHT / 2,
        y=-(cab_hy - WALL_THICK / 2), z=CABINET_HEIGHT / 2)
    # Back wall
    shell_geoms += """
          <geom name="cab_back" type="box" size="{wt:.4f} {hy:.4f} {hz:.4f}"
                pos="{x:.4f} 0 {z:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>""".format(
        wt=WALL_THICK / 2, hy=cab_hy, hz=CABINET_HEIGHT / 2,
        x=-(cab_hx - WALL_THICK / 2), z=CABINET_HEIGHT / 2)
    # Horizontal dividers
    for i in range(1, NUM_DRAWERS):
        div_z = WALL_THICK + i * (slot_h + WALL_THICK) - WALL_THICK / 2
        shell_geoms += """
          <geom name="cab_div_{i}" type="box" size="{hx:.4f} {hy:.4f} {wt:.4f}"
                pos="0 0 {z:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>""".format(
            i=i, hx=cab_hx, hy=cab_hy, wt=WALL_THICK / 2, z=div_z)

    # Drawer bodies
    drawer_bodies = ""
    for i in range(NUM_DRAWERS):
        drawer_bodies += _drawer_xml_snippet(
            i, slot_h, WALL_THICK, DRAWER_GAP,
            CABINET_LONG, CABINET_SHORT, DRAWER_SLIDE)

    # Cabinet orientation
    if cabinet_euler is not None:
        cab_quat = Rotation.from_euler('xyz', cabinet_euler, degrees=True).as_quat(scalar_first=True)
        cab_euler_str = 'quat="{:.8f} {:.8f} {:.8f} {:.8f}"'.format(*cab_quat)
    else:
        cab_euler_str = ''

    # fr3_with_hand.xml include filename
    scene_basename = os.path.basename(scene_xml)

    xml = """
    <mujoco model="fr3 scene with drawer cabinet">
      <include file="{scene_file}"/>
      <statistic center="0.3 0 0.4" extent="1.0"/>
      <visual>
        <headlight diffuse="0.4 0.4 0.4" ambient="0.45 0.43 0.40" specular="0.15 0.15 0.15"/>
        <rgba haze="0.15 0.20 0.25 1"/>
        <global offwidth="{cam_w}" offheight="{cam_h}"/>
        <quality shadowsize="4096"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.35 0.35 0.38" rgb2="0.18 0.18 0.20"
                 width="512" height="3072"/>
        <texture type="2d" name="labfloor" builtin="flat"
                 rgb1="0.35 0.35 0.35" width="1" height="1"/>
        <material name="labfloor" texture="labfloor" texuniform="true" reflectance="0.05"/>
        <material name="dark_table" rgba="0.12 0.14 0.18 1" specular="0.3" shininess="0.1" reflectance="0.08"/>
        <material name="wood_cabinet" rgba="0.72 0.58 0.42 1" specular="0.10" shininess="0.03" reflectance="0.03"/>
        <material name="wood_drawer" rgba="0.68 0.55 0.38 1" specular="0.08" shininess="0.02" reflectance="0.02"/>
        <material name="wood_face" rgba="0.75 0.62 0.45 1" specular="0.12" shininess="0.04" reflectance="0.04"/>
      </asset>
      <worldbody>
        <light pos="0.3 0.0 1.8" dir="0 0 -1" directional="true"
               diffuse="0.45 0.45 0.45" ambient="0.15 0.14 0.13" specular="0.1 0.1 0.1"
               castshadow="true"/>
        <light pos="0.8 0.5 1.2" dir="-0.4 -0.3 -1" directional="false"
               diffuse="0.2 0.2 0.2" specular="0.05 0.05 0.05"/>
        <geom name="labfloor" size="3 3 0.01" type="plane" material="labfloor"
              pos="0.3 0 -0.02"/>
        <body name="table" pos="0.3 0 -0.01">
          <geom name="table_top" type="box" size="0.55 0.45 0.01"
                material="dark_table" contype="1" conaffinity="1"/>
        </body>
        <camera name="{cam_name}"
                pos="{t0:.6f} {t1:.6f} {t2:.6f}"
                quat="{q0:.6f} {q1:.6f} {q2:.6f} {q3:.6f}"
                fovy="{fovy:.4f}"/>
        <camera name="front" pos="1.0 0.0 0.8" xyaxes="0 1 0 -0.6 0 0.8"/>
        <camera name="side"  pos="0.0 0.8 0.6" xyaxes="-1 0 0 0 -0.6 0.8"/>
        <camera name="top"   pos="0.4 0.0 1.5" xyaxes="0 1 0 -1 0 0"/>
        {wrist_cam}
        <body name="cabinet" pos="{cp0:.4f} {cp1:.4f} {cp2:.4f}" {cab_euler}>
          {shell}
        {drawers}
        </body>
      </worldbody>
    </mujoco>
    """.format(
        scene_file=scene_basename,
        cam_w=CAM_W, cam_h=CAM_H,
        cam_name=cam_name,
        t0=t_cam[0], t1=t_cam[1], t2=t_cam[2],
        q0=quat_wxyz_cam[0], q1=quat_wxyz_cam[1],
        q2=quat_wxyz_cam[2], q3=quat_wxyz_cam[3],
        fovy=fovy,
        wrist_cam=_wrist_cam_xml(),
        cp0=cp[0], cp1=cp[1], cp2=cp[2], cab_euler=cab_euler_str,
        shell=shell_geoms,
        drawers=drawer_bodies,
    )

    orig_dir = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(scene_xml)))
        model = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(orig_dir)
    _bind_wrist_cam(model)

    return model


def set_drawer_pos(model, data, drawer_idx, position):
    """Set drawer slide position (0=closed, DRAWER_SLIDE=fully open)."""
    jname = f"drawer_{drawer_idx}_slide"
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid >= 0:
        data.qpos[model.jnt_qposadr[jid]] = np.clip(position, 0, DRAWER_SLIDE)
        mujoco.mj_forward(model, data)


def get_drawer_pos(model, data, drawer_idx):
    """Get current drawer slide position."""
    jname = f"drawer_{drawer_idx}_slide"
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid >= 0:
        return float(data.qpos[model.jnt_qposadr[jid]])
    return 0.0


def load_cabinet_placement(path=None):
    """Load cabinet position/euler from JSON."""
    path = path or _CABINET_PLACEMENT
    with open(path) as f:
        cfg = json.load(f)
    return np.array(cfg["position"]), np.array(cfg["euler"])


def load_episode_mapping(path=None):
    """Load episode→floor mapping. Returns list of {real_ep, floor}."""
    path = path or _EPISODE_MAPPING
    with open(path) as f:
        return json.load(f)


class MuJoCoVecEnvDrawer:
    """Vectorised MuJoCo environment for Franka FR3 close drawer.

    Each env has one active drawer that starts open (DRAWER_SLIDE).
    The robot must push the drawer closed.

    Drawer physics is kinematic (TCP-face contact → set qpos),
    matching batch_replay_drawer.py which generated GR00T training data.
    """

    CAMERA_MAP = {
        "cam_base": "observation.images.cam_base",
        "cam_wrist": "observation.images.cam_wrist",
    }

    def __init__(
        self,
        num_envs: int = 1,
        active_drawers: list[int] | None = None,
        cabinet_pos: np.ndarray | None = None,
        cabinet_euler: np.ndarray | None = None,
        scene_xml: str | None = None,
        calib_path: str | None = None,
        max_episode_steps: int = 500,
        reward_type: str = "sparse",
        device: str = "cuda:0",
        rl_img_size: int = RL_IMG_SIZE,
        groot_img_size: int = 256,
        contact_threshold: float = float("inf"),
    ):
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.device = device
        self.rl_img_size = rl_img_size
        self.groot_img_size = groot_img_size
        self.contact_threshold = contact_threshold

        # Cabinet placement
        if cabinet_pos is None or cabinet_euler is None:
            _pos, _euler = load_cabinet_placement()
            cabinet_pos = cabinet_pos if cabinet_pos is not None else _pos
            cabinet_euler = cabinet_euler if cabinet_euler is not None else _euler
        self._cabinet_pos = cabinet_pos
        self._cabinet_euler = cabinet_euler

        # Active drawer per env (cycles through [2,3,4] if not specified)
        if active_drawers is None:
            active_drawers = [ACTIVE_DRAWERS[i % len(ACTIVE_DRAWERS)] for i in range(num_envs)]
        self._active_drawers = active_drawers

        # Camera calibration (drawer-specific — different cam position)
        calib_file = calib_path or _DEFAULT_CALIB_DRAWER
        self._T_base_cam = load_calib(calib_file)
        load_calib_wrist(calib_file)  # Update wrist cam globals for this task

        # ── Precompute drawer face geometry (for kinematic physics) ──
        d_hx = (CABINET_LONG - 2 * WALL_THICK) / 2 - DRAWER_GAP
        self._face_hx = WALL_THICK / 2
        self._face_x_closed = d_hx + self._face_hx
        self._face_hy = CABINET_SHORT / 2 - DRAWER_GAP

        # ── Create N environments ──
        self._envs: list[dict[str, Any]] = []
        for i in range(num_envs):
            env = self._init_single_env(scene_xml, active_drawers[i])
            self._envs.append(env)

        # ── Gymnasium spaces ──
        # State: eef_pos(3) + eef_quat(4) + gripper_width(1) = 8D
        self._state_dim = 8
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
        # Object state: drawer_slide(1) + active_drawer_idx(1) = 2D
        obs_spaces["observation.object_state"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_envs, 2), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Action: 8D absolute [pos3, quat_xyzw4, gripper_raw1]
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_envs, 8), dtype=np.float32,
        )

        # Per-env state
        self._step_counts = np.zeros(num_envs, dtype=np.int64)
        self._drawer_qpos = np.full(num_envs, DRAWER_SLIDE, dtype=np.float64)
        self._last_actions = np.zeros((num_envs, 8), dtype=np.float32)

    def _init_single_env(self, scene_xml, active_drawer):
        """Create one MuJoCo environment with cabinet."""
        model = make_model_with_cabinet(
            self._T_base_cam,
            cabinet_pos=self._cabinet_pos,
            cabinet_euler=self._cabinet_euler,
            scene_xml=scene_xml,
        )
        data = mujoco.MjData(model)
        ids = get_model_ids(model)

        # Camera IDs
        cam_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_base")
        cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")

        # Cabinet body for coordinate transforms
        cab_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cabinet")

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

        # Renderers (RGB only — no depth for drawer)
        renderer_base = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        renderer_wrist = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

        # Arm actuator IDs
        arm_actuator_ids = []
        for jid in ids["jnt_ids"]:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            arm_actuator_ids.append(aid)

        # Finger actuator IDs
        finger_actuator_ids = []
        for fname in ("finger_joint1", "finger_joint2"):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, fname)
            if aid >= 0:
                finger_actuator_ids.append(aid)

        # Disable the original 'gripper' actuator if present
        orig_grip_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        if orig_grip_aid >= 0:
            model.actuator_gainprm[orig_grip_aid, 0] = 0.0
            model.actuator_biasprm[orig_grip_aid, :] = 0.0

        # Extend finger joint range (same as stack)
        for fname in ("finger_joint1", "finger_joint2"):
            jid_f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
            if jid_f >= 0:
                model.jnt_range[jid_f] = [-0.01, 0.04]

        # Close Drawer: boost finger actuator gains so gripper stays firmly closed
        for aid in finger_actuator_ids:
            model.actuator_gainprm[aid, 0] = 200.0
            model.actuator_biasprm[aid, 1] = -200.0

        # Set robot to home position
        for i, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = DRAWER_HOME_QPOS[i]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.0  # gripper closed

        # Open the active drawer
        set_drawer_pos(model, data, active_drawer, DRAWER_SLIDE)
        mujoco.mj_forward(model, data)

        n_substeps = int(round(1.0 / (FPS * model.opt.timestep)))

        # Precompute active drawer z-range for kinematic physics
        slot_h = DRAWER_SLOT_HEIGHT
        dz = WALL_THICK + slot_h / 2 + active_drawer * (slot_h + WALL_THICK)

        return {
            "model": model,
            "data": data,
            "ids": ids,
            "n_substeps": n_substeps,
            "renderer_base": renderer_base,
            "renderer_wrist": renderer_wrist,
            "cam_base_id": cam_base_id,
            "cam_wrist_id": cam_wrist_id,
            "cab_body_id": cab_body_id,
            "opt_base": opt_base,
            "opt_wrist": opt_wrist,
            "active_drawer": active_drawer,
            "arm_actuator_ids": arm_actuator_ids,
            "finger_actuator_ids": finger_actuator_ids,
            "drawer_z_center": dz,
            "drawer_z_min": dz - slot_h / 2 - 0.02,
            "drawer_z_max": dz + slot_h / 2 + 0.02,
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
            # Close Drawer: gripper always closed (push task)
            self._apply_action(i, action[:3], action[3:7], 0.0)
            self._update_drawer_physics(i)
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
                      task_str: str = "Close the drawer") -> dict:
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
        """Apply IK + step physics."""
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
            q_ref=DRAWER_HOME_QPOS,
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

        # Finger actuators — always closed for close-drawer task
        for aid in env["finger_actuator_ids"]:
            data.ctrl[aid] = 0.0

        # Physics simulation
        for _ in range(env["n_substeps"]):
            # Force finger qpos=0 every substep so contact forces can't open gripper
            for fid in env["ids"]["finger_ids"]:
                if fid >= 0:
                    data.qpos[model.jnt_qposadr[fid]] = 0.0
                    data.qvel[model.jnt_dofadr[fid]] = 0.0
            mujoco.mj_step(model, data)

    def _update_drawer_physics(self, env_idx):
        """Kinematic drawer physics: TCP-face contact → update drawer qpos.

        Replicates batch_replay_drawer.py logic with added guard:
        TCP must be in front of the closed-face position to push.
        This prevents false triggers when TCP is behind the cabinet
        (e.g. at home position).
        """
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]
        active_drawer = env["active_drawer"]

        # Get TCP position in world frame
        hand_pos = data.xpos[ids["hand_id"]]
        hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
        tcp = hand_pos + hand_mat @ TCP_OFFSET

        # Transform TCP to cabinet-local frame
        cab_pos_w = data.xpos[env["cab_body_id"]]
        cab_mat = data.xmat[env["cab_body_id"]].reshape(3, 3)
        tcp_local = cab_mat.T @ (tcp - cab_pos_w)

        # Check if TCP is aligned with this drawer's face
        y_ok = abs(tcp_local[1]) < self._face_hy + 0.03
        z_ok = (tcp_local[2] > env["drawer_z_min"]) and (tcp_local[2] < env["drawer_z_max"])

        if y_ok and z_ok:
            # Contact gate: only push if TCP is near demo mean contact position
            if self.contact_threshold < float("inf"):
                face_hz = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP
                dz = env["drawer_z_min"] + face_hz + DRAWER_GAP
                y_norm = tcp_local[1] / self._face_hy if self._face_hy > 0 else 0.0
                z_norm = (tcp_local[2] - dz) / face_hz if face_hz > 0 else 0.0
                dm = DEMO_CONTACT_MEAN.get(active_drawer)
                if dm is not None:
                    dist = np.sqrt((y_norm - dm["y_norm"])**2 + (z_norm - dm["z_norm"])**2)
                    if dist > self.contact_threshold:
                        return  # Too far from demo mean — drawer doesn't move

            face_x_now = self._face_x_closed + self._drawer_qpos[env_idx]
            # Kinematic contact: if TCP x is beyond the closed-face line,
            # the drawer face tracks TCP position (clamped to valid range).
            # In batch_replay, each frame sets GT qpos so face always matches
            # TCP.  Here we must handle arbitrary TCP trajectories:
            #  - TCP behind closed face (x < face_x_closed): ignore (behind cabinet)
            #  - TCP between face_x_closed and face_x_now: push drawer partially
            #  - TCP beyond face_x_now: drawer gets pushed to its minimum
            #    (TCP has already passed through the face)
            if tcp_local[0] > self._face_x_closed:
                new_qpos = tcp_local[0] - self._face_x_closed
                new_qpos = max(0.0, min(DRAWER_SLIDE, new_qpos))
                if new_qpos < self._drawer_qpos[env_idx]:
                    self._drawer_qpos[env_idx] = new_qpos
                    set_drawer_pos(model, data, active_drawer, new_qpos)

    # ------------------------------------------------------------------
    # Internal: reset
    # ------------------------------------------------------------------

    def _reset_single_env(self, env_idx):
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # Robot to home
        for i, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = DRAWER_HOME_QPOS[i]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.0  # gripper closed

        # Close all drawers, then open active one
        for d in range(NUM_DRAWERS):
            set_drawer_pos(model, data, d, 0.0)
        set_drawer_pos(model, data, env["active_drawer"], DRAWER_SLIDE)
        self._drawer_qpos[env_idx] = DRAWER_SLIDE

        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        self._step_counts[env_idx] = 0
        self._last_actions[env_idx] = 0.0

        for aid, jid in zip(env["arm_actuator_ids"], ids["jnt_ids"]):
            if aid >= 0:
                data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
        for aid in env["finger_actuator_ids"]:
            data.ctrl[aid] = 0.0  # gripper closed

    # ------------------------------------------------------------------
    # Internal: reward and success
    # ------------------------------------------------------------------

    def _compute_reward(self, env_idx):
        if self.reward_type == "sparse":
            return 1.0 if self._is_success(env_idx) else 0.0

        if self.reward_type == "dense_simple":
            closed_frac = 1.0 - (self._drawer_qpos[env_idx] / DRAWER_SLIDE)
            return float(np.clip(closed_frac, 0.0, 1.0))

        # Stage-based dense reward (approach / contact / push / success)
        # Each stage contributes up to 0.25, total max = 1.0
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]
        active_drawer = env["active_drawer"]

        # TCP in cabinet-local frame
        hand_pos = data.xpos[ids["hand_id"]]
        hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
        tcp = hand_pos + hand_mat @ TCP_OFFSET
        cab_pos_w = data.xpos[env["cab_body_id"]]
        cab_mat = data.xmat[env["cab_body_id"]].reshape(3, 3)
        tcp_local = cab_mat.T @ (tcp - cab_pos_w)

        # Demo mean target in cabinet-local 3D
        dm = DEMO_CONTACT_MEAN.get(active_drawer)
        face_hz = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP
        dz_center = env["drawer_z_min"] + face_hz + DRAWER_GAP
        target_x = self._face_x_closed + self._drawer_qpos[env_idx]
        target_y = dm["y_norm"] * self._face_hy if dm else 0.0
        target_z = dz_center + dm["z_norm"] * face_hz if dm else dz_center
        target_local = np.array([target_x, target_y, target_z])

        # Stage 1: Approach — TCP distance to demo mean 3D position (max 0.25)
        dist_to_target = float(np.linalg.norm(tcp_local - target_local))
        approach = (1.0 - np.tanh(dist_to_target / 0.1)) * 0.25

        # Stage 2: Contact — TCP in face region AND within threshold of demo mean (0.25)
        y_ok = abs(tcp_local[1]) < self._face_hy + 0.03
        z_ok = (tcp_local[2] > env["drawer_z_min"]) and (tcp_local[2] < env["drawer_z_max"])
        contact = 0.0
        if y_ok and z_ok and dm is not None:
            y_norm = tcp_local[1] / self._face_hy if self._face_hy > 0 else 0.0
            z_norm = (tcp_local[2] - dz_center) / face_hz if face_hz > 0 else 0.0
            norm_dist = np.sqrt((y_norm - dm["y_norm"])**2 + (z_norm - dm["z_norm"])**2)
            if norm_dist <= 1.0:
                contact = 0.25

        # Stage 3: Push — fraction of drawer closed (max 0.25)
        closed_frac = 1.0 - (self._drawer_qpos[env_idx] / DRAWER_SLIDE)
        push = float(np.clip(closed_frac, 0.0, 1.0)) * 0.25

        # Stage 4: Success — drawer ≥90% closed (0.25 bonus)
        success = 0.25 if self._is_success(env_idx) else 0.0

        total = approach + contact + push + success
        return float(np.clip(total, 0.0, 1.0))

    def _is_success(self, env_idx) -> bool:
        """Drawer is considered closed when ≥90% shut (slide < 10% of full)."""
        return self._drawer_qpos[env_idx] < DRAWER_SLIDE * 0.1

    def get_drawer_closed_fraction(self, env_idx) -> float:
        """Return fraction of drawer closed (0=open, 1=closed)."""
        return 1.0 - (self._drawer_qpos[env_idx] / DRAWER_SLIDE)

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
            else:
                for key in images:
                    images[key].append(np.zeros((3, self.rl_img_size, self.rl_img_size), dtype=np.uint8))

            # Object state: drawer_slide + active_drawer_idx
            object_states.append(np.array([
                self._drawer_qpos[i],
                float(env["active_drawer"]),
            ], dtype=np.float32))

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
                if key in env:
                    try:
                        env[key].close()
                    except Exception:
                        pass
