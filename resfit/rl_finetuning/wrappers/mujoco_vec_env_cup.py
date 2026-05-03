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
        data = json.load(f)
    # Support both list format and dict format (episode_000, ...)
    if isinstance(data, list):
        return {f"episode_{i:03d}": item for i, item in enumerate(data)}
    return data


def _cup_env_worker_loop(pipe, init_kwargs):
    """Worker process for cup SubprocVecEnv: physics + IK + reward (no rendering).

    Protocol (via pipe):
      recv: ("step_batch", [(local_idx, pos, quat, grip), ...])
      send: [(state_8d, obj_state_8d, reward, terminated, truncated, grasped), ...]

      recv: ("reset_batch", [local_idx, ...])
      send: "ok"

      recv: ("get_qpos", local_idx)
      send: (qpos_copy, qvel_copy, grasp_state_copy)

      recv: ("close",)
      send: None  (then exit)
    """
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['MUJOCO_GL'] = 'osmesa'  # Workers don't render; avoid EGL conflicts

    import mujoco as _mj
    import numpy as _np
    from scipy.spatial.transform import Rotation as _Rot

    from utils import (
        HOME_QPOS as _HOME_QPOS,
        get_model_ids as _get_model_ids,
        get_tcp_pose as _get_tcp_pose,
        solve_ik as _solve_ik,
        load_calib as _load_calib,
        _wrist_cam_xml,
        _bind_wrist_cam,
    )

    num_local_envs = init_kwargs["num_local_envs"]
    cup_positions = init_kwargs["cup_positions"]
    cup_quats = init_kwargs["cup_quats"]
    scene_xml = init_kwargs["scene_xml"]
    calib_path = init_kwargs["calib_path"]
    reward_type = init_kwargs["reward_type"]
    max_episode_steps = init_kwargs["max_episode_steps"]
    cup_home_qpos = _np.array(init_kwargs["cup_home_qpos"])

    T_base_cam = _load_calib(calib_path)

    # Build environments (physics only, no renderers)
    envs = []
    for i in range(num_local_envs):
        cup_pos = _np.array(cup_positions[i], dtype=_np.float64)
        cup_quat_wxyz = _np.array(cup_quats[i], dtype=_np.float64)

        model = make_model_with_cup(
            T_base_cam, cup_pos=cup_pos, cup_quat_wxyz=cup_quat_wxyz,
            scene_xml=scene_xml, table_z_offset=TABLE_Z_OFFSET,
        )
        data = _mj.MjData(model)
        ids = _get_model_ids(model)

        # Arm actuators
        arm_actuator_ids = []
        for jid in ids["jnt_ids"]:
            jname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_JOINT, jid)
            aid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_ACTUATOR, jname)
            arm_actuator_ids.append(aid)

        # Gripper actuator
        gripper_actuator_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_ACTUATOR, "gripper")
        if gripper_actuator_id >= 0:
            model.actuator_gainprm[gripper_actuator_id, 0] = FINGER_GAIN
            model.actuator_biasprm[gripper_actuator_id, 0] = 0.0
            model.actuator_biasprm[gripper_actuator_id, 1] = -FINGER_GAIN
            model.actuator_biasprm[gripper_actuator_id, 2] = 0.0

        # Finger joint range
        for fname in ("finger_joint1", "finger_joint2"):
            jid_f = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, fname)
            if jid_f >= 0:
                model.jnt_range[jid_f] = [-0.01, 0.04]

        # Arm gains
        for aid in arm_actuator_ids:
            if aid >= 0:
                model.actuator_gainprm[aid, 0] = ARM_GAIN
                model.actuator_biasprm[aid, 0] = 0.0
                model.actuator_biasprm[aid, 1] = -ARM_GAIN
                model.actuator_biasprm[aid, 2] = -ARM_DAMPING

        # Finger friction
        for gi in range(model.ngeom):
            gname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, gi) or ""
            bname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ""
            if "finger" in gname or "finger" in bname or "pad" in gname or "lip" in gname:
                model.geom_friction[gi] = FINGER_FRICTION

        # Cup joint
        cup_jnt_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "cup_joint")
        cup_qposadr = model.jnt_qposadr[cup_jnt_id] if cup_jnt_id >= 0 else None
        cup_dofadr = model.jnt_dofadr[cup_jnt_id] if cup_jnt_id >= 0 else None

        # Cup collision geoms
        cup_col_gids = []
        finger_geom_ids = []
        for gi in range(model.ngeom):
            bname_g = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[gi]) or ""
            gname_g = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, gi) or ""
            if bname_g == "cup" and model.geom_contype[gi] > 0:
                cup_col_gids.append(gi)
            if "finger" in gname_g or "finger" in bname_g or "pad" in gname_g or "lip" in gname_g:
                finger_geom_ids.append(gi)

        # Rotational damping
        if cup_dofadr is not None:
            model.dof_damping[cup_dofadr + 3:cup_dofadr + 6] = CUP_ROTATIONAL_DAMPING

        n_substeps = int(round(1.0 / (FPS * model.opt.timestep)))

        # Init robot
        for ji, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = cup_home_qpos[ji]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = GRIPPER_MAX_WIDTH
        _mj.mj_forward(model, data)

        # Ctrl init
        for aid in arm_actuator_ids:
            if aid >= 0:
                jid_a = model.actuator_trnid[aid, 0]
                data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid_a]]
        if gripper_actuator_id >= 0:
            data.ctrl[gripper_actuator_id] = GRIPPER_MAX_WIDTH
        data.qvel[:] = 0.0

        # Settle cup
        for _ in range(N_SETTLE_INIT):
            for _ in range(n_substeps):
                _mj.mj_step(model, data)
        _mj.mj_forward(model, data)

        # Cup friction after settle
        for gi in cup_col_gids:
            model.geom_friction[gi] = CUP_FRICTION

        # Precompute finger side lookup
        finger_geom_id_set = set(finger_geom_ids)
        finger_geom_side = {gid: (idx % 2 == 0) for idx, gid in enumerate(finger_geom_ids)}

        envs.append({
            "model": model, "data": data, "ids": ids,
            "n_substeps": n_substeps,
            "arm_actuator_ids": arm_actuator_ids,
            "gripper_actuator_id": gripper_actuator_id,
            "cup_qposadr": cup_qposadr,
            "cup_dofadr": cup_dofadr,
            "cup_col_gids": set(cup_col_gids),
            "finger_geom_ids": finger_geom_ids,
            "finger_geom_id_set": finger_geom_id_set,
            "finger_geom_side": finger_geom_side,
            "grasp_state": {"grasped": False, "contact_count": 0},
            "cup_pos_init": cup_pos.copy(),
            "cup_quat_wxyz_init": cup_quat_wxyz.copy(),
            "step_count": 0,
        })

    # ── Local helper functions ──

    def _apply_action_local(env, target_pos, target_quat_xyzw, gripper_width_raw):
        model, data, ids = env["model"], env["data"], env["ids"]
        qn = _np.linalg.norm(target_quat_xyzw)
        if qn > 1e-6:
            target_quat_xyzw = target_quat_xyzw / qn
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()
        _solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
                  target_pos, target_quat_xyzw, max_iter=50, q_ref=cup_home_qpos)
        target_joint_pos = _np.array([
            data.qpos[model.jnt_qposadr[jid]] for jid in ids["jnt_ids"]
        ])
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        _mj.mj_forward(model, data)
        for aid, tq in zip(env["arm_actuator_ids"], target_joint_pos):
            if aid >= 0:
                data.ctrl[aid] = tq
        finger_target = _np.clip(gripper_width_raw, GRIPPER_MIN_WIDTH, GRIPPER_MAX_WIDTH)
        grip_aid = env["gripper_actuator_id"]
        if grip_aid >= 0:
            data.ctrl[grip_aid] = finger_target
        for _ in range(env["n_substeps"]):
            _mj.mj_step(model, data)
        _update_grasp_local(env)

    def _update_grasp_local(env):
        model, data = env["model"], env["data"]
        gs = env["grasp_state"]
        cup_gids = env["cup_col_gids"]
        finger_set = env["finger_geom_id_set"]
        finger_side = env["finger_geom_side"]
        left_contact = False
        right_contact = False
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if g1 in cup_gids or g2 in cup_gids:
                other = g2 if g1 in cup_gids else g1
                if other in finger_set:
                    if finger_side[other]:
                        left_contact = True
                    else:
                        right_contact = True
                    if left_contact and right_contact:
                        break
        bilateral = left_contact and right_contact
        finger_id = env["ids"]["finger_ids"][0]
        gripper_width = float(data.qpos[model.jnt_qposadr[finger_id]]) if finger_id >= 0 else GRIPPER_MAX_WIDTH
        gripper_closing = gripper_width < GRIPPER_CLOSE_THRESHOLD
        if gripper_closing and bilateral:
            gs["contact_count"] = min(gs["contact_count"] + 1, 10)
        else:
            gs["contact_count"] = max(gs["contact_count"] - 2, 0)
        was_grasped = gs["grasped"]
        if was_grasped and gripper_closing:
            gs["grasped"] = True
        else:
            gs["grasped"] = (gs["contact_count"] >= GRASP_CONTACT_THRESHOLD and gripper_closing)

    def _get_uprightness_local(env):
        cup_qpa = env["cup_qposadr"]
        if cup_qpa is None:
            return 0.0
        q = env["data"].qpos[cup_qpa + 3:cup_qpa + 7]
        w, x, y, z = q[0], q[1], q[2], q[3]
        return float(1.0 - 2.0 * (x * x + y * y))

    def _compute_reward_local(env):
        if reward_type == "sparse":
            return 1.0 if _is_success_local(env) else 0.0
        model, data, ids = env["model"], env["data"], env["ids"]
        tcp_pos, _ = _get_tcp_pose(model, data, ids["hand_id"])
        cup_qpa = env["cup_qposadr"]
        if cup_qpa is None:
            return 0.0
        cup_pos = data.qpos[cup_qpa:cup_qpa + 3]
        tcp_cup_dist = float(_np.linalg.norm(tcp_pos - cup_pos))
        approach_reward = 1.0 - float(_np.tanh(tcp_cup_dist / 0.10))
        grasped = env["grasp_state"]["grasped"]
        grasp_reward = 1.0 if grasped else 0.0
        uprightness = _get_uprightness_local(env)
        upright_reward = float(_np.clip(uprightness, 0.0, 1.0)) if grasped else 0.0
        reward = 0.20 * approach_reward + 0.15 * grasp_reward + 0.65 * upright_reward
        return float(_np.clip(reward, 0.0, 1.0))

    def _is_success_local(env):
        cup_qpa = env["cup_qposadr"]
        cup_dof = env["cup_dofadr"]
        if cup_qpa is None:
            return False
        data = env["data"]
        uprightness = _get_uprightness_local(env)
        cup_z = data.qpos[cup_qpa + 2]
        cup_vel = _np.linalg.norm(data.qvel[cup_dof:cup_dof + 6]) if cup_dof is not None else 0.0
        return (uprightness > 0.85 and cup_vel < 0.5
                and cup_z > 0.0 and cup_z < CUP_HEIGHT * 1.5)

    def _extract_state(env):
        model, data, ids = env["model"], env["data"], env["ids"]
        tcp_pos, tcp_R = _get_tcp_pose(model, data, ids["hand_id"])
        eef_quat_xyzw = _Rot.from_matrix(tcp_R).as_quat()
        finger_id = ids["finger_ids"][0]
        grip = max(0.0, float(data.qpos[model.jnt_qposadr[finger_id]])) if finger_id >= 0 else GRIPPER_MAX_WIDTH
        state = _np.concatenate([
            tcp_pos.astype(_np.float32),
            eef_quat_xyzw.astype(_np.float32),
            _np.array([grip], dtype=_np.float32),
        ])
        cup_qpa = env["cup_qposadr"]
        if cup_qpa is not None:
            obj_state = _np.concatenate([
                data.qpos[cup_qpa:cup_qpa + 3].astype(_np.float32),
                data.qpos[cup_qpa + 3:cup_qpa + 7].astype(_np.float32),
                _np.array([_get_uprightness_local(env)], dtype=_np.float32),
            ])
        else:
            obj_state = _np.zeros(8, dtype=_np.float32)
        return state, obj_state

    def _reset_local(env):
        model, data, ids = env["model"], env["data"], env["ids"]
        for ji, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = cup_home_qpos[ji]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = GRIPPER_MAX_WIDTH
        cup_qpa = env["cup_qposadr"]
        cup_dof = env["cup_dofadr"]
        if cup_qpa is not None:
            data.qpos[cup_qpa:cup_qpa + 3] = env["cup_pos_init"]
            data.qpos[cup_qpa + 3:cup_qpa + 7] = env["cup_quat_wxyz_init"]
        if cup_dof is not None:
            data.qvel[cup_dof:cup_dof + 6] = 0.0
        data.qvel[:] = 0.0
        _mj.mj_forward(model, data)
        for aid in env["arm_actuator_ids"]:
            if aid >= 0:
                jid_a = model.actuator_trnid[aid, 0]
                data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid_a]]
        grip_aid = env["gripper_actuator_id"]
        if grip_aid >= 0:
            data.ctrl[grip_aid] = GRIPPER_MAX_WIDTH
        # Settle
        for _ in range(N_SETTLE_INIT):
            for _ in range(env["n_substeps"]):
                _mj.mj_step(model, data)
        _mj.mj_forward(model, data)
        # Re-apply cup friction
        for gi in env["cup_col_gids"]:
            model.geom_friction[gi] = CUP_FRICTION
        env["grasp_state"] = {"grasped": False, "contact_count": 0}
        env["step_count"] = 0

    # ── Main event loop ──
    try:
        while True:
            cmd = pipe.recv()
            if cmd[0] == "step_batch":
                batch_cmds = cmd[1]
                results = []
                for local_idx, target_pos, target_quat, grip_w in batch_cmds:
                    env = envs[local_idx]
                    _apply_action_local(env, target_pos, target_quat, grip_w)
                    env["step_count"] += 1
                    reward = _compute_reward_local(env)
                    terminated = _is_success_local(env)
                    truncated = env["step_count"] >= max_episode_steps
                    state, obj_state = _extract_state(env)
                    results.append((state, obj_state, reward, terminated, truncated,
                                    env["grasp_state"]["grasped"]))
                pipe.send(results)
            elif cmd[0] == "reset_batch":
                for li in cmd[1]:
                    _reset_local(envs[li])
                pipe.send("ok")
            elif cmd[0] == "get_qpos":
                env = envs[cmd[1]]
                pipe.send((env["data"].qpos.copy(), env["data"].qvel.copy(),
                           dict(env["grasp_state"])))
            elif cmd[0] == "close":
                pipe.send(None)
                break
    except (EOFError, BrokenPipeError):
        pass


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
        parallel_envs: bool = True,
        num_workers: int = 8,
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

        # Preallocated buffers for _build_obs_dict (avoid per-step allocation)
        self._buf_states = np.zeros((num_envs, self._state_dim), dtype=np.float32)
        self._buf_obj_states = np.zeros((num_envs, 8), dtype=np.float32)
        self._buf_images = {
            k: np.zeros((num_envs, 3, rl_img_size, rl_img_size), dtype=np.uint8)
            for k in self.CAMERA_MAP.values()
        }
        self._zero_img = np.zeros((3, rl_img_size, rl_img_size), dtype=np.uint8)

        # Precompute finger_geom_ids as set for O(1) lookup in _update_grasp
        for env in self._envs:
            env["finger_geom_id_set"] = set(env["finger_geom_ids"])
            # Also build a side lookup: geom_id → (is_left: bool)
            fgids = env["finger_geom_ids"]
            env["finger_geom_side"] = {gid: (i % 2 == 0) for i, gid in enumerate(fgids)}

        # ── Parallel env workers (SubprocVecEnv) ──
        self._parallel = parallel_envs and num_envs > 1
        self._workers = []
        self._worker_pipes = []
        self._env_to_worker = {}  # env_idx → (worker_idx, local_idx)

        if self._parallel:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")

            actual_workers = min(num_workers, num_envs)
            envs_per_worker = [[] for _ in range(actual_workers)]
            for i in range(num_envs):
                w = i % actual_workers
                envs_per_worker[w].append(i)

            calib_file = calib_path or _DEFAULT_CALIB_CUP

            for w_idx in range(actual_workers):
                env_indices = envs_per_worker[w_idx]
                n_local = len(env_indices)
                for local_i, global_i in enumerate(env_indices):
                    self._env_to_worker[global_i] = (w_idx, local_i)

                parent_pipe, child_pipe = ctx.Pipe()
                init_kwargs = {
                    "num_local_envs": n_local,
                    "cup_positions": [self._cup_data[f"episode_{self._episode_ids[gi]:03d}"]["cup_position"] for gi in env_indices],
                    "cup_quats": [self._cup_data[f"episode_{self._episode_ids[gi]:03d}"]["cup_orientation_wxyz"] for gi in env_indices],
                    "scene_xml": _DEFAULT_SCENE_XML,
                    "calib_path": calib_file,
                    "reward_type": reward_type,
                    "max_episode_steps": max_episode_steps,
                    "cup_home_qpos": CUP_HOME_QPOS.tolist(),
                }
                proc = ctx.Process(target=_cup_env_worker_loop, args=(child_pipe, init_kwargs), daemon=True)
                proc.start()
                child_pipe.close()
                self._workers.append(proc)
                self._worker_pipes.append(parent_pipe)

            print(f"[MuJoCoVecEnvCup] Spawned {actual_workers} workers for {num_envs} envs")
            self._needs_qpos_sync = False

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
        if self._parallel:
            # Reset workers
            num_workers = len(self._worker_pipes)
            worker_local_envs = [[] for _ in range(num_workers)]
            for i in range(self.num_envs):
                w_idx, local_idx = self._env_to_worker[i]
                worker_local_envs[w_idx].append(local_idx)
            for w_idx in range(num_workers):
                if worker_local_envs[w_idx]:
                    self._worker_pipes[w_idx].send(("reset_batch", worker_local_envs[w_idx]))
            for w_idx in range(num_workers):
                if worker_local_envs[w_idx]:
                    self._worker_pipes[w_idx].recv()
            self._step_counts[:] = 0
            # Also reset main-process envs (for rendering)
            for i in range(self.num_envs):
                self._reset_single_env(i)
        else:
            for i in range(self.num_envs):
                self._reset_single_env(i)
        obs_dict = self._build_obs_dict(render_mode="full")
        return obs_dict, {}

    def reset_envs(self, env_ids: list[int]) -> None:
        if self._parallel:
            num_workers = len(self._worker_pipes)
            worker_resets = [[] for _ in range(num_workers)]
            for eid in env_ids:
                w_idx, local_idx = self._env_to_worker[eid]
                worker_resets[w_idx].append(local_idx)
            for w_idx in range(num_workers):
                if worker_resets[w_idx]:
                    self._worker_pipes[w_idx].send(("reset_batch", worker_resets[w_idx]))
            for w_idx in range(num_workers):
                if worker_resets[w_idx]:
                    self._worker_pipes[w_idx].recv()
            for eid in env_ids:
                self._reset_single_env(eid)
                self._step_counts[eid] = 0
        else:
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

        if self._parallel:
            # Dispatch to workers
            num_workers = len(self._worker_pipes)
            worker_batches: list[list[tuple]] = [[] for _ in range(num_workers)]
            worker_env_order: list[list[int]] = [[] for _ in range(num_workers)]

            for i in range(self.num_envs):
                w_idx, local_idx = self._env_to_worker[i]
                action = actions_np[i]
                worker_batches[w_idx].append(
                    (local_idx, action[:3].copy(), action[3:7].copy(), float(action[7]))
                )
                worker_env_order[w_idx].append(i)
                self._last_actions[i] = action

            # Send all batches
            for w_idx in range(num_workers):
                if worker_batches[w_idx]:
                    self._worker_pipes[w_idx].send(("step_batch", worker_batches[w_idx]))

            # Collect results
            for w_idx in range(num_workers):
                if not worker_batches[w_idx]:
                    continue
                results = self._worker_pipes[w_idx].recv()
                for j, global_i in enumerate(worker_env_order[w_idx]):
                    state, obj_state, reward, term, trunc, grasped = results[j]
                    self._step_counts[global_i] += 1
                    rewards[global_i] = reward
                    terminated[global_i] = term
                    truncated[global_i] = trunc
                    self._buf_states[global_i] = state
                    self._buf_obj_states[global_i] = obj_state
                    self._envs[global_i]["grasp_state"]["grasped"] = grasped

            # Sync qpos for rendering if needed
            if render_mode != "none":
                self._sync_qpos_from_workers()
                self._needs_qpos_sync = False
            else:
                self._needs_qpos_sync = True

            # Build obs (state/obj already in buffers from workers)
            obs_dict = self._build_obs_dict(render_mode=render_mode)
        else:
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

    def _sync_qpos_from_workers(self) -> None:
        """Sync qpos/qvel from workers to main-process models (for rendering)."""
        num_workers = len(self._worker_pipes)
        worker_envs: list[list[int]] = [[] for _ in range(num_workers)]
        for i in range(self.num_envs):
            w_idx, _ = self._env_to_worker[i]
            worker_envs[w_idx].append(i)

        for w_idx in range(num_workers):
            for global_i in worker_envs[w_idx]:
                _, local_idx = self._env_to_worker[global_i]
                self._worker_pipes[w_idx].send(("get_qpos", local_idx))
                qpos, qvel, gs = self._worker_pipes[w_idx].recv()
                env = self._envs[global_i]
                env["data"].qpos[:] = qpos
                env["data"].qvel[:] = qvel
                env["grasp_state"] = gs
                mujoco.mj_forward(env["model"], env["data"])

    def get_groot_obs(self, env_idx: int,
                      task_str: str = "Pick up the cup lying on its side and stand it upright") -> dict:
        """Build GR00T-format observation. Gripper in RAW METERS."""
        # Lazy sync from workers if parallel mode has pending state
        if self._parallel and self._needs_qpos_sync:
            self._sync_qpos_from_workers()
            self._needs_qpos_sync = False

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
        finger_set = env["finger_geom_id_set"]
        finger_side = env["finger_geom_side"]

        # Check bilateral finger-cup contact
        left_contact = False
        right_contact = False
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if g1 in cup_gids or g2 in cup_gids:
                other = g2 if g1 in cup_gids else g1
                if other in finger_set:
                    if finger_side[other]:
                        left_contact = True
                    else:
                        right_contact = True
                    if left_contact and right_contact:
                        break

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
        """Cup local z dot world z. 1.0 = perfectly upright.
        Uses direct quaternion math (avoids Rotation object overhead)."""
        env = self._envs[env_idx]
        cup_qpa = env["cup_qposadr"]
        if cup_qpa is None:
            return 0.0
        data = env["data"]
        q = data.qpos[cup_qpa + 3:cup_qpa + 7]  # wxyz
        w, x, y, z = q[0], q[1], q[2], q[3]
        # R[:,2] (third column of rotation matrix) dot [0,0,1] = R[2,2]
        # R[2,2] = 1 - 2*(x^2 + y^2)
        return float(1.0 - 2.0 * (x * x + y * y))

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
        states = self._buf_states
        obj_states = self._buf_obj_states
        do_render = render_mode in ("full", "rl_only")

        # In parallel mode with "none" render, state/obj buffers already filled by workers
        skip_state_compute = self._parallel and not do_render and self._needs_qpos_sync

        if not skip_state_compute:
            for i in range(self.num_envs):
                env = self._envs[i]
                model, data, ids = env["model"], env["data"], env["ids"]

                # State: 8D = eef_pos(3) + eef_quat(4) + gripper(1)
                tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
                eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()
                finger_id = ids["finger_ids"][0]
                grip = max(0.0, float(data.qpos[model.jnt_qposadr[finger_id]])) if finger_id >= 0 else GRIPPER_MAX_WIDTH
                states[i, :3] = tcp_pos
                states[i, 3:7] = eef_quat_xyzw
                states[i, 7] = grip

                # Images
                if do_render:
                    self._render_cameras_inplace(i)

                # Object state: cup_pos(3) + cup_quat_wxyz(4) + uprightness(1)
                cup_qpa = env["cup_qposadr"]
                if cup_qpa is not None:
                    obj_states[i, :3] = data.qpos[cup_qpa:cup_qpa + 3]
                    obj_states[i, 3:7] = data.qpos[cup_qpa + 3:cup_qpa + 7]
                    obj_states[i, 7] = self._get_uprightness(i)
                else:
                    obj_states[i] = 0.0
        else:
            # Buffers already filled by workers; just zero images
            for key in self._buf_images:
                self._buf_images[key][:] = 0

        obs = {
            "observation.state": torch.as_tensor(
                states, device=self.device, dtype=torch.float32),
            "observation.object_state": torch.as_tensor(
                obj_states, device=self.device, dtype=torch.float32),
        }
        for key in self._buf_images:
            obs[key] = torch.as_tensor(self._buf_images[key], device=self.device, dtype=torch.uint8)
        return obs

    def _render_cameras_inplace(self, env_idx):
        """Render cameras directly into preallocated image buffers."""
        env = self._envs[env_idx]
        data = env["data"]
        _rs = self.rl_img_size

        env["renderer_base"].update_scene(data, camera=env["cam_base_id"],
                                           scene_option=env["opt_base"])
        frame_base = env["renderer_base"].render()
        img_base = cv2.resize(frame_base, (_rs, _rs))
        self._buf_images["observation.images.cam_base"][env_idx] = np.transpose(img_base, (2, 0, 1))

        env["renderer_wrist"].update_scene(data, camera=env["cam_wrist_id"],
                                            scene_option=env["opt_wrist"])
        frame_wrist = env["renderer_wrist"].render()
        img_wrist = cv2.resize(frame_wrist, (_rs, _rs))
        self._buf_images["observation.images.cam_wrist"][env_idx] = np.transpose(img_wrist, (2, 0, 1))

    def _render_cameras(self, env_idx, images_dict):
        """Legacy render (used by get_groot_obs)."""
        env = self._envs[env_idx]
        data = env["data"]
        _rs = self.rl_img_size

        env["renderer_base"].update_scene(data, camera=env["cam_base_id"],
                                           scene_option=env["opt_base"])
        frame_base = env["renderer_base"].render().copy()
        img_base = cv2.resize(frame_base, (_rs, _rs))
        images_dict["observation.images.cam_base"].append(
            np.transpose(img_base, (2, 0, 1)).astype(np.uint8)
        )

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
        # Terminate workers
        if self._parallel:
            for pipe in self._worker_pipes:
                try:
                    pipe.send(("close",))
                    pipe.recv()
                except (EOFError, BrokenPipeError):
                    pass
            for w in self._workers:
                w.join(timeout=5)
            self._worker_pipes.clear()
            self._workers.clear()
        # Close renderers
        for env in self._envs:
            for key in ("renderer_base", "renderer_wrist"):
                if key in env and env[key] is not None:
                    env[key].close()
                    env[key] = None
