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
    _wrist_cam_xml,
    _bind_wrist_cam,
)

# ── Constants ──
RENDER_W = 640
RENDER_H = 360
RL_IMG_SIZE = 84
PNP_CUBE_HALF_SIZE = (0.04, 0.02, 0.02)  # 8×4×4 cm red cube
GRIPPER_CLOSE_THRESHOLD = 0.65
PNP_LIFT_THRESHOLD_M = 0.01  # 1cm lift to count as picked
PNP_PLACE_THRESHOLD_M = 0.095  # cube within bowl radius (9.5cm)
FPS = 20
BOWL_RADIUS = 0.095  # bowl radius in meters
BOWL_HEIGHT = 0.005  # bowl base height
DEFAULT_BOWL_POS = [0.42, 0.03, 0.0]  # from dataset analysis

# Physics simulation constants
PHYSICS_SUBSTEPS = 25       # 25 substeps × 0.002s = 0.05s per action (20Hz control)
GRASP_CONTACT_THRESHOLD = 1  # min number of finger-cube contacts to count as grasped
CUBE_SETTLED_VEL = 0.05     # max cube velocity to count as "settled" for success

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



def _make_model_with_cube_and_bowl(T_base_cam, cube_pos, bowl_pos,
                                    cube_quat_wxyz=None,
                                    cube_size=(0.04, 0.02, 0.02),
                                    bowl_radius=0.095,
                                    use_calibrated_wrist=False,
                                    scene_xml=None, cam_name="cam_base"):
    """Build MuJoCo model with red cube + white bowl (plate)."""
    from utils import DEFAULT_SCENE_XML, INTRINSICS
    scene_xml = scene_xml or DEFAULT_SCENE_XML
    R_cam = T_base_cam[:3, :3]
    t_cam = T_base_cam[:3, 3]
    R_mj = R_cam @ np.diag([1.0, -1.0, -1.0])
    quat_xyzw = Rotation.from_matrix(R_mj).as_quat()
    quat_wxyz_cam = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    fovy = 2 * np.degrees(np.arctan2(CAM_H / 2.0, INTRINSICS["fy"]))
    if cube_quat_wxyz is None:
        cube_quat_wxyz = [1, 0, 0, 0]
    cpos = " ".join(f"{v:.6f}" for v in cube_pos)
    cquat = " ".join(f"{v:.6f}" for v in cube_quat_wxyz)
    csz = " ".join(f"{v:.4f}" for v in cube_size)
    bpos = " ".join(f"{v:.6f}" for v in bowl_pos)


    # Wrist camera XML from calibrated utils
    wrist_xml = _wrist_cam_xml()

    xml = f"""
    <mujoco model="fr3 pnp scene">
      <include file="{scene_xml}"/>
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
        <material name="red_cube" rgba="0.85 0.15 0.10 1" specular="0.15" shininess="0.05" reflectance="0.03"/>
        <material name="white_bowl" rgba="0.92 0.90 0.88 1" specular="0.2" shininess="0.1" reflectance="0.05"/>
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
                pos="{t_cam[0]:.6f} {t_cam[1]:.6f} {t_cam[2]:.6f}"
                quat="{quat_wxyz_cam[0]:.6f} {quat_wxyz_cam[1]:.6f} {quat_wxyz_cam[2]:.6f} {quat_wxyz_cam[3]:.6f}"
                fovy="{fovy:.4f}"/>
        <camera name="front" pos="1.0 0.0 0.8" xyaxes="0 1 0 -0.6 0 0.8"/>
        {wrist_xml}
        <body name="cube" pos="{cpos}" quat="{cquat}">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="{csz}" material="red_cube"
                mass="0.03" friction="5.0 0.5 0.1"
                condim="6" solimp="0.95 0.99 0.001" solref="0.002 1"/>
          <site name="cube_site" size="0.001"/>
        </body>
        <body name="bowl" pos="{bpos}">
          <geom name="bowl_base" type="cylinder" size="{bowl_radius:.4f} 0.005"
                material="white_bowl" mass="0.3" pos="0 0 0.005"/>
        </body>
      </worldbody>
      <equality>
        <weld name="cube_grasp" body1="hand" body2="cube"
              relpose="0 0 0.1034 1 0 0 0"
              solref="0.0002 1" solimp="0.99 0.999 0.0001"
              active="false"/>
      </equality>
    </mujoco>
    """
    orig_dir = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(scene_xml)))
        model = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(orig_dir)
    return model


# ====================================================================
# PnP worker loop (top-level, picklable for multiprocessing.spawn)
# ====================================================================

def _pnp_env_worker_loop(pipe, init_kwargs):
    """Worker process for PnP environment: physics-only, no renderers.

    Protocol (via pipe):
      recv: ("step_batch", [(local_idx, pos3, quat4, grip), ...])
      send: [(state_10d, obj_state_10d, reward, terminated, truncated, grasped), ...]

      recv: ("reset_batch", [local_idx, ...])
      send: "ok"

      recv: ("get_qpos", local_idx)
      send: (qpos_copy, qvel_copy, grasp_state_dict)

      recv: ("get_qpos_all",)
      send: [(qpos, qvel, grasp_state_dict), ...]

      recv: ("close",)
      send: None  (then exit)
    """
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['MUJOCO_GL'] = 'osmesa'

    import mujoco as _mj
    import numpy as _np
    from scipy.spatial.transform import Rotation as _Rot

    from utils import (
        HOME_QPOS as _HOME_QPOS,
        get_model_ids as _get_model_ids,
        get_tcp_pose as _get_tcp_pose,
        solve_ik as _solve_ik,
        load_calib as _load_calib,
        _wrist_cam_xml as _wrist_cam_xml_fn,
        _bind_wrist_cam as _bind_wrist_cam_fn,
    )

    # Unpack init kwargs
    num_local_envs = init_kwargs["num_local_envs"]
    cube_positions = init_kwargs["cube_positions"]
    bowl_positions = init_kwargs["bowl_positions"]
    cube_quat_wxyz = init_kwargs["cube_quat_wxyz"]
    cube_size = tuple(init_kwargs["cube_size"])
    scene_xml = init_kwargs["scene_xml"]
    calib_path = init_kwargs["calib_path"]
    reward_type = init_kwargs["reward_type"]
    max_episode_steps = init_kwargs["max_episode_steps"]
    success_threshold = init_kwargs["success_threshold"]
    episode_positions = init_kwargs.get("episode_positions")
    random_cube_range = init_kwargs.get("random_cube_range")
    random_bowl_range = init_kwargs.get("random_bowl_range")
    cube_base_pos = _np.array(init_kwargs.get("cube_base_pos", [0.42, -0.03, 0.02]))
    bowl_base_pos = _np.array(init_kwargs.get("bowl_base_pos", DEFAULT_BOWL_POS))

    T_base_cam = _load_calib(calib_path)

    # Build environments (physics only, no renderers)
    envs = []
    for i in range(num_local_envs):
        model = _make_model_with_cube_and_bowl(
            T_base_cam,
            cube_pos=_np.array(cube_positions[i]),
            bowl_pos=_np.array(bowl_positions[i]),
            cube_quat_wxyz=cube_quat_wxyz,
            cube_size=cube_size,
            bowl_radius=BOWL_RADIUS,
            use_calibrated_wrist=False,
            scene_xml=scene_xml,
        )
        _bind_wrist_cam_fn(model)
        data = _mj.MjData(model)
        ids = _get_model_ids(model)

        cube_jnt_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qposadr = model.jnt_qposadr[cube_jnt_id] if cube_jnt_id >= 0 else None
        cube_geom_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_GEOM, "cube_geom")
        weld_eq_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_EQUALITY, "cube_grasp")
        bowl_body_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, "bowl")

        # Finger geom IDs
        finger_geom_ids = []
        for body_name in ("left_finger", "right_finger"):
            bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                continue
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid and model.geom_group[gid] == 3:
                    finger_geom_ids.append(gid)
        if not finger_geom_ids:
            for gid in range(model.ngeom):
                name = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, gid)
                if name and "finger" in name.lower():
                    finger_geom_ids.append(gid)

        # Actuator IDs
        arm_actuator_ids = []
        for jid in ids["jnt_ids"]:
            jname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_JOINT, jid)
            aid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_ACTUATOR, jname)
            arm_actuator_ids.append(aid)

        # Set high friction + kp on fingers (same as main process)
        for gid in finger_geom_ids:
            model.geom_friction[gid] = [5.0, 0.5, 0.1]
            model.geom_condim[gid] = 6
        for fname in ("finger_joint1", "finger_joint2"):
            aid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_ACTUATOR, fname)
            if aid >= 0:
                model.actuator_gainprm[aid, 0] = 5000.0
                model.actuator_biasprm[aid, 1] = -5000.0
                model.actuator_ctrlrange[aid] = [-0.01, 0.04]
        for fname in ("finger_joint1", "finger_joint2"):
            jid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, fname)
            if jid >= 0:
                model.jnt_range[jid] = [-0.01, 0.04]

        finger_actuator_ids = []
        for fname in ("finger_joint1", "finger_joint2"):
            aid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_ACTUATOR, fname)
            if aid >= 0:
                finger_actuator_ids.append(aid)

        # Init robot pose
        if cube_qposadr is not None:
            data.qpos[cube_qposadr:cube_qposadr + 3] = cube_positions[i]
            data.qpos[cube_qposadr + 3:cube_qposadr + 7] = cube_quat_wxyz
        for ji, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = _HOME_QPOS[ji]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04
        _mj.mj_forward(model, data)

        envs.append({
            "model": model,
            "data": data,
            "ids": ids,
            "cube_qposadr": cube_qposadr,
            "cube_geom_id": cube_geom_id,
            "weld_eq_id": weld_eq_id,
            "bowl_body_id": bowl_body_id,
            "finger_geom_ids": finger_geom_ids,
            "arm_actuator_ids": arm_actuator_ids,
            "finger_actuator_ids": finger_actuator_ids,
            "cube_pos_init": _np.array(cube_positions[i], dtype=_np.float64),
            "bowl_pos_init": _np.array(bowl_positions[i], dtype=_np.float64),
            "grasp_state": {"grasped": False, "contact_count": 0},
            "initial_cube_z": float(cube_positions[i][2]) if len(cube_positions[i]) >= 3 else 0.0,
            "step_count": 0,
        })

    # ── Helper functions (worker-local) ──

    def _get_contacts(env):
        data = env["data"]
        cube_geom_id = env["cube_geom_id"]
        finger_geom_ids = env["finger_geom_ids"]
        if cube_geom_id < 0 or not finger_geom_ids:
            finger_id = env["ids"]["finger_ids"][0]
            if finger_id >= 0:
                gw = data.qpos[env["model"].jnt_qposadr[finger_id]] / 0.04
                c = 1.0 if gw < GRIPPER_CLOSE_THRESHOLD else 0.0
                return _np.array([c, c, c], dtype=_np.float32)
            return _np.zeros(3, dtype=_np.float32)
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
        both_contact = min(left_contact, right_contact)
        return _np.array([left_contact, right_contact, both_contact], dtype=_np.float32)

    def _update_grasp(env, gripper_width):
        model, data = env["model"], env["data"]
        gs = env["grasp_state"]
        contacts = _get_contacts(env)
        any_contact = contacts[2] > 0.5
        gripper_closing = gripper_width < GRIPPER_CLOSE_THRESHOLD

        if gripper_closing and any_contact:
            gs["contact_count"] = min(gs["contact_count"] + 1, 10)
        else:
            gs["contact_count"] = max(gs["contact_count"] - 2, 0)

        was_grasped = gs["grasped"]
        if was_grasped and gripper_closing:
            gs["grasped"] = True
        else:
            gs["grasped"] = (gs["contact_count"] >= GRASP_CONTACT_THRESHOLD and gripper_closing)

        weld_id = env.get("weld_eq_id", -1)
        if weld_id >= 0:
            if gs["grasped"] and not was_grasped:
                hand_id = env["ids"]["hand_id"]
                cube_body_id = model.geom_bodyid[env["cube_geom_id"]]
                hand_pos = data.xpos[hand_id].copy()
                hand_mat = data.xmat[hand_id].reshape(3, 3).copy()
                cube_pos = data.xpos[cube_body_id].copy()
                cube_mat = data.xmat[cube_body_id].reshape(3, 3).copy()
                rel_pos = hand_mat.T @ (cube_pos - hand_pos)
                rel_mat = hand_mat.T @ cube_mat
                rel_quat = _np.zeros(4)
                _mj.mju_mat2Quat(rel_quat, rel_mat.flatten())
                model.eq_data[weld_id, 3:6] = rel_pos
                model.eq_data[weld_id, 6:10] = rel_quat
                model.eq_active0[weld_id] = 1
                data.eq_active[weld_id] = 1
            elif not gs["grasped"] and was_grasped:
                model.eq_active0[weld_id] = 0
                data.eq_active[weld_id] = 0

    def _apply_action(env, target_pos, target_quat_xyzw, gripper_width):
        model, data, ids = env["model"], env["data"], env["ids"]
        qn = _np.linalg.norm(target_quat_xyzw)
        if qn > 1e-6:
            target_quat_xyzw = target_quat_xyzw / qn
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()
        _solve_ik(model, data, ids["hand_id"], ids["jnt_ids"],
                  target_pos, target_quat_xyzw, max_iter=50)
        target_joint_pos = _np.array([
            data.qpos[model.jnt_qposadr[jid]] for jid in ids["jnt_ids"]
        ])
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        _mj.mj_forward(model, data)
        for aid, tq in zip(env["arm_actuator_ids"], target_joint_pos):
            if aid >= 0:
                data.ctrl[aid] = tq
        finger_target = _np.clip(gripper_width, 0.0, 1.0) * 0.04 - 0.010
        finger_target = max(finger_target, -0.01)
        for aid in env["finger_actuator_ids"]:
            data.ctrl[aid] = finger_target
        for _ in range(PHYSICS_SUBSTEPS):
            _mj.mj_step(model, data)
        _update_grasp(env, gripper_width)

    def _is_success(env):
        model, data = env["model"], env["data"]
        qa = env["cube_qposadr"]
        bowl_body_id = env["bowl_body_id"]
        if qa is None:
            return False
        cube_pos = data.qpos[qa:qa + 3].copy()
        if bowl_body_id >= 0:
            bowl_pos = model.body_pos[bowl_body_id].copy()
        else:
            bowl_pos = _np.array(DEFAULT_BOWL_POS)
        if float(_np.linalg.norm(cube_pos[:2] - bowl_pos[:2])) > success_threshold:
            return False
        if cube_pos[2] > BOWL_HEIGHT + 0.05:
            return False
        return True

    def _compute_reward(env):
        model, data, ids = env["model"], env["data"], env["ids"]
        qa = env["cube_qposadr"]
        bowl_body_id = env["bowl_body_id"]
        if qa is None:
            return 0.0
        cube_pos = data.qpos[qa:qa + 3].copy()
        tcp_pos, tcp_R = _get_tcp_pose(model, data, ids["hand_id"])
        grasped = env["grasp_state"]["grasped"]
        if bowl_body_id >= 0:
            bowl_pos = model.body_pos[bowl_body_id].copy()
        else:
            bowl_pos = _np.array(DEFAULT_BOWL_POS)

        if reward_type == "sparse":
            return 1.0 if _is_success(env) else 0.0

        elif reward_type == "dense_clipped":
            tcp_cube_dist = float(_np.linalg.norm(tcp_pos - cube_pos))
            cube_bowl_xy = float(_np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))
            approach_reward = (1.0 - _np.tanh(tcp_cube_dist / 0.1)) * 0.05
            grasp_reward = 0.1 if grasped else 0.0
            lift_delta = cube_pos[2] - env["initial_cube_z"]
            lift_reward = 0.0
            if grasped and lift_delta > PNP_LIFT_THRESHOLD_M:
                lift_reward = _np.tanh(lift_delta / 0.1) * 0.1
            transport_reward = 0.0
            if grasped and lift_delta > PNP_LIFT_THRESHOLD_M:
                transport_reward = (1.0 - _np.tanh(cube_bowl_xy / 0.1)) * 0.15
            place_reward = 0.6 if _is_success(env) else 0.0
            total = approach_reward + grasp_reward + lift_reward + transport_reward + place_reward
            return float(_np.clip(total, 0.0, 1.0))

        elif reward_type == "dense_v2":
            tcp_cube_dist = float(_np.linalg.norm(tcp_pos - cube_pos))
            cube_bowl_xy = float(_np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))
            if not grasped:
                distance_reward = (1.0 - _np.tanh(tcp_cube_dist / 0.1)) * 0.1
                contact_reward = 0.0
                transport_reward = 0.0
                success_reward = 0.0
            else:
                distance_reward = 0.1
                contact_reward = 0.2
                transport_reward = (1.0 - _np.tanh(cube_bowl_xy / 0.1)) * 0.5
                success_reward = 1.0 if _is_success(env) else 0.0
            total = distance_reward + contact_reward + transport_reward + success_reward
            return float(_np.clip(total, 0.0, 1.0))

        elif reward_type == "dense_v3":
            tcp_cube_dist = float(_np.linalg.norm(tcp_pos - cube_pos))
            cube_bowl_xy = float(_np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))
            cube_quat_wxyz = data.qpos[qa + 3:qa + 7]
            cube_R = _Rot.from_quat([cube_quat_wxyz[1], cube_quat_wxyz[2],
                                      cube_quat_wxyz[3], cube_quat_wxyz[0]]).as_matrix()
            grip_dir = tcp_R[:2, 1]
            cube_dir = cube_R[:2, 1]
            grip_dir_n = grip_dir / (_np.linalg.norm(grip_dir) + 1e-8)
            cube_dir_n = cube_dir / (_np.linalg.norm(cube_dir) + 1e-8)
            cos_align = abs(float(_np.dot(grip_dir_n, cube_dir_n)))
            approach_reward = (1.0 - _np.tanh(tcp_cube_dist / 0.1)) * 0.15
            proximity = max(0.0, 1.0 - tcp_cube_dist / 0.15)
            alignment_reward = cos_align * proximity * 0.10
            grasp_reward = 0.10 if grasped else 0.0
            lift_delta = cube_pos[2] - env["initial_cube_z"]
            lift_reward = 0.0
            if grasped and lift_delta > PNP_LIFT_THRESHOLD_M:
                lift_reward = _np.tanh(lift_delta / 0.1) * 0.10
            transport_reward = 0.0
            if grasped and lift_delta > PNP_LIFT_THRESHOLD_M:
                transport_reward = (1.0 - _np.tanh(cube_bowl_xy / 0.1)) * 0.30
            success_reward = 0.25 if _is_success(env) else 0.0
            total = (approach_reward + alignment_reward + grasp_reward +
                     lift_reward + transport_reward + success_reward)
            return float(_np.clip(total, 0.0, 1.0))

        elif reward_type == "dense_v4":
            tcp_cube_dist = float(_np.linalg.norm(tcp_pos - cube_pos))
            cube_bowl_xy = float(_np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))
            cube_quat_wxyz = data.qpos[qa + 3:qa + 7]
            cube_R = _Rot.from_quat([cube_quat_wxyz[1], cube_quat_wxyz[2],
                                      cube_quat_wxyz[3], cube_quat_wxyz[0]]).as_matrix()
            grip_dir = tcp_R[:2, 1]
            cube_dir = cube_R[:2, 1]
            grip_dir_n = grip_dir / (_np.linalg.norm(grip_dir) + 1e-8)
            cube_dir_n = cube_dir / (_np.linalg.norm(cube_dir) + 1e-8)
            cos_align = abs(float(_np.dot(grip_dir_n, cube_dir_n)))
            approach_reward = (1.0 - _np.tanh(tcp_cube_dist / 0.1)) * 0.20
            proximity = max(0.0, 1.0 - tcp_cube_dist / 0.15)
            alignment_reward = cos_align * proximity * 0.10
            grasp_reward = 0.20 if grasped else 0.0
            lift_delta = cube_pos[2] - env["initial_cube_z"]
            transport_reward = 0.0
            if grasped and lift_delta > PNP_LIFT_THRESHOLD_M:
                transport_reward = (1.0 - _np.tanh(cube_bowl_xy / 0.1)) * 0.20
            success_reward = 0.30 if _is_success(env) else 0.0
            total = (approach_reward + alignment_reward + grasp_reward +
                     transport_reward + success_reward)
            return float(_np.clip(total, 0.0, 1.0))

        elif reward_type == "dense":
            tcp_cube_dist = float(_np.linalg.norm(tcp_pos - cube_pos))
            cube_bowl_xy = float(_np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))
            distance_reward = (1.0 - _np.tanh(tcp_cube_dist / 0.1)) * 1.0
            contact_reward = float(grasped) * 2.0
            transport_reward = 0.0
            if grasped:
                transport_reward = (1.0 - _np.tanh(cube_bowl_xy / 0.1)) * 5.0
            success_reward = 100.0 if _is_success(env) else 0.0
            return float(distance_reward + contact_reward + transport_reward + success_reward)

        return 0.0

    def _build_state(env):
        model, data, ids = env["model"], env["data"], env["ids"]
        tcp_pos, tcp_R = _get_tcp_pose(model, data, ids["hand_id"])
        eef_quat_xyzw = _Rot.from_matrix(tcp_R).as_quat()
        grip = []
        for fid in ids["finger_ids"]:
            if fid >= 0:
                grip.append(data.qpos[model.jnt_qposadr[fid]])
            else:
                grip.append(0.04)
        contact_force = 1.0 if env["grasp_state"]["grasped"] else 0.0
        return _np.concatenate([
            tcp_pos.astype(_np.float32),
            eef_quat_xyzw.astype(_np.float32),
            _np.array(grip, dtype=_np.float32),
            _np.array([contact_force], dtype=_np.float32),
        ])

    def _get_object_state(env):
        qa = env["cube_qposadr"]
        bowl_body_id = env["bowl_body_id"]
        cube_state = _np.zeros(7, dtype=_np.float32)
        if qa is not None:
            cube_state[:3] = env["data"].qpos[qa:qa + 3]
            cube_state[3:7] = env["data"].qpos[qa + 3:qa + 7]
        if bowl_body_id >= 0:
            bowl_pos = env["model"].body_pos[bowl_body_id].copy().astype(_np.float32)
        else:
            bowl_pos = _np.array(DEFAULT_BOWL_POS, dtype=_np.float32)
        return _np.concatenate([cube_state, bowl_pos])

    def _reset_env(env):
        model, data, ids = env["model"], env["data"], env["ids"]
        for ji, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = _HOME_QPOS[ji]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04
        qa = env["cube_qposadr"]
        bowl_body_id = env["bowl_body_id"]

        if episode_positions is not None:
            ep = episode_positions[_np.random.randint(len(episode_positions))]
            cube_pos = _np.array(ep["cube_pos"], dtype=_np.float64)
            bowl_pos = _np.array(ep["bowl_pos"], dtype=_np.float64)
            env["cube_pos_init"] = cube_pos.copy()
            env["bowl_pos_init"] = bowl_pos.copy()
            if qa is not None:
                data.qpos[qa:qa + 3] = cube_pos
                if "cube_yaw_deg" in ep:
                    yaw = _np.radians(ep["cube_yaw_deg"])
                else:
                    yaw = _np.random.uniform(-_np.pi, _np.pi)
                q_xyzw = _Rot.from_euler("z", yaw).as_quat()
                data.qpos[qa + 3:qa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            if bowl_body_id >= 0:
                model.body_pos[bowl_body_id] = bowl_pos

        elif random_cube_range is not None:
            if isinstance(random_cube_range, list):
                rng = random_cube_range[_np.random.randint(len(random_cube_range))]
            else:
                rng = random_cube_range
            if rng is None:
                if qa is not None:
                    data.qpos[qa:qa + 3] = env["cube_pos_init"]
                    data.qpos[qa + 3:qa + 7] = cube_quat_wxyz
                if bowl_body_id >= 0:
                    model.body_pos[bowl_body_id] = env["bowl_pos_init"]
            else:
                for _attempt in range(50):
                    dx = _np.random.uniform(rng["dx"][0], rng["dx"][1])
                    dy = _np.random.uniform(rng["dy"][0], rng["dy"][1])
                    cp = cube_base_pos.copy()
                    cp[0] += dx
                    cp[1] += dy
                    bp = bowl_base_pos.copy()
                    if random_bowl_range is not None:
                        _brng = random_bowl_range
                        if isinstance(_brng, list):
                            _brng = _brng[_np.random.randint(len(_brng))]
                        if _brng is not None:
                            bp[0] += _np.random.uniform(_brng["dx"][0], _brng["dx"][1])
                            bp[1] += _np.random.uniform(_brng["dy"][0], _brng["dy"][1])
                    if _np.linalg.norm(cp[:2] - bp[:2]) > success_threshold:
                        break
                env["cube_pos_init"] = cp
                if qa is not None:
                    data.qpos[qa:qa + 3] = cp
                    yaw_lo, yaw_hi = rng.get("yaw", (0, 0))
                    yaw = _np.random.uniform(yaw_lo, yaw_hi)
                    q_xyzw = _Rot.from_euler("z", _np.radians(yaw)).as_quat()
                    data.qpos[qa + 3:qa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
                env["bowl_pos_init"] = bp
                if bowl_body_id >= 0:
                    model.body_pos[bowl_body_id] = bp
        else:
            if qa is not None:
                data.qpos[qa:qa + 3] = env["cube_pos_init"]
                data.qpos[qa + 3:qa + 7] = cube_quat_wxyz
            if bowl_body_id >= 0:
                model.body_pos[bowl_body_id] = env["bowl_pos_init"]

        data.qvel[:] = 0.0
        _mj.mj_forward(model, data)
        if qa is not None:
            env["initial_cube_z"] = float(data.qpos[qa + 2])
        else:
            env["initial_cube_z"] = 0.0
        env["step_count"] = 0
        env["grasp_state"] = {"grasped": False, "contact_count": 0}
        weld_id = env.get("weld_eq_id", -1)
        if weld_id >= 0:
            model.eq_active0[weld_id] = 0
            data.eq_active[weld_id] = 0
        for aid, jid in zip(env["arm_actuator_ids"], ids["jnt_ids"]):
            if aid >= 0:
                data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
        for aid in env["finger_actuator_ids"]:
            data.ctrl[aid] = 0.04

    # Initial reset
    for env in envs:
        _reset_env(env)

    pipe.send("ready")

    # ── Command loop ──
    try:
        while True:
            cmd = pipe.recv()
            if cmd[0] == "step_batch":
                results = []
                for item in cmd[1]:
                    local_idx, pos3, quat4, grip = item
                    env = envs[local_idx]
                    _apply_action(env, pos3, quat4, grip)
                    env["step_count"] += 1
                    reward = _compute_reward(env)
                    term = _is_success(env)
                    trunc = env["step_count"] >= max_episode_steps
                    state = _build_state(env)
                    obj_state = _get_object_state(env)
                    results.append((state, obj_state, reward, term, trunc,
                                    env["grasp_state"]["grasped"]))
                pipe.send(results)
            elif cmd[0] == "reset_batch":
                for local_idx in cmd[1]:
                    _reset_env(envs[local_idx])
                pipe.send("ok")
            elif cmd[0] == "get_qpos":
                _, local_idx = cmd
                env = envs[local_idx]
                pipe.send((env["data"].qpos.copy(), env["data"].qvel.copy(),
                           dict(env["grasp_state"])))
            elif cmd[0] == "get_qpos_all":
                all_data = []
                for env in envs:
                    extra = dict(env["grasp_state"])
                    # Include bowl position so main process can sync model.body_pos
                    bowl_id = env.get("bowl_body_id", -1)
                    if bowl_id >= 0:
                        extra["bowl_pos"] = env["model"].body_pos[bowl_id].copy()
                    extra["cube_pos_init"] = env.get("cube_pos_init", _np.zeros(3))
                    extra["bowl_pos_init"] = env.get("bowl_pos_init", _np.zeros(3))
                    all_data.append((env["data"].qpos.copy(), env["data"].qvel.copy(),
                                     extra))
                pipe.send(all_data)
            elif cmd[0] == "close":
                pipe.send(None)
                break
    except (EOFError, BrokenPipeError):
        pass


# ====================================================================
# PnP vectorised environment
# ====================================================================

from resfit.rl_finetuning.wrappers.subproc_vec_env import SubprocVecEnvMixin


class MuJoCoVecEnvPnP(SubprocVecEnvMixin):
    """Vectorised MuJoCo environment for Franka FR3 pick-and-place.

    Manages *N* independent ``(MjModel, MjData)`` pairs and persistent
    renderers.  Each environment has its own grasp-state tracker for
    kinematic cube attachment (same logic as ``infer_gr00t_mujoco.py``).

    Key differences from MuJoCoVecEnv (Lift):
    - Scene includes a white bowl (plate) as placement target
    - 2-phase reward: Phase 1 (pick) → Phase 2 (place)
    - Success = cube within PNP_PLACE_THRESHOLD_M of bowl center
    - Per-episode cube+bowl positions from JSON file
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
        bowl_positions: list[list[float]] | np.ndarray | None = None,
        episode_positions_file: str | None = None,
        cube_yaw_deg: float = 0.0,
        cube_yaw_degs: list[float] | None = None,
        random_cube_range: dict | None = None,
        random_bowl_range: dict | None = None,
        hover_offset: dict | None = None,
        cube_size: tuple[float, float, float] = PNP_CUBE_HALF_SIZE,
        scene_xml: str | None = None,
        calib_path: str | None = None,
        use_calibrated_wrist: bool = True,
        max_episode_steps: int = 300,
        success_threshold: float = PNP_PLACE_THRESHOLD_M,
        reward_type: str = "sparse",
        device: str = "cuda:0",
        rl_img_size: int = RL_IMG_SIZE,
        groot_img_size: int = 256,
        depth_norm: dict[str, dict[str, float]] | None = None,
        parallel_envs: bool = True,
        num_workers: int = 8,
    ):
        setup_egl()

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.reward_type = reward_type
        self.device = device
        self.rl_img_size = rl_img_size
        self.groot_img_size = groot_img_size
        self.cube_size = cube_size
        self._depth_norm = depth_norm or DEPTH_NORM_DEFAULTS
        self._use_calibrated_wrist = use_calibrated_wrist

        # Camera calibration
        self._T_base_cam = load_calib(calib_path)
        # load_calib_wrist(calib_path)  # Use hardcoded defaults (match MSRA sim data)

        # Load per-episode positions from JSON if provided
        self._episode_positions = None
        if episode_positions_file is not None:
            import json
            with open(episode_positions_file, "r") as f:
                self._episode_positions = json.load(f)
            print(f"[PnP] Loaded {len(self._episode_positions)} episode positions "
                  f"from {episode_positions_file}")

        # Default cube positions if not provided
        if cube_positions is None:
            cube_positions = [[0.42, -0.03, 0.02]] * num_envs
        cube_positions = np.asarray(cube_positions, dtype=np.float64)
        if cube_positions.shape[0] < num_envs:
            cube_positions = np.tile(cube_positions, (num_envs, 1))[:num_envs]

        # Default bowl positions if not provided
        if bowl_positions is None:
            bowl_positions = [DEFAULT_BOWL_POS] * num_envs
        bowl_positions = np.asarray(bowl_positions, dtype=np.float64)
        if bowl_positions.shape[0] < num_envs:
            bowl_positions = np.tile(bowl_positions, (num_envs, 1))[:num_envs]
        self._bowl_positions = bowl_positions

        # Cube quaternion (wxyz for MuJoCo) — per-env if cube_yaw_degs provided
        if cube_yaw_degs is not None:
            self._cube_quats_wxyz = []
            for yd in cube_yaw_degs:
                yaw = np.radians(yd)
                q_xyzw = Rotation.from_euler("z", yaw).as_quat()
                self._cube_quats_wxyz.append([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            # Pad/tile to num_envs if needed
            while len(self._cube_quats_wxyz) < num_envs:
                self._cube_quats_wxyz.append(self._cube_quats_wxyz[-1])
        else:
            self._cube_quats_wxyz = None

        yaw = np.radians(cube_yaw_deg)
        q_xyzw = Rotation.from_euler("z", yaw).as_quat()
        self._cube_quat_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        # Random cube/bowl placement config (if set, overrides fixed positions on reset)
        self._random_cube_range = random_cube_range
        self._random_bowl_range = random_bowl_range
        self._hover_offset = hover_offset
        self._cube_base_pos = np.array([0.42, -0.03, 0.02])
        self._bowl_base_pos = np.array(DEFAULT_BOWL_POS, dtype=np.float64)

        self._scene_xml = scene_xml
        self._calib_path = calib_path

        # ── Create N environments (main process — with renderers) ──
        self._envs: list[dict[str, Any]] = []
        for i in range(num_envs):
            env = self._init_single_env(cube_positions[i], bowl_positions[i], scene_xml)
            self._envs.append(env)

        # ── Parallel env workers (SubprocVecEnv) ──
        self._parallel = parallel_envs and num_envs > 1

        if self._parallel:
            def _make_worker_kwargs(env_indices: list[int]) -> dict:
                return {
                    "cube_positions": [cube_positions[gi].tolist() for gi in env_indices],
                    "bowl_positions": [bowl_positions[gi].tolist() for gi in env_indices],
                    "cube_quat_wxyz": self._cube_quat_wxyz,
                    "cube_size": list(self.cube_size),
                    "scene_xml": self._scene_xml,
                    "calib_path": calib_path,
                    "reward_type": reward_type,
                    "max_episode_steps": max_episode_steps,
                    "success_threshold": success_threshold,
                    "episode_positions": self._episode_positions,
                    "random_cube_range": self._random_cube_range,
                    "random_bowl_range": self._random_bowl_range,
                    "cube_base_pos": self._cube_base_pos.tolist(),
                    "bowl_base_pos": self._bowl_base_pos.tolist(),
                }

            self._init_parallel(
                num_envs=num_envs,
                num_workers=num_workers,
                worker_fn=_pnp_env_worker_loop,
                init_kwargs_fn=_make_worker_kwargs,
            )

            # Wait for workers to be ready
            for pipe in self._worker_pipes:
                msg = pipe.recv()
                assert msg == "ready", f"Worker init failed: {msg}"

            # Cache for parallel step results
            self._par_states = np.zeros((num_envs, 10), dtype=np.float32)
            self._par_object_states = np.zeros((num_envs, 10), dtype=np.float32)
        else:
            self._workers = []
            self._worker_pipes = []
            self._env_to_worker = {}

        self._needs_qpos_sync = False

        # ── Build gymnasium spaces ──
        self._state_dim = 10  # eef_pos(3)+eef_quat(4)+grip(2)+contact(1)
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
        # Object state (cube pose 7D + bowl pos 3D = 10D)
        obs_spaces["observation.object_state"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_envs, 10), dtype=np.float32,
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

    def _init_single_env(self, cube_pos: np.ndarray, bowl_pos: np.ndarray,
                         scene_xml: str | None) -> dict:
        """Create a single MuJoCo environment instance with cube + bowl."""
        model = _make_model_with_cube_and_bowl(
            self._T_base_cam,
            cube_pos=cube_pos,
            bowl_pos=bowl_pos,
            cube_quat_wxyz=self._cube_quat_wxyz,
            cube_size=self.cube_size,
            bowl_radius=BOWL_RADIUS,
            use_calibrated_wrist=self._use_calibrated_wrist,
            scene_xml=scene_xml,
        )

        # Bind calibrated wrist camera
        _bind_wrist_cam(model)

        data = mujoco.MjData(model)
        ids = get_model_ids(model)

        # Camera IDs
        cam_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_base")
        cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_wrist")
        front_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")

        # Hide hand/link6/link7 visual geoms (group 2 → 4) for wrist cam
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

        # Cube joint address
        cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qposadr = model.jnt_qposadr[cube_jnt_id] if cube_jnt_id >= 0 else None

        # Cube geom id (for contact detection)
        cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        # Weld constraint id for grasp
        weld_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "cube_grasp")

        # Bowl body id (for runtime repositioning)
        bowl_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl")

        # Finger geom ids (for contact detection)
        # Find collision geoms (group 3) belonging to finger bodies
        finger_geom_ids = []
        for body_name in ("left_finger", "right_finger"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                continue
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid and model.geom_group[gid] == 3:
                    finger_geom_ids.append(gid)
        # Fallback: try by name
        if not finger_geom_ids:
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if name and "finger" in name.lower():
                    finger_geom_ids.append(gid)
        print(f"[PnP] Found {len(finger_geom_ids)} finger collision geom(s)")

        # Actuator IDs for physics-based control
        arm_actuator_ids = []
        for jid in ids["jnt_ids"]:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            arm_actuator_ids.append(aid)
        # Set high friction on finger collision geoms for stable grasping
        for gid in finger_geom_ids:
            model.geom_friction[gid] = [5.0, 0.5, 0.1]
            model.geom_condim[gid] = 6  # full friction cone

        # Increase finger actuator kp for sufficient grasp force
        for fname in ("finger_joint1", "finger_joint2"):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, fname)
            if aid >= 0:
                model.actuator_gainprm[aid, 0] = 5000.0   # kp: 100 -> 5000
                model.actuator_biasprm[aid, 1] = -5000.0   # -kp for position actuator
                model.actuator_ctrlrange[aid] = [-0.01, 0.04]  # allow extra closing

        # Extend finger joint range to allow tighter closing
        for fname in ("finger_joint1", "finger_joint2"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
            if jid >= 0:
                model.jnt_range[jid] = [-0.01, 0.04]  # was [0, 0.04]

        finger_actuator_ids = []
        for fname in ("finger_joint1", "finger_joint2"):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, fname)
            if aid >= 0:
                finger_actuator_ids.append(aid)

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
            "bowl_pos_init": bowl_pos.copy(),
            "cube_geom_id": cube_geom_id,
            "weld_eq_id": weld_eq_id,
            "finger_geom_ids": finger_geom_ids,
            "bowl_body_id": bowl_body_id,
            "grasp_state": {"grasped": False, "contact_count": 0},
            "arm_actuator_ids": arm_actuator_ids,
            "finger_actuator_ids": finger_actuator_ids,
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset all environments and return observation dict."""
        if self._parallel:
            self._parallel_reset_all(self.num_envs)
        for i in range(self.num_envs):
            self._reset_single_env(i)
        # Sync main-process mirrors from workers so positions match
        if self._parallel:
            self._sync_qpos_all(self._envs, self.num_envs)
            self._needs_qpos_sync = False
            # Update initial_cube_z from synced qpos
            for i in range(self.num_envs):
                qa = self._envs[i]["cube_qposadr"]
                if qa is not None:
                    self._initial_cube_z[i] = self._envs[i]["data"].qpos[qa + 2]

        obs_dict = self._build_obs_dict()
        return obs_dict, {}

    def reset_envs(self, env_ids: list[int]) -> None:
        """Reset specific environments (auto-reset on done)."""
        if self._parallel:
            self._parallel_reset_envs(env_ids)
        for eid in env_ids:
            self._reset_single_env(eid)
        if self._parallel:
            self._sync_qpos_all(self._envs, self.num_envs)
            self._needs_qpos_sync = False
            for eid in env_ids:
                qa = self._envs[eid]["cube_qposadr"]
                if qa is not None:
                    self._initial_cube_z[eid] = self._envs[eid]["data"].qpos[qa + 2]

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

        if self._parallel:
            for i in range(self.num_envs):
                self._last_actions[i] = actions_np[i]

            all_results = self._parallel_step(actions_np, self.num_envs)
            for i, result in enumerate(all_results):
                state, obj_state, reward, term, trunc, grasped = result
                self._step_counts[i] += 1
                rewards[i] = reward
                terminated[i] = term
                truncated[i] = trunc
                self._par_states[i] = state
                self._par_object_states[i] = obj_state
                self._envs[i]["grasp_state"]["grasped"] = grasped

            if render_mode != "none":
                self._sync_qpos_all(self._envs, self.num_envs)
                self._needs_qpos_sync = False
            else:
                self._needs_qpos_sync = True
        else:
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

            for i in range(self.num_envs):
                _step_single_env(i)

        _dt_ik = time.time() - _t_ik

        # Build observations
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

    def get_groot_obs(self, env_idx: int,
                      task_str: str = "Pick up the red cube and place it onto the plate.") -> dict:
        """Build GR00T-format observation for a single environment."""
        # Lazy sync from workers if parallel mode has pending state
        if self._parallel and self._needs_qpos_sync:
            self._sync_qpos_all(self._envs, self.num_envs)
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
    # Internal: action application (PHYSICS-BASED with mj_step)
    # ------------------------------------------------------------------

    def _apply_action(
        self,
        env_idx: int,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        gripper_width: float,
    ) -> None:
        """Apply action via IK → ctrl targets → mj_step (physics simulation).

        1. Save current sim state
        2. Run IK on live data to find target joint positions
        3. Restore sim state
        4. Set data.ctrl with IK targets + finger targets
        5. Run mj_step × PHYSICS_SUBSTEPS for actual dynamics
        6. Update grasp state from contact detection
        """
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # Normalise quaternion
        qn = np.linalg.norm(target_quat_xyzw)
        if qn > 1e-6:
            target_quat_xyzw = target_quat_xyzw / qn

        # ── Step 1-3: IK on scratch state ──
        # Save current state
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()

        # Run IK (modifies data.qpos for arm joints internally)
        solve_ik(
            model, data, ids["hand_id"], ids["jnt_ids"],
            target_pos, target_quat_xyzw, max_iter=50,
        )

        # Extract IK solution (target joint positions)
        target_joint_pos = np.array([
            data.qpos[model.jnt_qposadr[jid]] for jid in ids["jnt_ids"]
        ])

        # Restore original state
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_forward(model, data)

        # ── Step 4: Set actuator ctrl targets ──
        # Arm actuators: position targets from IK
        for aid, target_q in zip(env["arm_actuator_ids"], target_joint_pos):
            if aid >= 0:
                data.ctrl[aid] = target_q

        # Finger actuators: gripper width → finger joint target
        # Subtract offset so fingers close 5mm tighter than commanded
        finger_target = np.clip(gripper_width, 0.0, 1.0) * 0.04 - 0.010
        finger_target = max(finger_target, -0.01)  # respect extended range
        for aid in env["finger_actuator_ids"]:
            data.ctrl[aid] = finger_target

        # ── Step 5: Physics simulation ──
        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(model, data)

        # ── Step 6: Update grasp state from contacts ──
        self._update_grasp_state(env_idx, gripper_width)

    def _update_grasp_state(self, env_idx: int, gripper_width: float) -> None:
        """Update grasp state based on physics contacts + weld constraint."""
        import numpy as np
        env = self._envs[env_idx]
        model, data = env["model"], env["data"]
        grasp_state = env["grasp_state"]
        contacts = self._get_contacts(env_idx)
        any_contact = contacts[2] > 0.5
        gripper_closing = gripper_width < GRIPPER_CLOSE_THRESHOLD

        if gripper_closing and any_contact:
            grasp_state["contact_count"] = min(
                grasp_state["contact_count"] + 1, 10
            )
        else:
            grasp_state["contact_count"] = max(
                grasp_state["contact_count"] - 2, 0
            )

        was_grasped = grasp_state["grasped"]
        # Once grasped (weld active), stay grasped while gripper closing
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
                cube_body_id = model.geom_bodyid[env["cube_geom_id"]]
                hand_pos = data.xpos[hand_id].copy()
                hand_mat = data.xmat[hand_id].reshape(3, 3).copy()
                cube_pos = data.xpos[cube_body_id].copy()
                cube_mat = data.xmat[cube_body_id].reshape(3, 3).copy()
                rel_pos = hand_mat.T @ (cube_pos - hand_pos)
                rel_mat = hand_mat.T @ cube_mat
                rel_quat = np.zeros(4)
                mujoco.mju_mat2Quat(rel_quat, rel_mat.flatten())
                model.eq_data[weld_id, 3:6] = rel_pos
                model.eq_data[weld_id, 6:10] = rel_quat
                model.eq_active0[weld_id] = 1
                data.eq_active[weld_id] = 1
                print(f"  [WELD] ACTIVATED at step, rel_pos={rel_pos}, rel_quat={rel_quat}")
            elif not grasp_state["grasped"] and was_grasped:
                model.eq_active0[weld_id] = 0
                data.eq_active[weld_id] = 0
                print(f"  [WELD] DEACTIVATED")

    # ------------------------------------------------------------------
    # Internal: reset
    # ------------------------------------------------------------------

    def _reset_single_env(self, env_idx: int) -> None:
        """Reset a single environment to initial state.

        In parallel mode, workers already set cube/bowl positions during their
        reset, so we skip position randomization here and let _sync_qpos_all
        (+ bowl sync) bring the authoritative state back.
        """
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        # In parallel mode, skip physics state changes — workers own the
        # authoritative (model, data).  We only reset main-process bookkeeping
        # (grasp_state, step_count, weld, ctrl) here; qpos/qvel and bowl_pos
        # are synced from workers afterwards.
        if self._parallel:
            self._step_counts[env_idx] = 0
            self._last_actions[env_idx] = 0.0
            env["grasp_state"] = {"grasped": False, "contact_count": 0}
            weld_id = env.get("weld_eq_id", -1)
            if weld_id >= 0:
                env["model"].eq_active0[weld_id] = 0
                env["data"].eq_active[weld_id] = 0
            return

        # Reset robot to home pose
        for i, jid in enumerate(ids["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = HOME_QPOS[i]
        for fid in ids["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04

        cube_qposadr = env["cube_qposadr"]
        bowl_body_id = env["bowl_body_id"]

        # Pick random episode positions if available
        if self._episode_positions is not None:
            ep = self._episode_positions[np.random.randint(len(self._episode_positions))]
            cube_pos = np.array(ep["cube_pos"], dtype=np.float64)
            bowl_pos = np.array(ep["bowl_pos"], dtype=np.float64)
            env["cube_pos_init"] = cube_pos.copy()
            env["bowl_pos_init"] = bowl_pos.copy()

            # Set cube position
            if cube_qposadr is not None:
                data.qpos[cube_qposadr:cube_qposadr + 3] = cube_pos
                # Use yaw from positions file if available, else random
                if "cube_yaw_deg" in ep:
                    yaw = np.radians(ep["cube_yaw_deg"])
                else:
                    yaw = np.random.uniform(-np.pi, np.pi)
                q_xyzw = Rotation.from_euler("z", yaw).as_quat()
                data.qpos[cube_qposadr + 3:cube_qposadr + 7] = [
                    q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
                ]

            # Reposition bowl
            if bowl_body_id >= 0:
                model.body_pos[bowl_body_id] = bowl_pos

        elif self._random_cube_range is not None:
            # Support list of ranges for cumulative difficulty (uniform random choice)
            if isinstance(self._random_cube_range, list):
                rng = self._random_cube_range[np.random.randint(len(self._random_cube_range))]
            else:
                rng = self._random_cube_range

            if rng is None:
                # No perturbation (easy mode) — use fixed positions
                if cube_qposadr is not None:
                    data.qpos[cube_qposadr:cube_qposadr + 3] = env["cube_pos_init"]
                    if self._cube_quats_wxyz is not None:
                        data.qpos[cube_qposadr + 3:cube_qposadr + 7] = self._cube_quats_wxyz[env_idx]
                    else:
                        data.qpos[cube_qposadr + 3:cube_qposadr + 7] = self._cube_quat_wxyz
                if bowl_body_id >= 0:
                    model.body_pos[bowl_body_id] = env["bowl_pos_init"]
            else:
                # Rejection sampling: ensure cube-bowl XY distance > success threshold
                for _attempt in range(50):
                    dx = np.random.uniform(rng["dx"][0], rng["dx"][1])
                    dy = np.random.uniform(rng["dy"][0], rng["dy"][1])
                    cube_pos = self._cube_base_pos.copy()
                    cube_pos[0] += dx
                    cube_pos[1] += dy

                    # Tentative bowl position
                    bowl_pos = self._bowl_base_pos.copy()
                    if self._random_bowl_range is not None:
                        _brng = self._random_bowl_range
                        if isinstance(_brng, list):
                            _brng = _brng[np.random.randint(len(_brng))]
                        if _brng is not None:
                            bowl_pos[0] += np.random.uniform(_brng["dx"][0], _brng["dx"][1])
                            bowl_pos[1] += np.random.uniform(_brng["dy"][0], _brng["dy"][1])

                    if np.linalg.norm(cube_pos[:2] - bowl_pos[:2]) > self.success_threshold:
                        break

                env["cube_pos_init"] = cube_pos
                if cube_qposadr is not None:
                    data.qpos[cube_qposadr:cube_qposadr + 3] = cube_pos
                    yaw_lo, yaw_hi = rng.get("yaw", (0, 0))
                    yaw = np.random.uniform(yaw_lo, yaw_hi)
                    q_xyzw = Rotation.from_euler("z", np.radians(yaw)).as_quat()
                    data.qpos[cube_qposadr + 3:cube_qposadr + 7] = [
                        q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
                    ]

                # Apply bowl position (already computed above)
                env["bowl_pos_init"] = bowl_pos
                if bowl_body_id >= 0:
                    model.body_pos[bowl_body_id] = bowl_pos
        else:
            # Fixed positions
            if cube_qposadr is not None:
                data.qpos[cube_qposadr:cube_qposadr + 3] = env["cube_pos_init"]
                if self._cube_quats_wxyz is not None:
                    data.qpos[cube_qposadr + 3:cube_qposadr + 7] = self._cube_quats_wxyz[env_idx]
                else:
                    data.qpos[cube_qposadr + 3:cube_qposadr + 7] = self._cube_quat_wxyz
            if bowl_body_id >= 0:
                model.body_pos[bowl_body_id] = env["bowl_pos_init"]

        # Reset velocities
        data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)

        # Record initial cube z
        if cube_qposadr is not None:
            self._initial_cube_z[env_idx] = data.qpos[cube_qposadr + 2]
        else:
            self._initial_cube_z[env_idx] = 0.0

        # Reset state
        self._step_counts[env_idx] = 0
        self._last_actions[env_idx] = 0.0
        env["grasp_state"] = {"grasped": False, "contact_count": 0}
        weld_id = env.get("weld_eq_id", -1)
        if weld_id >= 0:
            env["model"].eq_active0[weld_id] = 0
            env["data"].eq_active[weld_id] = 0

        # Set ctrl to current joint positions (physics-based init)
        for aid, jid in zip(env["arm_actuator_ids"], ids["jnt_ids"]):
            if aid >= 0:
                data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
        for aid in env["finger_actuator_ids"]:
            data.ctrl[aid] = 0.04  # open

    # ------------------------------------------------------------------
    # Internal: reward (PnP 2-phase)
    # ------------------------------------------------------------------

    def _compute_reward(self, env_idx: int) -> float:
        """Compute PnP reward for a single environment.

        Phase 1 (pick): approach cube + grasp
        Phase 2 (place): transport to bowl + release
        """
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]
        cube_qposadr = env["cube_qposadr"]
        bowl_body_id = env["bowl_body_id"]
        if cube_qposadr is None:
            return 0.0

        cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
        tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
        grasped = env["grasp_state"]["grasped"]

        # Bowl position
        if bowl_body_id >= 0:
            bowl_pos = model.body_pos[bowl_body_id].copy()
        else:
            bowl_pos = np.array(DEFAULT_BOWL_POS)

        if self.reward_type == "sparse":
            return 1.0 if self._is_success(env_idx) else 0.0

        elif self.reward_type == "dense_clipped":
            # 5-stage shaped reward for PnP (total range 0~1.0)
            tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
            cube_bowl_xy = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))

            # Stage 1: Approach cube (max 0.05)
            approach_reward = (1.0 - np.tanh(tcp_cube_dist / 0.1)) * 0.05

            # Stage 2: Grasp (0.1 bonus)
            grasp_reward = 0.1 if grasped else 0.0

            # Stage 3: Lift (max 0.1, only if grasped)
            lift_delta = cube_pos[2] - self._initial_cube_z[env_idx]
            lift_reward = 0.0
            if grasped and lift_delta > PNP_LIFT_THRESHOLD_M:
                lift_reward = np.tanh(lift_delta / 0.1) * 0.1

            # Stage 4: Transport to bowl (max 0.15, only if grasped + lifted)
            transport_reward = 0.0
            if grasped and lift_delta > PNP_LIFT_THRESHOLD_M:
                transport_reward = (1.0 - np.tanh(cube_bowl_xy / 0.1)) * 0.15

            # Stage 5: Place success (0.6 bonus)
            place_reward = 0.6 if self._is_success(env_idx) else 0.0

            total = approach_reward + grasp_reward + lift_reward + transport_reward + place_reward
            return float(np.clip(total, 0.0, 1.0))

        elif self.reward_type == "dense_v2":
            # 2-phase reward mirroring Lift structure (total 0~1.0)
            # Phase 1 (pre-grasp): approach cube — same as Lift's distance+contact
            # Phase 2 (post-grasp): transport to bowl — same as Lift's height+success
            tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
            cube_bowl_xy = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))

            if not grasped:
                # Phase 1: approach cube (max 0.3)
                distance_reward = (1.0 - np.tanh(tcp_cube_dist / 0.1)) * 0.1
                contact_reward = 0.0
                transport_reward = 0.0
                success_reward = 0.0
            else:
                # Phase 2: grasped — transport to bowl (max 1.0)
                distance_reward = 0.1  # full approach reward (already at cube)
                contact_reward = 0.2   # grasp bonus
                transport_reward = (1.0 - np.tanh(cube_bowl_xy / 0.1)) * 0.5
                success_reward = 1.0 if self._is_success(env_idx) else 0.0

            total = distance_reward + contact_reward + transport_reward + success_reward
            return float(np.clip(total, 0.0, 1.0))

        elif self.reward_type == "dense_v3":
            # 7-stage reward with gripper-cube alignment and continuous place (0~1.0)
            # Designed for random env perturbation training
            tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
            cube_bowl_xy = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))

            # Get gripper rotation matrix (already available from get_tcp_pose)
            _, tcp_R = get_tcp_pose(model, data, ids["hand_id"])

            # Cube yaw quaternion → rotation matrix
            cube_quat_wxyz = data.qpos[cube_qposadr + 3:cube_qposadr + 7]
            cube_R = Rotation.from_quat([
                cube_quat_wxyz[1], cube_quat_wxyz[2],
                cube_quat_wxyz[3], cube_quat_wxyz[0]
            ]).as_matrix()

            # Gripper-cube yaw alignment: compare gripper y-axis (finger opening
            # direction) with cube y-axis (short 4cm side) in XY plane.
            # Cube is 8×4×4cm so x=long, y=short. Gripper must align fingers
            # along the short side to grasp. |cos| handles 180° symmetry.
            grip_dir = tcp_R[:2, 1]  # gripper y-axis (finger opening dir) in XY
            cube_dir = cube_R[:2, 1]  # cube y-axis (short 4cm side) in XY
            grip_dir_n = grip_dir / (np.linalg.norm(grip_dir) + 1e-8)
            cube_dir_n = cube_dir / (np.linalg.norm(cube_dir) + 1e-8)
            cos_align = abs(float(np.dot(grip_dir_n, cube_dir_n)))  # 0~1, 1=aligned

            # Stage 1: Approach cube (max 0.15)
            approach_reward = (1.0 - np.tanh(tcp_cube_dist / 0.1)) * 0.15

            # Stage 2: Gripper-cube alignment (max 0.10, only when close to cube)
            proximity = max(0.0, 1.0 - tcp_cube_dist / 0.15)  # ramp: 1 at cube, 0 at 15cm
            alignment_reward = cos_align * proximity * 0.10

            # Stage 3: Grasp (0.10 bonus)
            grasp_reward = 0.10 if grasped else 0.0

            # Stage 4: Lift (max 0.10, only if grasped)
            lift_delta = cube_pos[2] - self._initial_cube_z[env_idx]
            lift_reward = 0.0
            if grasped and lift_delta > PNP_LIFT_THRESHOLD_M:
                lift_reward = np.tanh(lift_delta / 0.1) * 0.10

            # Stage 5: Transport to bowl (max 0.30, only if grasped + lifted)
            transport_reward = 0.0
            if grasped and lift_delta > PNP_LIFT_THRESHOLD_M:
                transport_reward = (1.0 - np.tanh(cube_bowl_xy / 0.1)) * 0.30

            # Stage 6: Success (0.25 bonus)
            success_reward = 0.25 if self._is_success(env_idx) else 0.0

            total = (approach_reward + alignment_reward + grasp_reward +
                     lift_reward + transport_reward + success_reward)
            return float(np.clip(total, 0.0, 1.0))

        elif self.reward_type == "dense_v4":
            # 4-stage reward: no lift stage, rebalanced weights (0~1.0)
            # approach 0.20, alignment 0.10, grasp 0.20, transport 0.20, success 0.30
            tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
            cube_bowl_xy = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))

            # Get gripper rotation matrix
            _, tcp_R = get_tcp_pose(model, data, ids["hand_id"])

            # Cube yaw quaternion → rotation matrix
            cube_quat_wxyz = data.qpos[cube_qposadr + 3:cube_qposadr + 7]
            cube_R = Rotation.from_quat([
                cube_quat_wxyz[1], cube_quat_wxyz[2],
                cube_quat_wxyz[3], cube_quat_wxyz[0]
            ]).as_matrix()

            # Gripper-cube yaw alignment
            grip_dir = tcp_R[:2, 1]
            cube_dir = cube_R[:2, 1]
            grip_dir_n = grip_dir / (np.linalg.norm(grip_dir) + 1e-8)
            cube_dir_n = cube_dir / (np.linalg.norm(cube_dir) + 1e-8)
            cos_align = abs(float(np.dot(grip_dir_n, cube_dir_n)))

            # Stage 1: Approach cube (max 0.20)
            approach_reward = (1.0 - np.tanh(tcp_cube_dist / 0.1)) * 0.20

            # Stage 2: Gripper-cube alignment (max 0.10, only when close)
            proximity = max(0.0, 1.0 - tcp_cube_dist / 0.15)
            alignment_reward = cos_align * proximity * 0.10

            # Stage 3: Grasp (0.20 bonus)
            grasp_reward = 0.20 if grasped else 0.0

            # Stage 4: Transport to bowl (max 0.20, only if grasped + lifted)
            lift_delta = cube_pos[2] - self._initial_cube_z[env_idx]
            transport_reward = 0.0
            if grasped and lift_delta > PNP_LIFT_THRESHOLD_M:
                transport_reward = (1.0 - np.tanh(cube_bowl_xy / 0.1)) * 0.20

            # Stage 5: Success (0.30 bonus)
            success_reward = 0.30 if self._is_success(env_idx) else 0.0

            total = (approach_reward + alignment_reward + grasp_reward +
                     transport_reward + success_reward)
            return float(np.clip(total, 0.0, 1.0))

        elif self.reward_type == "dense":
            # Simple dense reward
            tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
            cube_bowl_xy = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))

            distance_reward = (1.0 - np.tanh(tcp_cube_dist / 0.1)) * 1.0
            contact_reward = float(grasped) * 2.0

            transport_reward = 0.0
            if grasped:
                transport_reward = (1.0 - np.tanh(cube_bowl_xy / 0.1)) * 5.0

            success_reward = 100.0 if self._is_success(env_idx) else 0.0

            return float(distance_reward + contact_reward + transport_reward + success_reward)

        else:
            return 0.0

    def _is_success(self, env_idx: int) -> bool:
        """Check if cube is within PNP_PLACE_THRESHOLD_M of bowl center.

        Physics-based: check cube position + low velocity (settled).
        No explicit release check — physics handles that naturally.
        """
        env = self._envs[env_idx]
        model, data = env["model"], env["data"]
        cube_qposadr = env["cube_qposadr"]
        bowl_body_id = env["bowl_body_id"]
        if cube_qposadr is None:
            return False

        cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3].copy()

        # Bowl position
        if bowl_body_id >= 0:
            bowl_pos = model.body_pos[bowl_body_id].copy()
        else:
            bowl_pos = np.array(DEFAULT_BOWL_POS)

        # Check cube is near bowl (XY distance)
        cube_bowl_xy = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))
        if cube_bowl_xy > self.success_threshold:
            return False

        # Check cube is low (near table / bowl surface)
        if cube_pos[2] > BOWL_HEIGHT + 0.05:
            return False

        return True

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
        both_contact = min(left_contact, right_contact)  # 1.0 only if BOTH fingers touch
        return np.array([left_contact, right_contact, both_contact], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal: observation building
    # ------------------------------------------------------------------

    def _build_obs_dict(self, render_mode: str = "full") -> dict[str, torch.Tensor]:
        """Build RL observation dict for all environments."""
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

            # ── State (10D) ──
            if self._parallel and hasattr(self, '_par_states'):
                state = self._par_states[i]
            else:
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
            if render_mode == "full":
                self._render_cameras(i, images)
            else:
                for key in images:
                    images[key].append(np.zeros((3, self.rl_img_size, self.rl_img_size), dtype=np.uint8))

            # ── Depth images ──
            if render_mode != "none":
                self._render_depth(i, depth_images)
            else:
                for key in depth_images:
                    depth_images[key].append(np.zeros((1, self.rl_img_size, self.rl_img_size), dtype=np.float32))

            # ── Object state ──
            if self._parallel and hasattr(self, '_par_object_states'):
                object_states.append(self._par_object_states[i])
            else:
                object_states.append(self._get_object_state(i))

        out: dict[str, torch.Tensor] = {}

        out["observation.state"] = torch.as_tensor(
            np.stack(states), device=self.device, dtype=torch.float32,
        )
        out["observation.raw_joint_pos"] = torch.as_tensor(
            np.stack(raw_joint_pos_list), device=self.device, dtype=torch.float32,
        )
        out["observation.raw_gripper_frac"] = torch.as_tensor(
            np.stack(raw_gripper_frac_list), device=self.device, dtype=torch.float32,
        )

        for key, frames in images.items():
            stacked = np.stack(frames)
            out[key] = torch.as_tensor(stacked, device=self.device, dtype=torch.uint8)

        for key, frames in depth_images.items():
            stacked = np.stack(frames)
            out[key] = torch.as_tensor(stacked, device=self.device, dtype=torch.float32)

        out["observation.object_state"] = torch.as_tensor(
            np.stack(object_states), device=self.device, dtype=torch.float32,
        )

        return out

    def _build_state(self, env_idx: int) -> np.ndarray:
        """Build 10-D state vector: [eef_pos(3), eef_quat_xyzw(4), gripper_qpos(2), contact(1)]."""
        env = self._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]

        tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
        eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()

        grip = []
        for fid in ids["finger_ids"]:
            if fid >= 0:
                grip.append(data.qpos[model.jnt_qposadr[fid]])
            else:
                grip.append(0.04)

        contact_force = 1.0 if env["grasp_state"]["grasped"] else 0.0

        state = np.concatenate([
            tcp_pos.astype(np.float32),
            eef_quat_xyzw.astype(np.float32),
            np.array(grip, dtype=np.float32),
            np.array([contact_force], dtype=np.float32),
        ])
        return state

    def _render_cameras(
        self,
        env_idx: int,
        images_out: dict[str, list[np.ndarray]],
    ) -> None:
        """Render RGB cameras."""
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
                depth_raw = renderer.render().copy()
            else:
                depth_raw = np.zeros((self.rl_img_size, self.rl_img_size), dtype=np.float32)

            norm = self._depth_norm.get(cam_name, {"min": 0.0, "max": 1.0})
            d_min, d_max = norm["min"], norm["max"]
            depth_raw = np.clip(depth_raw, d_min, d_max)
            depth_raw = (depth_raw - d_min) / max(d_max - d_min, 1e-6)
            depth_raw = np.nan_to_num(depth_raw, nan=0.0, posinf=1.0, neginf=0.0)

            depth_out[key].append(depth_raw[np.newaxis].astype(np.float32))

    def _get_object_state(self, env_idx: int) -> np.ndarray:
        """Get cube pose + bowl pos as 10D: [cube_pos3, cube_quat_wxyz4, bowl_pos3]."""
        env = self._envs[env_idx]
        qa = env["cube_qposadr"]
        bowl_body_id = env["bowl_body_id"]
        cube_state = np.zeros(7, dtype=np.float32)
        if qa is not None:
            cube_state[:3] = env["data"].qpos[qa:qa + 3]
            cube_state[3:7] = env["data"].qpos[qa + 3:qa + 7]
        if bowl_body_id >= 0:
            bowl_pos = env["model"].body_pos[bowl_body_id].copy().astype(np.float32)
        else:
            bowl_pos = np.array(DEFAULT_BOWL_POS, dtype=np.float32)
        return np.concatenate([cube_state, bowl_pos])

    # ------------------------------------------------------------------
    # Cube / Bowl state access
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

    def get_bowl_pos(self, env_idx: int) -> np.ndarray:
        """Get bowl position (3,) in world frame."""
        env = self._envs[env_idx]
        bowl_body_id = env["bowl_body_id"]
        if bowl_body_id >= 0:
            return env["model"].body_pos[bowl_body_id].copy()
        return np.array(DEFAULT_BOWL_POS)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def fps(self) -> int:
        return FPS

    def close(self) -> None:
        """Release workers and renderers."""
        if self._parallel:
            self._close_workers()

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

    def get_frame(self, env_id: int, camera: str = "front",
                  size: tuple[int, int] = (128, 160)) -> np.ndarray:
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
