"""Realistic image renderer using depth-composited rendering with real backgrounds.

Builds a separate MuJoCo model via make_model_generic() for realistic
rendering. Physics runs on the VecEnv model; this only renders.
qpos is copied from VecEnv model → realistic model each frame.

Used by both:
  - Offline data collector (collect_offline_data_with_images.py)
  - Online RL training wrapper (mujoco_residual_wrapper_unified.py)
"""
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np


# Per-task object XML definitions (must match VecEnv joint structure exactly)
TASK_OBJECTS = {
    "pnp": {
        "objects_xml": """
        <body name="cube" pos="0.45 0.10 0.02">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="0.0400 0.0200 0.0200"
                rgba="0.85 0.15 0.10 1" mass="0.1" friction="1.0 0.005 0.0001"/>
        </body>
        <body name="bowl" pos="0.45 -0.10 0.0">
          <geom name="bowl_base" type="cylinder" size="0.0950 0.005"
                rgba="0.92 0.90 0.88 1" mass="0.5" contype="1" conaffinity="1"/>
        </body>
        """,
        "materials_xml": "",
        "shadow_bodies": ["cube"],
    },
    "stack": {
        "objects_xml": """
        <body name="white_cube" pos="0.45 0.22 0.02">
          <freejoint name="white_cube_joint"/>
          <geom name="white_cube_geom" type="box" size="0.02 0.02 0.02"
                rgba="0.9 0.9 0.85 1" mass="0.1" friction="1.0 0.005 0.0001"/>
        </body>
        <body name="green_cube" pos="0.45 -0.06 0.02">
          <freejoint name="green_cube_joint"/>
          <geom name="green_cube_geom" type="box" size="0.02 0.02 0.02"
                rgba="0.1 0.7 0.2 1" mass="0.2" friction="1.0 0.005 0.0001"/>
        </body>
        """,
        "materials_xml": "",
        "shadow_bodies": ["white_cube", "green_cube"],
    },
    "lift": {
        "objects_xml": """
        <body name="cube" pos="0.42 0.0 0.02">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="0.02 0.02 0.02"
                rgba="0.85 0.15 0.10 1" mass="0.1" friction="1.0 0.005 0.0001"/>
        </body>
        """,
        "materials_xml": "",
        "shadow_bodies": ["cube"],
    },
    "cup": {
        "objects_xml": """
        <body name="cup" pos="0.45 0.08 0.0">
          <freejoint name="cup_joint"/>
          <geom name="cup_body" type="cylinder" size="0.035 0.055"
                rgba="0.8 0.15 0.1 1" mass="0.15" friction="1.0 0.005 0.0001"/>
        </body>
        """,
        "materials_xml": "",
        "shadow_bodies": ["cup"],
    },
    # drawer is built dynamically — see _build_drawer_config()
    "drawer": None,
}

# Drawer cabinet constants (mirrors mujoco_vec_env_drawer.py)
_CABINET_LONG = 0.27
_CABINET_SHORT = 0.35
_CABINET_HEIGHT = 0.282
_NUM_DRAWERS = 5
_DRAWER_SLIDE = 0.13
_WALL_THICK = 0.006
_DRAWER_GAP = 0.001
_SLOT_H = (_CABINET_HEIGHT - _WALL_THICK * (_NUM_DRAWERS + 1)) / _NUM_DRAWERS

# Drawer-specific paths
_DRAWER_PHYSICS = str(Path(__file__).resolve().parents[4] / "MSRA" / "close_drawer" / "physics")
_DRAWER_CALIB = os.path.join(_DRAWER_PHYSICS, "camera_info.yaml")
_CABINET_PLACEMENT = os.path.join(_DRAWER_PHYSICS, "cabinet_placement.json")
_DRAWER_TMP = str(Path(__file__).resolve().parents[4] / "tmp_Mujoco_Franka_drawer")
_DRAWER_SCENE_XML = os.path.join(_DRAWER_TMP, "mujoco_menagerie", "franka_fr3", "fr3_with_hand_contact.xml")


def _drawer_xml_snippet(drawer_idx):
    """Generate XML for one drawer body with prismatic joint."""
    z_off = _WALL_THICK + _SLOT_H / 2 + drawer_idx * (_SLOT_H + _WALL_THICK)
    d_hx = (_CABINET_LONG - 2 * _WALL_THICK) / 2 - _DRAWER_GAP
    d_hy = (_CABINET_SHORT - 2 * _WALL_THICK) / 2 - _DRAWER_GAP
    d_hz = _SLOT_H / 2 - _DRAWER_GAP
    bt = 0.003  # bottom thickness
    face_hx = _WALL_THICK / 2
    face_hy = _CABINET_SHORT / 2 - _DRAWER_GAP
    face_hz = (_SLOT_H + _WALL_THICK) / 2 - _DRAWER_GAP
    wt = _WALL_THICK / 2
    name = f"drawer_{drawer_idx}"
    return f"""
        <body name="{name}" pos="0 0 {z_off:.4f}">
          <joint name="{name}_slide" type="slide" axis="1 0 0"
                 range="0 {_DRAWER_SLIDE:.4f}" damping="2.0" frictionloss="0.5"/>
          <geom name="{name}_bottom" type="box"
                size="{d_hx:.4f} {d_hy:.4f} {bt:.4f}"
                pos="0 0 {-d_hz + bt:.4f}" material="wood_drawer" contype="1" conaffinity="1"/>
          <geom name="{name}_left" type="box"
                size="{d_hx:.4f} {wt:.4f} {d_hz:.4f}"
                pos="0 {d_hy - wt:.4f} 0" material="wood_drawer" contype="1" conaffinity="1"/>
          <geom name="{name}_right" type="box"
                size="{d_hx:.4f} {wt:.4f} {d_hz:.4f}"
                pos="0 {-(d_hy - wt):.4f} 0" material="wood_drawer" contype="1" conaffinity="1"/>
          <geom name="{name}_back" type="box"
                size="{wt:.4f} {d_hy:.4f} {d_hz:.4f}"
                pos="{-(d_hx - wt):.4f} 0 0" material="wood_drawer" contype="1" conaffinity="1"/>
          <geom name="{name}_face" type="box"
                size="{face_hx:.4f} {face_hy:.4f} {face_hz:.4f}"
                pos="{d_hx + face_hx:.4f} 0 0" material="wood_face" contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>
        </body>"""


def _build_drawer_config():
    """Build TASK_OBJECTS-style config for drawer (dynamic XML from cabinet placement)."""
    with open(_CABINET_PLACEMENT) as f:
        cab_cfg = json.load(f)
    cab_pos = cab_cfg["position"]
    cab_euler = cab_cfg.get("euler")

    cab_hx = _CABINET_LONG / 2
    cab_hy = _CABINET_SHORT / 2

    # Cabinet shell geoms
    shell = ""
    shell += f'<geom name="cab_bottom" type="box" size="{cab_hx:.4f} {cab_hy:.4f} {_WALL_THICK/2:.4f}" pos="0 0 {_WALL_THICK/2:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>'
    shell += f'<geom name="cab_top" type="box" size="{cab_hx:.4f} {cab_hy:.4f} {_WALL_THICK/2:.4f}" pos="0 0 {_CABINET_HEIGHT - _WALL_THICK/2:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>'
    shell += f'<geom name="cab_left" type="box" size="{cab_hx:.4f} {_WALL_THICK/2:.4f} {_CABINET_HEIGHT/2:.4f}" pos="0 {cab_hy - _WALL_THICK/2:.4f} {_CABINET_HEIGHT/2:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>'
    shell += f'<geom name="cab_right" type="box" size="{cab_hx:.4f} {_WALL_THICK/2:.4f} {_CABINET_HEIGHT/2:.4f}" pos="0 {-(cab_hy - _WALL_THICK/2):.4f} {_CABINET_HEIGHT/2:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>'
    shell += f'<geom name="cab_back" type="box" size="{_WALL_THICK/2:.4f} {cab_hy:.4f} {_CABINET_HEIGHT/2:.4f}" pos="{-(cab_hx - _WALL_THICK/2):.4f} 0 {_CABINET_HEIGHT/2:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>'
    for i in range(1, _NUM_DRAWERS):
        div_z = _WALL_THICK + i * (_SLOT_H + _WALL_THICK) - _WALL_THICK / 2
        shell += f'<geom name="cab_div_{i}" type="box" size="{cab_hx:.4f} {cab_hy:.4f} {_WALL_THICK/2:.4f}" pos="0 0 {div_z:.4f}" material="wood_cabinet" contype="1" conaffinity="1"/>'

    # 5 drawer bodies
    drawers = ""
    for i in range(_NUM_DRAWERS):
        drawers += _drawer_xml_snippet(i)

    # Cabinet orientation
    cab_euler_str = ""
    if cab_euler is not None:
        from scipy.spatial.transform import Rotation
        q = Rotation.from_euler('xyz', cab_euler, degrees=True).as_quat(scalar_first=True)
        cab_euler_str = f'quat="{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}"'

    objects_xml = f"""
        <body name="cabinet" pos="{cab_pos[0]:.4f} {cab_pos[1]:.4f} {cab_pos[2]:.4f}" {cab_euler_str}>
          {shell}
          {drawers}
        </body>"""

    materials_xml = """
        <material name="wood_cabinet" rgba="0.58 0.57 0.47 1" specular="0.08" shininess="0.02" reflectance="0.02"/>
        <material name="wood_drawer" rgba="0.52 0.51 0.42 1" specular="0.06" shininess="0.02" reflectance="0.01"/>
        <material name="wood_face" rgba="0.55 0.54 0.44 1" specular="0.10" shininess="0.03" reflectance="0.02"/>"""

    shadow_bodies = ["cabinet"] + [f"drawer_{i}" for i in range(_NUM_DRAWERS)]

    return {
        "objects_xml": objects_xml,
        "materials_xml": materials_xml,
        "shadow_bodies": shadow_bodies,
    }


class RealisticImageRenderer:
    """Dual-model realistic renderer.

    Builds a separate MuJoCo model via make_model_generic() for realistic
    rendering. Physics runs on the VecEnv model; this only renders.
    qpos is copied from VecEnv model → realistic model each frame.
    """

    def __init__(self, task, vecenv, rl_img_size=84):
        import cv2 as _cv2
        self._cv2 = _cv2
        self.rl_img_size = rl_img_size

        franka_src = os.path.expanduser("~/Repos/Intern/Mujoco_Franka/src")
        sys.path.insert(0, franka_src)
        from scene_realistic import make_model_generic, RealisticRenderHelper
        from utils import load_calib, load_calib_wrist, CAM_W, CAM_H

        if task not in TASK_OBJECTS:
            raise ValueError(f"Realistic rendering not supported for task '{task}'. "
                             f"Supported: {list(TASK_OBJECTS.keys())}")

        # Drawer needs special handling: different calib + dynamic XML
        scene_xml = None
        if task == "drawer":
            # Load drawer-specific camera calibration (overrides globals)
            T_base_cam = load_calib(_DRAWER_CALIB)
            load_calib_wrist(_DRAWER_CALIB)
            # Reimport CAM_W/CAM_H after calib override
            import utils as _utils_mod
            CAM_W, CAM_H = _utils_mod.CAM_W, _utils_mod.CAM_H
            cfg = _build_drawer_config()
            scene_xml = _DRAWER_SCENE_XML
        else:
            T_base_cam = load_calib()
            cfg = TASK_OBJECTS[task]

        self.model = make_model_generic(
            T_base_cam,
            objects_xml=cfg["objects_xml"],
            materials_xml=cfg["materials_xml"],
            scene_xml=scene_xml,
        )
        self.data = mujoco.MjData(self.model)
        self.cam_w, self.cam_h = CAM_W, CAM_H

        # Verify nq matches VecEnv
        vecenv_nq = vecenv._envs[0]["model"].nq
        if self.model.nq != vecenv_nq:
            print(f"[realistic] WARNING: nq mismatch! VecEnv={vecenv_nq}, "
                  f"realistic={self.model.nq}. qpos copy may be wrong.",
                  file=sys.stderr)

        self.helper = RealisticRenderHelper(
            self.model,
            shadow_bodies=cfg["shadow_bodies"],
            cam_h=CAM_H, cam_w=CAM_W,
            wrist_h=CAM_H, wrist_w=CAM_W,
        )

        # Cache static body IDs for syncing (e.g. bowl in PnP)
        self._static_body_names = []
        if task == "pnp":
            self._static_body_names = ["bowl"]

        print(f"[realistic] Initialized: task={task}, nq={self.model.nq}, "
              f"cam={CAM_W}x{CAM_H} → {rl_img_size}x{rl_img_size}")

    def sync_and_render(self, vecenv, env_idx=0):
        """Copy qpos/qvel from VecEnv, forward, render realistic images.

        Returns dict: {camera_key: CHW uint8 array (3, rl_img_size, rl_img_size)}
        """
        src = vecenv._envs[env_idx]
        src_model, src_data = src["model"], src["data"]

        # Copy qpos and qvel
        nq = min(self.model.nq, src_model.nq)
        nv = min(self.model.nv, src_model.nv)
        self.data.qpos[:nq] = src_data.qpos[:nq]
        self.data.qvel[:nv] = src_data.qvel[:nv]

        # Sync static body positions (e.g. bowl)
        for bname in self._static_body_names:
            src_bid = mujoco.mj_name2id(src_model, mujoco.mjtObj.mjOBJ_BODY, bname)
            dst_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if src_bid >= 0 and dst_bid >= 0:
                self.model.body_pos[dst_bid] = src_model.body_pos[src_bid]

        mujoco.mj_forward(self.model, self.data)

        # Render base + wrist (returns BGR)
        img_base_bgr = self.helper.render_base(self.data)
        img_wrist_bgr = self.helper.render_wrist(self.data)

        # Convert BGR → RGB, resize, transpose to CHW
        sz = self.rl_img_size
        result = {}
        for cam_key, img_bgr in [("back", img_base_bgr), ("wrist", img_wrist_bgr)]:
            img_rgb = self._cv2.cvtColor(img_bgr, self._cv2.COLOR_BGR2RGB)
            img_small = self._cv2.resize(img_rgb, (sz, sz))
            result[cam_key] = np.transpose(img_small, (2, 0, 1)).astype(np.uint8)

        return result

    def close(self):
        try:
            self.helper.real_renderer.close()
        except Exception:
            pass
