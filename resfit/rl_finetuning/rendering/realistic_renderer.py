"""Realistic image renderer using depth-composited rendering with real backgrounds.

Builds a separate MuJoCo model via make_model_generic() for realistic
rendering. Physics runs on the VecEnv model; this only renders.
qpos is copied from VecEnv model → realistic model each frame.

Used by both:
  - Offline data collector (collect_offline_data_with_images.py)
  - Online RL training wrapper (mujoco_residual_wrapper_unified.py)
"""
import os
import sys

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
        from utils import load_calib, CAM_W, CAM_H

        if task not in TASK_OBJECTS:
            raise ValueError(f"Realistic rendering not supported for task '{task}'. "
                             f"Supported: {list(TASK_OBJECTS.keys())}")

        cfg = TASK_OBJECTS[task]
        T_base_cam = load_calib()

        self.model = make_model_generic(
            T_base_cam,
            objects_xml=cfg["objects_xml"],
            materials_xml=cfg["materials_xml"],
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
