#!/usr/bin/env python3
"""Realistic rendering test for each task.

Uses make_model_generic() + RealisticRenderHelper to render
base + wrist cameras with depth-based compositing.
"""
import os, sys, time
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
import cv2

# ── paths ────────────────────────────────────────────────────────
BASE_RL = os.path.dirname(os.path.abspath(__file__))
FRANKA_SRC = os.path.expanduser("~/Repos/Intern/Mujoco_Franka/src")
sys.path.insert(0, FRANKA_SRC)

from utils import (
    setup_egl, load_calib, HOME_QPOS, CAM_W, CAM_H,
    _bind_wrist_cam, get_model_ids, set_robot_pose,
)
from scene_realistic import make_model_generic, RealisticRenderHelper

setup_egl()

OUT_DIR = os.path.join(BASE_RL, "outputs", "render_test")
os.makedirs(OUT_DIR, exist_ok=True)

T_BASE_CAM = load_calib()

# ── per-task object definitions ──────────────────────────────────
CUBE_HALF = 0.02

TASKS = {
    "pnp": {
        "objects_xml": f"""
        <body name="cube" pos="0.45 0.10 {CUBE_HALF}">
          <freejoint name="cube_joint"/>
          <geom name="cube_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}"
                rgba="0.8 0.1 0.1 1" mass="0.1" friction="1.0 0.005 0.0001"/>
        </body>
        <body name="bowl" pos="0.45 -0.10 0.0">
          <geom name="bowl_geom" type="cylinder" size="0.05 0.005"
                rgba="0.9 0.85 0.75 1" mass="0.3" contype="1" conaffinity="1"/>
        </body>
        """,
        "materials_xml": "",
        "shadow_bodies": ["cube"],
    },
    "stack": {
        "objects_xml": f"""
        <body name="white_cube" pos="0.45 0.22 {CUBE_HALF}">
          <freejoint name="white_cube_joint"/>
          <geom name="white_cube_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}"
                rgba="0.9 0.9 0.85 1" mass="0.1" friction="1.0 0.005 0.0001"/>
        </body>
        <body name="green_cube" pos="0.45 -0.06 {CUBE_HALF}">
          <freejoint name="green_cube_joint"/>
          <geom name="green_cube_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}"
                rgba="0.1 0.7 0.2 1" mass="0.2" friction="1.0 0.005 0.0001"/>
        </body>
        """,
        "materials_xml": "",
        "shadow_bodies": ["white_cube", "green_cube"],
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
    "drawer": {
        "objects_xml": """
        <body name="cabinet" pos="0.50 0.0 0.0">
          <geom name="cabinet_body" type="box" size="0.10 0.12 0.06"
                rgba="0.35 0.25 0.18 1" mass="2.0" contype="1" conaffinity="1"/>
          <body name="drawer_body" pos="0.0 0.0 0.0">
            <joint name="drawer_slide" type="slide" axis="1 0 0" range="0 0.12"/>
            <geom name="drawer_geom" type="box" size="0.08 0.10 0.04" pos="0.08 0 0"
                  rgba="0.45 0.35 0.25 1" mass="0.3"/>
            <geom name="handle" type="cylinder" size="0.008 0.03" pos="0.16 0 0.02"
                  euler="0 90 0" rgba="0.6 0.6 0.6 1" mass="0.02"/>
          </body>
        </body>
        """,
        "materials_xml": "",
        "shadow_bodies": ["cabinet"],
    },
}


def test_task(task_name, cfg):
    print(f"[{task_name}] Building realistic model...")
    t0 = time.time()

    model = make_model_generic(
        T_BASE_CAM,
        objects_xml=cfg["objects_xml"],
        materials_xml=cfg["materials_xml"],
    )
    data = mujoco.MjData(model)

    # Set robot to home position
    nq_robot = min(len(HOME_QPOS), model.nq)
    data.qpos[:nq_robot] = HOME_QPOS[:nq_robot]
    mujoco.mj_forward(model, data)

    print(f"  Model built ({time.time()-t0:.1f}s), nq={model.nq}")

    # Create RealisticRenderHelper
    helper = RealisticRenderHelper(
        model,
        shadow_bodies=cfg["shadow_bodies"],
        cam_h=CAM_H, cam_w=CAM_W,
        wrist_h=CAM_H, wrist_w=CAM_W,
    )

    # Render
    t1 = time.time()
    img_base_bgr = helper.render_base(data)
    img_wrist_bgr = helper.render_wrist(data)
    render_ms = (time.time() - t1) * 1000

    # Save
    cv2.imwrite(os.path.join(OUT_DIR, f"{task_name}_realistic_base.png"), img_base_bgr)
    cv2.imwrite(os.path.join(OUT_DIR, f"{task_name}_realistic_wrist.png"), img_wrist_bgr)

    print(f"  ✓ base={img_base_bgr.shape}, wrist={img_wrist_bgr.shape}, render={render_ms:.0f}ms")
    helper.real_renderer.close()
    return True


if __name__ == "__main__":
    print(f"Realistic render test — output: {OUT_DIR}\n")

    for name, cfg in TASKS.items():
        try:
            test_task(name, cfg)
        except Exception as exc:
            print(f"  ✗ {name} FAILED: {exc}")
            import traceback; traceback.print_exc()

    print(f"\n✅ Done! Check {OUT_DIR}")
