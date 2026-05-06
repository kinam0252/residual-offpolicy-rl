#!/usr/bin/env python3
"""Real-robot GR00T VLA inference — Cartesian impedance control.

Publishes absolute EEF targets to /target_pose; reads feedback from /current_pose.
Uses GR00T base policy with dual cameras (cam_base + cam_wrist).

  1. Subscribe to cam_base + cam_wrist + EE pose via ROS2
  2. Run GR00T inference → 16-step absolute action chunk
  3. Publish Cartesian target poses to /target_pose
  4. Gripper commands

Output structure:
  output/{YYYYMMDD_HHMMSS}/{trial_N}/frames/step_XXXX.png
  output/{YYYYMMDD_HHMMSS}/{trial_N}/rollout.mp4
  output/{YYYYMMDD_HHMMSS}/{trial_N}/vision.mp4
  output/{YYYYMMDD_HHMMSS}/{trial_N}/ee_trajectory.png

Usage:
  python main_groot.py \
    --groot-checkpoint ~/kinam_dev/models/gr00t_sim_v2_checkpoint-70000 \
    --calib ../preproc/data/calib_result_base.calib \
    --desc "lift the cube"
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse, glob, os, queue, re, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import cv2
import torch

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from control_msgs.action import GripperCommand
from franka_msgs.action import Move as FrankaMove
from franka_msgs.action import Grasp as FrankaGrasp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────
_cwd = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_cwd, "..")
for p in [ROOT, os.path.join(ROOT, "utils")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Isaac-GR00T (gr00t package)
_groot_pkg = os.path.join(ROOT, "training", "Isaac-GR00T")
if _groot_pkg not in sys.path:
    sys.path.insert(0, _groot_pkg)
# GR00T inference
sys.path.insert(0, os.path.join(_cwd, "..", "groot_deploy"))
os.environ["TORCHDYNAMO_DISABLE"] = "1"
from groot_deploy.groot_inference import Gr00tInference

# Inline load_calib (avoids heavy open3d import from cam.py)
import yaml
from scipy.spatial.transform import Rotation as _R

# -- FoundationPose shared-memory reader --
import json as _json
FP_SHM_PATH = "/dev/shm/fp_cube_pose.json"
_FP_STALE_WARN_S = 2.0

def read_fp_pose(max_age_s=1.0):
    """Read latest cube pose from FP tracker via /dev/shm."""
    try:
        with open(FP_SHM_PATH, "r") as f:
            data = _json.load(f)
        if data.get("status") != "tracking":
            return None, None, None
        age = time.time() - data["timestamp"]
        pos = np.array(data["cube_pos"], dtype=np.float32)
        quat = np.array(data["cube_quat_wxyz"], dtype=np.float32)
        return pos, quat, age
    except (FileNotFoundError, KeyError, _json.JSONDecodeError):
        return None, None, None


# ── FoundationPose shared-memory reader ──────────────────────────────
import json as _json

FP_SHM_PATH = "/dev/shm/fp_cube_pose.json"
_FP_STALE_WARN_S = 2.0  # warn if pose older than this

def read_fp_pose(max_age_s=1.0):
    """Read latest cube pose from FP tracker (via /dev/shm).
    Returns (cube_pos_3, cube_quat_wxyz_4, age_s) or (None, None, None) if unavailable.
    """
    try:
        with open(FP_SHM_PATH, "r") as f:
            data = _json.load(f)
        if data.get("status") != "tracking":
            return None, None, None
        age = time.time() - data["timestamp"]
        pos = np.array(data["cube_pos"], dtype=np.float32)
        quat = np.array(data["cube_quat_wxyz"], dtype=np.float32)
        return pos, quat, age
    except (FileNotFoundError, KeyError, _json.JSONDecodeError):
        return None, None, None




# ── Trajectory comparison shared memory ──────────────────────────────
TRAJ_SHM_PATH = "/dev/shm/traj_compare.json"

def write_traj_shm(ee_pos, real_traj, sim_traj=None):
    """Write predicted trajectories to shared memory for MuJoCo viewer."""
    data = {
        "timestamp": time.time(),
        "ee_pos": ee_pos.tolist(),
        "real": real_traj.tolist(),
    }
    if sim_traj is not None:
        data["sim"] = sim_traj.tolist()
    tmp = TRAJ_SHM_PATH + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(data, f)
    os.replace(tmp, TRAJ_SHM_PATH)

def clear_traj_shm():
    try:
        os.remove(TRAJ_SHM_PATH)
    except FileNotFoundError:
        pass

# ── Checkpoint discovery & interactive selection ──────────────────────

CHECKPOINT_BASE_DIR = os.path.expanduser("~/kinam_dev/checkpoints")
DEFAULT_CALIB = os.path.expanduser(
    "~/kinam_dev/Mujoco/preproc/data/calib_result_base.calib")

# Task registry: each task has a description for GR00T and a results subdir
TASKS = {
    "cube_lift": {
        "desc": "lift the cube",
        "results_subdir": "cube_lift",
        "gripper_close_threshold": 0.75,  # 0~1 normalized
        "max_steps": 400,
        "gripper_latch": True,  # hold closed + require N open preds to release
        "gripper_settle": 2.0,
        "gripper_raw": False,   # input 0~1 normalized
        "init_pos": [0.3975, 0.2008, 0.1254],   # from lift_cube ep0
        "init_quat": [0.9998, -0.0032, -0.0179, 0.0013],
        # ── RL deploy config ──
        "as_mode": "wrapper",  # trained with per-task wrapper that has AS inside
        "rl_state_dim": 10,    # eef_pos(3)+eef_quat(4)+grip(2)+contact(1)
        "rl_object_state_dim": 7,  # cube pos(3)+quat_wxyz(4)
        "rl_action_scale": 0.1,
        "residual_pos_scale": 0.02,
        "residual_rot_scale": 0.05,
        "residual_grip_scale": 0.1,
        "grip_min": 0.0, "grip_max": 1.0,
    },
    "pick_and_place": {
        "desc": "Pick up the red cube and place it onto the plate.",
        "results_subdir": "pick_and_place",
        "gripper_close_threshold": 0.03,  # raw meters (~0.02 closed, ~0.04 open)
        "max_steps": 1000,
        "gripper_latch": False,  # PnP: model decides when to place (no latch)
        "gripper_settle": 0.0,
        "gripper_raw": True,    # input raw meters (finger_sum/2)
        "init_pos": [0.4391, -0.0311, 0.2470],   # from pnp_cube ep0
        "init_quat": [0.9985, 0.0224, 0.0466, -0.0193],
        # ── RL deploy config ──
        "as_mode": "script_only",  # trained with unified wrapper (no AS in wrapper)
        "rl_state_dim": 10,    # eef_pos(3)+eef_quat(4)+grip(2)+contact(1)
        "rl_object_state_dim": 10,  # cube pos(3)+quat(4) + plate pos(3)
        "rl_action_scale": 0.1,
        "residual_pos_scale": 0.02,
        "residual_rot_scale": 0.05,
        "residual_grip_scale": 0.1,
        "grip_min": 0.0, "grip_max": 1.0,
    },
    "close_drawer": {
        "desc": "Close the drawer",
        "results_subdir": "close_drawer",
        "gripper_close_threshold": 0.75,
        "max_steps": 600,
        "gripper_latch": False,
        "gripper_settle": 0.0,
        "gripper_raw": False,
        "gripper_always_closed": True,  # keep gripper closed throughout
        "init_pos": [0.2584, -0.2635, 0.3348],   # from close_drawer rosbag ep0
        "init_quat": [0.9993, -0.0241, -0.0257, -0.0100],  # xyzw from rosbag
        "ori_divergence_deg": 90.0,  # drawer task has wider orientation range
        # ── RL deploy config ──
        "as_mode": "wrapper",  # trained with per-task drawer wrapper that has AS inside
        "rl_state_dim": 8,     # eef_pos(3)+eef_quat(4)+grip(1) — no vel/contact
        "rl_object_state_dim": 5,  # drawer_slide(1)+active_drawer_idx(1)+face_center(3)
        "rl_action_scale": 0.05,  # from best checkpoint: d2only_as0.05_cw1000_delta_H (SR=100%)
        "residual_pos_scale": 0.02,
        "residual_rot_scale": 0.05,
        "residual_grip_scale": 0.004,
        "grip_min": 0.0, "grip_max": 0.04,
        "grip_always_zero": True,  # drawer is push task, grip always 0.0
    },
    "stack_cube": {
        "desc": "stack white cube on green cube",
        "results_subdir": "stack_cube",
        "gripper_close_threshold": 0.03,  # raw meters output (~0.0=closed, ~0.04=open)
        "max_steps": 600,
        "gripper_latch": 5,
        "gripper_settle": 1.0,
        "gripper_raw": True,    # model outputs raw meters
        "init_pos": [0.4088, 0.0450, 0.2814],   # from stack_cube rosbag ep0
        "init_quat": [0.9968, 0.0652, 0.0001, 0.0453],  # xyzw from rosbag
        # ── RL deploy config ──
        "as_mode": "wrapper",  # trained with per-task stack wrapper that has AS inside
        "rl_state_dim": 10,    # eef_pos(3)+eef_quat(4)+grip(2)+contact(1)
        "rl_object_state_dim": 10,  # white_cube(pos3+quat4=7) + green_pos(3) = 10D
        "rl_action_scale": 0.1,
        "residual_pos_scale": 0.02,
        "residual_rot_scale": 0.05,
        "residual_grip_scale": 0.004,
        "grip_min": 0.0, "grip_max": 0.04,
    },
    "stand_cup": {
        "desc": "stand cup up",
        "results_subdir": "stand_cup",
        "gripper_close_threshold": 0.03,  # raw meters output (~0.0=closed, ~0.04=open)
        "max_steps": 600,
        "gripper_latch": False,
        "gripper_settle": 0.0,
        "gripper_raw": True,    # model outputs raw meters
        "init_pos": [0.3433, 0.1202, 0.2756],   # from stand_cup rosbag ep0
        "init_quat": [0.9981, 0.0503, -0.0360, 0.0025],  # xyzw from rosbag
        # ── RL deploy config ──
        "as_mode": "script_only",  # trained with per-task cup wrapper (no AS in wrapper)
        "rl_state_dim": 8,     # eef_pos(3)+eef_quat(4)+grip(1)
        "rl_object_state_dim": 8,  # cup object state (pos+quat+uprightness etc)
        "rl_action_scale": 0.1,
        "residual_pos_scale": 0.02,
        "residual_rot_scale": 0.05,
        "residual_grip_scale": 0.004,
        "grip_min": 0.0, "grip_max": 0.04,
        "cup_gripper_latch": False,  # sim workaround; disable for real robot by default
        "cup_latch_close_thresh": 0.015,  # available if needed
        "cup_latch_open_thresh": 0.035,
        "cup_latch_open_steps": 32,
    },
}

_GROOT_TO_RL = {
    # cube_lift
    "groot_32ep": "gr00t_sim_32ep",
    "groot_66ep": "gr00t_sim_66ep",
    "groot_100ep": "gr00t_sim_100ep",
    # pick_and_place
    "groot_pnp_33ep": "gr00t_pnp_sim_33ep",
    "groot_pnp_66ep": "gr00t_pnp_sim_66ep",
    "groot_pnp_100ep": "gr00t_pnp_sim_100ep",
    # stack_cube
    "groot_stack_real_66ep": "gr00t_stack_real_66ep",
}


# Best sim eval success rates (from B200 async_eval_results)
_RL_BEST_SR = {
    "gr00t_sim_32ep": {
        "easy_s100": 72.5, "easy_s1L2_s2": 70.0, "easy_s42": 67.5,
        "easy_s7": 17.5, "easy_s99": 62.5, "easy_v2_noL2": 62.5,
        "hard_s100": 75.0, "hard_s13": 75.0, "hard_s1L2_s2": 42.5,
        "hard_s42": 52.5, "hard_s55": 37.5, "hard_s7": 50.0,
        "hard_s99": 12.5, "hard_v2_s42": 42.5,
        "normal_s100": 35.0, "normal_s1L2_s2": 52.5, "normal_s42": 87.5,
        "normal_s7": 0.0, "normal_s99": 40.0, "normal_v2_s42": 60.0,
    },
    "gr00t_sim_66ep": {
        "easy_s1L2_s2": 95.0, "easy_v2_noL2": 80.0, "easy_v2_s42": 62.5,
        "hard_s1L2_s2": 100.0, "hard_v2_s42": 100.0,
        "normal_s1L2_s2": 100.0,
    },
    "gr00t_sim_100ep": {
        "easy_s1L2_s2": 100.0, "easy_v2_noL2": 90.0,
        "hard_s1L2_s100": 97.5, "hard_s1L2_s13": 35.0,
        "hard_s1L2_s2": 100.0, "hard_s1L2_s42": 85.0,
        "hard_s1L2_s55": 35.0, "hard_s1L2_s7": 97.5,
        "hard_s1L2_s77": 35.0, "hard_s1L2_s99": 80.0,
        "normal_s1L2_s2": 100.0, "normal_v2_noL2": 95.0,
    },
}


def _rl_action_scale(variant):
    """Determine action_scale from variant name.
    s1L2/v2 variants: scale=1.0, old v1 variants (sN only): scale=0.2
    """
    parts = variant.split("_")
    for p in parts[1:]:
        if p in ("s1L2", "noL2") or p.startswith("v2"):
            return 1.0
    return 0.2


def _rl_variant_desc(variant, rl_sim_name=None, task_key=None):
    """Generate human-readable description from RL variant name."""
    parts = variant.split("_")
    diff = parts[0] if parts else "?"

    tags = []
    for p in parts[1:]:
        if p == "s1L2":
            tags.append("scale=1, L2")
        elif p == "noL2":
            tags.append("no L2")
        elif p.startswith("v2"):
            tags.append("v2")
        elif p.startswith("s") and p[1:].isdigit():
            tags.append(f"seed={p[1:]}")

    a_scale = _rl_action_scale(variant)
    # Task-level override for display
    if task_key and "rl_action_scale" in TASKS.get(task_key, {}):
        a_scale = TASKS[task_key]["rl_action_scale"]
    desc = f"{diff}, a={a_scale}"
    if tags:
        desc += ", " + ", ".join(tags)

    # Append sim SR if available
    if rl_sim_name and rl_sim_name in _RL_BEST_SR:
        sr = _RL_BEST_SR[rl_sim_name].get(variant)
        if sr is not None:
            desc += f" | SR={sr:.0f}%"

    return desc


def select_task():
    """Interactive task selection. Returns (task_key, task_desc, results_subdir)."""
    task_keys = list(TASKS.keys())
    if len(task_keys) == 1:
        tk = task_keys[0]
        t = TASKS[tk]
        print(f"\n  Task: {tk} (\"{t['desc']}\")")
        return tk, t["desc"], t["results_subdir"]

    print("\n" + "=" * 60)
    print("  SELECT TASK")
    print("=" * 60)
    for i, tk in enumerate(task_keys, 1):
        t = TASKS[tk]
        print(f"  [{i}] {tk}  —  \"{t['desc']}\"")
    print()

    choice = None
    while choice is None:
        try:
            raw = input("Select task [number]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(task_keys):
                choice = task_keys[idx]
            else:
                print(f"  Invalid. Enter 1-{len(task_keys)}")
        except (ValueError, EOFError):
            print(f"  Enter a number 1-{len(task_keys)}")

    t = TASKS[choice]
    print(f"  -> {choice} (\"{t['desc']}\")")
    return choice, t["desc"], t["results_subdir"]


def discover_and_select_checkpoints(ckpt_dir=CHECKPOINT_BASE_DIR,
                                     task_key="cube_lift"):
    """Auto-discover checkpoints and let user pick interactively.
    Returns (groot_path, residual_path_or_None, action_scale, epoch_tag).
    """
    ckpt_dir = os.path.expanduser(ckpt_dir)
    task_ckpt_dir = os.path.join(ckpt_dir, task_key)
    if not os.path.isdir(task_ckpt_dir):
        raise RuntimeError(f"No checkpoint dir for task: {task_ckpt_dir}")

    # ── Discover GR00T checkpoints ──
    groot_options = []
    for ep_dir in sorted(glob.glob(os.path.join(task_ckpt_dir, "groot_*"))):
        ep_name = os.path.basename(ep_dir)
        for step_name in sorted(os.listdir(ep_dir)):
            step_path = os.path.join(ep_dir, step_name)
            if os.path.isdir(step_path) and os.path.exists(
                    os.path.join(step_path, "config.json")):
                groot_options.append({
                    "path": step_path,
                    "label": f"{ep_name}/{step_name}",
                    "epoch_dir": ep_name,
                })

    if not groot_options:
        raise RuntimeError(
            f"No GR00T checkpoints found in {ckpt_dir}/groot_*/")

    # ── Print & select GR00T ──
    print("\n" + "=" * 60)
    print("  GR00T BASE POLICY CHECKPOINTS")
    print("=" * 60)
    for i, opt in enumerate(groot_options, 1):
        print(f"  [{i}] {opt['label']}")
    print()

    groot_choice = None
    while groot_choice is None:
        try:
            raw = input("Select GR00T checkpoint [number]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(groot_options):
                groot_choice = groot_options[idx]
            else:
                print(f"  Invalid. Enter 1-{len(groot_options)}")
        except (ValueError, EOFError):
            print(f"  Enter a number 1-{len(groot_options)}")

    print(f"  -> {groot_choice['label']}")
    groot_path = groot_choice["path"]
    epoch_dir = groot_choice["epoch_dir"]

    ep_match = re.search(r"(\d+ep)", epoch_dir)
    epoch_tag = ep_match.group(1) if ep_match else "unknown"

    # ── Discover ALL RL checkpoints ──
    matched_rl = _GROOT_TO_RL.get(epoch_dir)
    rl_best_dir = os.path.join(task_ckpt_dir, "rl_best")
    rl_options = []
    if os.path.isdir(rl_best_dir):
        for rl_sim_name in sorted(os.listdir(rl_best_dir)):
            rl_dir = os.path.join(rl_best_dir, rl_sim_name)
            if not os.path.isdir(rl_dir):
                continue
            for variant in os.listdir(rl_dir):
                best_pt = os.path.join(
                    rl_dir, variant, "checkpoints", "best.pt")
                if os.path.exists(best_pt):
                    sr = _RL_BEST_SR.get(rl_sim_name, {}).get(variant)
                    is_matched = (rl_sim_name == matched_rl)
                    rl_options.append({
                        "path": best_pt,
                        "label": f"{rl_sim_name}/{variant}",
                        "variant": variant,
                        "rl_sim_name": rl_sim_name,
                        "sr": sr if sr is not None else -1,
                        "matched": is_matched,
                        "action_scale": _rl_action_scale(variant),
                    })

    # Sort by SR descending (regardless of matching)
    rl_options.sort(key=lambda x: -x["sr"])
    TOP_N = 10

    # ── Print & select RL ──
    residual_path = None
    action_scale = 1.0

    def _print_rl_list(options, start=0, end=None):
        if end is None:
            end = len(options)
        for i in range(start, end):
            opt = options[i]
            desc = _rl_variant_desc(opt["variant"], opt["rl_sim_name"], task_key=task_key)
            tag = "*" if opt["matched"] else " "
            print(f"  [{i+1:2d}]{tag} {opt['label']:38s} ({desc})")

    if rl_options:
        print("\n" + "=" * 60)
        print(f"  RESIDUAL RL CHECKPOINTS  (* = matched with {matched_rl})")
        print("=" * 60)
        print("  [ 0] None (base policy only)")
        _print_rl_list(rl_options, 0, min(TOP_N, len(rl_options)))
        if len(rl_options) > TOP_N:
            print(f"  ... +{len(rl_options) - TOP_N} more (enter 'm' to show all)")
        print()

        rl_choice = None
        shown_all = len(rl_options) <= TOP_N
        while rl_choice is None:
            try:
                raw = input("Select RL checkpoint [number, 0=none]: ").strip()
                if raw.lower() == "m" and not shown_all:
                    _print_rl_list(rl_options, TOP_N)
                    shown_all = True
                    print()
                    continue
                idx = int(raw)
                if idx == 0:
                    rl_choice = "none"
                elif 1 <= idx <= len(rl_options):
                    rl_choice = rl_options[idx - 1]
                else:
                    print(f"  Invalid. Enter 0-{len(rl_options)}")
            except (ValueError, EOFError):
                print(f"  Enter a number 0-{len(rl_options)}")

        if rl_choice != "none":
            residual_path = rl_choice["path"]
            action_scale = rl_choice["action_scale"]
            # Task-level override (e.g. PnP uses 0.1 from training config)
            task_cfg = TASKS.get(task_key, {})
            if "rl_action_scale" in task_cfg:
                action_scale = task_cfg["rl_action_scale"]
            print(f"  -> {rl_choice['label']} (action_scale={action_scale})")
        else:
            print("  -> Base policy only (no RL)")
    else:
        print(f"\n  No RL checkpoints found for {epoch_dir}")

    print("=" * 60 + "\n")
    return groot_path, residual_path, action_scale, epoch_tag


# ── ActionScaler for deploy (mirrors training ActionScaler) ───────────
import json as _json_mod

class DeployActionScaler:
    """Min-max scaler: maps pos+grip to [-1,1] (mirrors training ActionScaler)."""

    def __init__(self, action_min, action_max, action_scale=1.0, no_clamp=False):
        self.action_min = np.asarray(action_min, dtype=np.float32)   # (4,) pos_x,y,z,grip
        self.action_max = np.asarray(action_max, dtype=np.float32)
        mid = (self.action_min + self.action_max) / 2
        half = (self.action_max - self.action_min) / 2
        half = np.maximum(half, 0.05)  # min_range_per_dim / 2
        expanded = half * (1 + action_scale)
        self._lo = mid - expanded
        self._hi = mid + expanded
        self._range = np.maximum(self._hi - self._lo, 1e-8)
        self.no_clamp = no_clamp

    def scale(self, pg):
        """Scale (4,) pos+grip → [-1,1]."""
        if self.no_clamp:
            return 2.0 * (pg - self._lo) / self._range - 1.0
        pg_c = np.clip(pg, self._lo, self._hi)
        return 2.0 * (pg_c - self._lo) / self._range - 1.0

    def unscale(self, scaled):
        """Unscale [-1,1] → original range."""
        if self.no_clamp:
            return self._lo + (scaled + 1.0) * (self._hi - self._lo) / 2.0
        scaled_c = np.clip(scaled, -1.0, 1.0)
        return self._lo + (scaled_c + 1.0) * (self._hi - self._lo) / 2.0

    @staticmethod
    def from_json(json_path, task_key, action_scale=1.0, no_clamp=False):
        """Load from action_scaler_stats.json."""
        with open(json_path) as f:
            stats = _json_mod.load(f)
        # Map deploy task_key → JSON key
        _key_map = {
            "cube_lift": "lift", "pick_and_place": "pnp",
            "close_drawer": "drawer", "stack_cube": "stack",
            "stand_cup": "cup",
        }
        jk = _key_map.get(task_key, task_key)
        if jk not in stats:
            raise KeyError(f"No ActionScaler stats for '{jk}' in {json_path}")
        entry = stats[jk]
        return DeployActionScaler(entry["min"], entry["max"], action_scale, no_clamp)


def build_rl_state(task_key, real_ee_pos, real_ee_quat, gripper_width,
                   obj_data, task_cfg):
    """Build RL state vector matching training: observation.state + observation.object_state.

    The actor receives [observation.state || observation.object_state || observation.base_action]
    where base_action is handled separately. This function builds state + object_state.

    Dims per task (must match checkpoint actor input - 7 for base_action):
        Lift:    state(10) + object(7)  = 17   → actor input 24
        PnP:     state(10) + object(10) = 20   → actor input 27
        Cup:     state(8)  + object(8)  = 16   → actor input 23
        Stack:   state(10) + object(10) = 20   → actor input 27
        Drawer:  state(8)  + object(5)  = 13   → actor input 20
    """
    _state_dim = task_cfg.get("rl_state_dim", 10)
    _obj_dim = task_cfg.get("rl_object_state_dim", 0)

    # ── Build observation.state ──
    if _state_dim == 10:
        # Lift/PnP/Stack: pos(3)+quat(4)+grip_qpos(2)+contact(1)
        gripper_raw = task_cfg.get("gripper_raw", False)
        if gripper_raw:
            gripper_qpos_2d = np.array([gripper_width, gripper_width], dtype=np.float32)
        else:
            gripper_qpos_2d = np.array([gripper_width * 0.04, gripper_width * 0.04], dtype=np.float32)
        contact_val = np.array([0.0], dtype=np.float32)
        state = np.concatenate([
            real_ee_pos.astype(np.float32),
            real_ee_quat.astype(np.float32),
            gripper_qpos_2d, contact_val,
        ])
    elif _state_dim == 8:
        # Cup/Drawer: pos(3)+quat(4)+grip(1) — no vel/contact
        gripper_raw = task_cfg.get("gripper_raw", False)
        if gripper_raw:
            grip_val = gripper_width
        else:
            grip_val = gripper_width * 0.04
        state = np.concatenate([
            real_ee_pos.astype(np.float32),
            real_ee_quat.astype(np.float32),
            np.array([grip_val], dtype=np.float32),
        ])
    else:
        raise ValueError(f"Unsupported rl_state_dim={_state_dim} for task {task_key}")

    # ── Build observation.object_state ──
    if _obj_dim == 0:
        return state

    if task_key == "cube_lift":
        # 7D: cube pos(3) + quat_wxyz(4)
        obj_state = np.concatenate([
            obj_data["cube_pos"], obj_data["cube_quat_wxyz"]
        ]).astype(np.float32)
    elif task_key == "pick_and_place":
        # 10D: cube pos(3)+quat(4) + plate/bowl pos(3)
        obj_state = np.concatenate([
            obj_data["cube_pos"], obj_data["cube_quat_wxyz"],
            obj_data["bowl_pos"],
        ]).astype(np.float32)
    elif task_key == "stack_cube":
        # 10D: white_cube pos(3)+quat_wxyz(4) + green_pos(3)
        obj_state = np.concatenate([
            obj_data["cube_pos"], obj_data["cube_quat_wxyz"],  # white cube
            obj_data["cube_b_pos"],                             # green cube position only
        ]).astype(np.float32)
    elif task_key == "stand_cup":
        # 8D: cup object state
        obj_state = np.concatenate([
            obj_data["cup_pos"], obj_data["cup_quat_wxyz"],
            np.array([obj_data.get("uprightness", 0.0)], dtype=np.float32),
        ]).astype(np.float32)
    elif task_key == "close_drawer":
        # 5D: drawer_slide(1) + active_drawer_idx(1) + face_center_world(3)
        # ── HARDCODED: matches sim training env (mujoco_vec_env_drawer.py) ──
        # All values fixed for D2 (drawer index 2) at episode start.
        # cabinet_pos=[0.5177, 0.2974, 0.005], euler=[0,0,-450]
        # drawer_slide = 0.13 (DRAWER_SLIDE, fully open)
        # face_center precomputed from cabinet geometry:
        #   D2=[0.5177, 0.1689, 0.1460], D3=[0.5177, 0.1689, 0.2004], D4=[0.5177, 0.1689, 0.2548]
        obj_state = np.array([
            0.13,    # HARDCODED: drawer_slide = fully open (episode start)
            2.0,     # HARDCODED: active_drawer_idx = D2
            0.5177,  # HARDCODED: face_center_world x (D2)
            0.1689,  # HARDCODED: face_center_world y (D2)
            0.1460,  # HARDCODED: face_center_world z (D2)
        ], dtype=np.float32)
    else:
        raise ValueError(f"Unknown task_key for object_state: {task_key}")

    assert obj_state.shape[0] == _obj_dim, (
        f"object_state dim mismatch for {task_key}: got {obj_state.shape[0]}, expected {_obj_dim}")

    return np.concatenate([state, obj_state])


def combine_actions_v2(base_pos, base_quat_xyzw, base_grip,
                       residual_7d, action_scaler=None, as_mode="script_only",
                       pos_scale=0.02, rot_scale=0.05, grip_scale=0.1,
                       grip_min=0.0, grip_max=1.0, grip_always_zero=False,
                       cup_latch_state=None):
    """Combine base + residual. Dual-mode matching training wrappers.

    Mode A (as_mode="wrapper" and action_scaler is not None):
        Position+Grip: normalize base → add residual (tanh*action_scale) → unscale
        Rotation: always physical delta (residual * rot_scale)
        This matches per-task wrappers (lift, stack, drawer) that have AS inside.

    Mode B (as_mode="script_only" or action_scaler is None):
        All physical delta: pos*pos_scale, rot*rot_scale, grip*grip_scale
        This matches unified wrapper and cup wrapper (no AS in wrapper).

    Args:
        cup_latch_state: if not None, dict with keys:
            'latched' (bool), 'open_count' (int),
            'close_thresh', 'open_thresh', 'open_steps'
            Will be mutated in-place. Grip is overridden by latch logic.
    """
    from scipy.spatial.transform import Rotation

    use_as_combine = (as_mode == "wrapper" and action_scaler is not None)

    if use_as_combine:
        # ── Mode A: ActionScaler combine (pos+grip in normalized space) ──
        base_pg = np.array([base_pos[0], base_pos[1], base_pos[2], base_grip],
                           dtype=np.float32)
        base_pg_norm = action_scaler.scale(base_pg)

        # Add residual in normalized space (residual is tanh*action_scale output)
        combined_pg_norm = base_pg_norm.copy()
        combined_pg_norm[:3] += residual_7d[:3]   # pos residual
        combined_pg_norm[3] += residual_7d[6]     # grip residual

        # Unscale back to physical
        combined_pg = action_scaler.unscale(combined_pg_norm)
        combined_pos = combined_pg[:3]
        combined_grip = float(np.clip(combined_pg[3], grip_min, grip_max))
    else:
        # ── Mode B: Physical delta ──
        combined_pos = base_pos + residual_7d[:3] * pos_scale
        combined_grip = float(np.clip(
            base_grip + residual_7d[6] * grip_scale, grip_min, grip_max))

    # Rotation: ALWAYS physical delta (both modes)
    res_euler = residual_7d[3:6] * rot_scale
    base_rot = Rotation.from_quat(base_quat_xyzw)
    delta_rot = Rotation.from_euler("xyz", res_euler)
    combined_rot = base_rot * delta_rot
    combined_quat = combined_rot.as_quat()

    # Drawer special: grip always 0.0
    if grip_always_zero:
        combined_grip = 0.0

    # Cup gripper latch (matches training cup wrapper logic)
    if cup_latch_state is not None:
        close_thresh = cup_latch_state.get("close_thresh", 0.015)
        open_thresh = cup_latch_state.get("open_thresh", 0.035)
        open_steps = cup_latch_state.get("open_steps", 32)

        if cup_latch_state["latched"]:
            # Latched closed — check if enough consecutive open commands
            if combined_grip > open_thresh:
                cup_latch_state["open_count"] += 1
                if cup_latch_state["open_count"] >= open_steps:
                    cup_latch_state["latched"] = False
                    cup_latch_state["open_count"] = 0
                    # Allow opening — keep combined_grip
                else:
                    combined_grip = grip_min  # still latched
            else:
                cup_latch_state["open_count"] = 0
                combined_grip = grip_min  # still latched
        else:
            # Not latched — check if close command
            if combined_grip < close_thresh:
                cup_latch_state["latched"] = True
                cup_latch_state["open_count"] = 0
                combined_grip = grip_min

    return combined_pos, combined_quat, combined_grip

class ResidualActor(torch.nn.Module):
    def __init__(self, input_dim=24, hidden_dim=256, output_dim=7,
                 action_scale=0.1):
        super().__init__()
        self.action_scale = action_scale
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(0.0),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(0.0),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Tanh(),
        )

    def forward(self, state, base_action_7d):
        x = torch.cat([state, base_action_7d], dim=-1)
        return self.policy(x) * self.action_scale

    @staticmethod
    def load_from_checkpoint(ckpt_path, device="cuda", action_scale=None):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Auto-read action_scale and use_action_scaler from checkpoint args
        ckpt_use_action_scaler = False
        if isinstance(ckpt, dict) and "args" in ckpt:
            ckpt_args = ckpt["args"]
            if isinstance(ckpt_args, dict):
                if action_scale is None and "action_scale" in ckpt_args:
                    action_scale = ckpt_args["action_scale"]
                ckpt_use_action_scaler = ckpt_args.get("use_action_scaler", False)
            elif hasattr(ckpt_args, "action_scale"):
                if action_scale is None:
                    action_scale = ckpt_args.action_scale
                ckpt_use_action_scaler = getattr(ckpt_args, "use_action_scaler", False)
        if action_scale is None:
            action_scale = 0.1
            print(f"  [ResidualActor] action_scale defaulting to {action_scale}")
        # Support both flat dict and nested { model: {...}} format
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

        actor_policy_sd = {}
        for k, v in state.items():
            if k.startswith("actor.policy."):
                actor_policy_sd[k.replace("actor.", "")]  = v

        input_dim = actor_policy_sd["policy.0.weight"].shape[1]
        hidden_dim = actor_policy_sd["policy.0.weight"].shape[0]
        output_dim = actor_policy_sd["policy.8.weight"].shape[0]

        actor = ResidualActor(input_dim, hidden_dim, output_dim,
                              action_scale=action_scale)
        actor.load_state_dict(actor_policy_sd)
        actor = actor.to(device).eval()
        print(f"  [ResidualActor] Loaded from {ckpt_path}")
        print(f"    input={input_dim}, hidden={hidden_dim}, output={output_dim}, action_scale={action_scale}")
        print(f"    use_action_scaler={ckpt_use_action_scaler} (from checkpoint)")
        return actor, ckpt_use_action_scaler

def combine_actions(base_pos, base_quat_xyzw, base_grip,
                    residual_7d, pos_scale=0.02, rot_scale=0.05, grip_scale=0.1):
    from scipy.spatial.transform import Rotation
    res_pos = residual_7d[:3] * pos_scale
    res_euler = residual_7d[3:6] * rot_scale
    res_grip = residual_7d[6] * grip_scale
    combined_pos = base_pos + res_pos
    base_rot = Rotation.from_quat(base_quat_xyzw)
    delta_rot = Rotation.from_euler("xyz", res_euler)
    combined_rot = base_rot * delta_rot
    combined_quat = combined_rot.as_quat()
    combined_grip = float(np.clip(base_grip + res_grip, 0.0, 1.0))
    return combined_pos, combined_quat, combined_grip


def build_base_action_7d(base_pos, base_quat_xyzw, base_grip):
    from scipy.spatial.transform import Rotation
    qn = np.linalg.norm(base_quat_xyzw)
    if qn > 1e-6:
        euler = Rotation.from_quat(base_quat_xyzw / qn).as_euler("xyz")
    else:
        euler = np.zeros(3)
    return np.concatenate([base_pos, euler, [base_grip]]).astype(np.float32)


def load_calib(calib_path):
    """Load .calib file and return 4x4 T_base_cam."""
    with open(calib_path, "r") as f:
        calib = yaml.safe_load(f)
    t = calib["transform"]["translation"]
    q = calib["transform"]["rotation"]  # x y z w
    T = np.eye(4)
    T[:3, :3] = _R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [t["x"], t["y"], t["z"]]
    return T

# Inline KeyListener (avoids utils.py which has heavy top-level imports)
class KeyListener:
    """Non-blocking keyboard listener that runs in a background thread."""
    def __init__(self):
        self._last_key = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            # Flush any leftover bytes in terminal input buffer
            import select as _sel
            while _sel.select([fd], [], [], 0)[0]:
                os.read(fd, 1024)
            while True:
                ch = sys.stdin.read(1)
                with self._lock:
                    self._last_key = ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def get_key(self):
        with self._lock:
            k = self._last_key
            self._last_key = None
            return k

_DEFAULT_INTRINSICS = {
    "fx": 906.0086, "fy": 904.6317,
    "cx": 650.8745, "cy": 374.5375,
    "width": 1280, "height": 720,
}

def project_points(pts_3d, T_base_cam, intrinsics, img_h, img_w):
    """Project (N, 3) base-frame points -> (N, 2) pixel coordinates."""
    T_cam_base = np.linalg.inv(T_base_cam)
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    pts_cam = (T_cam_base @ pts_h.T).T[:, :3]
    sx = img_w / intrinsics.get("width", 1280)
    sy = img_h / intrinsics.get("height", 720)
    fx, fy = intrinsics["fx"] * sx, intrinsics["fy"] * sy
    cx, cy = intrinsics["cx"] * sx, intrinsics["cy"] * sy
    u = fx * pts_cam[:, 0] / (pts_cam[:, 2] + 1e-8) + cx
    v = fy * pts_cam[:, 1] / (pts_cam[:, 2] + 1e-8) + cy
    return np.stack([u, v], axis=1)

# Init EE pose — defaults, overridden per-task in run()
INIT_EE_POS = np.array([0.3975, 0.2008, 0.1254])
INIT_EE_QUAT = np.array([0.9998, -0.0032, -0.0179, 0.0013])  # xyzw, cube_lift default

IMAGE_TOPICS = {
    "base_rgb": "/cam_base/camera/color/image_raw",
    "wrist_rgb": "/cam_hand/camera/color/image_raw",
}
DEPTH_TOPIC = "/cam_base/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/cam_base/camera/color/camera_info"

# Gripper thresholds (in normalized [0,1] space: 0=closed, 1=open)
# Training data: open=0.996, closed=0.494. Midpoint ~0.75.
GRIPPER_CLOSE_THRESHOLD = 0.75  # default; overridden per-task in run()

# Gripper latch: once closed, require N consecutive OPEN predictions to reopen
GRIPPER_LATCH_OPEN_COUNT = 8  # must see 8 consecutive open preds to unlatch
GRIPPER_SETTLE_TIME = 2.0     # seconds to wait after close before next inference
GRIPPER_HOLD_SEC = 3.0        # seconds to force-hold closed after first close

# ── Safety defaults ───────────────────────────────────────────────────
DEFAULT_WORKSPACE_MIN = np.array([0.2, -0.4, 0.0])
DEFAULT_WORKSPACE_MAX = np.array([0.7,  0.4, 0.5])
DEFAULT_MAX_STEP_POS_M = 0.02        # 2 cm per step (training max ~1.6cm at p99)
DEFAULT_MAX_STEP_ROT_RAD = 0.05      # ~2.9 deg per step (training max ~2.6deg at p99)
DEFAULT_SENSOR_TIMEOUT_S = 1.0       # stale sensor threshold
DEFAULT_ORI_DIVERGENCE_DEG = 45.0    # orientation anomaly threshold
DEFAULT_MAX_SPEED_M_S = 0.5          # velocity limit (m/s)


# ── Safety monitor ────────────────────────────────────────────────────

class SafetyMonitor:
    """Collects safety events and prints summary on shutdown."""

    def __init__(self):
        self.counts = {
            "workspace_clamp": 0,
            "step_clamp_pos": 0,
            "step_clamp_rot": 0,
            "nan_inf_skip": 0,
            "sensor_stale_stop": 0,
            "estop": 0,
            "ori_divergence_stop": 0,
            "speed_limit": 0,
        }
        self.log_lines = []

    def record(self, event, msg=""):
        self.counts[event] = self.counts.get(event, 0) + 1
        line = f"[SAFETY] {event}: {msg}" if msg else f"[SAFETY] {event}"
        self.log_lines.append(line)
        print(line, flush=True)

    def summary(self):
        print("\n" + "=" * 50, flush=True)
        print("SAFETY SUMMARY", flush=True)
        print("=" * 50, flush=True)
        total = sum(self.counts.values())
        for k, v in self.counts.items():
            if v > 0:
                print(f"  {k}: {v}", flush=True)
        if total == 0:
            print("  No safety events triggered.", flush=True)
        print("=" * 50 + "\n", flush=True)


def check_action_sanity(pos, quat):
    """Return True if action values are valid (no NaN/Inf, reasonable range)."""
    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
        return False
    if np.any(np.isnan(quat)) or np.any(np.isinf(quat)):
        return False
    if np.linalg.norm(pos) > 5.0:  # >5m from origin is clearly wrong
        return False
    if abs(np.linalg.norm(quat) - 1.0) > 0.1:  # quaternion should be unit
        return False
    return True


def clamp_workspace(pos, ws_min, ws_max, safety_monitor):
    """Clamp position to workspace bounds. Returns clamped position."""
    clamped = np.clip(pos, ws_min, ws_max)
    if not np.allclose(pos, clamped, atol=1e-6):
        safety_monitor.record(
            "workspace_clamp",
            f"[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] -> "
            f"[{clamped[0]:.3f},{clamped[1]:.3f},{clamped[2]:.3f}]")
    return clamped


def clamp_max_step(target_pos, target_quat, cur_pos, cur_quat,
                   max_pos_m, max_rot_rad, safety_monitor):
    """Clamp displacement per step. Returns (clamped_pos, clamped_quat)."""
    from scipy.spatial.transform import Rotation

    # Position clamp
    delta = target_pos - cur_pos
    dist = np.linalg.norm(delta)
    if dist > max_pos_m and dist > 1e-12:
        target_pos = cur_pos + delta * (max_pos_m / dist)
        safety_monitor.record(
            "step_clamp_pos", f"dist={dist:.4f}m -> {max_pos_m:.4f}m")

    # Rotation clamp
    cur_rot = Rotation.from_quat(cur_quat)   # xyzw
    tgt_rot = Rotation.from_quat(target_quat)
    rel = tgt_rot * cur_rot.inv()
    angle = rel.magnitude()
    if angle > max_rot_rad and angle > 1e-9:
        rel_clamped = Rotation.from_rotvec(
            rel.as_rotvec() * (max_rot_rad / angle))
        tgt_rot = rel_clamped * cur_rot
        target_quat = tgt_rot.as_quat()  # xyzw
        safety_monitor.record(
            "step_clamp_rot", f"angle={np.degrees(angle):.1f}deg -> "
            f"{np.degrees(max_rot_rad):.1f}deg")

    return target_pos, target_quat


def check_orientation_divergence(target_quat, cur_quat, max_deg):
    """Return True if orientation difference exceeds threshold."""
    q1 = cur_quat / (np.linalg.norm(cur_quat) + 1e-12)
    q2 = target_quat / (np.linalg.norm(target_quat) + 1e-12)
    dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    angle_deg = np.degrees(2.0 * np.arccos(dot))
    return angle_deg > max_deg, angle_deg


# ── Debug logger ──────────────────────────────────────────────────────

import json as _json

class DebugLogger:
    """Per-step debug log: input state, raw model output, post-safety output.

    Saves to {trial_dir}/debug_log.jsonl (one JSON object per line).
    Also saves input images every N steps to {trial_dir}/debug_images/.
    """

    def __init__(self, save_images_every=5):
        self._fh = None
        self._img_dir = None
        self._save_images_every = save_images_every
        self._step = 0

    def start(self, trial_dir):
        self._trial_dir = trial_dir
        path = os.path.join(trial_dir, "debug_log.jsonl")
        self._fh = open(path, "w")
        self._img_dir = os.path.join(trial_dir, "debug_images")
        os.makedirs(self._img_dir, exist_ok=True)
        self._step = 0
        print(f"  [DebugLogger] -> {path}", flush=True)

    def stop(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def _arr(self, a):
        """Numpy array to list for JSON serialization."""
        if a is None:
            return None
        return np.asarray(a).tolist()

    def log_step(self, step, *, input_state, raw_output, safe_output,
                 inference_time, base_output=None, residuals=None,
                 cam_base=None, cam_wrist=None):
        """Log one inference step.

        Args:
            input_state: dict with ee_pos, ee_quat, gripper_width
            raw_output: dict with pred_pos, pred_quat, pred_grip (after residual)
            base_output: dict with pred_pos, pred_quat, pred_grip (before residual, optional)
            residuals: list of 7D residual vectors per waypoint (optional)
            safe_output: dict with waypoints, grip_cmds (after safety)
            inference_time: float seconds
            cam_base/cam_wrist: RGB images (optional, saved periodically)
        """
        if self._fh is None:
            return

        # Compute deltas for analysis
        cur_pos = np.array(input_state["ee_pos"])
        raw_pos0 = np.array(raw_output["pred_pos"][0]) if len(raw_output["pred_pos"]) > 0 else cur_pos
        raw_delta_mm = float(np.linalg.norm(raw_pos0 - cur_pos) * 1000)

        # Raw action spread (how much the action chunk varies)
        raw_positions = np.array(raw_output["pred_pos"])
        if len(raw_positions) > 1:
            raw_spread_mm = float(np.linalg.norm(
                raw_positions[-1] - raw_positions[0]) * 1000)
        else:
            raw_spread_mm = 0.0

        entry = {
            "step": step,
            "time": time.time(),
            "inference_time_s": round(inference_time, 4),
            "input": {
                "ee_pos": self._arr(input_state["ee_pos"]),
                "ee_quat": self._arr(input_state["ee_quat"]),
                "gripper_width": round(float(input_state["gripper_width"]), 4),
            },
            "raw_output": {
                "pred_pos": self._arr(raw_output["pred_pos"]),
                "pred_quat": self._arr(raw_output["pred_quat"]),
                "pred_grip": self._arr(raw_output["pred_grip"]),
            },
            "analysis": {
                "raw_delta_from_current_mm": round(raw_delta_mm, 2),
                "raw_chunk_spread_mm": round(raw_spread_mm, 2),
                "n_waypoints_after_safety": len(safe_output.get("waypoints", [])),
            },
            "safe_output": {
                "waypoints": self._arr(safe_output.get("waypoints")),
                "grip_cmds": self._arr(safe_output.get("grip_cmds")),
            },
        }
        if base_output is not None:
            entry["base_output"] = {
                "pred_pos": self._arr(base_output["pred_pos"]),
                "pred_quat": self._arr(base_output["pred_quat"]),
                "pred_grip": self._arr(base_output["pred_grip"]),
            }
        if residuals is not None:
            entry["residuals"] = self._arr(residuals)
        self._fh.write(_json.dumps(entry) + "\n")
        self._fh.flush()

        # Save images periodically
        if (cam_base is not None and self._img_dir
                and self._step % self._save_images_every == 0):
            cv2.imwrite(
                os.path.join(self._img_dir, f"step_{step:04d}_base.jpg"),
                cv2.cvtColor(cam_base, cv2.COLOR_RGB2BGR))
            if cam_wrist is not None:
                cv2.imwrite(
                    os.path.join(self._img_dir, f"step_{step:04d}_wrist.jpg"),
                    cv2.cvtColor(cam_wrist, cv2.COLOR_RGB2BGR))

        self._step += 1


# ── ROS2 image conversion ────────────────────────────────────────────

def msg_to_numpy(msg):
    """Convert a sensor_msgs/Image to a numpy array."""
    channels = 1
    if msg.encoding == "16UC1":
        dtype = np.uint16
    elif msg.encoding in ("rgb8", "bgr8"):
        dtype = np.uint8
        channels = 3
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    img = np.frombuffer(msg.data, dtype=dtype)
    if channels > 1:
        img = img.reshape((msg.height, msg.width, channels))
        if msg.encoding == "bgr8":
            img = img[..., ::-1].copy()
    else:
        img = img.reshape((msg.height, msg.width))
    return img


# ── High-FPS vision recorder ─────────────────────────────────────────

class VisionRecorder:
    """Record base_rgb frames to a video file."""

    def __init__(self, resolution=(640, 480), fps=30):
        self.resolution = resolution
        self.fps = fps
        self._writer = None
        self._recording = False
        self._lock = threading.Lock()
        self.frame_count = 0
        self._video_path = None

    def start(self, video_path):
        with self._lock:
            if self._writer:
                self._writer.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                video_path, fourcc, self.fps, self.resolution)
            self._video_path = video_path
            self.frame_count = 0
            self._recording = True
        print(f"  [VisionRecorder] Recording -> {video_path}", flush=True)

    def stop(self):
        with self._lock:
            self._recording = False
            if self._writer:
                self._writer.release()
                self._writer = None
                print(f"  [VisionRecorder] Stopped "
                      f"({self.frame_count} frames -> {self._video_path})",
                      flush=True)

    def feed_frame(self, rgb_image):
        with self._lock:
            if self._recording and self._writer:
                resized = cv2.resize(rgb_image, self.resolution)
                bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                self._writer.write(bgr)
                self.frame_count += 1


# ── Rosbag recorder (for lerobot training data) ─────────────────────

class RosbagRecorder:
    """Record 6 ROS topics to a rosbag for later lerobot/FP conversion.

    Topics recorded:
      /franka_robot_state_broadcaster/current_pose  (PoseStamped, ~30Hz downsampled)
      /target_pose                                   (PoseStamped)
      /franka_gripper/joint_states                   (JointState, ~14Hz)
      /cam_base/camera/color/image_raw/compressed    (CompressedImage, ~15fps)
      /cam_hand/camera/color/image_raw/compressed    (CompressedImage, ~15fps)
      /cam_base/camera/aligned_depth_to_color/image_raw/compressedDepth
                                                     (CompressedImage, ~15fps)

    Uses an async writer thread to avoid blocking callbacks/control loop.
    Camera images are JPEG-encoded in the writer thread from raw numpy arrays.
    Depth is PNG-encoded (lossless 16-bit) for FoundationPose compatibility.
    """

    TOPICS = {
        "current_pose": ("/franka_robot_state_broadcaster/current_pose",
                         "geometry_msgs/msg/PoseStamped"),
        "target_pose": ("/target_pose",
                        "geometry_msgs/msg/PoseStamped"),
        "gripper": ("/franka_gripper/joint_states",
                    "sensor_msgs/msg/JointState"),
        "cam_base": ("/cam_base/camera/color/image_raw/compressed",
                     "sensor_msgs/msg/CompressedImage"),
        "cam_wrist": ("/cam_hand/camera/color/image_raw/compressed",
                      "sensor_msgs/msg/CompressedImage"),
        "depth": ("/cam_base/camera/aligned_depth_to_color/image_raw/compressedDepth",
                  "sensor_msgs/msg/CompressedImage"),
    }

    def __init__(self, jpeg_quality=95, pose_hz=30, queue_size=800):
        self._jpeg_quality = jpeg_quality
        self._pose_min_interval = 1.0 / pose_hz
        self._queue_size = queue_size
        self._queue = None
        self._thread = None
        self._recording = False
        self._last_pose_time = 0.0
        self._msg_count = 0
        self._drop_count = 0

    def start(self, trial_dir):
        """Open a new rosbag in trial_dir/rosbag/ and start the writer thread."""
        if self._recording:
            self.stop()
        bag_dir = os.path.join(trial_dir, "rosbag")
        self._queue = queue.Queue(maxsize=self._queue_size)
        self._recording = True
        self._last_pose_time = 0.0
        self._msg_count = 0
        self._drop_count = 0
        self._thread = threading.Thread(
            target=self._writer_loop, args=(bag_dir,), daemon=True)
        self._thread.start()
        print(f"  [RosbagRecorder] Recording -> {bag_dir}", flush=True)

    def stop(self):
        """Signal the writer thread to stop and wait for it."""
        if not self._recording:
            return
        self._recording = False
        if self._queue:
            self._queue.put(None)  # sentinel
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        drops = self._drop_count
        msgs = self._msg_count
        extra = f" ({drops} dropped)" if drops else ""
        print(f"  [RosbagRecorder] Stopped ({msgs} messages{extra})", flush=True)

    def push_ros_msg(self, key, msg, timestamp_ns):
        """Push a pre-built ROS message for direct serialization."""
        if not self._recording or self._queue is None:
            return
        if key == "current_pose":
            now = time.monotonic()
            if now - self._last_pose_time < self._pose_min_interval:
                return
            self._last_pose_time = now
        try:
            self._queue.put_nowait(("ros_msg", key, msg, timestamp_ns))
        except queue.Full:
            self._drop_count += 1

    def push_image(self, key, rgb_array, timestamp_ns):
        """Push a numpy RGB image for JPEG encoding in the writer thread."""
        if not self._recording or self._queue is None:
            return
        try:
            self._queue.put_nowait(("image", key, rgb_array, timestamp_ns))
        except queue.Full:
            self._drop_count += 1

    def push_depth(self, depth_uint16, timestamp_ns):
        """Push a 16UC1 depth array for PNG encoding in the writer thread."""
        if not self._recording or self._queue is None:
            return
        try:
            self._queue.put_nowait(("depth", "depth", depth_uint16, timestamp_ns))
        except queue.Full:
            self._drop_count += 1

    def _writer_loop(self, bag_dir):
        """Writer thread: open bag, process queue, close bag."""
        try:
            import rosbag2_py
            from rclpy.serialization import serialize_message
            from sensor_msgs.msg import CompressedImage

            writer = rosbag2_py.SequentialWriter()
            storage = rosbag2_py.StorageOptions(
                uri=bag_dir, storage_id="sqlite3")
            converter = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr")
            writer.open(storage, converter)

            for key, (topic_name, msg_type) in self.TOPICS.items():
                topic_meta = rosbag2_py.TopicMetadata(
                    name=topic_name,
                    type=msg_type,
                    serialization_format="cdr")
                writer.create_topic(topic_meta)

            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]

            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if not self._recording:
                        break
                    continue

                if item is None:
                    break

                try:
                    kind, key, data, ts_ns = item
                    topic_name = self.TOPICS[key][0]

                    if kind == "ros_msg":
                        serialized = serialize_message(data)
                        writer.write(topic_name, serialized, ts_ns)
                    elif kind == "image":
                        bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                        ok, buf = cv2.imencode(".jpg", bgr, jpeg_params)
                        if ok:
                            img_msg = CompressedImage()
                            img_msg.format = "jpeg"
                            img_msg.data = buf.tobytes()
                            serialized = serialize_message(img_msg)
                            writer.write(topic_name, serialized, ts_ns)
                    elif kind == "depth":
                        ok, buf = cv2.imencode(".png", data)
                        if ok:
                            img_msg = CompressedImage()
                            img_msg.format = "16UC1; compressedDepth png"
                            img_msg.data = buf.tobytes()
                            serialized = serialize_message(img_msg)
                            writer.write(topic_name, serialized, ts_ns)
                    self._msg_count += 1
                except Exception as e:
                    print(f"  [RosbagRecorder] Write error: {e}", flush=True)

            del writer
        except Exception as e:
            print(f"  [RosbagRecorder] Fatal error: {e}", flush=True)


# ── Per-step visualization ────────────────────────────────────────────

def save_step_frame(step, rgb, ee_pos, pred_positions, gripper_width,
                    ee_history, T_base_cam, intrinsics, frame_dir):
    """Save a 2-panel figure: [RGB w/ trajectories | XYZ plots].

    pred_positions: (N, 3) absolute predicted EE positions from GR00T.
    """
    if intrinsics is None:
        intrinsics = _DEFAULT_INTRINSICS
    H, W = rgb.shape[:2]

    vis = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

    # Past EE path (yellow)
    if len(ee_history) >= 2:
        past_pts = np.array(ee_history)
        past_uv = project_points(past_pts, T_base_cam, intrinsics, H, W).astype(np.int32)
        cv2.polylines(vis, [past_uv.reshape(-1, 1, 2)], False,
                      (0, 220, 255), 2, cv2.LINE_AA)

    # Predicted trajectory (red) -- includes current pos as first point
    all_pred = np.vstack([ee_pos[None, :], pred_positions])
    pred_uv = project_points(all_pred, T_base_cam, intrinsics, H, W).astype(np.int32)
    cv2.polylines(vis, [pred_uv.reshape(-1, 1, 2)], False,
                  (0, 0, 255), 3, cv2.LINE_AA)
    for i in range(1, len(pred_uv)):
        alpha = i / len(pred_uv)
        color = (0, int(50 + 200 * (1 - alpha)), 255)
        cv2.circle(vis, tuple(pred_uv[i]), 4, color, -1, cv2.LINE_AA)

    # Current EE dot
    cv2.circle(vis, tuple(pred_uv[0]), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, tuple(pred_uv[0]), 7, (0, 140, 255), 2, cv2.LINE_AA)

    # Text overlay
    info = (f"step={step}  pos=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]  "
            f"grip={gripper_width:.3f}")
    cv2.putText(vis, info, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, info, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # --- Right panel: XYZ plots ---
    ee_arr = np.array(ee_history) if len(ee_history) > 0 else np.zeros((1, 3))
    pred_steps = np.arange(len(ee_arr) - 1, len(ee_arr) - 1 + len(all_pred))
    colors_past = ["#e74c3c", "#27ae60", "#2980b9"]
    colors_pred = ["#ff9999", "#90ee90", "#87ceeb"]

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.4, wspace=0.3)

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(vis_rgb, aspect="auto")
    ax_img.set_axis_off()
    ax_img.set_title(f"Step {step} [GR00T]", fontsize=11, fontweight="bold")

    past_x = np.arange(len(ee_arr))
    for d, lbl in enumerate(["X", "Y", "Z"]):
        ax = fig.add_subplot(gs[d, 1])
        ax.plot(past_x, ee_arr[:, d], "-", color=colors_past[d],
                linewidth=1.5, label="past")
        ax.plot(pred_steps, all_pred[:, d], "--", color=colors_pred[d],
                linewidth=1.5, label="pred")
        ax.axvline(len(ee_arr) - 1, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel(f"{lbl} (m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7, loc="upper right")
        if d < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("step", fontsize=9)

    os.makedirs(frame_dir, exist_ok=True)
    path = os.path.join(frame_dir, f"step_{step:04d}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── GR00T Inference Node ──────────────────────────────────────────────

class InferenceNode(Node):
    """ROS2 node: subscribes to sensors, publishes Cartesian targets."""

    def __init__(self, args):
        super().__init__("groot_inference_node")
        self.args = args
        self.lock = threading.Lock()

        # ── Sensor state ──
        self.latest_images = {k: None for k in IMAGE_TOPICS}
        self.camera_intrinsics = None
        self.current_joint_pos = None
        self.current_ee_pos = None
        self.current_ee_quat = None  # xyzw
        self.gripper_finger_positions = np.array([0.04, 0.04], dtype=np.float32)

        # ── Sensor timestamps (for staleness detection) ──
        self._sensor_ts = {
            "base_rgb": 0.0,
            "wrist_rgb": 0.0,
            "ee_pose": 0.0,
            "gripper": 0.0,
        }

        # ── Rosbag recorder (set externally before trial) ──
        self.rosbag_recorder = None

        # ── ROS2 subscriptions ──
        self.image_subs = []
        for key, topic in IMAGE_TOPICS.items():
            sub = self.create_subscription(
                Image, topic,
                lambda msg, k=key: self._cb_image(msg, k), 10)
            self.image_subs.append(sub)

        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._cb_camera_info, 10)
        self.create_subscription(
            JointState, "/franka/joint_states", self._cb_joint_states, 10)
        self.create_subscription(
            PoseStamped, "/franka_robot_state_broadcaster/current_pose", self._cb_current_pose, 10)
        self.create_subscription(
            JointState, "/franka_gripper/joint_states", self._cb_gripper, 10)
        self.create_subscription(
            Image, DEPTH_TOPIC, self._cb_depth, 10)

        # ── ROS2 publishers ──
        self.target_pose_pub = self.create_publisher(
            PoseStamped, "/target_pose", 10)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            "/gripper/gripper_position_controller/commands", 10)
        self._gripper_action_client = ActionClient(
            self, GripperCommand, "/franka_gripper/gripper_action")
        self._gripper_move_client = ActionClient(
            self, FrankaMove, "/franka_gripper/move")
        self._gripper_grasp_client = ActionClient(
            self, FrankaGrasp, "/franka_gripper/grasp")

        self.get_logger().info(
            "GR00T InferenceNode ready -- Cartesian impedance mode")

    # ── Callbacks ──

    def _cb_image(self, msg, key):
        try:
            img = msg_to_numpy(msg)
            with self.lock:
                self.latest_images[key] = img
                self._sensor_ts[key] = time.time()
        except Exception as e:
            self.get_logger().error(f"Image error on {key}: {e}")

    def _cb_camera_info(self, msg):
        try:
            k = msg.k
            self.camera_intrinsics = {
                "width": msg.width, "height": msg.height,
                "fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5],
            }
        except Exception as e:
            self.get_logger().error(f"CameraInfo error: {e}")

    def _cb_joint_states(self, msg):
        try:
            idx = {name: i for i, name in enumerate(msg.name)}
            names = [f"fr3_joint{i}" for i in range(1, 8)]
            j = [msg.position[idx[n]] for n in names if n in idx]
            if len(j) == 7:
                self.current_joint_pos = np.array(j)
        except Exception:
            pass

    def _cb_current_pose(self, msg):
        try:
            p = msg.pose.position
            o = msg.pose.orientation
            self.current_ee_pos = np.array([p.x, p.y, p.z])
            self.current_ee_quat = np.array([o.x, o.y, o.z, o.w])
            self._sensor_ts["ee_pose"] = time.time()
            if self.rosbag_recorder:
                ts = self.get_clock().now().nanoseconds
                self.rosbag_recorder.push_ros_msg("current_pose", msg, ts)
        except Exception:
            pass

    def _cb_gripper(self, msg):
        """Handle /franka_gripper/joint_states."""
        if len(msg.position) >= 2:
            self.gripper_finger_positions = np.array(
                msg.position[:2], dtype=np.float32)
            self._sensor_ts["gripper"] = time.time()
            if self.rosbag_recorder:
                ts = self.get_clock().now().nanoseconds
                self.rosbag_recorder.push_ros_msg("gripper", msg, ts)

    def _cb_depth(self, msg):
        """Handle aligned depth image — forward to rosbag only."""
        if self.rosbag_recorder and self.rosbag_recorder._recording:
            try:
                raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    msg.height, msg.width)
                ts = self.get_clock().now().nanoseconds
                self.rosbag_recorder.push_depth(raw.copy(), ts)
            except Exception:
                pass

    # ── State query ──

    def get_state(self):
        with self.lock:
            images = {k: v for k, v in self.latest_images.items()}
        finger_sum = float(self.gripper_finger_positions.sum())
        if getattr(self, '_gripper_raw', False):
            gripper_width = finger_sum / 2.0  # half-width in meters for PnP
        else:
            gripper_width = float(np.clip(finger_sum / 0.08, 0.0, 1.0))
        return {
            "images": images,
            "joint_pos": (self.current_joint_pos.tolist()
                          if self.current_joint_pos is not None else None),
            "ee_pos": (self.current_ee_pos.tolist()
                       if self.current_ee_pos is not None else None),
            "ee_quat": (self.current_ee_quat.tolist()
                        if self.current_ee_quat is not None else None),
            "gripper_width": gripper_width,
            "camera_intrinsics": self.camera_intrinsics,
        }

    def sensors_alive(self, required=("base_rgb", "wrist_rgb", "ee_pose")):
        """Return True if all required sensors have ever published."""
        return all(self._sensor_ts.get(k, 0.0) > 0.0 for k in required)

    def check_staleness(self, timeout_s, required=("base_rgb", "wrist_rgb", "ee_pose")):
        """Return list of stale sensor names (empty = all fresh)."""
        now = time.time()
        stale = []
        for k in required:
            ts = self._sensor_ts.get(k, 0.0)
            if ts > 0.0 and (now - ts) > timeout_s:
                stale.append(k)
        return stale

    # ── Publishing ──

    def publish_target_pose(self, pos, quat_xyzw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.orientation.w = float(quat_xyzw[3])
        self.target_pose_pub.publish(msg)
        if self.rosbag_recorder:
            ts = self.get_clock().now().nanoseconds
            self.rosbag_recorder.push_ros_msg("target_pose", msg, ts)

    def publish_gripper(self, value):
        """Send gripper command. value: 1.0=open (Move), 0.0=closed (Grasp with force)."""
        if value > 0.5:
            # OPEN: use Move action
            goal = FrankaMove.Goal()
            goal.width = 0.08
            goal.speed = 0.1
            self._gripper_move_client.send_goal_async(goal)
        else:
            # CLOSE: use Grasp action (maintains force on object)
            goal = FrankaGrasp.Goal()
            goal.width = 0.02
            goal.epsilon.inner = 0.04
            goal.epsilon.outer = 0.04
            goal.speed = 0.1
            goal.force = 50.0  # Newtons - firm grip
            self._gripper_grasp_client.send_goal_async(goal)

    def publish_gripper_action(self, width_m, effort=50.0):
        """Publish gripper via action interface (width in meters)."""
        goal = GripperCommand.Goal()
        goal.command.position = float(width_m)
        goal.command.max_effort = effort
        self._gripper_action_client.send_goal_async(goal)

    def send_stop(self):
        """Freeze robot by republishing current pose."""
        if self.current_ee_pos is not None and self.current_ee_quat is not None:
            self.publish_target_pose(self.current_ee_pos, self.current_ee_quat)
            self.get_logger().info("STOP -- set target to current pose")

    def wait_for_convergence(self, target_pos, target_quat,
                             pos_thresh_mm=5.0, ori_thresh_deg=5.0,
                             timeout_sec=10.0):
        """Block until EE converges to target or timeout."""
        from scipy.spatial.transform import Rotation
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.current_ee_pos is None:
                continue
            pos_err = np.linalg.norm(self.current_ee_pos - target_pos) * 1000.0
            q1 = self.current_ee_quat / np.linalg.norm(self.current_ee_quat)
            q2 = target_quat / np.linalg.norm(target_quat)
            dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
            ori_err = np.degrees(2.0 * np.arccos(dot))
            if pos_err < pos_thresh_mm and ori_err < ori_thresh_deg:
                return True
        return False


# ── Main inference loop ───────────────────────────────────────────────

def run(node: InferenceNode, args):
    # ── MuJoCo live viewer (optional) ──
    args._viewer_proc = None
    if getattr(args, "viewer", False):
        viewer_script = os.path.join(os.path.dirname(__file__),
                                     "mujoco_live_viewer.py")
        viewer_python = os.path.expanduser(
            "~/miniconda3/envs/mujoco_viewer/bin/python")
        if os.path.exists(viewer_script) and os.path.exists(viewer_python):
            print("[Viewer] Launching MuJoCo live viewer (--with-objects)...")
            env = os.environ.copy()
            env["DISPLAY"] = ":1"
            env["MUJOCO_GL"] = "glx"
            # Ensure ROS2 libs are visible
            ros_lib = "/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu"
            ros_py = "/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages"
            env["LD_LIBRARY_PATH"] = ros_lib + ":" + env.get("LD_LIBRARY_PATH", "")
            env["PYTHONPATH"] = ros_py + ":" + env.get("PYTHONPATH", "")
            args._viewer_proc = subprocess.Popen(
                [viewer_python, viewer_script, "--with-objects", "--fps", "30"],
                env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print(f"[Viewer] PID={args._viewer_proc.pid}")
        else:
            print(f"[Viewer] WARNING: viewer script or mujoco_viewer env not found, skipping")

    # ── MuJoCo preview mode (interactive prompt) ──
    if not args.preview:
        try:
            ans = input("[Preview] Enable MuJoCo preview mode? (y/N): ").strip().lower()
            if ans == 'y':
                args.preview = True
                print("[Preview] ENABLED — will show planned trajectory before each execution.")
            else:
                print("[Preview] Disabled.")
        except (EOFError, KeyboardInterrupt):
            print("\n[Preview] Disabled.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safety = SafetyMonitor()

    # Safety parameters from args
    ws_min = np.array(args.workspace_min)
    ws_max = np.array(args.workspace_max)
    max_step_pos = args.max_step_pos
    max_step_rot = args.max_step_rot
    sensor_timeout = args.sensor_timeout
    ori_div_deg = args.ori_divergence_deg
    max_speed = args.max_speed

    # ── Task & checkpoint selection ──
    if not args.task:
        task_key, task_desc, task_results_subdir = select_task()
    else:
        task_key = args.task
        t = TASKS[task_key]
        task_desc, task_results_subdir = t["desc"], t["results_subdir"]
    if not args.desc:
        args.desc = task_desc

    # Set per-task gripper threshold
    global GRIPPER_CLOSE_THRESHOLD
    GRIPPER_CLOSE_THRESHOLD = TASKS[task_key].get("gripper_close_threshold", 0.75)
    args.max_steps = TASKS[task_key].get("max_steps", args.max_steps)
    use_gripper_latch = TASKS[task_key].get("gripper_latch", True)
    gripper_raw = TASKS[task_key].get("gripper_raw", False)
    args.gripper_settle = TASKS[task_key].get("gripper_settle", args.gripper_settle)
    node._gripper_raw = gripper_raw
    print(f"  Gripper close threshold: {GRIPPER_CLOSE_THRESHOLD}")
    print(f"  Gripper latch: {use_gripper_latch}")
    print(f"  Gripper raw input: {gripper_raw}")
    print(f"  Max steps: {args.max_steps}")
    args.out_dir = os.path.join(
        os.path.expanduser("~/kinam_dev/RESULTS"), task_results_subdir)

    if not args.groot_checkpoint:
        groot_path, residual_path, action_scale, epoch_tag = \
            discover_and_select_checkpoints(args.checkpoint_dir,
                                            task_key=task_key)
        args.groot_checkpoint = groot_path
        if residual_path:
            args.residual_checkpoint = residual_path
            args.residual_action_scale = action_scale
    else:
        ckpt_lower = args.groot_checkpoint.lower()
        if "100ep" in ckpt_lower:
            epoch_tag = "100ep"
        elif "66ep" in ckpt_lower:
            epoch_tag = "66ep"
        elif "32ep" in ckpt_lower or "checkpoint-90000" in ckpt_lower:
            epoch_tag = "32ep"
        else:
            epoch_tag = "unknown"

    mode_tag = "residual" if args.residual_checkpoint else "base"

    results_base = os.path.join(args.out_dir, epoch_tag, mode_tag)
    for diff in ("easy", "normal", "hard"):
        for outcome in ("success", "fail"):
            os.makedirs(os.path.join(results_base, diff, outcome), exist_ok=True)
    os.makedirs(os.path.join(results_base, "extra"), exist_ok=True)

    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(results_base, f"_staging_{session_tag}")
    os.makedirs(session_dir, exist_ok=True)
    trial_num = 0

    T_base_cam = load_calib(args.calib).astype(np.float64)

    # Load GR00T model
    print(f"Loading GR00T model from {args.groot_checkpoint}...", flush=True)
    inferencer = Gr00tInference(
        model_path=args.groot_checkpoint,
        task_description=args.desc,
        action_horizon=args.action_horizon,
    )
    horizon = args.action_horizon
    print(f"GR00T loaded. Action horizon={horizon}", flush=True)

    # ── Compare mode: load second GR00T model ──
    compare_inferencer = None
    if getattr(args, 'compare_ckpt', None):
        print(f"[Compare] Loading comparison model: {args.compare_ckpt}", flush=True)
        compare_inferencer = Gr00tInference(
            model_path=args.compare_ckpt,
            task_description=args.desc,
            action_horizon=args.action_horizon,
        )
        print(f"[Compare] Loaded. Red=primary, Blue=compare in viewer.", flush=True)

    # ── Startup checks: wait for ALL required sensors ──
    print("Waiting for ROS2 sensor data...", flush=True)
    required_sensors = ("base_rgb", "wrist_rgb", "ee_pose")
    t_wait_start = time.time()
    while not node.sensors_alive(required_sensors):
        rclpy.spin_once(node, timeout_sec=0.1)
        elapsed = time.time() - t_wait_start
        if elapsed > 30.0:
            missing = [s for s in required_sensors
                       if node._sensor_ts.get(s, 0.0) == 0.0]
            print(f"  WARNING: Still waiting for sensors: {missing} "
                  f"({elapsed:.0f}s)", flush=True)
            t_wait_start = time.time()
    # Also wait for images to be non-None
    state = node.get_state()
    while (state.get("ee_pos") is None or state.get("ee_quat") is None
           or state.get("images", {}).get("base_rgb") is None
           or state.get("images", {}).get("wrist_rgb") is None):
        rclpy.spin_once(node, timeout_sec=0.1)
        state = node.get_state()
    # ── Load residual actor ──
    residual_actor = None
    residual_device = "cuda" if torch.cuda.is_available() else "cpu"
    deploy_action_scaler = None   # v2: ActionScaler instance
    ckpt_use_action_scaler = False
    if args.residual_checkpoint:
        residual_actor, ckpt_use_action_scaler = ResidualActor.load_from_checkpoint(
            args.residual_checkpoint, device=residual_device)
        cube_pos_init = np.array(args.cube_pos, dtype=np.float32)
        cube_quat_wxyz_init = np.array(args.cube_quat_wxyz, dtype=np.float32)
        bowl_pos_init = np.array(getattr(args, 'bowl_pos', [0.42, 0.03, 0.0]), dtype=np.float32)
        use_fp = getattr(args, "use_fp_tracking", False)
        fp_max_age = getattr(args, "fp_max_age", 2.0)

        # v2: Determine ActionScaler mode
        use_action_scaler = args.use_action_scaler or ckpt_use_action_scaler
        if use_action_scaler:
            stats_path = args.action_scaler_stats
            if stats_path is None:
                # Auto-search: same dir as script, then .copilot/deploy/
                _candidates = [
                    os.path.join(os.path.dirname(__file__), "action_scaler_stats.json"),
                    os.path.join(os.path.dirname(__file__), "..", "action_scaler_stats.json"),
                ]
                for _c in _candidates:
                    if os.path.exists(_c):
                        stats_path = _c
                        break
            if stats_path and os.path.exists(stats_path):
                _as_scale = args.residual_action_scale  # action_scale used to expand range
                deploy_action_scaler = DeployActionScaler.from_json(
                    stats_path, task_key, action_scale=_as_scale,
                    no_clamp=args.action_scaler_no_clamp)
                print(f"  [ActionScaler] Loaded from {stats_path} (task={task_key}, "
                      f"scale={_as_scale}, no_clamp={args.action_scaler_no_clamp})")
            else:
                print(f"  [WARNING] ActionScaler enabled but no stats file found! Falling back to legacy mode.")
                use_action_scaler = False

        # Apply task config defaults for residual scales
        task_cfg = TASKS[task_key]
        if args.residual_pos_scale == 0.02 and "residual_pos_scale" in task_cfg:
            args.residual_pos_scale = task_cfg["residual_pos_scale"]
        if args.residual_rot_scale == 0.05 and "residual_rot_scale" in task_cfg:
            args.residual_rot_scale = task_cfg["residual_rot_scale"]
        if args.residual_grip_scale == 0.1 and "residual_grip_scale" in task_cfg:
            args.residual_grip_scale = task_cfg["residual_grip_scale"]

        print(f"  Residual RL enabled: action_scale={args.residual_action_scale}, "
              f"pos_scale={args.residual_pos_scale}, "
              f"rot_scale={args.residual_rot_scale}, grip_scale={args.residual_grip_scale}")
        print(f"  ActionScaler mode: {'ON' if use_action_scaler else 'OFF (legacy)'}")
        if use_fp:
            print(f"  Object state: LIVE FP tracking (max_age={fp_max_age}s, "
                  f"fallback={cube_pos_init.tolist()})")
        else:
            print(f"  Object state (fixed): pos={cube_pos_init.tolist()}, "
                  f"quat_wxyz={cube_quat_wxyz_init.tolist()}")
    else:
        cube_pos_init = np.zeros(3, dtype=np.float32)
        cube_quat_wxyz_init = np.array([1,0,0,0], dtype=np.float32)
        bowl_pos_init = np.array(getattr(args, 'bowl_pos', [0.42, 0.03, 0.0]), dtype=np.float32)
        use_action_scaler = False
        print("  Residual RL: DISABLED")

    print("All sensors active:", flush=True)
    for s in required_sensors:
        print(f"  {s}: OK", flush=True)

    # Per-task init pose override
    if "init_pos" in TASKS[task_key]:
        INIT_EE_POS[:] = TASKS[task_key]["init_pos"]
    if "init_quat" in TASKS[task_key]:
        INIT_EE_QUAT[:] = TASKS[task_key]["init_quat"]
    init_quat = INIT_EE_QUAT.copy()
    print(f"  Init pos:  {INIT_EE_POS.tolist()}")
    print(f"  Init quat: {INIT_EE_QUAT.tolist()}")

    # Per-task safety overrides
    if "ori_divergence_deg" in TASKS[task_key]:
        ori_div_deg = TASKS[task_key]["ori_divergence_deg"]
        print(f"  [Task override] ori_divergence_deg = {ori_div_deg}")
    if "residual_grip_scale" in TASKS[task_key]:
        args.residual_grip_scale = TASKS[task_key]["residual_grip_scale"]
        print(f"  [Task override] residual_grip_scale = {args.residual_grip_scale}")

    print(f"\nSafety config:", flush=True)
    print(f"  Workspace: [{ws_min.tolist()}] ~ [{ws_max.tolist()}]", flush=True)
    print(f"  Max step: pos={max_step_pos}m, rot={np.degrees(max_step_rot):.1f}deg",
          flush=True)
    print(f"  Sensor timeout: {sensor_timeout}s", flush=True)
    print(f"  Orientation divergence limit: {ori_div_deg}deg", flush=True)
    print(f"  Max speed: {max_speed}m/s\n", flush=True)

    # Vision recorder + rosbag recorder
    recorder = VisionRecorder(
        resolution=tuple(args.record_res), fps=args.record_fps)
    rosbag_rec = RosbagRecorder()
    node.rosbag_recorder = rosbag_rec
    _orig_cb = node._cb_image

    _IMAGE_KEY_TO_BAG = {"base_rgb": "cam_base", "wrist_rgb": "cam_wrist"}

    def _cb_image_with_record(msg, key):
        _orig_cb(msg, key)
        try:
            rgb = msg_to_numpy(msg)
        except Exception:
            return
        if key == "base_rgb":
            recorder.feed_frame(rgb)
        bag_key = _IMAGE_KEY_TO_BAG.get(key)
        if bag_key and rosbag_rec._recording:
            ts = node.get_clock().now().nanoseconds
            rosbag_rec.push_image(bag_key, rgb.copy(), ts)

    node._cb_image = _cb_image_with_record
    node.image_subs.clear()
    for key, topic in IMAGE_TOPICS.items():
        sub = node.create_subscription(
            Image, topic,
            lambda msg, k=key: node._cb_image(msg, k), 10)
        node.image_subs.append(sub)

    vis_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vis")
    debug_log = DebugLogger(save_images_every=3)

    # ── Keyboard + execution state ──
    keys = KeyListener()
    running = False
    executing = False
    estopped = False
    use_residual_this_trial = False
    trial_results_base = None
    step_count = 0
    ee_history = []
    gripper_latched = False
    gripper_open_streak = 0
    gripper_latch_time = 0.0  # time.time() when latch engaged
    prev_grip = None  # persist across chunks to prevent grasp re-send/cancel
    trial_dir = None

    # Cup gripper latch state (for RL combine, separate from base policy latch)
    _cup_latch_cfg = task_cfg.get("cup_gripper_latch", False)
    cup_latch_state = None
    if _cup_latch_cfg:
        cup_latch_state = {
            "latched": False,
            "open_count": 0,
            "close_thresh": task_cfg.get("cup_latch_close_thresh", 0.015),
            "open_thresh": task_cfg.get("cup_latch_open_thresh", 0.035),
            "open_steps": task_cfg.get("cup_latch_open_steps", 32),
        }

    # Move to init pose (slow)
    print(f"Moving to init pose... gripper={'CLOSED' if TASKS[task_key].get('gripper_always_closed', False) else 'OPEN'}", flush=True)

    def slow_move_to(target_pos, target_quat, n_waypoints=50,
                     interval=0.08, open_gripper=True):
        """Slowly interpolate from current pose to target. Press [e] to halt."""
        from scipy.spatial.transform import Rotation, Slerp
        nonlocal estopped

        rclpy.spin_once(node, timeout_sec=0.05)
        cur = node.get_state()
        if cur.get("ee_pos") is None:
            node.publish_target_pose(target_pos, target_quat)
            return
        start_pos = np.array(cur["ee_pos"])
        start_quat = np.array(cur["ee_quat"])

        if open_gripper:
            node.publish_gripper(1.0)

        rots = Rotation.from_quat([start_quat, target_quat])
        slerp = Slerp([0.0, 1.0], rots)

        print(f"  Slow move: {n_waypoints} waypoints, "
              f"{interval}s each, [e]=halt", flush=True)

        for i in range(1, n_waypoints + 1):
            t = i / n_waypoints
            wp_pos = start_pos + t * (target_pos - start_pos)
            wp_quat = slerp([t]).as_quat()[0]  # xyzw

            node.publish_target_pose(wp_pos, wp_quat)
            time.sleep(interval)

            # Check for e-stop during move
            rclpy.spin_once(node, timeout_sec=0.001)
            k = keys.get_key()
            if k in ('e', ' '):
                node.send_stop()
                estopped = True
                safety.record("estop", "User halted during slow move")
                print("  HALTED mid-move.", flush=True)
                return

        # Final convergence
        node.wait_for_convergence(
            target_pos, target_quat, timeout_sec=5.0)
        print("  Arrived.", flush=True)

    def emergency_stop(reason=""):
        nonlocal running, executing, estopped
        safety.record("estop", reason)
        print(f"\n!!! EMERGENCY STOP: {reason} !!!", flush=True)
        node.send_stop()
        time.sleep(0.3)
        if running:
            running = False
            finalize_trial()
            classify_trial()
        executing = False
        estopped = True
        print("E-STOP: Stopped. Press [r] to return to init, [a] to re-arm.\n", flush=True)

    def return_to_init():
        nonlocal executing, prev_grip
        gripper_always_closed = TASKS[task_key].get("gripper_always_closed", False)
        if gripper_always_closed:
            print("\n>>> Gripper stays closed (close_drawer mode)...", flush=True)
            node.send_stop()
            time.sleep(0.2)
            # Use Move (not Grasp) to keep fingers closed
            from franka_msgs.action import Move as FrankaMove
            goal = FrankaMove.Goal()
            goal.width = 0.0
            goal.speed = 0.1
            node._gripper_move_client.send_goal_async(goal)
            prev_grip = 0
        else:
            print("\n>>> Opening gripper...", flush=True)
            node.send_stop()
            time.sleep(0.2)
            node.publish_gripper(1.0)
            prev_grip = 1  # reset: gripper is now open
        time.sleep(1.0)
        print(">>> Returning to init pose...", flush=True)
        slow_move_to(INIT_EE_POS, init_quat, open_gripper=(not gripper_always_closed))
        executing = False

    def execute_async(waypoints, grip_cmds, n_act, interval):
        nonlocal step_count, executing, prev_grip
        try:
            for i, wp in enumerate(waypoints):
                if estopped:
                    break
                wp = np.asarray(wp)
                pos, quat = wp[:3], wp[3:7]

                # Gripper -- only publish on state change
                gripper_always_closed = TASKS[task_key].get("gripper_always_closed", False)
                if gripper_always_closed:
                    pass  # never change gripper
                elif grip_cmds is not None and i < len(grip_cmds):
                    grip_state = 1 if grip_cmds[i] > 0.5 else 0
                    if prev_grip is None or grip_state != prev_grip:
                        node.publish_gripper(1.0 if grip_state else 0.0)
                        node.get_logger().info(
                            f"Gripper: {'OPEN' if grip_state else 'CLOSED'}")
                        if prev_grip is not None:
                            time.sleep(1.0)
                        prev_grip = grip_state

                node.publish_target_pose(pos, quat)
                if interval > 0 and i < len(waypoints) - 1:
                    time.sleep(interval)

            # If chunk ended with gripper CLOSE, wait for it to physically settle
            if grip_cmds and grip_cmds[-1] < 0.5:
                settle = args.gripper_settle
                if settle > 0:
                    node.get_logger().info(
                        f"Gripper settle: waiting {settle:.1f}s for close")
                    time.sleep(settle)

            step_count += n_act
        except Exception as e:
            print(f"Trajectory error: {e}", flush=True)
        finally:
            executing = False

    def finalize_trial():
        vis_pool.submit(lambda: None).result()
        recorder.stop()
        rosbag_rec.stop()
        debug_log.stop()
        if trial_dir is None or len(ee_history) == 0:
            return

        if len(ee_history) > 1:
            ee = np.array(ee_history)
            fig, axes = plt.subplots(1, 3, figsize=(12, 3))
            for d, (ax, lbl) in enumerate(zip(axes, ["X", "Y", "Z"])):
                ax.plot(ee[:, d], "r-", linewidth=1.5)
                ax.set_title(f"EE {lbl}")
                ax.set_xlabel("step"); ax.set_ylabel("m")
                ax.grid(True, alpha=0.3)
            plt.suptitle("EE Trajectory")
            plt.tight_layout()
            plot_path = os.path.join(trial_dir, "ee_trajectory.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Plot: {plot_path}", flush=True)

    def classify_trial():
        """Prompt user to classify trial result and difficulty, then move it."""
        nonlocal trial_dir
        if trial_dir is None:
            return
        print("")
        print("  +======================================+")
        print("  |  Classify this trial:                |")
        print("  |    [s] = success                     |")
        print("  |    [f] = fail                        |")
        print("  |    [x] = extra                       |")
        print("  |    [d] = discard (delete)            |")
        print("  +======================================+")
        print(f"  Trial: {trial_dir}", flush=True)

        classification = None
        while classification is None:
            rclpy.spin_once(node, timeout_sec=0.01)
            k = keys.get_key()
            if k == "s":
                classification = "success"
            elif k == "f":
                classification = "fail"
            elif k == "x":
                classification = "extra"
            elif k == "d":
                import shutil
                shutil.rmtree(trial_dir, ignore_errors=True)
                print("  -> DISCARDED (deleted)", flush=True)
                trial_dir = None
                return

        # Ask difficulty for success/fail
        difficulty = None
        if classification in ("success", "fail"):
            print("  +--------------------------------------+")
            print("  |  Difficulty:                         |")
            print("  |    [e] = easy                        |")
            print("  |    [n] = normal                      |")
            print("  |    [h] = hard                        |")
            print("  +--------------------------------------+", flush=True)
            while difficulty is None:
                rclpy.spin_once(node, timeout_sec=0.01)
                k = keys.get_key()
                if k == "e":
                    difficulty = "easy"
                elif k == "n":
                    difficulty = "normal"
                elif k == "h":
                    difficulty = "hard"

        # Build destination path: results_base / difficulty / classification /
        _rb = trial_results_base if trial_results_base else results_base
        if difficulty:
            dst_base = os.path.join(_rb, difficulty, classification)
        else:
            dst_base = os.path.join(_rb, classification)
        os.makedirs(dst_base, exist_ok=True)
        trial_name = f"trial_{session_tag}_{trial_num}"
        dst = os.path.join(dst_base, trial_name)
        os.rename(trial_dir, dst)
        label = f"{difficulty}/{classification}" if difficulty else classification
        print(f"  -> {label.upper()}: {dst}", flush=True)
        trial_dir = None

    # ── Initial slow move to init pose ──
    _gripper_always_closed = TASKS[task_key].get("gripper_always_closed", False)
    if _gripper_always_closed:
        # Use Move (not Grasp) to close fingers — Grasp fails without an object
        from franka_msgs.action import Move as FrankaMove
        goal = FrankaMove.Goal()
        goal.width = 0.0
        goal.speed = 0.1
        node._gripper_move_client.send_goal_async(goal)
        time.sleep(1.0)

    # Preview: show init trajectory in MuJoCo before moving
    if args.preview:
        # Flush any leftover keys AND ignore keys for 0.5s
        _flush_deadline = time.time() + 0.5
        while time.time() < _flush_deadline:
            keys.get_key()
            time.sleep(0.02)

        rclpy.spin_once(node, timeout_sec=0.1)
        cur = node.get_state()
        if cur.get("ee_pos") is not None:
            cur_pos = np.array(cur["ee_pos"])
            # Generate interpolated path to init pose for preview
            n_preview = 10
            preview_pts = np.array([
                cur_pos + (INIT_EE_POS - cur_pos) * (i / n_preview)
                for i in range(1, n_preview + 1)
            ])
            write_traj_shm(cur_pos, preview_pts)
            time.sleep(0.3)  # Give viewer time to render trajectory
            print(f"  [Preview] Showing init move trajectory in MuJoCo. "
                  f"Press Enter/[a] to execute, [q] to abort.", flush=True)
            _preview_ok = False
            while not _preview_ok:
                rclpy.spin_once(node, timeout_sec=0.01)
                _pk = keys.get_key()
                if _pk is None:
                    time.sleep(0.02)
                    continue
                if _pk in ('\r', '\n', 'a'):
                    _preview_ok = True
                elif _pk == 'q':
                    print("  [Preview] Aborted by user.", flush=True)
                    return
                elif _pk in ('e', ' '):
                    print("  [Preview] E-STOP.", flush=True)
                    return

    slow_move_to(INIT_EE_POS, init_quat, open_gripper=(not _gripper_always_closed))
    time.sleep(0.5)

    # ── Main loop ──
    mode_str = "STEP-BY-STEP" if args.step_by_step else "CONTINUOUS"
    print(f"Mode: {mode_str}", flush=True)
    print("Press [a] to start, [r] to return to init, "
          "[e]/[space] to E-STOP, [Ctrl+C] to quit.", flush=True)
    if args.step_by_step:
        print("Step-by-step: [n]=execute waypoint, [s]=skip, [e]=E-STOP\n",
              flush=True)
    else:
        print("", flush=True)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            # ── Keyboard handling ──
            key = keys.get_key()
            if key in ('e', ' '):
                emergency_stop("User pressed E-STOP key")
                continue
            if key == 'r':
                if running:
                    running = False
                    finalize_trial()
                    classify_trial()
                return_to_init()
                if not estopped:  # only clear if move completed (not halted mid-move)
                    estopped = False
                print("Press [a] to restart...", flush=True)
                continue
            if key == 'g' and running:
                # Open gripper, stop trial, classify
                print("  [g] Opening gripper and ending trial...", flush=True)
                node.send_stop()
                node.publish_gripper(1.0)  # fully open
                time.sleep(1.0)
                running = False
                finalize_trial()
                classify_trial()
                print("Press [r] to return to init, [a] to start new trial...",
                      flush=True)
                continue
            if key == 'a' and not running:
                # Startup check: verify sensors still alive
                if not node.sensors_alive(required_sensors):
                    missing = [s for s in required_sensors
                               if node._sensor_ts.get(s, 0.0) == 0.0]
                    print(f"Cannot start: sensors missing: {missing}",
                          flush=True)
                    continue
                if trial_dir and len(ee_history) > 0:
                    finalize_trial()
                    classify_trial()
                # Ask mode if residual checkpoint is loaded
                if residual_actor is not None:
                    print("  +--------------------------------------+")
                    print("  |  Select mode:                        |")
                    print("  |    [b] = base policy only            |")
                    print("  |    [r] = residual RL                 |")
                    print("  +--------------------------------------+", flush=True)
                    mode_choice = None
                    while mode_choice is None:
                        rclpy.spin_once(node, timeout_sec=0.01)
                        mk = keys.get_key()
                        if mk == "b":
                            use_residual_this_trial = False
                            mode_choice = "base"
                        elif mk == "r":
                            use_residual_this_trial = True
                            mode_choice = "residual"
                    cur_mode_tag = mode_choice
                else:
                    use_residual_this_trial = False
                    cur_mode_tag = "base"

                trial_num += 1
                trial_results_base = os.path.join(args.out_dir, epoch_tag, cur_mode_tag)
                trial_dir = os.path.join(trial_results_base, f"_staging_{session_tag}", f"trial_{trial_num}")
                os.makedirs(trial_dir, exist_ok=True)
                mode_label = "RESIDUAL RL" if use_residual_this_trial else "BASE ONLY"
                running = True
                estopped = False
                ee_history.clear()
                step_count = 0
                gripper_latched = False
                gripper_open_streak = 0
                gripper_latch_time = 0.0
                # Reset cup latch state for new trial
                if cup_latch_state is not None:
                    cup_latch_state["latched"] = False
                    cup_latch_state["open_count"] = 0
                recorder.start(os.path.join(trial_dir, "vision.mp4"))
                rosbag_rec.start(trial_dir)
                debug_log.start(trial_dir)
                # Close gripper at trial start if gripper_always_closed
                if TASKS[task_key].get("gripper_always_closed", False):
                    from franka_msgs.action import Move as FrankaMove
                    goal = FrankaMove.Goal()
                    goal.width = 0.0
                    goal.speed = 0.1
                    node._gripper_move_client.send_goal_async(goal)
                    prev_grip = 0
                    time.sleep(0.5)
                print(f">>> Trial {trial_num} [{mode_label}] started! -> {trial_dir}",
                      flush=True)
            if not running or executing:
                continue
            if step_count >= args.max_steps:
                print(f"Max steps ({args.max_steps}) reached.", flush=True)
                running = False
                finalize_trial()
                classify_trial()
                node.send_stop()
                print("Press [r] to return to init, [a] to start new trial...", flush=True)
                continue

            # ── Sensor staleness check ──
            stale = node.check_staleness(sensor_timeout)
            if stale:
                safety.record("sensor_stale_stop", f"Stale: {stale}")
                emergency_stop(f"Sensor stale: {stale}")
                continue

            # ── Get state ──
            t_loop = time.time()

            t_state = time.time()
            state = node.get_state()
            if state is None:
                continue
            if (state.get("ee_pos") is None
                    or state.get("ee_quat") is None
                    or state.get("images", {}).get("base_rgb") is None
                    or state.get("images", {}).get("wrist_rgb") is None):
                continue
            dt_state = time.time() - t_state

            real_ee_pos = np.array(state["ee_pos"])
            real_ee_quat = np.array(state["ee_quat"])
            gripper_width = state["gripper_width"]
            ee_history.append(real_ee_pos.copy())

            # Prepare images for GR00T (expects RGB)
            cam_base = state["images"]["base_rgb"]
            cam_wrist = state["images"]["wrist_rgb"]
            cam_base_resized = Gr00tInference.resize_image(cam_base)
            cam_wrist_resized = Gr00tInference.resize_image(cam_wrist)

            # ── GR00T inference ──
            t_infer = time.time()
            result = inferencer.predict(
                cam_base=cam_base_resized,
                cam_wrist=cam_wrist_resized,
                eef_pos=real_ee_pos.astype(np.float32),
                eef_quat=real_ee_quat.astype(np.float32),
                gripper_width=gripper_width,
            )
            dt_infer = time.time() - t_infer

            # Parse GR00T output: absolute actions
            pred_pos = np.array(result["eef_pos"])       # (H, 3)
            pred_quat = np.array(result["eef_quat"])     # (H, 4) xyzw
            pred_grip = np.array(result["gripper_width"]) # (H, 1) or (H,)
            if pred_grip.ndim == 2:
                pred_grip = pred_grip[:, 0]

            # Save base (pre-residual) actions for logging
            base_pred_pos = pred_pos.copy()
            base_pred_quat = pred_quat.copy()
            base_pred_grip = pred_grip.copy()
            step_residuals = []  # list of 7D residual vectors per waypoint

            # ── SAFETY: Action sanity check (NaN/Inf/outlier) ──
            all_sane = True
            for k in range(len(pred_pos)):
                if not check_action_sanity(pred_pos[k], pred_quat[k]):
                    all_sane = False
                    break
            if not all_sane:
                safety.record("nan_inf_skip",
                              f"Step {step_count}: invalid action detected")
                node.send_stop()
                continue

            # ── Apply residual RL correction ──
            if residual_actor is not None and use_residual_this_trial:
                with torch.no_grad():
                    for k in range(len(pred_pos)):
                        # v2: Task-aware state construction via build_rl_state
                        if use_fp:
                            fp_pos, fp_quat, fp_age = read_fp_pose(fp_max_age)
                            if fp_pos is not None:
                                _cube_pos = fp_pos
                                _cube_quat = fp_quat
                                if fp_age > _FP_STALE_WARN_S:
                                    print(f"  [FP] pose stale: {fp_age:.1f}s", flush=True)
                            else:
                                _cube_pos = cube_pos_init
                                _cube_quat = cube_quat_wxyz_init
                        else:
                            _cube_pos = cube_pos_init
                            _cube_quat = cube_quat_wxyz_init
                        _bowl_pos = bowl_pos_init if bowl_pos_init is not None else np.array([0.42, 0.03, 0.0], dtype=np.float32)

                        obj_data = {
                            "cube_pos": _cube_pos, "cube_quat_wxyz": _cube_quat,
                            "bowl_pos": _bowl_pos,
                            # Stack: cube_b defaults (override with FP if available)
                            "cube_b_pos": getattr(args, 'cube_b_pos', np.zeros(3, dtype=np.float32)),
                            "cube_b_quat_wxyz": getattr(args, 'cube_b_quat_wxyz', np.array([1,0,0,0], dtype=np.float32)),
                            # Cup: same as cube for now
                            "cup_pos": _cube_pos, "cup_quat_wxyz": _cube_quat,
                            "uprightness": 0.0,
                            # Drawer: same as cube for now
                            "drawer_pos": _cube_pos, "drawer_quat_wxyz": _cube_quat,
                        }

                        rl_state = build_rl_state(task_key, real_ee_pos, real_ee_quat,
                                                  gripper_width, obj_data, task_cfg)

                        # Base action 7D: pos(3) + euler(3) + grip(1)
                        ba_7d = build_base_action_7d(pred_pos[k], pred_quat[k], pred_grip[k])

                        # ── Actor input normalization (dual-mode) ──
                        _as_mode = task_cfg.get("as_mode", "script_only")
                        if _as_mode == "wrapper" and use_action_scaler and deploy_action_scaler is not None:
                            # Mode A: normalize pos+grip via ActionScaler, euler=0
                            ba_pg = np.array([ba_7d[0], ba_7d[1], ba_7d[2], ba_7d[6]],
                                             dtype=np.float32)
                            ba_pg_norm = deploy_action_scaler.scale(ba_pg)
                            ba_7d_input = ba_7d.copy()
                            ba_7d_input[:3] = ba_pg_norm[:3]
                            ba_7d_input[3:6] = 0.0   # euler zeroed in AS-wrapper mode
                            ba_7d_input[6] = ba_pg_norm[3]
                        else:
                            # Mode B: raw base_action (unified/cup wrapper)
                            ba_7d_input = ba_7d

                        s_t = torch.as_tensor(rl_state, device=residual_device).unsqueeze(0)
                        ba_t = torch.as_tensor(ba_7d_input, device=residual_device, dtype=torch.float32).unsqueeze(0)
                        residual_7d = residual_actor(s_t, ba_t).squeeze(0).cpu().numpy()

                        step_residuals.append(residual_7d.copy())

                        # ── Combine (dual-mode) ──
                        _grip_min = task_cfg.get("grip_min", 0.0)
                        _grip_max = task_cfg.get("grip_max", 1.0)
                        _grip_always_zero = task_cfg.get("grip_always_zero", False)
                        pred_pos[k], pred_quat[k], pred_grip[k] = combine_actions_v2(
                            pred_pos[k], pred_quat[k], pred_grip[k],
                            residual_7d,
                            action_scaler=deploy_action_scaler if use_action_scaler else None,
                            as_mode=_as_mode,
                            pos_scale=args.residual_pos_scale,
                            rot_scale=args.residual_rot_scale,
                            grip_scale=args.residual_grip_scale,
                            grip_min=_grip_min,
                            grip_max=_grip_max,
                            grip_always_zero=_grip_always_zero,
                            cup_latch_state=cup_latch_state,
                        )

            # ── Compare inference (if enabled) ──
            compare_pred_pos = None
            if compare_inferencer is not None:
                compare_result = compare_inferencer.predict(
                    cam_base=cam_base_resized.copy(),
                    cam_wrist=cam_wrist_resized.copy(),
                    eef_pos=real_ee_pos.astype(np.float32).copy(),
                    eef_quat=real_ee_quat.astype(np.float32).copy(),
                    gripper_width=gripper_width,
                )
                compare_pred_pos = np.array(compare_result["eef_pos"])

            # ── Write trajectories to shared memory for viewer ──
            write_traj_shm(real_ee_pos, pred_pos,
                           compare_pred_pos if compare_inferencer else None)

            # ── Compare mode: pause for user confirmation ──
            if compare_inferencer is not None:
                print(f"  [Compare] Step {step_count}: red=primary, blue=compare. "
                      f"Enter=execute, q=stop", flush=True)
                try:
                    import select as _sel
                    ready, _, _ = _sel.select([sys.stdin], [], [], 60.0)
                    if ready:
                        user_in = sys.stdin.readline().strip()
                        if user_in.lower() == 'q':
                            print("  [Compare] User stopped.", flush=True)
                            running = False
                            finalize_trial()
                            classify_trial()
                            node.send_stop()
                            clear_traj_shm()
                            continue
                    else:
                        print("  [Compare] Timeout, executing.", flush=True)
                except Exception:
                    input("  [Compare] Press Enter...")

            # ── Visualize (background thread) — disabled to save disk ──
            # Frames were saved to {trial_dir}/frames/ and compiled into rollout.mp4
            # Now only vision.mp4 (raw camera) is recorded.
            # intrinsics = state.get("camera_intrinsics", _DEFAULT_INTRINSICS)
            # vis_pool.submit(
            #     save_step_frame,
            #     step_count, cam_base.copy(), real_ee_pos.copy(),
            #     pred_pos.copy(), gripper_width,
            #     list(ee_history), T_base_cam, intrinsics,
            #     os.path.join(trial_dir, "frames"))

            # ── Build Cartesian waypoints with safety ──
            t_cart = time.time()
            n_exec = min(args.exec_steps, len(pred_pos))
            cart_waypoints = []
            grip_cmds = []
            abort_trial = False

            for k in range(n_exec):
                tgt_pos = pred_pos[k].copy()
                tgt_quat = pred_quat[k].copy()

                # SAFETY 1: Workspace clamp
                tgt_pos = clamp_workspace(tgt_pos, ws_min, ws_max, safety)

                # SAFETY 2: Max-step clamp (relative to current EE)
                ref_pos = real_ee_pos if k == 0 else np.array(cart_waypoints[-1][:3])
                ref_quat = real_ee_quat if k == 0 else np.array(cart_waypoints[-1][3:7])
                tgt_pos, tgt_quat = clamp_max_step(
                    tgt_pos, tgt_quat, ref_pos, ref_quat,
                    max_step_pos, max_step_rot, safety)

                # SAFETY 3: Orientation divergence check
                diverged, angle_deg = check_orientation_divergence(
                    tgt_quat, real_ee_quat, ori_div_deg)
                if diverged:
                    safety.record("ori_divergence_stop",
                                  f"Step {step_count} wp{k}: "
                                  f"{angle_deg:.1f}deg > {ori_div_deg}deg")
                    emergency_stop(f"Orientation divergence: {angle_deg:.1f}deg")
                    abort_trial = True
                    break

                # SAFETY 4: Speed limit
                ref_for_speed = real_ee_pos if k == 0 else np.array(cart_waypoints[-1][:3])
                dist_m = np.linalg.norm(tgt_pos - ref_for_speed)
                if args.waypoint_interval > 0:
                    implied_speed = dist_m / max(args.waypoint_interval, 1e-6)
                    if implied_speed > max_speed:
                        safety.record("speed_limit",
                                      f"wp{k}: {implied_speed:.2f}m/s > "
                                      f"{max_speed}m/s")
                        scale = max_speed / implied_speed
                        tgt_pos = ref_for_speed + (tgt_pos - ref_for_speed) * scale

                wp = np.concatenate([tgt_pos, tgt_quat])
                cart_waypoints.append(wp.tolist())

                # Gripper command
                model_wants_close = pred_grip[k] < GRIPPER_CLOSE_THRESHOLD
                if use_gripper_latch:
                    # Latch logic + time-based hold (cube_lift)
                    if model_wants_close:
                        gripper_open_streak = 0
                        if not gripper_latched:
                            gripper_latched = True
                            gripper_latch_time = time.time()
                            node.get_logger().info(
                                f"Gripper LATCH engaged at step {step_count} wp{k}")
                    else:
                        if gripper_latched:
                            gripper_open_streak += 1

                    if gripper_latched:
                        hold_elapsed = time.time() - gripper_latch_time
                        if hold_elapsed < GRIPPER_HOLD_SEC:
                            grip_cmds.append(0.0)
                            if gripper_open_streak > 0:
                                node.get_logger().info(
                                    f"Gripper HOLD: {hold_elapsed:.1f}s/{GRIPPER_HOLD_SEC}s "
                                    f"(ignoring {gripper_open_streak} OPEN preds)")
                                gripper_open_streak = 0
                        elif gripper_open_streak >= args.gripper_latch:
                            gripper_latched = False
                            gripper_open_streak = 0
                            gripper_latch_time = 0.0
                            grip_cmds.append(1.0)
                            node.get_logger().info(
                                f"Gripper LATCH released at step {step_count} wp{k} "
                                f"(after {args.gripper_latch} consecutive OPEN)")
                        else:
                            grip_cmds.append(0.0)
                    else:
                        grip_cmds.append(
                            1.0 if pred_grip[k] > GRIPPER_CLOSE_THRESHOLD else 0.0)
                else:
                    # Direct passthrough (pick_and_place)
                    grip_cmds.append(0.0 if model_wants_close else 1.0)

            if abort_trial:
                continue

            # ── Chunk-level grip consistency (only when latch is active) ──
            if use_gripper_latch and len(grip_cmds) > 0 and grip_cmds[0] < 0.5:
                n_open = sum(1 for g in grip_cmds if g > 0.5)
                if n_open > 0:
                    node.get_logger().info(
                        f"Grip chunk override: first=CLOSE, forcing {n_open} OPEN→CLOSE")
                    grip_cmds = [0.0] * len(grip_cmds)
                    if gripper_latched:
                        gripper_open_streak = 0

            dt_cart = time.time() - t_cart
            dt_total = time.time() - t_loop

            # ── Debug log ──
            debug_log.log_step(
                step_count,
                input_state={
                    "ee_pos": real_ee_pos,
                    "ee_quat": real_ee_quat,
                    "gripper_width": gripper_width,
                    "joint_pos": (state["joint_pos"]
                                  if state.get("joint_pos") is not None
                                  else None),
                },
                raw_output={
                    "pred_pos": pred_pos,
                    "pred_quat": pred_quat,
                    "pred_grip": pred_grip,
                },
                base_output={
                    "pred_pos": base_pred_pos,
                    "pred_quat": base_pred_quat,
                    "pred_grip": base_pred_grip,
                } if use_residual_this_trial else None,
                residuals=step_residuals if step_residuals else None,
                safe_output={
                    "waypoints": cart_waypoints,
                    "grip_cmds": grip_cmds,
                },
                inference_time=dt_infer,
                cam_base=cam_base,
                cam_wrist=cam_wrist,
            )

            # ── Timing summary ──
            latch_str = " [LATCHED]" if gripper_latched else ""
            n_close_pred = sum(1 for g in pred_grip[:n_exec] if g < GRIPPER_CLOSE_THRESHOLD)
            print(f"  Step {step_count:3d}  "
                  f"pos=[{pred_pos[0,0]:.3f},{pred_pos[0,1]:.3f},"
                  f"{pred_pos[0,2]:.3f}]  "
                  f"grip={pred_grip[0]:.3f} ({n_close_pred}C/{n_exec-n_close_pred}O)  "
                  f"grip_in={gripper_width:.3f}{latch_str}  |  "
                  f"infer={dt_infer:.3f}s  "
                  f"total={dt_total:.3f}s", flush=True)

            # ── MuJoCo preview: pause for user confirmation ──
            if args.preview and not args.step_by_step:
                # Flush leftover keys
                while keys.get_key() is not None:
                    pass
                # Write planned waypoints to SHM so viewer shows them
                preview_pos = np.array([wp[:3] for wp in cart_waypoints])
                write_traj_shm(real_ee_pos, preview_pos)
                time.sleep(0.2)  # Give viewer time to render
                print(f"  [Preview] Showing {len(cart_waypoints)} waypoints in MuJoCo. "
                      f"Press Enter/[a] to execute, [q] to stop trial.", flush=True)
                # Use KeyListener (stdin is in cbreak mode)
                preview_done = False
                while not preview_done:
                    rclpy.spin_once(node, timeout_sec=0.01)
                    pk = keys.get_key()
                    if pk is None:
                        time.sleep(0.02)
                        continue
                    if pk == 'q':
                        print("  [Preview] User stopped trial.", flush=True)
                        running = False
                        finalize_trial()
                        classify_trial()
                        node.send_stop()
                        clear_traj_shm()
                        preview_done = True
                    elif pk in ('\r', '\n', 'a'):
                        # Enter or 'a' → proceed with execution
                        preview_done = True
                    elif pk in ('e', ' '):
                        emergency_stop("User E-STOP during preview")
                        preview_done = True
                if not running:
                    continue

            # ── Execute ──
            if args.step_by_step:
                # Step-by-step: execute one waypoint at a time, wait for keypress
                for wi, wp in enumerate(cart_waypoints):
                    wp = np.asarray(wp)
                    pos_w, quat_w = wp[:3], wp[3:7]
                    grip_w = grip_cmds[wi] if wi < len(grip_cmds) else 1.0

                    # Compute delta from CURRENT ee pose (refreshed)
                    rclpy.spin_once(node, timeout_sec=0.01)
                    cur_state = node.get_state()
                    if cur_state.get("ee_pos") is not None:
                        cur_pos = np.array(cur_state["ee_pos"])
                        cur_quat = np.array(cur_state["ee_quat"])
                    else:
                        cur_pos = real_ee_pos
                        cur_quat = real_ee_quat

                    delta_mm = np.linalg.norm(pos_w - cur_pos) * 1000
                    too_big = delta_mm > max_step_pos * 1000

                    status = "TOO BIG - auto-skip" if too_big else "[n]=exec [s]=skip [e]=estop"
                    print(f"    WP {wi}/{len(cart_waypoints)-1}  "
                          f"pos=[{pos_w[0]:.4f},{pos_w[1]:.4f},{pos_w[2]:.4f}]  "
                          f"grip={'OPEN' if grip_w > 0.5 else 'CLOSE'}  "
                          f"delta={delta_mm:.1f}mm  "
                          f"{status}", flush=True)

                    if too_big:
                        safety.record("step_clamp_pos",
                                      f"WP{wi} auto-skipped: {delta_mm:.1f}mm > "
                                      f"{max_step_pos*1000:.0f}mm")
                        continue

                    # Wait for keypress
                    confirm = None
                    while confirm is None:
                        rclpy.spin_once(node, timeout_sec=0.01)
                        confirm = keys.get_key()
                        if confirm == 'e' or confirm == ' ':
                            emergency_stop("User E-STOP during step-by-step")
                            break
                        if confirm == 's':
                            print("    -> skipped", flush=True)
                            break
                        if confirm == 'n':
                            # Execute this single waypoint
                            node.publish_gripper(1.0 if grip_w > 0.5 else 0.0)
                            node.publish_target_pose(pos_w, quat_w)
                            time.sleep(args.waypoint_interval)
                            print("    -> executed", flush=True)
                            break
                        if confirm == 'r':
                            if running:
                                running = False
                                finalize_trial()
                            return_to_init()
                            estopped = False
                            break
                        confirm = None  # unrecognized key, keep waiting

                    if confirm in ('e', ' ', 'r') or estopped:
                        break

                step_count += len(cart_waypoints)
            else:
                # Normal mode: execute all waypoints non-blocking
                executing = True
                threading.Thread(
                    target=execute_async,
                    args=(cart_waypoints, grip_cmds, n_exec,
                          args.waypoint_interval),
                    daemon=True).start()

    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        if running:
            finalize_trial()
            classify_trial()

    safety.summary()
    # Clean up empty staging dir
    if os.path.isdir(session_dir) and not os.listdir(session_dir):
        os.rmdir(session_dir)
    print("Done.", flush=True)


# ── Entry point ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Real-robot GR00T VLA inference "
                    "(Cartesian impedance control)")
    ap.add_argument("--groot-checkpoint", default=None,
                    help="Path to GR00T checkpoint (if omitted, interactive selection)")
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_BASE_DIR,
                    help="Base dir for auto-discovery (default: %%(default)s)")
    ap.add_argument("--calib", default=DEFAULT_CALIB,
                    help="Path to camera calibration file (.calib)")
    ap.add_argument("--task", default=None, choices=list(TASKS.keys()),
                    help="Task key (if omitted, interactive selection)")
    ap.add_argument("--desc", default=None,
                    help="Override task description for GR00T")
    ap.add_argument("--action-horizon", type=int, default=16,
                    help="GR00T action horizon (default: 16)")
    ap.add_argument("--exec-steps", type=int, default=16,
                    help="Waypoints to execute per GR00T inference")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out-dir",
                    default=os.path.expanduser("~/kinam_dev/RESULTS"))
    ap.add_argument("--waypoint-interval", type=float, default=0.05,
                    help="Seconds between Cartesian waypoint publishes")
    ap.add_argument("--record-fps", type=int, default=15)
    ap.add_argument("--record-res", type=int, nargs=2, default=[640, 480],
                    metavar=("W", "H"))
    # Safety arguments
    safety_grp = ap.add_argument_group("Safety")
    safety_grp.add_argument(
        "--workspace-min", type=float, nargs=3,
        default=DEFAULT_WORKSPACE_MIN.tolist(),
        metavar=("X", "Y", "Z"),
        help="Workspace lower bounds (meters)")
    safety_grp.add_argument(
        "--workspace-max", type=float, nargs=3,
        default=DEFAULT_WORKSPACE_MAX.tolist(),
        metavar=("X", "Y", "Z"),
        help="Workspace upper bounds (meters)")
    safety_grp.add_argument(
        "--max-step-pos", type=float, default=DEFAULT_MAX_STEP_POS_M,
        help="Max position change per step (meters)")
    safety_grp.add_argument(
        "--max-step-rot", type=float, default=DEFAULT_MAX_STEP_ROT_RAD,
        help="Max rotation change per step (radians)")
    safety_grp.add_argument(
        "--sensor-timeout", type=float, default=DEFAULT_SENSOR_TIMEOUT_S,
        help="Sensor staleness threshold (seconds)")
    safety_grp.add_argument(
        "--ori-divergence-deg", type=float, default=DEFAULT_ORI_DIVERGENCE_DEG,
        help="Max orientation divergence from current (degrees)")
    safety_grp.add_argument(
        "--max-speed", type=float, default=DEFAULT_MAX_SPEED_M_S,
        help="Max implied EE speed (m/s)")
    safety_grp.add_argument(
        "--step-by-step", action="store_true",
        help="Execute one waypoint at a time, requiring [n] keypress for each")
    safety_grp.add_argument(
        "--gripper-latch", type=int, default=GRIPPER_LATCH_OPEN_COUNT,
        metavar="N",
        help="Consecutive OPEN predictions needed to unlatch gripper "
             "(0=disable latch, default: %(default)s)")
    safety_grp.add_argument(
        "--gripper-settle", type=float, default=GRIPPER_SETTLE_TIME,
        help="Seconds to wait after gripper close before next inference "
             "(default: %(default)s)")
    # Residual RL
    res_grp = ap.add_argument_group("Residual RL")
    res_grp.add_argument("--residual-checkpoint", type=str, default=None,
        help="Path to residual actor best.pt (None=disable)")
    res_grp.add_argument("--residual-pos-scale", type=float, default=0.02,
        help="Residual position scale (meters)")
    res_grp.add_argument("--residual-rot-scale", type=float, default=0.05,
        help="Residual rotation scale (radians)")
    res_grp.add_argument("--residual-grip-scale", type=float, default=0.1,
        help="Residual gripper scale")
    res_grp.add_argument("--residual-action-scale", type=float, default=0.1,
        help="Actor output scale (tanh * scale). "
             "s1L2 checkpoints: 1.0, s42/s100 checkpoints: 0.2")
    res_grp.add_argument("--use-action-scaler", action="store_true", default=False,
        help="Enable ActionScaler mode (normalize pos+grip to [-1,1], "
             "add residual in normalized space). Auto-detected from checkpoint.")
    res_grp.add_argument("--action-scaler-stats", type=str, default=None,
        help="Path to action_scaler_stats.json. If not given, searches "
             "in same dir as this script or .copilot/deploy/")
    res_grp.add_argument("--action-scaler-no-clamp", action="store_true", default=False,
        help="Disable clamping in ActionScaler (pass-through for base policy)")
    res_grp.add_argument("--cube-pos", type=float, nargs=3,
        default=[0.45, -0.05, 0.02],
        help="Cube initial position xyz for object_state")
    res_grp.add_argument("--cube-quat-wxyz", type=float, nargs=4,
        default=[1.0, 0.0, 0.0, 0.0],
        help="Cube initial quaternion wxyz for object_state")
    res_grp.add_argument("--bowl-pos", type=float, nargs=3,
        default=[0.42, 0.03, 0.0], help="Bowl xyz position")
    res_grp.add_argument("--no-fp-tracking", dest="use_fp_tracking",
        action="store_false", default=True,
        help="Disable live FoundationPose tracking (default: enabled)")
    res_grp.add_argument("--fp-max-age", type=float, default=2.0,
        help="Max age (seconds) for FP pose before falling back to fixed pose")

    vis_grp = ap.add_argument_group("Visualisation")
    vis_grp.add_argument("--no-viewer", dest="viewer", action="store_false", default=True,
        help="Disable MuJoCo live viewer")
    vis_grp.add_argument("--preview", action="store_true", default=False,
        help="MuJoCo preview mode: show planned trajectory before execution, wait for Enter")
    vis_grp.add_argument("--compare-ckpt", type=str, default=None,
        help="Second GR00T checkpoint for trajectory comparison")

    args = ap.parse_args()

    # ── Auto-save log to file ──
    import datetime as _dt
    _log_dir = os.path.expanduser("~/kinam_dev/logs")
    os.makedirs(_log_dir, exist_ok=True)
    _log_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = os.path.join(_log_dir, f"run_{_log_ts}.log")

    class _Tee:
        """Write to both terminal and log file."""
        def __init__(self, stream, log_file):
            self._stream = stream
            self._log = log_file
        def write(self, data):
            self._stream.write(data)
            self._stream.flush()
            try:
                self._log.write(data)
                self._log.flush()
            except Exception:
                pass
        def flush(self):
            self._stream.flush()
            try:
                self._log.flush()
            except Exception:
                pass
        def fileno(self):
            return self._stream.fileno()
        def isatty(self):
            return self._stream.isatty()

    _log_file = open(_log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)
    print(f"[Log] Saving to: {_log_path}", flush=True)

    rclpy.init()
    node = InferenceNode(args)
    try:
        run(node, args)
    finally:
        clear_traj_shm()
        if getattr(args, "_viewer_proc", None) is not None:
            print("[Viewer] Shutting down MuJoCo viewer...")
            args._viewer_proc.terminate()
            args._viewer_proc.kill(); args._viewer_proc.wait(timeout=3)
        node.destroy_node()
        rclpy.shutdown()
        print(f"\n[Log] Saved to: {_log_path}", flush=True)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log_file.close()


if __name__ == "__main__":
    main()
