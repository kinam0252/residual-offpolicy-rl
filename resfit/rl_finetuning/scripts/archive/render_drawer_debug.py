#!/usr/bin/env python3
"""Render debug videos of GR00T drawer policy — base + wrist cameras, with TCP overlay."""
import os, sys, json, time
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
import importlib, types as _types
if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed"); _ds.__version__="0.0.0"
    _ds.__spec__=importlib.machinery.ModuleSpec("deepspeed",None); _ds.__path__=[]; _ds.__file__=__file__
    _dz = _types.ModuleType("deepspeed.zero"); _dz.__spec__=importlib.machinery.ModuleSpec("deepspeed.zero",None)
    _dz.Init=lambda *a,**kw:(lambda f:f); _ds.zero=_dz
    sys.modules["deepspeed"]=_ds; sys.modules["deepspeed.zero"]=_dz
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except: pass

import cv2
import mujoco

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, DRAWER_SLIDE, FLOOR_TO_DRAWER, load_episode_mapping,
    WALL_THICK, DRAWER_SLOT_HEIGHT, TCP_OFFSET, get_tcp_pose,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer

# ── Config ──
CKPT = os.environ.get("CKPT", os.path.expanduser(
    "~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000"))
EPISODES = [int(x) for x in os.environ.get("EPISODES", "1,2,0").split(",")]
MAX_STEPS = int(os.environ.get("MAX_STEPS", "500"))
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/drawer_debug_videos")
FPS = 15

os.makedirs(OUT_DIR, exist_ok=True)
print(f"=== Drawer Debug Video Render ===")
print(f"  ckpt: {CKPT}")
print(f"  episodes: {EPISODES}")
print(f"  output: {OUT_DIR}")

# ── Setup ──
env = MuJoCoVecEnvDrawer(num_envs=1, max_episode_steps=MAX_STEPS)
eps = load_episode_mapping()

wrapper = MuJoCoResidualWrapperDrawer(
    vec_env=env, groot_checkpoint=CKPT, policy_device="cuda:0"
)
print("Wrapper ready.\n")


def write_video(frames, path, fps=FPS):
    """Write frames (BGR) to mp4 with H.264 (avc1) via imageio-ffmpeg."""
    if not frames:
        return
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec='libx264',
                                 output_params=['-pix_fmt', 'yuv420p', '-crf', '23'])
    for frame in frames:
        # imageio expects RGB
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"  Saved: {path} ({len(frames)} frames)")


def add_overlay(frame, text_lines, y_start=20):
    """Add text overlay on frame (BGR)."""
    for i, line in enumerate(text_lines):
        y = y_start + i * 18
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)  # shadow
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


# ── Run episodes ──
for ei in EPISODES:
    ep = eps[ei]
    floor = ep["floor"]
    active_drawer = FLOOR_TO_DRAWER[floor]

    env._active_drawers[0] = active_drawer
    env_data = env._envs[0]
    env_data["active_drawer"] = active_drawer
    slot_h = DRAWER_SLOT_HEIGHT
    dz = WALL_THICK + slot_h / 2 + active_drawer * (slot_h + WALL_THICK)
    env_data["drawer_z_center"] = dz
    env_data["drawer_z_min"] = dz - slot_h / 2 - 0.02
    env_data["drawer_z_max"] = dz + slot_h / 2 + 0.02

    obs, _ = wrapper.reset()

    frames_base = []
    frames_wrist = []
    frames_combined = []

    print(f"ep{ei} floor={floor} drawer={active_drawer} ...", flush=True)
    t0 = time.time()

    for step in range(MAX_STEPS):
        e = env._envs[0]
        model, data, ids = e["model"], e["data"], e["ids"]

        # Render base + wrist
        e["renderer_base"].update_scene(data, camera=e["cam_base_id"],
                                         scene_option=e["opt_base"])
        img_base = e["renderer_base"].render().copy()  # RGB

        e["renderer_wrist"].update_scene(data, camera=e["cam_wrist_id"],
                                          scene_option=e["opt_wrist"])
        img_wrist = e["renderer_wrist"].render().copy()  # RGB

        # TCP info
        hand_pos = data.xpos[ids["hand_id"]]
        hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
        tcp_w = hand_pos + hand_mat @ TCP_OFFSET
        cab_p = data.xpos[e["cab_body_id"]]
        cab_m = data.xmat[e["cab_body_id"]].reshape(3, 3)
        tcp_l = cab_m.T @ (tcp_w - cab_p)
        face_x_now = env._face_x_closed + env._drawer_qpos[0]
        slide = env._drawer_qpos[0]
        closed_pct = (1 - slide / DRAWER_SLIDE) * 100

        # Convert to BGR for cv2
        base_bgr = cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

        # Resize to same height
        h_target = 360
        w_base = int(base_bgr.shape[1] * h_target / base_bgr.shape[0])
        w_wrist = int(wrist_bgr.shape[1] * h_target / wrist_bgr.shape[0])
        base_r = cv2.resize(base_bgr, (w_base, h_target))
        wrist_r = cv2.resize(wrist_bgr, (w_wrist, h_target))

        # Overlay text
        overlay = [
            f"Step {step:03d}/{MAX_STEPS}  Drawer {active_drawer} (floor {floor})",
            f"Slide: {slide:.4f}/{DRAWER_SLIDE:.4f}  Closed: {closed_pct:.0f}%",
            f"TCP local: [{tcp_l[0]:.3f}, {tcp_l[1]:.3f}, {tcp_l[2]:.3f}]",
            f"Face x now: {face_x_now:.3f}  Dist: {tcp_l[0]-face_x_now:.3f}",
            f"TCP world: [{tcp_w[0]:.3f}, {tcp_w[1]:.3f}, {tcp_w[2]:.3f}]",
        ]
        add_overlay(base_r, overlay)
        add_overlay(wrist_r, [f"WRIST Step {step:03d}"])

        # Side by side
        combined = np.hstack([base_r, wrist_r])
        frames_combined.append(combined)

        # Step
        residual = torch.zeros((1, 7), dtype=torch.float32)
        obs, rew, term, trunc, info = wrapper.step(residual)
        if term[0] or trunc[0]:
            break

    dt = time.time() - t0
    success = env._drawer_qpos[0] < 0.01  # might be reset already, check term
    tag = "SUCC" if term[0].item() else "FAIL"
    print(f"  => {tag} steps={step+1} closed={closed_pct:.0f}% t={dt:.1f}s", flush=True)

    # Write combined video
    vid_path = os.path.join(OUT_DIR, f"ep{ei}_floor{floor}_drawer{active_drawer}.mp4")
    write_video(frames_combined, vid_path)

print(f"\n=== Done. Videos in {OUT_DIR} ===")
