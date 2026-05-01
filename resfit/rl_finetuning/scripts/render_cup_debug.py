#!/usr/bin/env python3
"""Render debug videos of GR00T stand-cup policy — base + wrist cameras, with cup state overlay."""
import os, sys, time
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

from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import (
    MuJoCoVecEnvCup, CUP_HOME_QPOS, CUP_HEIGHT, TCP_OFFSET,
    GRIPPER_MAX_WIDTH, get_tcp_pose,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_cup import MuJoCoResidualWrapperCup

# ── Config ──
CKPT = os.environ.get("CKPT", os.path.expanduser(
    "~/DATA/INTERN/training/groot_cup_sim_29ep/checkpoint-100000"))
EPISODES = [int(x) for x in os.environ.get("EPISODES", "0,1,2").split(",")]
MAX_STEPS = int(os.environ.get("MAX_STEPS", "500"))
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/cup_debug_videos")
FPS = 15

os.makedirs(OUT_DIR, exist_ok=True)
print(f"=== Stand Cup Debug Video Render ===")
print(f"  ckpt: {CKPT}")
print(f"  episodes: {EPISODES}")
print(f"  output: {OUT_DIR}")


def write_video(frames, path, fps=FPS):
    """Write frames (BGR) to mp4 with H.264 (avc1) via imageio-ffmpeg."""
    if not frames:
        return
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec='libx264',
                                 output_params=['-pix_fmt', 'yuv420p', '-crf', '23'])
    for frame in frames:
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


# ── Run episodes one by one ──
for ei in EPISODES:
    print(f"\n--- Episode {ei} ---")
    env = MuJoCoVecEnvCup(num_envs=1, episode_ids=[ei], max_episode_steps=MAX_STEPS)

    wrapper = MuJoCoResidualWrapperCup(
        vec_env=env, groot_checkpoint=CKPT, policy_device="cuda:0"
    )
    print(f"Wrapper ready for ep{ei}.")

    obs, _ = wrapper.reset()

    frames_combined = []
    t0 = time.time()

    for step in range(MAX_STEPS):
        e = env._envs[0]
        model, data, ids = e["model"], e["data"], e["ids"]

        # Render base + wrist
        e["renderer_base"].update_scene(data, camera=e["cam_base_id"],
                                         scene_option=e["opt_base"])
        img_base = e["renderer_base"].render().copy()

        e["renderer_wrist"].update_scene(data, camera=e["cam_wrist_id"],
                                          scene_option=e["opt_wrist"])
        img_wrist = e["renderer_wrist"].render().copy()

        # TCP info
        tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])

        # Cup state
        cup_state = env.get_cup_state(0)
        cup_pos = cup_state["pos"]
        uprightness = cup_state["uprightness"]
        cup_vel = cup_state["vel"]

        # Gripper width
        finger_id = ids["finger_ids"][0]
        grip_w = max(0.0, float(data.qpos[model.jnt_qposadr[finger_id]])) if finger_id >= 0 else 0.0

        # Convert to BGR
        base_bgr = cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

        h_target = 360
        w_base = int(base_bgr.shape[1] * h_target / base_bgr.shape[0])
        w_wrist = int(wrist_bgr.shape[1] * h_target / wrist_bgr.shape[0])
        base_r = cv2.resize(base_bgr, (w_base, h_target))
        wrist_r = cv2.resize(wrist_bgr, (w_wrist, h_target))

        overlay = [
            f"Step {step:03d}/{MAX_STEPS}  Episode {ei}",
            f"TCP: [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]",
            f"Grip: {grip_w:.4f} / {GRIPPER_MAX_WIDTH}",
            f"Cup: [{cup_pos[0]:.3f}, {cup_pos[1]:.3f}, {cup_pos[2]:.3f}]",
            f"Upright: {uprightness:.3f}  Vel: {cup_vel:.3f}",
            f"Success: {'YES' if env._is_success(0) else 'no'}",
        ]
        add_overlay(base_r, overlay)
        add_overlay(wrist_r, [f"WRIST Step {step:03d}"])

        combined = np.hstack([base_r, wrist_r])
        frames_combined.append(combined)

        # Step with zero residual (base policy only)
        residual = torch.zeros((1, 7), dtype=torch.float32)
        obs, rew, term, trunc, info = wrapper.step(residual)
        if term[0] or trunc[0]:
            break

    dt = time.time() - t0
    tag = "SUCC" if term[0].item() else "FAIL"
    final_cup = env.get_cup_state(0)
    print(f"  => {tag} steps={step+1} upright={final_cup['uprightness']:.3f} "
          f"cup_z={final_cup['pos'][2]:.4f} vel={final_cup['vel']:.3f} t={dt:.1f}s")

    vid_path = os.path.join(OUT_DIR, f"ep{ei}_cup.mp4")
    write_video(frames_combined, vid_path)

    wrapper.close()

# Copy to .VIEW/
VIEW_DIR = os.path.expanduser("~/Repos/Intern/.VIEW")
os.makedirs(VIEW_DIR, exist_ok=True)
import shutil
for ei in EPISODES:
    src = os.path.join(OUT_DIR, f"ep{ei}_cup.mp4")
    if os.path.exists(src):
        ckpt_name = os.path.basename(CKPT).replace("checkpoint-", "")
        dst = os.path.join(VIEW_DIR, f"cup_ep{ei}_{ckpt_name}.mp4")
        shutil.copy2(src, dst)
        print(f"Copied → {dst}")

print(f"\n=== Done. Videos in {OUT_DIR} and {VIEW_DIR} ===")
