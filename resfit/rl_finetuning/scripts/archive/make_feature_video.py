"""Generate feature-overlay video for Stack Cube episodes.

Renders cam_base every step, overlays reward features as text + bars,
and saves as MP4 video.

Usage (login node with GPU):
    cd ~/Repos/Intern/residual-offpolicy-rl
    LD_LIBRARY_PATH=~/.local/lib/gl python resfit/rl_finetuning/scripts/make_feature_video.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

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

import numpy as np
import mujoco
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation
import imageio_ffmpeg

from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import (
    MuJoCoVecEnvStack, CUBE_HALF, get_tcp_pose,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack

REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])

NUM_POSITIONS = 5
MAX_STEPS = 1000
FPS = 15
GROOT_CKPT = os.path.expanduser(
    "~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000"
)
POSITIONS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
    'configs', 'stack_cube_positions.json'
)
OUT_DIR = Path("outputs/test_feature_collection")


def extract_features(mujoco_env, env_idx=0):
    env = mujoco_env._envs[env_idx]
    model, data, ids = env["model"], env["data"], env["ids"]
    qa = env["white_qposadr"]
    white_pos = data.qpos[qa:qa + 3].copy() if qa else np.zeros(3)
    tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
    grasped = env["grasp_state"]["grasped"]
    green_body_id = env["green_body_id"]
    green_pos = model.body_pos[green_body_id].copy() if green_body_id >= 0 else np.zeros(3)
    green_top = green_pos.copy(); green_top[2] += CUBE_HALF * 2

    tcp_white_dist = float(np.linalg.norm(tcp_pos - white_pos))
    white_to_green_top = float(np.linalg.norm(white_pos - green_top))
    white_green_xy = float(np.linalg.norm(white_pos[:2] - green_pos[:2]))

    white_gid = env["white_geom_id"]
    green_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_cube_geom")
    contact = False
    for ci in range(data.ncon):
        c = data.contact[ci]
        if {c.geom1, c.geom2} == {white_gid, green_gid}:
            contact = True; break

    return {
        "tcp_white_dist": tcp_white_dist,
        "white_to_green_top": white_to_green_top,
        "white_green_xy": white_green_xy,
        "white_z": float(white_pos[2]),
        "green_z": float(green_pos[2]),
        "grasped": grasped,
        "contact": contact,
        "tcp_pos": tcp_pos.copy(),
        "white_pos": white_pos.copy(),
        "green_pos": green_pos.copy(),
    }


def render_cam_base(mujoco_env, env_idx=0):
    env = mujoco_env._envs[env_idx]
    env["renderer_base"].update_scene(
        env["data"], camera=env["cam_base_id"], scene_option=env["opt_base"]
    )
    return env["renderer_base"].render().copy()


def draw_overlay(frame, feat, step, max_steps):
    """Draw feature overlay on frame. Returns BGR frame for cv2."""
    h, w = frame.shape[:2]
    # Convert RGB → BGR for cv2
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Semi-transparent overlay panel
    overlay = img.copy()
    panel_h = 200
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Text config
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45  # font scale
    thick = 1
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)

    y0 = 18
    dy = 20

    # Step counter
    cv2.putText(img, f"Step {step}/{max_steps}", (10, y0), font, 0.55, white, thick)

    # Features
    row = y0 + dy
    cv2.putText(img, f"tcp->white:  {feat['tcp_white_dist']:.4f} m", (10, row), font, fs, cyan, thick)
    _bar(img, 250, row - 10, feat['tcp_white_dist'], 0.4, cyan)

    row += dy
    cv2.putText(img, f"white->green_top: {feat['white_to_green_top']:.4f} m", (10, row), font, fs, green, thick)
    _bar(img, 250, row - 10, feat['white_to_green_top'], 0.4, green)

    row += dy
    cv2.putText(img, f"white<>green_xy:  {feat['white_green_xy']:.4f} m", (10, row), font, fs, (180, 180, 255), thick)
    _bar(img, 250, row - 10, feat['white_green_xy'], 0.4, (180, 180, 255))

    row += dy
    cv2.putText(img, f"white_z: {feat['white_z']:.4f}  green_z: {feat['green_z']:.4f}", (10, row), font, fs, white, thick)

    row += dy
    grasp_color = green if feat['grasped'] else red
    grasp_text = "YES" if feat['grasped'] else "NO"
    cv2.putText(img, f"Grasped: {grasp_text}", (10, row), font, fs, grasp_color, thick)

    contact_color = green if feat['contact'] else red
    contact_text = "YES" if feat['contact'] else "NO"
    cv2.putText(img, f"  White-Green Contact: {contact_text}", (180, row), font, fs, contact_color, thick)

    row += dy
    success = feat['white_z'] > feat['green_z'] and feat['contact']
    if success:
        cv2.putText(img, "STACKED!", (10, row), font, 0.7, green, 2)

    # Progress bar at bottom of panel
    progress = step / max(max_steps, 1)
    bar_y = panel_h - 8
    cv2.rectangle(img, (10, bar_y), (int(10 + (w - 20) * progress), bar_y + 5), yellow, -1)
    cv2.rectangle(img, (10, bar_y), (w - 10, bar_y + 5), (100, 100, 100), 1)

    return img


def _bar(img, x, y, value, max_val, color, bar_w=120, bar_h=12):
    """Draw a horizontal bar showing value/max_val."""
    ratio = min(value / max(max_val, 1e-6), 1.0)
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(bar_w * ratio), y + bar_h), color, -1)
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (100, 100, 100), 1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(POSITIONS_FILE) as f:
        all_positions = json.load(f)
    positions = all_positions[:NUM_POSITIONS]
    print(f"Generating video for {len(positions)} positions, max {MAX_STEPS} steps each")

    print("Creating env...")
    mujoco_env = MuJoCoVecEnvStack(
        num_envs=1, reward_type="dense", max_episode_steps=MAX_STEPS
    )
    print("Loading GR00T...")
    wrapper = MuJoCoResidualWrapperStack(
        vec_env=mujoco_env,
        groot_checkpoint=GROOT_CKPT,
        policy_device="cuda:0",
    )
    print("Ready.\n")

    for pos_idx, pos_info in enumerate(positions):
        print(f"=== Position {pos_idx} ===")
        video_path = OUT_DIR / f"pos{pos_idx}_features.mp4"

        # Reset
        obs, _ = wrapper.reset()
        env_data = mujoco_env._envs[0]
        data, model = env_data["data"], env_data["model"]
        wqa = env_data["white_qposadr"]
        green_body_id = env_data["green_body_id"]

        for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
        for fid in env_data["ids"]["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04

        white_pos = np.array(pos_info["white_cube_pos"], dtype=np.float64)
        green_pos = np.array(pos_info["green_cube_pos"], dtype=np.float64)
        if wqa is not None:
            data.qpos[wqa:wqa+3] = white_pos
            q = Rotation.from_euler("z", np.radians(pos_info.get("white_yaw_deg", 0.0))).as_quat()
            data.qpos[wqa+3:wqa+7] = [q[3], q[0], q[1], q[2]]
        if green_body_id >= 0:
            model.body_pos[green_body_id] = green_pos

        env_data["grasp_state"] = {"grasped": False, "contact_count": 0}
        weld_id = env_data.get("weld_eq_id", -1)
        if weld_id >= 0:
            model.eq_active0[weld_id] = 0
            data.eq_active[weld_id] = 0

        mujoco.mj_forward(model, data)
        mujoco_env._step_counts[:] = 0
        mujoco_env._initial_white_z[0] = data.qpos[wqa + 2] if wqa else 0.0

        # First frame
        frame0 = render_cam_base(mujoco_env, 0)
        h, w = frame0.shape[:2]

        # Use imageio-ffmpeg for H.264 (avc1) encoding
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        writer = imageio_ffmpeg.write_frames(
            str(video_path), (w, h), fps=FPS,
            codec="libx264",
            pix_fmt_in="bgr24",
            output_params=["-crf", "23", "-preset", "fast",
                           "-pix_fmt", "yuv420p"],
            ffmpeg_log_level="error",
        )
        writer.send(None)  # initialize generator

        feat = extract_features(mujoco_env, 0)
        overlay = draw_overlay(frame0, feat, 0, MAX_STEPS)
        writer.send(overlay.tobytes())

        t0 = time.time()
        for step in range(MAX_STEPS):
            residual = torch.zeros((1, 7), dtype=torch.float32)
            next_obs, reward, terminated, truncated, info = wrapper.step(residual)
            obs = next_obs

            feat = extract_features(mujoco_env, 0)
            frame = render_cam_base(mujoco_env, 0)
            overlay = draw_overlay(frame, feat, step + 1, MAX_STEPS)
            writer.send(overlay.tobytes())

            if terminated[0] or truncated[0]:
                break

        writer.close()
        elapsed = time.time() - t0
        n_frames = step + 2  # step 0 + steps done
        success = bool(terminated[0])
        tag = "SUCCESS" if success else "FAIL"
        print(f"  {tag} | {n_frames} frames | {elapsed:.1f}s | {video_path}")
        print(f"  Final: tcp_white={feat['tcp_white_dist']:.4f} "
              f"white_green_top={feat['white_to_green_top']:.4f} "
              f"grasped={feat['grasped']} contact={feat['contact']}")

    print(f"\nDone! Videos in {OUT_DIR}/")


if __name__ == "__main__":
    main()
