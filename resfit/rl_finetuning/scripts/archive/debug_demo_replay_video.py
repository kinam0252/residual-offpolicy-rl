"""Demo replay with video + reward overlay. 2 episodes per drawer."""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# Ensure EGL libs are found
_gl_path = os.path.expanduser("~/.local/lib/gl")
if os.path.isdir(_gl_path):
    os.environ['LD_LIBRARY_PATH'] = _gl_path + ":" + os.environ.get('LD_LIBRARY_PATH', '')
    import ctypes
    ctypes.CDLL(os.path.join(_gl_path, "libGLdispatch.so.0"))
    ctypes.CDLL(os.path.join(_gl_path, "libOpenGL.so.0"))
    ctypes.CDLL(os.path.join(_gl_path, "libEGL.so.1"))

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import numpy as np
import torch
import cv2
import imageio
import pandas as pd
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, load_episode_mapping, FLOOR_TO_DRAWER,
    DEMO_CONTACT_MEAN, DRAWER_SLIDE, DRAWER_SLOT_HEIGHT, DRAWER_GAP, TCP_OFFSET,
    WALL_THICK,
)

DATASET_DIR = Path(os.path.expanduser(
    "~/DATA/INTERN/datasets/CloseDrawer_sim_33ep/data/chunk-000"))
OUT_DIR = Path(os.path.expanduser("~/Repos/Intern/.VIEW/demo_replay/"))
FPS = 15


def compute_ndist(env, env_idx):
    e = env._envs[env_idx]
    model, data, ids = e["model"], e["data"], e["ids"]
    active_drawer = e["active_drawer"]
    hand_pos = data.xpos[ids["hand_id"]]
    hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
    tcp = hand_pos + hand_mat @ TCP_OFFSET
    cab_pos_w = data.xpos[e["cab_body_id"]]
    cab_mat = data.xmat[e["cab_body_id"]].reshape(3, 3)
    tcp_local = cab_mat.T @ (tcp - cab_pos_w)

    y_ok = abs(tcp_local[1]) < env._face_hy + 0.03
    z_ok = (tcp_local[2] > e["drawer_z_min"]) and (tcp_local[2] < e["drawer_z_max"])
    dm = DEMO_CONTACT_MEAN.get(active_drawer)
    if not (y_ok and z_ok) or dm is None:
        return float("inf"), 0.0, 0.0, y_ok, z_ok

    face_hz = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP
    dz_center = e["drawer_z_min"] + face_hz + DRAWER_GAP
    y_norm = tcp_local[1] / env._face_hy if env._face_hy > 0 else 0.0
    z_norm = (tcp_local[2] - dz_center) / face_hz if face_hz > 0 else 0.0
    ndist = np.sqrt((y_norm - dm["y_norm"])**2 + (z_norm - dm["z_norm"])**2)
    return ndist, y_norm, z_norm, y_ok, z_ok


def add_overlay(frame, lines, y_start=20):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for i, line in enumerate(lines):
        y = y_start + i * 18
        cv2.putText(bgr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_reward_bar(frame, reward, closed_frac, ndist, step, drawer_idx, total_steps):
    h, w = frame.shape[:2]
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Bottom bar: closed fraction
    bar_h, bar_y = 25, h - 35
    bar_x, bar_w = 10, w - 20
    cv2.rectangle(bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    fill_w = int(closed_frac * bar_w)
    color = (0, 200, 0) if closed_frac >= 0.9 else (0, 200, 255)
    if fill_w > 0:
        cv2.rectangle(bgr, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.rectangle(bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
    cv2.putText(bgr, f"Closed: {closed_frac*100:.0f}%", (bar_x + 5, bar_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # ndist indicator bar
    nd_bar_y = bar_y - 30
    cv2.rectangle(bgr, (bar_x, nd_bar_y), (bar_x + bar_w, nd_bar_y + 20), (40, 40, 40), -1)
    if ndist < 100:
        nd_frac = min(ndist / 3.0, 1.0)  # scale: 0-3 maps to bar
        nd_fill = int(nd_frac * bar_w)
        nd_color = (0, 255, 0) if ndist <= 1.0 else (0, 100, 255)
        if nd_fill > 0:
            cv2.rectangle(bgr, (bar_x, nd_bar_y), (bar_x + nd_fill, nd_bar_y + 20), nd_color, -1)
        # threshold=1.0 marker
        t_x = bar_x + int(1.0 / 3.0 * bar_w)
        cv2.line(bgr, (t_x, nd_bar_y), (t_x, nd_bar_y + 20), (0, 0, 255), 2)
    nd_str = f"{ndist:.2f}" if ndist < 100 else "inf"
    cv2.putText(bgr, f"NormDist: {nd_str} (|=1.0)", (bar_x + 5, nd_bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _set_active_drawer(env, env_idx, drawer_idx):
    """Properly switch active drawer: update env dict + z-range."""
    e = env._envs[env_idx]
    e["active_drawer"] = drawer_idx
    env._active_drawers[env_idx] = drawer_idx
    slot_h = DRAWER_SLOT_HEIGHT
    dz = WALL_THICK + slot_h / 2 + drawer_idx * (slot_h + WALL_THICK)
    e["drawer_z_center"] = dz
    e["drawer_z_min"] = dz - slot_h / 2 - 0.02
    e["drawer_z_max"] = dz + slot_h / 2 + 0.02


def run_replay(env, ei, drawer_idx, actions, video_path):
    _set_active_drawer(env, 0, drawer_idx)
    env.reset()
    e = env._envs[0]

    frames = []
    for t in range(len(actions)):
        # Pre-step state
        ndist, yn, zn, y_ok, z_ok = compute_ndist(env, 0)
        closed_frac = 1.0 - (env._drawer_qpos[0] / DRAWER_SLIDE)
        reward = env._compute_reward(0)

        # Render
        e["renderer_base"].update_scene(e["data"], camera=e["cam_base_id"],
                                        scene_option=e["opt_base"])
        img_base = e["renderer_base"].render().copy()
        e["renderer_wrist"].update_scene(e["data"], camera=e["cam_wrist_id"],
                                         scene_option=e["opt_wrist"])
        img_wrist = e["renderer_wrist"].render().copy()

        h_target = 360
        w_base = int(img_base.shape[1] * h_target / img_base.shape[0])
        w_wrist = int(img_wrist.shape[1] * h_target / img_wrist.shape[0])
        base_r = cv2.resize(img_base, (w_base, h_target))
        wrist_r = cv2.resize(img_wrist, (w_wrist, h_target))

        nd_str = f"{ndist:.2f}" if ndist < 100 else "inf"
        overlay_lines = [
            f"DEMO REPLAY  ep{ei} D{drawer_idx}  step {t}/{len(actions)}",
            f"reward={reward:.3f}  closed={closed_frac*100:.0f}%  ndist={nd_str}",
        ]
        base_r = add_overlay(base_r, overlay_lines)
        base_r = draw_reward_bar(base_r, reward, closed_frac, ndist, t, drawer_idx, len(actions))
        wrist_r = add_overlay(wrist_r, [f"WRIST {t:03d}"])

        combined = np.hstack([base_r, wrist_r])
        frames.append(combined)

        # Step
        action_t = torch.tensor(actions[t:t+1], dtype=torch.float32)
        obs, r, term, trunc, info = env.step(action_t)
        if term[0]:
            break

    # Save
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(video_path), fps=FPS, codec='libx264',
                                output_params=['-pix_fmt', 'yuv420p', '-crf', '23'])
    for f in frames:
        writer.append_data(f)
    writer.close()

    final_closed = 1.0 - (env._drawer_qpos[0] / DRAWER_SLIDE)
    success = env._is_success(0)
    tag = "SUCC" if success else "FAIL"
    print(f"  ep{ei} d{drawer_idx} {tag} closed={final_closed*100:.1f}% "
          f"frames={len(frames)} → {video_path}")


def main():
    eps = load_episode_mapping()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = MuJoCoVecEnvDrawer(
        num_envs=1, reward_type="dense", max_episode_steps=1000,
        contact_threshold=float("inf"),
    )

    # Pick 2 per drawer
    picked = {2: [], 3: [], 4: []}
    for ei, ep in enumerate(eps):
        d = FLOOR_TO_DRAWER.get(ep["floor"])
        if d and len(picked[d]) < 2:
            picked[d].append(ei)
        if all(len(v) == 2 for v in picked.values()):
            break

    for d in [2, 3, 4]:
        for ei in picked[d]:
            pq = DATASET_DIR / f"episode_{ei:06d}.parquet"
            df = pd.read_parquet(pq)
            actions = np.array(df["action"].tolist(), dtype=np.float32)
            video_path = str(OUT_DIR / f"demo_replay_ep{ei:02d}_d{d}.mp4")
            run_replay(env, ei, d, actions, video_path)

    print(f"\nAll videos → {OUT_DIR}")


if __name__ == "__main__":
    main()
