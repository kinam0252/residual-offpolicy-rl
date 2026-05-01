"""Reward debug with video: render rollout frames with reward bar overlay.

Runs base policy (zero residual) for D2/D3/D4, renders side-by-side video
with reward breakdown bar chart overlay.

Usage:
    python debug_reward_video.py
"""
import sys, os, importlib, types as _types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed"); _ds.__version__="0.0.0"
    _ds.__spec__=importlib.machinery.ModuleSpec("deepspeed",None); _ds.__path__=[]; _ds.__file__=__file__
    _dz = _types.ModuleType("deepspeed.zero"); _dz.__spec__=importlib.machinery.ModuleSpec("deepspeed.zero",None)
    _dz.Init=lambda *a,**kw:(lambda f:f); _ds.zero=_dz
    _dc = _types.ModuleType("deepspeed.comm"); _dc.__spec__=importlib.machinery.ModuleSpec("deepspeed.comm",None)
    _ds.comm=_dc
    sys.modules["deepspeed"]=_ds; sys.modules["deepspeed.zero"]=_dz; sys.modules["deepspeed.comm"]=_dc
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except: pass

import numpy as np
import torch
import cv2
import imageio
from pathlib import Path

torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, load_episode_mapping, FLOOR_TO_DRAWER,
    DEMO_CONTACT_MEAN, DRAWER_SLIDE, DRAWER_SLOT_HEIGHT, DRAWER_GAP, TCP_OFFSET,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer

FPS = 15
MAX_STEPS = 500


def reward_breakdown(mujoco_env, env_idx):
    """Compute reward stage breakdown."""
    env = mujoco_env._envs[env_idx]
    model, data, ids = env["model"], env["data"], env["ids"]
    active_drawer = env["active_drawer"]

    hand_pos = data.xpos[ids["hand_id"]]
    hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
    tcp = hand_pos + hand_mat @ TCP_OFFSET
    cab_pos_w = data.xpos[env["cab_body_id"]]
    cab_mat = data.xmat[env["cab_body_id"]].reshape(3, 3)
    tcp_local = cab_mat.T @ (tcp - cab_pos_w)

    dm = DEMO_CONTACT_MEAN.get(active_drawer)
    face_hz = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP
    dz_center = env["drawer_z_min"] + face_hz + DRAWER_GAP
    target_x = mujoco_env._face_x_closed + mujoco_env._drawer_qpos[env_idx]
    target_y = dm["y_norm"] * mujoco_env._face_hy if dm else 0.0
    target_z = dz_center + dm["z_norm"] * face_hz if dm else dz_center

    dist_3d = float(np.linalg.norm(tcp_local - np.array([target_x, target_y, target_z])))
    approach = (1.0 - np.tanh(dist_3d / 0.1)) * 0.25

    y_ok = abs(tcp_local[1]) < mujoco_env._face_hy + 0.03
    z_ok = (tcp_local[2] > env["drawer_z_min"]) and (tcp_local[2] < env["drawer_z_max"])
    contact = 0.0
    norm_dist = float('inf')
    if y_ok and z_ok and dm is not None:
        y_norm = tcp_local[1] / mujoco_env._face_hy if mujoco_env._face_hy > 0 else 0.0
        z_norm = (tcp_local[2] - dz_center) / face_hz if face_hz > 0 else 0.0
        norm_dist = np.sqrt((y_norm - dm["y_norm"])**2 + (z_norm - dm["z_norm"])**2)
        if norm_dist <= 1.0:
            contact = 0.25

    closed_frac = 1.0 - (mujoco_env._drawer_qpos[env_idx] / DRAWER_SLIDE)
    push = float(np.clip(closed_frac, 0.0, 1.0)) * 0.25
    success = 0.25 if mujoco_env._is_success(env_idx) else 0.0

    return {
        "approach": approach,
        "contact": contact,
        "push": push,
        "success": success,
        "total": approach + contact + push + success,
        "dist_3d": dist_3d,
        "norm_dist": norm_dist,
        "closed_frac": float(np.clip(closed_frac, 0, 1)),
    }


def draw_reward_bar(frame, bd, step, drawer_idx, max_steps):
    """Draw reward breakdown as stacked horizontal bar + text on the frame."""
    h, w = frame.shape[:2]
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # --- Text overlay (top-left) ---
    cf_pct = bd["closed_frac"] * 100
    nd_str = f"{bd['norm_dist']:.2f}" if bd["norm_dist"] < 100 else "inf"
    lines = [
        f"Step {step:03d}/{max_steps}  Drawer {drawer_idx}",
        f"Reward: {bd['total']:.3f}  Closed: {cf_pct:.0f}%",
        f"Dist3D: {bd['dist_3d']:.3f}m  NormDist: {nd_str}",
    ]
    for i, line in enumerate(lines):
        y = 18 + i * 18
        cv2.putText(bgr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # --- Stacked bar (bottom) ---
    bar_h = 30
    bar_y = h - bar_h - 10
    bar_x = 10
    bar_w = w - 20

    # Background
    cv2.rectangle(bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (40, 40, 40), -1)
    cv2.rectangle(bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (100, 100, 100), 1)

    # Stacked segments: approach(blue), contact(cyan), push(yellow), success(green)
    colors = [
        (255, 100, 50),   # approach - orange-ish blue in BGR
        (200, 200, 0),    # contact - cyan
        (0, 200, 255),    # push - yellow in BGR
        (0, 200, 0),      # success - green
    ]
    labels = ["App", "Con", "Push", "Suc"]
    values = [bd["approach"], bd["contact"], bd["push"], bd["success"]]

    x_cursor = bar_x
    for val, color, label in zip(values, colors, labels):
        seg_w = int(val / 1.0 * bar_w)  # max reward = 1.0
        if seg_w > 0:
            cv2.rectangle(bgr, (x_cursor, bar_y), (x_cursor + seg_w, bar_y + bar_h),
                          color, -1)
            if seg_w > 30:
                cv2.putText(bgr, f"{label} {val:.2f}", (x_cursor + 3, bar_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            x_cursor += seg_w

    # Total text
    cv2.putText(bgr, f"Total: {bd['total']:.3f}", (bar_x + bar_w - 100, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_video(frames, path, fps=FPS):
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264',
                                output_params=['-pix_fmt', 'yuv420p', '-crf', '23'])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  Saved video: {path} ({len(frames)} frames)")


def run_episode(wrapper, mujoco_env, drawer_idx, video_path):
    """Run one episode with video + reward overlay."""
    mujoco_env._active_drawers[0] = drawer_idx
    obs, _ = wrapper.reset()

    e = mujoco_env._envs[0]
    model, data = e["model"], e["data"]

    frames = []
    for step in range(MAX_STEPS):
        # Reward breakdown BEFORE step (current state)
        bd = reward_breakdown(mujoco_env, 0)

        # Render
        e["renderer_base"].update_scene(data, camera=e["cam_base_id"],
                                        scene_option=e["opt_base"])
        img_base = e["renderer_base"].render().copy()

        e["renderer_wrist"].update_scene(data, camera=e["cam_wrist_id"],
                                         scene_option=e["opt_wrist"])
        img_wrist = e["renderer_wrist"].render().copy()

        # Resize to same height
        h_target = 360
        w_base = int(img_base.shape[1] * h_target / img_base.shape[0])
        w_wrist = int(img_wrist.shape[1] * h_target / img_wrist.shape[0])
        base_r = cv2.resize(img_base, (w_base, h_target))
        wrist_r = cv2.resize(img_wrist, (w_wrist, h_target))

        # Reward overlay on base image
        base_r = draw_reward_bar(base_r, bd, step, drawer_idx, MAX_STEPS)

        # Wrist label
        bgr_w = cv2.cvtColor(wrist_r, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr_w, f"WRIST {step:03d}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        wrist_r = cv2.cvtColor(bgr_w, cv2.COLOR_BGR2RGB)

        combined = np.hstack([base_r, wrist_r])
        frames.append(combined)

        # Step
        residual = torch.zeros((1, 7), dtype=torch.float32)
        obs, reward, terminated, truncated, info = wrapper.step(residual)

        if step % 25 == 0 or bd["contact"] > 0 or bd["success"] > 0:
            print(f"  step={step:3d} r={bd['total']:.3f} "
                  f"app={bd['approach']:.3f} con={bd['contact']:.3f} "
                  f"push={bd['push']:.3f} suc={bd['success']:.3f} "
                  f"cf={bd['closed_frac']:.2f}")

        if terminated[0] or truncated[0]:
            # Capture final state reward
            bd_final = reward_breakdown(mujoco_env, 0)
            tag = "SUCC" if bool(terminated[0]) else "TIMEOUT"
            print(f"  >>> {tag} at step {step+1}: final_r={float(reward[0]):.3f}")
            break

    write_video(frames, video_path)


def main():
    ckpt = os.path.expanduser("~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000")
    out_dir = Path(os.path.expanduser("~/Repos/Intern/.VIEW/reward_debug/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = load_episode_mapping()

    print("Creating env (dense reward, contact_threshold=1.0)...")
    mujoco_env = MuJoCoVecEnvDrawer(
        num_envs=1,
        reward_type="dense",
        max_episode_steps=MAX_STEPS,
        contact_threshold=1.0,
    )
    print("Loading GR00T...")
    wrapper = MuJoCoResidualWrapperDrawer(
        vec_env=mujoco_env,
        groot_checkpoint=ckpt,
        policy_device="cuda:0",
    )
    print("Ready.\n")

    # Pick one episode per drawer
    picked = {}
    for ei, ep in enumerate(eps):
        d = FLOOR_TO_DRAWER.get(ep["floor"])
        if d and d not in picked:
            picked[d] = (ei, ep)
        if len(picked) == 3:
            break

    for drawer_idx in sorted(picked):
        ei, ep = picked[drawer_idx]
        print(f"{'='*60}")
        print(f"  Drawer {drawer_idx}  (ep{ei})")
        print(f"{'='*60}")
        video_path = str(out_dir / f"reward_debug_d{drawer_idx}.mp4")
        run_episode(wrapper, mujoco_env, drawer_idx, video_path)
        print()

    print(f"\nAll videos saved to: {out_dir}")


if __name__ == "__main__":
    main()
