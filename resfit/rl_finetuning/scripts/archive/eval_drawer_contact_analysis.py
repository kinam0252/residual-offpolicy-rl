"""Evaluate GR00T close-drawer base policy and record TCP-face contact geometry.

For each episode, records the TCP position in cabinet-local frame at every step,
identifies pushing steps (where drawer actually moves), and computes normalised
contact coordinates relative to the drawer face centre.

This answers: "Does the gripper push the centre of the drawer face, or the edges?"

Usage:
    python eval_drawer_contact_analysis.py \
        --ckpt_step 100000 \
        --output_dir outputs/drawer_contact_analysis
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

import torch
torch.backends.cuda.enable_cudnn_sdp(False)
import importlib, types as _types
if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed"); _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__ = []; _ds.__file__ = __file__
    _dz = _types.ModuleType("deepspeed.zero")
    _dz.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _dz.Init = lambda *a, **kw: (lambda f: f); _ds.zero = _dz
    sys.modules["deepspeed"] = _ds; sys.modules["deepspeed.zero"] = _dz
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except Exception:
    pass

import numpy as np
import cv2
import mujoco
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, DRAWER_SLIDE, FLOOR_TO_DRAWER,
    ACTIVE_DRAWERS, WALL_THICK, DRAWER_SLOT_HEIGHT,
    CABINET_SHORT, CABINET_LONG, DRAWER_GAP,
    TCP_OFFSET, get_tcp_pose, load_episode_mapping,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import (
    MuJoCoResidualWrapperDrawer,
)

FPS = 15


def write_video(frames, path, fps=FPS):
    """Write frames (RGB) to mp4 with H.264 via imageio-ffmpeg."""
    if not frames:
        return
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264',
                                output_params=['-pix_fmt', 'yuv420p', '-crf', '23'])
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def add_overlay(frame, text_lines, y_start=20):
    """Add text overlay on frame (BGR→ overlay → back to RGB for storage)."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for i, line in enumerate(text_lines):
        y = y_start + i * 18
        cv2.putText(bgr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# Precompute face geometry (same as mujoco_vec_env_drawer)
_SLOT_H = DRAWER_SLOT_HEIGHT
_D_HX = (CABINET_LONG - 2 * WALL_THICK) / 2 - DRAWER_GAP
_FACE_HX = WALL_THICK / 2
_FACE_X_CLOSED = _D_HX + _FACE_HX
_FACE_HY = CABINET_SHORT / 2 - DRAWER_GAP
_FACE_HZ = (_SLOT_H + WALL_THICK) / 2 - DRAWER_GAP

MAX_STEPS = 500


def get_tcp_local(env_data):
    """Get TCP position in cabinet-local frame."""
    model, data, ids = env_data["model"], env_data["data"], env_data["ids"]
    hand_pos = data.xpos[ids["hand_id"]]
    hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
    tcp = hand_pos + hand_mat @ TCP_OFFSET

    cab_pos_w = data.xpos[env_data["cab_body_id"]]
    cab_mat = data.xmat[env_data["cab_body_id"]].reshape(3, 3)
    tcp_local = cab_mat.T @ (tcp - cab_pos_w)
    return tcp_local


def run_one(wrapper, mujoco_env, drawer_idx, max_steps, save_video_path=None):
    """Run one episode, recording TCP-face contact geometry and optionally video."""
    mujoco_env._active_drawers[0] = drawer_idx
    env_data = mujoco_env._envs[0]
    env_data["active_drawer"] = drawer_idx

    slot_h = DRAWER_SLOT_HEIGHT
    dz = WALL_THICK + slot_h / 2 + drawer_idx * (slot_h + WALL_THICK)
    env_data["drawer_z_center"] = dz
    env_data["drawer_z_min"] = dz - slot_h / 2 - 0.02
    env_data["drawer_z_max"] = dz + slot_h / 2 + 0.02

    obs, _ = wrapper.reset()

    trajectory = []
    frames = []
    prev_slide = DRAWER_SLIDE

    for step in range(max_steps):
        tcp_local = get_tcp_local(env_data)
        slide = float(mujoco_env._drawer_qpos[0])
        pushing = slide < prev_slide - 1e-6

        trajectory.append({
            "step": step,
            "tcp_local": [round(float(tcp_local[0]), 5),
                          round(float(tcp_local[1]), 5),
                          round(float(tcp_local[2]), 5)],
            "slide": round(slide, 5),
            "pushing": pushing,
        })

        # Render frame for video
        if save_video_path is not None:
            e = mujoco_env._envs[0]
            model, data = e["model"], e["data"]

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

            # Contact info
            y_norm = tcp_local[1] / _FACE_HY if _FACE_HY > 0 else 0.0
            z_norm = (tcp_local[2] - dz) / _FACE_HZ if _FACE_HZ > 0 else 0.0
            closed_pct = (1 - slide / DRAWER_SLIDE) * 100

            overlay = [
                f"Step {step:03d}/{max_steps}  Drawer {drawer_idx}  {'PUSH' if pushing else '    '}",
                f"Slide: {slide:.4f}/{DRAWER_SLIDE:.4f}  Closed: {closed_pct:.0f}%",
                f"TCP local: [{tcp_local[0]:.3f}, {tcp_local[1]:.3f}, {tcp_local[2]:.3f}]",
                f"Contact Y={y_norm:.2f} Z={z_norm:.2f} (0=centre, ±1=edge)",
            ]
            base_r = add_overlay(base_r, overlay)
            wrist_r = add_overlay(wrist_r, [f"WRIST Step {step:03d}"])

            combined = np.hstack([base_r, wrist_r])
            frames.append(combined)

        prev_slide = slide

        residual = torch.zeros((1, 7), dtype=torch.float32)
        next_obs, reward, terminated, truncated, info = wrapper.step(residual)
        obs = next_obs

        if terminated[0] or truncated[0]:
            break

    # Save video immediately after episode
    if save_video_path is not None and frames:
        write_video(frames, save_video_path)

    success = bool(terminated[0]) if isinstance(terminated[0], (bool, np.bool_)) else bool(terminated[0].item())
    n_steps = step + 1

    # Extract contact points (pushing steps only) with normalised coords
    contact_points = []
    for t in trajectory:
        if t["pushing"]:
            y_norm = t["tcp_local"][1] / _FACE_HY if _FACE_HY > 0 else 0.0
            z_norm = (t["tcp_local"][2] - dz) / _FACE_HZ if _FACE_HZ > 0 else 0.0
            x_depth = t["tcp_local"][0] - _FACE_X_CLOSED
            contact_points.append({
                "step": t["step"],
                "y_norm": round(y_norm, 4),
                "z_norm": round(z_norm, 4),
                "x_depth": round(x_depth, 5),
                "slide": t["slide"],
            })

    # Summary statistics
    push_steps = len(contact_points)
    summary = {"push_steps": push_steps, "total_steps": n_steps}
    if push_steps > 0:
        y_vals = [c["y_norm"] for c in contact_points]
        z_vals = [c["z_norm"] for c in contact_points]
        summary["mean_y_norm"] = round(float(np.mean(y_vals)), 4)
        summary["std_y_norm"] = round(float(np.std(y_vals)), 4)
        summary["mean_z_norm"] = round(float(np.mean(z_vals)), 4)
        summary["std_z_norm"] = round(float(np.std(z_vals)), 4)
        summary["min_y_norm"] = round(float(np.min(y_vals)), 4)
        summary["max_y_norm"] = round(float(np.max(y_vals)), 4)
        summary["min_z_norm"] = round(float(np.min(z_vals)), 4)
        summary["max_z_norm"] = round(float(np.max(z_vals)), 4)

    min_slide = min(t["slide"] for t in trajectory) if trajectory else DRAWER_SLIDE
    closed_frac = 1.0 - (min_slide / DRAWER_SLIDE) if DRAWER_SLIDE > 0 else 0.0

    return {
        "success": success,
        "steps": n_steps,
        "drawer": drawer_idx,
        "min_slide": round(min_slide, 4),
        "closed_fraction": round(closed_frac, 4),
        "contact_points": contact_points,
        "trajectory": trajectory,
        "summary": summary,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Drawer contact analysis eval")
    p.add_argument("--ckpt_step", type=int, default=100000)
    p.add_argument("--output_dir", default="outputs/drawer_contact_analysis")
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no_video", action="store_true", help="Skip video rendering")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = os.path.expanduser(
        f"~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-{args.ckpt_step}"
    )

    # Load episode mapping
    eps = load_episode_mapping()
    print(f"=== Drawer Contact Analysis ===")
    print(f"  ckpt: {args.ckpt_step}")
    print(f"  episodes: {len(eps)}")
    print(f"  output: {out_dir}")
    print(f"  face_hy={_FACE_HY:.4f}  face_hz={_FACE_HZ:.4f}  face_x_closed={_FACE_X_CLOSED:.4f}")
    print()

    # Create env + wrapper
    mujoco_env = MuJoCoVecEnvDrawer(
        num_envs=1,
        reward_type="sparse",
        max_episode_steps=args.max_steps,
    )
    wrapper = MuJoCoResidualWrapperDrawer(
        vec_env=mujoco_env,
        groot_checkpoint=ckpt_path,
        policy_device=args.device,
    )
    print("Ready.\n")

    # Per-drawer aggregation
    drawer_stats = {d: {"success": 0, "total": 0, "y_norms": [], "z_norms": []} for d in ACTIVE_DRAWERS}
    t_start = time.time()

    for ei, ep in enumerate(eps):
        floor = ep["floor"]
        if floor not in FLOOR_TO_DRAWER:
            print(f"  ep{ei}: floor={floor} not in active drawers, skipping")
            continue
        drawer_idx = FLOOR_TO_DRAWER[floor]
        result_path = out_dir / f"ep{ei:02d}_d{drawer_idx}.json"

        if result_path.exists():
            print(f"  ep{ei}: already done, skipping")
            # Still load for aggregation
            with open(result_path) as f:
                result = json.load(f)
            ds = drawer_stats[drawer_idx]
            ds["total"] += 1
            if result["success"]:
                ds["success"] += 1
            ds["y_norms"].extend([c["y_norm"] for c in result["contact_points"]])
            ds["z_norms"].extend([c["z_norm"] for c in result["contact_points"]])
            continue

        print(f"  ep{ei} floor={floor} drawer={drawer_idx} ...", end=" ", flush=True)
        video_path = None
        if not args.no_video:
            video_path = out_dir / f"ep{ei:02d}_d{drawer_idx}.mp4"
        result = run_one(wrapper, mujoco_env, drawer_idx, args.max_steps,
                         save_video_path=video_path)
        result["episode"] = ei
        result["floor"] = floor

        # Save
        tmp = result_path.with_suffix(".tmp.json")
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2)
        tmp.rename(result_path)

        tag = "SUCC" if result["success"] else "FAIL"
        s = result["summary"]
        push_info = ""
        if s["push_steps"] > 0:
            push_info = f" push={s['push_steps']}steps y={s.get('mean_y_norm',0):.2f}±{s.get('std_y_norm',0):.2f} z={s.get('mean_z_norm',0):.2f}±{s.get('std_z_norm',0):.2f}"
        print(f"{tag} steps={result['steps']} closed={result['closed_fraction']*100:.0f}%{push_info}")

        # Aggregate
        ds = drawer_stats[drawer_idx]
        ds["total"] += 1
        if result["success"]:
            ds["success"] += 1
        ds["y_norms"].extend([c["y_norm"] for c in result["contact_points"]])
        ds["z_norms"].extend([c["z_norm"] for c in result["contact_points"]])

    elapsed = time.time() - t_start

    # Print per-drawer summary
    print(f"\n{'='*60}")
    print(f"Contact Analysis Summary (ckpt={args.ckpt_step}, {elapsed:.0f}s)")
    print(f"{'='*60}")
    for d in ACTIVE_DRAWERS:
        ds = drawer_stats[d]
        if ds["total"] == 0:
            continue
        sr = ds["success"] / ds["total"] * 100
        print(f"\n  Drawer {d}: SR={ds['success']}/{ds['total']}={sr:.0f}%")
        if ds["y_norms"]:
            y = np.array(ds["y_norms"])
            z = np.array(ds["z_norms"])
            print(f"    Y (left-right): mean={y.mean():.3f} std={y.std():.3f} range=[{y.min():.3f}, {y.max():.3f}]")
            print(f"    Z (up-down):    mean={z.mean():.3f} std={z.std():.3f} range=[{z.min():.3f}, {z.max():.3f}]")
            print(f"    Interpretation: Y=0 → centre, ±1 → edge | Z=0 → centre, ±1 → edge")

    # Save global summary
    global_summary = {
        "checkpoint": args.ckpt_step,
        "episodes": len(eps),
        "time_s": round(elapsed, 1),
        "face_geometry": {
            "face_hy": round(_FACE_HY, 4),
            "face_hz": round(_FACE_HZ, 4),
            "face_x_closed": round(_FACE_X_CLOSED, 4),
        },
        "per_drawer": {},
    }
    for d in ACTIVE_DRAWERS:
        ds = drawer_stats[d]
        if ds["total"] == 0:
            continue
        entry = {
            "success_rate": round(ds["success"] / ds["total"], 3),
            "n_episodes": ds["total"],
            "n_push_steps": len(ds["y_norms"]),
        }
        if ds["y_norms"]:
            y = np.array(ds["y_norms"])
            z = np.array(ds["z_norms"])
            entry["y_norm"] = {"mean": round(float(y.mean()), 4), "std": round(float(y.std()), 4),
                               "min": round(float(y.min()), 4), "max": round(float(y.max()), 4)}
            entry["z_norm"] = {"mean": round(float(z.mean()), 4), "std": round(float(z.std()), 4),
                               "min": round(float(z.min()), 4), "max": round(float(z.max()), 4)}
        global_summary["per_drawer"][f"D{d}"] = entry

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(global_summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    wrapper.close()
    print("Done.")


if __name__ == "__main__":
    main()
