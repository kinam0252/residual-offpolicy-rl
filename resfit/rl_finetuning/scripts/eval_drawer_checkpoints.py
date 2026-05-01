"""Evaluate GR00T close-drawer checkpoints with work-stealing.

Each worker scans for unclaimed (ckpt, episode) pairs, locks one,
runs it, saves result JSON + optional MP4, then loops. Multiple
workers can run in parallel without overlap.

Usage:
    python eval_drawer_checkpoints.py [--output_dir outputs/ckpt_eval_drawer]

Work-stealing protocol:
    outputs/ckpt_eval_drawer/
      ckpt_025000/
        ep00.json           # completed result
        ep00.mp4            # feature video (optional)
        ep00.lock/          # dir-lock while running
      ...
"""
import sys, os, json, time, argparse, shutil
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

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, DRAWER_SLIDE, FLOOR_TO_DRAWER,
    get_tcp_pose, get_drawer_pos, load_episode_mapping,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer

# Real robot initial joint configuration
REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])

# ── Checkpoints to evaluate ──
CKPT_BASE = os.path.expanduser("~/DATA/INTERN/training/groot_drawer_sim_33ep")
CKPT_STEPS = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000]

# Episode mapping (33 episodes with floor info)
_EPISODE_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
    '..', 'MSRA', 'close_drawer', 'physics', 'real_sim_episode_mapping.json'
)

MAX_STEPS = 500
FPS = 15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/ckpt_eval_drawer")
    p.add_argument("--no_video", action="store_true", default=True)
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    return p.parse_args()


def extract_features(mujoco_env, env_idx=0):
    """Extract task-relevant features for logging/overlay."""
    env = mujoco_env._envs[env_idx]
    model, data, ids = env["model"], env["data"], env["ids"]

    tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
    drawer_slide = mujoco_env._drawer_qpos[env_idx]
    closed_frac = mujoco_env.get_drawer_closed_fraction(env_idx)
    active_drawer = env["active_drawer"]

    # Compute TCP-to-cabinet-face distance (in cabinet local frame)
    cab_pos_w = data.xpos[env["cab_body_id"]]
    cab_mat = data.xmat[env["cab_body_id"]].reshape(3, 3)
    tcp_local = cab_mat.T @ (tcp_pos - cab_pos_w)

    return {
        "tcp_pos": tcp_pos.tolist(),
        "tcp_local_x": float(tcp_local[0]),
        "tcp_local_y": float(tcp_local[1]),
        "tcp_local_z": float(tcp_local[2]),
        "drawer_slide": float(drawer_slide),
        "closed_fraction": float(closed_frac),
        "active_drawer": int(active_drawer),
    }


def render_cam_base(mujoco_env, env_idx=0):
    env = mujoco_env._envs[env_idx]
    env["renderer_base"].update_scene(
        env["data"], camera=env["cam_base_id"], scene_option=env["opt_base"]
    )
    return env["renderer_base"].render().copy()


def draw_overlay(frame, feat, step, max_steps):
    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    overlay = img.copy()
    panel_h = 140
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thick = 0.45, 1
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)

    y, dy = 18, 20
    cv2.putText(img, f"Step {step}/{max_steps}  Drawer #{feat['active_drawer']}", (10, y), font, 0.55, white, thick)
    y += dy
    cv2.putText(img, f"Slide: {feat['drawer_slide']:.4f} / {DRAWER_SLIDE:.3f}", (10, y), font, fs, cyan, thick)
    _bar(img, 300, y - 10, feat['drawer_slide'], DRAWER_SLIDE, cyan)
    y += dy
    frac = feat['closed_fraction']
    fc = green if frac > 0.9 else yellow if frac > 0.5 else red
    cv2.putText(img, f"Closed: {frac*100:.1f}%", (10, y), font, fs, fc, thick)
    _bar_frac(img, 300, y - 10, frac, green)
    y += dy
    cv2.putText(img, f"TCP local: ({feat['tcp_local_x']:.3f}, {feat['tcp_local_y']:.3f}, {feat['tcp_local_z']:.3f})",
                (10, y), font, fs, white, thick)
    y += dy
    if feat['closed_fraction'] > 0.92:
        cv2.putText(img, "CLOSED!", (10, y), font, 0.7, green, 2)

    bar_y = panel_h - 8
    cv2.rectangle(img, (10, bar_y), (int(10 + (w - 20) * step / max(max_steps, 1)), bar_y + 5), yellow, -1)
    cv2.rectangle(img, (10, bar_y), (w - 10, bar_y + 5), (100, 100, 100), 1)
    return img


def _bar(img, x, y, val, mx, color, bw=120, bh=12):
    r = min(val / max(mx, 1e-6), 1.0)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(bw * r), y + bh), color, -1)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (100, 100, 100), 1)


def _bar_frac(img, x, y, frac, color, bw=120, bh=12):
    r = min(max(frac, 0.0), 1.0)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(bw * r), y + bh), color, -1)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (100, 100, 100), 1)


def try_lock(lock_path: Path) -> bool:
    try:
        lock_path.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        return False


def run_episode(wrapper, mujoco_env, ep_info, max_steps, video_path=None):
    """Run one episode. Returns result dict.

    ep_info: dict with 'floor' and 'real_ep' keys.
    """
    floor = ep_info["floor"]
    active_drawer = FLOOR_TO_DRAWER[floor]

    # Set active drawer and reset
    mujoco_env._active_drawers[0] = active_drawer
    env_data = mujoco_env._envs[0]
    env_data["active_drawer"] = active_drawer

    # Recompute drawer z-range for this drawer
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
        WALL_THICK, DRAWER_SLOT_HEIGHT,
    )
    slot_h = DRAWER_SLOT_HEIGHT
    dz = WALL_THICK + slot_h / 2 + active_drawer * (slot_h + WALL_THICK)
    env_data["drawer_z_center"] = dz
    env_data["drawer_z_min"] = dz - slot_h / 2 - 0.02
    env_data["drawer_z_max"] = dz + slot_h / 2 + 0.02

    obs, _ = wrapper.reset()
    model, data = env_data["model"], env_data["data"]

    # Set robot initial pose
    for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
    for fid in env_data["ids"]["finger_ids"]:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = 0.04

    mujoco.mj_forward(model, data)
    mujoco_env._step_counts[:] = 0

    # Video writer
    writer = None
    if video_path:
        frame0 = render_cam_base(mujoco_env, 0)
        h, w = frame0.shape[:2]
        writer = imageio_ffmpeg.write_frames(
            str(video_path), (w, h), fps=FPS,
            codec="libx264", pix_fmt_in="bgr24",
            output_params=["-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p"],
            ffmpeg_log_level="error",
        )
        writer.send(None)
        feat = extract_features(mujoco_env, 0)
        writer.send(draw_overlay(frame0, feat, 0, max_steps).tobytes())

    # Rollout
    feature_log = []
    t0 = time.time()
    success = False
    for step in range(max_steps):
        residual = torch.zeros((1, 7), dtype=torch.float32)
        next_obs, reward, terminated, truncated, info = wrapper.step(residual)
        obs = next_obs

        feat = extract_features(mujoco_env, 0)
        feature_log.append(feat)

        if writer:
            frame = render_cam_base(mujoco_env, 0)
            writer.send(draw_overlay(frame, feat, step + 1, max_steps).tobytes())

        if terminated[0] or truncated[0]:
            if terminated[0]:
                success = True
            break

    if writer:
        writer.close()

    elapsed = time.time() - t0
    n_steps = step + 1

    # Summary stats
    min_slide = min(f["drawer_slide"] for f in feature_log) if feature_log else DRAWER_SLIDE
    max_closed_frac = max(f["closed_fraction"] for f in feature_log) if feature_log else 0.0

    return {
        "success": success,
        "steps": n_steps,
        "time_s": round(elapsed, 1),
        "floor": floor,
        "active_drawer": active_drawer,
        "min_drawer_slide": round(min_slide, 4),
        "max_closed_fraction": round(max_closed_frac, 4),
        "final_drawer_slide": round(feature_log[-1]["drawer_slide"], 4) if feature_log else DRAWER_SLIDE,
        "final_closed_fraction": round(feature_log[-1]["closed_fraction"], 4) if feature_log else 0.0,
    }


def main():
    args = parse_args()
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Load episode mapping
    episodes = load_episode_mapping(_EPISODE_MAPPING_PATH)
    num_episodes = len(episodes)
    print(f"Loaded {num_episodes} episodes from mapping file")

    # Build work items: (ckpt_step, ep_idx)
    all_work = [(ckpt, ei) for ckpt in CKPT_STEPS for ei in range(num_episodes)]
    print(f"Total work items: {len(all_work)} ({len(CKPT_STEPS)} ckpts × {num_episodes} episodes)")

    mujoco_env = None
    wrapper = None
    current_ckpt = None

    completed = 0
    skipped = 0

    while True:
        claimed = False
        for ckpt_step, ep_idx in all_work:
            ckpt_tag = f"ckpt_{ckpt_step:06d}"
            ckpt_dir = out_base / ckpt_tag
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            result_path = ckpt_dir / f"ep{ep_idx:02d}.json"
            lock_path = ckpt_dir / f"ep{ep_idx:02d}.lock"
            video_path = ckpt_dir / f"ep{ep_idx:02d}.mp4"

            if result_path.exists():
                continue
            if not try_lock(lock_path):
                continue

            # Claimed!
            claimed = True
            ckpt_path = os.path.join(CKPT_BASE, f"checkpoint-{ckpt_step}")

            if current_ckpt != ckpt_step:
                print(f"\n{'='*60}")
                print(f"Loading checkpoint: {ckpt_step}")
                print(f"{'='*60}")
                if wrapper is not None:
                    del wrapper
                    torch.cuda.empty_cache()
                if mujoco_env is None:
                    mujoco_env = MuJoCoVecEnvDrawer(
                        num_envs=1, reward_type="sparse",
                        max_episode_steps=args.max_steps,
                    )
                wrapper = MuJoCoResidualWrapperDrawer(
                    vec_env=mujoco_env,
                    groot_checkpoint=ckpt_path,
                    policy_device="cuda:0",
                )
                current_ckpt = ckpt_step
                print("Ready.\n")

            ep_info = episodes[ep_idx]
            print(f"  [{ckpt_tag}] ep{ep_idx:02d} (floor={ep_info['floor']}) ...", end=" ", flush=True)

            try:
                vid = None if args.no_video else str(video_path)
                result = run_episode(wrapper, mujoco_env, ep_info, args.max_steps, video_path=vid)
                result["checkpoint"] = ckpt_step
                result["episode"] = ep_idx

                tmp = ckpt_dir / f"ep{ep_idx:02d}.tmp.json"
                with open(tmp, "w") as f:
                    json.dump(result, f, indent=2)
                tmp.rename(result_path)

                tag = "SUCC" if result["success"] else "FAIL"
                print(f"{tag} steps={result['steps']} closed={result['final_closed_fraction']*100:.0f}% "
                      f"t={result['time_s']}s")
                completed += 1
            except Exception as e:
                import traceback
                print(f"ERROR: {e}")
                traceback.print_exc()
            finally:
                shutil.rmtree(lock_path, ignore_errors=True)

            break

        if not claimed:
            break

    print(f"\n{'='*60}")
    print(f"Worker done. Completed: {completed}")
    _print_summary(out_base, episodes)


def _print_summary(out_base: Path, episodes: list):
    """Print success rate table by checkpoint and floor."""
    print(f"\n{'='*70}")
    print(f"{'Checkpoint':>12} | {'SR':>6} | {'Avg Closed%':>11} | {'F3 SR':>6} | {'F4 SR':>6} | {'F5 SR':>6}")
    print(f"{'-'*12}-+-{'-'*6}-+-{'-'*11}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for ckpt_step in CKPT_STEPS:
        ckpt_dir = out_base / f"ckpt_{ckpt_step:06d}"
        results = []
        for p in sorted(ckpt_dir.glob("ep*.json")):
            if p.name.endswith(".tmp.json"):
                continue
            with open(p) as f:
                results.append(json.load(f))

        if not results:
            print(f"{ckpt_step:>12} | {'--':>6} |")
            continue

        n = len(results)
        sr = sum(1 for r in results if r["success"]) / n * 100
        avg_closed = np.mean([r["final_closed_fraction"] for r in results]) * 100

        # Per-floor breakdown
        floor_srs = {}
        for floor in [3, 4, 5]:
            fr = [r for r in results if r["floor"] == floor]
            if fr:
                floor_srs[floor] = sum(1 for r in fr if r["success"]) / len(fr) * 100
            else:
                floor_srs[floor] = -1

        f3 = f"{floor_srs[3]:5.1f}%" if floor_srs[3] >= 0 else "  -- "
        f4 = f"{floor_srs[4]:5.1f}%" if floor_srs[4] >= 0 else "  -- "
        f5 = f"{floor_srs[5]:5.1f}%" if floor_srs[5] >= 0 else "  -- "

        print(f"{ckpt_step:>12} | {sr:5.1f}% | {avg_closed:9.1f}% | {f3} | {f4} | {f5}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
