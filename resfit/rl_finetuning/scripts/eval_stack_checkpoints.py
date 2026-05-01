"""Evaluate GR00T stack checkpoints with work-stealing.

Each worker scans for unclaimed (ckpt, pos) pairs, locks one, runs it,
saves result JSON + feature video, then loops. Multiple workers can run
in parallel without overlap.

Usage:
    python eval_stack_checkpoints.py [--output_dir outputs/ckpt_eval_stack]

Work-stealing protocol:
    outputs/ckpt_eval_stack/
      ckpt_025000/
        pos00.json          # completed result
        pos00.mp4           # feature video
        pos00.lock/         # dir-lock while running
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

from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import (
    MuJoCoVecEnvStack, CUBE_HALF, get_tcp_pose,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack

REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])

# ── Checkpoints to evaluate ──
CKPT_BASE = os.path.expanduser("~/DATA/INTERN/training/groot_stack_sim_66ep")
CKPT_STEPS = [25000, 50000, 75000, 100000, 125000, 150000,
              175000, 200000, 225000, 250000, 275000, 300000]

POSITIONS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
    'configs', 'stack_cube_positions.json'
)
NUM_POSITIONS = 66
MAX_STEPS = 1000
FPS = 15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/ckpt_eval_stack_66pos")
    p.add_argument("--no_video", action="store_true", default=True, help="Skip video generation (faster)")
    return p.parse_args()


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
    panel_h = 180
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
    cv2.putText(img, f"Step {step}/{max_steps}", (10, y), font, 0.55, white, thick)
    y += dy
    cv2.putText(img, f"tcp->white:  {feat['tcp_white_dist']:.4f}", (10, y), font, fs, cyan, thick)
    _bar(img, 250, y - 10, feat['tcp_white_dist'], 0.4, cyan)
    y += dy
    cv2.putText(img, f"white->green_top: {feat['white_to_green_top']:.4f}", (10, y), font, fs, green, thick)
    _bar(img, 250, y - 10, feat['white_to_green_top'], 0.4, green)
    y += dy
    cv2.putText(img, f"white<>green_xy:  {feat['white_green_xy']:.4f}", (10, y), font, fs, (180, 180, 255), thick)
    _bar(img, 250, y - 10, feat['white_green_xy'], 0.4, (180, 180, 255))
    y += dy
    cv2.putText(img, f"white_z: {feat['white_z']:.4f}  green_z: {feat['green_z']:.4f}", (10, y), font, fs, white, thick)
    y += dy
    gc = green if feat['grasped'] else red
    cv2.putText(img, f"Grasped: {'YES' if feat['grasped'] else 'NO'}", (10, y), font, fs, gc, thick)
    cc = green if feat['contact'] else red
    cv2.putText(img, f"  W-G Contact: {'YES' if feat['contact'] else 'NO'}", (180, y), font, fs, cc, thick)
    y += dy
    if feat['white_z'] > feat['green_z'] and feat['contact']:
        cv2.putText(img, "STACKED!", (10, y), font, 0.7, green, 2)

    bar_y = panel_h - 8
    cv2.rectangle(img, (10, bar_y), (int(10 + (w - 20) * step / max(max_steps, 1)), bar_y + 5), yellow, -1)
    cv2.rectangle(img, (10, bar_y), (w - 10, bar_y + 5), (100, 100, 100), 1)
    return img


def _bar(img, x, y, val, mx, color, bw=120, bh=12):
    r = min(val / max(mx, 1e-6), 1.0)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(bw * r), y + bh), color, -1)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (100, 100, 100), 1)


def try_lock(lock_path: Path) -> bool:
    """Atomic directory-based lock. Returns True if acquired."""
    try:
        lock_path.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        return False


def run_episode(wrapper, mujoco_env, pos_info, max_steps, video_path=None):
    """Run one episode. Returns result dict."""
    obs, _ = wrapper.reset()
    env_data = mujoco_env._envs[0]
    data, model = env_data["data"], env_data["model"]
    wqa = env_data["white_qposadr"]
    green_body_id = env_data["green_body_id"]

    # Set robot initial pose
    for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
    for fid in env_data["ids"]["finger_ids"]:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = 0.04

    # Set cube positions
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
            break

    if writer:
        writer.close()

    elapsed = time.time() - t0
    success = bool(terminated[0])
    n_steps = step + 1

    # Summary stats
    min_tcp_white = min(f["tcp_white_dist"] for f in feature_log) if feature_log else 999
    min_white_green = min(f["white_to_green_top"] for f in feature_log) if feature_log else 999
    any_grasped = any(f["grasped"] for f in feature_log)
    any_contact = any(f["contact"] for f in feature_log)

    return {
        "success": success,
        "steps": n_steps,
        "time_s": round(elapsed, 1),
        "min_tcp_white_dist": round(min_tcp_white, 4),
        "min_white_to_green_top": round(min_white_green, 4),
        "any_grasped": any_grasped,
        "any_contact": any_contact,
    }


def main():
    args = parse_args()
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Load positions
    with open(POSITIONS_FILE) as f:
        all_positions = json.load(f)
    positions = all_positions[:NUM_POSITIONS]

    # Build work items: (ckpt_step, pos_idx)
    all_work = [(ckpt, pi) for ckpt in CKPT_STEPS for pi in range(NUM_POSITIONS)]
    print(f"Total work items: {len(all_work)} ({len(CKPT_STEPS)} ckpts × {NUM_POSITIONS} positions)")

    # Env created once; wrapper re-created per checkpoint
    mujoco_env = None
    wrapper = None
    current_ckpt = None

    completed = 0
    skipped = 0

    while True:
        # Scan for unclaimed work
        claimed = False
        for ckpt_step, pos_idx in all_work:
            ckpt_tag = f"ckpt_{ckpt_step:06d}"
            ckpt_dir = out_base / ckpt_tag
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            result_path = ckpt_dir / f"pos{pos_idx:02d}.json"
            lock_path = ckpt_dir / f"pos{pos_idx:02d}.lock"
            video_path = ckpt_dir / f"pos{pos_idx:02d}.mp4"

            # Already done?
            if result_path.exists():
                continue
            # Try to claim
            if not try_lock(lock_path):
                continue

            # Claimed! Load checkpoint if needed
            claimed = True
            ckpt_path = os.path.join(CKPT_BASE, f"checkpoint-{ckpt_step}")

            if current_ckpt != ckpt_step:
                print(f"\n{'='*60}")
                print(f"Loading checkpoint: {ckpt_step}")
                print(f"{'='*60}")
                # Clean up old wrapper
                if wrapper is not None:
                    del wrapper
                    torch.cuda.empty_cache()
                if mujoco_env is None:
                    mujoco_env = MuJoCoVecEnvStack(
                        num_envs=1, reward_type="dense", max_episode_steps=MAX_STEPS
                    )
                wrapper = MuJoCoResidualWrapperStack(
                    vec_env=mujoco_env,
                    groot_checkpoint=ckpt_path,
                    policy_device="cuda:0",
                )
                current_ckpt = ckpt_step
                print("Ready.\n")

            # Run episode
            pos_info = positions[pos_idx]
            print(f"  [{ckpt_tag}] pos{pos_idx:02d} ...", end=" ", flush=True)

            try:
                vid = None if args.no_video else str(video_path)
                result = run_episode(wrapper, mujoco_env, pos_info, MAX_STEPS, video_path=vid)
                result["checkpoint"] = ckpt_step
                result["position"] = pos_idx

                # Save result atomically
                tmp = ckpt_dir / f"pos{pos_idx:02d}.tmp.json"
                with open(tmp, "w") as f:
                    json.dump(result, f, indent=2)
                tmp.rename(result_path)

                tag = "SUCC" if result["success"] else "FAIL"
                print(f"{tag} steps={result['steps']} grasped={result['any_grasped']} "
                      f"t={result['time_s']}s")
                completed += 1
            except Exception as e:
                print(f"ERROR: {e}")
            finally:
                shutil.rmtree(lock_path, ignore_errors=True)

            break  # Re-scan from top after each job

        if not claimed:
            break  # All work done or locked

    # Print summary
    print(f"\n{'='*60}")
    print(f"Worker done. Completed: {completed}")

    # Aggregate results if all done
    _print_summary(out_base)


def _print_summary(out_base: Path):
    """Print success rate table if results are available."""
    print(f"\n{'='*60}")
    print(f"{'Checkpoint':>12} | {'SR':>6} | {'Grasped':>8} | {'Avg Steps':>10}")
    print(f"{'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*10}")

    for ckpt_step in CKPT_STEPS:
        ckpt_dir = out_base / f"ckpt_{ckpt_step:06d}"
        results = []
        for p in sorted(ckpt_dir.glob("pos*.json")):
            if p.name.endswith(".tmp.json"):
                continue
            with open(p) as f:
                results.append(json.load(f))

        if not results:
            print(f"{ckpt_step:>12} | {'--':>6} |")
            continue

        n = len(results)
        sr = sum(1 for r in results if r["success"]) / n * 100
        grasp_rate = sum(1 for r in results if r["any_grasped"]) / n * 100
        avg_steps = np.mean([r["steps"] for r in results])
        print(f"{ckpt_step:>12} | {sr:5.1f}% | {grasp_rate:6.1f}% | {avg_steps:9.0f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
