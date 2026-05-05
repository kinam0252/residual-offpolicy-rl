#!/usr/bin/env python3
"""Demo replay for Stack Cube using the env wrapper (new physics).

Replays demo actions through MuJoCoVecEnvStack one episode at a time
to verify physics changes. Saves video (base cam) per episode.
"""
import os, sys, json, ctypes, gc, time
import numpy as np

# EGL setup
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
_gl = os.path.expanduser("~/.local/lib/gl")
if os.path.isdir(_gl):
    for lib in ["libGLdispatch.so.0", "libOpenGL.so.0", "libEGL.so.1"]:
        p = os.path.join(_gl, lib)
        if os.path.exists(p):
            ctypes.cdll.LoadLibrary(p)

import cv2
import imageio
import pyarrow.parquet as pq

# Add project paths
_proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _proj)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import MuJoCoVecEnvStack, CUBE_HALF, RENDER_W, RENDER_H
import mujoco

# ── Config ──
DATA_DIR = os.path.expanduser("~/DATA/INTERN/datasets/Stack_sim_66ep/data/chunk-000")
POS_JSON = os.path.join(_proj, "configs", "stack_cube_positions.json")
OUTPUT_DIR = os.path.join(_proj, "outputs", "stack_cube_replay_videos")
N_EPISODES = 66
MAX_QUICK = 6
FPS = 15


def load_demo(ep_idx: int) -> np.ndarray:
    """Load actions from parquet. Returns (T, 8) array."""
    pq_path = os.path.join(DATA_DIR, f"episode_{ep_idx:06d}.parquet")
    table = pq.read_table(pq_path, columns=["action"])
    actions = np.array(table["action"].to_pylist(), dtype=np.float32)
    return actions


def render_frame(env, env_idx=0) -> np.ndarray:
    """Render base camera frame (H, W, 3) BGR for video."""
    e = env._envs[env_idx]
    e["renderer_base"].update_scene(e["data"], camera=e["cam_base_id"],
                                     scene_option=e["opt_base"])
    rgb = e["renderer_base"].render().copy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run_replay(quick: bool = True):
    """Run demo replay with video output."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(POS_JSON) as f:
        positions = json.load(f)

    n_ep = MAX_QUICK if quick else N_EPISODES
    n_ep = min(n_ep, len(positions))

    results = []
    t0 = time.time()

    for ep_idx in range(n_ep):
        pos_info = positions[ep_idx]
        white_pos = np.array(pos_info["white_cube_pos"], dtype=np.float64)
        green_pos = np.array(pos_info["green_cube_pos"], dtype=np.float64)

        from scipy.spatial.transform import Rotation
        def yaw_to_quat_wxyz(yaw_deg):
            r = Rotation.from_euler("z", np.radians(yaw_deg))
            q_xyzw = r.as_quat()
            return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]

        white_quat = yaw_to_quat_wxyz(pos_info.get("white_yaw_deg", 0))
        green_quat = yaw_to_quat_wxyz(pos_info.get("green_yaw_deg", 0))

        # Create env (1 at a time to save RAM)
        env = MuJoCoVecEnvStack(
            num_envs=1,
            white_cube_positions=[white_pos.tolist()],
            green_cube_positions=[green_pos.tolist()],
            white_cube_quats_wxyz=[white_quat],
            green_cube_quats_wxyz=[green_quat],
            max_episode_steps=300,
            reward_type="dense",
        )
        env.reset()

        # Load and replay demo actions
        actions = load_demo(ep_idx)
        n_steps = len(actions)

        # Video writer (H.264 via imageio-ffmpeg)
        vid_path = os.path.join(OUTPUT_DIR, f"ep{ep_idx:02d}.mp4")
        writer = imageio.get_writer(
            vid_path, fps=FPS, codec="libx264",
            macro_block_size=1,
            output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
        )

        import torch
        for t in range(n_steps):
            action = actions[t]
            action_t = torch.from_numpy(action[np.newaxis]).float()
            obs, reward, terminated, truncated, info = env.step(action_t, render_mode="none")

            # Render every frame
            frame = render_frame(env)  # BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Add text overlay
            cv2.putText(frame_rgb, f"ep{ep_idx:02d} f={t}/{n_steps}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            writer.append_data(frame_rgb)

        writer.close()

        # Check final state
        env_d = env._envs[0]
        data = env_d["data"]
        white_qa = env_d["white_qposadr"]
        green_qa = env_d["green_qposadr"]

        white_final = data.qpos[white_qa:white_qa + 3].copy() if white_qa else np.zeros(3)
        green_final = data.qpos[green_qa:green_qa + 3].copy() if green_qa else green_pos

        xy_err = float(np.linalg.norm(white_final[:2] - green_final[:2]))
        z_target = green_final[2] + 2 * CUBE_HALF
        z_err = abs(white_final[2] - z_target)
        stacked = z_err < 0.015 and xy_err < 0.020
        success = env._is_success(0)

        results.append({
            "ep": ep_idx,
            "stacked": stacked,
            "success_contact": success,
            "white_z": float(white_final[2]),
            "green_z": float(green_final[2]),
            "z_err": float(z_err),
            "xy_err": float(xy_err),
            "n_steps": n_steps,
        })

        status = "✓" if stacked else "✗"
        print(f"  ep{ep_idx:02d}: {status} z_err={z_err:.4f} xy_err={xy_err:.4f} "
              f"wz={white_final[2]:.4f} gz={green_final[2]:.4f} ({n_steps} steps) → {vid_path}")

        env.close()
        del env
        gc.collect()

    elapsed = time.time() - t0
    n_success = sum(r["stacked"] for r in results)
    n_contact = sum(r["success_contact"] for r in results)
    print(f"\n{'='*60}")
    print(f"Results: {n_success}/{n_ep} stacked, {n_contact}/{n_ep} contact-success")
    print(f"Time: {elapsed:.1f}s ({elapsed/n_ep:.1f}s/ep)")
    print(f"Mean z_err: {np.mean([r['z_err'] for r in results]):.4f}")
    print(f"Mean xy_err: {np.mean([r['xy_err'] for r in results]):.4f}")
    print(f"Videos: {OUTPUT_DIR}/")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run all 66 episodes")
    args = parser.parse_args()
    run_replay(quick=not args.full)
