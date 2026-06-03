"""
Quick base-policy eval for Lift task on login node.
Runs GR00T only (no residual) and saves videos.

Usage:
    python scripts/eval_base_lift.py \
        --groot_checkpoint checkpoints/<TASK>/checkpoint \
        --num_envs 3 --num_episodes 1 --save_video
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import types as _types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DS_BUILD_OPS", "0")

# ── Deepspeed mock ──
if "deepspeed" not in sys.modules:
    _ds_mock = _types.ModuleType("deepspeed")
    _ds_mock.__version__ = "0.0.0"
    _ds_mock.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds_mock.__path__ = []
    _ds_mock.__file__ = __file__
    _ds_zero = _types.ModuleType("deepspeed.zero")
    _ds_zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _ds_zero.Init = lambda *a, **kw: (lambda f: f)
    _ds_mock.zero = _ds_zero
    sys.modules["deepspeed"] = _ds_mock
    sys.modules["deepspeed.zero"] = _ds_zero

# ── Patch huggingface_hub ──
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _patched(repo_id: str) -> None:
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")):
            return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _patched
except Exception:
    pass

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified


def main():
    p = argparse.ArgumentParser(description="Base policy eval for Lift")
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=3)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--reward_type", type=str, default="dense")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--output_dir", type=str, default="outputs/lift_base_eval")
    p.add_argument("--chunk_sync", action="store_true", default=True)
    p.add_argument("--parallel_envs", action="store_true", default=False,
                   help="Use parallel env workers (faster, no video)")
    p.add_argument("--async_prefetch", action="store_true", default=False,
                   help="Async GR00T prefetch (faster but needs EGL thread safety)")
    p.add_argument("--cube_positions_json", type=str, default=None,
                   help="JSON file with cube positions (list of {cube_pos, cube_yaw_deg})")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cube positions from JSON if provided
    cube_positions = None
    if args.cube_positions_json:
        import json
        with open(args.cube_positions_json) as f:
            pos_data = json.load(f)
        cube_positions = [p["cube_pos"] for p in pos_data[:args.num_envs]]
        print(f"[lift-eval] Loaded {len(cube_positions)} cube positions from {args.cube_positions_json}")
        for i, cp in enumerate(cube_positions):
            print(f"  env{i}: cube_pos={cp}")

    print(f"[lift-eval] Creating {args.num_envs} envs...")
    vec_env = MuJoCoVecEnv(
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        reward_type=args.reward_type,
        device=args.device,
        parallel_envs=args.parallel_envs,
        cube_positions=cube_positions,
    )
    # Wrap with GR00T
    env = MuJoCoResidualWrapperUnified(
        vec_env=vec_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device=args.device,
        task_description="lift the wooden block",
        open_loop_horizon=16,
        chunk_sync=args.chunk_sync,
        grip_min=0.0,
        grip_max=1.0,
        use_gripper_latch=True,
        async_prefetch=args.async_prefetch,
    )
    print(f"[lift-eval] Env created: {args.num_envs} envs, steps={args.max_episode_steps}")

    # Storage for videos
    if args.save_video:
        video_frames = {i: [] for i in range(args.num_envs)}

    total_success = 0
    total_episodes = 0

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        done_flags = np.zeros(args.num_envs, dtype=bool)
        ep_rewards = np.zeros(args.num_envs)
        grip_log = []
        ep_successes = np.zeros(args.num_envs, dtype=bool)

        for step in range(args.max_episode_steps):
            # Step with zero residual (base policy only)
            residual = torch.zeros(args.num_envs, 7, device=args.device)
            obs, rewards, terminated, truncated, info = env.step(residual)

            ep_rewards += rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards

            # Log gripper values and grasp state every step
            held = env._held_base_action  # (num_envs, 8) - GR00T raw output
            grip_vals = held[:, 7] if held.shape[1] > 7 else held[:, -1]
            # Also get actual combined grip (after latch)
            combined_grip = info.get("scaled_action", None)
            if combined_grip is not None:
                cg = combined_grip[:, 7].cpu().numpy() if hasattr(combined_grip, 'cpu') else combined_grip[:, 7]
            else:
                cg = grip_vals
            
            # Check grasp state and cube height
            grasp_info = []
            for i in range(args.num_envs):
                e = env.vec_env._envs[i]
                grasped = e["grasp_state"]["grasped"]
                cube_z = e["data"].qpos[e["cube_qposadr"] + 2] if e["cube_qposadr"] is not None else 0
                init_z = e.get("initial_cube_z", 0.02)
                lift = cube_z - init_z
                grasp_info.append((grasped, cube_z, lift))
            
            if step % 10 == 0:
                grip_str = [f'{g:.4f}' for g in grip_vals]
                cg_str = [f'{g:.4f}' for g in cg]
                grasp_str = [f'{"G" if gi[0] else "-"} lift={gi[2]:.3f}' for gi in grasp_info]
                print(f"  step={step:3d} raw_grip={grip_str} combined={cg_str} grasp={grasp_str}")
            grip_log.append(grip_vals.copy())

            # Capture frames for video
            if args.save_video:
                try:
                    frames_batch = vec_env.render()  # (num_envs, H, W, 3)
                    for i in range(args.num_envs):
                        if not done_flags[i]:
                            video_frames[i].append(frames_batch[i].copy())
                except Exception:
                    pass

            # Check success: terminated=True means _is_success() passed
            if isinstance(terminated, torch.Tensor):
                term_np = terminated.cpu().numpy()
            else:
                term_np = np.asarray(terminated)
            for i in range(args.num_envs):
                if not done_flags[i] and term_np[i]:
                    ep_successes[i] = True
                    print(f"  >>> env{i} SUCCESS at step={step}!")

            dones = terminated | truncated
            if isinstance(dones, torch.Tensor):
                dones = dones.cpu().numpy()
            done_flags |= dones

            if done_flags.all():
                break

        total_success += ep_successes.sum()
        total_episodes += args.num_envs

        # Grip analysis
        if grip_log:
            grip_arr = np.array(grip_log)  # (T, num_envs)
            for i in range(args.num_envs):
                g = grip_arr[:, i]
                below_08 = (g < 0.8).sum()
                below_05 = (g < 0.5).sum()
                print(f"  env{i} grip: min={g.min():.4f} max={g.max():.4f} mean={g.mean():.4f} <0.8={below_08} <0.5={below_05}/{len(g)}")

        for i in range(args.num_envs):
            status = "✅" if ep_successes[i] else "❌"
            print(f"  env{i} ep{ep}: {status} reward={ep_rewards[i]:.4f} steps={step+1}")

    sr = total_success / total_episodes * 100
    print(f"\n[lift-eval] SR = {total_success}/{total_episodes} = {sr:.1f}%")

    # Save videos (avc1/h264 codec)
    if args.save_video:
        try:
            import cv2
            for i in range(args.num_envs):
                frames = video_frames[i]
                if len(frames) == 0:
                    print(f"  env{i}: no frames captured")
                    continue
                vpath = output_dir / f"env{i}.mp4"
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"avc1"), 30, (w, h))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                for f in frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.shape[-1] == 3 else f)
                writer.release()
                try:
                    import imageio_ffmpeg
                    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                    tmp = str(vpath) + ".tmp.mp4"
                    import subprocess
                    subprocess.run([ffmpeg, "-y", "-i", str(vpath), "-c:v", "libx264",
                                    "-pix_fmt", "yuv420p", "-crf", "23", tmp],
                                   capture_output=True, timeout=60)
                    if os.path.exists(tmp):
                        os.replace(tmp, str(vpath))
                except Exception:
                    pass
                print(f"  env{i}: saved {len(frames)} frames → {vpath}")
        except ImportError:
            print("  cv2 not available, skipping video save")

    # Save results JSON
    results = {
        "groot_checkpoint": args.groot_checkpoint,
        "num_envs": args.num_envs,
        "num_episodes": args.num_episodes,
        "success_rate": sr,
        "total_success": int(total_success),
        "total_episodes": total_episodes,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[lift-eval] Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
