"""
Quick base-policy eval for Cup (StandCup) task on login node.
Runs GR00T only (no residual) and reports per-env success rate.

Usage:
    python scripts/eval_base_cup.py \
        --groot_checkpoint checkpoints/<TASK>/checkpoint \
        --num_envs 5 --num_episodes 2 --save_video
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

from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified


def main():
    p = argparse.ArgumentParser(description="Base policy eval for Cup (StandCup)")
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=5)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--reward_type", type=str, default="dense")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--output_dir", type=str, default="outputs/cup_base_eval")
    p.add_argument("--chunk_sync", action="store_true", default=True)
    p.add_argument("--parallel_envs", action="store_true", default=False,
                   help="Use parallel env workers (faster, no video)")
    p.add_argument("--async_prefetch", action="store_true", default=False)
    p.add_argument("--cup_positions_json", type=str,
                   default="configs/cup_positions.json",
                   help="JSON file with cup positions")
    p.add_argument("--episode_ids", type=str, default=None,
                   help="Comma-separated episode IDs to use (e.g. 0,1,2,3,4). "
                        "Default: first num_envs episodes")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse episode IDs
    episode_ids = None
    if args.episode_ids:
        episode_ids = [int(x) for x in args.episode_ids.split(",")]
        print(f"[cup-eval] Using episode IDs: {episode_ids}")
    else:
        episode_ids = list(range(args.num_envs))
        print(f"[cup-eval] Using first {args.num_envs} episodes")

    num_envs = len(episode_ids) if episode_ids else args.num_envs

    print(f"[cup-eval] Creating {num_envs} envs...")
    vec_env = MuJoCoVecEnvCup(
        num_envs=num_envs,
        episode_ids=episode_ids,
        cup_positions_path=args.cup_positions_json,
        max_episode_steps=args.max_episode_steps,
        reward_type=args.reward_type,
        device=args.device,
        parallel_envs=args.parallel_envs,
    )

    # Wrap with GR00T (zero residual = base policy only)
    env = MuJoCoResidualWrapperUnified(
        vec_env=vec_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device=args.device,
        task_description="Pick up the cup lying on its side and stand it upright.",
        open_loop_horizon=16,
        chunk_sync=args.chunk_sync,
        grip_min=0.0,
        grip_max=0.04,
        use_gripper_latch=True,
        grip_close_latch_thresh=0.015,
        grip_open_latch_thresh=0.035,
        grip_latch_open_steps=32,
        async_prefetch=args.async_prefetch,
        use_images=args.save_video,  # render cam_base images for video
    )
    print(f"[cup-eval] Env created: {num_envs} envs, steps={args.max_episode_steps}")

    # Storage for videos
    if args.save_video:
        video_frames = {i: [] for i in range(num_envs)}

    total_success = 0
    total_episodes = 0
    per_env_success = np.zeros(num_envs, dtype=int)
    per_env_total = np.zeros(num_envs, dtype=int)

    for ep in range(args.num_episodes):
        print(f"\n[cup-eval] Episode {ep+1}/{args.num_episodes}")
        obs, info = env.reset()
        done_flags = np.zeros(num_envs, dtype=bool)
        ep_rewards = np.zeros(num_envs)
        ep_successes = np.zeros(num_envs, dtype=bool)

        for step in range(args.max_episode_steps):
            # Step with zero residual (base policy only)
            residual = torch.zeros(num_envs, 7, device=args.device)
            obs, rewards, terminated, truncated, info = env.step(residual)

            ep_rewards += rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards

            # Log grip and cup state every 20 steps
            if step % 20 == 0:
                held = env._held_base_action
                grip_vals = held[:, 7] if held.shape[1] > 7 else held[:, -1]
                grip_str = [f'{g:.4f}' for g in grip_vals]
                print(f"  step={step:3d} grip={grip_str}")

            # Capture frames for video from obs images
            if args.save_video:
                cam_key = "observation.images.cam_base"
                if cam_key in obs:
                    imgs = obs[cam_key]  # (num_envs, C, H, W) or (num_envs, H, W, C)
                    for i in range(num_envs):
                        if not done_flags[i]:
                            frame = imgs[i].cpu().numpy() if hasattr(imgs[i], 'cpu') else imgs[i]
                            if frame.shape[0] == 3:  # CHW -> HWC
                                frame = frame.transpose(1, 2, 0)
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            video_frames[i].append(frame.copy())

            # Check success
            if isinstance(terminated, torch.Tensor):
                term_np = terminated.cpu().numpy()
            else:
                term_np = np.asarray(terminated)
            for i in range(num_envs):
                if not done_flags[i] and term_np[i]:
                    ep_successes[i] = True
                    print(f"  >>> env{i} (ep{episode_ids[i]}) SUCCESS at step={step}!")

            dones = terminated | truncated
            if isinstance(dones, torch.Tensor):
                dones = dones.cpu().numpy()
            done_flags |= dones

            if done_flags.all():
                break

        total_success += ep_successes.sum()
        total_episodes += num_envs
        per_env_success += ep_successes.astype(int)
        per_env_total += 1

        for i in range(num_envs):
            status = "✅" if ep_successes[i] else "❌"
            print(f"  env{i} (ep{episode_ids[i]}): {status} reward={ep_rewards[i]:.4f} steps={step+1}")

    # Summary
    sr = total_success / total_episodes * 100
    print(f"\n{'='*60}")
    print(f"[cup-eval] Overall SR = {total_success}/{total_episodes} = {sr:.1f}%")
    print(f"\nPer-env breakdown:")
    for i in range(num_envs):
        env_sr = per_env_success[i] / per_env_total[i] * 100
        print(f"  env{i} (ep{episode_ids[i]}): {per_env_success[i]}/{per_env_total[i]} = {env_sr:.0f}%")

    # Save videos (avc1/h264 codec)
    if args.save_video:
        try:
            import cv2
            for i in range(num_envs):
                frames = video_frames[i]
                if len(frames) == 0:
                    print(f"  env{i}: no frames captured")
                    continue
                vpath = output_dir / f"env{i}_ep{episode_ids[i]}.mp4"
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"avc1"), 30, (w, h))
                if not writer.isOpened():
                    # Fallback: write mp4v then re-encode with ffmpeg
                    writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                for f in frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.shape[-1] == 3 else f)
                writer.release()
                # Re-encode to h264 if mp4v was used (avc1 not available in OpenCV)
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
        "task": "cup",
        "groot_checkpoint": args.groot_checkpoint,
        "cup_positions_json": args.cup_positions_json,
        "episode_ids": episode_ids,
        "num_envs": num_envs,
        "num_episodes": args.num_episodes,
        "success_rate": sr,
        "total_success": int(total_success),
        "total_episodes": total_episodes,
        "per_env_sr": {
            f"ep{episode_ids[i]}": per_env_success[i] / per_env_total[i] * 100
            for i in range(num_envs)
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[cup-eval] Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
