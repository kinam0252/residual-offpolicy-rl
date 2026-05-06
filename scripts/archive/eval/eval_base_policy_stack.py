#!/usr/bin/env python3
"""Evaluate base GR00T policy on Stack Cube (no residual) with video saving.

Replicates the exact eval setup used by the training sweep:
  - 20 envs with 66 positions (first 20)
  - MuJoCoResidualWrapper (residual action = 0)
  - max_episode_steps same as training
  - Saves per-env videos for analysis
"""
import sys
from pathlib import Path

# Repo path
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Patch huggingface validate_repo_id for local checkpoints
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig_validate = _hf_val.validate_repo_id
    def _patched_validate_repo_id(repo_id: str) -> None:
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")):
            return
        return _orig_validate(repo_id)
    _hf_val.validate_repo_id = _patched_validate_repo_id
except Exception:
    pass

import argparse
import json
import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack as MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import MuJoCoVecEnvStack as MuJoCoVecEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_positions_file", type=str, required=True)
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--groot_embodiment_tag", type=str, default="NEW_EMBODIMENT")
    p.add_argument("--task_description", type=str,
                   default="Pick up the white cube and stack it on the green cube.")
    p.add_argument("--open_loop_horizon", type=int, default=16)
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--num_envs", type=int, default=20)
    p.add_argument("--num_episodes", type=int, default=3,
                   help="Num rounds (each round = all envs run 1 episode)")
    p.add_argument("--output_dir", type=str, default="outputs/stack_base_eval_videos")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--calib_path", type=str, default=None)
    p.add_argument("--reward_type", type=str, default="dense")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load positions
    with open(args.eval_positions_file) as f:
        pos_list = json.load(f)
    seen = set()
    unique_pos = []
    for pp in pos_list:
        if pp["episode"] not in seen:
            seen.add(pp["episode"])
            unique_pos.append(pp)

    n = min(args.num_envs, len(unique_pos))
    white_positions = [pp["white_cube_pos"] for pp in unique_pos[:n]]
    green_positions = [pp["green_cube_pos"] for pp in unique_pos[:n]]
    print(f"[base-eval] Using {n} env positions from {args.eval_positions_file}")

    # Auto-detect calib
    _calib_path = args.calib_path
    if _calib_path is None:
        ckpt_lower = args.groot_checkpoint.lower()
        if "66ep" in ckpt_lower or "100ep" in ckpt_lower:
            c = str(Path(__file__).resolve().parents[2] / "Mujoco_Franka" / "config" / "camera_info_66ep.yaml")
            if Path(c).exists():
                _calib_path = c
                print(f"[base-eval] Auto-detected calib: {c}")

    # Create env
    print(f"[base-eval] Creating {n} envs...")
    mujoco_env = MuJoCoVecEnv(
        num_envs=n,
        white_cube_positions=white_positions,
        green_cube_positions=green_positions,
        scene_xml=args.scene_xml,
        calib_path=_calib_path,
        max_episode_steps=args.max_episode_steps,
        reward_type=args.reward_type,
        device=args.device,
    )

    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=args.device,
        task_description=args.task_description,
        open_loop_horizon=args.open_loop_horizon,
        residual_pos_scale=0.0,  # no residual
        residual_rot_scale=0.0,
        residual_grip_scale=0.0,
        ema_alpha=0.0,
        action_scaler=None,
    )
    print("[base-eval] Environment ready.")

    # Run episodes
    total_success = 0
    total_episodes = 0

    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        env_done = [False] * n
        env_success = [False] * n
        # Video frames per env
        frames = [[] for _ in range(n)]

        for step in range(args.max_episode_steps):
            # Base policy only: residual = 0
            zero_action = torch.zeros((n, env.action_dim), device=device)
            obs, reward, terminated, truncated, info = env.step(zero_action)
            done = terminated | truncated

            # Capture frames from front camera for first round only
            if ep == 0:
                # Use front camera images from observation for video
                if "observation.images.front" in obs:
                    front_imgs = obs["observation.images.front"].cpu().numpy()  # (n, 3, 84, 84)
                    for i in range(n):
                        if not env_done[i]:
                            frame = np.transpose(front_imgs[i], (1, 2, 0))  # (84, 84, 3)
                            frames[i].append(frame)

            for i in range(n):
                if not env_done[i] and done[i]:
                    env_done[i] = True
                    if terminated[i]:
                        env_success[i] = True

            if all(env_done):
                break

        ep_successes = sum(env_success)
        total_success += ep_successes
        total_episodes += n
        sr = ep_successes / n * 100
        print(f"[base-eval] Episode {ep}: SR={sr:.1f}% ({ep_successes}/{n})")

        # Save videos from first round
        if ep == 0 and imageio is not None:
            for i in range(n):
                if len(frames[i]) > 0:
                    status = "success" if env_success[i] else "fail"
                    video_path = output_dir / f"env{i:02d}_{status}.mp4"
                    imageio.mimwrite(str(video_path), frames[i], fps=30, codec="libx264")
            print(f"[base-eval] Videos saved to {output_dir}")

    overall_sr = total_success / total_episodes * 100
    print(f"\n[base-eval] Overall SR: {overall_sr:.1f}% ({total_success}/{total_episodes})")
    print(f"[base-eval] {args.num_episodes} rounds × {n} envs = {total_episodes} episodes")


if __name__ == "__main__":
    main()
