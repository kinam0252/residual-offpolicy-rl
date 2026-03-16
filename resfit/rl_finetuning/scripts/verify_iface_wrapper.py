"""Quick verification: run IfaceEnvWrapper for N steps and save video.

Usage (inside isaaclab.sh):
  isaaclab.sh -p resfit/rl_finetuning/scripts/verify_iface_wrapper.py \
    --headless --num_envs 1 --csv_dir ... --groot_model_path ... \
    --max_steps 500 --output_dir /path/to/output
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_resfit_tasks_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "isaaclab", "source", "isaaclab_tasks")
_resfit_tasks_path = os.path.abspath(_resfit_tasks_path)
if os.path.isdir(_resfit_tasks_path) and _resfit_tasks_path not in sys.path:
    sys.path.insert(0, _resfit_tasks_path)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--csv_dir", type=str, required=True)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--policy_device", type=str, default="cuda:0")
parser.add_argument("--language_override", type=str, default=None)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--output_dir", type=str, default="/tmp/iface_wrapper_verify")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import imageio
from pathlib import Path

import isaaclab.sim as sim_utils
from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    env = IfaceEnvWrapper(
        sim=sim,
        csv_dir=args.csv_dir,
        groot_model_path=args.groot_model_path,
        embodiment_tag=args.embodiment_tag,
        policy_device=args.policy_device,
        language_override=args.language_override,
        task_description="pick up mushroom",
        max_episode_steps=args.max_steps,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    frames = []
    rewards = []

    for step in range(args.max_steps):
        # residual = None means pure GR00T (online equivalent)
        obs, reward, terminated, truncated, info = env.step(residual_action=None)
        rewards.append(float(reward[0].item()))

        # Capture front camera frame (84x84 CHW → HWC)
        frame = obs["observation.images.front"][0].cpu().numpy()
        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        frames.append(frame)

        if terminated[0].item():
            break

    # Save video
    vid_path = out_dir / "iface_wrapper_500.mp4"
    writer = imageio.get_writer(str(vid_path), fps=20)
    for fr in frames:
        writer.append_data(fr)
    writer.close()

    total_reward = sum(rewards)
    success = any(r >= 1.0 for r in rewards)
    print(f"[verify] steps={len(frames)}, total_reward={total_reward:.1f}, success={success}")
    print(f"[verify] video saved: {vid_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
